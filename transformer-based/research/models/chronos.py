"""Chronos-2 MPS adapter with explicit origin groups and observed-label loss."""
from __future__ import annotations
import types
import numpy as np
import torch
from ..data import decode


def observed_quantile_loss(self,quantile_preds,future_target,future_target_mask,patched_future_covariates_mask,loc_scale,num_output_patches):
    target,_=self.instance_norm(future_target,loc_scale)
    mask=future_target_mask.bool() if future_target_mask is not None else torch.isfinite(target)
    horizon=target.shape[-1]
    prediction=quantile_preds[...,:horizon]
    target=torch.where(mask,target,0)[:,None,:]
    q=self.quantiles[None,:,None].to(prediction)
    error=target-prediction
    pinball=2*torch.maximum(q*error,(q-1)*error)
    valid=mask[:,None,:].to(pinball)
    # Only observed targets count. Covariate rows and output patch padding cannot dilute loss.
    return (pinball*valid).sum()/(valid.sum().clamp_min(1)*self.num_quantiles)


class ChronosAdapter:
    def __init__(self,path,snapshot,pre,device='mps',lora=True,rank=8,alpha=16):
        from chronos.chronos2.model import Chronos2Model
        self.snapshot=snapshot; self.pre=pre; self.device=device
        model=Chronos2Model.from_pretrained(str(path),torch_dtype=torch.float32).to(device)
        model._compute_loss=types.MethodType(observed_quantile_loss,model)
        self.base=model; self.quantiles=model.quantiles.detach().cpu().numpy()
        if lora:
            from peft import LoraConfig,get_peft_model
            modules=[n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and n.endswith(('.q','.v')) and 'attention' in n]
            if not modules: raise ValueError('Pinned Chronos attention projections not found')
            model=get_peft_model(model,LoraConfig(r=rank,lora_alpha=alpha,target_modules=modules,lora_dropout=0,bias='none'))
            trainable=[n for n,p in model.named_parameters() if p.requires_grad]
            if not trainable or any('lora_' not in n for n in trainable): raise ValueError('Unexpected trainable Chronos base weights')
        self.model=model; self.pca={}
        if pre.get('text_sources'):
            self._fit_text_pca()

    def _fit_text_pca(self):
        from sklearn.decomposition import PCA
        origins=self.pre['training_origins']
        dates=set(self.snapshot.dates[np.unique(np.concatenate([np.arange(i-self.snapshot.lb+1,i+1) for i in origins]))])
        for source in self.pre['text_sources']:
            rows=self.snapshot.text[(self.snapshot.text.source==source)&self.snapshot.text.date.map(lambda d:d in dates)]
            vectors=np.stack(rows.embedding)
            if len(vectors)>=4:self.pca[source]=PCA(n_components=4,svd_solver='full').fit(vectors)

    def arrays(self,i,with_target=False):
        s=self.snapshot; sl=slice(i-s.lb+1,i+1)
        raw=s.frame.iloc[sl]
        a=raw.quidax_average.to_numpy(); upper=raw.quidax_high.to_numpy()-a; lower=a-raw.quidax_low.to_numpy()
        active=np.array(self.pre['active']); cov=s.x[sl][:,active]
        # Preserve native Chronos raw-level instance normalization.
        rows=[a,upper,lower]+[cov[:,j] for j in range(cov.shape[1])]
        if self.pca:
            sample=s.sample(i,self.pre)
            for source,pca in self.pca.items():
                j=self.pre['text_sources'].index(source); mask=sample['text_mask'][:,j]
                projected=pca.transform(sample['text'][:,j]); projected[~mask]=np.nan
                rows += [projected[:,k] for k in range(4)]
                rows += [sample['document_count'][:,j].astype(float),mask.astype(float)]
        context=np.stack(rows).astype('float32'); n=len(context)
        future=np.full((n,7),np.nan,dtype='float32')
        # Last four numerical columns are calendar terms when active; look up by name.
        events=s.sample(i,self.pre)['known_events']
        for j,name in enumerate(self.pre['names']):
            if name in ['weekday_sin','weekday_cos','month_sin','month_cos']:
                future[3+j]=events[:,['weekday_sin','weekday_cos','month_sin','month_cos'].index(name)]
        kwargs={'context':torch.tensor(context,device=self.device),'group_ids':torch.zeros(n,dtype=torch.long,device=self.device),
                'future_covariates':torch.tensor(future,device=self.device),'num_output_patches':1}
        if with_target:
            rates=s.rates[i+1:i+8]; values=np.stack([rates[:,0],rates[:,1]-rates[:,0],rates[:,0]-rates[:,2]])
            targets=np.zeros((n,7),dtype='float32'); targets[:3]=np.nan_to_num(values)
            mask=np.zeros((n,7),dtype=bool); mask[:3]=np.isfinite(values)
            kwargs['future_target']=torch.tensor(targets,device=self.device); kwargs['future_target_mask']=torch.tensor(mask,device=self.device)
        return kwargs

    def loss(self,i):
        kwargs=self.arrays(i,True)
        if not kwargs['future_target_mask'].any():return None
        return self.model(**kwargs).loss

    def predict(self,i,sample=None):
        self.model.eval()
        with torch.no_grad(): values=self.model(**self.arrays(i)).quantile_preds[:3,:,:7].float().cpu().numpy()
        median=int(np.argmin(np.abs(self.quantiles-.5))); a,u,l=values[:,median]
        return decode(a,u,l,self.snapshot.rates[i,0])

    def state_dict(self):
        return {k:v.detach().cpu() for k,v in self.model.state_dict().items() if 'lora_' in k}

    def load_state_dict(self,state):
        result=self.model.load_state_dict(state,strict=False)
        if result.unexpected_keys:raise ValueError('Checkpoint contains incompatible adapter keys')
