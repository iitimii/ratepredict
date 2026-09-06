"""Resumable experiment loops; only completed evaluations advance the model ladder."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np

from .core import State,ResourceBlocked,digest,file_hash,now,write_json
from .data import Snapshot,decode
from .evaluation import evaluate
from .tracking import Tracker


def run_identity(snapshot,model,fold,seed,settings):
    return digest({'snapshot':snapshot.manifest['id'],'model':model,'fold':fold,'seed':seed,'settings':settings})[:24]


def torch_save(path,payload,state):
    import torch
    # Model state is adapters/heads only. Keep previous complete checkpoint until fsync+rename.
    with state.reserve(180*1024**2,'optimizer and adapter checkpoint'):
        temp=path.with_suffix('.tmp'); torch.save(payload,temp)
        with temp.open('rb') as f:os.fsync(f.fileno())
        os.replace(temp,path)


def rng_state():
    import torch
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'mps':torch.mps.get_rng_state() if torch.backends.mps.is_available() else None}


def restore_rng(rng):
    import torch
    random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
    if rng.get('mps') is not None:torch.mps.set_rng_state(rng['mps'])


def train_chronos(cfg,snapshot_path,fold='dev1',seed=42,smoke=False,rotate=False):
    import torch
    from .models.chronos import ChronosAdapter
    from .models.downloads import download
    state=State(cfg); snapshot=Snapshot(snapshot_path); settings=dict(cfg['training']['chronos'])
    if smoke:settings.update(max_steps=1,accumulation=1,eval_every=1)
    train=snapshot.origins(fold,'train'); val=snapshot.origins(fold,'eval')
    if smoke:train=train[-8:];val=val[:2]
    pre=snapshot.fit_preprocessing(train); pre['training_origins']=train
    rid=run_identity(snapshot,'chronos_smoke' if smoke else 'chronos',fold,seed,settings)
    tracker=Tracker(cfg,rid,{'model':'chronos','snapshot':snapshot.manifest['id'],'fold':fold,'seed':seed,'settings':settings},online=not smoke)
    path=tracker.path;write_json(path/'preprocessing.json',pre)
    with state.connect() as db:db.execute('INSERT OR REPLACE INTO runs VALUES(?,?,?,?,?,?,?,?)',(rid,snapshot.manifest['id'],'chronos',fold,seed,'running',now(),None))
    started=time.monotonic(); status='failed'; latest=path/'latest.pt';best=math.inf;bad=0;update=0;position=0;order=[]
    try:
        with state.gpu():
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
            if not torch.backends.mps.is_available():raise ResourceBlocked('Chronos requires an available MPS device')
            backbone=download(cfg,'chronos',rotate=rotate)
            adapter=ChronosAdapter(backbone,snapshot,pre,rank=settings['rank'],alpha=settings['alpha'])
            optimizer=torch.optim.AdamW([p for p in adapter.model.parameters() if p.requires_grad],lr=settings['learning_rate'],fused=False)
            if latest.exists():
                checkpoint=torch.load(latest,map_location='cpu',weights_only=False)
                if checkpoint['identity']!=rid:raise ValueError('Resume configuration or snapshot mismatch')
                adapter.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer']);restore_rng(checkpoint['rng'])
                update=checkpoint['update'];position=checkpoint['position'];order=checkpoint['order'];best=checkpoint['best'];bad=checkpoint['bad']
            elif not smoke:
                result=evaluate(snapshot,val,adapter.predict,pre,path/'pretrained',tracker)
                tracker.log({'pretrained/score':result['score']})
            def checkpoint():
                return {'identity':rid,'model':adapter.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng_state(),
                        'update':update,'position':position,'order':order,'best':best,'bad':bad,'wandb_id':rid}
            while update<settings['max_steps'] and bad<settings['patience']:
                state.check_memory();adapter.model.train();optimizer.zero_grad();losses=[]
                for _ in range(settings['accumulation']):
                    if position>=len(order):order=np.random.permutation(train).tolist();position=0
                    i=order[position];position+=1
                    loss=adapter.loss(i)
                    if loss is None:continue
                    if not torch.isfinite(loss):raise ValueError('Nonfinite Chronos loss')
                    (loss/settings['accumulation']).backward();losses.append(float(loss.detach().cpu()))
                if not losses:continue
                if len(losses)!=settings['accumulation']:
                    for p in adapter.model.parameters():
                        if p.grad is not None:p.grad.mul_(settings['accumulation']/len(losses))
                torch.nn.utils.clip_grad_norm_([p for p in adapter.model.parameters() if p.requires_grad],1.0)
                optimizer.step();update+=1;torch.mps.synchronize()
                if update%10==0 or update==1:
                    tracker.log({'train/update':update,'train/loss':float(np.mean(losses)),
                        'train/seconds_per_update':(time.monotonic()-started)/update,
                        'train/eta_seconds':(time.monotonic()-started)/update*(settings['max_steps']-update),
                        'metal/allocated_gb':torch.mps.current_allocated_memory()/1024**3,**state.metrics()})
                if update%settings['eval_every']==0 or update==settings['max_steps']:
                    result=evaluate(snapshot,val,adapter.predict,pre,path/'validation',tracker)
                    if result['score']<best:
                        best=result['score'];bad=0;torch_save(path/'best.pt',checkpoint(),state)
                    else:bad+=1
                    torch_save(latest,checkpoint(),state)
            chosen=torch.load(path/'best.pt',map_location='cpu',weights_only=False);adapter.load_state_dict(chosen['model'])
            result=evaluate(snapshot,val,adapter.predict,pre,path/'evaluation',tracker)
            result.update(run_id=rid,snapshot=snapshot.manifest['id'],model='chronos',fold=fold,seed=seed,
                          optimizer_updates=update,smoke_only=smoke,seconds=time.monotonic()-started)
            write_json(path/'result.json',result);tracker.artifact(path/'evaluation/predictions.parquet',rid+'-predictions')
            status='smoke_passed' if smoke else 'complete';return result
    except BaseException as e:
        if 'adapter' in locals() and 'optimizer' in locals():
            try:torch_save(latest,checkpoint(),state)
            except Exception:pass
        write_json(path/'failure.json',{'type':type(e).__name__,'reason':str(e)[:600],'update':update,'at':now()})
        status='blocked' if isinstance(e,ResourceBlocked) else 'failed';raise
    finally:
        tracker.close('complete' if status in ('complete','smoke_passed') else status)
        with state.connect() as db:db.execute('UPDATE runs SET status=?,updated=?,result=? WHERE id=?',(status,now(),str(path/'result.json'),rid))


def baselines(cfg,snapshot_path,fold='dev1',seed=42):
    from sklearn.linear_model import Ridge
    state=State(cfg); snapshot=Snapshot(snapshot_path);train=snapshot.origins(fold,'train');val=snapshot.origins(fold,'eval')
    pre=snapshot.fit_preprocessing(train);root=state.root/'baselines'/snapshot.manifest['id']/fold
    persistence=evaluate(snapshot,val,lambda i,s:snapshot.baseline(i),pre,root/'persistence')
    def features(i):
        sample=snapshot.sample(i,pre);x=sample['quantitative'];m=sample['quantitative_mask']
        return np.concatenate([x[-1],m[-1],x[-7:].mean(axis=0),x[-30:].mean(axis=0),x[-90:].mean(axis=0)])
    x=np.stack([features(i) for i in train]); models=[]
    y=[];m=[]
    from .data import labels
    for i in train:
        a,b=labels(snapshot.rates,i);y.append(a.ravel());m.append(b.ravel())
    y,m=np.array(y),np.array(m)
    for c in range(21):
        if not m[:,c].any():raise ValueError('No training labels for output '+str(c))
        models.append(Ridge(alpha=100).fit(x[m[:,c]],y[m[:,c],c]))
    def predict(i,sample):
        p=np.array([model.predict(features(i)[None])[0] for model in models]).reshape(7,3)
        return decode(sample['anchor']+p[:,0],p[:,1],-p[:,2],sample['anchor'])
    ridge=evaluate(snapshot,val,predict,pre,root/'ridge')
    import pickle
    (root/'ridge/model.pkl').write_bytes(pickle.dumps({'models':models,'preprocessing':pre,'snapshot':snapshot.manifest['id']}))
    return {'persistence':persistence['score'],'ridge':ridge['score']}
