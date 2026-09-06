"""Shared masked metrics, immutable prediction artifacts, and paired uncertainty."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .data import CHANNELS,labels,decode
from .core import write_json

WEIGHTS=np.array([.5,.25,.25])


def metrics(pred,target,mask,scale,baseline=None):
    p,y,m=np.asarray(pred),np.asarray(target),np.asarray(mask,dtype=bool)
    if p.shape!=y.shape or m.shape!=y.shape or p.ndim!=3 or p.shape[1:]!=(7,3):
        raise ValueError('Expected identical [origins,7,3] arrays')
    if not np.isfinite(p).all(): raise ValueError('Nonfinite predictions')
    count=m.sum(axis=0); error=np.where(m,p-y,0)
    def average(v): return np.divide(v.sum(axis=0),count,out=np.full((7,3),np.nan),where=count>0)
    mae=average(np.abs(error)); rmse=np.sqrt(average(error**2))
    score=float(np.nanmean(np.nansum(mae/scale*WEIGHTS,axis=1)))
    out={'score':score,'observed_labels':int(m.sum()),'origins':len(p)}
    for h in range(7):
        for c,name in enumerate(CHANNELS):
            key=f'h{h+1}/{name}'; out[key+'/count']=int(count[h,c])
            out[key+'/mae']=float(mae[h,c]) if count[h,c] else None
            out[key+'/rmse']=float(rmse[h,c]) if count[h,c] else None
    observed=m[:,:,0]; nonzero=observed&(y[:,:,0]!=0)
    out['directional_accuracy_nonzero']=float((np.sign(p[:,:,0][nonzero])==np.sign(y[:,:,0][nonzero])).mean()) if nonzero.any() else None
    out['zero_change_labels']=int((observed&(y[:,:,0]==0)).sum())
    out['directional_accuracy_including_zero']=float((np.sign(p[:,:,0][observed])==np.sign(y[:,:,0][observed])).mean()) if observed.any() else None
    if baseline is not None:
        bs=metrics(baseline,y,m,scale)['score']; out['baseline_score']=bs; out['skill']=1-score/bs if bs else None
    return out


def daily_losses(pred,target,mask,scale):
    values=np.where(mask,np.abs(pred-target)/scale,0)
    count=mask.sum(axis=1)
    per_channel=np.divide(values.sum(axis=1),count,out=np.zeros_like(count,dtype=float),where=count>0)
    return (per_channel*WEIGHTS).sum(axis=-1)


def paired_bootstrap(dates,loss_a,loss_b,block_days=28,repeats=1000,seed=42):
    dates=pd.DatetimeIndex(dates)
    if not len(dates): return None
    calendar=pd.date_range(dates.min(),dates.max(),freq='D')
    delta=pd.Series(np.asarray(loss_a)-np.asarray(loss_b),index=dates).reindex(calendar).to_numpy()
    rng=np.random.default_rng(seed); results=[]
    # Calendar blocks preserve both gaps in eligible origins and overlapping horizons.
    for _ in range(repeats):
        starts=rng.integers(0,max(1,len(delta)-block_days+1),size=int(np.ceil(len(delta)/block_days)))
        sample=np.concatenate([delta[s:s+block_days] for s in starts])[:len(delta)]
        if np.isfinite(sample).any(): results.append(float(np.nanmean(sample)))
    return {'difference':float(np.nanmean(delta)),'ci95':np.quantile(results,[.025,.975]).tolist(),
            'block_calendar_days':block_days,'repeats':repeats,
            'limitation':'Retrospective data; overlapping horizons and short history limit independence.'}


def evaluate(snapshot,origins,predict,pre,path=None,tracker=None):
    predictions=[]; targets=[]; masks=[]; baseline=[]; rows=[]; corrections=0
    import time
    start=time.monotonic()
    for i in origins:
        sample=snapshot.sample(i,pre); answer=predict(i,sample)
        if isinstance(answer,tuple): p,c=answer; corrections+=c
        else: p=answer
        y,m=labels(snapshot.rates,i); b=snapshot.baseline(i)
        predictions.append(p); targets.append(y); masks.append(m); baseline.append(b)
        for h in range(7):
            row={'origin':snapshot.dates[i],'horizon':h+1,'target_date':snapshot.dates[i+h+1],
                 'observed':bool(m[h].all())}
            for c,n in enumerate(CHANNELS): row[n]=float(p[h,c]); row[n+'_actual']=float(y[h,c]) if m[h,c] else None
            rows.append(row)
    if not predictions: raise ValueError('No evaluation origins')
    p,y,m,b=map(np.array,(predictions,targets,masks,baseline))
    result=metrics(p,y,m,pre['target_scale'],b)
    result.update(corrections=corrections,inference_seconds=time.monotonic()-start)
    result['paired_baseline_difference']=paired_bootstrap(snapshot.dates[origins],daily_losses(p,y,m,pre['target_scale']),daily_losses(b,y,m,pre['target_scale']))
    quarters=snapshot.dates[origins].tz_localize(None).to_period('Q').astype(str)
    result['quarterly']={q:metrics(p[quarters==q],y[quarters==q],m[quarters==q],pre['target_scale'],b[quarters==q])['score'] for q in sorted(set(quarters))}
    regimes=m.sum(axis=(1,2))==21
    result['coverage_regimes']={name:metrics(p[idx],y[idx],m[idx],pre['target_scale'],b[idx])['score'] for name,idx in [('complete_window',regimes),('partial_window',~regimes)] if idx.any()}
    if path:
        path.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_parquet(path/'predictions.parquet',index=False); write_json(path/'metrics.json',result)
    if tracker: tracker.log({'validation/'+k:v for k,v in result.items() if isinstance(v,(float,int))})
    return result
