"""Calendar targets and immutable, lazy research snapshots. No production imports."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .core import ROOT, State, digest, file_hash, now, write_json

CHANNELS = ['avg_change', 'high_change', 'low_change']
OHLC = ['open', 'high', 'low', 'close']
ACTIVITY = ['volume', 'trade_count', 'buy_volume', 'sell_volume', 'buy_count',
            'sell_count', 'large_trade_count', 'large_trade_volume', 'btcngn_volume', 'btcngn_trade_count']
PLACEHOLDERS = set(ACTIVITY) - {'volume', 'btcngn_volume'}
VERSION = 'quidax-sampled-mean-v2'


def validate_bars(frame, cutoff=None, timestamp_semantics='start'):
    if timestamp_semantics not in ('start', 'end'):
        raise ValueError('Explicit bucket start/end semantics required')
    f = frame.copy()
    f['bucket_2h'] = pd.to_datetime(f['bucket_2h'], utc=True, errors='coerce', format='mixed')
    if timestamp_semantics == 'end':
        f['bucket_2h'] -= pd.Timedelta(hours=2)
    for c in OHLC + [c for c in ACTIVITY if c in f]:
        f[c] = pd.to_numeric(f[c], errors='coerce')
    cutoff = pd.Timestamp(cutoff or now())
    reasons = pd.Series('', index=f.index)
    def reject(mask, reason):
        reasons.loc[mask] += reason + ';'
    reject(f.bucket_2h.isna(), 'invalid_timestamp')
    reject(f.bucket_2h != f.bucket_2h.dt.floor('2h'), 'unaligned_bucket')
    reject(f.bucket_2h + pd.Timedelta(hours=2) > cutoff, 'unfinished')
    reject(~np.isfinite(f[OHLC]).all(axis=1) | (f[OHLC] <= 0).any(axis=1), 'invalid_price')
    reject((f.high < f[['open','close','low']].max(axis=1)) |
           (f.low > f[['open','close','high']].min(axis=1)), 'impossible_ohlc')
    # Conflicting versions are quarantined, never resolved using future outcomes.
    exact = f.drop_duplicates(['bucket_2h'] + OHLC)
    conflicts = exact.loc[exact.bucket_2h.notna() & exact.bucket_2h.duplicated(False), 'bucket_2h']
    reject(f.bucket_2h.isin(conflicts), 'conflicting_revision')
    f['quality_reason'] = reasons
    rejected = f.loc[reasons != ''].copy()
    valid = f.loc[reasons == ''].drop_duplicates(['bucket_2h'] + OHLC).sort_values('bucket_2h').copy()
    for c in ACTIVITY:
        if c in valid:
            valid.loc[valid[c] < 0, c] = np.nan
    # Extreme market movements remain. Flags are trailing, and cannot see tomorrow.
    changes = valid.close.diff()
    med = changes.shift(1).rolling(60, min_periods=20).median()
    mad = changes.shift(1).rolling(60, min_periods=20).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    valid['anomaly_flag'] = ((changes-med).abs() > 8*mad.clip(lower=1e-6)) | (valid.high > 3*valid.close)
    return valid, rejected


def aggregate_targets(bars):
    f = bars.copy()
    f['date'] = f.bucket_2h.dt.floor('D')
    grouped = f.groupby('date', sort=True)
    daily = grouped.agg(average=('close','mean'), high=('high','max'), low=('low','min'),
                        open=('open','first'), close=('close','last'), bar_count=('bucket_2h','nunique'),
                        anomaly_count=('anomaly_flag','sum'))
    for c in ACTIVITY:
        if c in f:
            daily[c] = grouped[c].sum(min_count=1)
    daily['coverage'] = daily.bar_count / 12
    daily['complete'] = daily.bar_count.eq(12)
    return daily


def labels(rates, origin, horizon=7):
    future = rates[origin+1:origin+1+horizon]
    if len(future) != horizon or not np.isfinite(rates[origin,0]):
        raise ValueError('Origin or forecast window unavailable')
    y = np.column_stack((future[:,0]-rates[origin,0],future[:,1]-future[:,0],future[:,2]-future[:,0]))
    mask = np.repeat(np.isfinite(future).all(axis=1)[:,None],3,axis=1)
    return np.where(mask, y, 0).astype('float32'), mask


def decode(average, upper, lower, anchor):
    """Projection affects forecasts only; returns exact required channel order."""
    a,u,l = np.broadcast_arrays(np.asarray(average),np.asarray(upper),np.asarray(lower))
    if not np.isfinite(np.stack([a,u,l])).all() or not np.isfinite(anchor) or anchor <= 0:
        raise ValueError('Nonfinite forecast or invalid reconstruction anchor')
    aa = np.maximum(a, 0.001)
    uu = np.maximum(u, 0)
    ll = np.minimum(np.maximum(l, 0), aa-0.001)
    corrections = int(np.count_nonzero(a != aa)+np.count_nonzero(u != uu)+np.count_nonzero(l != ll))
    return np.stack((aa-anchor,uu,-ll),axis=-1).astype('float32'), corrections


def output_records(prediction):
    p = np.asarray(prediction)
    if p.shape != (7,3) or not np.isfinite(p).all():
        raise ValueError('Forecast must contain finite float32 [7,3]')
    return [dict(day=i+1, **dict(zip(CHANNELS, map(float,row)))) for i,row in enumerate(p)]


def point_in_time(observations, dates):
    """Daily end-of-day views of released records. Later revisions do not rewrite earlier rows.

    Each row is the latest observed period available at that boundary, then latest
    available revision of that period. Unknown release timestamps are excluded.
    """
    if observations.empty:
        return pd.DataFrame(index=dates), pd.DataFrame(index=dates)
    obs = observations.copy()
    obs['available_at'] = pd.to_datetime(obs.available_at, utc=True, errors='coerce')
    obs['event_time'] = pd.to_datetime(obs.event_time, utc=True, errors='coerce')
    obs = obs.dropna(subset=['available_at','event_time','value'])
    values, ages = {}, {}
    for series,g in obs.groupby('series'):
        g = g.sort_values(['available_at','event_time','id'])
        rows = list(g.to_dict('records')); k=0; known={}; val=[]; age=[]
        for day in dates:
            boundary = day + pd.Timedelta(days=1)
            while k < len(rows) and rows[k]['available_at'] <= boundary:
                r=rows[k]; known[r['event_time']] = r; k+=1
            if known:
                r=known[max(known)]
                val.append(r['value']); age.append(max(0, (boundary-r['event_time']).total_seconds()/86400))
            else:
                val.append(np.nan); age.append(np.nan)
        values[series]=val; ages[series]=age
    return pd.DataFrame(values,index=dates),pd.DataFrame(ages,index=dates)


def freeze_snapshot(cfg, name='S0', include_text=False):
    state=State(cfg)
    files=[ROOT/'data/usdngn training data - usdngn training data.csv',ROOT/'data/latest/quidax_runtime_2h.csv']
    with state.connect() as db:
        # Connector-generated files are content-addressed and never replace imports.
        for r in db.execute("SELECT result FROM jobs WHERE kind='quidax' AND status='done'"):
            result=json.loads(r[0] or '{}')
            if result.get('bars_path'): files.append(Path(result['bars_path']))
        observations=pd.DataFrame([dict(r) for r in db.execute("SELECT * FROM observations WHERE quality='valid'")])
        source_rows=[json.loads(r[0]) for r in db.execute('SELECT metadata FROM sources ORDER BY id')]
    frames=[]; provenance=[]
    for p in files:
        if not p.exists(): continue
        f=pd.read_csv(p); f['import_path']=str(p.relative_to(ROOT))
        if 'runtime' in p.name or p.is_relative_to(state.root):
            for c in PLACEHOLDERS:
                if c in f: f[c]=np.nan
        frames.append(f); provenance.append({'path':str(p),'sha256':file_hash(p)})
    if not frames: raise ValueError('No Quidax inputs found')
    valid,rejected=validate_bars(pd.concat(frames,ignore_index=True),timestamp_semantics=cfg['bucket_timestamp'])
    daily=aggregate_targets(valid)
    dates=pd.date_range(pd.Timestamp(cfg['calendar_start'],tz='UTC'),daily.index.max(),freq='D')
    daily=daily.reindex(dates)
    target=daily[['average','high','low']].where(daily.complete.eq(True),np.nan).copy()
    target.columns=['target_average','target_high','target_low']
    numerical=daily.drop(columns=['complete']).add_prefix('quidax_')
    age=pd.DataFrame(np.where(numerical.notna(),0,np.nan),index=dates,columns=numerical.columns)
    pit,pit_age=point_in_time(observations,dates)
    if not pit.empty:
        numerical=numerical.join(pit.add_prefix('external_')); age=age.join(pit_age.add_prefix('external_'))
    calendar=pd.DataFrame({'weekday_sin':np.sin(2*np.pi*dates.dayofweek/7),
                           'weekday_cos':np.cos(2*np.pi*dates.dayofweek/7),
                           'month_sin':np.sin(2*np.pi*(dates.month-1)/12),
                           'month_cos':np.cos(2*np.pi*(dates.month-1)/12)},index=dates)
    numerical=numerical.join(calendar); age=age.join(calendar*0)
    features=list(numerical.columns)
    table=numerical.join(target)
    table.index.name='date'; age.index.name='date'
    rates=target.to_numpy(dtype='float64'); origins=[]
    h=cfg['horizon_days']; lb=cfg['lookback_days']
    for i in range(lb-1,len(dates)-h):
        if np.isfinite(rates[i,0]) and np.isfinite(rates[i+1:i+1+h]).all(axis=1).any(): origins.append(i)
    folds={}
    for key,f in cfg['folds'].items():
        cutoff=pd.Timestamp(f['train_end'],tz='UTC'); lo=pd.Timestamp(f['eval_start'],tz='UTC'); hi=pd.Timestamp(f['eval_end'],tz='UTC')
        folds[key]={'train':[i for i in origins if dates[i+h]<=cutoff],
                    'eval':[i for i in origins if dates[i+1]>=lo and dates[i+h]<=hi], 'locked':key=='final'}
    # Store sparse vectors only; do not allocate the 20-year source/day tensor here.
    text=[]
    if include_text:
        for p in sorted((state.root/'embeddings').glob('*.parquet')):
            text.append(pd.read_parquet(p))
    text_frame=pd.concat(text,ignore_index=True) if text else pd.DataFrame(columns=['date','source','embedding','document_count'])
    if not text_frame.empty:
        # A cache may contain multiple versions; selecting mixed encoders is invalid.
        if text_frame.encoder_revision.nunique()>1: raise ValueError('Mixed encoder revisions; choose one before snapshot')
        text_frame=text_frame.drop_duplicates(['date','source','content_hash'])
    source_ids=sorted({str(s.get('source_id',s.get('id'))) for s in source_rows})
    audit={'valid_bars':len(valid),'rejected_bars':len(rejected),'complete_days':int(daily.complete.eq(True).sum()),
           'partial_days':int(((daily.bar_count>0)&(daily.bar_count<12)).sum()),'eligible_origins':len(origins),
           'flagged_bars':int(valid.anomaly_flag.sum()),'rejection_reasons':rejected.quality_reason.value_counts().to_dict(),
           'bucket_semantics':'start; runtime extraction drops buckets >= current floor(2h); historical export uses same bucket_2h convention',
           'target_version':VERSION,'reference_USDt':'excluded: conflicting duplicate and unresolved availability',
           'unreleased_macro_policy':'excluded until validated available_at exists',
           'fold_counts':{k:{j:len(v[j]) for j in ('train','eval')} for k,v in folds.items()}}
    with state.reserve(80*1024**2,'snapshot and temporary build'):
        parent=state.root/'snapshots'; parent.mkdir(exist_ok=True)
        temp=Path(tempfile.mkdtemp(prefix='.building-',dir=parent))
        try:
            table.to_parquet(temp/'daily.parquet'); age.to_parquet(temp/'age.parquet'); text_frame.to_parquet(temp/'text.parquet')
            rejected.to_parquet(temp/'quarantined_bars.parquet',index=False)
            files_hash={p.name:file_hash(p) for p in sorted(temp.glob('*.parquet'))}
            identity={'version':VERSION,'features':features,'sources':source_ids,'lookback':lb,'horizon':h,
                      'inputs':provenance,'files':files_hash,'folds':folds,'label_policy':'masked',
                      'availability_policy':'UTC_end_of_day_v1','channels':CHANNELS}
            sid=digest(identity)[:20]; path=parent/sid
            manifest=dict(identity,id=sid,created_at=now(),audit=audit)
            write_json(temp/'manifest.json',manifest)
            if path.exists(): shutil.rmtree(temp)
            else: temp.rename(path)
        except BaseException:
            shutil.rmtree(temp,ignore_errors=True); raise
    write_json(state.root/f'{name}.json',{'snapshot':sid,'path':str(path)})
    return manifest


class Snapshot:
    def __init__(self, path, verify=True):
        self.path=Path(path); self.manifest=json.loads((self.path/'manifest.json').read_text())
        if verify:
            for name,h in self.manifest['files'].items():
                if file_hash(self.path/name)!=h: raise ValueError('Snapshot artifact changed: '+name)
        self.frame=pd.read_parquet(self.path/'daily.parquet'); self.dates=self.frame.index
        self.names=self.manifest['features']; self.x=self.frame[self.names].to_numpy(dtype='float32')
        self.mask=np.isfinite(self.x); self.age=pd.read_parquet(self.path/'age.parquet').to_numpy(dtype='float32')
        self.rates=self.frame[['target_average','target_high','target_low']].to_numpy(dtype='float64')
        self.text=pd.read_parquet(self.path/'text.parquet')
        self.lb=self.manifest['lookback']; self.h=self.manifest['horizon']

    def origins(self, fold, part, unlock=False):
        f=self.manifest['folds'][fold]
        if part=='eval' and f['locked'] and not unlock: raise ValueError('Locked test requires a frozen selection manifest')
        return f[part]

    def fit_preprocessing(self, origins):
        if not len(origins): raise ValueError('No training origins')
        dates=np.unique(np.concatenate([np.arange(i-self.lb+1,i+1) for i in origins]))
        x=self.x[dates].copy(); active=np.isfinite(x).any(axis=0)
        names=[n for n,a in zip(self.names,active) if a]; x=x[:,active]
        logs=np.array([any(c in n for c in ('volume','count')) for n in names])
        x[:,logs]=np.log1p(np.maximum(x[:,logs],0))
        center=np.nanmedian(x,axis=0); scale=np.nanquantile(x,.75,axis=0)-np.nanquantile(x,.25,axis=0)
        scale=np.where(scale>1e-6,scale,1)
        ys=[]
        for i in origins:
            y,m=labels(self.rates,i,self.h); ys.append(np.where(m,y,np.nan))
        y=np.stack(ys)
        target_scale=np.nanquantile(y,.75,axis=(0,1))-np.nanquantile(y,.25,axis=(0,1))
        target_scale=np.where(target_scale>1e-6,target_scale,1)
        sources=[]
        if not self.text.empty:
            training_dates=set(self.dates[dates])
            counts=self.text[self.text.date.map(lambda d:pd.Timestamp(d) in training_dates)].groupby('source').date.nunique()
            sources=sorted(counts[counts>=30].index)
        return {'active':active.tolist(),'names':names,'log1p':logs.tolist(),'center':center.tolist(),
                'scale':scale.tolist(),'target_scale':target_scale.tolist(),'text_sources':sources,
                'fit_origin_hash':digest(list(map(int,origins))),'unique_input_dates':len(dates)}

    def sample(self,i,pre=None):
        sl=slice(i-self.lb+1,i+1); raw=self.x[sl]; mask=self.mask[sl]; age=self.age[sl]
        if pre:
            active=np.array(pre['active']); raw=raw[:,active].copy(); mask=mask[:,active]; age=age[:,active]
            logs=np.array(pre['log1p']); raw[:,logs]=np.log1p(np.maximum(raw[:,logs],0))
            raw=(raw-pre['center'])/pre['scale']
        y,m=labels(self.rates,i,self.h)
        sources=pre.get('text_sources',[]) if pre else []
        text=np.zeros((self.lb,len(sources),384),dtype='float32'); tm=np.zeros((self.lb,len(sources)),bool)
        counts=np.zeros(tm.shape,dtype='int32')
        if sources:
            subset=self.text[(pd.to_datetime(self.text.date,utc=True)>=self.dates[i-self.lb+1]) &
                             (pd.to_datetime(self.text.date,utc=True)<=self.dates[i])]
            for (date,source),g in subset.groupby(['date','source']):
                if source not in sources: continue
                d=(pd.Timestamp(date)-self.dates[i-self.lb+1]).days; s=sources.index(source)
                v=np.mean(np.stack(g.embedding),axis=0); v=v/max(np.linalg.norm(v),1e-12)
                text[d,s]=v; tm[d,s]=True; counts[d,s]=g.document_count.sum()
        future_dates=self.dates[i+1:i+1+self.h]
        events=np.column_stack((np.sin(2*np.pi*future_dates.dayofweek/7),np.cos(2*np.pi*future_dates.dayofweek/7),
                                np.sin(2*np.pi*(future_dates.month-1)/12),np.cos(2*np.pi*(future_dates.month-1)/12))).astype('float32')
        return {'quantitative':np.where(mask,raw,0).astype('float32'),'quantitative_mask':mask,
                'quantitative_age':np.nan_to_num(age,nan=0).astype('float32'),
                'text':text,'text_mask':tm,'text_coverage':tm.astype('int8'),'document_count':counts,
                'known_events':events,'known_events_mask':np.ones_like(events,dtype=bool),
                'target':y,'target_mask':m,'anchor':float(self.rates[i,0]),
                'as_of':self.dates[i]+pd.Timedelta(days=1),'origin':i}

    def baseline(self,i):
        rates=self.rates[max(0,i-29):i+1]
        upper=np.nanmedian(rates[:,1]-rates[:,0]); lower=np.nanmedian(rates[:,0]-rates[:,2])
        return decode(np.repeat(self.rates[i,0],self.h),np.repeat(upper,self.h),np.repeat(lower,self.h),self.rates[i,0])[0]
