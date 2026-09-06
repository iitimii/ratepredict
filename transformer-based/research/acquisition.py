"""Bounded, persistent public-source acquisition. Indexes never count as bodies."""
from __future__ import annotations
import contextlib
import email.utils
import fcntl
import json
import re
import time
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urljoin,urlsplit,urldefrag
import xml.etree.ElementTree as ET

import httpx
import pandas as pd
from bs4 import BeautifulSoup

from .core import ROOT,State,ResourceBlocked,digest,now,write_json

RELEVANCE=re.compile(r'naira|nigeria|inflation|monetary|currency|exchange|forex|interest.rate|oil|treasury|gdp|capital.import|fomc|policy|auction|cpi|reserve|usdt',re.I)


def register_sources(state):
    inventory=ROOT/'outputs/unstructured-data/source_inventory.json'
    if not inventory.exists():
        raise FileNotFoundError('Run build_unstructured_report.py first to assemble all requested source definitions')
    data=json.loads(inventory.read_text()); sources=data['sources']
    sources += [{'id':'quidax','name':'Quidax USDT/NGN two-hour bars','kind':'quantitative',
                 'urls':['https://app.quidax.io/api/v1/markets/usdtngn/k?period=120&limit=1000'],
                 'origins':['local exports','app/services/market_data.py'],'status':'local_import_available'}]
    with state.connect() as db:
        for s in sources:
            s=dict(s); s['text_channel']=s.get('kind') not in ('quantitative','quantitative reference','dataset source') and not s['id'].startswith('macro_')
            s['required']=True
            db.execute('INSERT INTO sources VALUES(?,?) ON CONFLICT(id) DO UPDATE SET metadata=excluded.metadata',
                       (s['id'],json.dumps(s)))
    # Seed every existing record. Workers can process a bounded subset without losing pending counts.
    for r in data['records']:
        if not r.get('url','').startswith('http'): continue
        if r['kind']=='sitemap index': kind='sitemap'
        elif r['kind'] in ('document metadata','downloadable resource'): kind='document'
        else: continue
        state.enqueue(r['source_id'],kind,{'url':r['url'],'title':r.get('title',''),
                    'index_date':r.get('date'),'date_basis':r.get('date_basis'),'depth':0})
    for s in sources:
        for url in s.get('urls',[]):
            if not url.startswith('http'): continue
            if s['id']=='quidax': kind='quidax'
            elif 'sitemap' in url: kind='sitemap'
            elif 'feed' in url or 'rss' in url: kind='feed'
            else: kind='index'
            state.enqueue(s['id'],kind,{'url':url,'depth':0})
    # Reuse the source endpoint catalog, but not its unrestricted old downloader.
    from acquire_history import CBN_QUANT_ENDPOINTS,FRED_SERIES
    for name,(url,date_field) in CBN_QUANT_ENDPOINTS.items():
        state.enqueue('cbn_'+name,'cbn_quant',{'url':url,'date_field':date_field})
    for series in FRED_SERIES:
        state.enqueue('fred_'+series.lower(),'fred_quant',{'url':f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd=2006-01-01','series':series})
    return state.summary()


class AccessRestricted(RuntimeError): pass
class Retryable(RuntimeError):
    def __init__(self,message,delay=60): super().__init__(message); self.delay=delay


@contextlib.contextmanager
def network_lease(state,host):
    """Two global HTTP slots, one per host across processes; persisted pacing."""
    slot=None
    host_file=(state.root/('host-'+digest(host)[:16]+'.lock')).open('a')
    fcntl.flock(host_file,fcntl.LOCK_EX)
    try:
        while slot is None:
            for i in range(2):
                candidate=(state.root/f'http-{i}.lock').open('a')
                try: fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB); slot=candidate; break
                except BlockingIOError: candidate.close()
            if slot is None: time.sleep(.25)
        with state.connect() as db:
            r=db.execute('SELECT next_at FROM host_access WHERE host=?',(host,)).fetchone()
        if r: time.sleep(max(0,min(r[0]-time.time(),300)))
        yield
    finally:
        with state.connect() as db:
            db.execute('INSERT INTO host_access VALUES(?,?) ON CONFLICT(host) DO UPDATE SET next_at=excluded.next_at',
                       (host,time.time()+state.cfg['resources']['host_pause_seconds']))
        if slot: fcntl.flock(slot,fcntl.LOCK_UN); slot.close()
        fcntl.flock(host_file,fcntl.LOCK_UN); host_file.close()


def fetch(state,url):
    host=urlsplit(url).hostname
    if urlsplit(url).scheme not in ('https','http') or not host: raise ValueError('Public HTTP URL required')
    with network_lease(state,host),httpx.Client(timeout=state.cfg['resources']['request_timeout_seconds'],follow_redirects=True,
            headers={'User-Agent':'RatePredictResearch/1.0 (public economic research; bounded archive fetcher)'}) as client:
        with client.stream('GET',url) as response:
            if response.status_code in (401,403,451): raise AccessRestricted(f'HTTP {response.status_code} at {host}')
            if response.status_code==429 or response.status_code>=500:
                retry=response.headers.get('Retry-After','60')
                try: delay=float(retry)
                except ValueError:
                    try: delay=max(1,email.utils.parsedate_to_datetime(retry).timestamp()-time.time())
                    except (TypeError,ValueError): delay=60
                raise Retryable(f'HTTP {response.status_code} at {host}',delay)
            response.raise_for_status()
            max_bytes=int(state.cfg['resources']['max_download_mb']*1024**2)
            if int(response.headers.get('content-length',0))>max_bytes: raise ResourceBlocked('Document exceeds per-file budget')
            data=bytearray()
            with state.reserve(max_bytes,'bounded HTTP response'):
                for chunk in response.iter_bytes():
                    data.extend(chunk)
                    if len(data)>max_bytes: raise ResourceBlocked('Document exceeds per-file budget')
            return bytes(data),dict(response.headers),str(response.url)


def release_day(value):
    try:
        stamp=pd.Timestamp(value)
        if pd.isna(stamp): return None
        if stamp.tzinfo is None: stamp=stamp.tz_localize('UTC')
        else: stamp=stamp.tz_convert('UTC')
        # Midnight or date-only information conservatively enters at next boundary.
        if stamp==stamp.normalize(): stamp+=pd.Timedelta(days=1)
        if stamp>pd.Timestamp.now(tz='UTC')+pd.Timedelta(days=1): return None
        return stamp.isoformat()
    except (ValueError,TypeError): return None


def parse_quant(state,job,content,raw_hash):
    p=json.loads(job['payload']); stamp=now(); records=[]
    if job['kind']=='fred_quant':
        import io
        f=pd.read_csv(io.BytesIO(content)); field=f.columns[0]
        for row in f.to_dict('records'):
            records.append((p['series'],row[field],row.get(p['series']),'source_units'))
    else:
        payload=json.loads(content)
        if isinstance(payload,dict): payload=payload.get('data',payload.get('Data',[]))
        for row in payload:
            if not isinstance(row,dict): continue
            date=row.get(p['date_field'])
            for key,value in row.items():
                if key==p['date_field'] or key.lower() in ('id','year','month','currency','currencytype'): continue
                records.append((job['source_id']+'_'+key,date,value,'source_units'))
    count=0
    with state.connect() as db:
        for series,date,value,unit in records:
            try:
                v=float(str(value).replace(',','')); dt=pd.Timestamp(date)
                if pd.isna(dt) or not (-1e20<v<1e20): continue
                if dt.tzinfo is None: dt=dt.tz_localize('UTC')
            except (ValueError,TypeError): continue
            # Current downloads are NOT historical vintages. Safe for future forecasts only.
            key=digest([job['source_id'],series,str(dt),v,raw_hash])
            db.execute('INSERT OR IGNORE INTO observations VALUES(?,?,?,?,?,?,?,?,?,?)',
                (key,job['source_id'],series,dt.isoformat(),stamp,stamp,v,unit,'valid',raw_hash)); count+=1
    return {'observations':count,'availability_policy':'retrieval_time; historical release/vintage unverified'}


def process(state,job):
    p=json.loads(job['payload']); source=job['source_id']; kind=job['kind']
    body,headers,url=fetch(state,p['url']); h,path=state.raw(body,source,url,headers)
    if kind in ('cbn_quant','fred_quant'): return parse_quant(state,job,body,h)
    if kind=='quidax':
        payload=json.loads(body); rows=payload.get('data',payload)
        if isinstance(rows,dict): rows=rows.get('candles',rows.get('klines',[]))
        if not rows: raise ValueError('Quidax response contains no bars')
        f=pd.DataFrame(rows)
        if len(f.columns)==6: f.columns=['timestamp','open','high','low','close','volume']
        if 'timestamp' not in f: raise ValueError('Unrecognized Quidax schema')
        f['bucket_2h']=pd.to_datetime(pd.to_numeric(f.timestamp),unit='s',utc=True)
        f=f.loc[f.bucket_2h+pd.Timedelta(hours=2)<=pd.Timestamp.now(tz='UTC')].drop(columns=['timestamp'])
        output=state.root/'parsed'/f'{h}.csv'; output.parent.mkdir(exist_ok=True)
        f.to_csv(output,index=False)
        return {'bars_path':str(output),'bars':len(f),'raw_hash':h}
    if kind in ('sitemap','feed'):
        root=ET.fromstring(body); count=0; index=root.tag.split('}')[-1]=='sitemapindex'
        for elem in root:
            items={e.tag.split('}')[-1]:e for e in elem}
            link=items.get('loc') if kind=='sitemap' else items.get('link')
            if link is None: continue
            child=(link.text or link.attrib.get('href','')).strip()
            if not child.startswith('http'): continue
            title=items.get('title'); title=title.text if title is not None else ''
            next_kind='sitemap' if index else 'document'
            if next_kind=='document' and not RELEVANCE.search(child+' '+(title or '')): continue
            if p.get('depth',0)>=4 and next_kind=='sitemap': continue
            # Sitemap lastmod is deliberately not publication time.
            state.enqueue(source,next_kind,{'url':urldefrag(child)[0],'title':title or '',
                          'depth':p.get('depth',0)+1}); count+=1
        return {'enumerated_urls':count,'is_index':index,'raw_hash':h}
    if kind=='index':
        soup=BeautifulSoup(body,'html.parser'); n=0
        for a in soup.select('a[href]'):
            child=urljoin(url,a['href']); title=a.get_text(' ',strip=True)
            if urlsplit(child).hostname!=urlsplit(url).hostname: continue
            if child==url or not RELEVANCE.search(title+' '+child): continue
            if not (child.lower().endswith(('.pdf','.xlsx','.xls','.zip')) or
                    re.search(r'/20\d{2}/|/pressreleases/|/monetary-policy-',child)): continue
            state.enqueue(source,'document',{'url':urldefrag(child)[0],'title':title,'depth':0}); n+=1
        return {'enumerated_urls':n,'raw_hash':h}
    if kind=='document':
        payload=dict(p,path=str(path),raw_hash=h,content_type=headers.get('content-type',''),retrieved_at=now(),url=url)
        state.enqueue(source,'extract',payload)
        return {'raw_hash':h,'extraction_queued':True}
    raise ValueError('No handler for '+kind)


def work(cfg,limit=20,kinds=None):
    state=State(cfg); completed=0
    for _ in range(limit):
        state.check_memory(background=True)
        job=state.claim(kinds or ['quidax','cbn_quant','fred_quant','document','sitemap','feed','index'])
        if job is None: break
        try:
            result=process(state,job); state.finish(job['id'],'done',result); completed+=1
        except AccessRestricted as e:
            state.finish(job['id'],'suspended',error=str(e))
            with state.connect() as db:
                db.execute("UPDATE jobs SET status='suspended',error=? WHERE source_id=? AND status IN ('queued','retry')",
                           (str(e),job['source_id']))
        except ResourceBlocked as e:
            state.finish(job['id'],'retry',error=str(e),delay=300); break
        except Exception as e:
            attempts=job['attempts']+1; delay=max(getattr(e,'delay',0),min(3600,30*2**attempts))
            state.finish(job['id'],'failed' if attempts>=5 else 'retry',error=f'{type(e).__name__}: {str(e)[:300]}',delay=delay)
        write_json(state.root/'acquisition-status.json',{'at':now(),**state.summary(),**state.metrics()})
    return {'completed':completed,**state.summary()}


def schedule_refresh(state):
    stamp=pd.Timestamp.now(tz='UTC'); generations={'quidax':str(stamp.floor('2h')),'feed':str(stamp.floor('h')),
                                                  'index':str(stamp.floor('D')),'cbn_quant':str(stamp.floor('D')),'fred_quant':str(stamp.floor('D'))}
    with state.connect() as db:
        rows=db.execute("SELECT source_id,kind,payload FROM jobs WHERE kind IN ('quidax','feed','index','cbn_quant','fred_quant') GROUP BY source_id,kind,payload").fetchall()
    for row in rows: state.enqueue(row['source_id'],row['kind'],json.loads(row['payload']),generations[row['kind']])
