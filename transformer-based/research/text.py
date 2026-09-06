"""Versioned extraction and local tokenizer-aware document embeddings."""
from __future__ import annotations
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from .core import State,ResourceBlocked,digest,write_json,now
from .acquisition import release_day

EXTRACTION_VERSION='body-v1'
POOLING_VERSION='document-equal-source-day-v1'


def publication_metadata(html):
    soup=BeautifulSoup(html,'html.parser'); candidates=[]
    for name in ['article:published_time','datePublished','citation_publication_date','DC.date.issued']:
        tag=soup.find('meta',attrs={'property':name}) or soup.find('meta',attrs={'name':name})
        if tag and tag.get('content'): candidates.append((tag['content'],'html:'+name))
    def walk(x):
        if isinstance(x,list):
            for v in x: walk(v)
        if isinstance(x,dict):
            if x.get('datePublished'): candidates.append((x['datePublished'],'jsonld:datePublished'))
            for v in x.values():
                if isinstance(v,(dict,list)): walk(v)
    for script in soup.select('script[type="application/ld+json"]'):
        try: walk(json.loads(script.string or script.get_text()))
        except (ValueError,TypeError): pass
    published=None; basis=None
    for value,source in candidates:
        if release_day(value): published=release_day(value); basis=source; break
    modified=soup.find('meta',attrs={'property':'article:modified_time'})
    canonical=soup.find('link',attrs={'rel':'canonical'})
    return {'available_at':published,'date_basis':basis,
            'modified_at':release_day(modified.get('content')) if modified else None,
            'canonical_url':canonical.get('href') if canonical else None}


def extract_document(payload):
    path=Path(payload['path']); data=path.read_bytes(); result=dict(payload)
    result.update(extraction_version=EXTRACTION_VERSION,available_at=None,date_basis='unknown',sections=[])
    if data.startswith(b'%PDF'):
        from pypdf import PdfReader
        pages=PdfReader(path).pages
        texts=[p.extract_text() or '' for p in pages]
        if sum(len(t.strip()) for t in texts)<100:
            if not shutil.which('pdftoppm') or not shutil.which('tesseract'):
                result.update(status='ocr_required',text=''); return result
            # One worker, bounded pilot; retain unprocessed pages explicitly.
            with tempfile.TemporaryDirectory() as tmp:
                subprocess.run(['pdftoppm','-r','120','-f','1','-l',str(min(len(pages),20)),'-png',str(path),str(Path(tmp)/'page')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=180)
                texts=[subprocess.run(['tesseract',str(p),'stdout'],capture_output=True,text=True,check=True,timeout=60).stdout for p in sorted(Path(tmp).glob('*.png'))]
            result['ocr_pages']=len(texts)
            result['extraction_truncated']=len(pages)>20
        result['sections']=[{'page':i+1,'text':t} for i,t in enumerate(texts)]
        text='\n\n'.join(texts)
        # Confirm an index date against the document's printed day, not its PDF creation timestamp.
        if payload.get('index_date'):
            date=pd.Timestamp(payload['index_date'])
            variants=[date.strftime('%d %B %Y').lstrip('0'),date.strftime('%B %d, %Y').replace(' 0',' '),date.strftime('%d/%m/%Y')]
            compact=re.sub(r'\s+',' ',text[:8000]).lower()
            if any(v.lower() in compact for v in variants):
                result.update(available_at=release_day(payload['index_date']),date_basis='index_date_confirmed_in_document')
    elif data.startswith(b'PK') or payload.get('url','').lower().endswith(('.xls','.xlsx','.zip')):
        result.update(status='table_package_pending_parser',text=''); return result
    else:
        import trafilatura
        html=data.decode('utf-8',errors='replace'); result.update(publication_metadata(html))
        text=trafilatura.extract(html,include_tables=True,include_comments=False,favor_precision=True) or ''
        result['sections']=[{'section':'body','text':text}]
    text=re.sub(r'[ \t]+',' ',text).strip()
    result['text']=text; result['content_hash']=digest(text.encode())
    if len(text)<100: result['status']='extraction_failed'; return result
    from langdetect import detect,DetectorFactory
    DetectorFactory.seed=42
    try: language=detect(text)
    except Exception: language='unknown'
    result['language']=language
    result['status']='ready' if language=='en' and result['available_at'] and not result.get('extraction_truncated') else ('unsupported_language' if language!='en' else 'publication_date_unverified')
    return result


def extract_jobs(cfg,limit=10):
    state=State(cfg); count=0
    for _ in range(limit):
        state.check_memory(background=True); job=state.claim(['extract'])
        if job is None: break
        try:
            payload=json.loads(job['payload']); result=extract_document(payload); result['source']=job['source_id']
            key=digest([job['source_id'],payload['raw_hash'],EXTRACTION_VERSION])
            path=state.root/'documents'/f'{key}.parquet'; path.parent.mkdir(exist_ok=True)
            row=dict(result); row['sections']=json.dumps(row['sections'])
            with state.reserve(max(len(result['text'].encode())*3,1024**2),'extracted document'):
                pd.DataFrame([row]).to_parquet(path,index=False)
            with state.connect() as db:
                db.execute('INSERT OR IGNORE INTO documents VALUES(?,?,?,?,?)',(key,job['source_id'],payload['raw_hash'],result.get('available_at'),json.dumps(dict(result,path=str(path)))))
            state.finish(job['id'],'done',{'document_id':key,'status':result['status']}); count+=1
        except ResourceBlocked as e:
            state.finish(job['id'],'retry',error=str(e),delay=300); break
        except Exception as e: state.finish(job['id'],'failed',error=f'{type(e).__name__}: {str(e)[:250]}')
    return {'extracted':count}


def chunks(tokenizer,text,title='',body_tokens=384,overlap=64,max_tokens=512):
    if not 0<=overlap<body_tokens: raise ValueError('Invalid chunk overlap')
    title_ids=tokenizer.encode(title,add_special_tokens=False)[:64]
    available=min(body_tokens,max_tokens-len(title_ids)-tokenizer.num_special_tokens_to_add(pair=False))
    if available<=overlap: raise ValueError('Title leaves insufficient body tokens')
    tokens=tokenizer.encode(text,add_special_tokens=False)
    out=[]
    for start in range(0,len(tokens),available-overlap):
        ids=title_ids+tokens[start:start+available]
        ids=tokenizer.build_inputs_with_special_tokens(ids)
        if len(ids)>max_tokens: raise AssertionError('Tokenizer limit exceeded')
        out.append({'input_ids':ids,'attention_mask':[1]*len(ids),'body_start':start})
        if start+available>=len(tokens): break
    return out


def unit(vector):
    vector=np.asarray(vector,dtype='float32'); return vector/max(float(np.linalg.norm(vector)),1e-12)


def embed_jobs(cfg,limit=50,device='cpu'):
    import contextlib
    import torch
    from transformers import AutoTokenizer,AutoModel
    from huggingface_hub import HfApi
    state=State(cfg); spec=cfg['embedding']; directory=state.root/'embeddings'; directory.mkdir(exist_ok=True)
    with state.connect() as db:
        docs=[(r['id'],json.loads(r['data'])) for r in db.execute('SELECT * FROM documents WHERE available_at IS NOT NULL ORDER BY id')]
    docs=[(key,d) for key,d in docs if d['status']=='ready']
    if not docs:return {'embedded':0,'reason':'No extracted, publication-validated English documents'}
    revision=spec.get('revision')
    if not revision:
        pin=state.root/'embedding-pin.json'
        if pin.exists():revision=json.loads(pin.read_text())['revision']
        else:
            revision=HfApi().model_info(spec['repository']).sha
            write_json(pin,{'repository':spec['repository'],'revision':revision})
    selected=[]; seen=set()
    for key,d in docs:
        cache=digest([d['content_hash'],revision,EXTRACTION_VERSION,spec,POOLING_VERSION])
        if cache in seen or (directory/(cache+'.parquet')).exists():continue
        selected.append((cache,d)); seen.add(cache)
        if len(selected)>=limit:break
    if not selected:return {'embedded':0,'reason':'All eligible content cached'}
    lease=state.gpu() if device=='mps' else contextlib.nullcontext()
    with lease,state.reserve(400*1024**2,'embedding model and vector artifacts'):
        state.check_memory(background=True); torch.set_num_threads(2)
        cache_dir=state.root/'embedding-model'
        tok=AutoTokenizer.from_pretrained(spec['repository'],revision=revision,cache_dir=cache_dir)
        model=AutoModel.from_pretrained(spec['repository'],revision=revision,cache_dir=cache_dir).to(device).eval()
        for key,d in selected:
            state.check_memory(background=True)
            parts=chunks(tok,d['text'],d.get('title',''),spec['chunk_tokens'],spec['overlap_tokens'],spec['max_tokens']); vectors=[]
            for part in parts:
                batch={k:torch.tensor([part[k]],device=device) for k in ('input_ids','attention_mask')}
                with torch.no_grad(): vector=model(**batch).last_hidden_state[:,0].float().cpu().numpy()[0]
                vectors.append(unit(vector))
            pooled=unit(np.mean(vectors,axis=0))
            available=pd.Timestamp(d['available_at']); day=(available-pd.Timedelta(nanoseconds=1)).floor('D')
            row={'date':day,'source':d['source'],'embedding':pooled.tolist(),'document_count':1,
                 'chunk_count':len(parts),'content_hash':d['content_hash'],'encoder_revision':revision,
                 'extraction_version':EXTRACTION_VERSION,'pooling_version':POOLING_VERSION,
                 'available_at':available,'chunk_embeddings':[v.tolist() for v in vectors],
                 'chunk_token_offsets':[p['body_start'] for p in parts]}
            pd.DataFrame([row]).to_parquet(directory/(key+'.parquet'),index=False)
        del model
        if device=='mps':torch.mps.empty_cache()
    return {'embedded':len(selected),'encoder_revision':revision,'dimensions':384}
