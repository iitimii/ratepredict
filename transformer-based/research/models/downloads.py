"""Pinned model downloads, exact configuration checks, pipeline-owned cache rotation."""
import json
import shutil
from pathlib import Path
from ..core import State,ResourceBlocked,file_hash,write_json,GIB


def model_spec(cfg,name):
    return cfg['training']['chronos'] if name=='chronos' else cfg['models'][name]


def download(cfg,name,rotate=False):
    from huggingface_hub import HfApi,snapshot_download,hf_hub_download
    state=State(cfg); spec=model_spec(cfg,name); cache=state.root/'backbone-cache'
    marker=state.root/'backbone-current.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        if old['repository']!=spec['repository']:
            if not rotate: raise ResourceBlocked('A different backbone is cached; use the sequential supervisor to rotate it')
            # Only our dedicated reproducible cache, never the shared Hugging Face cache.
            if cache.exists(): shutil.rmtree(cache)
            marker.unlink()
    info=HfApi().model_info(spec['repository'],revision=spec['revision'],files_metadata=True)
    if info.sha!=spec['revision']: raise ValueError('Model revision was not resolved exactly')
    size=sum(s.size or 0 for s in info.siblings if s.rfilename.endswith('.safetensors'))
    patterns=['*.safetensors','*.json','*.model','tokenizer*','vocab*','merges.txt']
    existing=sum(p.stat().st_size for p in cache.rglob('*') if p.is_file() and not p.is_symlink()) if cache.exists() else 0
    # Downloads are resumed in place by HF; retain a bounded checkpoint/log allowance.
    reserve=max(0,size-existing)+300*1024**2
    with state.reserve(reserve,'backbone plus resumable checkpoint'):
        local=Path(snapshot_download(spec['repository'],revision=spec['revision'],cache_dir=cache,
                                    allow_patterns=patterns,max_workers=1))
        files={p.name:file_hash(p) for p in local.glob('*.safetensors')}
        if not files: raise ValueError('No safetensors weights downloaded')
        conf=json.loads((local/'config.json').read_text())
        if name.startswith('qwen'):
            official_path=hf_hub_download(spec['official'],'config.json',cache_dir=state.root/'model-configs')
            official=json.loads(Path(official_path).read_text()); actual=conf.get('text_config',conf); expected=official.get('text_config',official)
            for key in ['model_type','hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','vocab_size']:
                if actual.get(key)!=expected.get(key): raise ValueError('MLX conversion differs from official configuration: '+key)
            quant=conf.get('quantization',conf.get('quantization_config',{}))
            if quant.get('bits')!=4: raise ValueError('Expected four-bit MLX frozen weights')
        write_json(marker,{'model':name,'repository':spec['repository'],'revision':info.sha,
                           'path':str(local),'weight_bytes':size,'checksums':files})
    return local
