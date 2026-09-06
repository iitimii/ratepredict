"""Local durable metrics, W&B online setup gate, and resumable run identities."""
from __future__ import annotations
import json
import os
from pathlib import Path
import time

from .core import State, digest, now, write_json


class TrackingBlocked(RuntimeError):
    pass


class Tracker:
    def __init__(self,cfg,run_id,settings,online=True):
        self.cfg=cfg; self.id=run_id; self.path=Path(cfg['runtime'])/'runs'/run_id
        self.path.mkdir(parents=True,exist_ok=True); self.run=None; self.step=0
        self.journal=self.path/'metrics.jsonl'
        if self.journal.exists():
            for line in self.journal.read_text().splitlines():
                self.step=max(self.step,json.loads(line)['step']+1)
        write_json(self.path/'config.json',settings)
        if online:
            import wandb
            # No secrets are read into logs or stored in configuration.
            try:
                if not wandb.login(anonymous='never',relogin=False,force=False,timeout=10):
                    raise TrackingBlocked('Authenticate with transformer-based/.venv/bin/wandb login')
                self.run=wandb.init(project=cfg['tracking']['project'],entity=cfg['tracking'].get('entity'),
                    id=run_id,resume='allow',config=settings,dir=str(self.path),mode='online',
                    settings=wandb.Settings(init_timeout=45))
                if self.run is None or self.run.offline: raise TrackingBlocked('Initial W&B run must be online')
                write_json(Path(cfg['runtime'])/'tracking-ready.json',{'checked_at':now(),'url':self.run.url})
            except Exception as e:
                raise TrackingBlocked(f'W&B online setup failed ({type(e).__name__}); use wandb login and retry') from e

    def log(self,metrics):
        entry={'step':self.step,'time':now(),**metrics}
        with self.journal.open('a') as f:
            f.write(json.dumps(entry,allow_nan=False,default=str)+'\n'); f.flush(); os.fsync(f.fileno())
        if self.run:
            try: self.run.log(metrics,step=self.step)
            except Exception:
                # W&B also buffers its own stream; this journal remains authoritative.
                write_json(self.path/'tracking-buffered.json',{'at':now(),'next_step':self.step})
        self.step+=1

    def artifact(self,path,name,kind='evaluation'):
        if self.run:
            import wandb
            artifact=wandb.Artifact(name,type=kind)
            artifact.add_file(str(path)); self.run.log_artifact(artifact)

    def close(self,status='complete'):
        if self.run: self.run.finish(exit_code=0 if status=='complete' else 1)


def setup(cfg):
    t=Tracker(cfg,'setup-'+digest(now())[:10],{'purpose':'authentication and Apple resource telemetry'})
    t.log(State(cfg).metrics()); url=t.run.url; t.close(); return {'url':url,'online':True}
