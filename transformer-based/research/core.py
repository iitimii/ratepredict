from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime, timezone

import psutil
import yaml

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024 ** 3


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(value)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False)+'\n').encode())


def config(path=None):
    path = Path(path or ROOT/'transformer-based/configs/research.yaml')
    cfg = yaml.safe_load(path.read_text())
    cfg['runtime'] = str((ROOT/cfg['runtime_dir']).resolve())
    return cfg


class ResourceBlocked(RuntimeError):
    pass


class State:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = Path(cfg['runtime'])
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root/'state.sqlite3'
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY, metadata TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
              kind TEXT NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
              attempts INTEGER NOT NULL DEFAULT 0, next_at REAL NOT NULL DEFAULT 0,
              pid INTEGER, updated TEXT, error TEXT, result TEXT);
            CREATE TABLE IF NOT EXISTS raw (hash TEXT PRIMARY KEY, path TEXT NOT NULL,
              bytes INTEGER NOT NULL, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS fetches (id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
              url TEXT NOT NULL, hash TEXT NOT NULL, retrieved_at TEXT NOT NULL, headers TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
              raw_hash TEXT NOT NULL, available_at TEXT, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS observations (id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
              series TEXT NOT NULL, event_time TEXT NOT NULL, available_at TEXT,
              retrieved_at TEXT NOT NULL, value REAL, unit TEXT, quality TEXT NOT NULL, raw_hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, snapshot TEXT NOT NULL,
              model TEXT NOT NULL, fold TEXT NOT NULL, seed INTEGER NOT NULL,
              status TEXT NOT NULL, updated TEXT NOT NULL, result TEXT);
            CREATE TABLE IF NOT EXISTS reservations (owner TEXT PRIMARY KEY, bytes INTEGER NOT NULL, pid INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS host_access (host TEXT PRIMARY KEY, next_at REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS job_queue ON jobs(status, next_at);
            ''')

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA busy_timeout=30000')
        return db

    def enqueue(self, source, kind, payload, generation=''):
        key = digest([source, kind, payload, generation])
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO jobs(id,source_id,kind,payload,updated) VALUES(?,?,?,?,?)',
                       (key, source, kind, json.dumps(payload), now()))
        return key

    def claim(self, kinds=None):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for row in db.execute("SELECT id,pid FROM jobs WHERE status='running'").fetchall():
                if not row['pid'] or not psutil.pid_exists(row['pid']):
                    db.execute("UPDATE jobs SET status='queued',pid=NULL WHERE id=?",(row['id'],))
            sql = "SELECT * FROM jobs WHERE status IN ('queued','retry') AND next_at<=?"
            args = [time.time()]
            if kinds:
                sql += ' AND kind IN ('+','.join('?' for _ in kinds)+')'
                args += list(kinds)
            row = db.execute(sql+' ORDER BY updated,id LIMIT 1',args).fetchone()
            if row:
                db.execute("UPDATE jobs SET status='running',pid=?,attempts=attempts+1,updated=? WHERE id=?",
                           (os.getpid(), now(), row['id']))
                return dict(row)
        return None

    def finish(self, key, status, result=None, error=None, delay=0):
        with self.connect() as db:
            db.execute('UPDATE jobs SET status=?,result=?,error=?,next_at=?,updated=?,pid=NULL WHERE id=?',
                       (status,json.dumps(result),error,time.time()+delay,now(),key))

    def summary(self):
        with self.connect() as db:
            counts = {r['status']:r['n'] for r in db.execute('SELECT status,count(*) n FROM jobs GROUP BY status')}
            return {'jobs':counts,'sources':db.execute('SELECT count(*) FROM sources').fetchone()[0],
                    'documents':db.execute('SELECT count(*) FROM documents').fetchone()[0],
                    'observations':db.execute('SELECT count(*) FROM observations').fetchone()[0],
                    'raw_bytes':db.execute('SELECT coalesce(sum(bytes),0) FROM raw').fetchone()[0],
                    'runs':[dict(r) for r in db.execute('SELECT * FROM runs ORDER BY updated')]}

    def metrics(self):
        mem = psutil.virtual_memory()
        return {'system/rss_gb':psutil.Process().memory_info().rss/GIB,
                'system/available_gb':mem.available/GIB,'system/swap_gb':psutil.swap_memory().used/GIB,
                'system/disk_free_gb':shutil.disk_usage(self.root).free/GIB}

    def check_memory(self, background=False):
        m = self.metrics()
        limit = self.cfg['resources']['background_available_gb' if background else 'min_available_gb']
        if m['system/available_gb'] < limit or m['system/rss_gb'] > self.cfg['resources']['max_process_gb']:
            raise ResourceBlocked('Memory pressure: '+json.dumps(m))
        return m

    @contextlib.contextmanager
    def reserve(self, amount, purpose):
        owner = digest([os.getpid(),purpose,time.time_ns()])
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for r in db.execute('SELECT owner,pid FROM reservations').fetchall():
                if not psutil.pid_exists(r['pid']):
                    db.execute('DELETE FROM reservations WHERE owner=?',(r['owner'],))
            pending = db.execute('SELECT coalesce(sum(bytes),0) FROM reservations').fetchone()[0]
            free = shutil.disk_usage(self.root).free
            required = int(self.cfg['resources']['reserve_disk_gb']*GIB)+amount+pending
            if free < required:
                raise ResourceBlocked(f'{purpose}: need {required/GIB:.2f} GB free including reservations; have {free/GIB:.2f} GB')
            db.execute('INSERT INTO reservations VALUES(?,?,?)',(owner,amount,os.getpid()))
        try:
            yield
        finally:
            with self.connect() as db:
                db.execute('DELETE FROM reservations WHERE owner=?',(owner,))

    @contextlib.contextmanager
    def gpu(self):
        with (self.root/'gpu.lock').open('a') as f:
            try:
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                raise ResourceBlocked('Another pipeline process holds the GPU lease')
            try:
                self.check_memory()
                yield
            finally:
                fcntl.flock(f,fcntl.LOCK_UN)

    def raw(self, content, source, url, headers=None):
        h = digest(content)
        path = self.root/'raw'/h[:2]/h
        if not path.exists():
            total = self.summary()['raw_bytes']
            if total+len(content) > self.cfg['resources']['raw_budget_gb']*GIB:
                raise ResourceBlocked('Raw archive quota reached')
            with self.reserve(len(content), 'raw acquisition'):
                atomic_bytes(path,content)
        stamp = now()
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO raw VALUES(?,?,?,?)',(h,str(path),len(content),stamp))
            db.execute('INSERT OR IGNORE INTO fetches VALUES(?,?,?,?,?,?)',
                       (digest([h,url,stamp]),source,url,h,stamp,json.dumps(headers or {})))
        return h,path
