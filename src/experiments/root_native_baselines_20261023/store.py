"""Durable real measurements, atomic author batches and missing-only restarts."""
from pathlib import Path
import sqlite3
import numpy as np
from experiments.root_native_campaign_20261023 import common as c


class Store:
    def __init__(self,path,identity,record_ids,method):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.identity=identity;self.roster=list(record_ids);self.method=method
        c.require(len(self.roster)==len(set(self.roster)),'Repeated owned feature record')
        self.allowed=set(self.roster)
        self.db=sqlite3.connect(self.path)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS features(record_id TEXT PRIMARY KEY,vector BLOB NOT NULL,info REAL,tau REAL,length INTEGER,nan_count INTEGER NOT NULL,identity TEXT NOT NULL)')
        expected=dict(identity=identity,method=method,roster_sha256=c.key(self.roster))
        existing=dict(self.db.execute('SELECT key,value FROM metadata'))
        if existing:c.require(existing==expected,'Feature DB ownership/measurement/roster changed')
        else:
            self.db.executemany('INSERT INTO metadata VALUES(?,?)',expected.items());self.db.commit()
        self.existing={row[0] for row in self.db.execute('SELECT record_id FROM features')}
        c.require(self.existing<=self.allowed,'Foreign feature record in owned shard')
        c.require(all(row[0]==identity for row in self.db.execute('SELECT DISTINCT identity FROM features')),'Mixed measurement functions')

    def missing_batch(self,ids):
        c.require(set(ids)<=self.allowed and len(ids)==len(set(ids)),'Batch outside owned roster')
        present=[ident in self.existing for ident in ids]
        c.require(all(present) or not any(present),'Partially committed native batch; preserve corruption and diagnose')
        return not all(present)

    def commit(self,measurements):
        """All feature rows of one original author batch commit together."""
        ids=[row['record_id'] for row in measurements]
        c.require(measurements and self.missing_batch(ids),'Missing-only transaction required')
        packed=[]
        for row in measurements:
            dtype,dimension=('<f2',20480) if self.method=='coincide' else ('<f4',4096)
            vector=np.asarray(row['vector'],dtype=dtype)
            c.require(vector.shape==(dimension,) and np.isfinite(vector).all(),'Invalid actual feature vector')
            info,tau,length=row.get('info'),row.get('tau'),row.get('length')
            if self.method=='datatailor':
                c.require(info is not None and np.isfinite(info) and tau is not None and np.isfinite(tau) and tau>0 and length>0,'Invalid native SVD values')
            packed.append((row['record_id'],vector.tobytes(),info,tau,length,row.get('nan_count',0),self.identity))
        with self.db:self.db.executemany('INSERT INTO features VALUES(?,?,?,?,?,?,?)',packed)
        self.existing.update(ids)

    def close(self):
        self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)');self.db.close()


def load_complete(databases,method,identity,ids):
    """Read every measurement; hashes and exact ID/semantic coverage are mandatory."""
    values={};allowed=set(ids);dtype,dim=('<f2',20480) if method=='coincide' else ('<f4',4096)
    for descriptor in databases:
        c.require(c.desc(descriptor['path'])==descriptor,'Feature database changed')
        db=sqlite3.connect('file:'+str(Path(descriptor['path']).absolute())+'?mode=ro',uri=True)
        try:
            meta=dict(db.execute('SELECT key,value FROM metadata'))
            c.require(meta['identity']==identity and meta['method']==method,'Feature measurement identity differs')
            for ident,blob,info,tau,length,nans,actual in db.execute('SELECT * FROM features'):
                c.require(ident in allowed and ident not in values and actual==identity,'Duplicate/foreign feature')
                vector=np.frombuffer(blob,dtype=dtype)
                c.require(vector.shape==(dim,) and np.isfinite(vector).all(),'Truncated/nonfinite feature blob')
                if method=='datatailor':c.require(np.isfinite(info) and np.isfinite(tau) and tau>0 and length>0,'Corrupt SVD scalars')
                values[ident]=dict(vector=vector.copy(),info=info,tau=tau,length=length,nan_count=nans)
        finally:db.close()
    c.require(set(values)==allowed,'Full eligible-pool feature coverage not attained')
    return values
