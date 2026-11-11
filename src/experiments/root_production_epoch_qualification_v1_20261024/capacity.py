"""Conservative actual user-quota and filesystem bound, no global-free shortcut."""
import re
from decimal import Decimal
from . import protocol as p
def quota(raw,fs,minimum):
    rows=[x.split() for x in raw.splitlines() if x.strip().startswith('/scratch/user/u.lz341613 ')]
    p.require(len(rows)==1 and len(rows[0])==5,'Exact user quota row required')
    def quantity(x):
        m=re.fullmatch(r'(\d+(?:\.\d+)?)([KMGT]?)',x);p.require(m is not None,'Unknown quota unit');return Decimal(m[1])*Decimal(1024)**{'':0,'K':1,'M':2,'G':3,'T':4}[m[2]]
    _,used,limit,files,filelimit=rows[0];unit=re.search(r'[KMGT]$',used);uncertainty=quantity('1'+unit[0]) if unit else 1
    free=min(fs.f_bavail*fs.f_frsize,max(0,int(quantity(limit)-quantity(used)-uncertainty)));inodes=min(fs.f_favail,int(filelimit)-int(files));p.require(free>=minimum and inodes>=4000,'Actual user quota/stat cannot reserve actual collective bytes/4000inodes');return dict(available_bytes_conservative=free,available_inodes=inodes,actual_user_quota=raw)
