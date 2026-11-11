"""Reviewed logical-media index -> exact selected images on owned node TMPDIR.

PIL-only transport. No new jobs, downloads, model loads, data re-encoding, old
source edits, or broad pool readiness claims. Physical aliases are per attempt;
all logical data and transport inputs remain explicit, immutable descriptors.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import resource
import socket
import stat
import subprocess
import time
import uuid
import zipfile

REPORT_ROOTS=(Path('/data/user_data/luyangz/native_campaign_20261023'),
              Path('/scratch/user/u.lz341613/dsmm_native_campaign_20261023'))
READ_ROOTS=(Path('/data/hf_cache'),Path('/scratch/data'),
            Path('/scratch/user/u.lz341613/dsmm_native_campaign_20261023/assets'),
            Path('/data/user_data/luyangz/native_campaign_20261023/qualification-inputs-v1'))
REQUIRED_SOURCES={'coco','gqa','ocr_vqa','textvqa','vg'}


def require(value,message):
    if not value:raise ValueError(message)

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()
def sha(path,deadline=None):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):
            if deadline is not None and time.monotonic()>=deadline:raise InterruptedError('Media staging allocation reserve reached')
            h.update(block)
    return h.hexdigest()
def desc(path):
    path=Path(path).absolute()
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}
def checked(descriptor):
    require(desc(descriptor['path'])==descriptor,'Exact JSON descriptor changed: '+descriptor['path'])
    return json.loads(Path(descriptor['path']).read_text())
def seal(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=canonical(value)+b'\n'
    if path.exists():
        require(path.read_bytes()==payload,'Immutable receipt collision');return desc(path)
    with path.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    fd=os.open(str(path.parent),os.O_DIRECTORY|os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return desc(path)

def relative_name(value):
    require(isinstance(value,str) and value and '\\' not in value,'Unsafe logical image/member name')
    p=PurePosixPath(value)
    require(not p.is_absolute() and all(x not in ('','.','..') for x in value.split('/'))
            and str(p)==value,'Unsafe logical image/member path')
    return p

def literal(path,roots):
    path=Path(path)
    require(path.is_absolute() and '..' not in path.parts,'Literal absolute path required')
    matches=[Path(r) for r in roots if path==Path(r) or Path(r) in path.parents]
    require(matches,'Path outside reviewed namespace: '+str(path))
    root=max(matches,key=lambda r:len(r.parts));current=path
    while current!=root.parent:
        require(not current.is_symlink(),'Symlink ancestor refused: '+str(current));current=current.parent
    return path

def duration(value):
    require(value and value not in ('UNLIMITED','NOT_SET'),'Finite actual allocation required')
    days,clock=value.split('-',1) if '-' in value else ('0',value)
    hours,minutes,seconds=map(int,clock.split(':'))
    return int(days)*86400+hours*3600+minutes*60+seconds

def _actual():
    job=os.environ.get('SLURM_JOB_ID','')
    require(job.isdigit(),'Actual owned GPU allocation required')
    raw=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
    f=dict(x.split('=',1) for x in raw.split() if '=' in x)
    tres=dict(x.split('=',1) for x in f.get('AllocTRES','').split(',') if '=' in x)
    user=f.get('UserId','').split('(')[0];node=socket.gethostname().split('.')[0]
    require(f.get('JobId')==job and f.get('JobState')=='RUNNING'
            and user in ('luyangz','u.lz341613') and f.get('NodeList')==node
            and int(tres.get('gres/gpu','0'))>=1 and f.get('NumNodes')=='1',
            'Owned single-node GPU allocation not verified')
    return {'job_id':job,'node':node,'step_id':os.environ.get('SLURM_STEP_ID','batch'),
        'restart':os.environ.get('SLURM_RESTART_COUNT','0'),'user':user,
        'partition':f['Partition'],'account':f['Account'],'allocated_GPU_count':int(tres['gres/gpu']),
        'remaining_seconds':duration(f['TimeLimit'])-duration(f['RunTime'])}

def _job_root(actual):
    expected=Path('/tmp')/('job.'+actual['job_id'])
    configured=os.environ.get('TMPDIR')
    require(not configured or Path(configured) in (Path('/tmp'),expected),'Unexpected TMPDIR; no fallback to user scratch')
    literal(expected,[Path('/tmp')]);expected.mkdir(mode=0o700,exist_ok=True)
    require(expected.is_dir() and expected.stat().st_uid==os.getuid(),'Foreign job-local directory')
    return expected

def capacity(root,bytes_needed,inodes_needed,reserve_bytes,reserve_inodes):
    v=os.statvfs(root)
    measured={'free_bytes':v.f_bavail*v.f_frsize,'free_inodes':v.f_favail,
              'needed_bytes':bytes_needed,'needed_inodes':inodes_needed,
              'reserve_bytes':reserve_bytes,'reserve_inodes':reserve_inodes}
    require(measured['free_bytes']>=bytes_needed+reserve_bytes,'Insufficient measured node-local bytes')
    require(measured['free_inodes']>=inodes_needed+reserve_inodes,'Insufficient measured node-local inodes')
    return measured

def _inputs(pool_descriptor,records_descriptor,transport_descriptor,actual):
    pool=checked(pool_descriptor);records=checked(records_descriptor);cfg=checked(transport_descriptor)
    require(cfg['schema']=='native-media-transport-v1' and isinstance(records,list) and records,'Wrong transport/record schema')
    require(cfg['scope'] in ('ENGINEERING_ONLY','ALL_SOURCE'),'Explicit transport scope required')
    require(pool['image_manifest']==cfg['canonical_image_manifest'],'Logical pool image-manifest identity differs')
    assets=checked(cfg['canonical_image_manifest'])['assets'];index=checked(cfg['locator_index'])
    require(index['schema']=='native-media-locator-index-v1' and index['logical_pool']==pool_descriptor
        and index['canonical_image_manifest']==cfg['canonical_image_manifest']
        and index['scope']==cfg['scope'] and set(index['images'])==set(assets),'Incomplete/foreign logical locator index')
    coverage={}
    for key,item in assets.items():
        relative_name(key);group=key.split('/')[0];coverage[group]=coverage.get(group,0)+1
        require(item['bytes']>0 and len(item['sha256'])==64 and len(item['rgb_sha256'])==64
            and item['width']>0 and item['height']>0,'Canonical encoded/RGB identity incomplete')
    require(index['source_coverage']==coverage,'Declared image-source coverage differs')
    if cfg['scope']=='ALL_SOURCE':require(REQUIRED_SOURCES.issubset(coverage),'All-source index lacks native image domains')
    review=checked(cfg['index_review'])
    require(review['status']=='PASS_INDEPENDENT_COMPLETE_MEDIA_LOCATOR_INDEX'
        and review['locator_index']==cfg['locator_index'] and review['logical_pool']==pool_descriptor
        and review['canonical_image_manifest']==cfg['canonical_image_manifest'] and review['scope']==cfg['scope'],
        'Exact independent logical-index review required')
    ownership=checked(cfg['ownership'])
    require(ownership['status']=='ACTIVE_EXCLUSIVE_MEDIA_TRANSPORT_CLAIM'
        and ownership['job_id']==actual['job_id'] and ownership['logical_pool']==pool_descriptor
        and ownership['records_json']==records_descriptor and ownership['transport_index']==cfg['locator_index'],
        'Exact current consumer ownership required')
    roots=[Path(r) for r in cfg['allowed_read_roots']]
    require(roots and all(any(r==a or a in r.parents for a in READ_ROOTS) for r in roots),'Unapproved shared/archive read root')
    for key,locator in index['images'].items():
        require(locator['kind'] in ('zip','shared_file'),'Unsupported complete-index locator kind')
        if locator['kind']=='zip':
            relative_name(locator['member']);require(locator['archive'] in index['archives'],'Missing complete-index archive descriptor')
        else:
            path=Path(locator['path'])
            require(path.is_absolute() and '..' not in path.parts and any(path==r or r in path.parents for r in roots),'Shared locator outside approved roots')
    names=sorted({row['image'] for row in records if row.get('image')})
    for name in names:relative_name(name);require(name in assets,'Selected image absent from complete canonical pool')
    require(cfg['reserve_bytes']>=64<<20 and cfg['reserve_inodes']>=64,'Explicit node-local staging reserve required')
    return cfg,index,assets,names,roots

def _pixel_check(path,item):
    from PIL import Image
    require(Path(path).stat().st_size==item['bytes'] and sha(path)==item['sha256'],'Encoded bytes differ from canonical image')
    with Image.open(path) as opened:rgb=opened.convert('RGB')
    require(rgb.size==(item['width'],item['height']) and hashlib.sha256(rgb.tobytes()).hexdigest()==item['rgb_sha256'],
        'Decoded RGB differs from canonical image')

def _zip_members(archive,names):
    members={}
    for entry in archive.infolist():
        if entry.filename not in names:continue
        relative_name(entry.filename)
        require(entry.filename not in members and not entry.is_dir()
            and not stat.S_ISLNK(entry.external_attr>>16),'Duplicate/symlink selected ZIP member')
        members[entry.filename]=entry
    require(set(members)==set(names),'Selected ZIP member missing')
    return members

def prepare(pool_desc,records_desc,transport_desc,report_dir):
    started=time.monotonic();usage0=resource.getrusage(resource.RUSAGE_SELF);actual=_actual()
    cfg,index,canonical_assets,names,roots=_inputs(pool_desc,records_desc,transport_desc,actual)
    reserve=cfg.get('deadline_reserve_seconds',300)
    require(180<=reserve<=600,'Explicit accepted-boundary deadline reserve required')
    deadline=started+actual['remaining_seconds']-reserve
    if time.monotonic()>=deadline:raise InterruptedError('No staging time before allocation reserve')
    report=literal(report_dir,REPORT_ROOTS);report.mkdir(parents=True,exist_ok=True)
    receipt_path=report/'RUNTIME_IMAGES.json'
    if receipt_path.exists():
        result=desc(receipt_path);validate(result,pool_desc,records_desc);return result
    job_root=_job_root(actual)
    identity=hashlib.sha256(canonical([pool_desc,records_desc,cfg['locator_index'],actual['node'],actual['restart']])).hexdigest()[:24]
    stage=literal(job_root/('dsmm-native-media-'+identity),[job_root]);stage.mkdir(mode=0o700,exist_ok=True)
    with literal(stage/'OWNED_STAGE.lock',[stage]).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        before=capacity(stage,sum(canonical_assets[n]['bytes'] for n in names),len(names)*3+16,
            cfg['reserve_bytes'],cfg['reserve_inodes'])
        seal(report/'MEDIA_STAGE_START.json',{'actual':actual,'logical_pool':pool_desc,'records_json':records_desc,
            'transport':transport_desc,'image_root':str(stage),'measured_capacity':before})
        used_archives={index['images'][n]['archive'] for n in names if index['images'][n]['kind']=='zip'}
        archive_files={};archive_receipts={};archive_verified_bytes=0
        try:
            for key in sorted(used_archives):
                item=index['archives'][key];path=literal(item['path'],roots)
                provider=checked(item['provider'])
                require(provider['status'] in ('PASS_MEDIA_ARCHIVE','STAGED_EXACT_SELECTED_MEDIA_ARCHIVE')
                    and {k:provider['archive'][k] for k in ('path','bytes','sha256')}=={k:item[k] for k in ('path','bytes','sha256')},'Archive producer ownership/descriptor differs')
                require({'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path,deadline)}=={k:item[k] for k in ('path','bytes','sha256')},'Durable archive bytes changed')
                archive_verified_bytes+=item['bytes'];archive=zipfile.ZipFile(path)
                wanted={index['images'][n]['member'] for n in names if index['images'][n].get('archive')==key}
                members=_zip_members(archive,wanted);archive_files[key]=(archive,members);archive_receipts[key]={'path':str(path),'bytes':item['bytes'],'sha256':item['sha256']}
            runtime_assets={};staged_bytes=0;reused=0
            for name in names:
                if time.monotonic()>=deadline:raise InterruptedError('Media staging stopped at accepted image boundary')
                item=canonical_assets[name];locator=index['images'][name];destination=literal(stage/name,[stage])
                destination.parent.mkdir(parents=True,exist_ok=True)
                if destination.exists():_pixel_check(destination,item);reused+=1
                else:
                    temporary=destination.with_name(destination.name+'.partial-'+uuid.uuid4().hex)
                    if locator['kind']=='shared_file':
                        source=literal(locator['path'],roots);reader=source.open('rb')
                    else:
                        require(locator['kind']=='zip','Unsupported media locator kind')
                        archive,members=archive_files[locator['archive']];member=members[locator['member']]
                        require(member.file_size==item['bytes'],'ZIP encoded member size differs');reader=archive.open(member,'r')
                    with reader,temporary.open('xb') as output:
                        total=0
                        while True:
                            if time.monotonic()>=deadline:raise InterruptedError('Media staging stopped inside new partial file; canonical sources preserved')
                            block=reader.read(1<<20)
                            if not block:break
                            total+=len(block);require(total<=item['bytes'],'Oversized encoded image stream');output.write(block)
                        output.flush();os.fsync(output.fileno())
                    _pixel_check(temporary,item)
                    # New same-root temporary only: atomic no-overwrite install.
                    os.link(temporary,destination);temporary.unlink();staged_bytes+=item['bytes']
                runtime_assets[name]=dict(item,path=str(destination))
            after=capacity(stage,0,0,cfg['reserve_bytes'],cfg['reserve_inodes']);usage=resource.getrusage(resource.RUSAGE_SELF)
            receipt={'schema':'native-runtime-image-alias-v1','status':'PASS_BYTE_IDENTICAL_RUNTIME_IMAGES',
                'scope':cfg['scope'],'logical_pool':pool_desc,'records_json':records_desc,
                'canonical_image_manifest':cfg['canonical_image_manifest'],'transport':transport_desc,
                'locator_index':cfg['locator_index'],'index_review':cfg['index_review'],'ownership':cfg['ownership'],
                'image_root':str(stage),'assets':runtime_assets,'job_id':actual['job_id'],'node':actual['node'],'actual':actual,
                'materializer_source_sha256':sha(__file__),'verified_archives':archive_receipts,
                'cost':{'elapsed_seconds':time.monotonic()-started,'user_CPU_seconds':usage.ru_utime-usage0.ru_utime,
                    'system_CPU_seconds':usage.ru_stime-usage0.ru_stime,'staged_encoded_bytes':staged_bytes,
                    'archive_verified_bytes':archive_verified_bytes,'images':len(names),'reused_images':reused,
                    'capacity_before':before,'capacity_after':after},'scientific_pool_readiness_claim':False}
            return seal(receipt_path,receipt)
        except InterruptedError as error:
            seal(report/'MEDIA_STAGE_CONTINUATION_REQUIRED.json',{'actual':actual,'logical_pool':pool_desc,'records_json':records_desc,'transport':transport_desc,'reason':str(error),'same_node_cache_not_guaranteed':True,'no_scheduler_mutations':True})
            raise
        finally:
            for archive,members in archive_files.values():archive.close()

def validate(alias_desc,pool_desc,records_desc):
    value=checked(alias_desc);actual=_actual()
    require(value['status']=='PASS_BYTE_IDENTICAL_RUNTIME_IMAGES' and value['schema']=='native-runtime-image-alias-v1'
        and value['logical_pool']==pool_desc and value['records_json']==records_desc
        and value['job_id']==actual['job_id'] and value['node']==actual['node']
        and value['actual']['restart']==actual['restart'] and value['materializer_source_sha256']==sha(__file__),
        'Runtime alias is foreign/stale or has different logical/source identity')
    cfg,index,assets,names,roots=_inputs(pool_desc,records_desc,value['transport'],actual)
    stage=literal(value['image_root'],[_job_root(actual)])
    require(set(value['assets'])==set(names),'Runtime image coverage differs from selected unchanged records')
    for name in names:
        physical=literal(stage/name,[stage]);expected=dict(assets[name],path=str(physical))
        require(value['assets'][name]==expected,'Runtime alias changes canonical image identity')
        _pixel_check(physical,assets[name])
    return value
