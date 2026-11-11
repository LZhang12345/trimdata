"""Split the exact paid DT16 union, then run twelve full native source blocks.

Producer output is accepted only with a verified, completed physical binding and
an actual manifest; no future content hash or small-prefix qualification is used.
"""
import argparse, fcntl, hashlib, json, os, re, resource, socket, subprocess, time
from pathlib import Path
from contextlib import contextmanager
ROOT=Path(__file__).resolve().parents[2]
def require(ok,msg):
    if not ok:raise ValueError(msg)
def ref(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(r):
    require(set(r)=={'path','bytes','sha256'} and Path(r['path']).is_absolute() and not any(p.is_symlink()for p in(Path(r['path']),*Path(r['path']).parents))and ref(r['path'])==r,'Exact immutable input differs')
    return json.loads(Path(r['path']).read_bytes())
def seal(p,v):
    p=Path(p);b=(json.dumps(v,sort_keys=True,indent=2)+'\n').encode()
    with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
    return ref(p)
def np_save(p,a):
    import numpy as np
    with Path(p).open('xb')as f:np.save(f,a,allow_pickle=False);f.flush();os.fsync(f.fileno())
    return ref(p)
def validate(cfg):
    require(cfg['status']=='ROOT_DT_FULL_GEOMETRY_PREREQUISITES_V1' and cfg['owner']=='Root','Scope differs')
    require(cfg['budgets']==[0.05,0.10,0.15,0.20] and cfg['source_block_count']==12,'Full native scope differs')
    require(cfg['feature_acquisition_authorized']is False and cfg['student_training_authorized']is False,'Scope cannot train/acquire')
    require(cfg['full_ward_authorized']is True and cfg['epsilon_or_singleton_repair_authorized']is False,'Native geometry policy differs')
    out=Path(cfg['output']);require(str(out).startswith('/data/hf_cache/project_features/DS-MultiModal/dt_full_geometry_20261026/')and '..'not in out.parts,'Output route differs')
    for n,sha in cfg['sources'].items():
        p=ROOT/n;require(not Path(n).is_absolute()and '..'not in Path(n).parts and not p.is_symlink()and hashlib.sha256(p.read_bytes()).hexdigest()==sha,'Source differs: '+n)
    return cfg
@contextmanager
def exclusive(path):
    with Path(path).open('a+b')as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);yield

def authenticated(binding_path,mode):
    b=json.loads(Path(binding_path).read_bytes());cfg=validate(checked(b['configuration']));review=checked(b['independent_review'])
    require(review['status']=='PASS_INDEPENDENT_DT_FULL_GEOMETRY_SOURCE_CONFIG_V1'and review['independent']is True and review['reviewer']not in('Root','/root','')and review['configuration']==b['configuration']and review['sources']==cfg['sources'],'Independent exact review required')
    require(b['owner']=='Root'and b['mode']==mode,'Binding scope differs')
    j=os.environ.get('SLURM_JOB_ID','');array=os.environ.get('SLURM_ARRAY_JOB_ID');task=os.environ.get('SLURM_ARRAY_TASK_ID')
    require(j.isdecimal()and os.getuid()==2708899 and os.environ.get('SLURM_RESTART_COUNT','0')=='0','Fresh actual actor required')
    require((mode=='prepare'and j==b['job_id']and array is None)or(mode=='geometry'and array==b['job_id']and task is not None and int(task)in range(12)),'Actual task differs')
    selector=(array+'_'+task)if mode=='geometry'else j
    raw=subprocess.check_output(['scontrol','show','job','-o',selector],text=True);require(len([r for r in raw.splitlines()if r.strip()])==1,'One exact physical task required');f=dict(re.findall(r'(?:^|\s)([^\s=]+)=(\S+)',raw));memory='64G'if mode=='prepare'else'192G';limit='01:00:00'if mode=='prepare'else'1-00:00:00'
    require(f.get('JobId')==j and (mode=='prepare'or(f.get('ArrayJobId')==array and f.get('ArrayTaskId')==task)),'Actual physical job/task identity differs')
    require(all(f.get(k)==v for k,v in dict(JobState='RUNNING',UserId='luyangz(2708899)',Account='cx',QOS='normal',Partition='general',NumCPUs='8',NumTasks='1',TresPerNode='gres/gpu:L40S:1',Comment='dsmm-dt-full:'+cfg['nonce'],TimeLimit=limit,WorkDir=str(ROOT),Command=str(ROOT/'experiments/root_dt_full_geometry_v2_20261026/run.sbatch')).items()),'Actual resource/route differs')
    tres=dict(x.split('=',1)for x in f['AllocTRES'].split(',')if'='in x);require(tres.get('gres/gpu')=='1'and tres.get('mem')in(memory,str(int(memory[:-1])*1024)+'M'),'Allocation differs')
    spool=subprocess.check_output(['scontrol','write','batch_script',j,'-']);require(spool==(ROOT/'experiments/root_dt_full_geometry_v2_20261026/run.sbatch').read_bytes(),'Bound spool differs')
    claim=checked(b['ownership']);require(claim['owner']=='Root'and claim['job_id']==b['job_id']and claim['configuration']==b['configuration']and claim['no_duplicate_provider']is True,'Ownership differs')
    grant=checked(b['storage_reservation']);require(grant['owner']=='Root'and grant['configuration']==b['configuration']and grant['output']==cfg['output']and grant['collective_reserved_bytes']==32<<30 and grant['other_outstanding_reserved_bytes']>=0,'Collective measured storage grant required')
    measured=checked(grant['actual_storage_measurement']);require(measured['collective_measurement']['available_bytes']>=grant['collective_reserved_bytes']+grant['other_outstanding_reserved_bytes'],'Measured collective storage insufficient')
    return b,cfg,dict(job_id=j,array_job=array,task=task,node=socket.gethostname(),scheduler=raw)

def measurement_projection(original):
    # Exact previously executed qualify_union projection, not a relaxed header.
    require(isinstance(original['measurement_sources'],dict)and original['measurement_sources'],'Pinned measurement sources required')
    return dict(original,sources=original['measurement_sources'])

def prepare(cfg,out):
    import gc,numpy as np
    from experiments.root_native_campaign_v7_20261024 import selection as native
    from experiments.root_dt_oracle_overlay_recovery_v1_20261025 import consumer
    from experiments.root_dt_union_qualification_v1_20261025.worker import validate_original_complete,union_digest,eligible_blocks,runtime
    old=cfg['dt_config'];original=checked(old['original_configuration']);observed=runtime(original);measurement=measurement_projection(original);pool,records,ids=native.validate_pool(old['pool'],eligibility_desc=old['eligibility']);del records;gc.collect()
    require(len(ids)==660763 and ids==pool['record_ids'],'Exact full pool required');identity=consumer.native.identity(measurement,pool)
    for entry in old['feature_shards']:
        if entry['kind']=='original':
            roster=consumer.native.roster(dict(role='features',shard_count=16,shard_index=entry['shard_index']),[],ids)
            validate_original_complete(checked(entry['completion']),entry,old['pool'],identity,roster,measurement)
    values=consumer.load_full_pool(old['feature_shards'],ids,old['pool'],identity,old['actual_parity'],old['actual_parity_review'])
    digest=union_digest(values,ids);prior=checked(cfg['paid_union']);require(digest['ordered_paid_value_sha256']==prior['ordered_paid_value_sha256'],'Paid union changed')
    blocks=eligible_blocks(pool['source_indices']);result=[]
    for i,positions in enumerate(blocks):
        d=out/('block-%02d'%i);d.mkdir();hid=np.stack([values[ids[p]]['vector']for p in positions]);require(hid.dtype==np.dtype('<f4')and hid.shape==(len(positions),4096),'Paid float32 bits differ')
        item=dict(block=i,count=len(positions),hidden=np_save(d/'hidden.npy',hid),info=np_save(d/'info.npy',np.asarray([values[ids[p]]['info']for p in positions],dtype=np.float32)),positions=np_save(d/'positions.npy',np.asarray(positions,dtype=np.int64)),source_indices=[pool['source_indices'][p]for p in positions],record_ids_sha256=hashlib.sha256(json.dumps([ids[p]for p in positions],sort_keys=True,separators=(',',':')).encode()).hexdigest())
        result.append(item);del hid
    return dict(status='ACTUAL_FULL_DT12_BLOCK_INPUTS_V1',runtime=observed,measurement_projection='EXACT_QUALIFIED_PLAN_V4_measurement_sources',pool=old['pool'],paid_union=cfg['paid_union'],ordered_paid_value_sha256=digest['ordered_paid_value_sha256'],blocks=result,new_features=0)

def actual_manifest(binding,cfg):
    pb=checked(binding['producer_binding']);require(pb['mode']=='prepare'and pb['configuration']==binding['configuration'],'Actual producer differs')
    s=subprocess.check_output(['sacct','-n','-P','-j',pb['job_id'],'--format=JobID,State,ExitCode'],text=True)
    rows=[r.split('|')for r in s.splitlines()if r];require(any(r[:3]==[pb['job_id'],'COMPLETED','0:0']for r in rows),'Producer has not completed successfully')
    p=Path(cfg['output'])/'prepare'/'MANIFEST.json';result=checked(ref(p));require(result['status']=='ACTUAL_FULL_DT12_BLOCK_INPUTS_V1'and result['pool']==cfg['dt_config']['pool']and len(result['blocks'])==12 and result['actual']['job_id']==pb['job_id']and result['configuration']==pb['configuration'],'Actual manifest scope differs')
    start=checked(ref(Path(cfg['output'])/'prepare'/'START.json'));require(start['binding']==binding['producer_binding']and start['configuration']==pb['configuration']and start['actual']==result['actual'],'Actual producer result/binding join differs')
    return result,ref(p)

def geometry(b,cfg,out,task):
    import numpy as np,torch
    from experiments.root_native_baselines_20261023 import operators
    manifest,mref=actual_manifest(b,cfg);block=manifest['blocks'][task];require(block['block']==task,'Block order differs')
    for k in('hidden','info','positions'):require(ref(block[k]['path'])==block[k],'Actual block bytes differ')
    hidden=np.load(block['hidden']['path'],allow_pickle=False);info=np.load(block['info']['path'],allow_pickle=False);require(hidden.shape==(block['count'],4096)and hidden.dtype==np.dtype('<f4'),'Full block dtype/shape differs');torch.set_num_threads(8)
    labels=operators.datatailor_ward_block(torch.from_numpy(hidden));lref=np_save(out/'labels.npy',labels.numpy())
    seal(out/'WARD_COMPLETE.json',dict(status='ACTUAL_FULL_SOURCE_WARD_COMPLETE',block=task,count=len(hidden),labels=lref,producer_manifest=mref,clusters=int(labels.max())))
    try:
        unique,rep=operators.datatailor_relationships(torch.from_numpy(hidden),torch.from_numpy(info),labels,block['source_indices'])
        return dict(status='ACTUAL_FULL_SOURCE_RELATIONSHIPS_COMPLETE',block=task,labels=lref,uniqueness=np_save(out/'uniqueness.npy',unique.numpy()),representativeness=np_save(out/'representativeness.npy',rep.numpy()),producer_manifest=mref)
    except ValueError as e:
        seal(out/'RELATIONSHIP_REJECTED.json',dict(status='NATIVE_FULL_RELATIONSHIP_REJECTED',error_repr=repr(e),block=task,labels=lref,no_epsilon_or_fake_zero=True));raise

def run(binding_path,mode):
    b,cfg,actor=authenticated(binding_path,mode);base=Path(cfg['output']);base.mkdir(parents=True,exist_ok=True);task=int(actor['task'])if mode=='geometry'else None;out=base/('prepare'if mode=='prepare'else'geometry-%02d'%task)
    out.mkdir(exist_ok=False);start=time.monotonic()
    with exclusive(base/('prepare.lock'if mode=='prepare'else'geometry-%02d.lock'%task)):
        stat=os.statvfs(base);require(stat.f_bavail*stat.f_frsize>=(32<<30)+checked(b['storage_reservation'])['other_outstanding_reserved_bytes'],'32GiB additional output space required')
        seal(out/'START.json',dict(actual=actor,binding=ref(binding_path),configuration=b['configuration']))
        try:
            result=prepare(cfg,out)if mode=='prepare'else geometry(b,cfg,out,task)
            result.update(actual=actor,configuration=b['configuration'],elapsed_seconds=time.monotonic()-start,max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,new_neural_forwards=0,student_READY=False)
            return seal(out/('MANIFEST.json'if mode=='prepare'else'RESULT.json'),result)
        except BaseException as e:
            seal(out/'FAILURE.json',dict(actual=actor,error_repr=repr(e),elapsed_seconds=time.monotonic()-start,automatic_retry=False));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--mode',choices=['prepare','geometry'],required=True);a=p.parse_args();print(json.dumps(run(a.binding,a.mode)))
