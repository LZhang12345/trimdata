"""Finite Root General scalar continuation; no admission or old-store mutation."""
import argparse
import fcntl
import importlib
import os
from pathlib import Path
import pwd
import re
import signal
import socket
import subprocess
import sys
import time
from experiments.root_s7_dt_composite_general_v1_20261025 import worker as base
from . import consumer

c=base.c
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
SOURCE_ROOT=base.SOURCE_ROOT
CONTROL=Path('/home/luyangz/data_selection_multimodal/collaboration/root-dt-oracle-overlay-recovery-v1-20261025')
OUTPUT_ROOT=base.CAMPAIGN/'datatailor-oracle-overlay-v1'
AUTHOR='/root/storage_scope_review'
SCOPE='ROOT_MISSING_ONLY_S7_DT_PER_ROW_NATIVE91_PREFIX_ORACLE_V1'
CANDIDATES=(0,2,11,13,14,15)
REVIEW='PASS_INDEPENDENT_DT_ORACLE_OVERLAY_RECOVERY_SOURCE_CONFIG_V1'

def verify_config(cfg,ref,review):
    shard=cfg['global_shard_indices'][0]
    c.require(cfg['schema']=='root-dt-oracle-overlay-recovery-v1' and cfg['owner']=='Root'
        and cfg['scope']==SCOPE and cfg['source_root']==str(SOURCE_ROOT)
        and cfg['global_shard_indices']==[shard] and shard in CANDIDATES
        and cfg['shard_count']==16 and cfg['method']=='datatailor','Finite reviewed terminal-shard scope; live3 excluded')
    c.require(cfg['site']==dict(name='babel',partition='general',qos='normal',account='cx',user='luyangz',gpu_type='L40S'),'New Root route must be General; original profile remains historical')
    c.require(cfg['allocation']==dict(cpus=8,gpus=1,memory_mb=65536,minutes=120,reserve_seconds=300)
        and cfg['maximum_restarts']==3 and cfg['required_storage_bytes']==2<<30,'Exact finite General route/storage envelope')
    c.require(review['status']==REVIEW and review['configuration']==ref and review['sources']==cfg['sources']
        and isinstance(review.get('reviewer'),str) and review['reviewer'].strip()
        and review['reviewer'].strip()==review['reviewer'] and review['reviewer']!=AUTHOR,'Independent exact source/config review')
    for relative,digest in cfg['sources'].items():
        path=Path(relative);c.require(not path.is_absolute() and '..' not in path.parts and c.sha(ROOT/path)==digest,'Frozen source changed: '+relative)
    c.require(cfg['actual_parity']['sha256']==base.PARITY_SHA and cfg['actual_parity_review']['sha256']==base.PARITY_REVIEW_SHA,
        'Original independently reviewed NN33 result required; no fabricated expanded NN fixture')
    c.require(cfg['output']==str(OUTPUT_ROOT/('shard-%04d'%shard)) and cfg['retained_databases'],
        'One distinct new overlay, actual retained custody required')
    for entry in cfg['retained_databases']:
        c.require(entry['kind'] in ('original','composite_v3','oracle_overlay_v1')
            and set(entry['database'])=={'path','bytes','sha256'} and type(entry['database']['bytes']) is int
            and entry['database']['bytes']>0 and re.fullmatch('[0-9a-f]{64}',entry['identity']),
            'Explicit typed immutable retained ledger')
        base.within(entry['database']['path'],base.CAMPAIGN)
    home=Path('/home/luyangz/data_selection_multimodal')
    old_path=home/('collaboration/copilot7-fullpool-published-20261024/datatailor-preempt-v1/CONFIG.json' if shard==0 else 'collaboration/root-s7-datatailor-general-takeover-v1-20261024/CONFIG.json')
    c.require(cfg['original_spec']['path']==str(old_path) and cfg['actual_mapping']['path']==str(CONTROL/('g%d'%shard)/'MAPPING.json'),'Exact HOME original/mapping routes')
    old=c.checked(cfg['original_spec']);base.native_config(cfg,old)
    from .sources import sources
    c.require(cfg['sources']==sources(old),'Complete additive runtime source closure required')
    for field in ('python','feature_profile','published_sources','runtime_component','runtime_versions'):
        c.require(cfg[field]==old[field],'Original runtime/provider identity changed: '+field)
    c.require(shard in old['global_shard_indices'] and old['method']=='datatailor','Original global-index mapping differs')
    mapping=c.checked(cfg['actual_mapping'])
    c.require(mapping['status']=='EXACT_DT_SHARD_MAPPING_FROM_ACTUAL_SACCT'
        and mapping['global_shard_index']==shard and mapping['original_configuration']==cfg['original_spec']
        and mapping['producer_aliases']==cfg['predecessor_aliases'],'Actual logical/physical mapping required; no numeric inference')
    return shard

def actual(binding,cfg,runner=subprocess.check_output):
    c.require(ROOT==SOURCE_ROOT and os.getuid()==2708899 and pwd.getpwuid(os.getuid()).pw_name=='luyangz',
        'Actual Root actor/source before DATA')
    job=os.environ.get('SLURM_JOB_ID','');restart=os.environ.get('SLURM_RESTART_COUNT','0')
    c.require(job.isdecimal() and job==binding['job_id'] and binding['owner']=='Root'
        and not os.environ.get('SLURM_ARRAY_JOB_ID') and not os.environ.get('SLURM_ARRAY_TASK_ID')
        and restart.isdecimal() and int(restart)<=cfg['maximum_restarts']
        and re.fullmatch('[0-9a-f]{32}',binding['nonce']),'Exact scalar binding/restart')
    raw=runner(['scontrol','show','job','-o',job],text=True,timeout=30)
    c.require(len(raw.strip().splitlines())==1 and raw.startswith('JobId='),'One actual scalar record')
    fields=dict(re.findall(r'(?:^|\s)([^\s=]+)=(\S+)',raw));shard=cfg['global_shard_indices'][0]
    wanted=dict(JobId=job,JobState='RUNNING',UserId='luyangz(2708899)',Account='cx',Partition='general',QOS='normal',
        JobName='dsmm-root-dt-oracle-g%d-v1'%shard,Comment='root-dt-oracle-v1:'+binding['nonce'],
        Command=str(HERE/'run.sbatch'),WorkDir=str(ROOT),NumCPUs='8',NumNodes='1',NumTasks='1',
        Restarts=restart,Requeue='1',TimeLimit='02:00:00',NodeList=socket.gethostname().split('.')[0],TresPerNode='gres/gpu:L40S:1')
    c.require(all(fields.get(k)==v for k,v in wanted.items()),'Current scalar ownership/route differs')
    for name in ('ReqTRES','AllocTRES'):
        tres=dict(v.split('=',1) for v in fields.get(name,'').split(',') if '=' in v)
        c.require(tres.get('cpu')=='8' and tres.get('mem') in ('64G','65536M') and tres.get('gres/gpu')==tres.get('gres/gpu:l40s')=='1',
            'Actual requested/allocated CPU8 RAM64GiB L40S1 required')
    c.require(runner(['scontrol','write','batch_script',job,'-'],timeout=30)==(HERE/'run.sbatch').read_bytes(),'Actual spool changed')
    remaining=base.duration(fields['TimeLimit'])-base.duration(fields['RunTime']);c.require(remaining>300,'No accepted-boundary reserve')
    return dict(job_id=job,restart=int(restart),node=fields['NodeList'],raw_slurm=raw,deadline=time.time()+remaining-300,
        control_attempt=str(CONTROL/('g%d'%shard)/('attempt-'+job+'-r'+restart)))

def custody(binding,cfg,current,runner=subprocess.check_output):
    claim=c.checked(binding['function_claim']);grant=c.checked(binding['retained_custody']);storage=c.checked(binding['storage_reservation'])
    c.require(claim['owner']=='Root' and claim['scope']==SCOPE and claim['configuration']==binding['configuration']
        and claim['no_competing_overlay_provider'] is True,'Actual distinct Root function claim')
    c.require(grant['status']=='PASS_ROOT_EXACT_DT_RETAINED_CUSTODY_V1' and grant['owner']=='Root'
        and grant['configuration']==binding['configuration'] and grant['global_shard_index']==cfg['global_shard_indices'][0]
        and grant['retained_databases']==cfg['retained_databases'] and grant['old_writers_terminal'] is True
        and grant['producer_aliases']==cfg['predecessor_aliases'] and grant['actual_mapping']==cfg['actual_mapping'],
        'Actual terminal source/custody review; no live-writer takeover')
    raw=runner(['squeue','-r','-h','-u','luyangz','-o','%i|%T'],text=True,timeout=30)
    aliases=set(cfg['predecessor_aliases'])
    c.require(aliases and all(re.fullmatch(r'[0-9]+(?:_[0-9]+)?',v) for v in aliases),'Exact predecessor aliases')
    for line in raw.splitlines():
        fields=line.split('|');c.require(len(fields)==2 and re.fullmatch(r'[0-9]+(?:_[0-9]+)?',fields[0]),'Malformed expanded current queue')
        c.require(fields[0] not in aliases,'A retained source writer is still live/pending')
    c.require(storage['owner']=='Root' and storage['configuration']==binding['configuration'] and storage['output']==cfg['output']
        and storage['reserved_bytes']>=cfg['required_storage_bytes'] and storage['collective_claims_accounted'] is True,
        'Genuine measured collective subreservation required')
    c.seal(Path(current['control_attempt'])/'RETAINED_CUSTODY_CURRENT_QUEUE.json',dict(custody=binding['retained_custody'],raw_queue=raw))
    return storage

def execute(cfg,binding,current,storage):
    from experiments.copilot7_fullpool_published_20261024 import worker as original_worker
    old=c.checked(cfg['original_spec']);model=base.native_config(cfg,old)
    runtime=original_worker.runtime(old)
    pool,records,ids=base.load_pool(cfg,current)
    c.require(pool['count']==len(records)==len(ids)==660763,'Exact eligible whole-conversation population')
    base.verify_assets(old);original_worker.verify_fidelity(old)
    output=base.within(cfg['output'],OUTPUT_ROOT);output.parent.mkdir(parents=True,exist_ok=True)
    vfs=os.statvfs(output.parent)
    c.require(vfs.f_bavail*vfs.f_frsize>=storage['reserved_bytes']+storage['other_outstanding_reserved_bytes']
        and vfs.f_favail>=2000,'Actual current capacity below collective reservation')
    c.seal(Path(current['control_attempt'])/'ACTUAL_RUNTIME_AND_STORAGE.json',dict(runtime=runtime,
        available_bytes=vfs.f_bavail*vfs.f_frsize,available_inodes=vfs.f_favail,storage=binding['storage_reservation']))
    sys.path.insert(0,str(ROOT/c.NATIVE));module=importlib.import_module('llava.train.train')
    c.require(Path(module.__file__).resolve()==(ROOT/c.NATIVE/'llava/train/train.py').resolve(),'Wrong native constructor')
    stop=[False]
    def request_stop(*_):stop[0]=True
    signal.signal(signal.SIGUSR1,request_stop);signal.signal(signal.SIGTERM,request_stop)
    def should_stop():
        fs=os.statvfs(output.parent);c.require(fs.f_bavail*fs.f_frsize>=512<<20 and fs.f_favail>=500,'Actual storage safety floor')
        return stop[0] or time.time()>=current['deadline']
    return consumer.acquire(model,pool,records,ids,cfg['retained_databases'],cfg['actual_parity'],cfg['actual_parity_review'],
        module,output,output/('attempt-'+current['job_id']+'-r'+str(current['restart'])),should_stop,bound_configuration=binding['configuration'])

def complete_or_continue(ref,current,cfg,runner=subprocess.check_output):
    progress=c.checked(ref);control=Path(current['control_attempt'])
    c.seal(control/'RESULT.json',dict(progress=ref,actual=current,new_forwards=progress['new_forwards'],
        retained_records=progress['retained_records'],before=progress['before'],after=progress['after'],required=progress['required'],
        expandedNN_fixture_or_scientific_PASS=False,all_attempt_cost_requires_sacct=True))
    if progress['status']=='COMPLETE_COMPATIBLE_COMPOSITE_FEATURE_SHARD':return 0
    c.require(progress['status']=='MISSING_ONLY_COMPATIBLE_FEATURE_PROGRESS' and progress['after']<progress['required']
        and current['restart']<cfg['maximum_restarts'],'Finite incomplete attempt requires Root diagnosis')
    c.require(not (control/'REQUEUE_INTENT.json').exists(),'Existing ambiguous continuation: reconcile, do not retry')
    c.seal(control/'REQUEUE_INTENT.json',dict(job_id=current['job_id'],restart=current['restart'],accepted_progress=ref))
    try:ack=runner(['scontrol','requeue',current['job_id']],text=True,timeout=30)
    except BaseException as error:
        c.seal(control/'REQUEUE_AMBIGUOUS.json',dict(error_class=type(error).__name__,no_retry=True));raise
    c.seal(control/'REQUEUE_ACK.json',dict(stdout=ack,requested_only=True));return 0

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--binding',required=True);args=parser.parse_args(argv)
    binding=c.read(base.within(args.binding,CONTROL))
    base.within(binding['configuration']['path'],CONTROL);base.within(binding['review']['path'],CONTROL)
    cfg=c.checked(binding['configuration'])
    verify_config(cfg,binding['configuration'],c.checked(binding['review']));current=actual(binding,cfg)
    control=base.within(current['control_attempt'],CONTROL);control.mkdir(parents=True,exist_ok=False)
    c.seal(control/'ACTUAL_START.json',dict(actual=current,configuration=binding['configuration']))
    try:
        with (control.parent/'ROOT_SCALAR_EXECUTION.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            storage=custody(binding,cfg,current)
            ref=execute(cfg,binding,current,storage)
            return complete_or_continue(ref,current,cfg)
    except BaseException as error:
        c.seal(control/'FAILURE.json',dict(error_class=type(error).__name__,error=str(error),actual=current,
            no_blind_retry=True,preserve_all_paid_inputs=True));raise

if __name__=='__main__':raise SystemExit(main())
