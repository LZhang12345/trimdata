"""One owned CPU+GPU job: uninterrupted arm, accepted-stop arm, actual requeue.

Never submits a successor. Native scientific students remain separately held.
"""
import argparse
import fcntl
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import uuid
from . import common as c
from . import study as validation


def parse_job(raw):return dict(re.findall(r'(\w+)=(\S+)',raw))

def actual(binding,study,config,runner=subprocess.check_output):
    profile=c.checked(config['site_profile']);job=os.environ.get('SLURM_JOB_ID')
    c.require(binding['schema']=='native-engineering-binding-v1' and binding['owner']=='Root' and job==binding['job_id'],'Exact engineering job binding required')
    c.require(not os.environ.get('SLURM_ARRAY_JOB_ID'),'Engineering parity uses one scalar physical job')
    intent=c.checked(binding['intent'])
    c.require(intent['status']=='INTENT_BEFORE_SUBMISSION' and intent['owner']=='Root'
        and intent['study']==binding['study'] and intent['driver_sources']==study['driver_sources']
        and intent['site_profile']==config['site_profile'] and intent['allocation']==config['allocation'],
        'Durable exact Root engineering submission intent differs')
    raw=runner(['scontrol','show','job','-o',job],text=True);fields=parse_job(raw)
    exact=dict(JobId=job,JobName='dsmm-eng-'+study['id'],Account=profile['account'],Partition=profile['partition'],QOS=profile['qos'],
        NumNodes='1',NumTasks='1',JobState='RUNNING',WorkDir=profile['code_root'],Command=str(c.HERE/'run.sbatch'),
        Comment='dsmm-eng:'+intent['nonce'],TresPerNode='gres/gpu:'+profile['gpu_type']+':1')
    for name,value in exact.items():c.require(fields.get(name)==value,'Actual engineering Slurm '+name+' differs')
    c.require(fields.get('UserId','').split('(')[0]==profile['slurm_user'],'Actual engineering user differs')
    c.require(fields.get('Restarts','').isdecimal() and int(fields['Restarts'])==int(os.environ.get('SLURM_RESTART_COUNT','0')),
        'Actual scheduler Restarts differs from engineering attempt environment')
    c.require(int(fields['NumCPUs'])>=config['allocation']['cpus'],'Actual engineering CPU allocation too small')
    tres=dict(item.split('=',1) for item in fields['ReqTRES'].split(',') if '=' in item)
    memory=re.fullmatch(r'(\d+(?:\.\d+)?)([KMGT]?)',tres.get('mem',''))
    c.require(memory is not None,'Unknown allocated engineering memory')
    mb=float(memory[1])*{'K':1/1024,'':1,'M':1,'G':1024,'T':1024**2}[memory[2]]
    c.require(mb==config['allocation']['memory_mb'],'Actual engineering memory differs')
    c.require(c.within(study['report'],profile['registry_root']),'Engineering report outside owned registry')
    return profile,raw


def invoke(study,study_descriptor,arm,phase,config,profile,attempt,stop_at_epoch,current_stop):
    attempt.mkdir(parents=True,exist_ok=False);stop=attempt/'STOP_REQUESTED';current_stop[0]=stop
    modules,_=c.target_modules(study['native_target'])
    identity=validation.native_identity(config,modules)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HOME=profile['cache_root'],HF_HUB_CACHE=str(Path(profile['cache_root'])/'hub'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS=str(config['allocation']['cpus']),
        NATIVE_STOP_PATH=str(stop),NATIVE_ATTEMPT=str(attempt),NATIVE_RUN_IDENTITY=identity)
    command=[profile['python'],'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=1','-m',
        'experiments.root_native_qualification_v6_20261024.child','--study',study_descriptor['path'],'--arm',arm,'--phase',phase,
        '--attempt',str(attempt),'--stop-at-epoch',str(stop_at_epoch)]
    c.seal(attempt/'INVOCATION.json',dict(command=command,study=study_descriptor,arm=arm,phase=phase,identity=identity,
        UTC=c.now(),absolute_stop_epoch=stop_at_epoch,actual_job_id=os.environ['SLURM_JOB_ID'],
        actual_restart_count=int(os.environ.get('SLURM_RESTART_COUNT','0'))))
    started=time.monotonic()
    with open(attempt/'torchrun.log','xb',buffering=0) as log:
        child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT);rc=child.wait()
    current_stop[0]=None
    result=attempt/'CHILD_RESULT.json'
    c.seal(attempt/'SUPERVISOR_RESULT.json',dict(rc=rc,actual_elapsed_seconds=time.monotonic()-started,
        child_result=c.desc(result) if result.exists() else None))
    c.require(rc==0 and result.exists(),'Actual native attempt failed; preserve logs and accepted state, no blind retry')
    return c.desc(result)


def requeue(binding,report,interrupted):
    intent=c.seal(report/'REQUEUE_INTENT.json',dict(job_id=binding['job_id'],interrupted=interrupted,UTC=c.now(),same_physical_job_only=True))
    response=subprocess.run(['scontrol','requeue',binding['job_id']],capture_output=True,text=True)
    c.seal(report/'REQUEUE_RESPONSE.json',dict(intent=intent,rc=response.returncode,stdout=response.stdout,stderr=response.stderr))
    c.require(response.returncode==0,'Actual same-job requeue unacknowledged; do not submit a new successor')


def run(binding):
    started=time.monotonic();study=c.checked(binding['study']);configs,modules=validation.validate_study(study)
    profile,slurm=actual(binding,study,configs['reference'])
    report=Path(study['report']);report.mkdir(parents=True,exist_ok=True)
    lock=open(report/'EXECUTION.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run_id=str(binding['job_id'])+'-r'+os.environ.get('SLURM_RESTART_COUNT','0')+'-'+uuid.uuid4().hex[:10]
    attempt=report/'attempts'/run_id;attempt.mkdir(parents=True,exist_ok=False)
    c.seal(attempt/'START.json',dict(binding=binding,actual_slurm=slurm,UTC=c.now(),driver_sources=study['driver_sources']))
    current_stop=[None]
    def stop(*_):
        if current_stop[0] is not None:current_stop[0].touch()
    signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
    end_epoch=int(os.environ.get('SLURM_JOB_END_TIME','0'))
    stop_at=end_epoch-configs['reference']['allocation']['reserve_seconds']
    c.require(stop_at>time.time(),'Insufficient actual allocation remaining after CPU+GPU validation')
    if (report/'RAW_READOUT.json').exists():
        c.seal(attempt/'RESULT.json',dict(status='EXISTING_RAW_READOUT_NO_TRAINING_REPLAY',readout=c.desc(report/'RAW_READOUT.json')))
        return
    descriptors={}
    ref=report/'PHASE_REFERENCE.json';cut=report/'PHASE_INTERRUPTED.json';resume=report/'PHASE_RESUMED.json'
    if not ref.exists():
        c.require(not cut.exists() and not resume.exists(),'Out-of-order phase state')
        result=invoke(study,binding['study'],'reference','reference',configs['reference'],profile,attempt/'reference',stop_at,current_stop)
        c.require(c.checked(result)['status']=='NATIVE_TRAJECTORY_COMPLETE','Reference was interrupted: cannot call it uninterrupted, new reviewed study required')
        c.seal(ref,result)
    descriptors['reference']=c.read(ref)
    if not cut.exists():
        c.require(not resume.exists(),'Resumed arm exists without an interruption')
        c.require(stop_at-time.time()>study['minimum_launch_seconds'],'Not enough measured walltime to start interruption arm safely; new reviewed allocation needed')
        result=invoke(study,binding['study'],'resume','interrupted',configs['resume'],profile,attempt/'interrupted',stop_at,current_stop)
        value=c.checked(result)
        c.require(value['status']=='ACCEPTED_NATIVE_CONTINUATION','Expected accepted deliberate interruption, not a terminal trajectory')
        checkpoint=Path(configs['resume']['output'])/'training'/('checkpoint-'+str(study['interrupt_step']))
        identity=validation.native_identity(configs['resume'],modules)
        modules['native'].checkpoint_receipt(checkpoint,identity)
        c.seal(cut,result);requeue(binding,report,result)
        return
    descriptors['interrupted']=c.read(cut)
    previous=c.checked(descriptors['interrupted'])
    c.require(previous['actual_job_id']==binding['job_id'] and int(os.environ.get('SLURM_RESTART_COUNT','0'))>previous['actual_restart_count'],
        'Restarted phase requires actual same physical scheduler job requeue')
    if not resume.exists():
        result=invoke(study,binding['study'],'resume','resumed',configs['resume'],profile,attempt/'resumed',stop_at,current_stop)
        c.require(c.checked(result)['status']=='NATIVE_TRAJECTORY_COMPLETE','Post-restart arm did not complete; preserve actual continuation for diagnosis')
        c.seal(resume,result)
    descriptors['resumed']=c.read(resume)
    phases=c.seal(report/'PHASE_RESULTS.json',descriptors)
    try:
        from .readout import compare
        raw=compare(study,descriptors)
    except BaseException as exc:
        c.seal(report/'READOUT_FAILURE.json',dict(status='FAILED_ACTUAL_READOUT',error=repr(exc),phase_results=phases,UTC=c.now()))
        raise
    readout=c.seal(report/'RAW_READOUT.json',raw)
    c.seal(attempt/'RESULT.json',dict(status=raw['status'],raw_readout=readout,
        actual_supervisor_seconds=time.monotonic()-started,actual_supervisor_gpu_hours=(time.monotonic()-started)/3600,
        all_allocated_startup_prior_attempts_and_setup_costs_must_be_added=True,qualification_promoted=False))
    c.require(raw['status']=='PASS_ACTUAL_ENGINEERING_READOUT_UNREVIEWED','Actual parity failed; inspect counterevidence before any repair')


def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);a=p.parse_args();run(c.read(a.binding))

if __name__=='__main__':main()
