"""GPU-allocation supervisor, exact binding and same-job-only continuation."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
from . import common as c
from .config import validate

def actual(binding, config, profile, runner=subprocess.check_output):
    job=os.environ.get('SLURM_JOB_ID','')
    c.require(job.isdecimal() and job==binding['job_id'], 'Actual scalar job identity differs')
    c.require(not os.environ.get('SLURM_ARRAY_JOB_ID'), 'This worker uses scalar immutable job identity')
    raw=runner(['scontrol','show','job','-o',job],text=True)
    import re
    fields=dict(re.findall(r'(\w+)=(\S+)',raw))
    for key,value in dict(JobName='dsmm-native-'+config['id'],Account=profile['account'],
        Partition=profile['partition'],QOS=profile['qos'],NumNodes='1',NumTasks='1',JobState='RUNNING',
        WorkDir=profile['code_root'],Command=str(c.HERE/'run.sbatch')).items():
        c.require(fields.get(key)==value, 'Actual Slurm '+key+' differs')
    c.require(fields.get('UserId','').split('(')[0]==profile['slurm_user'],'Actual Slurm user differs')
    c.require(fields.get('Restarts','').isdecimal()
        and int(fields['Restarts'])==int(os.environ.get('SLURM_RESTART_COUNT','0')),
        'Attempt restart identity differs from actual Slurm Restarts')
    expected='gres/gpu:'+profile['gpu_type']+':'+str(config['allocation']['gpus'])
    c.require(fields.get('TresPerNode')==expected, 'Case-sensitive GPU type/count differs')
    c.require(fields.get('Comment')=='dsmm-native:'+c.checked(binding['intent'])['nonce'], 'Actual ownership nonce differs')
    c.require(int(fields['NumCPUs'])>=config['allocation']['cpus'], 'Allocated CPU count too small')
    tres=dict(item.split('=',1) for item in fields['ReqTRES'].split(',') if '=' in item)
    memory=re.fullmatch(r'(\d+(?:\.\d+)?)([KMGT]?)',tres.get('mem',''))
    c.require(memory is not None,'Actual memory request unrecognized')
    memory_mb=float(memory[1])*{'K':1/1024,'':1,'M':1,'G':1024,'T':1024**2}[memory[2]]
    c.require(memory_mb==config['allocation']['memory_mb'],'Actual memory request differs')
    return raw

def main():
    supervisor_started=time.monotonic()
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);a=p.parse_args()
    binding=c.read(a.binding);config=c.checked(binding['configuration']);profile,_=validate(config)
    raw=actual(binding,config,profile)
    report=Path(config['report']);report.mkdir(parents=True,exist_ok=True)
    lock=open(report/'EXECUTION.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=c.run_identity(config)
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    attempt=f"{binding['job_id']}-r{os.environ.get('SLURM_RESTART_COUNT','0')}-{uuid.uuid4().hex[:10]}"
    attempt_path=report/'attempts'/attempt;attempt_path.mkdir(parents=True)
    stop=attempt_path/'STOP_REQUESTED'
    def request_stop(*_):
        stop.touch()
    signal.signal(signal.SIGUSR1,request_stop);signal.signal(signal.SIGTERM,request_stop)
    c.seal(attempt_path/'START.json',dict(identity=identity,binding=c.desc(a.binding),
        UTC=c.now(),actual_slurm=raw,allocation=config['allocation']))
    if (output/'TRAIN_COMPLETE.json').exists():
        from .native import checkpoint_receipt
        complete=c.read(output/'TRAIN_COMPLETE.json')
        c.require(complete['identity']==identity,'Completed run identity differs')
        checkpoint_receipt(Path(complete['accepted']['path']).parent,identity)
        c.seal(attempt_path/'RESULT.json',dict(status='ALREADY_COMPLETE_NO_TRAINING_REPLAY',identity=identity))
        return
    env=dict(os.environ,HF_HOME=profile['cache_root'],HF_HUB_CACHE=str(Path(profile['cache_root'])/'hub'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS=str(max(1,config['allocation']['cpus']//config['allocation']['gpus'])),
        NATIVE_STOP_PATH=str(stop),NATIVE_ATTEMPT=str(attempt_path),NATIVE_RUN_IDENTITY=identity)
    end_epoch=int(os.environ.get('SLURM_JOB_END_TIME','0'))
    stop_epoch=end_epoch-config['allocation']['reserve_seconds']
    c.require(stop_epoch>time.time(), 'No remaining allocation time after preparation/checkpoint reserve')
    # torchrun one process/GPU on ONE NODE, in the exact native environment.
    command=[profile['python'],'-B','-m','torch.distributed.run','--standalone',
        '--nproc_per_node='+str(config['allocation']['gpus']),'-m',
        'experiments.root_native_campaign_v7_20261024.native','--config',binding['configuration']['path'],
        '--stop-at-epoch',str(stop_epoch)]
    start=time.monotonic()
    with open(attempt_path/'torchrun.log','ab',buffering=0) as log:
        child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        rc=child.wait()
    final=output/'TRAIN_COMPLETE.json'
    continuation=attempt_path/'CONTINUATION.json'
    status='TRAINING_COMPLETE' if final.exists() and c.read(final)['identity']==identity else 'ACCEPTED_CONTINUATION' if continuation.exists() else 'FAILED_NO_ACCEPTED_CONTINUATION'
    c.seal(attempt_path/'RESULT.json',dict(status=status,rc=rc,identity=identity,elapsed_seconds=time.monotonic()-start,
        supervisor_elapsed_seconds=time.monotonic()-supervisor_started,
        supervisor_gpu_hours=config['allocation']['gpus']*(time.monotonic()-supervisor_started)/3600,
        pre_supervisor_slurm_cost_must_be_added=True,
        scientific_evaluation_complete=False,qualification_promoted=False))
    if status=='ACCEPTED_CONTINUATION':
        c.require(rc==0,'A rank failed despite a rank-zero continuation receipt; preserve state and diagnose, do not requeue')
        receipt=c.read(continuation)
        c.require(receipt['identity']==identity, 'Continuation identity differs')
        # Same physical job only: never submit a new CPU/GPU successor at a site ceiling.
        c.seal(attempt_path/'REQUEUE_INTENT.json',dict(job_id=binding['job_id'],accepted_checkpoint=receipt['accepted'],UTC=c.now()))
        result=subprocess.run(['scontrol','requeue',binding['job_id']],capture_output=True,text=True)
        c.seal(attempt_path/'REQUEUE_RESPONSE.json',dict(rc=result.returncode,stdout=result.stdout,stderr=result.stderr))
        c.require(result.returncode==0, 'Same-job requeue not acknowledged; do not submit a replacement')
        return
    c.require(status=='TRAINING_COMPLETE' and rc==0, 'Native failed: preserve log and accepted state; diagnosis required')

if __name__=='__main__':main()
