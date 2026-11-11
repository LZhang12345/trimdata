"""One fresh process per owned cell. No automatic reruns or scheduler mutations."""
import argparse
from datetime import datetime
import fcntl
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
from . import protocol as p

def actual(binding,study,cfg,runner=subprocess.check_output):
    cell=binding['cell'];job=os.environ.get('SLURM_JOB_ID','')
    p.require(binding['schema']=='native7-replay-run-binding-v1' and binding['owner']=='Root'
        and cell in p.CELLS and job==binding['job_id']==study['jobs'][cell]
        and not os.environ.get('SLURM_ARRAY_JOB_ID'),'Exact scalar replay binding required')
    held=p.checked(binding['held_binding']);intent=p.checked(held['intent'])
    p.require(held['design']==study['design'] and held['cell']==cell and held['job_id']==job
        and held['spool_reconciled'] is True and held['driver_sources']==p.sources()
        and intent['status']=='INTENT_BEFORE_SUBMISSION','Exact reconciled held intent required')
    profile=p.checked(cfg['site_profile']);raw=runner(['scontrol','show','job','-o',job],text=True,timeout=30)
    f=dict(re.findall(r'(?:^|\s)([^\s=]+)=(\S+)',raw))
    expected=dict(JobId=job,JobName='dsmm-replay-'+cell,JobState='RUNNING',Account=profile['account'],
        Partition='gpu',QOS='normal',NumNodes='1',NumTasks='1',NumCPUs='8',
        WorkDir=str(p.ROOT),Command=str(p.HERE/'run.sbatch'),Comment='dsmm-replay:'+intent['nonce'],
        TimeLimit='00:55:00',TresPerNode='gres/gpu:h100:1',Restarts='0')
    for name,value in expected.items():p.require(f.get(name)==value,'Actual replay scheduler field differs: '+name)
    p.require(f.get('UserId')==profile['slurm_user']+'('+str(os.getuid())+')'
        and f.get('NodeList')==socket.gethostname().split('.')[0]
        and f.get('StartTime') not in (None,'Unknown','N/A') and f.get('CPUs/Task')=='8',
        'Actual UID/node/start/CPU binding differs')
    tres=dict(x.split('=',1) for x in f.get('ReqTRES','').split(',') if '=' in x)
    p.require(tres.get('mem') in ('64G','65536M') and tres.get('gres/gpu')=='1','Actual memory/GPU differs')
    p.require(os.environ.get('SLURM_RESTART_COUNT','0')=='0','Fresh-process control cannot be a resumed job')
    os.environ['SLURM_RESTART_COUNT']='0'
    end=int(datetime.fromisoformat(f['EndTime']).timestamp())
    supplied=os.environ.get('SLURM_JOB_END_TIME')
    p.require(supplied is None or abs(int(supplied)-end)<=1,'Stale hard allocation deadline')
    os.environ['SLURM_JOB_END_TIME']=str(end)
    p.require(end-time.time()>=cfg['allocation']['reserve_seconds']+600,'Actual allocation lacks staging/training reserve')
    return profile,raw,end

def run(binding):
    study=p.checked(binding['study']);pre=p.checked(study['native_configurations'][binding['cell']])
    profile,raw,end=actual(binding,study,pre)
    # Cheap actual API attestation precedes all cold model payload validation.
    from experiments.root_native_qualification_v10_20261024.preflight import runtime_preflight
    runtime=runtime_preflight(p.checked(study['anchor_study']),site='aces')
    configs,modules=p.validate(study);cell=binding['cell'];cfg=configs[cell]
    report=p.literal(cfg['report']);report.mkdir(parents=True,exist_ok=True)
    with (report/'EXECUTION.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p.require(not (report/'ATTEMPT_START.json').exists(),'Single fresh trajectory already attempted; no blind rerun')
        p.require(not p.literal(cfg['output']).exists(),'Fresh trajectory output already exists')
        started=time.monotonic();attempt=report/'attempt';attempt.mkdir(exist_ok=False)
        p.seal(report/'ATTEMPT_START.json',dict(binding=binding,actual_slurm=raw,UTC=p.now(),current_runtime=runtime))
        stop=attempt/'STOP_REQUESTED'
        def request_stop(*_):stop.touch()
        signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGUSR1,request_stop)
        from experiments.root_native_media_transport_20261023 import transport
        selected=p.checked(cfg['providers']['selection'])
        alias=transport.prepare(cfg['providers']['pool'],selected['records_json'],cfg['providers']['media_transport'],attempt/'transport')
        physical=p.h.physical_config(cfg,alias,validate_native=modules['config'].validate,run_identity=p.n.run_identity)
        physical_ref=p.seal(attempt/'PHYSICAL_CONFIG.json',physical)
        stop_at=end-cfg['allocation']['reserve_seconds']
        p.require(stop_at-time.time()>600,'Staging consumed real launch reserve')
        env=dict(os.environ,NATIVE_STOP_PATH=str(stop),NATIVE_ATTEMPT=str(attempt),
            NATIVE_RUN_IDENTITY=p.n.run_identity(physical),HF_HOME=profile['cache_root'],
            HF_HUB_CACHE=str(Path(profile['cache_root'])/'hub'),OMP_NUM_THREADS='8')
        argv=[profile['python'],'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=1','-m',p.PACKAGE+'.child',
            '--study',binding['study']['path'],'--cell',cell,'--physical-config',physical_ref['path'],
            '--alias',alias['path'],'--attempt',str(attempt),'--stop-at-epoch',str(stop_at)]
        p.seal(attempt/'INVOCATION.json',dict(argv=argv,binding=binding,physical_config=physical_ref,actual_end=end))
        with (attempt/'torchrun.log').open('xb',buffering=0) as log:
            rc=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT).wait()
        child=attempt/'CHILD_RESULT.json'
        p.seal(report/'RESULT.json',dict(status='ACTUAL_REPLAY_CELL_RETURNED_NOT_PROMOTED',returncode=rc,
            cell=cell,actual_job_id=binding['job_id'],child=p.desc(child) if child.exists() else None,
            seconds_including_preflight_transport_and_observation=time.monotonic()-started,
            scheduler_setup_and_failed_attempts_count_separately=True,qualification_promoted=False))
        p.require(rc==0 and child.exists(),'Actual replay failure; preserve states before any repair')
        v=p.checked(p.desc(child));o=p.checked(v['observation'])
        p.expected_endpoint(cell,v['status'],p.checked(v['files']['STEP-'+str(p.CELLS[cell])+'.json'])['global_step'],o['actual_record_exposure'])
        complete=p.seal(report/'CELL_COMPLETE.json',dict(status='COMPLETED_FINITE_REPLAY_CONTROL_NOT_QUALIFICATION',
            cell=cell,study=binding['study'],child=p.desc(child),qualification_promoted=False))
        if end-time.time()>=600 and all((Path(value['report'])/'CELL_COMPLETE.json').exists() for value in configs.values()):
            from .readout import collect
            collect(binding['study'])
        return complete

def main():
    a=argparse.ArgumentParser();a.add_argument('--binding',required=True);x=a.parse_args();run(p.read(x.binding))

if __name__=='__main__':main()
