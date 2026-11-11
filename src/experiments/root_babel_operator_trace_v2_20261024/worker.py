"""One owned fresh L40S trace; canonical Babel images and paid metadata reuse."""
import argparse
from datetime import datetime
import fcntl,os,re,signal,socket,subprocess,time
from pathlib import Path
from . import protocol as p

def actual(binding,study,cfg,runner=subprocess.check_output):
    job=os.environ.get('SLURM_JOB_ID','');cell=binding['cell']
    p.require(binding['schema']=='babel-native7-operator-trace-run-binding-v2' and binding['owner']=='Root'
        and cell=='T1' and job==binding['job_id']==study['jobs'][cell] and not os.environ.get('SLURM_ARRAY_JOB_ID'),'Exact scalar Babel trace binding')
    held=p.checked(binding['held_binding']);intent=p.checked(held['intent']);profile=p.checked(cfg['site_profile'])
    p.require(held['design']==study['design'] and held['cell']==cell and held['job_id']==job and held['spool_reconciled'] is True
        and held['driver_sources']==p.sources() and intent['status']=='INTENT_BEFORE_SUBMISSION','Exact current-job held intent required')
    raw=runner(['scontrol','show','job','-o',job],text=True,timeout=30);f=dict(re.findall(r'(?:^|\s)([^\s=]+)=(\S+)',raw))
    expected=dict(JobId=job,JobName='dsmm-babel-optrace-T1',JobState='RUNNING',Account='cx',Partition='general',QOS='normal',NumNodes='1',NumTasks='1',NumCPUs='8',WorkDir=str(p.ROOT),Command=str(p.HERE/'run.sbatch'),Comment='dsmm-babel-optrace:'+intent['nonce'],TimeLimit='00:55:00',TresPerNode='gres/gpu:L40S:1',Restarts='0')
    p.require(all(f.get(k)==v for k,v in expected.items()) and profile['site']=='babel' and profile['slurm_user']=='luyangz'
        and f.get('UserId')=='luyangz('+str(os.getuid())+')' and f.get('NodeList')==socket.gethostname().split('.')[0]
        and f.get('StartTime') not in (None,'Unknown','None','N/A') and f.get('CPUs/Task')=='8','Actual owned Babel General L40S route differs')
    allocated=dict(x.split('=',1) for x in f.get('AllocTRES','').split(',') if '=' in x)
    p.require(allocated.get('cpu')=='8' and allocated.get('mem') in ('64G','65536M') and allocated.get('gres/gpu')=='1','Actual CPU/GPU/memory allocation differs')
    p.require(os.environ.get('SLURM_RESTART_COUNT','0')=='0','Fresh-process diagnostic cannot restart')
    os.environ['SLURM_RESTART_COUNT']='0'
    end=int(datetime.fromisoformat(f['EndTime']).timestamp());given=os.environ.get('SLURM_JOB_END_TIME')
    p.require(given is None or abs(int(given)-end)<=1,'Stale allocation deadline');os.environ['SLURM_JOB_END_TIME']=str(end)
    p.require(p.RUNTIME_BUDGET['minimum_initial_remaining_seconds']<=end-time.time()<=3305,'Real55-minute deadline lacks launch reserve')
    spool=runner(['scontrol','write','batch_script',job,'-'],timeout=30)
    p.require(spool==(p.HERE/'run.sbatch').read_bytes(),'Actual current-job batch spool changed')
    return profile,raw,end

def child_argv(profile,study_ref,cell,config_ref,attempt,stop_at):
    return [profile['python'],'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=1','-m',p.PACKAGE+'.child',
        '--study',study_ref['path'],'--cell',cell,'--physical-config',config_ref['path'],'--attempt',str(attempt),'--stop-at-epoch',str(stop_at)]

def run(binding):
    study=p.checked(binding['study']);cfg=p.checked(study['native_configurations'][binding['cell']])
    profile,raw,end=actual(binding,study,cfg)
    from experiments.root_native_qualification_v10_20261024.preflight import runtime_preflight
    runtime=runtime_preflight(p.checked(study['anchor_study']),site='babel')
    pair,modules=p.validate(study);cfg=pair['T1'];report=p.literal(cfg['report']);report.mkdir(parents=True,exist_ok=True)
    with (report/'EXECUTION.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p.require(not (report/'ATTEMPT_START.json').exists() and not p.literal(cfg['output']).exists(),'One fresh diagnostic attempt only; preserve prior results')
        began=time.monotonic();attempt=report/'attempt';attempt.mkdir(exist_ok=False)
        p.seal(report/'ATTEMPT_START.json',dict(binding=binding,actual_slurm=raw,current_runtime=runtime,UTC=p.now()))
        # Reuse authentic paid Babel readout only. H100 G1a/G1b/U2 tensors are
        # different hardware controls and are not relabelled as Babel results.
        p.seal(attempt/'PAID_BABEL_READOUT_REUSE.json',dict(source=study['paid_anchor_readout'],status='REUSED_ACTUAL_FAILED_BABEL_V10_METADATA_ONLY',new_tensor_readout_or_training=False,qualification_promoted=False))
        stop=attempt/'STOP_REQUESTED'
        def stopping(*_):stop.touch()
        signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGUSR1,stopping)
        # Full model/assets/current canonical image checks run in this owned GPU
        # allocation before a new neural child, and consume its real deadline.
        modules['config'].validate(cfg)
        stop_at=end-cfg['allocation']['reserve_seconds']
        p.require(stop_at-time.time()>=p.RUNTIME_BUDGET['minimum_neural_seconds'],'Cold validation consumed real native launch window')
        cfg_ref=study['native_configurations']['T1'];env=dict(os.environ,NATIVE_STOP_PATH=str(stop),NATIVE_ATTEMPT=str(attempt),NATIVE_RUN_IDENTITY=p.n.run_identity(cfg),HF_HOME=profile['cache_root'],HF_HUB_CACHE=str(Path(profile['cache_root'])/'hub'),OMP_NUM_THREADS='8')
        argv=child_argv(profile,binding['study'],'T1',cfg_ref,attempt,stop_at)
        p.seal(attempt/'INVOCATION.json',dict(argv=argv,binding=binding,native_configuration=cfg_ref,actual_end=end))
        with (attempt/'torchrun.log').open('xb',buffering=0) as log:rc=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT).wait()
        child=attempt/'CHILD_RESULT.json'
        p.seal(report/'RESULT.json',dict(status='ACTUAL_BABEL_TRACE_RETURNED_NOT_PROMOTED',returncode=rc,actual_job_id=binding['job_id'],child=p.desc(child) if child.exists() else None,seconds_including_validation_and_observation=time.monotonic()-began,qualification_promoted=False))
        p.require(rc==0 and child.exists(),'Preserve actual failed trace before any repair')
        result=p.checked(p.desc(child));observed=p.checked(result['observation']);step=p.checked(result['files']['STEP-1.json'])
        p.expected_endpoint('T1',result['status'],step['global_step'],observed['actual_record_exposure'])
        complete=p.seal(report/'CELL_COMPLETE.json',dict(status='COMPLETED_FINITE_BABEL_TRACE_NOT_QUALIFICATION',cell='T1',study=binding['study'],child=p.desc(child),qualification_promoted=False))
        if end-time.time()>=p.RUNTIME_BUDGET['final_readout_minimum_seconds']:
            from .readout import collect
            collect(binding['study'])
        return complete

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--binding',required=True);args=parser.parse_args();run(p.read(args.binding))
if __name__=='__main__':main()
