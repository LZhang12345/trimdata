"""One real ownedGPU: finite F8 or reference8/cut4/same-job-requeue/resume8."""
import argparse,fcntl,os,pwd,re,signal,socket,subprocess,time,uuid
from pathlib import Path
from . import protocol as p,runtime

def fields(raw):return dict(re.findall(r'(?:^|\s)([^\s=]+)=(\S+)',raw))
def allocation(binding,study,runner=subprocess.check_output):
    cfg=p.checked(study['native_configurations']['reference']);profile=p.checked(cfg['site_profile']);job=os.environ.get('SLURM_JOB_ID','')
    p.require(job==binding['job_id']==study['job_id'] and job.isdecimal() and not os.environ.get('SLURM_ARRAY_JOB_ID') and pwd.getpwuid(os.getuid()).pw_name==profile['slurm_user']=='luyangz','Actual scalar Root-owned job required before DATA access')
    held=p.checked(study['held_binding']);intent=p.checked(held['intent']);raw=runner(['scontrol','show','job','-o',job],text=True,timeout=30);f=fields(raw)
    expected=dict(JobId=job,JobName='dsmm-d8rep43v3-'+study['cell'],JobState='RUNNING',UserId='luyangz('+str(os.getuid())+')',Account='cx',Partition='general',QOS='normal',NodeList=socket.gethostname().split('.')[0],NumNodes='1',NumTasks='1',NumCPUs='8',TimeLimit='00:55:00',Restarts=os.environ.get('SLURM_RESTART_COUNT','0'),Command=str(p.HERE/'run.sbatch'),WorkDir=str(p.ROOT),Comment='dsmm-d8rep43v3:'+intent['nonce'],TresPerNode='gres/gpu:L40S:1')
    p.require(all(f.get(k)==v for k,v in expected.items()),'Actual follow-on scheduler ownership/resources differ')
    tres=dict(x.split('=',1) for x in f.get('AllocTRES','').split(',') if '=' in x)
    p.require(tres.get('cpu')=='8' and tres.get('mem') in ('64G','65536M') and tres.get('gres/gpu')=='1','Actual allocatedTRES differs')
    from datetime import datetime
    end=int(datetime.fromisoformat(f['EndTime']).timestamp())
    if os.environ.get('SLURM_JOB_END_TIME'):p.require(int(os.environ['SLURM_JOB_END_TIME'])==end,'Runtime EndTime differs from actual scheduler')
    p.require(end>time.time()+p.ALLOCATION['reserve_seconds'],'Actual lifetime exhausted')
    return profile,raw,end

def requeue(job,attempt,reason,study,binding,runner=subprocess.run,query=subprocess.check_output):
    """Actual same-job restart; no new controller, retry or successor."""
    from . import admit
    cfg=p.checked(study['native_configurations']['reference']);profile=p.checked(cfg['site_profile'])
    admit.actor(profile)
    with admit.lock_path().open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        # Source/config/intent/ACK/spool, current two-slot census and site caps
        # are refreshed under the canonical lock before this mutation too.
        p.validate_study(study,gpu=False)
        current=admit.census(profile,need_submission_slot=False)
        authorization=p.checked(binding['Root_window_authorization'])
        p.require(any(r==dict(job_id=job,job_name='dsmm-d8rep43v3-'+study['cell'],gpus=1) for r in authorization['current_root_general_commitments']),'Current running job missing from exact Root authorization roster')
        eligible=admit.eligible_root_general(authorization,current,runner=query)
        p.require(eligible<=2,'Actual eligible Root commitments exceed author allocation')
        profile,raw,end=allocation(binding,study,runner=query)
        p.require(job==binding['job_id'],'Requeue exact current job only')
        spool=query(['scontrol','write','batch_script',job,'-'],timeout=30)
        p.require(spool==(p.HERE/'run.sbatch').read_bytes(),'Actual running spool differs')
        intent_path=attempt/'REQUEUE_INTENT.json'
        p.require(not intent_path.exists(),'Prior requeue intent requires reconciliation; no blind retry')
        intent=p.seal(intent_path,dict(owner='Root',prepared_by=p.AUTHOR_IDENTITY,job_id=job,reason=reason,
            same_physical_job_only=True,actual_slurm=raw,actual_census=current,
            refreshed_eligible_Root_GPUs=eligible,Root_window_authorization=binding['Root_window_authorization'],
            study=binding['study'],spool_sha256=p.sha(p.HERE/'run.sbatch'),UTC=p.now()))
        try:r=runner(['scontrol','requeue',job],capture_output=True,text=True,timeout=30)
        except subprocess.TimeoutExpired:
            p.seal(attempt/'REQUEUE_AMBIGUOUS.json',dict(intent=intent,no_automatic_retry=True));raise
        p.seal(attempt/'REQUEUE_ACK.json',dict(intent=intent,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
        p.require(r.returncode==0,'Requeue acknowledgement failed; no successor or blind retry')

def invoke(study_ref,study,configs,arm,phase,profile,attempt,stop_at,current):
    attempt.mkdir(parents=True,exist_ok=False);stop=attempt/'STOP_REQUESTED';current[0]=stop
    _,pre=p.completed_trace(study['completed_trace'],study['cell'])
    env=runtime.launch_environment(os.environ,pre,study['cell']);env.update(NATIVE_STOP_PATH=str(stop),NATIVE_ATTEMPT=str(attempt),NATIVE_RUN_IDENTITY=p.n.run_identity(configs[arm]),HF_HOME=profile['cache_root'],HF_HUB_CACHE=str(Path(profile['cache_root'])/'hub'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8')
    command=[profile['python'],'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=1','-m',p.PACKAGE+'.child','--study',study_ref['path'],'--arm',arm,'--phase',phase,'--attempt',str(attempt),'--stop-at-epoch',str(stop_at)]
    p.seal(attempt/'INVOCATION.json',dict(argv=command,study=study_ref,phase=phase,absolute_stop_epoch=stop_at,CUDA_runtime_factor=study['runtime'],actual_job_id=os.environ['SLURM_JOB_ID'],actual_restart_count=int(os.environ.get('SLURM_RESTART_COUNT','0')),UTC=p.now()))
    with (attempt/'torchrun.log').open('xb',buffering=0) as log:rc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT).wait()
    current[0]=None;result=attempt/'CHILD_RESULT.json';p.seal(attempt/'RETURN.json',dict(returncode=rc,result=p.desc(result) if result.exists() else None))
    p.require(rc==0 and result.exists(),'Actual native child failed; preserve logs/states, no automatic retry')
    value=p.read(result);p.require(value['phase']==phase and value['actual_job_id']==study['job_id'] and value['driver_sources']==p.sources(),'Child result identity differs')
    return p.desc(result)

def run(binding):
    study=p.checked(binding['study']);p.require(binding['owner']=='Root' and binding['schema']=='native-deterministic-replication-run-binding-v3','Exact binding required')
    # No pool, images, model files, tensor states, or output stat before actual
    # current ownedGPU identity and real EndTime have been authenticated.
    profile,raw,end=allocation(binding,study)
    configs,modules=p.validate_study(study,gpu=False)
    from experiments.root_native_qualification_v10_20261024.preflight import runtime_preflight,verify_shared
    initial=runtime_preflight(p.checked(study['anchor_study']),site='babel')
    report=p.literal(study['report']);report.mkdir(parents=True,exist_ok=True)
    with (report/'EXECUTION.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempt=report/'attempts'/(binding['job_id']+'-r'+os.environ.get('SLURM_RESTART_COUNT','0')+'-'+uuid.uuid4().hex[:8]);attempt.mkdir(parents=True,exist_ok=False)
        fs=os.statvfs(p.literal(profile['output_root']));reserve=p.CELLS[study['cell']]['reserved_bytes']
        p.require(fs.f_bavail*fs.f_frsize>=reserve and fs.f_favail>=4000,'Actual current output capacity below collective reservation')
        p.seal(attempt/'ACTUAL_CURRENT_STORAGE.json',dict(status='ACTUAL_CURRENT_OWNED_GPU_STORAGE_RECHECK',job_id=binding['job_id'],step_id=os.environ.get('SLURM_STEP_ID','batch'),available_bytes=fs.f_bavail*fs.f_frsize,available_inodes=fs.f_favail,required_bytes=reserve,prior_measurement=p.PRIOR_STORAGE,actual_slurm=raw,actual_EndTime=end,UTC=p.now()))
        p.seal(attempt/'START.json',dict(binding=binding,current_runtime=initial,actual_slurm=raw,actual_remaining_seconds=end-time.time(),UTC=p.now()))
        verify_shared(p.checked(study['anchor_study']),configs['reference'],raw=True)
        current=[None]
        def stopping(*_):
            if current[0] is not None:current[0].touch()
        signal.signal(signal.SIGUSR1,stopping);signal.signal(signal.SIGTERM,stopping)
        stop_at=end-p.ALLOCATION['reserve_seconds'];ref=report/'PHASE_REFERENCE.json';cut=report/'PHASE_INTERRUPTED.json';resume=report/'PHASE_RESUMED.json'
        if (report/'RAW_READOUT.json').exists():raise ValueError('Prior raw result exists; no automatic replay or relabelling')
        if not ref.exists():
            p.require(not cut.exists() and not resume.exists(),'Out-of-order accepted phase')
            result=invoke(binding['study'],study,configs,'reference','reference',profile,attempt/'reference',stop_at,current)
            p.require(p.checked(result)['status']=='NATIVE_TRAJECTORY_COMPLETE','Reference was interrupted; no claim of uninterrupted proof')
            p.seal(ref,result)
        if not cut.exists():
            # Cold validation+training costs remain real. If a complete first
            # reference consumed the attempt, requeue the SAME job at its saved
            # boundary before starting cut; never split the reference itself.
            if stop_at-time.time()<study['minimum_launch_seconds']:
                requeue(binding['job_id'],attempt,'COMPLETED_REFERENCE_BOUNDARY_BEFORE_NEW_CUT_PHASE',study,binding);return
            result=invoke(binding['study'],study,configs,'resume','interrupted',profile,attempt/'interrupted',stop_at,current)
            value=p.checked(result);p.require(value['status']=='ACCEPTED_NATIVE_CONTINUATION','Deliberate cut must stop at accepted update4')
            snapshot=p.checked(value['files']['STEP-4.json']);p.require(snapshot['global_step']==4,'Unplanned early stop cannot represent cut4')
            modules['native'].checkpoint_receipt(Path(configs['resume']['output'])/'training/checkpoint-4',p.n.run_identity(configs['resume']))
            p.seal(cut,result);requeue(binding['job_id'],attempt,'ACCEPTED_CUT4_ACTUAL_RESTART',study,binding);return
        interrupted=p.checked(p.read(cut))
        p.require(interrupted['actual_job_id']==binding['job_id'] and int(os.environ.get('SLURM_RESTART_COUNT','0'))>interrupted['actual_restart_count'],'Actual same physical job restart required')
        if not resume.exists():
            result=invoke(binding['study'],study,configs,'resume','resumed',profile,attempt/'resumed',stop_at,current)
            p.require(p.checked(result)['status']=='NATIVE_TRAJECTORY_COMPLETE','Resumed arm incomplete; no automatic retry')
            p.seal(resume,result)
        phases={name:p.read(path) for name,path in [('reference',ref),('interrupted',cut),('resumed',resume)]}
        from .readout import compare
        result=compare(study,phases);result.update(runtime_factor=study['runtime'],qualification_promoted=False,production_runtime_adapter_qualified=False,not_a_stochastic_variance_bound=True)
        p.seal(report/'RAW_READOUT.json',result)
        p.require(result['status']=='PASS_ACTUAL_ENGINEERING_READOUT_UNREVIEWED','Original strict numerical tests failed; preserve result')
        return result

def main():
    a=argparse.ArgumentParser();a.add_argument('--binding',required=True);v=a.parse_args();run(p.read(v.binding))
if __name__=='__main__':main()
