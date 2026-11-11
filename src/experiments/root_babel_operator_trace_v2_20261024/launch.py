"""Exact held-job config review; prior measured reservation and fresh GPU check."""
import argparse,copy,fcntl,os,pwd,socket,subprocess,time
from pathlib import Path
from . import protocol as p,admit as a,worker as w

def validate_launch(value,reviewed=True,config_reviewed=True):
    p.require(value.get('schema')=='babel-native7-exact-config-launch-v3' and value['owner']=='Root' and value['driver_sources']==p.sources(),'Exact new launch source required')
    design=p.validate_design(p.checked(value['design']));held=p.checked(value['held_binding']);intent=p.checked(held['intent']);cfg=p.checked(value['native_configuration'])
    p.require(value['job_id']==held['job_id'] and value['job_id'].isdecimal() and held['cell']=='T1' and held['owner']=='Root' and held['design']==value['design'] and held['driver_sources']==p.sources() and held['spool_reconciled'] is True,'Exact held job/design binding required')
    shell=dict(anchor_study=design['anchor_study'],paid_anchor_readout=design['paid_anchor_readout'],native_target=design['native_target'],report=design['report'])
    old,original,modules=p.anchor(shell);p.invariant_config(cfg,original,'T1')
    p.require(p.design_contract(shell,{'T1':cfg})==p.bare(design),'Actual config changed full independently reviewed design')
    claim=p.checked(cfg['providers']['ownership']);storage=p.checked(cfg['storage_reservation'])
    p.require(claim['actual_held_job_id']==value['job_id'] and claim['cell_id']==cfg['id'] and claim['plan']==cfg['plan'] and claim['pool']==cfg['providers']['pool'] and claim['site_profile']==cfg['site_profile'] and claim['replicate_site']=='babel' and claim['owner']=='Root' and claim['status']=='ACTIVE_EXCLUSIVE_NATIVE_PLAN_CELL_CLAIM' and claim['held_binding']==value['held_binding'],'Exact actual held-job native ownership required')
    prior=p.checked(p.PRIOR_STORAGE)
    p.require(prior['status']=='ACTUAL_OWNED_TRACE_GPU_OUTPUT_FILESYSTEM_MEASUREMENT' and prior['actual_job_id']=='10913252' and prior['output_root']==p.OUTPUT_BASE and prior['available_bytes']>=8*p.GIB and prior['available_inodes']>=2000,'Authentic prior measurement, not a fabricated current-job reading')
    p.require(storage['status']=='PASS_EXCLUSIVE_MEASURED_STORAGE_RESERVATION' and storage['cell_id']==cfg['id'] and storage['output']==cfg['output'] and storage['checkpoint_transaction_bytes']==8*p.GIB and storage['retained_accepted_states']==1 and storage['allow_new_intermediate_retirement'] is False and storage['prior_actual_measurement']==p.PRIOR_STORAGE and storage['measurement_timing']=='PRIOR_10913252_SNAPSHOT_NOT_CURRENT_ALLOCATION' and storage['fresh_recheck_required_before_neural'] is True,'Reviewed Root reservation must remain explicit about prior measurement timing')
    p.require(storage['Root_collective_authorization']==dict(owner='Root',study_id=p.ID,total_reserved_bytes=8*p.GIB,maximum_new_final_capsules=1,collective_consumers_reconciled=True),'Exact Root reassigned8GiB claim required')
    if config_reviewed:
        review=p.checked(cfg['review']);p.require(review['status']=='PASS_INDEPENDENT_EXACT_SOURCE_AND_CONFIG' and review['config_identity']==p.n.run_identity(cfg) and review['sources']==cfg['sources'] and review['reviewer'] not in ('/root/native_campaign_build','native_campaign_build'),'Genuine actual exact config review required; no PLAN/READY fallback')
    else:p.require(cfg['review'] is None,'Only unreviewed config draft may await independent config review')
    if reviewed:
        review=p.checked(value['review']);p.require(review['status']=='PASS_INDEPENDENT_BABEL_EXACT_CONFIG_TRACE_LAUNCH' and review['launch_identity']==p.key(p.bare(value)) and review['driver_sources']==p.sources() and review['reviewer'] not in ('/root/native_campaign_build','native_campaign_build'),'Independent actual job-bound launch review required')
    return design,cfg,held,intent

def freeze(request_ref):
    request=p.checked(request_ref);p.require(request['status']=='REGISTERED_BEFORE_EXECUTION' and request['owner']=='Root','Root finite registration required')
    design=p.validate_design(p.checked(request['design']));held=p.checked(request['held_binding']);cfg=copy.deepcopy(design['configurations']['T1']);job=held['job_id'];root=p.literal(Path(design['report']).parent)
    p.require(held['design']==request['design'] and held['driver_sources']==p.sources() and held['spool_reconciled'] is True,'Exact held source/design graph required')
    auth=request['Root_collective_authorization'];p.require(auth==dict(owner='Root',study_id=p.ID,total_reserved_bytes=8*p.GIB,maximum_new_final_capsules=1,collective_consumers_reconciled=True),'Root8GiB reassignment required')
    prior=p.checked(p.PRIOR_STORAGE)
    p.require(prior['actual_job_id']=='10913252' and prior['available_bytes']>=8*p.GIB,'Pinned prior factual measurement required')
    accounting=subprocess.check_output(['sacct','-X','-nP','-j','10913252','-o','JobIDRaw,State,ExitCode,ElapsedRaw,Restarts'],text=True,timeout=30)
    rows=[line.split('|') for line in accounting.splitlines() if line.split('|')[0]=='10913252']
    p.require(len(rows)==1 and rows[0][1:5]==['FAILED','1:0','577','0'],'Old failed trace must be terminal before reservation reassignment')
    terminal=p.seal(root/'PRIOR_FAILED_ACCOUNTING.json',dict(raw=accounting,prior_measurement=p.PRIOR_STORAGE,actual_old_job='10913252'))
    cfg['providers']['ownership']=p.seal(root/'T1/OWNERSHIP.json',dict(status='ACTIVE_EXCLUSIVE_NATIVE_PLAN_CELL_CLAIM',owner='Root',cell_id=cfg['id'],plan=cfg['plan'],pool=cfg['providers']['pool'],site_profile=cfg['site_profile'],replicate_site='babel',actual_held_job_id=job,held_binding=request['held_binding'],purpose='INTENTIONAL_L40S_OPERATOR_TRACE_EXACT_HELD_JOB'))
    cfg['storage_reservation']=p.seal(root/'T1/STORAGE.json',dict(status='PASS_EXCLUSIVE_MEASURED_STORAGE_RESERVATION',owner='Root',cell_id=cfg['id'],output=cfg['output'],retained_accepted_states=1,allow_new_intermediate_retirement=False,checkpoint_transaction_bytes=8*p.GIB,prior_actual_measurement=p.PRIOR_STORAGE,measurement_timing='PRIOR_10913252_SNAPSHOT_NOT_CURRENT_ALLOCATION',fresh_recheck_required_before_neural=True,Root_collective_authorization=auth,prior_failed_accounting=terminal))
    cfg['review']=None;config_ref=p.seal(root/'T1/CONFIG_DRAFT.json',cfg)
    value=dict(schema='babel-native7-exact-config-launch-v3',owner='Root',driver_sources=p.sources(),design=request['design'],held_binding=request['held_binding'],job_id=job,native_configuration=config_ref,registered_request=request_ref)
    validate_launch(value,reviewed=False,config_reviewed=False)
    return p.seal(root/'EXACT_CONFIG_REVIEW_REQUEST.json',dict(status='PENDING_GENUINE_EXACT_HELD_JOB_CONFIG_REVIEW',configuration=config_ref,config_identity=p.n.run_identity(cfg),launch_draft=p.seal(root/'FROZEN_LAUNCH_DRAFT.json',value)))

def bind(request_ref):
    request=p.checked(request_ref);p.require(request['status']=='REGISTERED_BEFORE_EXECUTION' and request['owner']=='Root','Root review bind registration')
    value=p.checked(request['launch_draft']);cfg=p.checked(value['native_configuration']);review=p.checked(request['configuration_review'])
    p.require(review['status']=='PASS_INDEPENDENT_EXACT_SOURCE_AND_CONFIG' and review['config_identity']==p.n.run_identity(cfg) and review['sources']==cfg['sources'] and review['reviewer'] not in ('/root/native_campaign_build','native_campaign_build'),'Genuine exact-source/config review only')
    cfg['review']=request['configuration_review'];value['native_configuration']=p.seal(Path(value['native_configuration']['path']).with_name('CONFIG.json'),cfg)
    validate_launch(value,reviewed=False)
    return p.seal(Path(request['launch_draft']['path']).with_name('FROZEN_LAUNCH_CONFIG_REVIEWED_DRAFT.json'),value)
def release(launch_ref,source_review):
    a.source_review(source_review);value=p.checked(launch_ref);design,cfg,held,intent=validate_launch(value)
    profile=p.checked(cfg['site_profile']);a.actor(profile)
    with a.lock_path().open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        dest=Path(value['held_binding']['path']).parent;job=value['job_id']
        p.require(not (dest/'RELEASE_INTENT.json').exists(),'Prior release intent requires reconciliation')
        raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=30)
        a.held_fields(raw,job,'T1',intent,profile);a.spool(job,dest/'PRE_RELEASE_SPOOL.sh')
        binding=p.seal(dest/'RUN_BINDING.json',dict(schema='babel-native7-operator-trace-run-binding-v2',owner='Root',cell='T1',job_id=job,held_binding=value['held_binding'],study=launch_ref))
        before=p.seal(dest/'RELEASE_INTENT.json',dict(owner='Root',job_id=job,binding=binding,actual_held_scheduler=raw,UTC=p.now()))
        try:reply=subprocess.run(['scontrol','release',job],capture_output=True,text=True,timeout=30)
        except subprocess.TimeoutExpired:
            p.seal(dest/'RELEASE_AMBIGUOUS.json',dict(intent=before,status='NO_AUTOMATIC_RELEASE_RETRY'));raise
        result=p.seal(dest/'RELEASE_RESPONSE.json',dict(intent=before,returncode=reply.returncode,stdout=reply.stdout,stderr=reply.stderr))
        p.require(reply.returncode==0,'Release acknowledgement failed; reconcile, no retry')
        return result

def current_job(value):
    job=os.environ.get('SLURM_JOB_ID','')
    p.require(job==value['job_id'] and job.isdecimal() and os.environ.get('SLURM_RESTART_COUNT','0')=='0'
        and pwd.getpwuid(os.getuid()).pw_name=='luyangz','Actual current fresh Root job required before DATA validation')
    raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=30);fields=a.fields(raw)
    held=p.checked(value['held_binding']);intent=p.checked(held['intent'])
    expected=dict(JobId=job,JobName='dsmm-babel-optrace-T1',JobState='RUNNING',UserId='luyangz('+str(os.getuid())+')',Account='cx',Partition='general',QOS='normal',NodeList=socket.gethostname().split('.')[0],NumNodes='1',NumTasks='1',NumCPUs='8',TimeLimit='00:55:00',Restarts='0',Command=str(p.HERE/'run.sbatch'),WorkDir=str(p.ROOT),Comment='dsmm-babel-optrace:'+intent['nonce'],TresPerNode='gres/gpu:L40S:1')
    allocated=dict(x.split('=',1) for x in fields.get('AllocTRES','').split(',') if '=' in x)
    p.require(all(fields.get(k)==v for k,v in expected.items()) and allocated.get('cpu')=='8' and allocated.get('mem') in ('64G','65536M') and allocated.get('gres/gpu')=='1','Current conditional job ownership/allocation differs')
    return raw

def validate_runtime(study):
    p.require(study['id']==p.ID and study['owner']=='Root' and study['purpose']=='BABEL_OPERATOR_DISPATCH_TRACE_ONLY' and study['cells']==p.CELLS and study['total_scheduler_steps']==8 and study['numeric_tolerance']==p.TOLERANCE and study['runtime_budget']==p.RUNTIME_BUDGET,'Exact runtime study metadata')
    value=p.checked(study['conditional_launch']);design,cfg,held,intent=validate_launch(value);current_job(value)
    p.require(study['driver_sources']==p.sources() and study['jobs']=={'T1':value['job_id']} and study['design']==value['design'] and study['native_configurations']=={'T1':value['native_configuration']},'Runtime cannot change reviewed actual config/job')
    snapshot=p.checked(study['fresh_storage_recheck'])
    p.require(snapshot['status']=='ACTUAL_CURRENT_TRACE_GPU_STORAGE_RECHECK' and snapshot['actual_job_id']==value['job_id'] and snapshot['launch']==study['conditional_launch'] and snapshot['available_bytes']>=8*p.GIB and snapshot['available_inodes']>=2000,'Actual current GPU recheck required in addition to prior measured reservation')
    old,original,modules=p.anchor(study);p.invariant_config(cfg,original,'T1')
    # Actual unchanged Native7 validator takes the independently reviewed exact
    # configuration branch. There is deliberately no readiness key or fakeREADY.
    modules['config'].validate(cfg)
    return {'T1':cfg},modules

def worker(binding):
    launch_ref=binding['study'];value=p.checked(launch_ref);design,cfg,held,intent=validate_launch(value)
    profile,raw,end=w.actual(binding,dict(jobs={'T1':value['job_id']},design=value['design']),cfg)
    from experiments.root_native_qualification_v10_20261024.preflight import runtime_preflight
    runtime=runtime_preflight(p.checked(design['anchor_study']),site='babel');report=p.literal(cfg['report']);report.mkdir(parents=True,exist_ok=True)
    with (report/'EXECUTION.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p.require(not (report/'ATTEMPT_START.json').exists() and not p.literal(cfg['output']).exists(),'One new fresh attempt; preserve old failed files')
        attempt=report/'attempt';attempt.mkdir(exist_ok=False)
        p.seal(report/'ATTEMPT_START.json',dict(binding=binding,actual_slurm=raw,current_runtime=runtime,UTC=p.now()))
        fs=os.statvfs(p.literal(profile['output_root']));available=fs.f_bavail*fs.f_frsize
        p.require(available>=8*p.GIB and fs.f_favail>=2000,'Actual current output capacity below reservation; no NN')
        fresh=p.seal(attempt/'ACTUAL_STORAGE_RECHECK.json',dict(status='ACTUAL_CURRENT_TRACE_GPU_STORAGE_RECHECK',actual_job_id=value['job_id'],actual_step=os.environ.get('SLURM_STEP_ID','batch'),launch=launch_ref,available_bytes=available,available_inodes=fs.f_favail,output_root=profile['output_root'],scheduler=raw,UTC=p.now()))
        study=dict(schema='babel-native7-operator-trace-runtime-study-v3',id=p.ID,owner='Root',purpose='BABEL_OPERATOR_DISPATCH_TRACE_ONLY',driver_sources=p.sources(),conditional_launch=launch_ref,design=value['design'],jobs={'T1':value['job_id']},native_configurations={'T1':value['native_configuration']},fresh_storage_recheck=fresh,native_target=design['native_target'],anchor_study=design['anchor_study'],paid_anchor_readout=design['paid_anchor_readout'],runtime_budget=p.RUNTIME_BUDGET,cells=p.CELLS,total_scheduler_steps=8,numeric_tolerance=p.TOLERANCE,report=design['report'])
        study_ref=p.seal(attempt/'RUNTIME_STUDY.json',study);pair,modules=validate_runtime(study)
        stop_at=end-cfg['allocation']['reserve_seconds'];p.require(stop_at-time.time()>=p.RUNTIME_BUDGET['minimum_neural_seconds'],'Cold validation consumed real55min neural launch window')
        p.seal(attempt/'PAID_BABEL_READOUT_REUSE.json',dict(source=study['paid_anchor_readout'],status='REUSED_ACTUAL_FAILED_BABEL_V10_METADATA_ONLY',new_tensor_readout_or_training=False,qualification_promoted=False))
        import signal
        stop=attempt/'STOP_REQUESTED'
        def stopping(*_):stop.touch()
        signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGUSR1,stopping)
        env=dict(os.environ,NATIVE_STOP_PATH=str(stop),NATIVE_ATTEMPT=str(attempt),NATIVE_RUN_IDENTITY=p.n.run_identity(cfg),HF_HOME=profile['cache_root'],HF_HUB_CACHE=str(Path(profile['cache_root'])/'hub'),OMP_NUM_THREADS='8')
        argv=w.child_argv(profile,study_ref,'T1',value['native_configuration'],attempt,stop_at)
        p.seal(attempt/'INVOCATION.json',dict(argv=argv,binding=binding,runtime_study=study_ref,actual_end=end))
        with (attempt/'torchrun.log').open('xb',buffering=0) as log:rc=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT).wait()
        child=attempt/'CHILD_RESULT.json';p.seal(report/'RESULT.json',dict(status='ACTUAL_BABEL_TRACE_RETURNED_NOT_PROMOTED',returncode=rc,actual_job_id=value['job_id'],child=p.desc(child) if child.exists() else None,qualification_promoted=False))
        p.require(rc==0 and child.exists(),'Actual failed trace preserved')
        result=p.checked(p.desc(child));observed=p.checked(result['observation']);step=p.checked(result['files']['STEP-1.json']);p.expected_endpoint('T1',result['status'],step['global_step'],observed['actual_record_exposure'])
        complete=p.seal(report/'CELL_COMPLETE.json',dict(status='COMPLETED_FINITE_BABEL_TRACE_NOT_QUALIFICATION',cell='T1',study=study_ref,child=p.desc(child),qualification_promoted=False))
        if end-time.time()>=p.RUNTIME_BUDGET['final_readout_minimum_seconds']:
            from .readout import collect
            collect(study_ref)
        return complete

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('freeze','bind','release','worker'));parser.add_argument('--request');parser.add_argument('--launch');parser.add_argument('--source-review');parser.add_argument('--binding');args=parser.parse_args()
    if args.action in ('freeze','bind'):
        if not args.request:parser.error('--request required')
        result=globals()[args.action](p.desc(args.request))
    elif args.action=='release':
        if not args.launch or not args.source_review:parser.error('--launch and --source-review required')
        result=release(p.desc(args.launch),p.desc(args.source_review))
    else:
        if not args.binding:parser.error('--binding required')
        result=worker(p.read(args.binding))
    print(p.n.canonical(result).decode())
if __name__=='__main__':main()
