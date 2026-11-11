"""Explicit physical outer S1 driver; unchanged logical297 child/state semantics."""
import os,signal,subprocess,time
from pathlib import Path
from experiments.root_s1_shared_student_v1_20261025 import protocol as p,driver as base_driver,collect
from experiments.s1_fulltrajectory_held_cohort_v2_20261025 import protocol as held
from experiments.root_production_student_launcher_v4_20261025 import control as transport
from experiments.root_production_epoch_qualification_v1_20261024 import runtime
from . import route
current=route.current
execution=route.execution
accepted_cursor_guard=base_driver.accepted_cursor_guard


def physical_check(binding_ref,activation_ref,cfg,path,output):
    profile=p.checked(cfg['site_profile'])
    argv=[profile['python'],'-B','-m',route.PACKAGE+'.route','--binding',binding_ref['path'],'--activation',activation_ref['path'],'--mode','capsule','--output',str(output)]
    result=subprocess.run(argv,cwd=p.ROOT,env=runtime.launch_environment(os.environ),text=True,capture_output=True,timeout=240)
    p.seal(str(output)+'.RPC.json',dict(argv=argv,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,physical_executed_sources=route.sources()))
    p.require(result.returncode==0,'Explicit physical capsule process failed; no model or requeue')
    guard=p.checked(p.desc(output));proof=p.checked(p.desc(str(output)+'.PHYSICAL.json'))
    b,r,actual,a=current(binding_ref,activation_ref,minimum_seconds=30)
    p.require(proof['guard']==p.desc(output) and proof['physical_extension']==actual['physical_extension']
        and proof['executed_sources']==route.sources() and guard['config_identity']==p.n.run_identity(cfg)
        and guard['physical_job_id']==b['job_id'] and proof['checkpoint_export']['export_identity']==route.EXPORT
        and proof['checkpoint_export']['actual_job_id']==actual['job_id'] and proof['checkpoint_export']['actual_restart']==actual['restart']
        and proof['checkpoint_export']['observed_device']==proof['checkpoint_export']['after_segment_observation']['observed_device'],
        'Exact physical source/capsule custody')
    return guard


def child_exit(code,path,configuration):
    ref,logical=base_driver.child_exit(code,path,configuration)
    proof=p.checked(p.desc(Path(path)/'PHYSICAL_EXECUTION_PROOF.json'))
    extension=proof['physical_extension'];b,r,actual,a=current(extension['binding'],extension['activation'],minimum_seconds=30)
    eref,e,cfg,cursor,root=execution(extension['binding'],extension['activation'],actual)
    outcome=p.checked(ref);receipt=p.checked(outcome['receipt'])
    p.require(proof['configuration']==configuration and proof['config_identity']==p.n.run_identity(cfg)
        and proof['logical_exit_code']==logical and proof['executed_sources']==route.sources()
        and proof['logical_sources']==p.sources() and proof['physical_extension']==actual['physical_extension']
        and extension['execution']==eref and proof['child_receipt']==outcome['receipt']
        and proof['operators']==receipt['operators']
        and p.checked(proof['operators'])['observational_execution_extension']==extension,
        'Child success alone cannot bypass mandatory actual physical execution proof')
    route.verify_physical_receipt(outcome['receipt'],extension['binding'],extension['activation'],actual['restart'])
    return ref,logical


def run_conditional_student(binding_ref,activation_ref):
 started=time.time();b,request,actual,a=current(binding_ref,activation_ref);eref,e,cfg,cursor,root=execution(binding_ref,activation_ref,actual)
 root.mkdir(parents=True,exist_ok=True);name=b['job_id']+'-r'+str(actual['restart'])+'-'+b['nonce'][:8]
 attempt=root/'attempts'/name;attempt.mkdir(parents=True,exist_ok=False)
 guard=physical_check(binding_ref,activation_ref,cfg,None,attempt/'BEFORE_TRAINING_CAPSULES.json')
 accepted_cursor_guard(guard,cursor,e['accepted_history'][-1] if e['accepted_history'] else None,a['configuration'])
 childpath=Path(cfg['report'])/'attempts'/name/'child'
 launch=p.seal(attempt/'CHILD_BINDING.json',dict(schema='root-longwall-student-child-binding-v1',owner='Root',configuration=a['configuration'],
     authorization=e['authorization'],adapter_sources=p.sources(),physical_binding=binding_ref,activation=activation_ref,
     physical_job_id=b['job_id'],actual_restart_count=actual['restart'],nonce=b['nonce'],command=b['command'],child_attempt=str(childpath),
     current_storage_guard=p.desc(attempt/'BEFORE_TRAINING_CAPSULES.json'),accepted_prior_attempts=e['accepted_history'],
     physical_execution=eref,physical_sources=route.sources(),physical_extension=actual['physical_extension'],
     physical_source_packet=actual['physical_extension']['source_packet'],physical_capsule_guard=p.desc(str(attempt/'BEFORE_TRAINING_CAPSULES.json')+'.PHYSICAL.json')))
 stop=actual['end_epoch']-600
 p.require(stop>time.time()+120,'No useful actual model window')
 profile=p.checked(cfg['site_profile']);argv=[profile['python'],'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=1','-m',route.PACKAGE+'.route',
     '--binding',binding_ref['path'],'--activation',activation_ref['path'],'--mode','child','--launch',launch['path'],'--attempt',str(childpath),'--stop-at-epoch',str(stop)]
 env=runtime.launch_environment(os.environ);env.update(NATIVE_RUN_IDENTITY=p.n.run_identity(cfg),NATIVE_ATTEMPT=str(childpath),NATIVE_STOP_PATH=str(childpath/'STOP'),
     HF_HOME=profile['cache_root'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SLURM_JOB_END_TIME=str(actual['end_epoch']),OMP_NUM_THREADS='8')
 p.seal(attempt/'ATTEMPT_START.json',dict(actual=actual,execution=eref,accepted_cursor=cursor,argv=argv,source=p.sources(),qualification_mode=False,shared_canary_mode=e['mode']=='shared_canary',
     full_horizon=cfg['expected_horizon'],scientific_max_steps=-1,started_epoch=started,actual_stop_at_epoch=stop,
     requested_segment_seconds=None,ordinary_accepted_checkpoint_control=e['mode']=='shared_canary',
     physical_extension=actual['physical_extension'],physical_executed_sources=route.sources()))
 requested=[False]
 def request_stop(*_):requested[0]=True
 handlers={sig:signal.signal(sig,request_stop) for sig in (signal.SIGUSR1,signal.SIGTERM)}
 try:
  with (attempt/'torchrun.log').open('xb') as log:
   proc=subprocess.Popen(argv,cwd=p.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   while proc.poll() is None:
    if (requested[0] or time.time()>=stop) and childpath.is_dir():(childpath/'STOP').touch(exist_ok=True)
    if time.time()>=actual['end_epoch']-60:
     proc.terminate()
     try:proc.wait(timeout=15)
     except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=15)
     raise ValueError('Unaccepted actual deadline failure; preserve all state, no automatic retry')
    time.sleep(1)
   code=proc.returncode
 finally:
  for sig,handler in handlers.items():signal.signal(sig,handler)
 # Preserve even an unaccepted/raw process failure before interpreting receipts.
 p.seal(attempt/'PROCESS_RETURN.json',dict(returncode=code,actual=actual,elapsed_seconds=time.time()-started,all_in_cost_requires_sacct_D=True))
 outcome,logical=child_exit(code,childpath,a['configuration'])
 p.seal(attempt/'CHILD_RETURN.json',dict(returncode=code,logical_exit_code=logical,child_exit=outcome,actual=actual,elapsed_seconds=time.time()-started,all_in_cost_requires_sacct_D=True))
 if logical==0:return collect.collect(a['configuration'],e['accepted_history']+[p.desc(childpath/'STUDENT_COMPLETE.json')],root/'COMPLETE.json')
 p.require(logical==75,'Unaccepted child failure is not an automatic continuation')
 cont=p.desc(childpath/'ACCEPTED_CONTINUATION_V1.json');value=p.checked(cont)
 p.require(value['configuration']==a['configuration'] and value['actual_job_id']==b['job_id'] and cursor<value['global_step']<=cfg['expected_horizon']['natural_updates'],
     'Only verified positive accepted progress may continue')
 after=physical_check(binding_ref,activation_ref,cfg,None,attempt/'AFTER_TRAINING_CAPSULES.json')
 accepted_cursor_guard(after,value['global_step'],cont,a['configuration'])
 if actual['restart']+1>=e['max_physical_attempts'] and e['mode']=='production':
  return p.seal(attempt/'CONTINUATION_ALLOWANCE_EXHAUSTED.json',dict(
      status='ACCEPTED_PROGRESS_PRESERVED_CONTINUATION_ALLOWANCE_EXHAUSTED_NOT_COMPLETE',
      execution=eref,continuation=cont,physical_guard=p.desc(attempt/'AFTER_TRAINING_CAPSULES.json'),
      actual=actual,max_physical_attempts=e['max_physical_attempts'],scheduler_action_taken=False,
      requires_new_independent_outer_policy_review=True,scientific_performance_measured=False))
 action='requeue';dest=root/'transitions'/('r'+str(actual['restart']))
 with transport.lock_path().open('r+') as lock:
  import fcntl
  fcntl.flock(lock,fcntl.LOCK_EX);held.actor('babel');census=transport.census(profile,0)
  current(binding_ref,activation_ref,minimum_seconds=120)
  dest.mkdir(parents=True,exist_ok=False)
  intent=p.seal(dest/'INTENT.json',dict(owner='Root',action=action,binding=binding_ref,execution=eref,continuation=cont,
      from_restart=actual['restart'],to_restart=actual['restart']+1,actual_census=census,physical_guard=p.desc(attempt/'AFTER_TRAINING_CAPSULES.json'),
      explicit_qualification_hold=e['mode']=='shared_canary',no_automatic_retry=True))
  if action=='requeue':
   nextvalue=dict(e,actual_restart=actual['restart']+1,accepted_history=e['accepted_history']+[cont])
   nextvalue['predecessor_transition']=intent
   p.seal(root/'requests'/('restart-'+str(actual['restart']+1)+'.json'),nextvalue)
  _,response=transport.rpc(['scontrol',action,b['job_id']],dest,intent)
  p.seal(dest/'ACK.json',dict(intent=intent,response=response,may_be_absent_after_parent_termination=True))
 while time.time()<actual['end_epoch']-15:
  raw=subprocess.check_output(['scontrol','show','job','-o',b['job_id']],text=True,timeout=30);fields=transport.fields(raw)
  if fields.get('Restarts')==str(actual['restart']+1) and fields.get('JobState') in ('PENDING','RUNNING'):
   return p.seal(attempt/'OBSERVED_TRANSITION.json',dict(intent=intent,actual_scheduler=raw,actual_qualified=False))
  time.sleep(5)
 raise ValueError('Transition unresolved; preserveintent, externalRoot reconciliation, no APIretry')
