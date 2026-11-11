"""Babel longwall target; one child per physical attempt, exact accepted progress."""
import os,signal,socket,subprocess,time
from pathlib import Path
from datetime import datetime
from . import protocol as p,child,collect
from experiments.s1_fulltrajectory_held_cohort_v2_20261025 import protocol as held
from experiments.root_fulltrajectory_held_cohort_v2_20261025.admit import wall
from experiments.root_production_student_launcher_v4_20261025 import control as transport
from experiments.root_production_epoch_qualification_v1_20261024 import runtime

def current(binding_ref,activation_ref,runner=None,minimum_seconds=600):
 runner=runner or subprocess.check_output
 b,request,profile,a=held.activation(binding_ref,activation_ref);held.actor('babel')
 p.require(request['site']=='babel' and profile['site']=='babel' and a['target_module']==__name__
     and a['target_sources']==p.sources(),'This target is Babel-only, exact complete source closure')
 p.require(b['schema']=='s1-fulltrajectory-held-binding-v2' and b['sources']==held.sources()
     and b['command']==p.desc(held.HERE/'run.sbatch') and binding_ref==p.desc(Path(request['control_root'])/'BINDING.json'),
     'Original cohort physical/source binding changed')
 intent=p.checked(b['intent']);ack=p.checked(b['ack'])
 p.require(intent['request']==b['request'] and intent['nonce']==b['nonce'] and ack['job_id']==b['job_id']
     and ack['intent']==b['intent'],'Original intent/ACK graph differs')
 job=os.environ.get('SLURM_JOB_ID','');restart=os.environ.get('SLURM_RESTART_COUNT','0')
 p.require(job==b['job_id'] and job.isdecimal() and restart.isdecimal(),'Actual current physical job/restart required')
 raw=runner(['scontrol','show','job','-o',job],text=True,timeout=30);p.require(len(raw.strip().splitlines())==1,'One current scheduler record')
 f=transport.fields(raw);expected=dict(JobId=job,JobName=b['job_name'],JobState='RUNNING',UserId='luyangz(2708899)',Account='cx',Partition='general',QOS='normal',
     NumNodes='1',NumTasks='1',NumCPUs='8',TimeLimit=wall(request['outer_allocation']['minutes']),Command=b['command']['path'],WorkDir=str(p.ROOT),
     Comment=held.COMMENT+b['nonce'],TresPerNode='gres/gpu:L40S:1',Restarts=restart,NodeList=socket.gethostname().split('.')[0])
 p.require(all(f.get(k)==v for k,v in expected.items()) and f.get('ArrayJobId') in (None,'(null)'),'Current actual Root allocation/route differs')
 t=held.SITES['babel'];allocated=dict(v.split('=',1) for v in f.get('AllocTRES','').split(',') if '=' in v)
 p.require(allocated.get('cpu')=='8' and allocated.get('mem') in ('64G','65536M') and allocated.get('gres/gpu')=='1','Exact allocated CPU8/64GiB/L40S1')
 spool=runner(['scontrol','write','batch_script',job,'-'],timeout=30)
 p.require(p.desc(b['spool']['path'])==b['spool'] and spool==Path(b['spool']['path']).read_bytes(),'Current original cohort spool differs')
 end=int(datetime.fromisoformat(f['EndTime']).timestamp());p.require(end>time.time()+minimum_seconds,'Insufficient actual lifetime')
 p.require(not os.environ.get('SLURM_JOB_END_TIME') or int(os.environ['SLURM_JOB_END_TIME'])==end,'Truthful actual EndTime required')
 return b,request,dict(job_id=job,restart=int(restart),node=f['NodeList'],end_epoch=end,scheduler=raw),a

def execution(binding_ref,activation_ref,actual):
 b=p.checked(binding_ref);request=p.checked(b['request']);a=p.checked(activation_ref);cfg=p.checked(a['configuration'])
 root=Path(cfg['report'])/'ROOT_LONG';path=root/'requests'/('restart-'+str(actual['restart'])+'.json');ref=p.desc(path);e=p.checked(ref)
 p.require(e['schema']=='babel-s1-shared-runtime-outer-execution-v1' and e['owner']=='Root' and e['binding']==binding_ref
     and e['activation']==activation_ref and e['configuration']==a['configuration'] and e['config_identity']==p.n.run_identity(cfg)
     and e['actual_restart']==actual['restart'] and e['sources']==p.sources()
     and e['mode']=='production','Actual immutable longwall outer execution request required')
 limit=e['max_physical_attempts']
 p.require(type(limit) is int and actual['restart']<limit<=cfg['expected_horizon']['natural_updates']
     and e['mode']=='production',
     'Finite independently reviewed continuation allowance required; no inherited three-requeue ceiling')
 review=p.checked(e['review']);p.require(review['status']=='PASS_INDEPENDENT_EXACT_BABEL_S1_SHARED_OUTER_EXECUTION_V1'
     and review['execution_policy_identity']==p.key({k:v for k,v in e.items() if k not in ('review','actual_restart','accepted_history','predecessor_transition')})
     and review.get('reviewer') not in (None,'',p.AUTHOR),'Genuine independent exact outer execution review required')
 history=e['accepted_history'];p.require(len(history)==actual['restart'],'One accepted progressing segment per actual physical restart')
 cursor=0
 for h in history:
  v=p.checked(h);header=p.checked(v['accepted_archive'])
  p.require(v['configuration']==a['configuration'] and v['config_identity']==p.n.run_identity(cfg) and v['actual_job_id']==b['job_id']
      and v['adapter_sources']==p.sources() and cursor<v['global_step']<=cfg['expected_horizon']['natural_updates']
      and header['identity']==p.n.run_identity(cfg) and header['global_step']==v['global_step'],'Foreign/nonprogressing accepted history')
  cursor=v['global_step']
 if actual['restart']:
  previous=p.checked(e['predecessor_transition'])
  p.require(previous['binding']==binding_ref and previous['to_restart']==actual['restart'] and previous['continuation']==history[-1], 'Actual predecessor intent/history required')
 from .shared_runtime import verify as runtime_class
 runtime_class(e['runtime_certificate'],cfg,a['configuration'],binding_ref,e['mode'])
 permit=p.checked(e['authorization'])
 p.require(permit['runtime_certificate']==e['runtime_certificate'] and permit['physical_binding']==binding_ref
     and permit['configuration']==a['configuration'] and permit['phase']==e['mode'],
     'Outer execution/certificate/physical permit graph differs')
 return ref,e,cfg,cursor,root

def physical_check(binding_ref,activation_ref,cfg,path,output):
 profile=p.checked(cfg['site_profile']);argv=[profile['python'],'-B','-m',p.PACKAGE+'.capsule','--binding',binding_ref['path'],
     '--activation',activation_ref['path'],'--output',str(output)]
 result=subprocess.run(argv,cwd=p.ROOT,env=runtime.launch_environment(os.environ),text=True,capture_output=True,timeout=240)
 p.seal(str(output)+'.RPC.json',dict(argv=argv,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
 p.require(result.returncode==0,'Current GPU capsule guard failed; no model/requeue')
 value=p.checked(p.desc(output));p.require(value['config_identity']==p.n.run_identity(cfg) and value['physical_job_id']==p.checked(binding_ref)['job_id'],
     'Foreign physical state guard')
 return value

def child_exit(code,path,configuration):
 p.require(code==0,'Unaccepted launcher failure is not an automatic continuation')
 ref=p.desc(Path(path)/'CHILD_EXIT.json');value=p.checked(ref)
 p.require(value['schema']=='root-longwall-child-exit-v1' and value['configuration']==configuration
     and value['config_identity']==p.n.run_identity(p.checked(configuration)) and value['adapter_sources']==p.sources()
     and value['torchrun_process_exit_code']==0 and value['logical_exit_code'] in (0,75),
     'Exact source/config-bound logical child outcome required')
 receipt=p.checked(value['receipt']);expected='STUDENT_COMPLETE.json' if value['logical_exit_code']==0 else 'ACCEPTED_CONTINUATION_V1.json'
 p.require(value['receipt']==p.desc(Path(path)/expected) and receipt['configuration']==configuration
     and receipt['config_identity']==value['config_identity'] and receipt['adapter_sources']==p.sources()
     and receipt['status']==('COMPLETE_STUDENT_TRAINING_NOT_EVALUATION' if value['logical_exit_code']==0 else 'ACCEPTED_STUDENT_CONTINUATION_REQUIRES_ROOT_SAME_JOB_ACTION'),
     'Foreign/failed/missing receipt cannot be promoted by process status')
 return ref,value['logical_exit_code']

def accepted_cursor_guard(guard,cursor,continuation,configuration):
 """Current physical state must be the recorded prefix, not only its step."""
 p.require(guard['latest_step']==cursor,'Saved cursor differs from actual accepted history; no replay')
 if cursor==0:
  p.require(continuation is None and guard['capsules']==[],'Fresh execution cannot inherit any unrecorded capsule')
  return
 p.require(continuation is not None,'Nonzero accepted cursor requires recorded custody')
 value=p.checked(continuation);archive=p.checked(value['accepted_archive'])
 matches=[row for row in guard['capsules'] if row['checkpoint']==str(Path(value['accepted']['path']).parent)]
 p.require(len(matches)==1 and value['configuration']==configuration and value['global_step']==cursor
     and value['config_identity']==p.n.run_identity(p.checked(configuration)) and value['adapter_sources']==p.sources()
     and matches[0]['accepted']==value['accepted']
     and value['accepted']['bytes']==value['accepted_archive']['bytes'] and value['accepted']['sha256']==value['accepted_archive']['sha256']
     and p.checked(matches[0]['accepted'])==archive and archive['global_step']==cursor
     and archive['identity']==value['config_identity'],
     'Current accepted capsule/header differs from recorded prefix custody')

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
     current_storage_guard=p.desc(attempt/'BEFORE_TRAINING_CAPSULES.json'),accepted_prior_attempts=e['accepted_history']))
 stop=actual['end_epoch']-600
 p.require(stop>time.time()+120,'No useful actual model window')
 profile=p.checked(cfg['site_profile']);argv=[profile['python'],'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=1','-m',p.PACKAGE+'.child',
     '--configuration',a['configuration']['path'],'--authorization',e['authorization']['path'],'--launch-binding',launch['path'],'--attempt',str(childpath),'--stop-at-epoch',str(stop)]
 env=runtime.launch_environment(os.environ);env.update(NATIVE_RUN_IDENTITY=p.n.run_identity(cfg),NATIVE_ATTEMPT=str(childpath),NATIVE_STOP_PATH=str(childpath/'STOP'),
     HF_HOME=profile['cache_root'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SLURM_JOB_END_TIME=str(actual['end_epoch']),OMP_NUM_THREADS='8')
 p.seal(attempt/'ATTEMPT_START.json',dict(actual=actual,execution=eref,accepted_cursor=cursor,argv=argv,source=p.sources(),qualification_mode=False,shared_canary_mode=e['mode']=='shared_canary',
     full_horizon=cfg['expected_horizon'],scientific_max_steps=-1,started_epoch=started,actual_stop_at_epoch=stop,
     requested_segment_seconds=None,ordinary_accepted_checkpoint_control=e['mode']=='shared_canary'))
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
