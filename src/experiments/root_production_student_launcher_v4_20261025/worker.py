"""Root-owned same-scalar supervision. This file is NOT a claim of atomic RPC."""
import argparse,os,signal,socket,subprocess,time
from pathlib import Path
from datetime import datetime
from . import protocol as p,control
from experiments.root_production_student_adapter_v2_20261025 import child,collect
from experiments.root_production_epoch_qualification_v1_20261024 import runtime

def allocation(binding,config,run=None,minimum_seconds=600):
 run=run or subprocess.check_output
 profile=p.checked(config['site_profile']);p.actor(profile);job=os.environ.get('SLURM_JOB_ID','')
 p.require(job==binding['job_id'] and job.isdecimal(),'Actual physical Root job differs')
 raw=run(['scontrol','show','job','-o',job],text=True,timeout=30);f=control.fields(raw)
 expected=dict(JobId=job,JobName=binding['job_name'],JobState='RUNNING',UserId='luyangz('+str(os.getuid())+')',Account='cx',Partition='general',QOS='normal',
     NumNodes='1',NumTasks='1',NumCPUs='8',TimeLimit='00:55:00',Command=binding['command']['path'],WorkDir=str(p.ROOT),
     Comment='dsmm-production-student:'+binding['nonce'],TresPerNode='gres/gpu:L40S:1',Restarts=os.environ.get('SLURM_RESTART_COUNT','0'),NodeList=socket.gethostname().split('.')[0])
 p.require(all(f.get(k)==v for k,v in expected.items()),'Current UID/nonce/spool/resources/node/restart differ')
 tres=dict(v.split('=',1) for v in f.get('AllocTRES','').split(',') if '=' in v)
 p.require(tres.get('cpu')=='8' and tres.get('mem') in ('64G','65536M') and tres.get('gres/gpu')=='1','Actual CPU/GPU allocation differs')
 end=int(datetime.fromisoformat(f['EndTime']).timestamp());p.require(end>time.time()+minimum_seconds,'Insufficient actual worker lifetime')
 p.require(not os.environ.get('SLURM_JOB_END_TIME') or int(os.environ['SLURM_JOB_END_TIME'])==end,'EndTime mismatch')
 spool=run(['scontrol','write','batch_script',job,'-'],timeout=30);p.require(spool==Path(binding['spool']['path']).read_bytes(),'Current running spool changed')
 return dict(job_id=job,restart=int(f['Restarts']),end_epoch=end,node=f['NodeList'],scheduler=raw)

def launch_binding(binding,config_ref,execution,child_path):
 partials=[ref for ref in execution['accepted_history'] if p.checked(ref).get('pilot_progress_pending',False)]
 p.require(len(partials)<=3,'Only actual accepted1/2/3 belong in pilot-prefix history')
 return dict(schema='production-student-launch-binding-v1',owner='Root',configuration=config_ref,authorization=execution['authorization'],
     physical_job_id=binding['job_id'],adapter_sources=p.a.sources(),nonce=binding['nonce'],command=binding['command'],
     actual_restart_count=execution['actual_restart_count'],child_attempt=str(child_path),accepted_prior_attempts=partials)

def child_argv(profile,config_ref,permit_ref,launch_ref,child_path,stop):
 return [profile['python'],'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=1','-m',p.a.PACKAGE+'.child',
     '--configuration',config_ref['path'],'--authorization',permit_ref['path'],'--launch-binding',launch_ref['path'],
     '--attempt',str(child_path),'--stop-at-epoch',str(stop)]

def validate_history(config,execution,binding,checkpoint_check):
 """Full GPU revalidation before child/model, even if held metadata was valid."""
 history=execution['accepted_history'];cursor=0
 for ref in history:
  v=p.checked(ref);archive=p.checked(v['accepted_archive'])
  p.require(v['configuration']==binding['configuration'] and v['config_identity']==p.a.n.run_identity(config)
      and v['adapter_sources']==p.a.sources() and v['actual_job_id']==binding['job_id']
      and cursor<v['global_step']<=config['expected_horizon']['natural_updates']
      and v['accepted_archive']['bytes']==v['accepted']['bytes'] and v['accepted_archive']['sha256']==v['accepted']['sha256']
      and archive['status']=='ACCEPTED_FULL_NATIVE_STATE' and archive['identity']==v['config_identity']
      and archive['global_step']==v['global_step'],'Foreign/nonincreasing accepted history')
  cursor=v['global_step']
 p.require(len(history)==execution['actual_restart_count'],'Each restart must advance one accepted segment; unknown restarts need reconciliation')
 if history:
  last=p.checked(history[-1]);child.validate_continuation(last,config,Path(last['segment']['path']).parent,binding['job_id'],checkpoint_check)
 else:
  p.require(not list((Path(config['output'])/'training').glob('checkpoint-*')),'Fresh request cannot hide existing checkpoint state')
 return cursor

def physical_check(binding_ref,path,identity,output):
 """Separate finite process releases its guard-only CUDA context before NN."""
 b=p.checked(binding_ref);cfg=p.checked(b['configuration']);profile=p.checked(cfg['site_profile'])
 argv=[profile['python'],'-B','-m',p.PACKAGE+'.capsule','--binding',binding_ref['path'],'--checkpoint',str(path),'--output',str(output)]
 r=subprocess.run(argv,env=runtime.launch_environment(os.environ),cwd=p.ROOT,text=True,capture_output=True,timeout=180)
 p.seal(str(output)+'.RPC.json',dict(argv=argv,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
 p.require(r.returncode==0,'Physical capsule guard failed; no training/requeue')
 value=p.checked(p.desc(output));p.require(value['config_identity']==identity and value['physical_job_id']==b['job_id'],'Physical guard identity differs')
 return value['accepted']

def transition(binding_ref,execution_ref,continuation_ref,root,run=None):
 """Intent survives parent termination. No promise that an ACK will survive."""
 binding=p.checked(binding_ref);execution=p.checked(execution_ref);request=p.checked(binding['request']);cfg,profile,_,_=p.metadata(request);p.actor(profile)
 p.validate_binding(binding,binding['request'],request)
 v=p.checked(continuation_ref)
 # Physical accepted state is checked in this actual ownedGPU, before mutation.
 actual=allocation(binding,cfg,minimum_seconds=240)
 p.require(actual['restart']==execution['actual_restart_count'],'Transition restart changed')
 guardpath=root/'physical-checks'/('before-transition-r'+str(actual['restart'])+'.json')
 child.validate_continuation(v,cfg,Path(v['segment']['path']).parent,binding['job_id'],lambda path,identity:physical_check(binding_ref,path,identity,guardpath))
 p.require(v['actual_restart_count']==actual['restart'] and v['global_step']<=cfg['expected_horizon']['natural_updates'],'Foreign/out-of-budget continuation')
 previous=p.checked(execution['accepted_history'][-1])['global_step'] if execution['accepted_history'] else 0
 p.require(v['global_step']>previous,'No missing-only progress; do not requeue setup/failure')
 action='requeuehold' if v['gate_hold_required'] else 'requeue'
 dest=p.literal_home(root/'transitions'/('r'+str(actual['restart'])))
 with control.lock_path().open('r+') as lock:
  import fcntl
  fcntl.flock(lock,fcntl.LOCK_EX);p.actor(profile);c=control.census(profile,0)
  p.require(time.time()+65<actual['end_epoch'],'No actual bounded RPC window after guard/lock; preserve accepted state for external reconciliation')
  dest.mkdir(parents=True,exist_ok=False)
  intent=p.seal(dest/'INTENT.json',dict(owner='Root',action=action,physical_binding=binding_ref,execution=execution_ref,
      continuation=continuation_ref,from_restart=actual['restart'],to_restart=actual['restart']+1,
      actual_before=actual,actual_census=c,config_identity=p.a.n.run_identity(cfg),physical_capsule_revalidation=p.desc(guardpath),
      API_advertised_only_atomicity_NOT_qualified=True,UTC=p.now()))
  if action=='requeue':
   history=execution['accepted_history']+[continuation_ref]
   nextvalue=control.execution_request(binding_ref,execution['authorization'],actual['restart']+1,history,intent)
   p.seal(root/'requests'/('restart-'+str(actual['restart']+1)+'.json'),nextvalue)
  _,response=control.rpc(['scontrol',action,binding['job_id']],dest,intent,run=run)
  p.seal(dest/'ACK.json',dict(intent=intent,response=response,UTC=p.now(),may_be_absent_if_parent_terminated=True))
 return intent

def run(binding_ref):
 started=time.time()
 b=p.checked(binding_ref);request=p.checked(b['request']);cfg,profile,_,_=p.metadata(request)
 p.validate_binding(b,b['request'],request);actual=allocation(b,cfg)
 root=p.literal_home(Path(cfg['report'])/'ROOT_LAUNCH')
 execution_ref=p.desc(root/'requests'/('restart-'+str(actual['restart'])+'.json'));e=p.checked(execution_ref)
 expected=control.execution_request(binding_ref,e['authorization'],actual['restart'],e['accepted_history'],e['predecessor_transition'])
 p.require(e==expected,'Current immutable execution request mismatch')
 # All allocation authentication above is HOME/scheduler only. DATA starts here.
 cursor=validate_history(cfg,e,b,lambda path,identity:physical_check(binding_ref,path,identity,root/'physical-checks'/('before-training-r'+str(actual['restart'])+'.json')))
 name=b['job_id']+'-r'+str(actual['restart'])+'-'+b['nonce'][:8]
 parent=p.literal_home(root/'attempts'/name);parent.mkdir(parents=True,exist_ok=False)
 childpath=Path(cfg['report'])/'attempts'/name/'child'
 launch=p.seal(parent/'CHILD_LAUNCH_BINDING.json',launch_binding(b,b['configuration'],e,childpath))
 p.seal(parent/'ATTEMPT_START.json',dict(actual=actual,execution=execution_ref,accepted_cursor=cursor,
     remaining_natural_updates=cfg['expected_horizon']['natural_updates']-cursor,
     continuation_allowance='STRICT_ACCEPTED_PROGRESS_UP_TO_NATURAL_HORIZON_NO_ENGINEERING_REQUEUE_CAP',launcher_sources=p.sources(),UTC=p.now()))
 stop=actual['end_epoch']-600;p.require(stop>time.time(),'No actual cold/preparation/training window remains')
 argv=child_argv(profile,b['configuration'],e['authorization'],launch,childpath,stop)
 env=runtime.launch_environment(os.environ);env.update(NATIVE_RUN_IDENTITY=p.a.n.run_identity(cfg),NATIVE_ATTEMPT=str(childpath),NATIVE_STOP_PATH=str(childpath/'STOP'),
     HF_HOME=profile['cache_root'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SLURM_JOB_END_TIME=str(actual['end_epoch']),OMP_NUM_THREADS='8')
 requested=[False]
 def signal_stop(*_):requested[0]=True
 old={sig:signal.signal(sig,signal_stop) for sig in (signal.SIGUSR1,signal.SIGTERM)}
 p.seal(parent/'INVOCATION.json',dict(argv=argv,preCUDA_policy='CUBLAS4096_EXPLICIT_CUTLASS_DETERMINISTIC',execution=execution_ref,environment_secrets_recorded=False))
 try:
  with (parent/'torchrun.log').open('xb') as log:
   proc=subprocess.Popen(argv,cwd=p.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   while proc.poll() is None:
    if requested[0] or time.time()>=stop:
     if childpath.is_dir():(childpath/'STOP').touch(exist_ok=True)
    if time.time()>=actual['end_epoch']-60:
     proc.terminate()
     try:proc.wait(timeout=10)
     except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
     raise ValueError('No accepted result within actual deadline; retain failure, no auto continuation')
    time.sleep(1)
   code=proc.returncode
 finally:
  for sig,handler in old.items():signal.signal(sig,handler)
 p.seal(parent/'CHILD_RETURN.json',dict(returncode=code,UTC=p.now(),log=p.desc(parent/'torchrun.log'),
     wall_seconds_since_worker_entry=time.time()-started,setup_failure_signal_and_guard_costs_count=True,
     actual_job_id=b['job_id'],actual_restart_count=actual['restart'],full_Slurm_allocation_cost_requires_actual_accounting=True))
 if code==0:
  final=p.desc(childpath/'STUDENT_COMPLETE.json')
  return collect.collect(b['configuration'],e['accepted_history']+[final],root/'COMPLETE_TRAJECTORY.json')
 p.require(code==75,'Unaccepted child failure is not an automatic continuation')
 cont=p.desc(childpath/'ACCEPTED_CONTINUATION_V1.json');accepted=p.checked(cont)
 delta=accepted['global_step']-cursor;p.require(delta>0,'No accepted progress for continuation allowance')
 remaining=cfg['expected_horizon']['natural_updates']-accepted['global_step']
 p.seal(parent/'MEASURED_CONTINUATION_PLANNING.json',dict(actual_job_id=b['job_id'],actual_restart=actual['restart'],
     accepted=cont,actual_accepted_updates_this_attempt=delta,actual_worker_wall_seconds=time.time()-started,
     remaining_natural_updates=remaining,projection_attempts_at_this_observed_rate=(remaining+delta-1)//delta,
     projection_is_not_a_GPU_cap_or_scientific_result=True,hard_progress_bound=cfg['expected_horizon']['natural_updates'],
     no_hidden_max_steps=True,Slurm_total_cost_still_requires_actual_accounting=True))
 intent=transition(binding_ref,execution_ref,cont,root)
 # An accepted RPC can terminate this process before ACK, or after it. Do not
 # return normal COMPLETED while assuming a requeue. Root reconciles raw state.
 while time.time()<actual['end_epoch']-15:
  raw=subprocess.check_output(['scontrol','show','job','-o',b['job_id']],text=True,timeout=30);f=control.fields(raw)
  if f.get('Restarts')==str(actual['restart']+1) and f.get('JobState')=='PENDING':
   return p.seal(parent/'OBSERVED_TRANSITION.json',dict(intent=intent,actual_scheduler=raw,not_atomicity_or_pilot_PASS=True))
  time.sleep(5)
 raise ValueError('Transition unresolved before allocation end; external reconciliation required, no RPC retry')

def main():
 a=argparse.ArgumentParser();a.add_argument('--binding',required=True);v=a.parse_args();print(p.a.canonical(run(p.desc(v.binding))).decode())
if __name__=='__main__':main()
