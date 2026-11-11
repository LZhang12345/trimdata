"""Finite current-job supervisor, three phases, accepted-boundary requeue only."""
import argparse,fcntl,os,pwd,re,signal,socket,subprocess,time,uuid
from datetime import datetime
from pathlib import Path
from . import protocol as p,runtime

def allocation(binding,study,runner=subprocess.check_output):
 cfg=p.checked(study['native_configurations']['reference']);profile=p.checked(cfg['site_profile']);job=os.environ.get('SLURM_JOB_ID','');held=p.checked(study['held_binding']);intent=p.checked(held['intent']);site=study['site']
 p.require(binding['owner']=='Root' and job==binding['job_id']==study['job_id']==held['job_id'] and job.isdecimal() and not os.environ.get('SLURM_ARRAY_JOB_ID') and pwd.getpwuid(os.getuid()).pw_name==profile['slurm_user'],'Actual scalar owned GPU before DATA')
 p.require(intent['driver_sources']==p.sources() and intent['status']=='INTENT_BEFORE_SUBMISSION' and held['spool_reconciled'] is True,'Authentic source/intent/spool binding required')
 raw=runner(['scontrol','show','job','-o',job],text=True,timeout=30);f=dict(re.findall(r'(?:^|\s)([^\s=]+)=(\S+)',raw))
 expected=dict(JobId=job,JobState='RUNNING',UserId=profile['slurm_user']+'('+str(os.getuid())+')',JobName='dsmm-epoch-'+site+'-s'+str(study['seed']),Account=profile['account'],Partition=profile['partition'],QOS=profile['qos'],NumCPUs='8',NumNodes='1',NumTasks='1',Restarts=os.environ.get('SLURM_RESTART_COUNT','0'),TimeLimit='00:55:00',NodeList=socket.gethostname().split('.')[0],Command=str(p.HERE/('run_'+site+'.sbatch')),WorkDir=str(p.ROOT),Comment='dsmm-epoch:'+intent['nonce'],TresPerNode='gres/gpu:'+profile['gpu_type']+':1')
 p.require(all(f.get(k)==v for k,v in expected.items()),'Actual current site/UID/nonce/spool/resources differ')
 tres=dict(x.split('=',1) for x in f.get('AllocTRES','').split(',') if '=' in x);p.require(tres.get('cpu')=='8' and tres.get('mem') in ('64G','65536M') and tres.get('gres/gpu')=='1','Actual allocation differs')
 end=int(datetime.fromisoformat(f['EndTime']).timestamp());p.require(not os.environ.get('SLURM_JOB_END_TIME') or int(os.environ['SLURM_JOB_END_TIME'])==end,'Actual EndTime differs');p.require(end>time.time()+300,'No actual reserve remains');os.environ['SLURM_JOB_END_TIME']=str(end)
 return profile,raw,end

def requeue(binding,study,attempt,reason):
 profile,raw,end=allocation(binding,study);lockpath=Path(p.LOCKS[study['site']]);p.require(lockpath.is_file() and not any(x.is_symlink() for x in (lockpath,*lockpath.parents)),'Existing canonical admission lock only')
 with lockpath.open('r+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);p.validate_study(study);profile,raw,end=allocation(binding,study)
  if study['site']=='babel':
   from experiments.root_native_deterministic_replication_v3_20261024 import admit
   census=admit.census(profile,need_submission_slot=False)  # New author policy: actual site/account limits, no Root2 fence.
  else:
   limits=subprocess.check_output(['sacctmgr','-nP','show','qos','normal','format=Name,MaxSubmitPU'],text=True,timeout=30);rows=[v.split('|') for v in limits.splitlines() if v.startswith('normal|')];p.require(len(rows)==1 and rows[0][1].isdigit(),'Actual ACES submission ceiling missing')
   queue=subprocess.check_output(['squeue','-r','-h','-u',profile['slurm_user'],'-o','%i'],text=True,timeout=30);p.require(all(re.fullmatch(r'\d+(?:_\d+)?',v) for v in queue.splitlines() if v.strip()),'Malformed expanded census');p.require(len(queue.splitlines())<=int(rows[0][1]),'Actual existing-job count exceeds site ceiling');census=dict(limits=limits,all_user_queue=queue,no_new_submission=True)
  spool=subprocess.check_output(['scontrol','write','batch_script',binding['job_id'],'-'],timeout=30);p.require(spool==(p.HERE/('run_'+study['site']+'.sbatch')).read_bytes(),'Actual running spool changed')
  p.require(int(os.environ.get('SLURM_RESTART_COUNT','0'))<3,'Finite at most three physical requeues; no private loop')
  target=attempt/'REQUEUE_INTENT.json';p.require(not target.exists(),'Prior ambiguous requeue requires reconciliation')
  intent=p.seal(target,dict(status='INTENT_BEFORE_RPC',job_id=binding['job_id'],reason=reason,same_physical_job_only=True,actual_slurm=raw,census=census,study=binding['study'],UTC=p.now()))
  try:reply=subprocess.run(['scontrol','requeue',binding['job_id']],capture_output=True,text=True,timeout=30)
  except subprocess.TimeoutExpired:p.seal(attempt/'REQUEUE_AMBIGUOUS.json',dict(intent=intent,no_automatic_retry=True));raise
  p.seal(attempt/'REQUEUE_ACK.json',dict(intent=intent,returncode=reply.returncode,stdout=reply.stdout,stderr=reply.stderr));p.require(reply.returncode==0,'Actual same-job requeue not acknowledged')

def invoke(study_ref,study,configs,arm,phase,profile,target,stop_at,current):
 target.mkdir(parents=True,exist_ok=False);stop=target/'STOP_REQUESTED';current[0]=stop
 env=runtime.launch_environment(os.environ);env.update(NATIVE_STOP_PATH=str(stop),NATIVE_ATTEMPT=str(target),NATIVE_RUN_IDENTITY=p.n.run_identity(configs[arm]),HF_HOME=profile['cache_root'],HF_HUB_CACHE=str(Path(profile['cache_root'])/'hub'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8')
 argv=[profile['python'],'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=1','-m',p.PACKAGE+'.child','--study',study_ref['path'],'--arm',arm,'--phase',phase,'--attempt',str(target),'--stop-at-epoch',str(stop_at)]
 p.seal(target/'INVOCATION.json',dict(argv=argv,study=study_ref,phase=phase,stop_at=stop_at,actual_job_id=os.environ['SLURM_JOB_ID'],actual_restart_count=int(os.environ.get('SLURM_RESTART_COUNT','0'))))
 with (target/'torchrun.log').open('xb',buffering=0) as stream:code=subprocess.Popen(argv,env=env,stdout=stream,stderr=subprocess.STDOUT).wait()
 current[0]=None;result=target/'CHILD_RESULT.json';p.seal(target/'RETURN.json',dict(returncode=code,result=p.desc(result) if result.exists() else None));p.require(code==0 and result.exists(),'Actual child failure preserved; no automatic retry')
 value=p.read(result);p.require(value['actual_job_id']==study['job_id'] and value['driver_sources']==p.sources() and value['phase']==phase,'Child ownership/source/phase mismatch');return p.desc(result)

def run(binding):
 study=p.checked(binding['study']);profile,raw,end=allocation(binding,study);configs,modules=p.validate_study(study)
 report=Path(study['report']);p.require(p.n.within(report,profile['registry_root']) and not any(x.is_symlink() for x in (report,*report.parents)),'Approved literal registry only');report.mkdir(parents=True,exist_ok=True)
 with (report/'EXECUTION.lock').open('r+' if (report/'EXECUTION.lock').exists() else 'w+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p.require(not (report/'RAW_READOUT.json').exists(),'Existing result forbids replay')
  attempt=report/'attempts'/(binding['job_id']+'-r'+os.environ.get('SLURM_RESTART_COUNT','0')+'-'+uuid.uuid4().hex[:8]);attempt.mkdir(parents=True,exist_ok=False)
  fs=os.statvfs(profile['output_root']);required=study['collective_reserved_bytes'];p.require(required>=p.MIN_COLLECTIVE and fs.f_bavail*fs.f_frsize>=required and fs.f_favail>=4000,'Actual collective capacity insufficient')
  capacity=dict(actual_job_id=binding['job_id'],available_bytes=fs.f_bavail*fs.f_frsize,available_inodes=fs.f_favail,required_bytes=required,actual_slurm=raw,actual_EndTime=end)
  if study['site']=='aces':
   from .capacity import quota
   capacity['actual_user_quota']=quota(subprocess.check_output(['showquota'],text=True,timeout=30),fs,required)
   p.require(capacity['actual_user_quota']['available_bytes_conservative']>=required,'Actual user quota below collective reservation')
  p.seal(attempt/'ACTUAL_STORAGE.json',capacity);current=[None]
  def stop(*_):
   if current[0] is not None:current[0].touch()
  signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
  stop_at=end-300;ref=report/'PHASE_REFERENCE.json';cut=report/'PHASE_INTERRUPTED.json';resume=report/'PHASE_RESUMED.json'
  if not ref.exists():
   p.require(not cut.exists() and not resume.exists(),'Out of order phases');value=invoke(binding['study'],study,configs,'reference','reference',profile,attempt/'reference',stop_at,current);p.require(p.checked(value)['status']=='NATIVE_TRAJECTORY_COMPLETE','Reference must be uninterrupted');p.seal(ref,value)
  if not cut.exists():
   if stop_at-time.time()<study['minimum_launch_seconds']:requeue(binding,study,attempt,'ACCEPTED_COMPLETE_REFERENCE_BEFORE_NEW_PHASE');return
   value=invoke(binding['study'],study,configs,'resume','interrupted',profile,attempt/'interrupted',stop_at,current);child=p.checked(value);p.require(child['status']=='ACCEPTED_NATIVE_CONTINUATION' and p.checked(child['files']['STEP-4.json'])['global_step']==4,'Deliberate accepted cut4 required');modules['native'].checkpoint_receipt(Path(configs['resume']['output'])/'training/checkpoint-4',p.n.run_identity(configs['resume']));p.seal(cut,value);requeue(binding,study,attempt,'ACCEPTED_CUT4_ACTUAL_RESTART');return
  p.require(int(os.environ.get('SLURM_RESTART_COUNT','0'))>p.checked(p.read(cut))['actual_restart_count'],'Genuine same physical job restart required')
  if not resume.exists():
   value=invoke(binding['study'],study,configs,'resume','resumed',profile,attempt/'resumed',stop_at,current);p.require(p.checked(value)['status']=='NATIVE_TRAJECTORY_COMPLETE','Actual resumed endpoint required');p.seal(resume,value)
  from .readout import compare
  result=compare(study,{name:p.read(path) for name,path in [('reference',ref),('interrupted',cut),('resumed',resume)]});p.seal(report/'RAW_READOUT.json',result);p.require(result['status']=='PASS_ACTUAL_DECLARED_EPOCH_RUNTIME_UNREVIEWED','Numerical or naturalendpoint criteria failed; preserve failures');return result

def main():
 a=argparse.ArgumentParser();a.add_argument('--binding',required=True);v=a.parse_args();run(p.read(v.binding))
if __name__=='__main__':main()
