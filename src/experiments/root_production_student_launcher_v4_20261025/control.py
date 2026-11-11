"""Explicit Root CLI only. Every mutation has an immutable intent; no RPC retries."""
import argparse,fcntl,os,re,subprocess,uuid
from pathlib import Path
from . import protocol as p

def fields(raw):return dict(re.findall(r'(?:^|\s)([^\s=]+)=(\S+)',raw))

def census(profile,needed,run=None):
 run=run or subprocess.check_output
 raw=run(['squeue','-r','-h','-u',profile['slurm_user'],'-o','%i|%u|%q|%T|%b|%j'],text=True,timeout=30)
 rows=[line.split('|') for line in raw.splitlines() if line.strip()]
 p.require(all(len(r)==6 and re.fullmatch(r'\d+(?:_\d+)?',r[0]) and r[1]==profile['slurm_user'] for r in rows)
     and len({r[0] for r in rows})==len(rows),'Successful expanded all-user census required')
 limits=run(['sacctmgr','-nP','show','qos','normal','format=Name,MaxTRESPU,MaxJobsPU,MaxSubmitPU'],text=True,timeout=30)
 values=[r.split('|') for r in limits.splitlines() if r.startswith('normal|')]
 p.require(len(values)==1 and len(values[0])>=4,'Current actual normal QoS limits absent')
 _,tres,jobs,submits,*_=values[0];tm=dict(v.split('=',1) for v in tres.split(',') if '=' in v)
 p.require(tm.get('gres/gpu')=='8' and jobs.isdecimal() and submits.isdecimal() and int(jobs)>0 and int(submits)>0,'Advertised site/account limits changed; re-review')
 normal=[r for r in rows if r[2]=='normal'];active=[r for r in normal if r[3] in ('RUNNING','COMPLETING','CONFIGURING','SUSPENDED')]
 gpu=0
 for r in normal:
  if r[4] in ('','N/A','(null)','None'):continue
  m=re.fullmatch(r'(?:gres/)?gpu(?::[A-Za-z0-9_]+)?:(\d+)',r[4]);p.require(m is not None and int(m[1])>0,'Unknown GPU multiplicity')
  if r in active:gpu+=int(m[1])
 p.require(len(normal)+needed<=int(submits) and len(active)<=int(jobs) and gpu<=8,'Actual all-user site ceiling reached')
 return dict(raw=raw,limits=limits,normal_requests=len(normal),running_jobs=len(active),running_GPUs=gpu,
     MaxSubmitPU=int(submits),MaxJobsPU=int(jobs),MaxGPUsPU=8,new_requests=needed,private_two_or_four_fence=False)

def held(raw,job,binding,initial):
 f=fields(raw)
 expected=dict(JobId=job,JobName=binding['job_name'],JobState='PENDING',Reason='JobHeldUser',UserId='luyangz('+str(os.getuid())+')',
     Account='cx',Partition='general',QOS='normal',NumTasks='1',NumCPUs='8',TimeLimit='00:55:00',Requeue='1',
     Command=binding['command']['path'],WorkDir=str(p.ROOT),Comment='dsmm-production-student:'+binding['nonce'],TresPerNode='gres/gpu:L40S:1')
 p.require(all(f.get(k)==v for k,v in expected.items()) and f.get('NumNodes') in ('1','1-1')
     and f.get('ArrayJobId') in (None,'(null)') and f.get('AllocTRES') in (None,'(null)'), 'Exact owned scalar held identity/resources differ')
 t=dict(v.split('=',1) for v in f.get('ReqTRES','').split(',') if '=' in v)
 p.require(t.get('cpu')=='8' and t.get('mem') in ('64G','65536M') and t.get('gres/gpu')=='1','Requested resources differ')
 if initial:p.require(f.get('Restarts')=='0' and f.get('RunTime')=='00:00:00' and f.get('StartTime')=='Unknown'
     and f.get('Dependency') in (None,'(null)','None'),'Initial held request already started or has hidden dependency')
 else:p.require(f.get('Restarts','').isdecimal(),'Actual resumed held restart absent')
 return f

def rpc(argv,dest,intent,run=None):
 """Timeout/nonzero/unknown result is never retried, including requeuehold."""
 run=run or subprocess.run
 try:r=run(argv,text=True,capture_output=True,timeout=60)
 except (subprocess.TimeoutExpired,OSError) as exc:
  p.seal(dest/'AMBIGUOUS_RPC.json',dict(intent=intent,exception_type=type(exc).__name__,automatic_retry=False));raise
 ref=p.seal(dest/'RPC_RESPONSE.json',dict(intent=intent,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
 if r.returncode!=0:
  p.seal(dest/'AMBIGUOUS_RPC.json',dict(intent=intent,response=ref,automatic_retry=False));raise ValueError('RPC unacknowledged; external exact reconciliation required')
 return r,ref

def lock_path():
 path=Path(p.LOCK)
 p.require(path.is_file() and not any(x.is_symlink() for x in (path,*path.parents)),'Existing canonical shared admission lock only')
 return path

def submit(request_ref):
 request=p.checked(request_ref);cfg,profile,permit,_=p.metadata(request);p.actor(profile)
 p.require(permit['phase']=='production' or cfg['id']==p.a.cell_id('5:uniform',17),'Only genuine canonical pilot or reviewed production permit')
 root=p.literal_home(Path(cfg['report'])/'ROOT_LAUNCH');command=p.desc(p.HERE/'run.sbatch')
 with lock_path().open('r+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);p.actor(profile)
  p.require(not root.exists(),'Any prior intent/binding requires reconciliation; no automatic resubmission')
  c=census(profile,1);name='dsmm-production-student-'+cfg['id']
  p.require(not any(r[5]==name for r in [v.split('|') for v in c['raw'].splitlines() if v]),'Same trajectory already queued')
  root.mkdir(parents=True,exist_ok=False);nonce=uuid.uuid4().hex;binding_path=root/'BINDING.json'
  argv=['sbatch','--parsable','--hold','--requeue','--open-mode=append','--nodes=1','--ntasks=1','--account=cx','--partition=general','--qos=normal',
      '--gres=gpu:L40S:1','--cpus-per-task=8','--mem=64G','--time=00:55:00','--signal=B:USR1@600',
      '--job-name='+name,'--comment=dsmm-production-student:'+nonce,'--chdir='+str(p.ROOT),'--output='+str(root/'slurm-%j.out'),command['path'],str(binding_path),profile['python']]
  intent=p.seal(root/'SUBMISSION_INTENT.json',dict(owner='Root',status='INTENT_BEFORE_SUBMISSION',request=request_ref,nonce=nonce,argv=argv,
      launcher_sources=p.sources(),actual_census=c,UTC=p.now()))
  r,response=rpc(argv,root,intent);match=re.fullmatch(r'(\d+)(?:;[^\s]+)?\s*',r.stdout)
  if match is None:
   p.seal(root/'AMBIGUOUS_ACK.json',dict(intent=intent,response=response,automatic_retry=False));raise ValueError('No exact scalar ACK; do not retry')
  job=match[1];ack=p.seal(root/'ACK.json',dict(job_id=job,intent=intent,response=response,UTC=p.now()))
  binding=dict(schema='root-production-student-physical-binding-v1',owner='Root',job_id=job,nonce=nonce,job_name=name,
      request=request_ref,configuration=request['configuration'],command=command,launcher_sources=p.sources(),adapter_sources=p.a.sources(),intent=intent,ack=ack)
  raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=30);held(raw,job,binding,True)
  spool=subprocess.check_output(['scontrol','write','batch_script',job,'-'],timeout=30)
  p.require(spool==(p.HERE/'run.sbatch').read_bytes(),'Held spool differs after ACK; preserve and stop')
  spool_path=root/'ACTUAL_SPOOL.sbatch'
  with spool_path.open('xb') as f:f.write(spool);f.flush();os.fsync(f.fileno())
  binding.update(spool=p.desc(spool_path),actual_initial_held=raw)
  return p.seal(binding_path,binding)

def archived_history(cfg,binding,history):
 """Authenticate HOME archives only; never claim current physical DATA state."""
 cursor=0
 for ordinal,ref in enumerate(history):
  p.literal_home(ref['path']);v=p.checked(ref);p.literal_home(v['accepted_archive']['path']);state=p.checked(v['accepted_archive'])
  p.require(v['status']=='ACCEPTED_STUDENT_CONTINUATION_REQUIRES_ROOT_SAME_JOB_ACTION'
      and v['configuration']==binding['configuration'] and v['config_identity']==p.a.n.run_identity(cfg)
      and v['actual_job_id']==binding['job_id'] and v['actual_restart_count']==ordinal
      and v['adapter_sources']==p.a.sources() and v['runtime']==p.a.RUNTIME
      and cursor<v['global_step']<=cfg['expected_horizon']['natural_updates']
      and v['accepted_archive']['bytes']==v['accepted']['bytes'] and v['accepted_archive']['sha256']==v['accepted']['sha256']
      and state['status']=='ACCEPTED_FULL_NATIVE_STATE' and state['identity']==v['config_identity']
      and state['global_step']==v['global_step'],'Archived accepted history is foreign/noncontiguous/changed')
  cursor=v['global_step']
 return cursor

def execution_request(binding_ref,authorization_ref,restart,history,transition=None):
 """Pure immutable request. Production gate remains the genuine125 verifier."""
 b=p.checked(binding_ref);request=p.checked(b['request']);cfg,profile,_,load=p.metadata(request)
 p.validate_binding(b,b['request'],request)
 providers={name:load(ref) for name,ref in cfg['providers'].items()}
 # Import explicitly; no dependency on incidental package attributes.
 from experiments.root_production_student_adapter_v2_20261025.authorization import verify
 permit=verify(cfg,providers,profile,authorization_ref,load)
 p.require(type(restart) is int and 0<=restart<=cfg['expected_horizon']['natural_updates'] and len(history)==restart,'Finite accepted-only continuation allowance exceeded')
 archived_history(cfg,b,history)
 if restart==0:p.require(not history and transition is None,'Fresh initial request has no accepted history')
 else:
  p.require(history and transition is not None,'Restart requires authentic accepted same-job transition')
  previous=p.checked(history[-1]);t=p.checked(transition)
  p.require(previous['actual_job_id']==b['job_id'] and previous['actual_restart_count']==restart-1
      and previous['config_identity']==p.a.n.run_identity(cfg) and t['continuation']==history[-1]
      and t['physical_binding']==binding_ref and t['from_restart']==restart-1 and t['to_restart']==restart,
      'Missing accepted transition/config/physical lineage')
  guard=p.checked(t['physical_capsule_revalidation'])
  p.require(guard['status']=='ACTUAL_OWNED_GPU_CAPSULE_REVALIDATION_NOT_TRAINING_QUALIFICATION'
      and guard['config_identity']==p.a.n.run_identity(cfg) and guard['physical_job_id']==b['job_id']
      and guard['actual_restart']==restart-1 and guard['accepted']==p.checked(previous['accepted_archive']),
      'Missing authentic pre-transition GPU capsule check/archive binding')
  if previous['gate_hold_required']:p.require(permit['phase']=='production','Accepted4 requires genuine new independently reviewed production permit')
 return dict(schema='root-production-student-execution-request-v1',owner='Root',physical_binding=binding_ref,
     configuration=b['configuration'],authorization=authorization_ref,actual_restart_count=restart,accepted_history=history,
     predecessor_transition=transition,launcher_sources=p.sources(),config_identity=p.a.n.run_identity(cfg))

def write_execution(binding_ref,authorization_ref,restart,history,transition,output):
 b=p.checked(binding_ref);cfg=p.checked(b['configuration']);p.actor(p.checked(cfg['site_profile']))
 expected=Path(cfg['report'])/'ROOT_LAUNCH/requests'/('restart-'+str(restart)+'.json')
 p.require(p.literal_home(output)==expected,'Only exact current physical restart request route')
 value=execution_request(binding_ref,authorization_ref,restart,history,transition)
 return p.seal(output,value)

def release(binding_ref,execution_ref):
 """Metadata only. Actual DATA state is revalidated by resumed GPU before NN."""
 b=p.checked(binding_ref);request=p.checked(b['request']);cfg,profile,_,_=p.metadata(request);p.actor(profile);p.validate_binding(b,b['request'],request)
 e=p.checked(execution_ref);expected=execution_request(binding_ref,e['authorization'],e['actual_restart_count'],e['accepted_history'],e['predecessor_transition'])
 p.require(e==expected,'Execution request drift')
 root=p.literal_home(Path(cfg['report'])/'ROOT_LAUNCH');restart=e['actual_restart_count']
 p.require(Path(execution_ref['path'])==root/'requests'/('restart-'+str(restart)+'.json'),'Only immutable actual restart request route')
 with lock_path().open('r+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);p.actor(profile);c=census(profile,0)
  raw=subprocess.check_output(['scontrol','show','job','-o',b['job_id']],text=True,timeout=30);f=held(raw,b['job_id'],b,restart==0)
  p.require(int(f['Restarts'])==restart,'Physical held restart differs')
  spool=subprocess.check_output(['scontrol','write','batch_script',b['job_id'],'-'],timeout=30);p.require(spool==Path(b['spool']['path']).read_bytes(),'Actual current spool differs')
  if restart:
   t=p.checked(e['predecessor_transition']);reconciled=p.checked(p.desc(root/'transitions'/('r'+str(restart-1))/'RECONCILED.json'))
   p.require(reconciled['intent']==e['predecessor_transition'] and reconciled['actual_held'] is True
       and t['action']=='requeuehold','Only externally reconciled accepted pilot hold can be released')
  dest=root/'releases'/('r'+str(restart));p.literal_home(dest);dest.mkdir(parents=True,exist_ok=False)
  intent=p.seal(dest/'INTENT.json',dict(owner='Root',action='release',binding=binding_ref,execution=execution_ref,actual_before=raw,census=c,UTC=p.now(),physical_DATA_validation_on_login=False))
  _,ack=rpc(['scontrol','release',b['job_id']],dest,intent)
  return p.seal(dest/'ACK.json',dict(intent=intent,response=ack,job_id=b['job_id'],UTC=p.now(),runtime_PASS=False))

def reconcile(binding_ref,transition_ref):
 """Read-only Root reconciliation after ACK loss/termination; NEVER repeats RPC."""
 b=p.checked(binding_ref);request=p.checked(b['request']);cfg,profile,_,_=p.metadata(request);p.actor(profile);p.validate_binding(b,b['request'],request)
 t=p.checked(transition_ref);p.require(t['physical_binding']==binding_ref and t['action'] in ('requeue','requeuehold'),'Foreign transition')
 raw=subprocess.check_output(['scontrol','show','job','-o',b['job_id']],text=True,timeout=30);f=fields(raw)
 p.require(f.get('UserId')=='luyangz('+str(os.getuid())+')' and f.get('JobId')==b['job_id']
     and f.get('Comment')=='dsmm-production-student:'+b['nonce'] and f.get('Command')==b['command']['path']
     and f.get('Restarts')==str(t['to_restart']),'No actual next physical restart; retain unresolved intent')
 spool=subprocess.check_output(['scontrol','write','batch_script',b['job_id'],'-'],timeout=30);p.require(spool==Path(b['spool']['path']).read_bytes(),'Actual spool changed')
 isheld=f.get('JobState')=='PENDING' and f.get('Reason')=='JobHeldUser'
 if t['action']=='requeuehold':held(raw,b['job_id'],b,False);p.require(isheld,'Pilot is not actually held')
 else:p.require(f.get('JobState') in ('PENDING','RUNNING','CONFIGURING'),'No actual accepted requeue state')
 path=Path(transition_ref['path']).parent/'RECONCILED.json'
 return p.seal(path,dict(status='ACTUAL_SAME_JOB_TRANSITION_RECONCILED_NOT_NUMERICAL_QUALIFICATION',intent=transition_ref,
     actual_scheduler=raw,actual_held=isheld,actual_restart=t['to_restart'],physical_DATA_validation_on_login=False,UTC=p.now()))

def main():
 parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
 s=sub.add_parser('submit');s.add_argument('--request',required=True)
 s=sub.add_parser('request');s.add_argument('--binding',required=True);s.add_argument('--authorization',required=True);s.add_argument('--restart',type=int,required=True);s.add_argument('--history',required=True);s.add_argument('--transition');s.add_argument('--output',required=True)
 s=sub.add_parser('release');s.add_argument('--binding',required=True);s.add_argument('--execution',required=True)
 s=sub.add_parser('reconcile');s.add_argument('--binding',required=True);s.add_argument('--transition',required=True)
 v=parser.parse_args()
 if v.action=='submit':result=submit(p.desc(v.request))
 elif v.action=='release':result=release(p.desc(v.binding),p.desc(v.execution))
 elif v.action=='reconcile':result=reconcile(p.desc(v.binding),p.desc(v.transition))
 else:result=write_execution(p.desc(v.binding),p.desc(v.authorization),v.restart,p.read(v.history),p.desc(v.transition) if v.transition else None,v.output)
 print(p.a.canonical(result).decode())
if __name__=='__main__':main()
