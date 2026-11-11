"""Root CLI: one reviewed conditional HELD scalar; no release/requeue API."""
import argparse,fcntl,os,re,subprocess,uuid
from pathlib import Path
from . import protocol as p

def fields(raw):return dict(re.findall(r'(?:^|\s)([^\s=]+)=(\S+)',raw))
def wall(minutes):
 days,rem=divmod(minutes,1440);hours,mins=divmod(rem,60)
 return (str(days)+'-' if days else '')+f'{hours:02d}:{mins:02d}:00'

def census(site,run=subprocess.check_output):
 s=p.SITES[site]
 raw=run(['squeue','-r','-h','-u',s['user'],'-o','%i|%u|%q|%T|%b|%j'],text=True,timeout=30)
 rows=[x.split('|') for x in raw.splitlines() if x.strip()]
 p.c.require(all(len(x)==6 and re.fullmatch(r'\d+(?:_\d+)?',x[0]) and x[1]==s['user'] for x in rows)
     and len({x[0] for x in rows})==len(rows),'Successful expanded actual all-user census required')
 rawlimits=run(['sacctmgr','-nP','show','qos','normal','format=Name,MaxTRESPU,MaxJobsPU,MaxSubmitPU'],text=True,timeout=30)
 candidates=[x.split('|') for x in rawlimits.splitlines() if x.startswith('normal|')]
 p.c.require(len(candidates)==1 and len(candidates[0])>=4,'One actual normal QoS row required')
 _,tres,maxjobs,maxsubmit,*_=candidates[0];t=dict(x.split('=',1) for x in tres.split(',') if '=' in x)
 gpu=t.get('gres/gpu') if site=='babel' else t.get('gres/gpu:h100')
 expectedgpu=8 if site=='babel' else 10
 p.c.require(gpu==str(expectedgpu) and maxjobs.isdecimal() and maxsubmit.isdecimal()
     and int(maxsubmit)==s['limit'] and int(maxjobs)==(10 if site=='babel' else 40),'Actual advertised site limits changed')
 normal=[x for x in rows if x[2]=='normal'];active=[x for x in normal if x[3] in ('RUNNING','CONFIGURING','COMPLETING','SUSPENDED')]
 gpus=0
 for row in active:
  if row[4] in ('','N/A','(null)','None'):continue
  m=re.fullmatch(r'(?:gres/)?gpu(?::([A-Za-z0-9_]+))?:(\d+)',row[4]);p.c.require(m is not None and int(m[2])>0,'Unknown active GPU multiplicity')
  if site=='aces':p.c.require(m[1] is not None,'Untyped active ACES GPU cannot be assumed non-H100')
  if site=='babel' or m[1]=='h100':gpus+=int(m[2])
 p.c.require(len(normal)+1<=int(maxsubmit) and len(active)<=int(maxjobs) and gpus<=expectedgpu,'Actual all-user capacity reached; preserve missing candidate')
 return dict(raw=raw,limits=rawlimits,normal_requests=len(normal),running_jobs=len(active),running_relevant_GPUs=gpus,private_cap=False)

def held(raw,job,b,request):
 lines=[x for x in raw.splitlines() if x.strip()];p.c.require(len(lines)==1,'One scalar held scheduler row required');f=fields(lines[0]);s=p.SITES[request['site']]
 expected=dict(JobId=job,JobName=b['job_name'],JobState='PENDING',Reason='JobHeldUser',UserId=s['user']+'('+str(s['uid'])+')',
     Account=s['account'],Partition=s['partition'],QOS=s['qos'],NumCPUs='8',NumTasks='1',TimeLimit=wall(request['outer_allocation']['minutes']),
     Command=b['command']['path'],WorkDir=str(p.ROOT),Comment='dsmm-fulltrajectory:'+b['nonce'],
     TresPerNode='gres/gpu:'+s['gpu']+':1',Restarts='0',RunTime='00:00:00',StartTime='Unknown',ExitCode='0:0')
 p.c.require(all(f.get(k)==v for k,v in expected.items()) and f.get('NumNodes') in ('1','1-1')
     and all(f.get(k) in (None,'(null)','None') for k in ('AllocTRES','ArrayJobId','ArrayTaskId','NodeList','Dependency')),'Exact never-started owned scalar/route/nonce differs')
 t=dict(x.split('=',1) for x in f.get('ReqTRES','').split(',') if '=' in x)
 p.c.require(t.get('cpu')=='8' and t.get('mem') in ('64G','65536M') and t.get('gres/gpu')=='1','Held requested resources differ')
 return f

def submit(request_ref,run=subprocess.run,query=subprocess.check_output):
 preliminary=p.c.checked(request_ref);p.actor(preliminary['site']);request,cell,profile=p.reviewed(request_ref,admission=True)
 s=p.SITES[request['site']];lock=Path(s['lock'])
 p.c.require(lock.is_file() and not any(x.is_symlink() for x in (lock,*lock.parents)),'Existing canonical shared admission lock required')
 root=Path(request['control_root']);command=p.c.desc(Path(__file__).with_name('run.sbatch'))
 with lock.open('r+') as stream:
  fcntl.flock(stream,fcntl.LOCK_EX);p.actor(request['site']);request,cell,profile=p.reviewed(request_ref,admission=True)
  p.c.require(not root.exists(),'Existing intent/ACK/function claim requires reconciliation; no retry')
  current=census(request['site'],query);name='dsmm-fulltrajectory-'+request['site']+'-'+request['original_cell_id']
  oldname='dsmm-production-student-'+request['original_cell_id']
  predecessor_name='dsmm-fulltrajectory-'+request['site']+'-'+request['original_cell_id']
  p.c.require(not any(x.split('|')[5] in (name,oldname,predecessor_name) for x in current['raw'].splitlines() if x),'Same logical trajectory already queued')
  root.mkdir(parents=True,exist_ok=False);nonce=uuid.uuid4().hex
  claim=p.c.seal(root/'CONDITIONAL_FUNCTION_CLAIM.json',dict(owner='Root',status='HELD_CONDITIONAL_NO_ACTIVE_TRAINING_RIGHT',
      scientific_design=request['scientific_design'],site=request['site'],request=request_ref,reserved_storage_bytes=0,
      future_eligible_storage_reservation_required=True,source_ready_not_runtime_READY=True))
  argv=['sbatch','--parsable','--hold','--requeue','--open-mode=append','--nodes=1','--ntasks=1','--account='+s['account'],
      '--partition='+s['partition'],'--qos='+s['qos'],'--gres=gpu:'+s['gpu']+':1','--cpus-per-task=8','--mem=64G',
      '--time='+wall(request['outer_allocation']['minutes']),'--job-name='+name,'--comment=dsmm-fulltrajectory:'+nonce,
      '--chdir='+str(p.ROOT),'--output='+str(root/'slurm-%j.out'),command['path'],str(root/'BINDING.json'),profile['python']]
  intent=p.c.seal(root/'SUBMISSION_INTENT.json',dict(owner='Root',request=request_ref,nonce=nonce,argv=argv,sources=p.sources(),
      function_claim=claim,actual_census=current,held_only=True,release_authorized=False,UTC=p.c.now()))
  env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SBATCH_','SRUN_'))}
  try:r=run(argv,text=True,capture_output=True,timeout=60,env=env)
  except (OSError,subprocess.TimeoutExpired) as exc:
   p.c.seal(root/'AMBIGUOUS_SUBMISSION.json',dict(intent=intent,error=type(exc).__name__,automatic_retry=False));raise
  response=p.c.seal(root/'SUBMISSION_RESPONSE.json',dict(intent=intent,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
  m=re.fullmatch(r'(\d+)(?:;[^\s]+)?\s*',r.stdout)
  if r.returncode!=0 or m is None:
   p.c.seal(root/'AMBIGUOUS_SUBMISSION.json',dict(intent=intent,response=response,automatic_retry=False));raise ValueError('Unknown scheduler acknowledgement: no automatic retry')
  job=m[1];ack=p.c.seal(root/'ACK.json',dict(intent=intent,response=response,job_id=job,nonce=nonce))
  b=dict(schema='root-fulltrajectory-held-binding-v2',owner='Root',job_id=job,nonce=nonce,job_name=name,request=request_ref,
      command=command,sources=p.sources(),intent=intent,ack=ack,function_claim=claim,held_only=True)
  raw=query(['scontrol','show','job','-o',job],text=True,timeout=30);held(raw,job,b,request)
  spool=query(['scontrol','write','batch_script',job,'-'],timeout=30);p.c.require(spool==Path(command['path']).read_bytes(),'Held spool differs')
  target=root/'batch_script.sh'
  with target.open('xb') as out:out.write(spool);out.flush();os.fsync(out.fileno())
  b['spool']=p.c.desc(target)
  p.c.seal(root/'HELD_READBACK.json',dict(job_id=job,raw=raw,command=command,spool=b['spool'],no_release=True))
  return p.c.seal(root/'BINDING.json',b)

def main():
 a=argparse.ArgumentParser(description='One conditional HELD request only. No release command.');a.add_argument('--request',required=True);v=a.parse_args();print(p.c.canonical(submit(p.c.desc(v.request))).decode())
if __name__=='__main__':main()
