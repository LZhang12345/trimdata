"""Current read-only allocation authentication for Root's common launcher API."""
import os,pwd,re,socket,subprocess,time
from datetime import datetime
from pathlib import Path
from . import protocol as p

def require_current(ref,config_ref,permit_ref,config,profile,runner=subprocess.check_output):
 value=p.checked(ref);job=os.environ.get('SLURM_JOB_ID','')
 p.require(value['schema']=='production-student-launch-binding-v1' and value['owner']=='Root'
     and value['configuration']==config_ref and value['authorization']==permit_ref
     and value['physical_job_id']==job and job.isdecimal()
     and value['adapter_sources']==p.sources() and re.fullmatch('[a-f0-9]{32}',value['nonce']),
     'No exact current Root launch binding')
 p.require(pwd.getpwuid(os.getuid()).pw_name==profile['slurm_user']=='luyangz'
     and profile['site']=='babel' and profile['partition']=='general' and profile['qos']=='normal'
     and profile['account']=='cx' and profile['gpu_type']=='L40S', 'Wrong actual CPU+GPU/site owner')
 command=value['command'];path=Path(command['path'])
 p.require(path.is_absolute() and not any(v.is_symlink() for v in (path,*path.parents))
     and p.desc(path)==command,'Current Root launcher source differs')
 raw=runner(['scontrol','show','job','-o',job],text=True,timeout=30)
 fields=dict(re.findall(r'(?:^|\s)([^\s=]+)=(\S+)',raw));allocation=config['allocation']
 minutes=allocation['minutes'];days,remaining=divmod(minutes,1440);hours,mins=divmod(remaining,60)
 expected_time=(str(days)+'-' if days else '')+f'{hours:02d}:{mins:02d}:00'
 wanted=dict(TimeLimit=expected_time,JobId=job,JobState='RUNNING',UserId=profile['slurm_user']+'('+str(os.getuid())+')',
     Account='cx',Partition='general',QOS='normal',NumNodes='1',NumTasks='1',
     NumCPUs=str(allocation['cpus']),Restarts=os.environ.get('SLURM_RESTART_COUNT','0'),
     Command=command['path'],WorkDir=str(p.ROOT),Comment='dsmm-production-student:'+value['nonce'],
     TresPerNode='gres/gpu:L40S:1',NodeList=socket.gethostname().split('.')[0])
 p.require(all(fields.get(k)==v for k,v in wanted.items()), 'Actual current UID/nonce/spool/resources differ')
 tres=dict(v.split('=',1) for v in fields.get('AllocTRES','').split(',') if '=' in v)
 p.require(tres.get('gres/gpu')=='1' and tres.get('cpu')==str(allocation['cpus'])
     and tres.get('mem') in (str(allocation['memory_mb'])+'M',str(allocation['memory_mb']//1024)+'G'),
     'Actual allocation differs from reviewed student CPU/GPU/memory')
 end=int(datetime.fromisoformat(fields['EndTime']).timestamp())
 p.require(end>time.time()+allocation['reserve_seconds'],'Insufficient owned deadline reserve')
 p.require(not os.environ.get('SLURM_JOB_END_TIME') or int(os.environ['SLURM_JOB_END_TIME'])==end,'EndTime drift')
 return dict(binding=ref,actual_slurm=raw,actual_job_id=job,actual_EndTime=end,
     scheduler_mutations=False,Root_owns_atomic_requeuehold=True)

def require_attempt(attempt,config,profile,current,binding):
 attempt=Path(attempt)
 report=Path(config['report']);registry=Path(profile['registry_root'])
 p.require(attempt.is_absolute() and report.is_absolute() and '..' not in attempt.parts
     and p.n.within(report,registry) and not any(v.is_symlink() for v in (attempt,*attempt.parents)),
     'Child attempt must remain in literal owned report route')
 name=current['actual_job_id']+'-r'+os.environ.get('SLURM_RESTART_COUNT','0')+'-'+binding['nonce'][:8]
 expected=report/'attempts'/name/'child'
 p.require(attempt==expected and binding['child_attempt']==str(expected)
     and binding['actual_restart_count']==int(os.environ.get('SLURM_RESTART_COUNT','0')),
     'Wrong current owned child attempt/restart route')
 p.require(not attempt.exists(),'New immutable child attempt required')
 return expected

def require_metadata_registration(ref,runner=subprocess.check_output):
 value=p.checked(ref);job=os.environ.get('SLURM_JOB_ID','')
 p.require(value['schema']=='registered-production-student-metadata-cpu-gpu-v1'
     and value['owner']=='Root' and value['scope']=='REGISTERED_PRODUCTION_STUDENT_METADATA_CPU_GPU'
     and job.isdecimal() and value['actual_job_id']==job
     and pwd.getpwuid(os.getuid()).pw_name=='luyangz','No actual Root CPU+GPU metadata registration')
 command=value['command'];path=Path(command['path'])
 p.require(path.is_absolute() and not any(v.is_symlink() for v in (path,*path.parents))
     and p.desc(path)==command and re.fullmatch('[a-f0-9]{32}',value['nonce']),
     'Registered metadata command/source/nonce differs')
 raw=runner(['scontrol','show','job','-o',job],text=True,timeout=30)
 f=dict(re.findall(r'(?:^|\s)([^\s=]+)=(\S+)',raw))
 expected=dict(JobId=job,JobState='RUNNING',UserId='luyangz('+str(os.getuid())+')',
     Account='cx',Partition='general',QOS='normal',NumNodes='1',
     Command=command['path'],WorkDir=value['workdir'],Comment=value['comment'],
     TresPerNode='gres/gpu:L40S:1',Restarts=os.environ.get('SLURM_RESTART_COUNT','0'))
 p.require(all(f.get(k)==v for k,v in expected.items()) and value['nonce'] in value['comment'],
     'Actual owned metadata parent identity differs')
 tres=dict(v.split('=',1) for v in f.get('AllocTRES','').split(',') if '=' in v)
 p.require(tres.get('gres/gpu')=='1' and int(tres.get('cpu','0'))>=4,
     'Heavy full-pool metadata requires actual GPU+at least4CPU')
 p.require(tres.get('mem','').endswith(('M','G')),'Actual memory field absent')
 mem=tres['mem'];memory_mb=int(mem[:-1])*(1024 if mem[-1]=='G' else 1)
 end=int(datetime.fromisoformat(f['EndTime']).timestamp())
 p.require(memory_mb>=16384 and end>time.time()+120,'Metadata preparation RAM/deadline insufficient')
 return dict(registration=ref,actual_job_id=job,actual_slurm=raw,actual_EndTime=end)
