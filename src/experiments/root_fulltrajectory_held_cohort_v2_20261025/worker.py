"""No NN until a genuine later activation; held admission is not readiness."""
import argparse,importlib,os,socket,subprocess
from pathlib import Path
from . import protocol as p
from .admit import wall

def run(binding_ref,runner=subprocess.check_output):
 b=p.c.checked(binding_ref);request,_,profile=p.reviewed(b['request']);p.actor(request['site'])
 intent=p.c.checked(b['intent']);ack=p.c.checked(b['ack'])
 p.c.require(intent['request']==b['request'] and intent['nonce']==b['nonce'] and intent['sources']==p.sources()
     and ack['intent']==b['intent'] and ack['job_id']==b['job_id'] and ack['nonce']==b['nonce'], 'Actual intent/ACK/source binding required')
 p.c.require(binding_ref==p.c.desc(Path(request['control_root'])/'BINDING.json'),'Canonical actual binding path required')
 p.c.require(b['sources']==p.sources() and b['command']==p.c.desc(Path(__file__).with_name('run.sbatch')),'Held source/spool command changed')
 job=os.environ.get('SLURM_JOB_ID','');p.c.require(job.isdecimal() and job==b['job_id'],'Actual scalar binding required')
 raw=runner(['scontrol','show','job','-o',job],text=True,timeout=30);lines=[x for x in raw.splitlines() if x.strip()]
 p.c.require(len(lines)==1,'Exact one scalar scheduler record required');f=dict(p.re.findall(r'(?:^|\s)([^\s=]+)=(\S+)',lines[0]));s=p.SITES[request['site']]
 expected=dict(JobId=job,JobState='RUNNING',UserId=s['user']+'('+str(s['uid'])+')',Account=s['account'],Partition=s['partition'],QOS=s['qos'],NumCPUs='8',NumTasks='1',NumNodes='1',TimeLimit=wall(request['outer_allocation']['minutes']),Restarts=os.environ.get('SLURM_RESTART_COUNT','0'),TresPerNode='gres/gpu:'+s['gpu']+':1',
     Comment='dsmm-fulltrajectory:'+b['nonce'],Command=b['command']['path'],WorkDir=str(p.ROOT),NodeList=socket.gethostname().split('.')[0])
 p.c.require(all(f.get(k)==v for k,v in expected.items()) and f.get('ArrayJobId') in (None,'(null)'), 'Current actual ownership/resource/route differs')
 t=dict(x.split('=',1) for x in f.get('AllocTRES','').split(',') if '=' in x)
 p.c.require(t.get('cpu')=='8' and t.get('mem') in ('64G','65536M') and t.get('gres/gpu')=='1','Actual allocated CPU/GPU/memory differs')
 spool=runner(['scontrol','write','batch_script',job,'-'],timeout=30)
 p.c.require(spool==Path(b['spool']['path']).read_bytes() and p.c.desc(b['spool']['path'])==b['spool'],'Actual spool changed')
 # Activation is created once after genuine reviews. A premature release fails
 # here before DATA/model/CUDA imports and never fabricates a future PASS.
 path=Path(request['control_root'])/'ACTIVATION.json'
 p.c.require(path.is_file(),'CONDITIONAL_HELD_GATES_UNSATISFIED: no actual activation; no NN allowed')
 _,_,_,a=p.activation(binding_ref,p.c.desc(path))
 # Separately reviewed activation target owns full allocation/storage/media/
 # capsule/config validation and accepted-progress continuation. It is not
 # supplied or presumed qualified by this held-only source packet.
 module=importlib.import_module(a['target_module']);return getattr(module,a['target_callable'])(binding_ref,p.c.desc(path))

def main():
 a=argparse.ArgumentParser();a.add_argument('--binding',required=True);v=a.parse_args();print(p.c.canonical(run(p.c.desc(v.binding))).decode())
if __name__=='__main__':main()
