import hashlib,json,os,pwd,re,socket,subprocess,sys,time
from datetime import datetime
from pathlib import Path
from experiments.session6_official_eval_v2_20261024 import common as old
from . import UPSTREAM_SOURCES
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
SRC=Path('/scratch/user/u.lz341613/dsmm_s6_official_eval_20261024/native-start-execution-v6/src')
BASE=Path('/scratch/user/u.lz341613/dsmm_root_scienceqa_start_v1_20261025')
PYTHON='/scratch/user/u.lz341613/dsmm_native_campaign_20261023/envs/native-py310-cu121-v1/bin/python'
AUTHOR='/root/scientific_packet_review';STATUS='PASS_INDEPENDENT_ACES_SCIENCEQA_START_SOURCE_CONFIG_V1'
require=old.require;checked=old.checked;desc=old.desc;seal=old.seal;key=old.key;canonical=old.canonical

def sources():
 names=['__init__.py','common.py','audit.py','worker.py','run.sbatch']
 extra=['experiments/session6_scienceqa_bank_v1_20261024/'+n for n in ('__init__.py','scoring.py')]
 return dict(sorted({**UPSTREAM_SOURCES,**{str((HERE/n).relative_to(ROOT)):old.sha(HERE/n) for n in names},**{n:old.sha(ROOT/n) for n in extra}}.items()))

def validate(cfg):
 require(cfg['schema']=='root-aces-scienceqa-native-start-v1' and cfg['owner']=='Root' and cfg['sources']==sources() and cfg['bank']=='ScienceQA_image' and cfg['questions']==2017 and cfg['shards']==8,'Exact ScienceQA2017/source/8shard scope')
 require(cfg['output']==str(BASE) and cfg['python']==PYTHON and cfg['cache_root']=='/scratch/user/u.lz341613/dsmm_native_campaign_20261023/assets/hf_cache','Exact owned outputs/cache/runtime')
 require(cfg['training_updates']==0 and cfg['model_downloads']==0 and cfg['reserved_bytes']==256<<20 and cfg['max_attempts']==3,'No training/downloads; finite accepted QA continuation')
 require(cfg['model_result']['sha256']=='5520c7bdf835f2e4baaa79e85b7c3d2ce752e7737fad2552fb7d6fcbd28eb157','Actual qualified untouched eager START model')
 return cfg

def actual(binding):
 require(ROOT==SRC and Path.cwd()==ROOT and os.getuid()==56219 and pwd.getpwuid(os.getuid()).pw_name=='u.lz341613','Actual Root source/actor before DATA')
 require(re.fullmatch('[0-9a-f]{32}',binding['nonce']),'Actual32hex nonce')
 phase=binding['phase'];require(phase in ('qualify','predict','score') and binding['owner']=='Root','Exact finite Root phase')
 parent=binding['job_id'];task=None;job=os.environ.get('SLURM_JOB_ID','');require(job.isdecimal() and parent.isdecimal(),'Actual numeric IDs')
 if phase=='predict':
  task=os.environ.get('SLURM_ARRAY_TASK_ID','');require(task in tuple(map(str,range(8))) and os.environ.get('SLURM_ARRAY_JOB_ID')==parent,'Actual exact8 task array')
  query=parent+'_'+task
 else:require(not os.environ.get('SLURM_ARRAY_JOB_ID') and job==parent,'Scalar exactphase');query=job
 raw=subprocess.check_output(['scontrol','show','job','-o',query],text=True,timeout=30);require(len(raw.splitlines())==1,'Explicit parent_task must return one record')
 f=dict(re.findall(r'(?:^|\s)([^\s=]+)=(\S+)',raw));cpu=4 if phase=='score' else 8;mem=8 if phase=='score' else 64;minutes=15 if phase in ('qualify','score') else 20
 expected=dict(JobId=job,JobName='dsmm-root-sqa-'+phase+'-v1',JobState='RUNNING',UserId='u.lz341613(56219)',Account='158416130568',Partition='cpu' if phase=='score' else 'gpu',QOS='normal',NumCPUs=str(cpu),NumTasks='1',TimeLimit='00:%02d:00'%minutes,WorkDir=str(ROOT),Command=str(HERE/'run.sbatch'),NodeList=socket.gethostname().split('.')[0],Comment='root-sqa-start-v1:'+binding['nonce'])
 if task is not None:expected.update(ArrayJobId=parent,ArrayTaskId=task)
 diff={k:dict(expected=v,actual=f.get(k)) for k,v in expected.items() if f.get(k)!=v};require(not diff,'Actual resource/nonce/identity differences '+json.dumps(diff,sort_keys=True))
 t=dict(x.split('=',1) for x in f.get('AllocTRES','').split(',') if '=' in x);require(t.get('cpu')==str(cpu) and t.get('mem') in (str(mem)+'G',str(mem*1024)+'M'),'Exact allocated CPU/memory')
 if phase=='score':require(not any(k.startswith('gres/gpu') and int(v)>0 for k,v in t.items()),'Score uses CPU/noGPU')
 else:require(f.get('TresPerNode')=='gres/gpu:h100:1' and t.get('gres/gpu')=='1' and t.get('gres/gpu:h100')=='1','Exact H1001 allocation')
 rest=f.get('Restarts','');require(rest.isdecimal() and int(rest)<3 and os.environ.get('SLURM_RESTART_COUNT',rest)==rest,'Bounded actualrestart')
 require(subprocess.check_output(['scontrol','write','batch_script',job,'-'],timeout=30)==(HERE/'run.sbatch').read_bytes(),'Exact ownedspool')
 require(str(Path(sys.executable).absolute())==PYTHON,'Exact current interpreter')
 end=datetime.fromisoformat(f['EndTime']).timestamp();require(end-time.time()>120,'Actual remaining lifetime')
 return dict(job_id=job,parent_job_id=parent,task_id=None if task is None else int(task),restart=int(rest),node=f['NodeList'],scheduler=raw,end_epoch=end,nonce=binding['nonce'])

def source_guard(cfg):
 for name,digest in cfg['sources'].items():require(old.sha(ROOT/name)==digest,'Frozen source changed '+name)
 for name,m in tuple(sys.modules.items()):
  if name.startswith('experiments.') and getattr(m,'__file__',None):
   path=Path(m.__file__).resolve();require(path.is_relative_to(ROOT) and str(path.relative_to(ROOT)) in cfg['sources'],'Unbound sourceimport '+name)

def producer_roster_key(rows):
 # Original S5 frozen.canonical is compact ASCII JSON WITHOUT newline.
 return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()

def load_inputs(cfg):
 proof=checked(cfg['bank_provider']);require(proof['status']=='COMPLETE_FULL_SCIENCEQA2017_QA_GT_INPUT_AND_RGB_NOT_MODEL_GENERATION' and proof['questions']==proof['images']==2017 and proof['neural_forwards']==proof['downloads']==0 and proof['GPU_qualification'] is False,'Actual CPU2017 provider; no modelREADY inference')
 review=checked(proof['review']);require(review['status']=='PASS_INDEPENDENT_ROOT_SCIENCEQA_PROVIDER_SOURCE_CONFIG_V1' and review['configuration']==proof['configuration'],'Exact CPU source/config review')
 roster=checked(proof['roster']);metadata=checked(proof['metadata_ready']);pixels=checked(proof['original_provider_complete']);aliases=checked(pixels['image_aliases'])
 require(roster['expected_questions']==len(roster['records'])==2017 and roster['inference_answers_excluded'] and pixels['images']==2017 and pixels['physical_images_ready'] is True and pixels['model_generation_scoring_qualified'] is False,'Actual complete input-only scope')
 require(producer_roster_key(roster['records'])==metadata['semantic_roster_sha256'] and len(set(x['semantic_key'] for x in roster['records']))==2017,'Full native semantic order')
 return proof,roster,aliases
