"""Small metadata-only boundary; physical validation belongs to owned GPUs."""
import json,os,pwd,re
from pathlib import Path
from experiments.root_production_student_adapter_v2_20261025 import protocol as a
from experiments.root_production_student_adapter_v2_20261025 import authorization
from experiments.root_production_epoch_qualification_v1_20261024.metadata import MetadataBindings

ROOT=a.ROOT;HERE=Path(__file__).resolve().parent
AUTHOR='/root/native_campaign_build'
PACKAGE='experiments.root_production_student_launcher_v4_20261025'
LOCK='/home/luyangz/data_selection_multimodal/collaboration/babel-two-track-20261019/ADMISSION.lock'
PROFILE_SHA='9fcf6ea41650162f5aa25fa8704d40b158052fa6bb7572d3d5c23f0fca96b226'
REGISTRY_ROOT=Path('/home/luyangz/data_selection_multimodal/native_campaign_v3_20261023/registry')
API_SHA='990ecff475b47668b0d68fdf524d64965297a70a3dafe97e58e30bb9b1b564c9'
require=a.require;checked=a.checked;read=a.read;desc=a.desc;seal=a.seal;key=a.key;sha=a.sha;now=a.now

def sources():
 own={str(f.relative_to(ROOT)):sha(f) for f in HERE.iterdir() if f.suffix in ('.py','.sbatch') and not f.name.startswith('test_')}
 return dict(sorted({**a.sources(),**own}.items()))

def literal_home(path):
 path=Path(path)
 # The exact pinned profile uses sibling src and registry roots. Check lexical
 # membership BEFORE any filesystem operation; DATA/arbitrary HOME stay barred.
 require(path.is_absolute() and '..' not in path.parts and any(path==root or root in path.parents for root in (ROOT,REGISTRY_ROOT)),
     'Literal approved source or exact pinned registry path required')
 require(not any(x.is_symlink() for x in (path,*path.parents)),'No symlink in approved source/registry route')
 return path

def lexical(path,root):
 path=Path(path);base=Path(root)
 require(path.is_absolute() and '..' not in path.parts and path!=base and base in path.parents,'Outside approved literal logical route')
 return path

def actor(profile):
 require(os.getuid()==2708899 and pwd.getpwuid(os.getuid()).pw_name==profile['slurm_user']=='luyangz','Actual Babel Root UID required before mutation/RPC')
 require(profile['site']=='babel' and tuple(profile[k] for k in ('account','partition','qos','gpu_type'))==('cx','general','normal','L40S')
     and profile['code_root']==str(ROOT),'Only installed Babel General L40S route')

def independent(value,status):
 require(value['status']==status and value.get('reviewer') not in (None,'',AUTHOR,'native_campaign_build'), 'Genuine independent launcher review required')
 return value

def metadata(request):
 """No DATA reads/stats, no fabricated native readiness or config review."""
 require(request['schema']=='root-production-student-request-v1' and request['owner']=='Root'
     and request['launcher_sources']==sources(),'Exact reviewed launcher request required')
 load=MetadataBindings(request['metadata_bindings'])
 require(request['slurm_API_evidence']['sha256']==API_SHA,'Exact actual Root Slurm API advertisement required')
 api=load(request['slurm_API_evidence']);observed=json.loads(api['stdout'])
 require(api['returncode']==0 and observed['version']['rc']==0 and observed['version']['stdout'].strip()=='slurm 26.05.1'
     and observed['help']['rc']==0 and 'requeuehold <job_id>' in observed['help']['selected'], 'API evidence absent; advertisement alone never proves atomic behavior')
 cfg=load(request['configuration']);profile=load(cfg['site_profile'])
 require(cfg['site_profile']['sha256']==PROFILE_SHA,'Exact shared natural qualification/student profile required')
 # Inventory expiry and site fields remain checked by the unchanged validator.
 a.native_config.validate_profile(profile)
 require(profile['site']=='babel' and profile['partition']=='general' and profile['gpu_type']=='L40S'
     and cfg['sources']==a.n.sources() and cfg['recipe']==a.native_config.RECIPE
     and cfg['role']=='student' and cfg['owner']=='Root','Frozen native91/recipe/student scope required')
 require(cfg['allocation']==dict(gpus=1,cpus=8,memory_mb=65536,minutes=55,reserve_seconds=300,microbatch=4,loader_workers=0)
     and cfg.get('qualification_max_steps') is None,'Exact uncapped world1 55-minute attempt required')
 lexical(cfg['output'],profile['output_root']);literal_home(cfg['report'])
 lexical(cfg['report'],profile['registry_root'])
 providers={name:load(ref) for name,ref in cfg['providers'].items()}
 permit=authorization.verify(cfg,providers,profile,request['authorization'],load=load)
 review=load(cfg['review'])
 require(review['status']=='PASS_INDEPENDENT_EXACT_SOURCE_AND_CONFIG' and review['config_identity']==a.n.run_identity(cfg)
     and review['sources']==cfg['sources'] and review.get('reviewer') not in (None,'',a.AUTHOR),'Genuine exact scientific config review required')
 plan=load(cfg['student_candidate_plan']);cells=[v for v in plan['cells'] if v['id']==cfg['id']]
 require(plan['status']=='PREPARED_LATE_BOUND_STUDENT_CANDIDATES_NOT_RELEASE_AUTHORITY' and plan['adapter_sources']==a.sources()
     and len(plan['cells'])==21 and {v['id'] for v in plan['cells']}=={a.cell_id(k,s) for k in a.KEYS for s in a.SEEDS}
     and len(cells)==1,'Actual finite21 candidate plan required')
 cell=cells[0]
 for name in ('method','fraction','training_seed','selection_seed'):
  require(cell[name]==cfg[name],'Candidate factor changed: '+name)
 require(cell['selection']==cfg['providers']['selection'] and cell['providers']['pool']==cfg['providers']['pool']
     and cell['providers']['eligibility']==cfg['providers']['eligibility']
     and cell['actual_K']==providers['selection']['count'] and cell['expected_horizon']==cfg['expected_horizon']==a.horizon(cell['actual_K']), 'Candidate selected population/horizon changed')
 storage=load(cfg['storage_reservation'])
 require(storage['status']=='PASS_EXCLUSIVE_MEASURED_STORAGE_RESERVATION' and storage['cell_id']==cfg['id']
     and storage['output']==cfg['output'] and storage['checkpoint_transaction_bytes']==a.CAPSULE_BYTES
     and storage['retained_accepted_states']==2 and storage['allow_new_intermediate_retirement'] is True,
     'Genuine collective21GiB reservation required; no idle-GPU storage inference')
 owner=providers['ownership'];function=key(dict(recipe=cfg['recipe'],pool_records=providers['pool']['records_json']['sha256'],
     selected_records=providers['selection']['records_json']['sha256'],training_seed=cfg['training_seed'],role='student',qualification_max_steps=None,replicate_site='babel'))
 require(owner['status']=='ACTIVE_EXCLUSIVE_NATIVE_RUN_CLAIM' and owner['function_identity']==function
     and owner['owner']=='Root' and owner['site_profile']==cfg['site_profile'],'Exact pre-admission immutable function claim required')
 source=independent(load(request['source_review']),'PASS_INDEPENDENT_ROOT_PRODUCTION_STUDENT_LAUNCHER_SOURCE_V4')
 require(source['runtime_sources']==sources(),'Launcher source review mismatch')
 return cfg,profile,permit,load

def validate_binding(binding,request_ref,request):
 require(binding['schema']=='root-production-student-physical-binding-v1' and binding['owner']=='Root'
     and binding['request']==request_ref and binding['configuration']==request['configuration']
     and binding['launcher_sources']==sources() and binding['adapter_sources']==a.sources()
     and str(binding['job_id']).isdecimal() and re.fullmatch('[0-9a-f]{32}',binding['nonce']), 'Foreign physical binding')
 require(desc(binding['command']['path'])==binding['command'] and binding['command']==desc(HERE/'run.sbatch'), 'Bound wrapper changed')
 require(desc(binding['spool']['path'])==binding['spool'] and Path(binding['spool']['path']).read_bytes()==(HERE/'run.sbatch').read_bytes(),'Bound reconciled spool changed')
 intent=checked(binding['intent']);ack=checked(binding['ack'])
 require(intent['request']==request_ref and intent['nonce']==binding['nonce'] and ack['intent']==binding['intent']
     and ack['job_id']==binding['job_id'],'Physical intent/ACK mismatch')
 return binding
