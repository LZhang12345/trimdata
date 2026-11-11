"""New explicit runtime/epoch contract; unchanged Native7 is only its core."""
from pathlib import Path
from experiments.root_native_campaign_v7_20261024 import common as n
from experiments.root_native_deterministic_replication_v3_20261024 import protocol as prior
from experiments.root_native_operator_trace_v4_20261024 import protocol as oldaces
ROOT=n.ROOT;HERE=Path(__file__).resolve().parent;PACKAGE='experiments.root_production_epoch_qualification_v1_20261024'
CAPSULE_CAP=5*(1<<30);TRANSIENT_RESERVE=6*(1<<30);MIN_COLLECTIVE=3*CAPSULE_CAP+TRANSIENT_RESERVE
AUTHOR_IDENTITY='/root/native_campaign_build';TOLERANCE={'absolute':1e-6,'relative':1e-5}
require=n.require;checked=n.checked;seal=n.seal;read=n.read;desc=n.desc;key=n.key;sha=n.sha;canonical=n.canonical;now=n.now
TARGET='a704d9716719a9ad2a0cd0890c24fec35c15bd085b7c2c98fe38ea0c98961830'
LOCKS=dict(babel='/home/luyangz/data_selection_multimodal/collaboration/babel-two-track-20261019/ADMISSION.lock',aces='/scratch/user/u.lz341613/dsmm_root_initial24_aces_20261023/ADMISSION.lock')

def sources():
 own={str(f.relative_to(ROOT)):sha(f) for f in HERE.iterdir() if f.suffix in ('.py','.sbatch') and not f.name.startswith('test_')}
 # The live tokenizer bridge imports the measured V5 package; S1 label helpers
 # import V3 metadata. Bind these concrete paths too, not just their version.
 extra=list((ROOT/'experiments/root_native_campaign_v5_20261024').glob('*.py'))+[ROOT/'experiments/root_native_campaign_v5_20261024/zero3.json']
 extra += [ROOT/'experiments/root_native_campaign_v3_20261023'/name for name in ('__init__.py','common.py','selection.py')]
 extra += [ROOT/'experiments/root_native_campaign_20261023/__init__.py']
 own.update({str(f.relative_to(ROOT)):sha(f) for f in extra if not f.name.startswith('test_')})
 return dict(sorted({**prior.sources(),**oldaces.sources(),**own}.items()))
def identifier(site,seed):
 require(site in ('babel','aces') and seed in (17,29,43),'Finite six registered site/seed units only');return 'production-epoch-qual-v1-'+site+'-s'+str(seed)
def specification(site):
 require(site in ('babel','aces'),'Exact declared site')
 return dict(site=site,count=1153 if site=='babel' else 1024,total_steps=9 if site=='babel' else 8,cut=4,natural_epoch=site=='babel',runtime='explicit_xformers_cutlass_fw_bw_deterministic_v1',CUBLAS_WORKSPACE_CONFIG=':4096:8',gpus=1,cpus=8,memory_mb=65536,minutes=55,reserve_seconds=300,microbatch=4,loader_workers=0)
def bare(value):return {k:v for k,v in value.items() if k!='review'}
def native_identity(cfg,modules):return modules['common'].run_identity(cfg)
def lexical_child(path,root):
 value=Path(path);base=Path(root)
 return value.is_absolute() and base.is_absolute() and '..' not in value.parts and value!=base and value.is_relative_to(base)

def validate_study(study,*,reviewed=True,load=checked):
 spec=specification(study['site']);require(study['schema']=='declared-production-epoch-qualification-v1' and study['owner']=='Root' and study['id']==identifier(study['site'],study['seed']) and study['specification']==spec and study['driver_sources']==sources() and study['numeric_tolerance']==TOLERANCE,'One exact registered finite adapter/source/specification required')
 require(study['collective_reserved_bytes']>=MIN_COLLECTIVE,'Three bounded capsules plus transient must fit measured collective reservation')
 require(study['total_steps']==spec['total_steps'] and study['interrupt_step']==4 and study['cell']=='D8','Exact paired horizon/cut policy')
 modules,packet=prior.b.target_modules(study['native_target']);require(study['native_target']['source_packet']['sha256']==TARGET and len(packet['sources'])==91,'Only unchanged Native7/91 core')
 configs={arm:load(study['native_configurations'][arm]) for arm in ('reference','resume')};base=configs['reference']
 for arm,cfg in configs.items():
  require(cfg['sources']==packet['sources']==n.sources() and cfg['recipe']==modules['config'].RECIPE and cfg['role']=='qualify' and cfg['method']=='full' and cfg['fraction']=='1' and cfg['training_seed']==study['seed'] and cfg['selection_seed']==1234 and cfg['qualification_max_steps']==spec['total_steps'],'Unchanged recipe/full fixture/seed and declared helper cap only')
  require(cfg['execution_adapter']==dict(package=PACKAGE,sources=sources(),specification=spec) and cfg['id']==study['id']+'-'+arm,'New execution adapter must be explicitly bound in native identity')
  require(cfg['allocation']=={k:spec[k] for k in ('gpus','cpus','memory_mb','minutes','reserve_seconds','microbatch','loader_workers')} and cfg['checkpoint_seconds']>3300,'Original world1/micro4/GA32, finite55min, no periodic rotation')
  profile_value=load(cfg['site_profile']);inventory_path=Path(profile_value['inventory']['path']);require(str(inventory_path).startswith(('/home/','/scratch/user/')) and not any(x.is_symlink() for x in (inventory_path,*inventory_path.parents)),'Inventory must be literal HOME/scratch metadata');profile=modules['config'].validate_profile(profile_value);require(profile['site']==study['site'] and profile['code_root']==str(ROOT),'Exact current site profile/source root')
  require(lexical_child(cfg['output'],profile['output_root']) and lexical_child(cfg['report'],profile['registry_root']) and lexical_child(study['report'],profile['registry_root']),'Approved lexical output/registry routes; physical DATA checks stay inside GPU')
  route=dict(babel=('luyangz','cx','general','normal','L40S'),aces=('u.lz341613','158416130568','gpu','normal','h100'))[study['site']]
  require(tuple(profile[k] for k in ('slurm_user','account','partition','qos','gpu_type'))==route,'Wrong advertised site/account/General route')
  pool=load(cfg['providers']['pool']);require(pool['schema']=='native-engineering-pool-v1' and pool['engineering_only'] is True and pool['count']==spec['count'],'Separate truthful engineering fixture scope')
  for name in ('assets','environment','pool','selection','eligibility'):require(cfg['providers'][name]==base['providers'][name],'Paired input provider mismatch '+name)
  require(cfg['site_profile']==base['site_profile'],'Paired hardware differs')
  if study['site']=='babel':
   from .fixture import COUNTS
   require(pool['source_counts']==COUNTS and pool['parent_pool']['sha256']=='c76040a17ca0dc1e99fea14514618edade3171a9b05f6ccc3e4751a5939e4964','Actual all-six-source derived1153 fixture required')
  else:require(pool['records_json']['sha256']=='3cf04ac5cb0b74c39de779ef94250bb4a3fb7d27fc09eb36ff6f16d4a9db3e90' and 'media_transport' in cfg['providers'],'ACES preserves real1024 original fixture/transport, no broad-source claim')
  review=load(cfg['review']);require(review.get('reviewer') not in (None,AUTHOR_IDENTITY,'native_campaign_build') and review['status']=='PASS_INDEPENDENT_EXACT_SOURCE_AND_CONFIG' and review['sources']==cfg['sources'] and review['config_identity']==n.run_identity(cfg),'Genuine exact config review, no synthesized READY/plan branch')
  owner=load(cfg['providers']['ownership']);require(owner['actual_held_job_id']==study['job_id'] and owner['qualification_study']==study['id'] and owner['qualification_arm']==arm and owner['cell_id']==cfg['id'] and owner['owner']=='Root','Current paired native ownership required')
  storage=load(cfg['storage_reservation']);require(storage['status']=='PASS_EXCLUSIVE_MEASURED_STORAGE_RESERVATION' and storage['owner']=='Root' and storage['output']==cfg['output'] and storage['cell_id']==cfg['id'] and storage['retained_accepted_states']==(1 if arm=='reference' else 2) and storage['allow_new_intermediate_retirement'] is False and storage['checkpoint_transaction_bytes']==CAPSULE_CAP and storage['transient_reserve_bytes']==TRANSIENT_RESERVE,'Genuine three-capsule hardcaps+transient collective reservation, no intermediate pruning')
 if reviewed:
  review=load(study['review']);require(review['status']=='PASS_INDEPENDENT_DECLARED_EPOCH_RUNTIME_STUDY' and review['study_identity']==key(bare(study)) and review['driver_sources']==sources() and review['reviewer']!=AUTHOR_IDENTITY,'Genuine independently reviewed actual study required')
 return configs,modules

def validate_study_metadata(study,bindings):
 from .metadata import MetadataBindings
 return validate_study(study,load=MetadataBindings(bindings))
