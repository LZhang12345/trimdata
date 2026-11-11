"""Immutable scientific design and independently reviewed future activation."""
import ast,importlib,json,os,pwd,re
from pathlib import Path
from experiments.root_native_campaign_v7_20261024 import common as c,config as ncfg
ROOT=c.ROOT;HERE=Path(__file__).resolve().parent;AUTHOR='/root/native_campaign_build'
SEEDS=(17,29,43);BUDGETS={5:33038,10:66076,20:132152};BABEL_MINUTES={5:360,10:720,20:1440};ACES_MINUTES={5:300,10:600,20:1200}
ORIGINAL_SHA='2c077d3db2268945dc4b2722d2f4c833e15636407c4f6bcb3626e8e694e2f205'
REVIEW='PASS_INDEPENDENT_CONDITIONAL_FULLTRAJECTORY_HELD_DESIGN_V2'
SITES={'babel':dict(uid=2708899,user='luyangz',account='cx',partition='general',qos='normal',gpu='L40S',limit=50,lock='/home/luyangz/data_selection_multimodal/collaboration/babel-two-track-20261019/ADMISSION.lock'),
 'aces':dict(uid=56219,user='u.lz341613',account='158416130568',partition='gpu',qos='normal',gpu='h100',limit=90,lock='/scratch/user/u.lz341613/dsmm_root_initial24_aces_20261023/ADMISSION.lock')}
NONEXECUTABLE_SOURCES={
 'collaboration/published-foundation-20261022/reference_sources/LLaVA/c121f0432da27facab705978f83c4ada465e46fd/pyproject.toml':'b71f686721b00c97360cc9def0311d3869d45cb13f2fe51885f94d5656440d54',
 'experiments/root_native_campaign_v7_20261024/zero3.json':'b415e5999d186e3da007512b9502da78248ac4842dd6c43ffde56ff5e3acaae5',
 'experiments/root_native_campaign_v5_20261024/zero3.json':'b415e5999d186e3da007512b9502da78248ac4842dd6c43ffde56ff5e3acaae5'}

def sources():return {str(f.relative_to(ROOT)):c.sha(f) for f in sorted(HERE.iterdir()) if f.suffix in ('.py','.sbatch') and not f.name.startswith('test_')}
def actor(site):
 s=SITES[site];c.require(os.getuid()==s['uid'] and pwd.getpwuid(os.getuid()).pw_name==s['user'],'Actual Root site UID required before writes/RPC')
def independent(value,status):
 c.require(value['status']==status and isinstance(value.get('reviewer'),str) and value['reviewer'].strip() and value['reviewer'] not in (AUTHOR,'native_campaign_build'),'Genuine independent exact design/review required');return value

def scheduler_minutes(evidence,site):
 c.require(evidence['returncode']==0,'Actual scheduler evidence failed');v=json.loads(evidence['stdout']);part=v['partitions']
 c.require(part['returncode']==0,'Actual partition query failed')
 rows=part['stdout'].replace('\\n','\n').splitlines();rows=[x for x in rows if x.startswith('PartitionName='+SITES[site]['partition']+' ')]
 c.require(len(rows)==1,'Exact advertised target partition required');f=dict(re.findall(r'(?:^|\s)([^\s=]+)=(\S+)',rows[0]))
 c.require(f['State']=='UP' and f['MaxTime']=='2-00:00:00','Reviewed48h site ceiling changed');return 2880

def validate_historical_profile(profile,load=c.checked):
 """Frozen route ancestry, not a current runtime/readiness assertion."""
 c.require(profile['schema']=='native-site-profile-v1','Historical native profile schema')
 inventory=load(profile['inventory'])
 c.require(inventory['status']=='PASS_CURRENT_SITE_PATHS_AND_SCHEDULER' and inventory['site']==profile['site'],
     'Exact historical independently observed inventory anchor required')
 for name in ('partition','qos','account','gpu_type','slurm_user','python','cache_root','code_root','output_root','registry_root'):
  c.require(isinstance(profile[name],str) and profile[name] and '<' not in profile[name] and inventory[name]==profile[name],
      'Historical profile/inventory route differs: '+name)
 for name in ('python','cache_root','code_root','output_root','registry_root'):
  c.require(Path(profile[name]).is_absolute(),'Historical site paths must be absolute')
 for name in ('max_minutes','max_gpus_per_node','max_cpus_per_request','max_memory_mb'):
  c.require(type(profile[name]) is int and profile[name]>0 and inventory[name]==profile[name],
      'Historical profile/inventory resource bounds differ')
 c.require(isinstance(inventory['valid_until'],str) and inventory['valid_until'],'Preserve actual historical expiry, never normalize')
 return profile

def design(request,load=c.checked,admission=False):
 c.require(request['schema']=='root-fulltrajectory-held-request-v2' and request['owner']=='Root' and request['status']=='CONDITIONAL_HELD_ONLY_NOT_EXECUTION_AUTHORITY'
     and request['sources']==sources() and request['native_sources']==c.sources() and request['recipe']==ncfg.RECIPE,'Exact prospective held design/source required')
 site=request['site'];c.require(site in SITES,'Registered site only');parent=load(request['parent_candidate_plan'])
 c.require(request['parent_candidate_plan']['sha256']==ORIGINAL_SHA and parent['owner']=='Root' and parent['eligible_N']==660763
     and parent['status']=='PREPARED_LATE_BOUND_STUDENT_CANDIDATES_NOT_RELEASE_AUTHORITY' and parent['native_sources']==c.sources()
     and len(parent['cells'])==21,'Actual immutable10923033 logical candidate21 required')
 matches=[v for v in parent['cells'] if v['id']==request['original_cell_id']];c.require(len(matches)==1,'One exact paid candidate required');cell=matches[0]
 percent=int(cell['selection_key'].split(':')[0]) if cell['selection_key']!='full' else 100
 c.require(percent in BUDGETS and cell['method'] in ('uniform','source_sqrt') and cell['training_seed'] in SEEDS
     and cell['actual_K']==cell['requested_K']==BUDGETS[percent] and cell['selection_seed']==1234,'Only existing exact18 full-conversation designs')
 c.require(not(site=='babel' and cell['selection_key']=='5:uniform' and cell['training_seed']==17),'Active canonical trajectory must be reused, never duplicated')
 expected=dict(method=cell['method'],fraction=cell['fraction'],training_seed=cell['training_seed'],selection_seed=1234,
     selected_count=cell['actual_K'],selection=cell['selection'],logical_pool=cell['providers']['pool'],logical_eligibility=cell['providers']['eligibility'],
     epoch=1,max_steps=-1,global_batch=128,replicate_site=site)
 c.require(request['scientific_design']==expected,'Scientific factors or paid selected population changed')
 profile=validate_historical_profile(load(request['site_profile']),load)
 if admission:ncfg.validate_profile(profile)
 c.require(profile['site']==site and profile['code_root']==str(ROOT),'Authentic current source/site profile required')
 route=SITES[site];c.require((profile['slurm_user'],profile['account'],profile['partition'],profile['qos'],profile['gpu_type'])==tuple(route[k] for k in ('user','account','partition','qos','gpu')),'Route differs from actual profile')
 maximum=scheduler_minutes(load(request['scheduler_time_evidence']),site);allocation=request['outer_allocation']
 c.require(allocation['gpus']==1 and allocation['cpus']==8 and allocation['memory_mb']==65536 and 0<allocation['minutes']<=maximum,'Finite outer8CPU64G world1 route required')
 timing=load(request['timing_basis'])
 c.require(timing['site']==site and timing['status']=='MEASURED_QUALIFIER_RATE_EXTRAPOLATION_NOT_FULL_STUDENT_ETA'
     and timing['source_evidence'] and timing['minutes_by_percent'][str(percent)]==allocation['minutes']
     and timing['no_padding_to_GPU_hour_target'] is True,'Own-site measured timing rationale required')
 for ref in timing['source_evidence']:
  c.require(Path(ref['path']).suffix=='.json' and not ref['path'].startswith('/data/') and ref['bytes']<=1<<20,'Bounded HOME/scratch timing metadata only');load(ref)
 c.require(allocation['minutes']==(BABEL_MINUTES if site=='babel' else ACES_MINUTES)[percent],'Reviewed own-site useful fulltrajectory wall envelopes only')
 c.require(request['future_release_gates']==['actual_matching_natural_runtime','actual_full_source_student_inputs','actual_full_selected_canonical_pilot','reviewed_longwall_runtime_entrypoint','actual_collective_storage_reservation','exact_config_and_outer_permit_reviews']
     and request['release_authorized'] is False and request['reserved_storage_bytes']==0,'Held work has no fabricated future gate or storage reservation')
 path=Path(request['control_root']);registry=Path(profile['registry_root'])
 c.require(path.is_absolute() and '..' not in path.parts and registry in path.parents
     and not any(x.is_symlink() for x in (path,*path.parents)),'Literal approved registry control route required')
 return cell,profile

def reviewed(request_ref,load=c.checked,admission=False):
 request=load(request_ref);cell,profile=design(request,load,admission);review=independent(load(request['review']),REVIEW)
 c.require(review['request_identity']==c.key({k:v for k,v in request.items() if k!='review'}) and review['sources']==sources()
     and review['native_sources']==c.sources() and review['held_only'] is True,'Exact independent held design/source review mismatch')
 return request,cell,profile

def activation(binding_ref,activation_ref,load=c.checked):
 """No activation exists at held admission. All later leaves are real reviews."""
 b=load(binding_ref);request,cell,profile=reviewed(b['request'],load);a=load(activation_ref)
 c.require(a['schema']=='root-fulltrajectory-actual-activation-v2' and a['binding']==binding_ref
     and a['scientific_design']==request['scientific_design'] and a['held_sources']==sources(), 'Foreign physical/scientific activation')
 review=independent(load(a['review']),'PASS_INDEPENDENT_ACTUAL_FULLTRAJECTORY_JOB_ACTIVATION_V2')
 c.require(review['activation_identity']==c.key({k:v for k,v in a.items() if k!='review'}) and review['binding']==binding_ref
     and review['target_sources']==a['target_sources'] and review['native_sources']==c.sources() and review['actual_gate_descriptors']==a['gates'],'Actual job/source/config/gates must be independently reviewed')
 c.require(set(a['gates'])==set(request['future_release_gates']) and all(a['gates'].values()),'All actual gates must be present; no future placeholders')
 for ref in a['gates'].values():load(ref)
 cfg=load(a['configuration']);runtime_profile=ncfg.validate_profile(load(cfg['site_profile']))
 route=SITES[request['site']]
 c.require(runtime_profile['site']==request['site'] and runtime_profile['code_root']==str(ROOT)
     and tuple(runtime_profile[k] for k in ('slurm_user','account','partition','qos','gpu_type'))==tuple(route[k] for k in ('user','account','partition','qos','gpu'))
     and runtime_profile['max_minutes']>=request['outer_allocation']['minutes'] and runtime_profile['max_cpus_per_request']>=8
     and runtime_profile['max_memory_mb']>=65536 and runtime_profile['max_gpus_per_node']>=1,
     'Fresh actual activation profile must authorize this longwall/runtime route')
 c.require(cfg['sources']==c.sources() and cfg['recipe']==ncfg.RECIPE and cfg['owner']=='Root'
     and cfg['role']=='student' and cfg['method']==cell['method'] and cfg['fraction']==cell['fraction']
     and cfg['training_seed']==cell['training_seed'] and cfg['selection_seed']==1234 and cfg.get('qualification_max_steps') is None,
     'Actual config changed immutable native scientific design')
 c.require(cfg['allocation']['minutes']==request['outer_allocation']['minutes'] and cfg['allocation']['gpus']==1
     and cfg['allocation']['cpus']==8 and cfg['allocation']['memory_mb']==65536,'Actual config/outer allocation mismatch')
 logical=a['logical_input_registration'];c.require(logical['original_pool']==cell['providers']['pool'] and logical['original_selection']==cell['selection']
     and logical['actual_pool']==cfg['providers']['pool'] and logical['actual_selection']==cfg['providers']['selection'],
     'Explicit original-to-current full source/selection binding required')
 target=a['target_module'];c.require(re.fullmatch(r'experiments\.root_[a-z0-9_]+\.[a-z0-9_]+',target)
     and target.replace('.','/')+'.py' in a['target_sources'] and a['target_callable']=='run_conditional_student',
     'Explicit separately reviewed complete runtime entrypoint required')
 verify_target_sources(a['target_sources'],target);return b,request,runtime_profile,a


def verify_target_sources(packet,target):
 """Verify a larger independently reviewed closure without replacing Native91."""
 c.require(all(packet.get(k)==v for k,v in c.sources().items()),'Future target must preserve every exact Native91 source')
 for name,value in packet.items():
  path=Path(name)
  permitted=path.suffix in ('.py','.sbatch') or NONEXECUTABLE_SOURCES.get(name)==value
  c.require(not path.is_absolute() and '..' not in path.parts and permitted
      and not any(x.is_symlink() for x in (ROOT/path,*(ROOT/path).parents))
      and c.sha(ROOT/path)==value,'Future target source bytes/path changed')
 parts=target.split('.')
 for i in range(1,len(parts)):
  init=Path(*parts[:i])/'__init__.py'
  if (ROOT/init).is_file():c.require(str(init) in packet,'Imported package initializer missing from target closure')
 path=target.replace('.','/')+'.py';c.require(path in packet,'Actual target module absent')
 tree=ast.parse((ROOT/path).read_text())
 c.require(any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='run_conditional_student' for n in tree.body),
     'Actual declared entrypoint absent from source')
 return path
