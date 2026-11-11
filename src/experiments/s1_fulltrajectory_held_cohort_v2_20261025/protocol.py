"""Exact six S1 designs, new physical bindings, fixed complete-source verifier."""
import re
from pathlib import Path
from experiments.root_fulltrajectory_held_cohort_v2_20261025 import protocol as rp
from experiments.s1_fulltrajectory_held_cohort_v1_20261025 import protocol as prior
c=rp.c;ncfg=rp.ncfg;ROOT=rp.ROOT;HERE=Path(__file__).resolve().parent
OWNER='Root';AUTHOR='/root/aces_fixture_metadata';SCHEMA='s1-fulltrajectory-held-request-v2'
STATUS='CONDITIONAL_HELD_ONLY_NOT_EXECUTION_AUTHORITY'
REVIEW='PASS_INDEPENDENT_EXACT_SIX_S1_HELD_REPLACEMENT_DESIGN_V2'
ACTIVATION_REVIEW='PASS_INDEPENDENT_ACTUAL_S1_FULLTRAJECTORY_JOB_ACTIVATION_V2'
COMMENT='dsmm-root-s1-fulltrajectory-v2:';REGISTRY='s1-conditional-fulltrajectory-v2'
SITES={'babel':rp.SITES['babel']};SEEDS=(17,29,43);ARMS=('0R','H0');GATES=prior.GATES
OLD_JOBS={'s1-student-v1-10-0R-s17':'10926092','s1-student-v1-10-0R-s29':'10926093','s1-student-v1-10-0R-s43':'10926094',
 's1-student-v1-10-H0-s17':'10926095','s1-student-v1-10-H0-s29':'10926096','s1-student-v1-10-H0-s43':'10926097'}
actor=rp.actor

def sources():
 own={str(f.relative_to(ROOT)):c.sha(f) for f in HERE.iterdir() if f.suffix in ('.py','.sbatch') and not f.name.startswith('test_')}
 return dict(sorted({**prior.sources(),**rp.sources(),**own}.items()))

def bare(v):return {k:x for k,x in v.items() if k!='review'}
def independent(v,status):
 c.require(v['status']==status and isinstance(v.get('reviewer'),str) and v['reviewer'].strip()
     and v['reviewer'] not in (AUTHOR,OWNER,'Session1'),'Genuine independent exact source/config decision required')
 return v

def predecessor(request,load=c.checked):
 old,science,_=prior.reviewed(request['original_request'],load)
 ident=request['cell_id'];c.require(ident in OLD_JOBS and old['cell_id']==ident and old['site']=='babel'
     and old['arm'] in ARMS and old['fraction']=='0.10' and old['training_seed'] in SEEDS,
     'Only six actual paid0R/H0 Babel cells; no other arm/site/budget')
 b=load(request['old_binding']);c.require(b['schema']=='s1-fulltrajectory-held-binding-v1' and b['owner']=='Session1'
     and b['job_id']==OLD_JOBS[ident] and b['request']==request['original_request'] and b['sources']==prior.sources()
     and b['command']==c.desc(prior.HERE/'run.sbatch') and re.fullmatch('[0-9a-f]{32}',b['nonce'])
     and request['old_binding']['path']==str(Path(old['control_root'])/'BINDING.json'),
     'Exact original physical binding must be authenticated, not inferred from job order')
 # Root publishes this only AFTER genuine safe cancellation of the never-started
 # old scalar. The raw scheduler records remain in the immutable receipt.
 retire=load(request['predecessor_retirement']);before=retire['before_scontrol'];after=retire['after_sacct']
 c.require(retire['status']=='ROOT_RECONCILED_EXACT_NEVER_STARTED_S1_REPLACEMENT_V2' and retire['owner']=='Root'
     and retire['old_binding']==request['old_binding'] and retire['cell_id']==ident
     and retire['scientific_design']==science and retire['no_live_duplicate_function'] is True
     and retire['old_sources_and_all_paid_selections_preserved'] is True,
     'Actual Root predecessor retirement/identity reconciliation required before new admission')
 c.require(before['returncode']==after['returncode']==0,'Successful actual before/after scheduler observations required')
 lines=[x for x in before['stdout'].splitlines() if x.strip()];c.require(len(lines)==1,'One original pending scalar')
 f=dict(re.findall(r'(?:^|\s)([^\s=]+)=(\S+)',lines[0]))
 expected=dict(JobId=b['job_id'],JobState='PENDING',Reason='JobHeldUser',UserId='luyangz(2708899)',Account='cx',Partition='general',QOS='normal',
     NumCPUs='8',NumTasks='1',TimeLimit='12:00:00',Restarts='0',RunTime='00:00:00',StartTime='Unknown',
     Comment=prior.COMMENT+b['nonce'],Command=b['command']['path'],WorkDir=str(ROOT),TresPerNode='gres/gpu:L40S:1')
 c.require(all(f.get(k)==v for k,v in expected.items()) and f.get('ArrayJobId') in (None,'(null)')
     and f.get('AllocTRES') in (None,'(null)','None'),'Only genuinely never-started exact owned predecessor may be replaced')
 rows=[x.split('|') for x in after['stdout'].splitlines() if x.strip()]
 c.require(len(rows)==1 and len(rows[0]) in (4,5) and (len(rows[0])==4 or rows[0][-1]=='') and rows[0][0]==b['job_id'] and rows[0][1].split()[0].rstrip('+')=='CANCELLED'
     and re.fullmatch(r'\d+:\d+',rows[0][2]) and rows[0][3]=='0','Actual terminal cancelled zero-elapsed predecessor required; no live duplicate')
 return old,science,b

def design(request,load=c.checked,admission=False):
 c.require(request['schema']==SCHEMA and request['owner']=='Root' and request['status']==STATUS and request['site']=='babel'
     and request['sources']==sources() and request['native_sources']==c.sources() and request['recipe']==ncfg.RECIPE,
     'Exact additive S1 held design/source only')
 old,science,b=predecessor(request,load)
 c.require(request['scientific_design']==science and request['selection_receipt']==old['selection_receipt']
     and request['paired_root_requests']==old['paired_root_requests'] and request['future_release_gates']==GATES
     and request['release_authorized'] is False and request['reserved_storage_bytes']==0,
     'Immutable paid selection/scientific factors/control pairing/gates must remain exact')
 profile=rp.validate_historical_profile(load(request['site_profile']),load)
 if admission:ncfg.validate_profile(profile)
 c.require(profile['site']=='babel' and profile['code_root']==str(ROOT)
     and tuple(profile[k] for k in ('slurm_user','account','partition','qos','gpu_type'))==('luyangz','cx','general','normal','L40S')
     and request['outer_allocation']==dict(gpus=1,cpus=8,memory_mb=65536,minutes=720)
     and profile['max_minutes']>=720 and profile['max_cpus_per_request']>=8 and profile['max_memory_mb']>=65536,
     'Current compatible GeneralL40S12h CPU8/64GiB profile only')
 path=Path(request['control_root']);c.require(path.parent==Path(profile['registry_root'])/REGISTRY and path.name==request['cell_id']
     and path.is_absolute() and '..' not in path.parts and not any(x.is_symlink() for x in (path,*path.parents)),
     'New literal prospective S1V2 control namespace; never overwrite predecessor')
 return science,profile

def reviewed(request_ref,load=c.checked,admission=False):
 request=load(request_ref);science,profile=design(request,load,admission)
 review=independent(load(request['review']),REVIEW)
 c.require(review['request_identity']==c.key(bare(request)) and review['sources']==sources()
     and review['native_sources']==c.sources() and review['held_only'] is True,'Independent exact held replacement review required')
 return request,science,profile

def activation(binding_ref,activation_ref,load=c.checked):
 b=load(binding_ref);request,science,_=reviewed(b['request'],load);a=load(activation_ref)
 c.require(a['schema']=='s1-fulltrajectory-actual-activation-v2' and a['binding']==binding_ref
     and a['scientific_design']==science and a['held_sources']==sources(),'Exact new S1 physical activation')
 review=independent(load(a['review']),ACTIVATION_REVIEW)
 c.require(review['activation_identity']==c.key(bare(a)) and review['binding']==binding_ref
     and review['target_sources']==a['target_sources'] and review['native_sources']==c.sources()
     and review['actual_gate_descriptors']==a['gates'],'Exact independent actual source/config/selection gate review')
 c.require(set(a['gates'])==set(GATES) and all(a['gates'].values()),'All seven genuine S1 runtime/input/selection gates required')
 for ref in a['gates'].values():load(ref)
 cfg=load(a['configuration']);profile=ncfg.validate_profile(load(cfg['site_profile']))
 c.require(cfg['owner']=='Root' and cfg['role']=='student' and cfg['id']==request['cell_id']
     and cfg['sources']==c.sources() and cfg['recipe']==ncfg.RECIPE and cfg.get('qualification_max_steps') is None,
     'Native91 shared recipe and own target identity required')
 for field in ('method','fraction','training_seed','selection_seed'):c.require(cfg[field]==science[field],'Scientific target changed '+field)
 c.require(all(cfg['allocation'][k]==v for k,v in request['outer_allocation'].items()) and profile['site']=='babel',
     'Actual new native allocation/site differs')
 logical=a['logical_input_registration']
 c.require(logical==dict(original_pool=science['logical_pool'],original_selection=science['selection'],
     actual_pool=cfg['providers']['pool'],actual_selection=cfg['providers']['selection'])
     and cfg['providers']['pool']==science['logical_pool'] and cfg['providers']['selection']==science['selection'],
     'Exact already-paid logical pool/selection; no regeneration or relabeling')
 target=a['target_module'];c.require(re.fullmatch(r'experiments\.root_s1_[a-z0-9_]+\.[a-z0-9_]+',target)
     and a['target_callable']=='run_conditional_student','Only separately reviewed S1 runtime entrypoint')
 rp.verify_target_sources(a['target_sources'],target)
 return b,request,profile,a
