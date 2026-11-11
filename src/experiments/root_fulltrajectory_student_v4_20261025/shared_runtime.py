"""Exact per-target compatibility; original observations keep their real identities.

This module is authorization metadata, not a substitute for Native91 input,
claim, cold-device, positive-target or accepted-state checks.
"""
from pathlib import Path
from . import protocol as p
from experiments.root_fulltrajectory_student_v2_20261025 import protocol as old
from experiments.root_fulltrajectory_held_cohort_v2_20261025.protocol import validate_historical_profile
BASE='/home/luyangz/data_selection_multimodal/'
ANCHOR_READOUT=dict(path=BASE+'native_campaign_v3_20261023/registry/production-students/production-student-v1-5-source_sqrt-s17-longwall-v2/ROOT_LONG/ACTUAL_LONGWALL_QUALIFICATION_V1.json',bytes=44505,sha256='1b01f8f87459d8542914dc84c4a7bb278b5cd838595d721fbae9c15cdd3bb1ec')
ANCHOR_REVIEW=dict(path=BASE+'collaboration/root-held-release-audit-20261025/continued-release-v1/INDEPENDENT_ACTUAL_LONGWALL_RUNTIME_REVIEW_V1.json',bytes=45219,sha256='afd7406e7bee8f4b273cf555ceb11a06d402306fed6e29cbb53785fdce5471ee')
ANCHOR_CONFIG=dict(path=BASE+'native_campaign_v3_20261023/registry/production-students/production-student-v1-5-source_sqrt-s17-longwall-v2/CONFIG.json',bytes=57107,sha256='57cb66a3b8c1827fd8f54dcb4aea0b296639d8e396abfeaff4535929479bc992')
ANCHOR_ID='ae1ddc3f51470edb799322c0deb9707d8e7c49cb6faa970c08214bcd9e039b60'
BRIDGE_STATUS='PASS_INDEPENDENT_BABEL_SHARED_NUMERICAL_RUNTIME_SOURCE_BRIDGE_V4'
CERT_STATUS='PASS_INDEPENDENT_EXACT_BABEL_SHARED_RUNTIME_TARGET_V4'
PROFILE_FIELDS=('site','partition','qos','account','gpu_type','slurm_user','python','cache_root','code_root','registry_root')
NUMERICAL_FILES=('observer.py','state_audit.py','capsule.py','student_config.py')
DELTA_FIELDS=('id','method','fraction','training_seed','site_profile','output','report','providers.selection','providers.ownership','providers.qualification','storage_reservation','review','execution_adapter','student_candidate_plan','expected_horizon','runtime_compatibility','allocation.minutes')

def reviewer(value):
 p.require(isinstance(value.get('reviewer'),str) and value['reviewer'].strip()
     and value['reviewer'] not in (p.AUTHOR,'Root','storage_scope_review'), 'Genuine independent named reviewer required')

def numerical_bodies():
 result={}
 for name in NUMERICAL_FILES:
  p.require((p.HERE/name).read_bytes()==(old.HERE/name).read_bytes(),'Changed qualified numerical/observer body '+name)
  result[name]=p.sha(p.HERE/name)
 return result

def source_bridge(ref,load=p.checked):
 v=load(ref);reviewer(v)
 p.require(v['status']==BRIDGE_STATUS and v['anchor_sources']==old.sources()
     and v['target_sources']==p.sources() and v['native_sources']==p.n.sources()
     and v['numerical_bodies']==numerical_bodies() and v['runtime']==p.RUNTIME
     and v['allow_old_checkpoint_identity_migration'] is False and v['natural_runtime_seed_policy']==p.SEED_POLICY,
     'Exact independent additive source bridge required; no identity migration')
 return v

def anchor(load=p.checked):
 q=load(ANCHOR_READOUT);review=load(ANCHOR_REVIEW);cfg=load(ANCHOR_CONFIG)
 reviewer(review)
 p.require(q['status']=='ACTUAL_LONGWALL_RESTORE_AND_SEGMENTS_REQUIRES_INDEPENDENT_REVIEW'
     and q['configuration']==ANCHOR_CONFIG and q['config_identity']==ANCHOR_ID==p.n.run_identity(cfg)
     and q['sources']==old.sources() and q['actual_same_job_restore'] is True
     and q['accepted_prefix_end']==8 and q['actual_records']==1024 and q['max_steps']==-1
     and len(q['accepted_segments'])==2 and q['expected_natural_horizon']==p.horizon(33038)
     and review['status']=='PASS_INDEPENDENT_ACTUAL_LONGWALL_STUDENT_RUNTIME'
     and review['readout']==ANCHOR_READOUT and review['configuration']==ANCHOR_CONFIG
     and review['binding']==q['binding'] and review['config_identity']==ANCHOR_ID
     and review['sources']==old.sources() and review['actual_same_job_restore'] is True
     and review['accepted_prefix_end']==8 and review['actual_records']==1024,
     'Only the genuine original Babel0-to1-to8 readout and review anchor this class')
 return cfg,q,review

def invariants(cfg,load=p.checked,historical=False):
 profile=(validate_historical_profile(load(cfg['site_profile']),load) if historical
     else p.native_config.validate_profile(load(cfg['site_profile'])))
 p.require(cfg['sources']==p.n.sources() and cfg['recipe']==p.native_config.RECIPE
     and cfg['official_commit']==p.n.PIN and cfg['owner']=='Root' and cfg['role']=='student'
     and cfg['replication_block']==dict(replicate_site='babel')
     and cfg['allocation']['gpus']==1 and cfg['allocation']['cpus']==8
     and cfg['allocation']['memory_mb']==65536 and cfg['allocation']['microbatch']==4
     and cfg['allocation']['loader_workers']==0 and cfg['allocation']['reserve_seconds']==300
     and cfg['allocation']['minutes'] in (360,720,1440) and cfg['checkpoint_seconds']==1200
     and cfg.get('qualification_max_steps') is None, 'Outside qualified numerical/runtime allocation class')
 p.require(profile['site']=='babel' and profile['partition']=='general' and profile['gpu_type']=='L40S'
     and profile['cache_root']=='/data/hf_cache'
     and profile['max_minutes']>=cfg['allocation']['minutes'] and profile['max_gpus_per_node']>=1
     and profile['max_cpus_per_request']>=8 and profile['max_memory_mb']>=65536,
     'Babel General L40S class/current resource envelope only; no ACES transfer')
 return dict(natural_runtime_seed_policy=p.SEED_POLICY,native_sources=cfg['sources'],official_commit=cfg['official_commit'],recipe=cfg['recipe'],runtime=p.RUNTIME,
     profile={k:profile[k] for k in PROFILE_FIELDS},allocation={k:v for k,v in cfg['allocation'].items() if k!='minutes'},
     checkpoint_seconds=cfg['checkpoint_seconds'],providers={k:cfg['providers'][k] for k in ('pool','eligibility','environment','assets')},
     replication_block=cfg['replication_block'],native_cli=dict(max_steps=-1,epochs=1,model_max_length=2048),
     numerical_bodies=numerical_bodies())

def target(cfg,configuration,binding_ref,load=p.checked):
 p.supported(cfg);profile=p.native_config.validate_profile(load(cfg['site_profile']))
 p.require(profile['output_root']==p.CHECKPOINT_ROOT and p.n.within(cfg['output'],p.CHECKPOINT_ROOT)
     and cfg['output']!=p.CHECKPOINT_ROOT and p.n.within(cfg['report'],profile['registry_root']),
     'Fresh targets use only the explicitly approved shared project checkpoint namespace')
 out=Path(cfg['output']);root=Path(p.CHECKPOINT_ROOT)
 p.require(out.is_absolute() and '..' not in out.parts and root in out.parents
     and not any(path.is_symlink() for path in (out,*out.parents)),
     'Literal non-symlink owned shared checkpoint path required')
 selection=load(cfg['providers']['selection']);pool=load(cfg['providers']['pool'])
 expected_count={'0.05':33038,'0.10':66076,'0.20':132152}[cfg['fraction']]
 p.require(pool['count']==660763 and pool['pool_kind']=='all_source_protected_bridge'
     and selection['count']==expected_count and len(selection['record_ids'])==expected_count
     and len(set(selection['record_ids']))==expected_count
     and selection['pool']==cfg['providers']['pool'] and selection['method']==cfg['method']
     and selection['fraction']==cfg['fraction'] and selection['selection_seed']==cfg['selection_seed']==1234
     and cfg['expected_horizon']==p.horizon(expected_count),
     'Exact complete eligible pool, target selected budget/membership/horizon required')
 b=load(binding_ref);request=load(b['request'])
 # Full held.activation and actual UID/nonce/spool checks remain in driver.current.
 p.require(b['schema']=='root-fulltrajectory-held-binding-v2' and b['job_id'].isdecimal()
     and request['site']=='babel' and request['original_cell_id']==cfg['id'], 'Exact new physical target cell binding required')
 storage=load(cfg['storage_reservation']);ownership=load(cfg['providers']['ownership'])
 function=p.key(dict(recipe=cfg['recipe'],pool_records=pool['records_json']['sha256'],selected_records=selection['records_json']['sha256'],
     training_seed=cfg['training_seed'],role='student',qualification_max_steps=None,replicate_site='babel'))
 p.require(storage['status']=='PASS_EXCLUSIVE_MEASURED_STORAGE_RESERVATION' and storage['output']==cfg['output']
     and storage.get('cell_id')==cfg['id'] and storage['checkpoint_transaction_bytes']==p.CAPSULE_BYTES
     and storage['retained_accepted_states']==2 and storage['allow_new_intermediate_retirement'] is True
     and type(storage['other_outstanding_reserved_bytes']) is int and storage['other_outstanding_reserved_bytes']>=0
     and ownership['owner']=='Root' and ownership['site_profile']==cfg['site_profile']
     and ownership['status']=='ACTIVE_EXCLUSIVE_NATIVE_RUN_CLAIM' and ownership['function_identity']==function,
     'Measured owned target storage and native function ownership required')
 return dict(configuration=configuration,config_identity=p.n.run_identity(cfg),binding=binding_ref,
     selection=cfg['providers']['selection'],selected_count=expected_count,ordered_record_ids_sha256=p.key(selection['record_ids']),
     pool=cfg['providers']['pool'],function_identity=function,training_seed=cfg['training_seed'],expected_horizon=cfg['expected_horizon'],
     providers=cfg['providers'],site_profile=cfg['site_profile'],output=cfg['output'],report=cfg['report'],
     storage_reservation=cfg['storage_reservation'],storage_body_sha256=p.key(storage),ownership_body_sha256=p.key(ownership),
     allocation=cfg['allocation'],checkpoint_root=p.CHECKPOINT_ROOT,
     all_in_cost_policy='ALL_SETUP_FAILED_SIGNAL_TRAINING_NATIVE_AND_CACHED_ACQUISITION_COSTS_COUNT')

def canary(proof_ref,review_ref,fp,load=p.checked):
 q=load(proof_ref);r=load(review_ref);reviewer(r);cfg=load(q['configuration'])
 p.require(q['status']=='ACTUAL_BABEL_SHARED_RUNTIME_CANARY_REQUIRES_INDEPENDENT_REVIEW'
     and cfg['id']==p.CANARY_ID and cfg['output'].startswith(p.CHECKPOINT_ROOT+'/')
     and q['sources']==p.sources() and q['actual_same_job_restore'] is True
     and q['config_identity']==p.n.run_identity(cfg) and q['expected_natural_horizon']==p.horizon(33038)
     and 1<q['accepted_prefix_end']<258
     and q['actual_records']==q['accepted_prefix_end']*128 and len(q['accepted_segments'])==2
     and q['max_steps']==-1 and q['ordinary_accepted_checkpoint_control'] is True
     and q['actual_checkpoint_root']==p.CHECKPOINT_ROOT and q['runtime_invariants']==fp
     and invariants(cfg,load,historical=True)==fp
     and r['status']=='PASS_INDEPENDENT_ACTUAL_BABEL_SHARED_RUNTIME_CANARY_V4'
     and r['readout']==proof_ref and r['configuration']==q['configuration'] and r['binding']==q['binding']
     and r['sources']==p.sources() and r['runtime_invariants']==fp
     and r['actual_shared_filesystem_save_restore'] is True,
     'One genuine new V4 cold/accepted/shared-filesystem EXACT restore canary is mandatory before bulk production')
 return q

def verify(ref,cfg,configuration,binding_ref,phase,load=p.checked):
 v=load(ref);reviewer(v)
 p.require(v['status']==CERT_STATUS and v['schema']=='babel-shared-runtime-exact-target-v4'
     and v['owner']=='Root' and v['phase']==phase and phase in ('shared_canary','production')
     and v['anchor_readout']==ANCHOR_READOUT and v['anchor_review']==ANCHOR_REVIEW
     and v['anchor_configuration']==ANCHOR_CONFIG and v['sources']==p.sources()
     and v['natural_runtime_seed_policy']==p.SEED_POLICY and v['allowed_delta_fields']==list(DELTA_FIELDS) and v['checkpoint_identity_migration'] is False,
     'Exact independently reviewed per-target certificate required, never wildcard or oldweight migration')
 source_bridge(v['source_bridge'],load);oldcfg,_,_=anchor(load)
 fp=invariants(cfg,load);p.require(invariants(oldcfg,load,historical=True)==fp,'Numerical/input/runtime invariant mismatch')
 p.require(v['runtime_invariants']==fp and v['target']==target(cfg,configuration,binding_ref,load),
     'Certificate target/selection/seed/horizon/storage/physical binding differs')
 # Pin every undeclared logical difference, rather than merely listing some invariants.
 def fixed(c):
  import copy
  x=copy.deepcopy(c)
  for field in DELTA_FIELDS:
   d=x;parts=field.split('.')
   for part in parts[:-1]:d=d.get(part,{})
   d.pop(parts[-1],None)
  return x
 p.require(fixed(cfg)==fixed(oldcfg),'Undeclared logical configuration delta')
 if phase=='shared_canary':
  p.require(cfg['id']==p.CANARY_ID and v.get('actual_v4_canary') is None
      and v.get('actual_v4_canary_review') is None,'Bootstrap only exact uniform5 seed29; no fabricated future canary')
 else:canary(v['actual_v4_canary'],v['actual_v4_canary_review'],fp,load)
 return v

def draft(configuration,binding_ref,bridge_ref,phase,*,actual_canary=None,actual_canary_review=None,load=p.checked):
 """Read-only request builder. It deliberately cannot mint a reviewed PASS."""
 cfg=load(configuration);source_bridge(bridge_ref,load);oldcfg,_,_=anchor(load)
 fp=invariants(cfg,load)
 p.require(invariants(oldcfg,load,historical=True)==fp,'Unqualified invariant delta in request')
 p.require(phase in ('shared_canary','production'),'Explicit finite execution phase required')
 if phase=='shared_canary':
  p.require(cfg['id']==p.CANARY_ID and actual_canary is None and actual_canary_review is None,
      'Only exact prospective canary, no invented future outcome')
 else:canary(actual_canary,actual_canary_review,fp,load)
 return dict(schema='babel-shared-runtime-exact-target-v4',status='DRAFT_REQUIRES_INDEPENDENT_EXACT_TARGET_REVIEW_V4',
     owner='Root',reviewer=None,phase=phase,natural_runtime_seed_policy=p.SEED_POLICY,anchor_readout=ANCHOR_READOUT,anchor_review=ANCHOR_REVIEW,
     anchor_configuration=ANCHOR_CONFIG,sources=p.sources(),source_bridge=bridge_ref,runtime_invariants=fp,
     allowed_delta_fields=list(DELTA_FIELDS),checkpoint_identity_migration=False,
     target=target(cfg,configuration,binding_ref,load),actual_v4_canary=actual_canary,actual_v4_canary_review=actual_canary_review)
