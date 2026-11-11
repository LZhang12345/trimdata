"""Exact S1 scientific/physical bridge to genuine shared-FS Root V4 evidence."""
import copy
from pathlib import Path
from . import protocol as p
from experiments.root_fulltrajectory_student_v4_20261025 import protocol as qualified
from experiments.root_fulltrajectory_student_v4_20261025 import shared_runtime as base
from experiments.s1_fulltrajectory_held_cohort_v2_20261025 import protocol as held
BRIDGE_STATUS='PASS_INDEPENDENT_BABEL_S1_NUMERICAL_AND_SELECTION_SOURCE_BRIDGE_V1'
CERT_STATUS='PASS_INDEPENDENT_EXACT_BABEL_S1_SHARED_RUNTIME_TARGET_V1'
DELTA_FIELDS=base.DELTA_FIELDS


def numerical_bodies():
 for name in base.NUMERICAL_FILES:
  p.require((p.HERE/name).read_bytes()==(qualified.HERE/name).read_bytes(),'Changed S1 numerical/observer/checkpoint body '+name)
 return base.numerical_bodies()

def source_bridge(ref,load=p.checked):
 v=load(ref);base.reviewer(v)
 p.require(v['status']==BRIDGE_STATUS and v['qualified_runtime_sources']==qualified.sources()
     and v['target_sources']==p.sources() and v['held_sources']==held.sources()
     and v['native_sources']==p.n.sources() and v['numerical_bodies']==numerical_bodies()
     and v['runtime']==p.RUNTIME and v['natural_runtime_seed_policy']==p.SEED_POLICY
     and v['selection_pins']==p.SELECTIONS and v['allowed_s1_cells']==sorted(held.OLD_JOBS)
     and v['old_checkpoint_identity_migration'] is False and v['six_redundant_GPU_pilots_required'] is False,
     'Independent exact numerical/custom-selection/physical source bridge required')
 return v


def target(cfg,configuration,binding_ref,load=p.checked):
 arm=p.supported(cfg);profile=p.native_config.validate_profile(load(cfg['site_profile']))
 out=Path(cfg['output']);root=Path(p.CHECKPOINT_ROOT)
 p.require(profile['output_root']==p.CHECKPOINT_ROOT and out.is_absolute() and '..' not in out.parts and root in out.parents
     and not any(v.is_symlink() for v in (out,*out.parents)) and p.n.within(cfg['report'],profile['registry_root']),
     'Only literal owned shared checkpoint namespace and approved registry')
 b=load(binding_ref);request,science,_=held.reviewed(b['request'],load)
 p.require(b['schema']=='s1-fulltrajectory-held-binding-v2' and b['owner']=='Root' and b['sources']==held.sources()
     and b['job_id'].isdecimal() and b['job_id'] not in held.OLD_JOBS.values() and request['cell_id']==cfg['id']
     and request['scientific_design']==science and science['selection']==p.SELECTIONS[arm]['selection'],
     'Exact new physical S1 replacement binding and original selected science')
 selected=load(cfg['providers']['selection']);pool=load(cfg['providers']['pool']);method_review=load(selected['method_review'])
 p.require(cfg['providers']['pool']==science['logical_pool'] and cfg['providers']['eligibility']==science['logical_eligibility']
     and pool['count']==660763 and pool['pool_kind']=='all_source_protected_bridge'
     and selected['schema']=='native-record-selection-v1' and selected['status']=='PASS_SELECTION'
     and selected['pool']==cfg['providers']['pool'] and selected['count']==len(selected['record_ids'])==len(set(selected['record_ids']))==66076
     and selected['fraction']==cfg['fraction']=='0.10' and selected['method']==cfg['method']==p.SELECTIONS[arm]['method']
     and selected['selection_seed']==cfg['selection_seed']==1234 and selected['method_review']==p.SELECTIONS[arm]['method_review']
     and selected['budget']==dict(unit='whole_conversation_count',value=66076,fraction='0.10',eligible_pool_count=660763)
     and cfg['expected_horizon']==p.horizon(66076), 'Exact previously paid finalized S1 selection/count/budget/horizon')
 p.require(method_review['status']=='PASS_EXACT_SELECTOR_IMPLEMENTATION_AND_OUTPUT' and method_review['pool']==cfg['providers']['pool']
     and method_review['records_json']==selected['records_json'] and method_review['method']==cfg['method']
     and method_review['selection_seed']==1234 and method_review['fraction']=='0.10'
     and method_review['requested_K']==method_review['actual_K']==66076,
     'Genuine pinned selection review must join actual whole-record manifest; no relabel as baseline')
 function=p.key(dict(recipe=cfg['recipe'],pool_records=pool['records_json']['sha256'],selected_records=selected['records_json']['sha256'],
     training_seed=cfg['training_seed'],role='student',qualification_max_steps=None,replicate_site='babel'))
 ownership=load(cfg['providers']['ownership']);storage=load(cfg['storage_reservation'])
 p.require(ownership['status']=='ACTIVE_EXCLUSIVE_NATIVE_RUN_CLAIM' and ownership['owner']=='Root'
     and ownership['function_identity']==function and ownership['site_profile']==cfg['site_profile']
     and storage['status']=='PASS_EXCLUSIVE_MEASURED_STORAGE_RESERVATION' and storage['cell_id']==cfg['id']
     and storage['output']==cfg['output'] and storage['checkpoint_transaction_bytes']==p.CAPSULE_BYTES
     and storage['retained_accepted_states']==2 and storage['allow_new_intermediate_retirement'] is True
     and type(storage['other_outstanding_reserved_bytes']) is int and storage['other_outstanding_reserved_bytes']>=0,
     'Current exact function ownership/collective measured storage required')
 return dict(configuration=configuration,config_identity=p.n.run_identity(cfg),binding=binding_ref,scientific_request=b['request'],
     original_request=request['original_request'],old_binding=request['old_binding'],predecessor_retirement=request['predecessor_retirement'],
     method=cfg['method'],selection=cfg['providers']['selection'],method_review=selected['method_review'],selection_receipt=request['selection_receipt'],
     selected_count=66076,ordered_record_ids_sha256=p.key(selected['record_ids']),function_identity=function,
     expected_horizon=cfg['expected_horizon'],training_seed=cfg['training_seed'],providers=cfg['providers'],
     output=cfg['output'],report=cfg['report'],site_profile=cfg['site_profile'],allocation=cfg['allocation'],
     storage_reservation=cfg['storage_reservation'],storage_body_sha256=p.key(storage),ownership_body_sha256=p.key(ownership),
     all_in_cost_policy='ALL_SETUP_FAILED_SIGNAL_TRAINING_NATIVE_AND_CACHED_ACQUISITION_COSTS_COUNT')


def fixed(cfg):
 result=copy.deepcopy(cfg)
 for field in DELTA_FIELDS:
  d=result;parts=field.split('.')
  for part in parts[:-1]:d=d.get(part,{})
  d.pop(parts[-1],None)
 return result


def verify(ref,cfg,configuration,binding_ref,phase,load=p.checked):
 v=load(ref);base.reviewer(v)
 p.require(v['schema']=='babel-s1-shared-runtime-exact-target-v1' and v['status']==CERT_STATUS and v['owner']=='Root'
     and phase==v['phase']=='production' and v['sources']==p.sources() and v['qualified_runtime_sources']==qualified.sources()
     and v['anchor_readout']==base.ANCHOR_READOUT and v['anchor_review']==base.ANCHOR_REVIEW
     and v['natural_runtime_seed_policy']==p.SEED_POLICY and v['allowed_delta_fields']==list(DELTA_FIELDS)
     and v['checkpoint_identity_migration'] is False,'Exact independently reviewed S1 production-only certificate')
 source_bridge(v['source_bridge'],load);oldcfg,_,_=base.anchor(load)
 fp=base.invariants(cfg,load);numerical_bodies()
 p.require(fp==base.invariants(oldcfg,load,historical=True)==v['runtime_invariants'] and fixed(cfg)==fixed(oldcfg),
     'Unqualified numerical/input/recipe/logical delta; selector identity alone is allowed')
 p.require(v['target']==target(cfg,configuration,binding_ref,load),'Exact S1 target/seed/selection/physical/storage binding differs')
 # This deliberately calls the frozen V4 consumer of its ACTUAL V4 readout.
 # The V4 measurement keeps V4's real source/config/job identities, never S1 labels.
 base.canary(v['actual_v4_canary'],v['actual_v4_canary_review'],fp,load)
 return v


def draft(configuration,binding_ref,bridge_ref,*,actual_v4_canary,actual_v4_canary_review,load=p.checked):
 cfg=load(configuration);source_bridge(bridge_ref,load);oldcfg,_,_=base.anchor(load);fp=base.invariants(cfg,load)
 p.require(fp==base.invariants(oldcfg,load,historical=True) and fixed(cfg)==fixed(oldcfg),'Unqualified S1 target delta')
 base.canary(actual_v4_canary,actual_v4_canary_review,fp,load)
 return dict(schema='babel-s1-shared-runtime-exact-target-v1',status='DRAFT_REQUIRES_INDEPENDENT_EXACT_S1_TARGET_REVIEW_V1',
     owner='Root',reviewer=None,phase='production',sources=p.sources(),qualified_runtime_sources=qualified.sources(),
     anchor_readout=base.ANCHOR_READOUT,anchor_review=base.ANCHOR_REVIEW,natural_runtime_seed_policy=p.SEED_POLICY,
     source_bridge=bridge_ref,runtime_invariants=fp,allowed_delta_fields=list(DELTA_FIELDS),checkpoint_identity_migration=False,
     target=target(cfg,configuration,binding_ref,load),actual_v4_canary=actual_v4_canary,actual_v4_canary_review=actual_v4_canary_review)
