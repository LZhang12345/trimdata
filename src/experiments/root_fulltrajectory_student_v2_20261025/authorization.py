"""Outer execution permission changes; scientific config/checkpoint identity does not."""
from contextlib import contextmanager
from contextvars import ContextVar
from . import protocol as p
from experiments.root_production_student_adapter_v2_20261025 import protocol as prior
_CURRENT=ContextVar('reviewed_student_authorization',default=None)

def independent(ref,status,identity_field,identity,load):
 value=load(ref)
 p.require(value.get('status')==status and value.get(identity_field)==identity
     and value.get('reviewer') not in (None,'',p.AUTHOR,'storage_scope_review'),
     'Missing genuine independent actual evidence/review')
 return value

def natural_evidence(ref,config,load=p.checked,compatibility_ref=None):
 bundle=load(ref);raw=load(bundle['raw_readout']);study=load(bundle['study'])
 review=independent(bundle['independent_review'],'PASS_INDEPENDENT_ACTUAL_DECLARED_EPOCH_RUNTIME_FOR_STUDENT_PROBE',
     'raw_readout',bundle['raw_readout'],load)
 p.require(review['study']==bundle['study'] and review['driver_sources']==p.qualification.sources(),
     'Independent epoch review/source mismatch')
 p.require(raw['status']=='PASS_ACTUAL_DECLARED_EPOCH_RUNTIME_UNREVIEWED'
     and raw['driver_sources']==p.qualification.sources() and raw['native_sources']==config['sources']
     and raw['study_identity']==p.key(p.bare(study)) and raw['paired_initialization_seed']==config['training_seed'],
     'Unqualified/foreign raw epoch/runtime result')
 spec=study['specification'];units=raw['epoch_and_objective_units']
 p.require(study['site']=='babel' and spec==p.qualification.specification('babel')
     and units['passed'] is True and units['natural_epoch'] is True
     and units['selected_count']==1153 and units['actual_optimized_count']==1152
     and units['tail_count']==1 and units['no_tail_replacement_or_topup'] is True
     and set(units['actual_forwarded_sources'])=={'coco','gqa','ocr_vqa','vg','textvqa','text_only'},
     'H100 capped/partial-source evidence cannot qualify broad natural-epoch students')
 p.require(set(raw['numerical_comparisons'])=={'portable_weights','Adam_FP32_master_scaler','saved_RNG','scheduler','microbatch_loss_trajectory'}
     and set(raw['criteria'])=={'reference','interrupted','resumed'},'Incomplete actual parity/mechanism evidence')
 p.require(raw['actual_same_job_requeue'] is True
     and all(v['pass_numeric'] is True for v in raw['numerical_comparisons'].values())
     and raw['pre_restart_reference_cut4']['passed'] is True
     and raw['actual_update_cursor_checks']['passed'] is True
     and all(v['all_pass'] is True for v in raw['criteria'].values())
     and raw['optimizer_groups_equal'] is True and raw['semantic_learning_rate_trajectories']['passed'] is True,
     'Actual numerical/mask/optimizer/LR/restart requirements failed')
 reference=load(study['native_configurations']['reference'])
 p.require(reference['recipe']==config['recipe'] and reference['sources']==config['sources'], 'Different qualified recipe/core')
 from .compatibility import verify as compatible
 compatible(compatibility_ref or config['runtime_compatibility'],config,reference['site_profile'],load)
 for name in ('environment','assets'):
  p.require(reference['providers'][name]==config['providers'][name], 'Different qualified '+name)
 for name in ('gpus','microbatch','cpus','memory_mb','loader_workers'):
  p.require(reference['allocation'][name]==config['allocation'][name], 'Different qualified allocation '+name)
 return bundle

def verify(config,providers,profile,permit_ref,load=p.checked):
 gate=providers['qualification'];permit=load(permit_ref)
 p.require(gate['status']==p.GATE and gate['schema']=='production-student-runtime-gate-contract-v1'
     and gate['ordinary_native_launch_forbidden'] is True and gate['requires_separate_outer_authorization'] is True
     and gate['student_adapter']==dict(package=p.PACKAGE,sources=p.sources())
     and gate['native_sources']==config['sources'] and gate['recipe']==config['recipe']
     and gate['runtime']==p.RUNTIME and gate['site']=='babel' and profile['site']=='babel'
     and gate['site_profile']==config['site_profile'], 'Wrong explicit runtime gate/source/site')
 for name in ('environment','assets'):
  p.require(gate[name]==config['providers'][name], 'Gate provider mismatch '+name)
 p.require(config['execution_adapter']==dict(package=p.PACKAGE,sources=p.sources(),runtime=p.RUNTIME,
     natural_epoch=True,ordinary_native_launch_forbidden=True), 'Student adapter absent from logical identity')
 natural_evidence(gate['qualification_evidence'],config,load)
 p.require(permit['schema']=='production-student-outer-authorization-v1' and permit['owner']=='Root'
     and permit['config_identity']==p.n.run_identity(config) and permit['adapter_sources']==p.sources()
     and permit['runtime']==p.RUNTIME and permit['phase'] in ('canonical_pilot','production'),
     'Unreviewed/foreign outer execution authorization')
 independent(permit['review'],'PASS_INDEPENDENT_EXACT_STUDENT_EXECUTION_AUTHORIZATION',
     'authorization_identity',p.key(p.bare(permit)),load)
 selection=providers['selection'];expected=p.horizon(selection['count'])
 p.require(permit['expected_horizon']==expected and config.get('qualification_max_steps') is None,
     'Missing horizon/secret qualification cap')
 if permit['phase']=='canonical_pilot':
  p.require(config['id']==p.cell_id('5:uniform',17) and config['method']=='uniform'
      and config['fraction']=='0.05' and config['training_seed']==17 and selection['count']==33038
      and expected['natural_updates']==258 and permit['stop_after_global_step']==4,
      'Engineering permit authorizes only canonical full-selected uniform5percent seed17 cut4')
 else:
  p.require(permit['stop_after_global_step'] is None,'Production cannot retain a cap')
  proof=load(permit['canonical_pilot_readout']);pilot_cfg=load(proof['configuration'])
  p.require(proof['status']=='COMPLETE_CANONICAL_STUDENT_PILOT_ACCEPTED_BOUNDARY_NOT_FULL_EPOCH'
      and proof['config_identity']==p.n.run_identity(pilot_cfg)
      and pilot_cfg['id']==p.cell_id('5:uniform',17) and proof['accepted_global_step']==4
      and proof['expected_horizon']==p.horizon(33038) and proof['actual_args_max_steps']==-1
      and proof['actual_selected_count']==33038 and proof['actual_native_max_steps']==258
      and proof['adapter_sources']==prior.sources() and proof['runtime']==p.RUNTIME
      and proof['actual_learning_rates']['passed'] is True and proof['DS_skipped_steps']==0
      and proof['actual_record_count']==512 and proof['actual_microbatches']==128,
      'Absent/foreign/capped actual canonical production-CLI pilot')
  prefix=load(proof['pilot_prefix_evidence'])
  p.require(prefix['status']=='COMPLETE_ACTUAL_CONTIGUOUS_PILOT_PREFIX_NOT_FULL_EPOCH'
      and prefix['accepted_prefix_end']==4 and prefix['complete_pilot'] is True
      and prefix['config_identity']==proof['config_identity'] and prefix['adapter_sources']==prior.sources()
      and prefix['runtime']==p.RUNTIME and prefix['aggregate_exposure']['record_count']==512
      and prefix['aggregate_exposure']['microbatches']==128
      and len(prefix['aggregate_exposure']['ordered_record_ids'])==len(set(prefix['aggregate_exposure']['ordered_record_ids']))==512,
      'Missing genuine contiguous accepted pilot prefix')
  from .compatibility import verify as compatible
  compatible(config['runtime_compatibility'],config,pilot_cfg['site_profile'],load)
  p.require(pilot_cfg['execution_adapter']==dict(package=prior.PACKAGE,sources=prior.sources(),runtime=p.RUNTIME,natural_epoch=True,ordinary_native_launch_forbidden=True),'Only actual V2 canonical source can anchor this metadata-only extension')
  for name in ('recipe','sources'):
   p.require(pilot_cfg[name]==config[name], 'Canonical pilot recipe/core/profile/adapter mismatch '+name)
  for name in ('gpus','microbatch','cpus','memory_mb','loader_workers'):
   p.require(pilot_cfg['allocation'][name]==config['allocation'][name], 'Canonical pilot allocation mismatch '+name)
  for name in ('pool','assets','environment','eligibility'):
   p.require(pilot_cfg['providers'][name]==config['providers'][name], 'Pilot full provider mismatch '+name)
  p.require(proof['accepted_archive']['sha256']==proof['accepted']['sha256']
      and proof['accepted_archive']['bytes']==proof['accepted']['bytes'],'Pilot accepted header archive changed')
  accepted=load(proof['accepted_archive'])
  p.require(accepted['status']=='ACCEPTED_FULL_NATIVE_STATE' and accepted['identity']==proof['config_identity']
      and accepted['global_step']==4,'Pilot accepted state header/identity differs')
  operator=load(proof['operators'])
  p.require(operator['explicit_policy']['policy']==p.RUNTIME and operator['explicit_policy']['calls']>0
      and operator['explicit_policy']['no_operator_fallback'] is True,'Actual pilot did not exercise declared runtime')
  independent(permit['canonical_pilot_review'],'PASS_INDEPENDENT_ACTUAL_CANONICAL_STUDENT_PILOT',
      'pilot_readout',permit['canonical_pilot_readout'],load)
 return permit

@contextmanager
def scope(config,providers,profile,permit_ref,load=p.checked):
 permit=verify(config,providers,profile,permit_ref,load)
 token=_CURRENT.set((p.n.run_identity(config),permit))
 try:yield permit
 finally:_CURRENT.reset(token)

def require_current(config,providers,profile):
 value=_CURRENT.get()
 p.require(value is not None and value[0]==p.n.run_identity(config),
     'No explicit reviewed student execution scope; ordinary native launch is forbidden')
 p.require(providers['qualification']['status']==p.GATE and profile['site']=='babel', 'Wrong scoped gate/site')
 return value[1]
