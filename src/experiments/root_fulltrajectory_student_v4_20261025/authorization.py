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
 bundle=load(ref);p.require(bundle['raw_readout']==p.NATURAL_ANCHOR_READOUT,'Only accurately labeled genuine seed17 natural runtime anchor')
 raw=load(bundle['raw_readout']);study=load(bundle['study'])
 review=independent(bundle['independent_review'],'PASS_INDEPENDENT_ACTUAL_DECLARED_EPOCH_RUNTIME_FOR_STUDENT_PROBE',
     'raw_readout',bundle['raw_readout'],load)
 p.require(review['study']==bundle['study'] and review['driver_sources']==p.qualification.sources(),
     'Independent epoch review/source mismatch')
 p.require(raw['status']=='PASS_ACTUAL_DECLARED_EPOCH_RUNTIME_UNREVIEWED'
     and raw['driver_sources']==p.qualification.sources() and raw['native_sources']==config['sources']
     and raw['study_identity']==p.key(p.bare(study)) and raw['paired_initialization_seed']==p.NATURAL_ANCHOR_SEED
     and study['seed']==p.NATURAL_ANCHOR_SEED and study['id']==p.qualification.identifier('babel',p.NATURAL_ANCHOR_SEED)
     and config['training_seed'] in p.SEEDS,
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
 p.require(reference['recipe']==config['recipe'] and reference['sources']==config['sources']
     and reference['training_seed']==p.NATURAL_ANCHOR_SEED, 'Different qualified recipe/core or mislabeled actual anchor seed')
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
     and gate['site_profile']==config['site_profile']
     and gate['natural_runtime_seed_policy']==p.SEED_POLICY and gate['target_initialization_seed']==config['training_seed'], 'Wrong explicit runtime gate/source/site')
 for name in ('environment','assets'):
  p.require(gate[name]==config['providers'][name], 'Gate provider mismatch '+name)
 p.require(config['execution_adapter']==dict(package=p.PACKAGE,sources=p.sources(),runtime=p.RUNTIME,
     natural_epoch=True,ordinary_native_launch_forbidden=True), 'Student adapter absent from logical identity')
 natural_evidence(gate['qualification_evidence'],config,load)
 p.require(permit['schema']=='babel-shared-runtime-outer-authorization-v4' and permit['owner']=='Root'
     and permit['configuration'] and load(permit['configuration'])==config
     and permit['config_identity']==p.n.run_identity(config) and permit['adapter_sources']==p.sources()
     and permit['runtime']==p.RUNTIME and permit['phase'] in ('shared_canary','production')
     and permit['stop_after_global_step'] is None and config.get('qualification_max_steps') is None,
     'Exact new shared-runtime target authorization required; no hidden scientific cap')
 independent(permit['review'],'PASS_INDEPENDENT_EXACT_BABEL_SHARED_EXECUTION_AUTHORIZATION_V4',
     'authorization_identity',p.key(p.bare(permit)),load)
 p.require(permit['expected_horizon']==p.horizon(providers['selection']['count']), 'Wrong natural target horizon')
 from .shared_runtime import verify as compatibility
 compatibility(permit['runtime_certificate'],config,permit['configuration'],permit['physical_binding'],permit['phase'],load)
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
