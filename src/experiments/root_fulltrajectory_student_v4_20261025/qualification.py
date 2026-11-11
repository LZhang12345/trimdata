"""Join real prefix objective/LR/state observations; no full-epoch/science PASS."""
import argparse,json,math
from pathlib import Path
from . import protocol as p
from experiments.root_native_qualification_v10_20261024.observer import validate_lr_trajectory

def readout(configuration,binding_ref,receipts,output):
 cfg=p.checked(configuration);binding=p.checked(binding_ref);selected=p.checked(cfg['providers']['selection']);expected=p.horizon(selected['count']);p.require(cfg['expected_horizon']==expected,'Exact selectedN/horizon required');allowed=set(selected['record_ids']);cursor=0;all_ids=[];previous=None;rows=[];evidence=[]
 p.require(len(receipts)==2 and cfg['id']==p.CANARY_ID,'Exactly two actual shared-filesystem canary segments required')
 p.supported(cfg)
 from .shared_runtime import invariants
 profile=p.native_config.validate_profile(p.checked(cfg['site_profile']))
 p.require(profile['output_root']==p.CHECKPOINT_ROOT and p.n.within(cfg['output'],p.CHECKPOINT_ROOT),'Exact new shared checkpoint namespace required')
 for ordinal,ref in enumerate(receipts):
  value=p.checked(ref);end=value['global_step'];base=Path(value['exposure']['path']).parent
  name=binding['job_id']+'-r'+str(ordinal)+'-'+binding['nonce'][:8]
  outer=Path(cfg['report'])/'ROOT_LONG/attempts'/name
  started=p.read(outer/'ATTEMPT_START.json');returned=p.read(outer/'CHILD_RETURN.json')
  outcome=p.checked(returned['child_exit'])
  p.require(started['actual']['job_id']==binding['job_id'] and started['actual']['restart']==ordinal
      and started['source']==p.sources() and started['qualification_mode'] is False and started['shared_canary_mode'] is True
      and started['ordinary_accepted_checkpoint_control'] is True and started['requested_segment_seconds'] is None and started['scientific_max_steps']==-1
      and returned['returncode']==0 and returned['logical_exit_code']==75 and returned['actual']==started['actual']
      and outcome['configuration']==configuration and outcome['config_identity']==p.n.run_identity(cfg)
      and outcome['adapter_sources']==p.sources() and outcome['logical_exit_code']==75 and outcome['receipt']==ref
      and math.isfinite(returned['elapsed_seconds']) and returned['elapsed_seconds']>0,
      'Actual ordinary-checkpoint canary driver/return evidence required, not only child labels')
  p.require(value['status']=='ACCEPTED_STUDENT_CONTINUATION_REQUIRES_ROOT_SAME_JOB_ACTION' and value['configuration']==configuration
      and value['config_identity']==p.n.run_identity(cfg) and value['adapter_sources']==p.sources()
      and value['actual_job_id']==binding['job_id'] and value['actual_restart_count']==ordinal and cursor<end<expected['natural_updates'],
      'Actual same-config/job/source progressing qualification segments only')
  ordinary=p.read(base.parent/'ORDINARY_ACCEPTED_CANARY_BOUNDARY_V4.json')
  p.require(ordinary['schema']=='babel-shared-runtime-canary-boundary-v4'
      and ordinary['configuration_identity']==p.n.run_identity(cfg) and ordinary['sources']==p.sources()
      and ordinary['accepted']==value['accepted'] and ordinary['global_step']==end
      and ordinary['actual_job_id']==binding['job_id'] and ordinary['actual_restart']==ordinal
      and ordinary['no_save_requested'] is True and ordinary['no_scientific_step_cap'] is True
      and ordinary['ordinary_checkpoint_seconds']==1200 and ordinary['shared_checkpoint_root']==p.CHECKPOINT_ROOT,
      'Actual ordinary accepted callback proof required')
  device=p.read(base.parent/'ACTUAL_DEVICE.json');storage=p.read(base.parent/'ACTUAL_STORAGE.json')
  argv=p.read(base.parent/'ACTUAL_STUDENT_ARGV.json');ops=p.checked(value['operators'])
  p.require(device['device_count']==1 and 'L40S' in device['GPU'] and device['torch_version']=='2.1.2+cu121'
      and device['CUDA_version']=='12.1' and storage['actual_job_id']==binding['job_id']
      and storage['reservation']==cfg['storage_reservation'] and storage['available_bytes']>=storage['required_bytes']
      and storage['checkpoint_root']==p.CHECKPOINT_ROOT and type(storage['checkpoint_root_device']) is int
      and argv['no_max_steps'] is True and argv['natural_epoch'] is True and '--max_steps' not in argv['argv']
      and argv['config_identity']==p.n.run_identity(cfg) and ops['explicit_policy']['policy']==p.RUNTIME
      and ops['explicit_policy']['calls']>0 and ops['explicit_policy']['no_operator_fallback'] is True,
      'Cold actual device/storage/uncapped CLI/operator evidence required')
  archive=p.checked(value['accepted_archive']);p.require(archive['identity']==p.n.run_identity(cfg) and archive['global_step']==end
      and Path(value['accepted']['path'])==Path(cfg['output'])/'training'/('checkpoint-'+str(end))/'ACCEPTED.json'
      and value['accepted_archive']['sha256']==value['accepted']['sha256'],'Actual accepted archive identity differs')
  horizon=p.read(base/'ACTUAL_HORIZON.json');exposure=p.checked(value['exposure'])
  p.require(horizon['actual_resume_step']==cursor and horizon['args_max_steps']==-1 and horizon['native_max_steps']==expected['natural_updates']
      and horizon['selected_count']==expected['selected_count'] and exposure['record_count']==len(exposure['ordered_record_ids'])==(end-cursor)*128
      and exposure['microbatches']==(end-cursor)*32,'Actual uncapped horizon/cursor/exposure differs')
  lr=[json.loads(x) for x in (base/'LEARNING_RATES.jsonl').read_text().splitlines()];validate_lr_trajectory(lr,cursor,end,expected['natural_updates'])
  units=[json.loads(x) for x in (base/'OBJECTIVE_UNITS.jsonl').read_text().splitlines()]
  p.require(all(x['skipped_steps']==0 for x in lr) and len(units)==(end-cursor)*32
      and [v for x in units for v in x['record_ids']]==exposure['ordered_record_ids']
      and all(x['microbatch_size']==4 and min(x['per_record_shifted_targets'])>0 and x['declared_accumulation']==32
          and math.isfinite(x['native_unscaled_loss']) and sum(x['per_record_shifted_targets'])==x['expanded_shifted_targets'] for x in units),
      'Missing/invalid native mask/objective/update evidence')
  begin=p.read(base.parent/('LONGWALL_STATE_BEGIN-'+str(cursor)+'.json'))
  saved=p.read(base.parent/('LONGWALL_STATE_SAVE-'+str(end)+'.json'))
  interval=saved['observed_epoch']-begin['observed_epoch']
  p.require(math.isfinite(interval) and interval>=0,'Actual first-micro to saved-state timing required')
  # StudentAudit writes exposure under child/student_observations; state audit
  # writes beside that directory in the child. This is the actual path relation.
  p.require(begin['state']['global_step']==begin['state']['actual_DS_global_steps']==cursor and saved['state']['global_step']==saved['state']['actual_DS_global_steps']==end,'Actual saved/restored optimizer cursor differs')
  for state in (begin,saved):p.require(state['configuration_identity']==p.n.run_identity(cfg) and state['sources']==p.sources()
      and state['state']['all_weights_finite'] is True and state['state']['DS_skipped_steps']==0,'Foreign/nonfinite actual state snapshot')
  if previous is not None:
   fields=sorted(set(begin['state'])|set(previous['state']))
   comparison=dict(previous_saved=previous['state'],actual_restored=begin['state'],
       fields_equal={key:begin['state'].get(key)==previous['state'].get(key) for key in fields},
       exact_equal=begin['state']==previous['state'],configuration=configuration,
       actual_job_id=binding['job_id'],actual_restart=ordinal,scientific_performance_measured=False)
   p.seal(str(output)+'.restore-'+str(ordinal)+'.json',comparison)
   p.require(comparison['exact_equal'],'Restored live weights/Adam/scheduler/RNG/cursors are not EXACT saved state')
   restore=p.read(base.parent/'VERIFIED_DS_RESTORE-rank0.json')
   p.require(restore['identity']==p.n.run_identity(cfg) and restore['module_restore']['loaded_keys']
       and restore['module_restore']['verified_frozen_missing_keys'],'Actual verified DeepSpeed module restore absent')
  p.require(set(exposure['ordered_record_ids'])<=allowed,'Foreign optimizer-used record ID')
  evidence.append(dict(segment=ref,actual_node=started['actual']['node'],checkpoint_root_device=storage['checkpoint_root_device'],attempt_start=p.desc(outer/'ATTEMPT_START.json'),child_return=p.desc(outer/'CHILD_RETURN.json'),actual_all_in_seconds=returned['elapsed_seconds'],requested_outer_child_window_seconds=None,outer_child_window_includes_cold_setup=True,actual_first_micro_to_save_seconds=interval,horizon=p.desc(base/'ACTUAL_HORIZON.json'),learning_rates=p.desc(base/'LEARNING_RATES.jsonl'),objective_units=p.desc(base/'OBJECTIVE_UNITS.jsonl'),begin=p.desc(base.parent/('LONGWALL_STATE_BEGIN-'+str(cursor)+'.json')),saved=p.desc(base.parent/('LONGWALL_STATE_SAVE-'+str(end)+'.json'))))
  all_ids+=exposure['ordered_record_ids'];previous=saved;cursor=end;rows.append(ref)
 node_devices={}
 for row in evidence:
  node=row['actual_node'];p.require(isinstance(node,str) and node,'Actual node identity required')
  if node in node_devices:p.require(node_devices[node]==row['checkpoint_root_device'],'Shared checkpoint filesystem device changed on same actual node')
  node_devices[node]=row['checkpoint_root_device']
 # Linux device numbers are node-local mount IDs, not a cross-node filesystem identity.
 # Exact configured root, original accepted bytes and live restored state bind the shared path.
 p.require(len(all_ids)==len(set(all_ids))==cursor*128,'Repeated/missing actual optimized records across resume')
 return p.seal(output,dict(status='ACTUAL_BABEL_SHARED_RUNTIME_CANARY_REQUIRES_INDEPENDENT_REVIEW',configuration=configuration,
     binding=binding_ref,config_identity=p.n.run_identity(cfg),sources=p.sources(),actual_same_job_restore=True,ordinary_accepted_checkpoint_control=True,actual_checkpoint_root=p.CHECKPOINT_ROOT,runtime_invariants=invariants(cfg),
     exact_live_state_restore_fields=['weights','optimizer','scheduler','rng','global_step','actual_DS_global_steps','DS_skipped_steps'],
     accepted_prefix_end=cursor,accepted_segments=rows,evidence=evidence,actual_records=len(all_ids),expected_natural_horizon=expected,
     max_steps=-1,full_epoch_completed=False,scientific_performance_measured=False,production_authorized=False,
     qualification_may_be_positive_or_fail_no_tolerance_change=True))

def main():
 a=argparse.ArgumentParser()
 for name in ('configuration','binding','receipts','output'):a.add_argument('--'+name,required=True)
 v=a.parse_args();print(p.canonical(readout(p.desc(v.configuration),p.desc(v.binding),p.read(v.receipts),v.output)).decode())
if __name__=='__main__':main()
