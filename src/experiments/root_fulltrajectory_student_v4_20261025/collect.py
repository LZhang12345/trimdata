"""Finite accepted-attempt accounting; full epoch completion is not evaluation."""
import math
from pathlib import Path
from . import protocol as p
from experiments.root_native_qualification_v10_20261024.observer import validate_lr_trajectory

def collect(configuration,attempt_receipts,output,load=p.checked):
 cfg=load(configuration);selected=load(cfg['providers']['selection']);expected=p.horizon(selected['count'])
 allowed=set(selected['record_ids']);cursor=0;ids=[];bound=[];last_lr=None;last_accepted=None
 p.require(attempt_receipts,'No actual accepted attempt history')
 for ref in attempt_receipts:
  value=load(ref)
  p.require(value['configuration']==configuration and value['config_identity']==p.n.run_identity(cfg)
      and value['adapter_sources']==p.sources() and value['runtime']==p.RUNTIME,
      'Foreign immutable student attempt')
  attempt=Path(value['exposure']['path']).parent
  horizon=load(p.desc(attempt/'ACTUAL_HORIZON.json'))
  p.require(horizon['actual_resume_step']==cursor and horizon['args_max_steps']==-1
      and horizon['native_max_steps']==expected['natural_updates'] and horizon['selected_count']==selected['count'],
      'Attempt chain has missing/repeated accepted cursor or wrong natural horizon')
  if value['status']=='ACCEPTED_STUDENT_CONTINUATION_REQUIRES_ROOT_SAME_JOB_ACTION':
   end=value['global_step'];last_accepted=value['accepted']
  else:
   p.require(value['status']=='COMPLETE_STUDENT_TRAINING_NOT_EVALUATION','Unaccepted/failed attempts do not advance durable accounting')
   complete=load(value['native_complete']);end=complete['global_step'];last_accepted=complete['accepted']
  p.require(cursor<end<=expected['natural_updates'],'Nonincreasing/out-of-range actual accepted update cursor')
  archive=value['accepted_archive']
  p.require(archive['sha256']==last_accepted['sha256'] and archive['bytes']==last_accepted['bytes'],
      'Archived accepted metadata differs from original capsule receipt')
  state=load(archive)
  p.require(state['status']=='ACCEPTED_FULL_NATIVE_STATE' and state['identity']==p.n.run_identity(cfg)
      and state['global_step']==end,'Accepted capsule identity/cursor mismatch')
  exposure=load(value['exposure']);seen=exposure['ordered_record_ids']
  p.require(exposure['record_count']==len(seen)==(end-cursor)*128 and exposure['microbatches']==(end-cursor)*32
      and set(seen)<=allowed and len(set(seen))==len(seen),'Actual accepted exposure count/source/repetition differs')
  lr=[__import__('json').loads(line) for line in (attempt/'LEARNING_RATES.jsonl').read_text().splitlines()]
  validate_lr_trajectory(lr,cursor,end,expected['natural_updates'])
  p.require(all(row['skipped_steps']==0 for row in lr) and (last_lr is None or lr[0]==last_lr),
      'Skipped update or scheduler discontinuity across accepted restart')
  objective=[__import__('json').loads(line) for line in (attempt/'OBJECTIVE_UNITS.jsonl').read_text().splitlines()]
  p.require(len(objective)==(end-cursor)*32 and [v for row in objective for v in row['record_ids']]==seen
      and all(row['microbatch_size']==4 and len(row['per_record_shifted_targets'])==4
          and min(row['per_record_shifted_targets'])>0 and sum(row['per_record_shifted_targets'])==row['expanded_shifted_targets']
          and row['declared_accumulation']==32 and math.isfinite(row['native_unscaled_loss'])
          and math.isfinite(row['returned_native_scaled_loss']) for row in objective),
      'Incomplete/nonpositive actual native objective-unit accounting')
  ids+=seen;cursor=end;last_lr=lr[-1];bound.append(ref)
 p.require(cursor==expected['natural_updates'] and len(ids)==expected['optimized_records']
     and len(set(ids))==len(ids),'Natural student epoch is incomplete or repeated records')
 optimized=set(ids);tail=[v for v in selected['record_ids'] if v not in optimized]
 p.require(len(tail)==expected['unoptimized_tail_count'],'Actual unoptimized tail differs; no topup')
 return p.seal(output,dict(status='COMPLETE_ACTUAL_NATURAL_STUDENT_TRAJECTORY_NOT_EVALUATION',
     configuration=configuration,config_identity=p.n.run_identity(cfg),attempts=bound,accepted=last_accepted,
     expected=expected,actual_optimized_records=len(ids),actual_unoptimized_tail_record_ids=tail,
     ordered_record_ids_sha256=p.key(ids),actual_optimizer_updates=cursor,
     native_objective_scaling_unchanged=True,actual_no_skipped_optimizer_updates=True,
     adapter_sources=p.sources(),runtime=p.RUNTIME,downstream_evaluation_complete=False,
     scientific_improvement_demonstrated=False))
