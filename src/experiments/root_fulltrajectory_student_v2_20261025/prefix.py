"""Real contiguous accepted pilot prefix, including time/signal continuations."""
import json,math
from pathlib import Path
from . import protocol as p
from experiments.root_native_qualification_v10_20261024.observer import validate_lr_trajectory

def resume_cursor(config):
 """Cheap admission guard; unchanged Native7 revalidates full saved semantics."""
 root=Path(config['output'])/'training';candidates=[]
 for path in root.glob('checkpoint-*'):
  p.require(not path.is_symlink() and (path/'ACCEPTED.json').is_file(),
      'Unaccepted/aliased pilot state requires owner reconciliation')
  receipt=p.checked(p.desc(path/'ACCEPTED.json'))
  step=receipt['global_step']
  p.require(receipt['status']=='ACCEPTED_FULL_NATIVE_STATE' and receipt['identity']==p.n.run_identity(config)
      and type(step) is int and step>0 and path.name=='checkpoint-'+str(step),
      'Foreign pilot accepted identity/cursor')
  candidates.append(step)
 cursor=max(candidates,default=0)
 p.require(cursor<4,'Accepted pilot4 already exists: independent review/production permit required, no replay')
 return cursor

def aggregate(config,current_segment,prior_attempts,load=p.checked,*,target_end=4):
 """No synthesized observations: join actual immutable accepted segment files."""
 p.require(type(target_end) is int and 0<target_end<=4,'Only an actual bounded accepted pilot prefix')
 p.require(isinstance(prior_attempts,list) and len(prior_attempts)<=3,'Finite at most3 proper accepted partial pilot prefixes')
 selected=load(config['providers']['selection']);allowed=set(selected['record_ids'])
 segments=[]
 for ref in prior_attempts:
  receipt=load(ref)
  p.require(receipt['status']=='ACCEPTED_STUDENT_CONTINUATION_REQUIRES_ROOT_SAME_JOB_ACTION'
      and receipt['pilot_progress_pending'] is True and receipt['gate_hold_required'] is False
      and receipt['pilot_readout'] is None,'Only actual incomplete pilot prefixes may be joined')
  segments.append(receipt['segment'])
 segments.append(current_segment)
 cursor=0;ordered=[];combined=[];first=None;metadata=[]
 for ref in segments:
  value=load(ref)
  p.require(value['status']=='ACTUAL_ACCEPTED_STUDENT_SEGMENT_METADATA_V1'
      and value['config_identity']==p.n.run_identity(config) and value['adapter_sources']==p.sources()
      and value['actual_job_id']==load(current_segment)['actual_job_id'] and value['runtime']==p.RUNTIME,
      'Foreign pilot segment/source/job/runtime')
  end=value['global_step'];p.require(cursor<end<=4,'Pilot prefix missing/repeated/out-of-range cursor')
  archived=value['accepted_archive'];original=value['accepted'];state=load(archived)
  p.require(archived['bytes']==original['bytes'] and archived['sha256']==original['sha256']
      and state['status']=='ACCEPTED_FULL_NATIVE_STATE' and state['identity']==p.n.run_identity(config)
      and state['global_step']==end,'Actual archived pilot acceptance changed')
  path=Path(value['exposure']['path']).parent
  horizon=load(p.desc(path/'ACTUAL_HORIZON.json'))
  p.require(horizon['actual_resume_step']==cursor and horizon['args_max_steps']==-1
      and horizon['native_max_steps']==258 and horizon['selected_count']==33038,
      'Pilot segment was capped/repeated or used wrong natural horizon')
  if first is None:first=horizon
  exposure=load(value['exposure']);ids=exposure['ordered_record_ids']
  p.require(exposure['record_count']==len(ids)==(end-cursor)*128
      and exposure['microbatches']==(end-cursor)*32 and set(ids)<=allowed
      and len(set(ids))==len(ids),'Pilot accepted exposure count/identity/repetition mismatch')
  rates=[json.loads(line) for line in (path/'LEARNING_RATES.jsonl').read_text().splitlines()]
  validate_lr_trajectory(rates,cursor,end,258)
  p.require(all(row['skipped_steps']==0 for row in rates)
      and (not combined or combined[-1]==rates[0]),'Actual pilot scheduler continuity failed')
  units=[json.loads(line) for line in (path/'OBJECTIVE_UNITS.jsonl').read_text().splitlines()]
  p.require(len(units)==(end-cursor)*32 and [v for row in units for v in row['record_ids']]==ids
      and all(row['microbatch_size']==4 and len(row['per_record_shifted_targets'])==4
          and min(row['per_record_shifted_targets'])>0
          and sum(row['per_record_shifted_targets'])==row['expanded_shifted_targets']
          and row['declared_accumulation']==32 and math.isfinite(row['native_unscaled_loss'])
          and math.isfinite(row['returned_native_scaled_loss']) for row in units),
      'Missing/invalid actual accepted pilot objective units')
  ordered+=ids;combined+=rates if not combined else rates[1:];cursor=end
  metadata.append(dict(segment=ref,horizon=p.desc(path/'ACTUAL_HORIZON.json'),
      learning_rates=p.desc(path/'LEARNING_RATES.jsonl'),objective_units=p.desc(path/'OBJECTIVE_UNITS.jsonl')))
 p.require(cursor==target_end and len(ordered)==len(set(ordered))==target_end*128,'Pilot prefix missing/repeated actual accepted exposure')
 return dict(status='COMPLETE_ACTUAL_CONTIGUOUS_PILOT_PREFIX_NOT_FULL_EPOCH',accepted_prefix_end=target_end,complete_pilot=target_end==4,
     config_identity=p.n.run_identity(config),adapter_sources=p.sources(),runtime=p.RUNTIME,
     initial_actual_horizon=first,aggregate_exposure=dict(record_count=target_end*128,microbatches=target_end*32,ordered_record_ids=ordered),
     combined_actual_learning_rates=combined,actual_segments=metadata,
     scope='ACTUAL_CONTIGUOUS_ACCEPTED_PILOT_PREFIX_NOT_A_FABRICATED_FRESH_ATTEMPT')

def validate_prior_history(config,prior_attempts,cursor,actual_job_id,load=p.checked):
 p.require(isinstance(prior_attempts,list) and len(prior_attempts)<=3,'Finite accepted pilot history required')
 if cursor==0:
  p.require(not prior_attempts,'Fresh canonical pilot cannot inherit stale exposure');return None
 p.require(0<cursor<4 and prior_attempts,'Partial saved pilot requires its genuine accepted history before NN')
 last=load(prior_attempts[-1])
 p.require(last['global_step']==cursor and last['actual_job_id']==actual_job_id
     and last['pilot_progress_pending'] is True and last['gate_hold_required'] is False,
     'Saved pilot cursor/history/job does not match')
 current=Path(config['output'])/'training'/('checkpoint-'+str(cursor))/'ACCEPTED.json'
 p.require(not current.is_symlink() and not current.parent.is_symlink()
     and p.desc(current)==last['accepted'],
     'Current physical accepted state differs from measured historical pilot prefix')
 return aggregate(config,last['segment'],prior_attempts[:-1],load,target_end=cursor)
