"""Source-bound child API; Root alone owns scheduler/atomic requeuehold."""
import argparse,importlib,os,sys,time,traceback
from pathlib import Path
from . import protocol as p,authorization,student_config
from .observer import StudentAudit
from experiments.root_production_epoch_qualification_v1_20261024 import runtime

def student_cli(config,profile,providers):
 argv=student_config.cli(config,profile,providers)
 p.require(config['role']=='student' and '--max_steps' not in argv
     and argv[argv.index('--num_train_epochs')+1]=='1'
     and argv[argv.index('--seed')+1]==str(config['training_seed'])
     and argv[argv.index('--data_seed')+1]==str(config['training_seed']), 'Student must use actual natural oneepoch CLI')
 return argv

def validate_continuation(value,config,attempt,job_id,checkpoint_check):
 p.require(value['status']=='ACCEPTED_STUDENT_CONTINUATION_REQUIRES_ROOT_SAME_JOB_ACTION'
     and value['config_identity']==p.n.run_identity(config) and value['actual_job_id']==job_id
     and value['adapter_sources']==p.sources() and value['requires_atomic_pending_before_exit'] is True,
     'Absent/foreign durable accepted continuation')
 path=Path(value['checkpoint_directory'])
 p.require(path.parent==Path(config['output'])/'training' and path.name=='checkpoint-'+str(value['global_step'])
     and p.desc(path/'ACCEPTED.json')==value['accepted'], 'Continuation capsule/path changed')
 receipt=checkpoint_check(path,p.n.run_identity(config))
 p.require(receipt['global_step']==value['global_step'] and receipt['global_step']>0,
     'No accepted actual update boundary')
 if value.get('pilot_progress_pending',False):
  p.require(config['id']==p.cell_id('5:uniform',17) and 0<receipt['global_step']<4
      and value['gate_hold_required'] is False and value['pilot_readout'] is None,
      'Partial pilot must continue without qualification promotion')
 if value['gate_hold_required']:
  p.require(config['id']==p.cell_id('5:uniform',17) and receipt['global_step']==4
      and p.checked(value['pilot_readout'])['config_identity']==p.n.run_identity(config),
      'Only actual canonical accepted4 may request qualification hold')
 return receipt

def require_device(torch):
 value=dict(device_count=torch.cuda.device_count(),GPU=torch.cuda.get_device_name(),
     torch_version=torch.__version__,CUDA_version=torch.version.cuda)
 p.require(value['device_count']==1 and 'L40S' in value['GPU'] and value['CUDA_version']=='12.1'
     and value['torch_version']=='2.1.2+cu121','Actual CUDA device/count/runtime differs from qualified Babel site')
 return value

def pilot_readout(common,accepted,archived,horizon,exposure,lr_rows):
 from experiments.root_native_qualification_v10_20261024.observer import validate_lr_trajectory
 expected=p.horizon(33038)
 p.require(horizon['args_max_steps']==-1 and horizon['selected_count']==33038
     and horizon['native_max_steps']==258 and horizon['actual_resume_step']==0,
     'Only real fresh canonical uncapped258 horizon may produce pilot evidence')
 lr_check=validate_lr_trajectory(lr_rows,0,4,258)
 p.require(exposure['record_count']==512 and exposure['microbatches']==128
     and len(set(exposure['ordered_record_ids']))==512 and all(row['skipped_steps']==0 for row in lr_rows),
     'Pilot did not execute512 distinct records/four unskipped natural-schedule updates')
 return dict(status='COMPLETE_CANONICAL_STUDENT_PILOT_ACCEPTED_BOUNDARY_NOT_FULL_EPOCH',
     accepted_global_step=4,accepted=accepted,accepted_archive=archived,expected_horizon=expected,
     actual_args_max_steps=horizon['args_max_steps'],actual_selected_count=horizon['selected_count'],
     actual_native_max_steps=horizon['native_max_steps'],full_epoch_completed=False,
     actual_learning_rates=lr_check,DS_skipped_steps=0,actual_record_count=512,actual_microbatches=128,**common)

def run(config_ref,permit_ref,launch_ref,attempt,stop_at):
 """Called inside Root's authenticated torchrun GPU child; returns0 or75.

 Root must have durably registered actual current UID/job/nonce/spool/resources
 before invoking. This API never requeues or creates a scheduler request.
 """
 attempt=Path(attempt)
 config=p.checked(config_ref);profile=p.checked(config['site_profile'])
 from .launch import require_current
 current=require_current(launch_ref,config_ref,permit_ref,config,profile)
 from .launch import require_attempt
 require_attempt(attempt,config,profile,current,p.checked(launch_ref))
 attempt.mkdir(parents=True,exist_ok=False)
 p.seal(attempt/'ACTUAL_ALLOCATION.json',current)
 p.require(stop_at<=current['actual_EndTime']-config['allocation']['reserve_seconds'],'Stop deadline overruns actual reserve')
 fs=os.statvfs(profile['output_root'])
 storage=p.checked(config['storage_reservation'])
 guard=p.checked(p.checked(launch_ref)['current_storage_guard'])
 total=storage['checkpoint_transaction_bytes']*(storage['retained_accepted_states']+1)+p.TRANSIENT_BYTES
 p.require(guard['configuration']==config_ref and guard['config_identity']==p.n.run_identity(config)
     and guard['physical_job_id']==current['actual_job_id'] and guard['actual_restart']==int(os.environ.get('SLURM_RESTART_COUNT','0'))
     and guard['total_claim_bytes']==total and total==21<<30,'Actual current GPU semantic/own-state storage proof required')
 required=max(0,total-guard['own_materialized_allocated_lower_bound'])+storage['other_outstanding_reserved_bytes']
 p.require(storage['checkpoint_transaction_bytes']==p.CAPSULE_BYTES
     and fs.f_bavail*fs.f_frsize>=required and fs.f_favail>=4000,'Actual same-job remaining collective storage insufficient')
 p.seal(attempt/'ACTUAL_STORAGE.json',dict(available_bytes=fs.f_bavail*fs.f_frsize,available_inodes=fs.f_favail,
     actual_job_id=current['actual_job_id'],required_bytes=required,reservation=config['storage_reservation'],
     checkpoint_root=profile['output_root'],checkpoint_root_device=os.stat(profile['output_root']).st_dev))
 from .driver import accepted_cursor_guard
 history=p.checked(launch_ref)['accepted_prior_attempts']
 accepted_cursor_guard(guard,guard['latest_step'],history[-1] if history else None,config_ref)
 providers={name:p.checked(ref) for name,ref in config['providers'].items()}
 p.require(config['role']=='student' and os.environ['NATIVE_RUN_IDENTITY']==p.n.run_identity(config)
     and os.environ.get('SLURM_JOB_ID','').isdecimal() and os.environ.get('WORLD_SIZE')=='1',
     'Actual current world1 student identity required')
 p.require(time.time()<stop_at and Path(os.environ['NATIVE_ATTEMPT'])==attempt
     and Path(os.environ['NATIVE_STOP_PATH']).parent==attempt,'Actual attempt/stop deadline ownership differs')
 # Cheap truthful authorization precedes cold asset hashing/model/CUDA work.
 with authorization.scope(config,providers,profile,permit_ref) as permit:
  p.require(not (Path(config['output'])/'TRAIN_COMPLETE.json').exists(),'Existing complete trajectory must be collected, not replayed')
  if permit['phase']=='canonical_pilot':
   from .prefix import resume_cursor,validate_prior_history
   cursor=resume_cursor(config)
   validate_prior_history(config,p.checked(launch_ref).get('accepted_prior_attempts',[]),cursor,os.environ['SLURM_JOB_ID'])
  policy,trace,pre=runtime.install();p.seal(attempt/'PRE_CUDA_RUNTIME_POLICY.json',pre)
  import torch
  p.seal(attempt/'ACTUAL_DEVICE.json',require_device(torch))
  from experiments.root_native_campaign_v7_20261024 import native
  source=p.ROOT/p.n.NATIVE;sys.path.insert(0,str(source))
  patch=importlib.import_module('llava.train.llama_xformers_attn_monkey_patch');patch.replace_llama_attn_with_xformers_attn()
  module=importlib.import_module('llava.train.train')
  p.require(Path(module.__file__).resolve()==source/'llava/train/train.py','Foreign actual native trainer')
  parent=module.LLaVATrainer;audit=StudentAudit(config,permit,attempt)
  from .state_audit import install as observe_state
  module.LLaVATrainer=observe_state(audit.install(parent),config,attempt)
  original_validate=native.validate;original_cli=native.cli;original_argv=sys.argv[:]
  p.require(original_validate.__module__=='experiments.root_native_campaign_v7_20261024.config',
      'Unexpected prior native validate substitution')
  def cli(cfg,prof,prov):
   argv=student_cli(cfg,prof,prov)
   p.seal(attempt/'ACTUAL_STUDENT_ARGV.json',dict(argv=argv,native_cli_source=p.desc(Path(student_config.cli.__code__.co_filename)),
       no_max_steps=True,natural_epoch=True,config_identity=p.n.run_identity(config)))
   return argv
  native.validate=student_config.validate;native.cli=cli
  error=None
  try:
   runtime.require_current()
   sys.argv=['explicit-production-student','--config',config_ref['path'],'--stop-at-epoch',str(stop_at)]
   native.main()
  except BaseException as exc:error=repr(exc);traceback.print_exc()
  finally:
   native.validate=original_validate;native.cli=original_cli;module.LLaVATrainer=parent;sys.argv=original_argv
  evidence=audit.finish();operators=p.seal(attempt/'ACTUAL_RUNTIME_OPERATOR.json',runtime.finish(policy,trace))
  p.require(error is None,'Actual student failed; preserve all evidence, no automatic retry')
  return finish_native(config_ref,permit_ref,config,permit,launch_ref,attempt,evidence,operators,native.checkpoint_receipt)

def finish_native(config_ref,permit_ref,config,permit,launch_ref,attempt,evidence,operators,checkpoint_check):
 """Actual post-Native metadata path; continuation is authenticated by descriptor."""
 attempt=Path(attempt)
 p.require(config['execution_adapter']==dict(package=p.PACKAGE,sources=p.sources(),runtime=p.RUNTIME,natural_epoch=True,ordinary_native_launch_forbidden=True),'Newtarget logicalsource only; oldcheckpoint identities require their own recovery')
 common=dict(configuration=config_ref,config_identity=p.n.run_identity(config),adapter_sources=p.sources(),
     actual_job_id=os.environ['SLURM_JOB_ID'],actual_restart_count=int(os.environ.get('SLURM_RESTART_COUNT','0')),
     outer_authorization=permit_ref,runtime=p.RUNTIME,exposure=evidence,operators=operators,
     actual_cli=p.desc(attempt/'ACTUAL_STUDENT_ARGV.json'),qualification_promoted=False,
     scientific_performance_measured=False)
 complete=Path(config['output'])/'TRAIN_COMPLETE.json'
 continuation=attempt/'CONTINUATION.json'
 if complete.exists():
  p.require(permit['phase']=='production','Pilot must not complete full epoch before independent gate')
  final=p.read(complete);accepted=final['accepted'];header=p.checked(accepted)
  archived=p.seal(attempt/'ACCEPTED_HEADER_ARCHIVE.json',header)
  p.require(archived['sha256']==accepted['sha256'] and archived['bytes']==accepted['bytes'],'Final accepted archive changed')
  result=p.seal(attempt/'STUDENT_COMPLETE.json',dict(status='COMPLETE_STUDENT_TRAINING_NOT_EVALUATION',
      native_complete=p.desc(complete),accepted=accepted,accepted_archive=archived,**common));return 0,result
 p.require(continuation.exists(),'Native returned with neither accepted continuation nor complete endpoint')
 accepted=p.checked(p.desc(continuation))['accepted'];receipt=p.checked(accepted);step=receipt['global_step']
 path=Path(accepted['path']).parent;checkpoint_check(path,p.n.run_identity(config))
 archived=p.seal(attempt/'ACCEPTED_HEADER_ARCHIVE.json',receipt)
 p.require(archived['bytes']==accepted['bytes'] and archived['sha256']==accepted['sha256'],
     'Accepted metadata archive is not byte-identical to original GPU receipt')
 pilot=None
 pending_pilot=permit['phase']=='canonical_pilot' and step<4
 if permit['phase']=='canonical_pilot':p.require(0<step<=4,'Pilot continuation exceeded exact accepted4')
 segment=p.seal(attempt/'ACCEPTED_SEGMENT.json',dict(status='ACTUAL_ACCEPTED_STUDENT_SEGMENT_METADATA_V1',
     accepted=accepted,accepted_archive=archived,global_step=step,**common))
 if permit['phase']=='canonical_pilot' and step==4:
  from .prefix import aggregate
  prefix=aggregate(config,segment,p.checked(launch_ref).get('accepted_prior_attempts',[]))
  prefix_ref=p.seal(attempt/'PILOT_PREFIX_EVIDENCE.json',prefix)
  pilot=p.seal(attempt/'CANONICAL_PILOT_READOUT.json',pilot_readout(
      dict(common,pilot_prefix_evidence=prefix_ref),accepted,archived,prefix['initial_actual_horizon'],
      prefix['aggregate_exposure'],prefix['combined_actual_learning_rates']))
 value=dict(status='ACCEPTED_STUDENT_CONTINUATION_REQUIRES_ROOT_SAME_JOB_ACTION',
     checkpoint_directory=str(path),accepted=accepted,accepted_archive=archived,global_step=step,
     gate_hold_required=permit['phase']=='canonical_pilot' and step==4,
     pilot_progress_pending=pending_pilot,segment=segment,requires_atomic_pending_before_exit=True,
     pilot_readout=pilot,root_must_record_intent_and_same_job_requeue_before_exit=True,
     root_must_requeuehold_before_exit=permit['phase']=='canonical_pilot' and step==4,**common)
 validate_continuation(value,config,attempt,os.environ['SLURM_JOB_ID'],checkpoint_check)
 result=p.seal(attempt/'ACCEPTED_CONTINUATION_V1.json',value)
 return 75,result

def emit_exit(config_ref,code,receipt,attempt):
 """torchrun success is0; accepted75 is an authenticated logical outcome."""
 value=p.checked(receipt)
 p.require(code in (0,75) and value['configuration']==config_ref
     and value['config_identity']==p.n.run_identity(p.checked(config_ref)) and value['adapter_sources']==p.sources()
     and value['status']==('COMPLETE_STUDENT_TRAINING_NOT_EVALUATION' if code==0 else 'ACCEPTED_STUDENT_CONTINUATION_REQUIRES_ROOT_SAME_JOB_ACTION'),
     'Only genuine source-bound child success/accepted receipt may use successful process exit')
 return p.seal(Path(attempt)/'CHILD_EXIT.json',dict(schema='root-longwall-child-exit-v1',configuration=config_ref,
     config_identity=value['config_identity'],adapter_sources=p.sources(),logical_exit_code=code,receipt=receipt,torchrun_process_exit_code=0))

def main():
 a=argparse.ArgumentParser()
 for name in ('configuration','authorization','launch-binding','attempt'):a.add_argument('--'+name,required=True)
 a.add_argument('--stop-at-epoch',type=int,required=True);v=a.parse_args()
 code,receipt=run(p.desc(v.configuration),p.desc(v.authorization),p.desc(v.launch_binding),v.attempt,v.stop_at_epoch)
 outcome=emit_exit(p.desc(v.configuration),code,receipt,v.attempt)
 print(p.canonical(dict(logical_exit_code=code,receipt=receipt,child_exit=outcome)).decode());return 0
if __name__=='__main__':raise SystemExit(main())
