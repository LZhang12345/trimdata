"""Canary stops AFTER an ordinary accepted checkpoint, never requests a save.

Numerical Native91 / StudentAudit / state-audit bodies remain byte-identical.
This additional callback only publishes the ordinary accepted continuation and
uses Native91's existing nonnumerical AcceptedContinuation control exception.
"""
import os,time
from pathlib import Path
from . import protocol as p

def install(Parent,config,permit,attempt,stop_at,*,callback_base=None):
 p.require(permit['phase']=='shared_canary' and config['id']==p.CANARY_ID,
     'Only exact reviewed bootstrap target can use canary control')
 if callback_base is None:
  from transformers import TrainerCallback
  callback_base=TrainerCallback
 attempt=Path(attempt)
 class Boundary(callback_base):
  def on_save(self,args,state,control,**kwargs):
   step=int(state.global_step)
   p.require(args.max_steps==-1 and 0<step<int(state.max_steps)==config['expected_horizon']['natural_updates']
       and time.time()<stop_at and not Path(os.environ['NATIVE_STOP_PATH']).exists(),
       'Canary must observe ordinary accepted save, not forced deadline/final/cap')
   accepted=p.desc(Path(config['output'])/'training'/('checkpoint-'+str(step))/'ACCEPTED.json')
   header=p.checked(accepted)
   p.require(header['status']=='ACCEPTED_FULL_NATIVE_STATE' and header['identity']==p.n.run_identity(config)
       and header['global_step']==step,'Native semantic acceptance must precede canary stop')
   saved=p.checked(p.desc(attempt/('LONGWALL_STATE_SAVE-'+str(step)+'.json')))
   p.require(saved['configuration_identity']==p.n.run_identity(config) and saved['sources']==p.sources()
       and saved['state']['global_step']==step,'Unchanged state observer must run before canary callback')
   p.seal(attempt/'ORDINARY_ACCEPTED_CANARY_BOUNDARY_V4.json',dict(schema='babel-shared-runtime-canary-boundary-v4',
       configuration_identity=p.n.run_identity(config),sources=p.sources(),accepted=accepted,
       global_step=step,actual_job_id=os.environ['SLURM_JOB_ID'],actual_restart=int(os.environ.get('SLURM_RESTART_COUNT','0')),
       no_save_requested=True,no_scientific_step_cap=True,ordinary_checkpoint_seconds=config['checkpoint_seconds'],
       shared_checkpoint_root=p.CHECKPOINT_ROOT))
   p.seal(attempt/'CONTINUATION.json',dict(identity=p.n.run_identity(config),accepted=accepted))
   from experiments.root_native_campaign_v7_20261024.native import AcceptedContinuation
   raise AcceptedContinuation()
 class Observed(Parent):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs);self.add_callback(Boundary())
 return Observed
