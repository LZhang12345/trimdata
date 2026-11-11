"""Observe exact engine state at restore/save; unchanged qualified snapshot code."""
from pathlib import Path
import time
from . import protocol as p
from experiments.root_native_qualification_v10_20261024.state import snapshot

def install(Parent,config,attempt):
 from transformers import TrainerCallback
 attempt=Path(attempt);holder={}
 class Events(TrainerCallback):
  def record(self,kind):
   value=snapshot(holder['trainer'])
   return p.seal(attempt/('LONGWALL_STATE_'+kind+'-'+str(value['global_step'])+'.json'),dict(
       kind=kind,configuration_identity=p.n.run_identity(config),sources=p.sources(),state=value,observed_epoch=time.time(),
       observer_does_not_consume_RNG=True,no_optimizer_or_objective_change=True))
  def on_save(self,args,state,control,**kwargs):self.record('SAVE')
 class Observed(Parent):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs);holder['trainer']=self;self._longwall_events=Events();self.add_callback(self._longwall_events)
  def training_step(self,*args,**kwargs):
   # HF4.37 restores saved RNG after on_train_begin/data skipping. Observe at
   # the unchanged qualified Native10 first-micro boundary, before parent work.
   if not getattr(self,'_longwall_first_micro_observed',False):
    self._longwall_events.record('BEGIN');self._longwall_first_micro_observed=True
   return super().training_step(*args,**kwargs)
 return Observed
