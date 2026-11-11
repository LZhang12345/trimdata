"""Explicit qualifier CLI adaptation and observations; no changed native loss."""
from contextlib import contextmanager
import json,math,time
from pathlib import Path
from . import protocol as p

def natural_cli(native_arguments):
 args=list(native_arguments);positions=[i for i,x in enumerate(args) if x=='--max_steps']
 p.require(len(positions)==1 and positions[0]+1<len(args) and args[positions[0]+1]=='9','Only declared qualifier cap9 can be removed')
 index=positions[0];out=args[:index]+args[index+2:]
 p.require('--max_steps' not in out and '--num_train_epochs' in out and out[out.index('--num_train_epochs')+1]=='1','Natural oneepoch must preserve original native recipe')
 return out

@contextmanager
def native_cli_adapter(native,study,attempt):
 original=native.cli
 def declared(config,profile,providers):
  result=original(config,profile,providers)
  if study['specification']['natural_epoch']:result=natural_cli(result)
  p.seal(Path(attempt)/'ACTUAL_NATIVE_ARGV.json',dict(original_qualifier_argv=original(config,profile,providers),executed_argv=result,natural_epoch=study['specification']['natural_epoch'],no_other_cli_change=True))
  return result
 native.cli=declared
 try:yield
 finally:native.cli=original

class EpochAudit:
 def __init__(self,study,config,output):
  self.study=study;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True);self.current=None;self.denominator=None;self.raw_loss=None
  pool=p.checked(config['providers']['pool']);self.source_indices=pool['source_indices'];self.record_ids=pool['record_ids']
 def append(self,name,value):
  import os
  with (self.output/name).open('ab') as stream:stream.write(p.canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())
 def install(self,Parent):
  from transformers import TrainerCallback
  audit=self;spec=self.study['specification']
  class Observed(Parent):
   def __init__(self,*args,**kwargs):
    super().__init__(*args,**kwargs);base=self.model.get_base_model();original=base.prepare_inputs_labels_for_multimodal
    def prepared(*args,**kwargs):
     result=original(*args,**kwargs);labels=result[5]
     p.require(labels is not None,'Actual expanded labels absent')
     audit.denominator=dict(expanded_shifted_valid_targets=int((labels[...,1:]!=-100).sum().item()),per_record_expanded_shifted_targets=[int(v) for v in (labels[...,1:]!=-100).sum(dim=-1).tolist()],expanded_shape=list(labels.shape))
     p.require(audit.denominator['expanded_shifted_valid_targets']>0 and all(v>0 for v in audit.denominator['per_record_expanded_shifted_targets']),'Zero actual shifted target denominator')
     return result
    base.prepare_inputs_labels_for_multimodal=prepared;self.add_callback(Events());audit.trainer=self
   def get_train_dataloader(self):
    loader=super().get_train_dataloader();p.seal(audit.output/'ACTUAL_DATALOADER.json',dict(selected_count=len(self.train_dataset),loader_batches=len(loader),drop_last=bool(loader.drop_last),sampler=type(loader.sampler).__module__+'.'+type(loader.sampler).__qualname__,gradient_accumulation_steps=self.args.gradient_accumulation_steps,original_num_train_epochs=self.args.num_train_epochs,actual_args_max_steps=self.args.max_steps));return loader
   def compute_loss(self,*args,**kwargs):
    value=super().compute_loss(*args,**kwargs);loss=value[0] if isinstance(value,tuple) else value;audit.raw_loss=float(loss.detach().cpu());return value
   def training_step(self,model,inputs):
    indices=list(inputs['_engineering_record_indices']);unexpanded=int((inputs['labels'][...,1:]!=-100).sum().item());audit.denominator=None;audit.raw_loss=None
    value=super().training_step(model,inputs)
    p.require(audit.denominator is not None and audit.raw_loss is not None,'Actual forward denominator/loss evidence missing')
    audit.append('ACTUAL_OBJECTIVE_UNITS.jsonl',dict(fixture_indices=indices,original_source_indices=[audit.source_indices[i] for i in indices],record_ids=[audit.record_ids[i] for i in indices],microbatch_size=len(indices),unexpanded_shifted_valid_targets=unexpanded,**audit.denominator,native_unscaled_loss=audit.raw_loss,returned_native_scaled_loss=float(value.detach().cpu()),declared_accumulation=32,scaling_policy='UNCHANGED_NATIVE_MEAN_VALID_SHIFTED_TARGETS_PER_MICROBATCH_THEN_NATIVE_GA32',trainer_global_step_before=int(self.state.global_step)))
    return value
  class Events(TrainerCallback):
   def on_train_begin(self,args,state,control,**kwargs):
    loader=p.read(audit.output/'ACTUAL_DATALOADER.json');expected=spec['total_steps']
    p.require(loader['selected_count']==spec['count'] and loader['gradient_accumulation_steps']==32 and state.max_steps==expected and args.num_train_epochs==1,'Actual loader/epoch/horizon differs')
    if spec['natural_epoch']:p.require(args.max_steps==-1 and loader['loader_batches']==math.ceil(1153/4) and loader['loader_batches']//32==9 and loader['drop_last'] is False,'No natural oneepoch floor9 endpoint; cap forbidden')
    else:p.require(args.max_steps==8,'ACES1024 fixed engineering8 cap must remain explicit')
    p.seal(audit.output/'ACTUAL_HORIZON.json',dict(status='ACTUAL_DECLARED_EPOCH_HORIZON',selected_count=spec['count'],actual_loader_batches=loader['loader_batches'],actual_max_steps=state.max_steps,args_max_steps=args.max_steps,num_train_epochs=args.num_train_epochs,natural_epoch=spec['natural_epoch'],tail_policy='UNCHANGED_NATIVE_FLOOR_UPDATES_NO_TOPUP_NO_RESCALE'))
   def on_train_end(self,args,state,control,**kwargs):p.seal(audit.output/'ACTUAL_STOP.json',dict(global_step=int(state.global_step),state_max_steps=int(state.max_steps),epoch=float(state.epoch),control_should_training_stop=bool(control.should_training_stop),args_max_steps=args.max_steps,num_train_epochs=args.num_train_epochs,natural_epoch=spec['natural_epoch']))
  return Observed
