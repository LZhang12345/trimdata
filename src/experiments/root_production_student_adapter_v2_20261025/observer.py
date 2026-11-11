"""Actual student horizon/exposure/objective units; no changed loss or sampler."""
import math,os
from pathlib import Path
from . import protocol as p
from experiments.root_native_qualification_v10_20261024.observer import actual_lr_row,validate_lr_row

class StudentAudit:
 def __init__(self,config,permit,attempt):
  self.config=config;self.permit=permit;self.attempt=Path(attempt);self.output=self.attempt/'student_observations'
  self.output.mkdir(parents=True,exist_ok=False)
  selected=p.checked(config['providers']['selection']);pool=p.checked(config['providers']['pool'])
  index=dict(zip(pool['record_ids'],pool['source_indices']))
  self.ids=selected['record_ids'];self.indices=[index[v] for v in self.ids]
  self.expected=p.horizon(selected['count']);self.exposed=[];self.micro=0;self.denominator=None;self.loss=None
 def append(self,name,value):
  with (self.output/name).open('ab') as stream:stream.write(p.canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())
 def install(self,Parent):
  from transformers import TrainerCallback
  audit=self
  class Observed(Parent):
   def __init__(self,*args,**kwargs):
    super().__init__(*args,**kwargs);audit.trainer=self
    original=self.train_dataset;collate=self.data_collator
    class Indexed:
     @property
     def list_data_dict(self):return original.list_data_dict
     def __len__(self):return len(original)
     @property
     def lengths(self):return original.lengths
     @property
     def modality_lengths(self):return original.modality_lengths
     def __getitem__(self,index):return dict(original[index],_student_index=index)
    self.train_dataset=Indexed()
    def indexed_collate(instances):
     return dict(collate(instances),_student_indices=[v['_student_index'] for v in instances])
    self.data_collator=indexed_collate
    base=self.model.get_base_model();prepare=base.prepare_inputs_labels_for_multimodal
    def expanded(*args,**kwargs):
     result=prepare(*args,**kwargs);labels=result[5]
     p.require(labels is not None,'Missing actual expanded assistant targets')
     counts=(labels[...,1:]!=-100).sum(dim=-1).tolist()
     p.require(all(v>0 for v in counts),'Zero actual expanded target denominator')
     audit.denominator=dict(expanded_shifted_targets=sum(counts),per_record_shifted_targets=counts)
     return result
    base.prepare_inputs_labels_for_multimodal=expanded;self.add_callback(Events())
   def get_train_dataloader(self):
    loader=super().get_train_dataloader()
    p.seal(audit.output/'DATALOADER.json',dict(selected_count=len(self.train_dataset),loader_batches=len(loader),
        drop_last=bool(loader.drop_last),sampler=type(loader.sampler).__module__+'.'+type(loader.sampler).__qualname__))
    return loader
   def compute_loss(self,*args,**kwargs):
    value=super().compute_loss(*args,**kwargs);loss=value[0] if isinstance(value,tuple) else value
    audit.loss=float(loss.detach().cpu());return value
   def training_step(self,model,inputs):
    positions=inputs.pop('_student_indices');audit.denominator=None;audit.loss=None
    value=super().training_step(model,inputs)
    p.require(audit.denominator is not None and audit.loss is not None
        and math.isfinite(audit.loss) and math.isfinite(float(value.detach().cpu())),'Missing/nonfinite actual objective evidence')
    ids=[audit.ids[i] for i in positions];audit.exposed+=ids;audit.micro+=1
    audit.append('OBJECTIVE_UNITS.jsonl',dict(microbatch=audit.micro,record_ids=ids,
        original_source_indices=[audit.indices[i] for i in positions],microbatch_size=len(ids),
        native_unscaled_loss=audit.loss,returned_native_scaled_loss=float(value.detach().cpu()),
        trainer_step_before=int(self.state.global_step),declared_accumulation=32,**audit.denominator))
    return value
  class Events(TrainerCallback):
   def on_train_begin(self,args,state,control,**kwargs):
    loader=p.read(audit.output/'DATALOADER.json');expected=audit.expected
    p.require(args.max_steps==-1 and args.num_train_epochs==1 and args.gradient_accumulation_steps==32
        and args.per_device_train_batch_size==4 and state.max_steps==expected['natural_updates']
        and loader['loader_batches']==expected['loader_batches'] and loader['selected_count']==expected['selected_count']
        and loader['drop_last'] is False,'Actual student horizon/loader has cap or different policy')
    p.seal(audit.output/'ACTUAL_HORIZON.json',dict(expected=expected,args_max_steps=args.max_steps,
        native_max_steps=state.max_steps,selected_count=loader['selected_count'],actual_resume_step=int(state.global_step)))
    row=actual_lr_row(audit.trainer);validate_lr_row(row);audit.append('LEARNING_RATES.jsonl',row)
   def on_step_end(self,args,state,control,**kwargs):
    row=actual_lr_row(audit.trainer);validate_lr_row(row);audit.append('LEARNING_RATES.jsonl',row)
    if audit.permit['phase']=='canonical_pilot' and state.global_step==4:
     Path(os.environ['NATIVE_STOP_PATH']).touch()
   def on_save(self,args,state,control,**kwargs):
    # Native Boundaries handles semantic acceptance and actual early stop.
    p.seal(audit.output/('ACCEPTED_CURSOR-'+str(state.global_step)+'.json'),
        dict(global_step=int(state.global_step),native_max_steps=int(state.max_steps)))
  return Observed
 def finish(self):
  return p.seal(self.output/'ATTEMPT_EXPOSURE.json',dict(ordered_record_ids=self.exposed,
      record_count=len(self.exposed),microbatches=self.micro,expected=self.expected,
      rows_include_only_current_attempt=True,full_epoch_not_claimed=True))
