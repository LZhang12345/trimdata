"""Observe the unchanged native training math, data order and accepted boundaries."""
import inspect
import json
import resource
import os
from pathlib import Path
import time
from . import common as c
from . import state
from .masking import author_v1

def expanded_labels(ids,labels,attention,image_token,image_patches,max_length):
    """Independent image-span expansion oracle; no native preprocessing call."""
    output=[]
    for tokens,targets,mask in zip(ids,labels,attention):
        row=[]
        for token,target,keep in zip(tokens,targets,mask):
            if not keep:continue
            row.extend([-100]*image_patches if token==image_token else [target])
        output.append(row[:max_length])
    width=max(map(len,output))
    return [row+[-100]*(width-len(row)) for row in output]

class Audit:
    def __init__(self,config,phase,output,interrupt_step):
        self.config=config;self.phase=phase;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.interrupt_step=interrupt_step;self.micro=0;self.inputs=[];self.step_inputs=[]
        self.mask_count=0;self.expanded_target_count=0;self.mask_failures=0
        self.gradients={};self.attention_calls={};self.kernel_calls=0;self.noncausal_kernel_calls=0;self.started=time.monotonic();self.audit_seconds=0.;self.training_started=None
        self.base_rows=c.checked(c.checked(config['providers']['pool'])['records_json'])
        self.record_ids=c.checked(config['providers']['pool'])['record_ids'];self.current_indices=[]

    def append(self,name,value):
        with open(self.output/name,'ab') as stream:
            stream.write(c.canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())

    def install(self,Parent):
        import torch
        from transformers import TrainerCallback
        audit=self
        class ObservedTrainer(Parent):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs);audit.trainer=self
                original=self.train_dataset
                class Indexed:
                    list_data_dict=original.list_data_dict
                    def __len__(self):return len(original)
                    @property
                    def lengths(self):return original.lengths
                    @property
                    def modality_lengths(self):return original.modality_lengths
                    def __getitem__(self,index):return dict(original[index],_engineering_record_index=index)
                self.train_dataset=Indexed();collate=self.data_collator
                def indexed_collate(instances):
                    indices=[row['_engineering_record_index'] for row in instances]
                    return dict(collate(instances),_engineering_record_indices=indices)
                self.data_collator=indexed_collate
                c.require(not self.args.remove_unused_columns,'Native metadata observation requires existing remove_unused_columns=False')
                base=self.model.get_base_model();original_prepare=base.prepare_inputs_labels_for_multimodal
                import llava.conversation as conversation
                from transformers.models.llama.modeling_llama import LlamaAttention
                template=conversation.default_conversation.copy()
                attention=[]
                for name,layer in base.named_modules():
                    if not isinstance(layer,LlamaAttention):continue
                    forward=layer.forward
                    origin=dict(name=name,cls=type(layer).__module__+'.'+type(layer).__qualname__,
                        function_module=forward.__module__,function_name=forward.__name__,
                        source_file=forward.__code__.co_filename)
                    c.require(type(layer).__name__=='LlamaAttention' and forward.__module__=='llava.train.llama_xformers_attn_monkey_patch'
                        and forward.__name__=='xformers_forward','Declared xFormers objective bypassed by actual attention implementation')
                    attention.append(origin);audit.attention_calls[name]=0
                    def count_attention(_module,_inputs,name=name):audit.attention_calls[name]+=1
                    layer.register_forward_pre_hook(count_attention)
                c.require(len(attention)==32,'Actual7B attention-layer census differs')
                import xformers.ops as ops
                kernel=ops.memory_efficient_attention
                def traced_kernel(query,key,value,*args,**kwargs):
                    audit.kernel_calls+=1
                    bias=kwargs.get('attn_bias',args[0] if args else None)
                    causal=isinstance(bias,ops.LowerTriangularMask)
                    audit.noncausal_kernel_calls+=int(not causal)
                    c.require(causal,'Actual declared autoregressive training attention became noncausal')
                    c.require(query.dtype==key.dtype==value.dtype==torch.float16 and query.is_cuda,
                        'Actual xFormers kernel input is not declared CUDAFP16')
                    if audit.kernel_calls==1:
                        c.seal(audit.output/'KERNEL_FIRST_CALL.json',dict(function=kernel.__module__+'.'+kernel.__name__,
                            query_shape=list(query.shape),key_shape=list(key.shape),value_shape=list(value.shape),
                            dtype=str(query.dtype),device=str(query.device),attn_bias=type(bias).__module__+'.'+type(bias).__qualname__,
                            actual_original_xformers_operator_called=True))
                    return kernel(query,key,value,*args,**kwargs)
                ops.memory_efficient_attention=traced_kernel
                c.seal(audit.output/'ATTENTION.json',dict(attention=attention,
                    actual_implementation=getattr(base.config,'_attn_implementation',None),
                    actual_model_class=type(base).__module__+'.'+type(base).__qualname__))
                patches=base.get_vision_tower().num_patches
                c.require(patches==576,'Expected actual ViT336 native576visualpatches')
                def traced_prepare(input_ids,position_ids,attention_mask,past_key_values,labels,images,image_sizes=None):
                    before=time.monotonic()
                    original_ids=input_ids.detach().cpu().tolist();original_labels=labels.detach().cpu().tolist()
                    for index,ident in enumerate(audit.current_indices):
                        oracle=author_v1(audit.base_rows[ident],self.tokenizer,template)
                        native_ids=original_ids[index][:min(len(oracle['input_ids']),2048)]
                        expected_ids=oracle['input_ids'][:2048]
                        native_targets=original_labels[index][:len(expected_ids)]
                        c.require(native_ids==expected_ids and native_targets==oracle['labels'][:2048],
                            'Actual native assistant masks/input IDs differ from independently reconstructed original conversation')
                        c.require(all(v==-100 for v in original_labels[index][len(expected_ids):]),'Padding became supervised')
                    attention=attention_mask.detach().cpu().tolist() if attention_mask is not None else [[True]*len(row) for row in original_ids]
                    out=original_prepare(input_ids,position_ids,attention_mask,past_key_values,labels,images,image_sizes)
                    actual=out[-1].detach().cpu().tolist()
                    expected=expanded_labels(original_ids,original_labels,attention,-200,patches,2048)
                    matches=actual==expected;audit.mask_count+=len(actual)
                    audit.expanded_target_count+=sum(sum(v!=-100 for v in row[1:]) for row in actual)
                    if not matches:audit.mask_failures+=1
                    sample=dict(micro=audit.micro,record_ids=[audit.record_ids[i] for i in audit.current_indices],
                        native_labels_sha256=c.key(actual),independent_expected_sha256=c.key(expected),matches=matches,
                        unexpanded_targets=sum(sum(v!=-100 for v in row[1:]) for row in original_labels),
                        expanded_targets=sum(sum(v!=-100 for v in row[1:]) for row in actual),image_patches=patches)
                    if audit.micro<=8:
                        sample.update(original_input_ids=original_ids,native_unexpanded_labels=original_labels,
                            native_expanded_labels=actual,independent_expanded_labels=expected,
                            original_conversations=[audit.base_rows[i]['conversations'] for i in audit.current_indices],
                            decoded_native_target_text=[self.tokenizer.decode([v for v in row if v!=-100]) for row in actual])
                    audit.append('MASKING.jsonl',sample)
                    audit.audit_seconds+=time.monotonic()-before
                    c.require(matches,'Native image expansion/padding/truncation masked labels differ from independent oracle')
                    return out
                base.prepare_inputs_labels_for_multimodal=traced_prepare
                # Observe oneLoRABparameter per projection plus projector, without
                # accumulating/changing any gradient or consuming randomness.
                names={}
                for name,param in self.model.named_parameters():
                    if not param.requires_grad:continue
                    key=None
                    if '.lora_B.' in name:key=name.split('.lora_')[0].split('.')[-1]
                    elif 'mm_projector' in name and name.endswith('weight'):key='projector'
                    if key and key not in names:names[key]=(name,param)
                c.require(set(names)=={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','projector'},'Gradient observation support incomplete')
                for key,(name,param) in names.items():
                    audit.gradients[key]=dict(name=name,calls=0,finite_calls=0,nonzero_finite_calls=0,nonfinite_calls=0)
                    def observe(gradient,key=key):
                        row=audit.gradients[key];row['calls']+=1
                        finite=bool(torch.isfinite(gradient).all())
                        row['finite_calls']+=int(finite);row['nonfinite_calls']+=int(not finite)
                        if finite:row['nonzero_finite_calls']+=int(bool((gradient!=0).any()))
                        return gradient
                    param.register_hook(observe)
                self.add_callback(Events())

            def create_optimizer(self):
                result=super().create_optimizer()
                if not hasattr(self,'_engineering_optimizer_audited'):
                    named={id(p):name for name,p in self.model.named_parameters() if p.requires_grad};seen=[];groups=[]
                    for group in self.optimizer.param_groups:
                        names=[named[id(p)] for p in group['params']]
                        for name in names:
                            expected=2e-5 if 'mm_projector' in name else 2e-4
                            c.require(group['lr']==expected,'Actual parameter group LR differs: '+name)
                        seen+=names;groups.append(dict(names=names,learning_rate=group['lr'],weight_decay=group['weight_decay']))
                    c.require(set(seen)==set(named.values()) and len(seen)==len(set(seen)),'Optimizer omits/duplicates trainables')
                    c.seal(audit.output/'PARAMETER_GROUPS.json',dict(status='ACTUAL_NATIVE_GROUPS_MATCH',groups=groups,
                        native_optimizer_class=type(self.optimizer).__module__+'.'+type(self.optimizer).__qualname__))
                    self._engineering_optimizer_audited=True
                return result

            def training_step(self,model,inputs):
                indices=inputs.pop('_engineering_record_indices');audit.current_indices=indices
                audit.micro+=1
                if audit.micro==1:
                    c.seal(audit.output/'FIRST_MICRO_STATE.json',state.snapshot(self))
                    audit.training_started=time.monotonic()
                ids=[audit.record_ids[i] for i in indices];audit.inputs+=ids;audit.step_inputs+=ids
                before=time.monotonic();result=super().training_step(model,inputs)
                loss=float(result.detach().cpu())
                c.require(torch.isfinite(result).all(),'Nonfinite actual native training loss')
                audit.append('MICROBATCHES.jsonl',dict(micro=audit.micro,global_step_before=int(self.state.global_step),
                    record_ids=ids,returned_native_scaled_loss=loss,observed_step_seconds=time.monotonic()-before))
                return result

        class Events(TrainerCallback):
            def on_train_begin(self,args,state_,control,**kwargs):
                torch.cuda.reset_peak_memory_stats()
                c.require(torch.backends.cuda.matmul.allow_tf32 is False,'Actual TF32 matmul setting differs from recipe')
                c.seal(audit.output/'TRAIN_BEGIN.json',dict(global_step=int(state_.global_step),
                    actual_TF32_matmul=torch.backends.cuda.matmul.allow_tf32,actual_TF32_cudnn=torch.backends.cudnn.allow_tf32,
                    actual_torch_version=torch.__version__,
                    actual_DS_save_signature=str(inspect.signature(audit.trainer.model_wrapped.save_checkpoint)),
                    actual_DS_load_signature=str(inspect.signature(audit.trainer.model_wrapped.load_checkpoint)),
                    actual_DS_engine_class=type(audit.trainer.model_wrapped).__module__+'.'+type(audit.trainer.model_wrapped).__qualname__,
                    GPU=torch.cuda.get_device_name(),allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
                    parameter_dtype_counts=dict(__import__('collections').Counter(str(p.dtype) for p in audit.trainer.model.parameters()))))

            def on_step_end(self,args,state_,control,**kwargs):
                started=time.monotonic();snapshot=state.snapshot(audit.trainer)
                snapshot.update(record_ids=audit.step_inputs[:],gradients=dict(audit.gradients),
                    mask_records=audit.mask_count,mask_failures=audit.mask_failures,
                    expanded_assistant_targets=audit.expanded_target_count,CUDA_peak_allocated=torch.cuda.max_memory_allocated(),
                    CUDA_peak_reserved=torch.cuda.max_memory_reserved())
                c.seal(audit.output/f'STEP-{state_.global_step}.json',snapshot);audit.step_inputs=[]
                audit.audit_seconds+=time.monotonic()-started
                if audit.phase=='interrupted' and state_.global_step==audit.interrupt_step:
                    Path(os.environ['NATIVE_STOP_PATH']).touch()
                return control

            def on_save(self,args,state_,control,**kwargs):
                checkpoint=Path(audit.config['output'])/'training'/f'checkpoint-{state_.global_step}'
                saved=torch.load(checkpoint/'rng_state.pth',map_location='cpu')
                saved_hash=state.digest(state.normalize_saved_rng(saved));current_hash=state.digest(state.rng_state())
                c.require(saved_hash==current_hash,'Saved RNG differs at the accepted native boundary')
                c.seal(audit.output/f'ACCEPTED-{state_.global_step}.json',dict(accepted=c.desc(checkpoint/'ACCEPTED.json'),
                    saved_RNG_sha256=saved_hash,current_RNG_sha256=current_hash,exact_RNG_match=True))
                return control

        return ObservedTrainer

    def finish(self,status):
        import torch
        elapsed=time.monotonic()-self.started
        return c.seal(self.output/'OBSERVATION.json',dict(status=status,phase=self.phase,
            actual_microbatches=self.micro,actual_record_exposure=len(self.inputs),ordered_record_ids=self.inputs,
            masks_checked=self.mask_count,mask_failures=self.mask_failures,expanded_assistant_targets=self.expanded_target_count,
            gradient_support=self.gradients,actual_attention_forward_calls=self.attention_calls,actual_xformers_kernel_calls=self.kernel_calls,noncausal_kernel_calls=self.noncausal_kernel_calls,elapsed_seconds=elapsed,observer_overhead_seconds=self.audit_seconds,
            observed_training_seconds=time.monotonic()-self.training_started if self.training_started is not None else None,
            observed_records_per_second=len(self.inputs)/elapsed if elapsed else None,
            CUDA_peak_allocated=torch.cuda.max_memory_allocated(),CUDA_peak_reserved=torch.cuda.max_memory_reserved(),
            process_CPU_peak_RSS_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            includes_observation_overhead=True,production_throughput_not_established=True))
