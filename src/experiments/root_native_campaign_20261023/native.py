"""Additive lifecycle around native LLaVA; original dataset/masking/loss untouched.

UNQUALIFIED on GPUs until independent numerical, masking and restart tests pass.
The c121 upstream source is a declared adaptation of the COINCIDE author fork.
"""
import argparse
import importlib
import os
from pathlib import Path
import shutil
import sys
import time
from . import common as c
from .config import validate,cli

class Captured(Exception):
    def __init__(self,trainer):self.trainer=trainer

class AcceptedContinuation(Exception):pass

def checkpoint_receipt(path,identity):
    path=Path(path);accepted=c.read(path/'ACCEPTED.json')
    c.require(accepted['identity']==identity and accepted['status']=='ACCEPTED_FULL_NATIVE_STATE', 'Wrong checkpoint identity')
    for name,expected in accepted['files'].items():
        c.require(c.sha(path/name)==expected['sha256'] and (path/name).stat().st_size==expected['bytes'], 'Checkpoint changed: '+name)
    return accepted

def checkpoint_inventory(path,gpus):
    path=Path(path)
    files={str(p.relative_to(path)):dict(bytes=p.stat().st_size,sha256=c.sha(p)) for p in sorted(path.rglob('*')) if p.is_file() and p.name!='ACCEPTED.json'}
    names=set(files)
    c.require('trainer_state.json' in names, 'Missing trainer state')
    c.require('non_lora_trainables.bin' in names, 'Missing trainable projector')
    optim_names=[n for n in names if n.endswith('optim_states.pt')]
    model_names=[n for n in names if n.endswith('model_states.pt')]
    c.require(len(optim_names)==gpus, 'Missing/extra per-rank native ZeRO optimizer shards')
    c.require(len(model_names)==gpus, 'Missing/extra per-rank native ZeRO3 model shards')
    c.require(any(n in names for n in ('adapter_model.safetensors','adapter_model.bin')), 'Missing portable LoRA state')
    expected={f'rng_state_{i}.pth' for i in range(gpus)} if gpus>1 else {'rng_state.pth'}
    c.require(expected<=names, 'Incomplete per-rank Python/NumPy/Torch/CUDA RNG state')
    # Transformers/DeepSpeed stores scheduler and FP16 scaler in the DS state,
    # not necessarily standalone scheduler.pt/scaler.pt. These are trusted,
    # newly created checkpoints, never untrusted downloaded pickle files.
    import torch
    for name in model_names:
        state=torch.load(path/name,map_location='cpu')
        nested=isinstance(state.get('lr_scheduler'),dict)
        standalone='scheduler.pt' in names and isinstance(torch.load(path/'scheduler.pt',map_location='cpu'),dict)
        c.require((nested or standalone) and state.get('global_steps') is not None, 'Missing actual DS/native scheduler/step state')
    for name in optim_names:
        state=torch.load(path/name,map_location='cpu')['optimizer_state_dict']
        c.require(state.get('zero_stage')==3 and state.get('partition_count')==gpus, 'DS optimizer partition identity differs')
        c.require('loss_scaler' in state and 'optimizer_state_dict' in state and 'fp32_flat_groups' in state, 'Missing FP16 scaler/Adam/FP32 master state')
    return files

def retire_new_intermediates(training,report,identity,keep=2):
    """Only this run's newly accepted checkpoints; raw metadata stays in the report.

    Called collectively after the next state has been accepted. Never touches old
    project trees, unaccepted failures, final output, or externally owned states.
    """
    training=Path(training);accepted=[]
    for path in training.glob('checkpoint-*'):
        c.require(not path.is_symlink() and c.within(path,training),'Checkpoint symlink/path escape')
        if (path/'ACCEPTED.json').exists():
            receipt=checkpoint_receipt(path,identity)
            accepted.append((receipt['global_step'],path,receipt))
    for _,path,receipt in sorted(accepted)[:-keep]:
        # Full file list and receipt survive, including scientific failure evidence.
        target=Path(report)/'retired-new-intermediates'/path.name
        c.seal(target/'ACCEPTED_ARCHIVE.json',receipt)
        c.seal(target/'DELETION_INTENT.json',dict(identity=identity,path=str(path),files=receipt['files'],UTC=c.now()))
        for name in sorted(receipt['files'],key=lambda n:len(Path(n).parts),reverse=True):
            (path/name).unlink()
        (path/'ACCEPTED.json').unlink()
        for folder in sorted((p for p in path.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):folder.rmdir()
        path.rmdir()
        c.seal(target/'DELETION_RECEIPT.json',dict(identity=identity,path=str(path),deleted_logical_bytes=receipt['logical_bytes'],UTC=c.now()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--stop-at-epoch',type=int,required=True);a=p.parse_args()
    config=c.read(a.config);profile,providers=validate(config,gpu_check=True)
    import torch
    import torch.distributed as dist
    # Match the official train_xformers import order before importing train/Transformers.
    source=str(c.ROOT/c.NATIVE);sys.path.insert(0,source)
    patch=importlib.import_module('llava.train.llama_xformers_attn_monkey_patch')
    patch.replace_llama_attn_with_xformers_attn()
    module=importlib.import_module('llava.train.train')
    c.require(Path(module.__file__).resolve()==Path(source)/'llava/train/train.py', 'Foreign LLaVA module imported')
    from transformers import TrainerCallback
    output=Path(config['output']);training=output/'training';training.mkdir(parents=True,exist_ok=True)
    identity=os.environ['NATIVE_RUN_IDENTITY'];attempt=Path(os.environ['NATIVE_ATTEMPT'])
    rank=int(os.environ['RANK']);gpus=config['allocation']['gpus'];start=time.monotonic()
    Parent=module.LLaVATrainer
    if rank==0:c.seal(output/'OWNER.json',dict(identity=identity,owner='Root',schema='new-native-run-only-v1'))

    def barrier():
        if dist.is_initialized():dist.barrier()

    def from_rank_zero(value):
        values=[value if rank==0 else None]
        dist.broadcast_object_list(values,src=0)
        return values[0]

    class NativeTrainer(Parent):
        def train(self,*args,**kwargs):
            if getattr(self,'_capture',True):raise Captured(self)
            return super().train(*args,**kwargs)

        def _rotate_checkpoints(self,*args,**kwargs):
            # Acceptance and retirement are our transaction, never HF pre-acceptance deletion.
            return

        def _save_checkpoint(self,model,trial,metrics=None):
            step=int(self.state.global_step);path=training/f'checkpoint-{step}'
            exists=from_rank_zero((path/'ACCEPTED.json').exists())
            if exists:
                if rank==0:checkpoint_receipt(path,identity)
                barrier();return
            existing_unaccepted=from_rank_zero(path.exists())
            c.require(not existing_unaccepted, 'Unaccepted checkpoint target exists: preserve failed attempt, do not overwrite')
            # Collective cache-space reservation comes from an independent current receipt.
            storage=c.checked(config['storage_reservation'])
            c.require(storage['status']=='PASS_EXCLUSIVE_MEASURED_STORAGE_RESERVATION' and storage['output']==config['output'], 'No current measured storage reservation')
            c.require(shutil.disk_usage(output).free>=storage['checkpoint_transaction_bytes'], 'Insufficient checkpoint free space')
            # HF4.37.2 excludes frozen weights only for PEFT + compatible DS signature.
            # Fail rather than silently duplicating the7B frozen base in each checkpoint.
            import inspect
            from transformers.trainer import _is_peft_model
            c.require(_is_peft_model(self.model) and 'exclude_frozen_parameters' in inspect.signature(self.model_wrapped.save_checkpoint).parameters,
                'DeepSpeed save must support PEFT frozen-base exclusion')
            if not storage.get('allow_new_intermediate_retirement',False):
                retained=from_rank_zero(len(list(training.glob('checkpoint-*/ACCEPTED.json'))))
                c.require(retained<2, 'Two retained states reached; reviewed owned-state retirement is required before another checkpoint')
            super()._save_checkpoint(model,trial,metrics)
            # Gather all trainable non-LoRA parameters on ALL ranks (ZeRO3 collective).
            extra=module.get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters())
            if rank==0:
                c.require(extra and all('mm_projector' in name for name in extra), 'Unexpected non-LoRA trainable support')
                torch.save(extra,path/'non_lora_trainables.bin')
            barrier()
            if rank==0:
                files=checkpoint_inventory(path,gpus)
                logical=sum(d['bytes'] for d in files.values())
                c.require(logical<=storage['checkpoint_transaction_bytes'], 'Measured checkpoint exceeds reservation')
                for name in files:
                    with open(path/name,'rb') as stream:os.fsync(stream.fileno())
                c.seal(path/'ACCEPTED.json',dict(status='ACCEPTED_FULL_NATIVE_STATE',identity=identity,
                    global_step=step,epoch=self.state.epoch,files=files,world_size=gpus,
                    logical_bytes=logical,UTC=c.now(),original_native_sampler_and_loss=True))
                c.seal(attempt/f'CHECKPOINT-{step}.json',dict(accepted=c.desc(path/'ACCEPTED.json')))
            barrier()
            if storage.get('allow_new_intermediate_retirement',False):
                if rank==0:retire_new_intermediates(training,config['report'],identity)
                barrier()

        def _load_from_checkpoint(self,resume_from_checkpoint,model=None):
            checkpoint_receipt(resume_from_checkpoint,identity)
            super()._load_from_checkpoint(resume_from_checkpoint,model=model)
            extra=torch.load(Path(resume_from_checkpoint)/'non_lora_trainables.bin',map_location='cpu')
            parameters=dict(self.model.named_parameters())
            expected={name for name,param in parameters.items() if param.requires_grad and 'lora_' not in name}
            c.require(set(extra)==expected, 'Resume projector support differs')
            # Trainer's DeepSpeed restore path may bypass this method. The DS checkpoint
            # includes projector weights; actual qualification MUST test that route.
            from deepspeed import zero
            for name in sorted(extra):
                with zero.GatheredParameters([parameters[name]],modifier_rank=0):
                    if rank==0:
                        parameters[name].data.copy_(extra[name].to(parameters[name].device))

    class Boundaries(TrainerCallback):
        def on_step_end(self,args,state,control,**kwargs):
            due=time.time()>=a.stop_at_epoch or Path(os.environ['NATIVE_STOP_PATH']).exists()
            flag=torch.tensor([int(due)],device=trainer.args.device)
            dist.all_reduce(flag,op=dist.ReduceOp.MAX)
            self.stop=bool(flag.item()) and state.global_step<state.max_steps
            # Save final/stop states and elapsed-time accepted boundaries, not every step.
            periodic=time.monotonic()-getattr(self,'last_saved',start)>=config['checkpoint_seconds']
            save_flag=torch.tensor([int(self.stop or periodic or state.global_step>=state.max_steps)],device=trainer.args.device)
            dist.all_reduce(save_flag,op=dist.ReduceOp.MAX)
            control.should_save=bool(save_flag.item())
            return control

        def on_save(self,args,state,control,**kwargs):
            self.last_saved=time.monotonic()
            if getattr(self,'stop',False):
                if rank==0:
                    c.seal(attempt/'CONTINUATION.json',dict(identity=identity,
                        accepted=c.desc(training/f'checkpoint-{state.global_step}'/'ACCEPTED.json')))
                barrier();raise AcceptedContinuation()
            return control

    module.LLaVATrainer=NativeTrainer
    module.transformers.set_seed(config['training_seed'])
    sys.argv=['native-c121-xformers']+cli(config,profile,providers)
    try:
        module.train()
        raise ValueError('Native constructor failed to enter trainer')
    except Captured as captured:trainer=captured.trainer
    finally:module.LLaVATrainer=Parent
    trainer._capture=False
    c.require(dist.is_initialized() and dist.get_world_size()==gpus,'Actual distributed runtime not initialized')
    c.require(trainer.args.world_size*trainer.args.per_device_train_batch_size*trainer.args.gradient_accumulation_steps==128,'Actual global batch differs')
    names=[name for name,param in trainer.model.named_parameters() if param.requires_grad]
    c.require(any('mm_projector' in n for n in names) and not any('vision_tower' in n or 'lm_head' in n for n in names),'Trainable support differs')
    c.require(all('lora_' in n or 'mm_projector' in n for n in names),'Unexpected trainable base parameter')
    peft=trainer.model.peft_config['default']
    c.require(peft.r==128 and peft.lora_alpha==256 and peft.lora_dropout==.05 and peft.bias=='none','Actual LoRA configuration differs')
    support={n.split('.lora_')[0].split('.')[-1] for n in names if 'lora_' in n}
    c.require(support=={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'},'Native language LoRA targets differ')
    # Add an identity/valid-target guard at the lazy loading seam, without changing
    # native tokenization, assistant masking, collator, loss, sampling or images.
    base=trainer.train_dataset
    image_assets=c.checked(providers['pool']['image_manifest'])['assets']
    image_root=Path(providers['pool']['image_root'])
    from PIL import Image
    import hashlib
    class VerifiedDataset:
        def __len__(self):return len(base)
        @property
        def lengths(self):return base.lengths
        @property
        def modality_lengths(self):return base.modality_lengths
        def __getitem__(self,index):
            row=base.list_data_dict[index]
            if row.get('image'):
                path=(image_root/row['image']).absolute();expected=image_assets[row['image']]
                c.require(c.within(path,image_root) and str(path)==expected['path'] and c.sha(path)==expected['sha256'],'Image identity changed at native loading')
                with Image.open(path) as image:
                    rgb=image.convert('RGB')
                    c.require(rgb.size==(expected['width'],expected['height']) and hashlib.sha256(rgb.tobytes()).hexdigest()==expected['rgb_sha256'],'Decoded image/RGB identity changed')
            value=base[index]
            c.require(int((value['labels'][1:]!=-100).sum())>0,'All supervised assistant targets were masked/truncated')
            return value
    trainer.train_dataset=VerifiedDataset()
    if rank==0:
        c.seal(attempt/'NATIVE_MODEL_AUDIT.json',dict(trainable_names=names,lora_targets=sorted(support),
            native_masking_is_unmodified=True,masking_efficacy_requires_independent_GPU_audit=True,
            global_batch=128,recipe=config['recipe'],official_source_adaptation=c.PIN))
    trainer.add_callback(Boundaries())
    # Refuse any partial/foreign checkpoint; never let HF pick an unaccepted directory.
    candidates=[]
    if rank==0:
        for path in training.glob('checkpoint-*'):
            c.require((path/'ACCEPTED.json').exists(),'Unaccepted checkpoint remains: quarantine requires separate reviewed repair; no replay overwrite')
            if (path/'ACCEPTED.json').exists():
                receipt=checkpoint_receipt(path,identity);candidates.append((receipt['global_step'],path))
    resume=from_rank_zero(str(max(candidates)[1]) if candidates else None)
    try:
        trainer.train(resume_from_checkpoint=resume)
    except AcceptedContinuation:
        return
    c.require(trainer.state.global_step==trainer.state.max_steps,'Native trajectory stopped before expected endpoint')
    trainer._save_checkpoint(trainer.model,None)
    barrier()
    if rank==0:
        accepted=c.desc(training/f'checkpoint-{trainer.state.global_step}'/'ACCEPTED.json')
        c.seal(output/'TRAIN_COMPLETE.json',dict(status='COMPLETE_NATIVE_TRAINING_ONLY',identity=identity,
            accepted=accepted,global_step=trainer.state.global_step,epoch=trainer.state.epoch,
            role=config['role'],selected_count=providers['selection']['count'],
            evaluation_complete=False,qualification_promoted=False,original_native_sampler_may_drop_partial_accumulation=True))
    barrier()

if __name__=='__main__':main()
