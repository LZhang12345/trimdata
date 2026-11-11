"""Narrow native checkpoint restore and actual attention-class assertions.

Frozen base bytes are reconstructed from the independently verified pinned assets.
Only the exact frozen-parameter keys deliberately omitted by the native DS saver
may be missing; trainable parameters and persistent buffers remain strict.
The original HF/DS optimizer, scheduler, FP16 scaler/master and RNG paths are kept.
Actual GPU parity is required; CPU/API tests alone never qualify native restart.
"""
from contextlib import contextmanager
import json
import math
from pathlib import Path
from . import common as c

def persisted_record(value):
    """Exact JSON-domain identity, with explicit collision/nonfinite rejection.

    HF configs contain integer id2label keys; JSON checkpoint persistence turns
    them into strings. Compare the same persisted representation on both sides.
    No fields or values are omitted and no model/checkpoint loading is relaxed.
    """
    def visit(item):
        if isinstance(item,dict):
            names=[]
            for key,child in item.items():
                c.require(type(key) in (str,int), 'Unsupported reconstruction mapping key')
                names.append(str(key));visit(child)
            c.require(len(names)==len(set(names)), 'Ambiguous reconstruction JSON mapping keys')
        elif isinstance(item,(list,tuple)):
            for child in item:visit(child)
        else:
            c.require(item is None or type(item) in (str,bool,int,float),
                'Unsupported reconstruction JSON value')
            if type(item) is float:c.require(math.isfinite(item),'Nonfinite reconstruction JSON value')
    visit(value)
    return json.loads(c.canonical(value))


def reconstruction_record(model, config):
    from .optimizer_groups import model_inventory
    def entry(name,param):
        shape=getattr(param,'ds_shape',param.shape)
        return dict(name=name,shape=list(shape),dtype=str(param.dtype))
    return persisted_record(dict(schema='native-frozen-reconstruction-v1',assets=config['providers']['assets'],
        recipe=config['recipe'],official_commit=c.PIN,
        frozen=[entry(n,p) for n,p in model.named_parameters() if not p.requires_grad],
        trainable=[entry(n,p) for n,p in model.named_parameters() if p.requires_grad],
        native_config=model.config.to_dict(),optimizer_group_inventory=model_inventory(model)))

def restore_exact_trainable_and_buffers(src,dst):
    """Mirror DS0.12.6 module_state_dict's exact frozen-name exclusion."""
    current=dst.state_dict()
    frozen={n for n,p in dst.named_parameters() if not p.requires_grad and n in current}
    c.require(frozen,'No frozen support: this adapter is only for excluded-frozen PEFT checkpoints')
    expected=set(current)-frozen
    c.require(set(src)==expected,'Saved module keys differ from exact trainable/buffer support')
    for name in expected:
        c.require(src[name].dtype==current[name].dtype,'Checkpoint dtype differs: '+name)
        c.require(tuple(src[name].shape)==tuple(current[name].shape),'Checkpoint shape differs: '+name)
    # PyTorch still raises shape mismatches with strict=False. Only missing-key
    # enforcement is delegated to the exact explicit inventory below.
    incompatible=dst.load_state_dict(src,strict=False)
    c.require(not incompatible.unexpected_keys and set(incompatible.missing_keys)==frozen,
        'Only deliberately excluded frozen parameters may be missing')
    return dict(loaded_keys=sorted(expected),verified_frozen_missing_keys=sorted(frozen))

def assert_attention(model,patch):
    from transformers.models.llama.modeling_llama import LlamaAttention
    modules=[(n,m) for n,m in model.named_modules() if isinstance(m,LlamaAttention)]
    c.require(len(modules)==model.config.num_hidden_layers,'Missing decoder attention layers')
    rows=[]
    for name,module in modules:
        forward=getattr(module.forward,'__func__',None)
        c.require(type(module) is LlamaAttention and forward is patch.xformers_forward,
            'HF automatic attention bypassed the declared patched eager implementation')
        rows.append(dict(name=name,type=type(module).__module__+'.'+type(module).__name__,
            forward_module=forward.__module__,forward_name=forward.__name__))
    return dict(status='PASS_CONSTRUCTED_PATCHED_EAGER_ATTENTION_CLASS_ONLY',layers=rows,
        actual_xformers_kernel_invocation_still_requires_GPU_evidence=True)

@contextmanager
def install_verified_restore(config,identity,attempt,checkpoint_receipt):
    import transformers.trainer as trainer_module
    original=trainer_module.deepspeed_load_checkpoint
    def verified(engine,checkpoint_path):
        checkpoint=Path(checkpoint_path)
        checkpoint_receipt(checkpoint,identity)
        saved=c.read(checkpoint/'FROZEN_RECONSTRUCTION.json')
        current=reconstruction_record(engine.module,config)
        if saved!=current:
            c.seal(Path(attempt)/'RECONSTRUCTION_MISMATCH.json',dict(
                saved_checkpoint=str(checkpoint/'FROZEN_RECONSTRUCTION.json'),
                current_reconstruction=current,
                differing_top_level_fields=[k for k in sorted(set(saved)|set(current))
                    if saved.get(k)!=current.get(k)]))
        c.require(saved==current,
            'Current pinned frozen reconstruction/model support differs from accepted state')
        loader=engine.load_module_state_dict
        calls=[]
        def scoped(checkpoint,strict=True,custom_load_fn=None,fetch_z3_params=False):
            c.require(strict is True and custom_load_fn is None and fetch_z3_params is False,
                'Unqualified DeepSpeed restore mode')
            c.require(checkpoint.get('frozen_param_fragments') is None,
                'Frozen fragments unexpectedly present in excluded-frozen checkpoint')
            def exact(src,dst):
                c.require(dst is engine.module,'Foreign restore destination')
                calls.append(restore_exact_trainable_and_buffers(src,dst))
            # Preserve original DeepSpeed gather/fragments/optimizer behavior;
            # narrow only module missing-key handling through its supported hook.
            return loader(checkpoint,strict=True,custom_load_fn=exact,fetch_z3_params=False)
        engine.load_module_state_dict=scoped
        try:
            result=original(engine,checkpoint_path)
        finally:
            engine.load_module_state_dict=loader
        c.require(len(calls)==1,'Expected exactly one verified module restore per rank')
        import os
        c.seal(Path(attempt)/('VERIFIED_DS_RESTORE-rank'+os.environ.get('RANK','0')+'.json'),
            dict(identity=identity,checkpoint=str(checkpoint_path),module_restore=calls[0],
                optimizer_scheduler_scaler_master_RNG_parity_requires_actual_GPU_qualification=True))
        return result
    trainer_module.deepspeed_load_checkpoint=verified
    try:yield
    finally:trainer_module.deepspeed_load_checkpoint=original
