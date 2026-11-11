"""Reconstruct pinned Stage1 + accepted PEFT/non-LoRA state without base copies."""
import gc
import importlib.metadata
import json
import re
from pathlib import Path

from experiments.root_gqa_bank_qualification_v1_20261026 import worker as u
from experiments.root_native_campaign_v7_20261024 import common as n
from experiments.root_native_state_guards_20261023 import checkpoint as semantic
from experiments.session6_start_model_qualification_v4_eager_20261024 import model as stage1,worker as base_identity
from experiments.session6_native_start_execution_v6_20261024 import identity as identity_io
from experiments.copilot7_coincide_native_exec_v2_20261028 import protocol as student
from experiments.root_native_trained_loader_v4_20261026 import generation

def portable_support(reconstruction,lora,non_lora):
    import torch
    trainable={r['name']:r for r in reconstruction['trainable']}
    u.require(len(trainable)==len(reconstruction['trainable']) and trainable,'Exact unique native trainable support required')
    names={name for name in trainable if '.lora_A.' in name or '.lora_B.' in name}
    projector={name for name in trainable if 'mm_projector' in name}
    u.require(names and projector and names.isdisjoint(projector) and names|projector==set(trainable),'Only native LoRA and projector trainables supported')
    portable={semantic._portable_name(name):name for name in names}
    u.require(len(portable)==len(names) and set(lora)==set(portable) and set(non_lora)==projector,'Portable trainable keys differ')
    expected={portable[k]:v for k,v in lora.items()};expected.update(non_lora)
    for name,value in expected.items():
        r=trainable[name]
        u.require(torch.is_tensor(value) and list(value.shape)==r['shape'] and str(value.dtype)==r['dtype']=='torch.float16'
            and bool(torch.isfinite(value).all()),'Portable tensor shape/dtype/finite values differ: '+name)
    return expected

def accepted_checkpoint(accepted_ref,training_ref):
    import torch
    training=n.checked(training_ref);accepted=n.checked(accepted_ref);path=u.literal(Path(accepted_ref['path']).parent)
    student.supported(training)
    u.require(accepted['status']=='ACCEPTED_FULL_NATIVE_STATE' and accepted['identity']==n.run_identity(training)
        and accepted['world_size']==1 and type(accepted['global_step']) is int and accepted['global_step']==516
        and path==Path(training['output'])/'training'/('checkpoint-'+str(accepted['global_step']))
        and Path(training['output'])==Path(student.CHECKPOINT_ROOT)/training['id']
        and training['execution_adapter']==dict(package=student.PACKAGE,sources=student.sources(),
            runtime=student.RUNTIME,natural_epoch=True,ordinary_native_launch_forbidden=True),
        'Only exact final516 COINCIDE trained checkpoint in its separately granted namespace')
    u.require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Actual oneGPU semantic state qualification required')
    proof=semantic.validate_checkpoint_directory(path,accepted['identity'],cuda_state_sizes=[int(torch.cuda.get_rng_state(0).numel())])
    original=n.checked(accepted['semantic_validation'])
    u.require(proof['status']==original['status']=='PASS_SEMANTIC_NATIVE_WORLD1_ZERO3_STATE'
        and proof['global_steps']==original['global_steps']==accepted['global_step'] and proof['skipped_steps']==0
        and original['files_verified_before_and_after'] is True,'Accepted original/current semantic proofs differ or updates skipped')
    reconstruction=n.read(path/'FROZEN_RECONSTRUCTION.json')
    u.require(reconstruction['schema']=='native-frozen-reconstruction-v1' and reconstruction['assets']==training['providers']['assets']
        and reconstruction['recipe']==training['recipe'] and reconstruction['official_commit']==n.PIN,'Frozen Stage1/recipe reconstruction differs')
    return path,training,accepted,reconstruction,proof

def verify_parameter_layout(rows,reconstruction):
    inventory=reconstruction['frozen']+reconstruction['trainable']
    named={r['name']:r for r in rows}
    u.require(len(rows)==len(named) and len(inventory)==len({r['name'] for r in inventory}) and set(named)=={r['name'] for r in inventory},'Loaded model parameter names differ from frozen reconstruction')
    for r in inventory:
        p=named[r['name']]
        u.require(p['shape']==r['shape'] and p['dtype']==r['dtype']=='torch.float16','Reconstructed parameter shape/dtype differs: '+r['name'])
    return dict(exact_parameter_inventory=True,frozen_parameters=len(reconstruction['frozen']),trained_parameters=len(reconstruction['trainable']))

def verify_loaded_parameters(model,reconstruction,expected):
    import torch
    named=dict(model.named_parameters())
    layout=verify_parameter_layout([dict(name=k,shape=list(v.shape),dtype=str(v.dtype))for k,v in named.items()],reconstruction)
    for name,value in expected.items():u.require(torch.equal(named[name].detach().cpu(),value),'Portable trained value not exactly loaded: '+name)
    return dict(**layout,portable_trainable_values_exact=True,LoRA_tensors=sum('.lora_' in k for k in expected),projector_tensors=sum('mm_projector' in k for k in expected))

def native_adapter_targets(reconstruction):
    pairs={}
    pattern=r'base_model\.model\.(model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj)))\.lora_([AB])\.default\.weight'
    for row in reconstruction['trainable']:
        if '.lora_'not in row['name']:continue
        match=re.fullmatch(pattern,row['name']);u.require(match is not None,'Unexpected saved native language adapter support')
        target,side=match.groups();u.require(side not in pairs.setdefault(target,set()),'Duplicate saved adapter side');pairs[target].add(side)
    u.require(pairs and all(sides=={'A','B'}for sides in pairs.values()),'Exhaustive native A/B adapter pairs required')
    return set(pairs)

def attach_native_adapter(base,adapter,reconstruction,factory):
    """Mirror native train.py: inject PEFT before vision linear modules exist.

    The same fully loaded frozen vision object is held aside and restored in a
    finally block. No tensor/config/target policy is replaced or remapped.
    """
    import torch
    expected=native_adapter_targets(reconstruction);generic=adapter['target_modules']
    u.require(type(generic)is list and len(generic)==len(set(generic))
        and set(generic)=={name.rsplit('.',1)[-1]for name in expected},'Original saved generic target policy differs from trained support')
    core=base.get_model();vision=base.get_vision_tower()
    u.require(core._modules.get('vision_tower')is vision and vision is not None and vision.is_loaded is True,
        'One exact fully loaded original vision object required')
    def candidates():return {name for name,module in base.named_modules()if name.rsplit('.',1)[-1]in generic and isinstance(module,torch.nn.Linear)}
    initial=candidates();u.require(expected<=initial and all(name.startswith('model.vision_tower.')for name in initial-expected),
        'Unexpected pre-injection target support outside original vision namespace')
    core.vision_tower=None
    try:
        u.require(candidates()==expected,'Visible native injection targets differ from exact saved language support')
        result=factory(base)
    finally:core.vision_tower=vision
    u.require(result.get_model()is core and result.get_vision_tower()is vision and core._modules['vision_tower']is vision,
        'Exact original frozen vision object must be restored before use')
    return result,dict(native_training_order_preserved=True,vision_object_restored_identically=True,
        original_saved_target_policy_unchanged=True,language_adapter_targets=sorted(expected),
        loaded_vision_generic_targets_excluded=sorted(initial-expected),no_forward_during_structural_injection=True)

def restore_persistent_buffers(model,state,trainable_names):
    """Restore only authenticated DS persistent buffers, no parameter overwrite."""
    import torch
    names=state['buffer_names'];module=state['module']
    u.require(type(names) is list and all(type(k) is str for k in names) and len(names)==len(set(names))
        and set(names).isdisjoint(trainable_names) and set(module)==set(trainable_names)|set(names),
        'Saved persistent-buffer/trainable DS support differs')
    current=model.state_dict();parameters=set(dict(model.named_parameters()))
    expected={k for k in dict(model.named_buffers()) if k in current}
    u.require(set(current)-parameters==expected==set(names),'Exact current/saved persistent buffer inventory differs')
    buffers={k:module[k] for k in names}
    for k,value in buffers.items():
        u.require(torch.is_tensor(value) and value.shape==current[k].shape and value.dtype==current[k].dtype
            and bool(torch.isfinite(value).all()),'Saved persistent buffer shape/dtype/finite values differ: '+k)
    if buffers:
        incompatible=model.load_state_dict(buffers,strict=False)
        u.require(not incompatible.unexpected_keys and set(incompatible.missing_keys)==set(current)-set(buffers),
            'Persistent-buffer load changed support outside exact saved buffers')
    actual=model.state_dict()
    u.require(all(torch.equal(actual[k].detach().cpu(),v) for k,v in buffers.items()),'Saved persistent buffer not exactly restored')
    return dict(exact_persistent_buffer_support=True,persistent_buffer_values_exact=True,
        restored_buffer_names=sorted(names),restored_buffer_count=len(names),parameters_overwritten=False)

def authenticated_archive(path,accepted_file,loader):
    ref=dict(path=str(path),**accepted_file)
    u.require(n.desc(path)==ref,'Accepted tensor archive changed before actual load')
    value=loader(path)
    u.require(n.desc(path)==ref,'Accepted tensor archive changed during actual load')
    return value,ref

def compose(accepted_ref,training_ref,diagnostic_path=None):
    import torch
    import peft
    u.require(importlib.metadata.version('peft')=='0.7.1','Exact qualified native PEFT0.7.1 required')
    path,training,accepted,reconstruction,proof=accepted_checkpoint(accepted_ref,training_ref)
    assets=stage1.verified_assets(training['providers']['assets'],'/data/hf_cache')
    gc.collect();tokenizer,base,processor=stage1.compose(assets,config_path=path)
    from transformers import AutoTokenizer
    saved_tokenizer=AutoTokenizer.from_pretrained(str(path),use_fast=False,model_max_length=2048,padding_side='right',local_files_only=True)
    fields=('special_tokens_map','pad_token_id','eos_token_id','model_max_length','padding_side')
    u.require(saved_tokenizer.get_vocab()==tokenizer.get_vocab()
        and all(getattr(saved_tokenizer,k)==getattr(tokenizer,k) for k in fields),
        'Accepted trained tokenizer differs from exact frozen Stage1 tokenizer policy')
    tokenizer=saved_tokenizer
    names={'adapter_model.safetensors','adapter_model.bin'}&set(accepted['files'])
    u.require(len(names)==1,'One accepted portable LoRA archive required')
    portable_name=next(iter(names))
    if portable_name.endswith('.safetensors'):
        from safetensors.torch import load_file
        load_lora=lambda p:load_file(str(p),device='cpu')
    else:load_lora=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    lora,lora_ref=authenticated_archive(path/portable_name,accepted['files'][portable_name],load_lora)
    non_lora,non_lora_ref=authenticated_archive(path/'non_lora_trainables.bin',accepted['files']['non_lora_trainables.bin'],
        lambda p:torch.load(p,map_location='cpu',weights_only=False))
    expected=portable_support(reconstruction,lora,non_lora)
    adapter=n.read(path/'adapter_config.json');recipe=training['recipe']
    u.require(adapter['r']==recipe['lora_r']==128 and adapter['lora_alpha']==recipe['lora_alpha']==256
        and adapter['lora_dropout']==recipe['lora_dropout']==0.05 and adapter['bias']=='none' and adapter['task_type']=='CAUSAL_LM', 'Accepted native adapter configuration differs')
    model,injection=attach_native_adapter(base,adapter,reconstruction,
        lambda value:peft.PeftModel.from_pretrained(value,str(path),adapter_name='default',is_trainable=False,local_files_only=True))
    current=model.state_dict();u.require(set(non_lora)<=set(current),'Missing exact trained projector parameters')
    incompatible=model.load_state_dict(non_lora,strict=False)
    u.require(not incompatible.unexpected_keys and set(incompatible.missing_keys)==set(current)-set(non_lora),'Only non-LoRA trained parameters may overwrite the Stage1 base')
    model.to(device='cuda:0',dtype=torch.float16).eval().requires_grad_(False)
    if diagnostic_path is not None:
        expected_names={r['name']for r in reconstruction['frozen']+reconstruction['trainable']}
        observed=[dict(name=k,shape=list(v.shape),dtype=str(v.dtype))for k,v in model.named_parameters()]
        actual_names={r['name']for r in observed}
        u.publish(Path(diagnostic_path),dict(observed=observed,expected_count=len(expected_names),actual_count=len(observed),
            missing=sorted(expected_names-actual_names),extra=sorted(actual_names-expected_names),injection=injection))
    verification=verify_loaded_parameters(model,reconstruction,expected)
    state_names=[k for k in accepted['files'] if k.endswith('model_states.pt')]
    u.require(len(state_names)==1,'One exact accepted world1 DS model-state archive required')
    state_path=path/state_names[0]
    ds_state,expected_state=authenticated_archive(state_path,accepted['files'][state_names[0]],
        lambda p:torch.load(p,map_location='cpu',weights_only=False))
    buffer_verification=restore_persistent_buffers(model,ds_state,set(expected))
    u.require(model.active_adapters==['default'] and hasattr(model,'peft_config') and model.peft_config['default'].inference_mode is True,
        'Exact active inference adapter required; never merge or label START')
    layers=[m for m in model.modules() if hasattr(m,'lora_A')]
    u.require(layers and all(not m.merged and not m.disable_adapters for m in layers),'LoRA must stay active and unmerged')
    u.require(not model.training and all(not p.requires_grad and p.dtype==torch.float16 and p.device.type=='cuda' for p in model.parameters()),'Inference-only FP16 model required')
    u.require(n.desc(path/'ACCEPTED.json')==accepted_ref,'Accepted checkpoint header changed after reconstruction')
    del lora,non_lora,expected,ds_state;gc.collect()
    return (tokenizer,model,processor),dict(semantic_state=proof,portable_value_verification=verification,accepted=accepted_ref,
        native_adapter_construction=injection,
        persistent_buffer_verification=buffer_verification,persistent_buffer_archive=expected_state,
        actual_portable_archive_reads=[lora_ref,non_lora_ref],
        training_configuration=training_ref,training_optimizer_updates=accepted['global_step'],successful_updates=proof['successful_updates'],
        native_recipe=recipe,adapter_active=True,adapter_merged=False,no_frozen_base_saved=True)

def identity(loaded,proof,runtime):
    tokenizer,model,processor=loaded
    d=base_identity.diagnostic(loaded)
    u.require(d['predicates']['causal_classes_eager'] and model.config._attn_implementation=='eager','Matched official native eager evaluation required')
    value=dict(schema='native-trained-active-peft-evaluation-model-identity-v1',accepted=proof['accepted'],training_configuration=proof['training_configuration'],
        training_optimizer_updates=proof['training_optimizer_updates'],successful_updates=proof['successful_updates'],adapter_active=True,adapter_merged=False,
        source_commit=n.PIN,parameter_sha256=stage1.parameter_digest(model),buffers=base_identity.buffer_manifest(model),
        generation_config=stage1.generation_configuration(model),native_config=model.config.to_dict(),runtime=runtime,
        tokenizer=dict(length=len(tokenizer),vocabulary_sha256=n.key(tokenizer.get_vocab()),special_tokens=tokenizer.special_tokens_map,
            pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,model_max_length=tokenizer.model_max_length,padding_side=tokenizer.padding_side,class_name=type(tokenizer).__name__),
        processor=processor.to_dict(),language_attention=model.config._attn_implementation,
        vision_attention=model.get_vision_tower().vision_tower.config._attn_implementation,
        matched_START_component_protocol_only_not_START_model=True)
    u.require(value['generation_config']['num_beams']==1,'Inherited native greedy beam policy differs')
    return identity_io.json_domain(value)


def compose_v4(accepted_ref,training_ref,diagnostic_path=None):
    loaded,proof=compose(accepted_ref,training_ref,diagnostic_path)
    proof=dict(proof,generation_API_bridge=generation.install(loaded[1]))
    return loaded,proof


def identity_v4(loaded,proof,runtime):
    value=identity(loaded,proof,runtime)
    value['generation_API_bridge']=proof['generation_API_bridge']
    return value


def qualified_body_proof():
    import ast
    from experiments.root_native_trained_loader_v3_20261026 import model as original
    before={node.name:node for node in ast.parse(Path(original.__file__).read_bytes()).body
            if isinstance(node,ast.FunctionDef)}
    after={node.name:node for node in ast.parse(Path(__file__).read_bytes()).body
           if isinstance(node,ast.FunctionDef)}
    names=set(before)-{'accepted_checkpoint'}
    u.require(all(ast.dump(before[name],include_attributes=False)==ast.dump(after[name],include_attributes=False)
                  for name in names),'Qualified loader numerical/reconstruction body changed')
    return dict(anchor=n.desc(original.__file__),target=n.desc(__file__),
        unchanged_function_bodies=sorted(names),path_guard_only='COINCIDE_FINAL516_EXACT_NATIVE_CONFIGURATION',
        generation_bridge_source=n.desc(generation.__file__),
        actual_new_GPU_qualification_claimed=False)
