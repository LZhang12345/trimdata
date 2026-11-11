"""Only native positional-token → PEFT071 inputs= call-binding compatibility.

Original PEFT.generate remains responsible for native prepare hook installation,
exception restoration and base dispatch. No tensor conversion, decoding change,
merge, direct base.generate shortcut or parameter/buffer write is performed.
"""
import importlib,importlib.metadata,inspect
from pathlib import Path
from experiments.root_gqa_bank_qualification_v1_20261026 import worker as u
from experiments.root_native_campaign_v7_20261024 import common as n

PEFT_VERSION='0.7.1'
PEFT_MODEL_SHA='baeb93af9c9a75f1987f47e32e0ae512ea1b3b41158548d18296f8e408d74872'
PEFT_INIT_SHA='15aa49cc585e46520bba019b8161b82481e2dc40569f3e27cc1404618fc6aa02'
NATIVE_PATH=n.ROOT/n.NATIVE/'llava/model/language_model/llava_llama.py'
NATIVE_SHA='00ab011e0543bb502e3ce26d402899022a4bc866364de1d645678cba853d05cc'
POLICY=dict(status='PINNED_PEFT071_NATIVE_POSITIONAL_INPUTS_BRIDGE_V1',peft_version=PEFT_VERSION,
    peft_class='peft.peft_model.PeftModelForCausalLM',peft_source_sha256=PEFT_MODEL_SHA,peft_init_sha256=PEFT_INIT_SHA,
    native_class='llava.model.language_model.llava_llama.LlavaLlamaForCausalLM',native_source_sha256=NATIVE_SHA,
    official_positional_audit_remains_outside_bridge=True,positional_argument_zero_to_keyword='inputs',
    original_bound_PEFT_generate_invoked=True,direct_base_generate_called=False,weight_merge_performed=False,
    tensor_conversion_performed=False,generation_controls_changed=False)

def same_method(a,b):
    return getattr(a,'__self__',None)is getattr(b,'__self__',None)and getattr(a,'__func__',a)is getattr(b,'__func__',b)

def active(model):
    u.require(model.active_adapters==['default']and model.peft_config['default'].inference_mode is True,
        'Exact active default inference adapter required')
    layers=[m for m in model.modules()if hasattr(m,'lora_A')]
    u.require(layers and all(not m.merged and not m.disable_adapters for m in layers),'Native LoRA must remain active and unmerged')

def api(model):
    peft=importlib.import_module('peft');module=importlib.import_module('peft.peft_model')
    u.require(importlib.metadata.version('peft')==PEFT_VERSION and type(model)is module.PeftModelForCausalLM,
        'Exact installed PEFT071 causal model required')
    original=model.generate;native=model.get_base_model();native_module=importlib.import_module('llava.model.language_model.llava_llama')
    u.require(getattr(original,'__self__',None)is model and getattr(original,'__func__',None)is module.PeftModelForCausalLM.generate
        and type(native)is native_module.LlavaLlamaForCausalLM and getattr(native.generate,'__func__',None)is native_module.LlavaLlamaForCausalLM.generate,
        'Unwrapped pinned original PEFT and native generate methods required')
    paths=dict(peft_model=Path(inspect.getsourcefile(module.PeftModelForCausalLM)),peft_init=Path(peft.__file__),
        native=Path(inspect.getsourcefile(native_module.LlavaLlamaForCausalLM)))
    for k,expected in(('peft_model',PEFT_MODEL_SHA),('peft_init',PEFT_INIT_SHA),('native',NATIVE_SHA)):
        u.require(n.sha(paths[k])==expected,'Pinned generation API source changed: '+k)
    ps=inspect.signature(original);ns=inspect.signature(native.generate)
    u.require(list(ps.parameters)==['kwargs']and ps.parameters['kwargs'].kind is inspect.Parameter.VAR_KEYWORD,
        'PEFT071 generate keyword-only API changed')
    u.require(list(ns.parameters)==['inputs','images','image_sizes','kwargs']and ns.parameters['kwargs'].kind is inspect.Parameter.VAR_KEYWORD
        and all(ns.parameters[k].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for k in('inputs','images','image_sizes')),
        'Pinned native inputs API changed')
    u.require(same_method(model.base_model.prepare_inputs_for_generation,model.base_model_prepare_inputs_for_generation),
        'Original canonical PEFT saved preparation hook required')
    active(model)
    return original,dict(policy=POLICY,actual_sources={k:n.desc(path)for k,path in paths.items()},
        original_PEFT_signature=str(ps),native_signature=str(ns))

class PositionalInputsBridge:
    def __init__(self,model,original,proof):
        self.model=model;self.original=original;self.proof=proof;self.records=[]
    def __call__(self,*args,**kwargs):
        u.require(len(args)==1 and not({'inputs','input_ids','inputs_embeds'}&set(kwargs)),
            'Exactly original one-positional-token native call; no duplicate/alternate inputs')
        model=self.model;active(model)
        saved=model.base_model_prepare_inputs_for_generation
        u.require(same_method(model.base_model.prepare_inputs_for_generation,saved),'Foreign preparation hook before native generation')
        call=dict(kwargs,inputs=args[0]);record=dict(input_keyword='inputs',input_object_unchanged=call['inputs']is args[0],
            all_keyword_objects_unchanged=all(call[k]is v for k,v in kwargs.items()),keyword_names=sorted(kwargs),
            original_PEFT_generate_invoked=True,success=False,prepare_hook_restored=False,output_object_unchanged=False)
        try:
            output=self.original(**call);record['success']=True;record['output_object_unchanged']=True
            return output
        finally:
            record['prepare_hook_restored']=same_method(model.base_model.prepare_inputs_for_generation,saved)
            self.records.append(record)
            u.require(record['prepare_hook_restored'],'Original PEFT preparation hook did not restore')
            active(model)

def install(model):
    u.require(not hasattr(model,'_dsmm_native_inputs_bridge'),'Generation bridge already installed')
    original,proof=api(model);bridge=PositionalInputsBridge(model,original,proof)
    model.generate=bridge;model._dsmm_native_inputs_bridge=bridge
    return proof

def readout(model):
    bridge=model._dsmm_native_inputs_bridge
    u.require(type(bridge)is PositionalInputsBridge and model.generate is bridge,'Exact installed positional bridge must be restored after outer audit')
    active(model);u.require(same_method(model.base_model.prepare_inputs_for_generation,model.base_model_prepare_inputs_for_generation),
        'Original PEFT preparation hook must remain restored')
    return dict(status='COMPLETE_PINNED_PEFT071_NATIVE_POSITIONAL_BRIDGE_CALLS',policy=POLICY,
        runtime_API_proof=bridge.proof,calls=len(bridge.records),records=list(bridge.records),adapter_active=True,adapter_merged=False)
