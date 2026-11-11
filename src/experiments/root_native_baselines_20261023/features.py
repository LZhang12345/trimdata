"""Native published feature acquisition, once per full-pool measurement function.

COINCIDE retains fixed four author chunks, batch8, shuffle42, layer activations,
prompt31 and729 vision positions. DataTailor retains the full unpadded pre-final-
norm token matrix, FP32 SVD and native BF16/TF32 forward measurement. Neither is
the project's cheap feature proxy. Actual GPU qualification remains mandatory.
"""
import ast
import hashlib
import importlib
import math
from pathlib import Path
import sqlite3
import sys
import types
import numpy as np
from experiments.root_native_campaign_20261023 import common as c
from . import pins,operators
from .store import Store


def logical_pool(pool):
    images=c.checked(pool['image_manifest'])['assets']
    semantic={name:{k:value[k] for k in ('sha256','rgb_sha256','width','height')}
              for name,value in images.items()}
    return c.key(dict(records_sha256=pool['records_json']['sha256'],record_ids=pool['record_ids'],
        source_indices=pool['source_indices'],images=semantic,
        protected_exclusions_sha256=pool['protected_exclusions']['sha256']))


def measurement_identity(config,pool):
    return c.key(dict(method=config['method'],logical_pool=logical_pool(pool),
        assets=config['feature_assets'],environment=config['environment'],
        sources=config['sources'],measurement_contract=config['measurement_contract']))


CONTRACTS={
    'coincide':dict(proxy='TinyLLaVA-2.0B',precision='fp16',layers=[3,7,11,15,19],
        hook='self_attn.o_proj_output_activations',prompt_len=31,vision_positions=729,
        global_shuffle=42,native_chunks=4,batch_size=8,conversation='phi',embedding_dimension=20480,
        tanh=True,per_layer_modality_L2=True,normalization_sqrt=10,
        author_nan_policy='replace_NaN_with_zero_after_float16_cast_and_record_count'),
    'datatailor':dict(proxy='Stage1_Vicuna7B_CLIP336_projector',precision='bf16',tf32=True,
        microbatch=1,hidden='full_unpadded_last_decoder_before_final_RMSNorm',
        hidden_dimension=4096,svd='torch.svd_float32',singular_normalization='L1',
        entropy='author_Python_probability_log_loop_stored_float32',
        tau='maximum_normalized_singular_value_stored_float32',last_valid_token=True,
        frozen=True,optimizer=None,training_updates=0)}


def parameter_hash(model):
    import torch
    result={}
    for name,value in model.named_parameters():
        raw=value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
        result[name]=dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(raw).hexdigest())
    return c.key(result)


def tensor_hash(value):
    import torch
    return hashlib.sha256(value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def batches(ids,method,index,total):
    """A scheduling shard never changes the author's batching or padding peers."""
    c.require(0<=index<total,'Invalid acquisition shard')
    if method=='datatailor':
        return [[i] for i,ident in enumerate(ids)
                if int(c.key(['datatailor-native-shard-v1',ident]),16)%total==index]
    c.require(method=='coincide' and total%4==0,'COINCIDE shards must stripe four fixed author chunks')
    shuffled=np.arange(len(ids));np.random.RandomState(42).shuffle(shuffled)
    chunk_size=math.ceil(len(ids)/4);chunk=index//(total//4);lane=index%(total//4)
    positions=shuffled[chunk*chunk_size:(chunk+1)*chunk_size]
    return [positions[start:start+8].tolist() for j,start in enumerate(range(0,len(positions),8))
            if j%(total//4)==lane]


def verify_image(pool,row):
    if not row.get('image'):return
    from PIL import Image
    name=row['image']
    c.require('_verified_image_assets' in pool,'Image manifest must be verified once before acquisition')
    expected=pool['_verified_image_assets'][name]
    path=(Path(pool['image_root'])/name).absolute()
    c.require(c.within(path,pool['image_root']) and str(path)==expected['path'] and c.sha(path)==expected['sha256'],'Encoded native image changed')
    with Image.open(path) as image:
        rgb=image.convert('RGB')
        c.require(rgb.size==(expected['width'],expected['height']) and hashlib.sha256(rgb.tobytes()).hexdigest()==expected['rgb_sha256'],'Decoded native RGB changed')


def datatailor(config,pool,rows,ids,store,should_stop,engineering=False):
    import torch
    from experiments.root_native_campaign_20261023.config import cli
    source=str(c.ROOT/c.NATIVE);sys.path.insert(0,source)
    patch=importlib.import_module('llava.train.llama_xformers_attn_monkey_patch')
    patch.replace_llama_attn_with_xformers_attn()
    module=importlib.import_module('llava.train.train')
    c.require(Path(module.__file__).resolve()==Path(source)/'llava/train/train.py','Wrong native LLaVA source')
    class Captured(Exception):
        def __init__(self,trainer):self.trainer=trainer
    Parent=module.LLaVATrainer
    class Capture(Parent):
        def train(self,*a,**k):raise Captured(self)
    native_assets=c.checked(config['feature_assets'])
    fake=dict(allocation=dict(gpus=1,microbatch=1,loader_workers=0),training_seed=1234,role='student',output=config['output'])
    providers=dict(assets=native_assets,pool=pool,selection=dict(records_json=pool['records_json']))
    arguments=cli(fake,c.checked(config['site_profile']),providers)
    values=dict(zip(arguments[::2],arguments[1::2]));values.pop('--deepspeed')
    values.update({'--lora_enable':'False','--gradient_checkpointing':'False',
        '--gradient_accumulation_steps':'1','--fp16':'False','--bf16':'True','--tf32':'True'})
    module.LLaVATrainer=Capture
    sys.argv=['native-datatailor-frozen']+[item for k,v in values.items() for item in (k,v)]
    try:
        module.train();raise ValueError('No native trainer captured')
    except Captured as result:trainer=result.trainer
    finally:module.LLaVATrainer=Parent
    model=trainer.model.requires_grad_(False).eval().to('cuda')
    c.require(trainer.optimizer is None and trainer.lr_scheduler is None,'Frozen acquisition constructed an optimizer/scheduler')
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    c.require(not any(p.requires_grad for p in model.parameters()),'Frozen feature model unexpectedly trainable')
    capture={}
    def before_norm(module,args):
        c.require(args[0].ndim==3 and len(args[0])==1,'DataTailor native B1 valid-token matrix required')
        capture['states']=args[0][0].detach()
    hook=model.get_model().norm.register_forward_pre_hook(before_norm)
    before=parameter_hash(model) if engineering else None
    try:
        for positions in batches(ids,'datatailor',config['shard_index'],config['shard_count']):
            i=positions[0];ident=ids[i]
            if ident not in store.allowed or not store.missing_batch([ident]):continue
            if should_stop():break
            verify_image(pool,rows[i])
            batch=trainer.data_collator([trainer.train_dataset[i]])
            c.require(bool(batch['attention_mask'].all()),'DataTailor B1 unexpectedly padded; no unqualified valid-token filtering')
            batch={k:([v.to('cuda') for v in value] if isinstance(value,list) else value.to('cuda')) for k,value in batch.items()}
            capture.clear()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                prepared=model.prepare_inputs_labels_for_multimodal(batch['input_ids'],None,batch.get('attention_mask'),None,batch['labels'],batch.get('images'),None)
                input_ids,position_ids,attention,past,embeds,labels=prepared
                c.require(int((labels[...,1:]!=-100).sum())>0,'No assistant targets after native expansion/truncation')
                c.require(attention is None or bool(attention.all()),'Expanded B1 sequence contains invalid tokens')
                model(input_ids=input_ids,position_ids=position_ids,attention_mask=attention,
                    past_key_values=past,inputs_embeds=embeds,labels=labels,use_cache=False,return_dict=True)
            states=capture.get('states')
            c.require(states is not None and states.shape[1]==4096,'Native pre-final-norm hidden capture absent')
            value=operators.datatailor_values(states,verify_author=engineering)
            if engineering:
                c.seal(Path(config['output'])/'engineering-evidence'/('record-'+c.key(ident)+'.json'),dict(
                    status='ACTUAL_ENGINEERING_FEATURE_EVIDENCE_NOT_A_FULL_POOL_PASS',record_id=ident,
                    original_source_index=pool['source_indices'][i],input_ids_sha256=tensor_hash(batch['input_ids']),
                    input_labels_sha256=tensor_hash(batch['labels']),expanded_labels_sha256=tensor_hash(labels),
                    supervised_targets=int((labels[...,1:]!=-100).sum()),prefinal_hidden_shape=list(states.shape),
                    prefinal_hidden_sha256=tensor_hash(states),author_SVD_loop_exact_parity=True,
                    informativeness=value['informativeness'],tau=value['tau'],parameter_tree_before=before,
                    original_multiturn_masking=True,source_adapter='c121_xformers_B1_unpadded'))
            store.commit([dict(record_id=ident,vector=value['hidden'].float().numpy(),
                info=value['informativeness'],tau=value['tau'],length=value['length'])])
    finally:hook.remove()
    if engineering:
        c.require(parameter_hash(model)==before,'Frozen Stage1 parameters changed')
        c.seal(Path(config['output'])/'engineering-evidence/FROZEN_PARAMETERS.json',dict(
            before=before,after=before,actual_checked=True,optimizer=None,updates=0,full_pool_qualified=False))


def coincide(config,pool,rows,ids,store,should_stop,engineering=False):
    """Run the unchanged author outer feature-loop AST on original fixed batches."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    sys.path.insert(0,str(pins.COIN_CODE))
    module=importlib.import_module('tinyllava.eval.score.coincide.extract_embed')
    c.require(Path(module.__file__).resolve()==pins.COIN_SCORE/'extract_embed.py','Wrong author feature module')
    assets=c.checked(config['feature_assets'])
    tiny=assets['assets']['tiny_proxy']['snapshot'];vision=assets['assets']['tiny_siglip']['snapshot']
    # Only redirect the author's immutable cache locator. Keep encoder, processor,
    # model, tokenizer, hook, tokenization and numerical operators unchanged.
    vision_module=importlib.import_module('tinyllava.model.multimodal_encoder.siglip_encoder')
    Parent=vision_module.SigLipVisionTower.load_model
    def pinned_load(self):
        c.require(self.vision_tower_name in ('bczhou/TinyLLaVA-2.0B-SigLIP',vision),'Unexpected proxy vision tower')
        self.vision_tower_name=vision
        return Parent(self)
    vision_module.SigLipVisionTower.load_model=pinned_load
    try:
        module.disable_torch_init()
        tokenizer,model,processor,_=module.load_pretrained_model(tiny,None,'TinyLLaVA-2.0B')
    finally:vision_module.SigLipVisionTower.load_model=Parent
    model.requires_grad_(False).eval()
    before=parameter_hash(model) if engineering else None
    c.require(model.config.hidden_size==2048 and len(model.model.layers)==24,'Wrong exact Tiny proxy')
    from tinyllava import conversation as conversation_lib
    conversation_lib.default_conversation=conversation_lib.conv_templates['phi']
    args=types.SimpleNamespace(data_path=pool['records_json']['path'],image_folder=pool['image_root'],
        image_aspect_ratio='pad',is_multimodal=True,mm_use_im_start_end=False,version='phi',
        num_chunks=4,chunk_idx=config['shard_index']//(config['shard_count']//4),
        batch_size=8,prompt_len=31,layer_list=[3,7,11,15,19])
    c.require(tokenizer.model_max_length==2048,'Proxy tokenizer differs from frozen2048 author contract')
    dataset=module.LazySupervisedDataset_w_split(args.data_path,tokenizer,processor,args)
    collator=module.DataCollatorForSupervisedDataset(tokenizer,31)
    shuffled=np.arange(len(ids));np.random.RandomState(42).shuffle(shuffled)
    chunk_size=math.ceil(len(ids)/4);offset=args.chunk_idx*chunk_size
    local_to_global=shuffled[offset:offset+len(dataset)]
    expected=batches(ids,'coincide',config['shard_index'],config['shard_count'])
    global_to_local={int(g):i for i,g in enumerate(local_to_global)}
    pending=[]
    for positions in expected:
        batch_ids=[ids[i] for i in positions]
        if not set(batch_ids)<=store.allowed:continue
        if store.missing_batch(batch_ids):pending.append([global_to_local[i] for i in positions])
    class Remaining:
        def __len__(self):return len(pending)
        def __iter__(self):
            for positions in pending:
                if should_stop():break
                for i in positions:verify_image(pool,rows[int(local_to_global[i])])
                batch=collator([dataset[i] for i in positions])
                if engineering:
                    c.seal(Path(config['output'])/'engineering-evidence'/('batch-'+c.key(positions)+'.json'),dict(
                        status='ACTUAL_ENGINEERING_FEATURE_EVIDENCE_NOT_A_FULL_POOL_PASS',
                        original_positions=[int(local_to_global[i]) for i in positions],
                        record_ids=[ids[int(local_to_global[i])] for i in positions],
                        input_ids_sha256=tensor_hash(batch['input_ids']),input_labels_sha256=tensor_hash(batch['labels']),
                        language_length=batch['language_length'].tolist(),conversation='phi',batch_size=len(positions),
                        author_outer_feature_loop_AST_unchanged=True,parameter_tree_before=before))
                yield batch
    class Sink:
        def __setitem__(self,indices,value):
            vector=np.asarray(value,dtype='<f2')
            c.require(vector.shape==(len(indices),20480),'Author feature dimension changed')
            measurements=[]
            for local,row in zip(indices.tolist(),vector):
                nan=np.isnan(row);count=int(nan.sum());row=row.copy();row[nan]=0
                c.require(np.isfinite(row).all(),'Author NaN policy does not repair infinity')
                measurements.append(dict(record_id=ids[int(local_to_global[local])],vector=row,nan_count=count))
            store.commit(measurements)
    layers=model.model.layers
    for layer in args.layer_list:module.autograd_hacks.add_hooks(layers[layer].self_attn.o_proj)
    tree=ast.parse((pins.COIN_SCORE/'extract_embed.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='eval_model')
    loop=next(n for n in function.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='batch')
    namespace=dict(module.__dict__)
    namespace.update(model=model,args=args,data_loader=Remaining(),layers=layers,
        num_layers=len(layers),num_target_layers=5,prompt_len=31,sim_act=nn.Tanh(),msa_task_emb=Sink())
    code=ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[]))
    with torch.no_grad():exec(compile(code,str(pins.COIN_SCORE/'extract_embed.py'),'exec'),namespace)
    if engineering:
        c.require(parameter_hash(model)==before,'Frozen Tiny proxy parameters changed')
        c.seal(Path(config['output'])/'engineering-evidence/FROZEN_PARAMETERS.json',dict(
            before=before,after=before,actual_checked=True,optimizer=None,updates=0,full_pool_qualified=False))

