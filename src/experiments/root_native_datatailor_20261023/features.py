"""DataTailor native Stage1 BF16 feature constructor, separate from frozen V2.

The author FP32 SVD/entropy/tau operators are reused unchanged. This c121 eager
xFormers/B1 adaptation requires actual native BF16/SVD parity before production.
No old feature constructor or Qwen features are imported or called.
"""
import hashlib
import importlib
from pathlib import Path
import sys
from experiments.root_native_campaign_v3_20261023 import common as c
from experiments.root_native_campaign_v3_20261023.config import cli
from experiments.root_native_campaign_v3_20261023.resume_adapter import assert_attention
from experiments.root_native_baselines_20261023 import operators

CONTRACT=dict(proxy='Stage1_Vicuna7B_CLIP336_projector',precision='bf16',tf32=True,
    microbatch=1,attention='explicit_eager_c121_xformers',
    hidden='full_unpadded_last_decoder_before_final_RMSNorm',hidden_dimension=4096,
    svd='torch.svd_float32',singular_normalization='L1',
    entropy='author_Python_probability_log_loop_stored_float32',
    tau='maximum_normalized_singular_value_stored_float32',last_valid_token=True,
    frozen=True,optimizer=None,training_updates=0)


def tensor_hash(value):
    import torch
    return hashlib.sha256(value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def parameter_hash(model):
    return c.key({n:dict(shape=list(p.shape),dtype=str(p.dtype),sha256=tensor_hash(p)) for n,p in model.named_parameters()})


def logical_pool(pool):
    images=c.checked(pool['image_manifest'])['assets']
    return c.key(dict(records_sha256=pool['records_json']['sha256'],record_ids=pool['record_ids'],source_indices=pool['source_indices'],
        images={n:{k:x[k] for k in ('sha256','rgb_sha256','width','height')} for n,x in images.items()},
        protected_exclusions_sha256=pool['protected_exclusions']['sha256']))


def identity(config,pool):
    return c.key(dict(method='datatailor',adapter='explicit-eager-native-bf16-v1',logical_pool=logical_pool(pool),
        feature_assets=config['feature_assets'],environment=config['environment'],
        sources=config['sources'],native_sources=config['native_sources'],measurement_contract=CONTRACT))


def roster(config,rows,ids):
    if config['role']=='qualify_features':
        selected=config['engineering_ids']
        c.require(8<=len(selected)<=32 and len(selected)==len(set(selected)) and set(selected)<=set(ids),
            'Qualifier must declare8–32 exact original engineering records')
        original=dict(zip(ids,rows));chosen=[original[i] for i in selected]
        c.require(any(not r.get('image') for r in chosen) and any(r.get('image') for r in chosen)
            and any(len(r['conversations'])>=4 for r in chosen)
            and any(sum(len(t['value']) for t in r['conversations'])>=2000 for r in chosen),
            'Actual image/text/multiturn/long qualification coverage required')
        return selected
    total=config['shard_count'];index=config['shard_index']
    c.require(isinstance(total,int) and 0<=index<total,'Invalid native feature shard')
    return [ident for ident in ids if int(c.key(['datatailor-native-shard-v1',ident]),16)%total==index]


def engineering_ids(rows,ids):
    """Data-only fixed32 qualifier: source, modality, long and multiturn coverage."""
    c.require(len(ids)>=32 and len(rows)==len(ids),'Actual engineering construction needs32records')
    groups={}
    for i,row in zip(ids,rows):groups.setdefault(row.get('image','text').split('/')[0],[]).append(i)
    original=dict(zip(ids,rows));chosen=[]
    def add(candidates):
        c.require(candidates,'Missing declared engineering coverage')
        ident=min(candidates,key=lambda i:(c.key(['datatailor-engineering-v1',i]),i))
        if ident not in chosen:chosen.append(ident)
    for group in sorted(groups):add(groups[group])
    add([i for i,r in original.items() if len(r['conversations'])>=4])
    add([i for i,r in original.items() if sum(len(t['value']) for t in r['conversations'])>=2000])
    for i in sorted(ids,key=lambda i:(c.key(['datatailor-engineering-v1',i]),i)):
        if i not in chosen:chosen.append(i)
        if len(chosen)==32:break
    return [i for i in ids if i in set(chosen)]


def compute(config,pool,rows,ids,store,should_stop,engineering,attempt):
    import torch
    import xformers.ops as ops
    from PIL import Image
    from experiments.root_native_qualification_20261023.masking import author_v1
    source=str(c.ROOT/c.NATIVE);sys.path.insert(0,source)
    patch=importlib.import_module('llava.train.llama_xformers_attn_monkey_patch')
    patch.replace_llama_attn_with_xformers_attn()
    module=importlib.import_module('llava.train.train')
    c.require(Path(module.__file__).resolve()==Path(source)/'llava/train/train.py','Foreign native constructor')
    class Captured(Exception):
        def __init__(self,trainer):self.trainer=trainer
    Parent=module.LLaVATrainer
    class Capture(Parent):
        def train(self,*args,**kwargs):raise Captured(self)
    assets=c.checked(config['feature_assets']);profile=c.checked(config['site_profile'])
    fake=dict(allocation=dict(gpus=1,microbatch=1,loader_workers=0),training_seed=1234,role='student',output=config['output'])
    args=cli(fake,profile,dict(assets=assets,pool=pool,selection=dict(records_json=pool['records_json'])))
    values=dict(zip(args[::2],args[1::2]));values.pop('--deepspeed')
    values.update({'--lora_enable':'False','--gradient_checkpointing':'False','--gradient_accumulation_steps':'1',
        '--fp16':'False','--bf16':'True','--tf32':'True'})
    previous_argv=sys.argv[:];module.LLaVATrainer=Capture
    try:
        sys.argv=['native-datatailor-bf16-eager']+[item for key,value in values.items() for item in (key,value)]
        module.train(attn_implementation='eager')
        raise ValueError('No original native constructor captured')
    except Captured as captured:trainer=captured.trainer
    finally:module.LLaVATrainer=Parent;sys.argv=previous_argv
    c.require(trainer.optimizer is None and trainer.lr_scheduler is None,'Feature constructor created training state')
    model=trainer.model.requires_grad_(False).eval().to(device='cuda',dtype=torch.bfloat16)
    c.require(torch.cuda.is_bf16_supported() and not any(p.requires_grad for p in model.parameters()),'Actual frozen BF16 CUDA support absent')
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    audit=assert_attention(model,patch);capture={};calls=[0]
    c.seal(Path(config['output'])/'ATTENTION.json',audit)
    def before_norm(_module,args):
        c.require(args[0].ndim==3 and len(args[0])==1 and args[0].dtype==torch.bfloat16,'Native full B1/BF16 token matrix required')
        capture['states']=args[0][0].detach()
    hook=model.get_model().norm.register_forward_pre_hook(before_norm)
    original_kernel=ops.memory_efficient_attention
    def observed_kernel(query,key,value,*args,**kwargs):
        bias=kwargs.get('attn_bias',args[0] if args else None)
        c.require(query.dtype==key.dtype==value.dtype==torch.bfloat16 and query.is_cuda
            and isinstance(bias,ops.LowerTriangularMask),'Actual native BF16 causal xFormers path differs')
        calls[0]+=1
        return original_kernel(query,key,value,*args,**kwargs)
    ops.memory_efficient_attention=observed_kernel
    images=c.checked(pool['image_manifest'])['assets'];positions={ident:i for i,ident in enumerate(ids)}
    try:
        before=parameter_hash(model) if engineering else None
        for ident in roster(config,rows,ids):
            if ident not in store.allowed or not store.missing_batch([ident]):continue
            if should_stop():break
            i=positions[ident];row=rows[i]
            if row.get('image'):
                expected=images[row['image']];path=(Path(pool['image_root'])/row['image']).absolute()
                c.require(c.within(path,pool['image_root']) and str(path)==expected['path'] and c.sha(path)==expected['sha256'],'Encoded native image changed')
                with Image.open(path) as image:
                    rgb=image.convert('RGB')
                    c.require(rgb.size==(expected['width'],expected['height']) and hashlib.sha256(rgb.tobytes()).hexdigest()==expected['rgb_sha256'],'Native RGB changed')
            batch=trainer.data_collator([trainer.train_dataset[i]])
            if engineering:
                import llava.conversation as conversation
                oracle=author_v1(row,trainer.tokenizer,conversation.default_conversation.copy())
                c.require(batch['input_ids'][0].tolist()==oracle['input_ids'][:2048]
                    and batch['labels'][0].tolist()==oracle['labels'][:2048],
                    'Native full-conversation inputs/masks differ from independent assistant oracle')
            batch={k:([v.to('cuda') for v in value] if isinstance(value,list) else value.to('cuda')) for k,value in batch.items()}
            capture.clear()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                prepared=model.prepare_inputs_labels_for_multimodal(batch['input_ids'],None,batch.get('attention_mask'),None,batch['labels'],batch.get('images'),None)
                input_ids,position_ids,attention,past,embeds,labels=prepared
                c.require(int((labels[...,1:]!=-100).sum())>0 and (attention is None or bool(attention.all())),
                    'Expanded B1 assistant targets absent or padding introduced')
                if engineering:
                    expected=[]
                    for token,target,keep in zip(batch['input_ids'][0].tolist(),batch['labels'][0].tolist(),batch['attention_mask'][0].tolist()):
                        if keep:expected.extend([-100]*576 if token==-200 else [target])
                    c.require(labels[0].tolist()==expected[:2048],'Actual native image expansion/truncation differs from independent label oracle')
                model(input_ids=input_ids,position_ids=position_ids,attention_mask=attention,past_key_values=past,
                    inputs_embeds=embeds,labels=labels,use_cache=False,return_dict=True)
            states=capture.get('states')
            c.require(states is not None and states.shape[1]==4096,'Actual native pre-final-norm capture absent')
            value=operators.datatailor_values(states,verify_author=engineering)
            if engineering:
                c.seal(Path(config['output'])/'engineering-evidence'/('record-'+c.key(ident)+'.json'),dict(
                    status='ACTUAL_DATATAILOR_ENGINEERING_EVIDENCE_NOT_QUALIFICATION',record_id=ident,
                    source_index=pool['source_indices'][i],input_ids_sha256=tensor_hash(batch['input_ids']),
                    input_labels_sha256=tensor_hash(batch['labels']),expanded_labels_sha256=tensor_hash(labels),
                    targets=int((labels[...,1:]!=-100).sum()),hidden_shape=list(states.shape),hidden_sha256=tensor_hash(states),
                    actual_hidden_dtype=str(states.dtype),author_SVD_exact_parity=True,
                    informativeness=value['informativeness'],tau=value['tau'],parameter_hash_before=before,
                    original_multiturn_masking=True,independent_assistant_and_expansion_oracle_passed=True,
                    source_adaptation='c121_explicit_eager_xformers_B1_BF16'))
            store.commit([dict(record_id=ident,vector=value['hidden'].float().numpy(),info=value['informativeness'],tau=value['tau'],length=value['length'])])
    finally:
        hook.remove();ops.memory_efficient_attention=original_kernel
    if engineering:
        after=parameter_hash(model);c.require(before==after,'Frozen Stage1 parameters changed')
        c.seal(Path(attempt)/'ENGINEERING_MODEL_EVIDENCE.json',dict(
            status='ACTUAL_ENGINEERING_EVIDENCE_REQUIRES_INDEPENDENT_REVIEW',
            before=before,after=after,actual_BF16_causal_xformers_calls=calls[0],
            optimizer=None,updates=0,actual_full_pool_qualification=False,author_source_adaptation_disclosed=True))
