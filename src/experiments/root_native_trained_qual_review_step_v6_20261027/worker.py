"""Real evidence review, independently implemented; no models or generation.

The subject's compare()/promotion functions are never called. Tokenizer, image
processor and pinned native prompt primitives are used only to independently
recompute the two real CPU input commitments. Every PASS is conditional on an
actual completed subject, immutable evidence, exact state and all four answers.
"""
import argparse,copy,fcntl,hashlib,json,os,re,socket,stat,subprocess,sys,time
from pathlib import Path
from contextlib import contextmanager

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
HOME=Path('/home/luyangz/data_selection_multimodal')
CONTROL=HOME/'collaboration/root-native-trained-qual-review-step-v6-20261027'
REVIEW_OUTPUT=HOME/'collaboration/root-native-trained-evaluation-v1-20261026/INDEPENDENT_ACTUAL_QUALIFICATION_REVIEW_V4.json'
LOCK=HOME/'collaboration/babel-two-track-20261019/ADMISSION.lock'
AUTHOR='/root/cache_alias_finish'
REPAIR_AUTHOR='/root/telemetry_finish'
REVIEWER=AUTHOR+'/registered-actual-trained-loader-review-v4'
SOURCE_REVIEW='PASS_NONAUTHOR_INDEPENDENT_TRAINED_QUAL_REVIEW_REGISTERED_STEP_SOURCE_V6'
PASS='PASS_INDEPENDENT_ACTUAL_NATIVE_TRAINED_EVAL_QUALIFICATION_V4'
SCOPE='INDEPENDENT_ACTUAL_PAID28_TRAINED_LOADER_FOUR_GENERATION_READOUT_REVIEW'

def require(ok,msg):
    if not ok:raise ValueError(msg)

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def key(v):return hashlib.sha256(canonical(v)).hexdigest()
def evaluator_canonical(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def evaluator_key(v):return hashlib.sha256(evaluator_canonical(v)).hexdigest()

def tensor_commitment(value):
    import torch
    value=value.detach().cpu().contiguous()
    return hashlib.sha256(evaluator_canonical([str(value.dtype),list(value.shape)])+value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

def literal(path):
    p=Path(path);require(p.is_absolute() and '..' not in p.parts
        and not any(q.is_symlink()for q in(p,*p.parents)), 'Literal non-symlink evidence required');return p

def descriptor(path):
    p=literal(path);fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW);digest=hashlib.sha256();size=0
    try:
        before=os.fstat(fd);require(stat.S_ISREG(before.st_mode),'Regular evidence required')
        while True:
            b=os.read(fd,8<<20)
            if not b:break
            digest.update(b);size+=len(b)
        after=os.fstat(fd)
        signature=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
        require(signature(before)==signature(after) and after.st_nlink>0 and size==before.st_size,'Evidence changed while hashing')
    finally:os.close(fd)
    literal(p);return dict(path=str(p),bytes=size,sha256=digest.hexdigest())

def checked(ref):
    require(set(ref)=={'path','bytes','sha256'} and type(ref['bytes'])is int and 0<ref['bytes']<=32<<20
        and re.fullmatch('[0-9a-f]{64}',ref['sha256']), 'Bounded exact metadata descriptor required')
    require(descriptor(ref['path'])==ref,'Metadata identity differs')
    value=json.loads(Path(ref['path']).read_bytes());require(descriptor(ref['path'])==ref,'Metadata changed during parse');return value

def publish(path,value):
    p=literal(path);require(CONTROL in p.parents or p==REVIEW_OUTPUT,'Only independent review outputs allowed')
    b=canonical(value)+b'\n';require(len(b)<32<<20,'Bounded review output required')
    with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
    d=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(d)
    finally:os.close(d)
    return descriptor(p)

def sources():
    paths=list(HERE.glob('*.py'))+[ROOT/'experiments/root_native_trained_qual_review_v5_20261027'/n for n in('__init__.py','cpu_inputs.py')]
    return {str(p.relative_to(ROOT)):descriptor(p)['sha256']for p in sorted(paths)if not p.name.startswith('test_')}

def fields(raw):
    require(len(raw.strip().splitlines())==1,'One exact scheduler actor required')
    matches=list(re.finditer(r'(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=',raw.strip()));result={}
    for i,m in enumerate(matches):
        require(m[1]not in result,'Duplicate scheduler field')
        result[m[1]]=raw.strip()[m.end():matches[i+1].start()if i+1<len(matches)else len(raw.strip())].strip()
    return result

def query(argv):return subprocess.check_output(argv,text=True,timeout=60)

def actual_accounting(raw,job):
    rows=[r.split('|')for r in raw.splitlines()if r.strip()]
    rows=[r for r in rows if r[0]==job]
    require(len(rows)==1,'One actual subject accounting allocation required');r=rows[0]
    require(len(r)>=11 and r[1:8]==['COMPLETED','0:0','luyangz','cx','general','normal','8']
        and r[8].isdecimal() and int(r[8])>0,'Actual subject must complete successfully on exact allocated route')
    tres=dict(x.split('=',1)for x in r[9].split(',')if'='in x)
    require(tres.get('cpu')=='8' and tres.get('gres/gpu')=='1'and tres.get('mem')in('64G','65536M'),
        'Actual subject GPU/CPU/memory accounting differs')
    return dict(job_id=job,elapsed_allocated_seconds=int(r[8]),allocated_CPUs=8,allocated_GPUs=1,node=r[10],raw=raw)

def inspect_actor(raw,job,nonce,source_root,phase=None):
    f=fields(raw);wanted=dict(JobId=job,JobState='RUNNING',UserId='luyangz(2708899)',Account='cx',Partition='general',QOS='normal',
        JobName='dsmm-native-trained-eval-qual-v4',NumCPUs='8',NumTasks='1',NumNodes='1',Restarts='0',TimeLimit='01:00:00',
        TresPerNode='gres/gpu:L40S:1',WorkDir=source_root,Command=source_root+'/experiments/root_native_trained_eval_qual_v4_20261026/run.sbatch',
        Comment='dsmm-trained-eval-qual:'+nonce)
    require(all(f.get(k)==v for k,v in wanted.items()),'Actual captured subject actor differs')
    tres=dict(x.split('=',1)for x in f['AllocTRES'].split(',')if'='in x)
    require(tres.get('cpu')=='8'and tres.get('gres/gpu')=='1'and tres.get('mem')in('64G','65536M'),'Captured subject allocation differs')
    return f

def held_subject(value,binding_ref,subject,job):
    require(value['job_id']==job and value['binding']==binding_ref,'Actual held subject binding differs')
    f=fields(value['scheduler'])
    wanted=dict(JobId=job,JobState='PENDING',Reason='JobHeldUser',UserId='luyangz(2708899)',Account='cx',Partition='general',QOS='normal',
        JobName='dsmm-native-trained-eval-qual-v4',NumCPUs='8',NumTasks='1',Restarts='0',Requeue='0',TimeLimit='01:00:00',RunTime='00:00:00',
        TresPerNode='gres/gpu:L40S:1',WorkDir=subject['source_root'],Command=subject['source_root']+'/experiments/root_native_trained_eval_qual_v4_20261026/run.sbatch',
        Comment='dsmm-trained-eval-qual:'+subject['nonce'])
    require(all(f.get(k)==v for k,v in wanted.items())and f.get('NumNodes')in('1','1-1')and f.get('AllocTRES')in(None,'(null)'),
        'Authentic never-started held subject resource/source proof differs')
    tres=dict(x.split('=',1)for x in f['ReqTRES'].split(',')if'='in x)
    require(tres.get('cpu')=='8'and tres.get('node')=='1'and tres.get('gres/gpu')=='1'and tres.get('mem')in('64G','65536M'),
        'Held subject requested resources differ')

def logical_identity(cfg):
    value=copy.deepcopy({k:v for k,v in cfg.items()if k!='review'});value.get('providers',{}).pop('runtime_images',None);return key(value)

def expected_injection(reconstruction):
    """Independently derive saved language targets and excluded loadedvision support."""
    trainable=reconstruction['trainable'];frozen=reconstruction['frozen']
    require(len(trainable)==452 and len(frozen)==682
        and len({r['name']for r in trainable+frozen})==1134,'Exact authentic native1134parameter support required')
    pairs={};prefix='base_model.model.'
    for row in trainable:
        name=row['name']
        if '.lora_'not in name:continue
        match=re.fullmatch(r'base_model\.model\.(model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj)))\.lora_([AB])\.default\.weight',name)
        require(match is not None,'Only original native language A/B support is eligible')
        target,side=match.groups();require(side not in pairs.setdefault(target,set()),'Duplicate native adapter side')
        pairs[target].add(side)
    require(len(pairs)==224 and all(v=={'A','B'}for v in pairs.values()),'Exhaustive original224paired language adapters required')
    generic={n.rsplit('.',1)[-1]for n in pairs};vision=[]
    for row in frozen:
        name=row['name']
        if name.startswith(prefix+'model.vision_tower.')and name.endswith('.weight'):
            module=name[len(prefix):-len('.weight')]
            if module.rsplit('.',1)[-1]in generic:vision.append(module)
    require(len(vision)==len(set(vision))==72,'Exact72loadedvision generic-name collisions required')
    return dict(native_training_order_preserved=True,vision_object_restored_identically=True,
        original_saved_target_policy_unchanged=True,language_adapter_targets=sorted(pairs),
        loaded_vision_generic_targets_excluded=sorted(vision),no_forward_during_structural_injection=True)

def inspect_loaded_inventory(value,reconstruction,injection):
    expected={r['name']:{k:r[k]for k in('name','shape','dtype')}for r in reconstruction['frozen']+reconstruction['trainable']}
    rows=value['observed'];actual={r['name']:r for r in rows}
    require(len(rows)==len(actual)==len(expected)==1134 and actual==expected
        and value['expected_count']==value['actual_count']==1134 and value['missing']==value['extra']==[]
        and value['injection']==injection,'Actual observed parameter inventory differs from authentic saved support')

def inspect_models(subject,first,second,identities,proofs,accepted,reconstruction,training,ds_buffers):
    require(identities[0]==identities[1],'Fresh model/tokenizer/processor/buffer identities differ')
    identity=identities[0];step=accepted['global_step']
    require(accepted['status']=='ACCEPTED_FULL_NATIVE_STATE'and step==28 and accepted['world_size']==1
        and accepted['identity']==logical_identity(training),'Actual paid28 training identity differs')
    require(identity['schema']=='native-trained-active-peft-evaluation-model-identity-v1'
        and identity['accepted']==subject['accepted_checkpoint']and identity['training_configuration']==subject['training_configuration']
        and identity['training_optimizer_updates']==identity['successful_updates']==step and identity['adapter_active']is True
        and identity['adapter_merged']is False and identity['source_commit']=='c121f0432da27facab705978f83c4ada465e46fd'
        and identity['language_attention']=='eager'and re.fullmatch('[0-9a-f]{64}',identity['parameter_sha256'])
        and identity['generation_config']['num_beams']==1,'Actual trained identity/updates/native policy differ')
    trainable=reconstruction['trainable'];require(len({r['name']for r in trainable})==len(trainable)>0,'Unique trainable reconstruction required')
    lora=sum('.lora_A.'in r['name']or'.lora_B.'in r['name']for r in trainable);projector=sum('mm_projector'in r['name']for r in trainable)
    require(lora>0 and projector==4 and lora+projector==len(trainable),'Native portable support differs')
    injection=expected_injection(reconstruction)
    for proof in proofs:
        state=proof['semantic_state'];portable=proof['portable_value_verification'];buffer=proof['persistent_buffer_verification']
        require(proof['accepted']==subject['accepted_checkpoint']and proof['training_configuration']==subject['training_configuration']
            and proof['training_optimizer_updates']==proof['successful_updates']==step and proof['native_recipe']==training['recipe']
            and proof['adapter_active']is True and proof['adapter_merged']is False and proof['no_frozen_base_saved']is True,
            'Actual model reconstruction checkpoint/recipe differs')
        require(state['status']=='PASS_SEMANTIC_NATIVE_WORLD1_ZERO3_STATE'and state['global_steps']==state['successful_updates']==step
            and state['skipped_steps']==0 and state['files_verified_before_and_after']is True
            and state['accepted_descriptor']==subject['accepted_checkpoint']and state['accepted']==accepted,
            'Actual full saved-state semantic proof differs')
        require(portable==dict(exact_parameter_inventory=True,frozen_parameters=len(reconstruction['frozen']),
            trained_parameters=len(trainable),portable_trainable_values_exact=True,LoRA_tensors=lora,projector_tensors=projector),
            'Portable trainable values/support not actually verified')
        require(proof['native_adapter_construction']==injection,'Actual native adapter construction/restoration proof differs')
        require(buffer==dict(exact_persistent_buffer_support=True,persistent_buffer_values_exact=True,
            restored_buffer_names=sorted(ds_buffers),restored_buffer_count=len(ds_buffers),parameters_overwritten=False),
            'Saved persistent buffer support/values not exactly restored')
        expected_archives=[]
        for name in accepted['files']:
            if name in ('adapter_model.safetensors','adapter_model.bin','non_lora_trainables.bin'):
                expected_archives.append(dict(path=str(Path(subject['accepted_checkpoint']['path']).parent/name),**accepted['files'][name]))
        require(len(expected_archives)==2 and {key(r)for r in proof['actual_portable_archive_reads']}=={key(r)for r in expected_archives},
            'Actual portable archive read references differ')
        model_files=[name for name in accepted['files']if name.endswith('model_states.pt')]
        require(len(model_files)==1 and proof['persistent_buffer_archive']==dict(path=str(Path(subject['accepted_checkpoint']['path']).parent/model_files[0]),**accepted['files'][model_files[0]]),
            'Persistent-buffer archive differs')
    require(first['function']==second['function'],'Fresh subject functions differ')
    return identity

def expected_function(subject,identity,roster):
    return dict(schema='native-trained-active-adapter-official-bank-function-v1',bank='POPE',role='ENGINEERING_CHECKPOINT_RELOAD_QUALIFICATION_ONLY',
        accepted_checkpoint=subject['accepted_checkpoint'],training_configuration=subject['training_configuration'],model_identity_sha256=evaluator_key(identity),
        model_generation_config=identity['generation_config'],training_optimizer_updates=28,adapter_active=True,adapter_merged=False,
        full_roster_sha256=evaluator_key(roster),decoding=subject['decoding'],sources_sha256=evaluator_key(subject['sources']))

def inspect_answers(first,second,fn,rows,commitments,identity,subject,job):
    require(first['status']==second['status']=='COMPLETE_TWO_REAL_TRAINED_QAS'and first['new_answers']==second['new_answers']==2
        and len(first['entries'])==len(second['entries'])==len(rows)==2 and first['function']==second['function']==fn,
        'Exactly four real answers on two actual semantic questions required')
    a,b=first['actual'],second['actual']
    require(a['phase']=='first'and b['phase']=='repeat'and a['job_id']==b['job_id']==job and a['nonce']==b['nonce']==subject['nonce']
        and a['process_nonce']!=b['process_nonce']and a['pid']!=b['pid']and min(a['pid'],b['pid'])>0,
        'Two genuine fresh same-job model processes required')
    for child in (first,second):
        actor=inspect_actor(child['actual']['scheduler'],job,subject['nonce'],subject['source_root'])
        require(child['actual']['node']==actor['NodeList']and child['actual']['restart']==0,'Actual child node/restart differs')
        for entry,row,commitment in zip(child['entries'],rows,commitments):
            raw=entry['raw'];audit=entry['audit'];g=audit['native_generation']
            require(entry['semantic_key']==audit['semantic_key']==g['semantic_key']==row['semantic_key']
                and audit['function_sha256']==evaluator_key(fn)and audit['actual']==child['actual']
                and type(raw['question_id'])is type(row['native_id'])and raw['question_id']==row['native_id']
                and raw['prompt']==row['native_input']['text']and raw['text']==g['decoded']
                and g['image_reference']==row['image_reference'],'Actual raw/function/typed semantic QA binding differs')
            require(all(g[k]==v for k,v in commitment.items()),'Independently recomputed prompt/token/RGB/processor commitment differs')
            require(g['explicit_kwargs']==dict(do_sample=False,temperature=0,top_p=None,num_beams=1,max_new_tokens=128,use_cache=True)
                and audit['generation_config_binding']==dict(model_generation_config_sha256=evaluator_key(identity['generation_config']),native_num_beams_keyword_present=True,effective_num_beams=1),
                'Actual native generation kwargs/config binding differs')
            ids=g['output_ids'];require(type(ids)is list and ids and all(type(v)is int and v>=0 for v in ids)
                and g['raw_output_token_count']==len(ids),'Actual generated token sequence invalid')
            eos=identity['generation_config']['eos_token_id'];eos=eos if isinstance(eos,list)else[eos]
            stop='EOS'if ids[-1]in eos else'LENGTH'
            require(g['stop_reason']==stop and (stop=='EOS'or len(ids)>=128),
                'Actual native EOS/length termination differs')
    for a,b in zip(first['entries'],second['entries']):
        require(a['audit']['native_generation']==b['audit']['native_generation']
            and {k:v for k,v in a['raw'].items()if k!='answer_id'}=={k:v for k,v in b['raw'].items()if k!='answer_id'},
            'Fresh repeat actual tokens/inputs/raw answer differ')

BRIDGE_POLICY=dict(status='PINNED_PEFT071_NATIVE_POSITIONAL_INPUTS_BRIDGE_V1',peft_version='0.7.1',
    peft_class='peft.peft_model.PeftModelForCausalLM',peft_source_sha256='baeb93af9c9a75f1987f47e32e0ae512ea1b3b41158548d18296f8e408d74872',
    peft_init_sha256='15aa49cc585e46520bba019b8161b82481e2dc40569f3e27cc1404618fc6aa02',
    native_class='llava.model.language_model.llava_llama.LlavaLlamaForCausalLM',native_source_sha256='00ab011e0543bb502e3ce26d402899022a4bc866364de1d645678cba853d05cc',
    official_positional_audit_remains_outside_bridge=True,positional_argument_zero_to_keyword='inputs',
    original_bound_PEFT_generate_invoked=True,direct_base_generate_called=False,weight_merge_performed=False,
    tensor_conversion_performed=False,generation_controls_changed=False)


def inspect_generation_bridge(value,subject,observe=descriptor):
    require(value['status']=='COMPLETE_PINNED_PEFT071_NATIVE_POSITIONAL_BRIDGE_CALLS'
        and value['policy']==BRIDGE_POLICY and type(value['calls'])is int and value['calls']==len(value['records'])==2
        and value['adapter_active']is True and value['adapter_merged']is False,
        'Exact actual two-call native PEFT071 bridge proof required')
    api=value['runtime_API_proof'];require(api['policy']==BRIDGE_POLICY,'Actual API policy differs')
    refs=api['actual_sources'];require(set(refs)=={'peft_model','peft_init','native'},'Exact real API source support')
    require(Path(refs['peft_model']['path']).parent==Path(refs['peft_init']['path']).parent
        and Path(refs['peft_model']['path']).parts[-3:]==('site-packages','peft','peft_model.py')
        and Path(refs['peft_init']['path']).name=='__init__.py'
        and refs['native']['path']==str(Path(subject['source_root'])/'collaboration/published-foundation-20261022/reference_sources/LLaVA/c121f0432da27facab705978f83c4ada465e46fd/llava/model/language_model/llava_llama.py'),
        'Actual installed PEFT/native source paths differ')
    for name,expected in [('peft_model',BRIDGE_POLICY['peft_source_sha256']),('peft_init',BRIDGE_POLICY['peft_init_sha256']),('native',BRIDGE_POLICY['native_source_sha256'])]:
        ref=refs[name];require(ref['sha256']==expected and observe(ref['path'])==ref,'Pinned actual API source changed')
    require(api['original_PEFT_signature']=='(**kwargs)'and type(api['native_signature'])is str and api['native_signature'].startswith('(inputs:'),
        'Original PEFT keyword-only and native positional-input signatures required')
    keywords=sorted(['images','image_sizes','do_sample','temperature','top_p','num_beams','max_new_tokens','use_cache'])
    for record in value['records']:
        require(record['input_keyword']=='inputs'and record['keyword_names']==keywords
            and all(record[k]is True for k in('success','input_object_unchanged','all_keyword_objects_unchanged',
                'prepare_hook_restored','original_PEFT_generate_invoked','output_object_unchanged')),
            'Actual original PEFT call/input/keyword/hook/output preservation differs')


def inspect_post_generation_identity(before,after):
    require(after==before,'Post-generation full parameter/buffer/tokenizer/processor identity differs')
    require(after['adapter_active']is True and after['adapter_merged']is False
        and after['training_optimizer_updates']==after['successful_updates']==28,
        'Generation must retain actual trained active unmerged LoRA identity')


def inspect_bridge_paths(child,subject,phase):
    base=Path(subject['output'])/phase
    require(child['generation_bridge_readout']['path']==str(base/'GENERATION_BRIDGE_READOUT.json')
        and child['post_generation_model_identity']['path']==str(base/'POST_GENERATION_MODEL_IDENTITY.json'),
        'Exact actual generation bridge and post-generation identity paths required')

@contextmanager
def no_models_or_network():
    import torch
    old=torch.nn.Module.__init__;connect=socket.socket.connect
    def denied(*a,**kw):raise RuntimeError('Independent readout review forbids models and network')
    torch.nn.Module.__init__=denied;socket.socket.connect=denied
    try:yield
    finally:torch.nn.Module.__init__=old;socket.socket.connect=connect

def recompute_inputs(subject,identity,work):
    # Scientific modules come from the verified actual subject root, not a
    # silently substituted canonical checkout or unreviewed new model loader.
    sys.path.insert(0,subject['source_root'])
    from experiments.root_babel_start_evaluation_v2_20261025 import worker as inputs
    from experiments.session6_start_model_qualification_v4_eager_20261024 import model as metadata
    require(Path(inputs.__file__).is_relative_to(subject['source_root'])and Path(metadata.__file__).is_relative_to(subject['source_root']),
        'Actual subject metadata primitives must retain their verified source route')
    import torch
    require(not torch.cuda.is_initialized(),'Independent reviewer must not initialize CUDA')
    base=checked(subject['official_POPE_configuration']);runtime=inputs.module_provenance()
    require(runtime=={k:identity['runtime'][k]for k in ('modules','torch_C')}==base['runtime_module_bindings'],
        'Actual model runtime origins differ from original literal provenance')
    roster,inventory,targets,snapshot=inputs.bound_inputs(base);require(len(roster['records'])==8910,'Full original POPE provider required')
    training=checked(subject['training_configuration'])
    with no_models_or_network():
        assets=metadata.verified_assets(training['providers']['assets'],'/data/hf_cache')
        from transformers import AutoTokenizer,CLIPImageProcessor
        tokenizer=AutoTokenizer.from_pretrained(str(Path(subject['accepted_checkpoint']['path']).parent),use_fast=False,
            model_max_length=2048,padding_side='right',local_files_only=True)
        processor=CLIPImageProcessor.from_pretrained(assets['clip336']['snapshot'],local_files_only=True)
        require(key(tokenizer.get_vocab())==identity['tokenizer']['vocabulary_sha256']and processor.to_dict()==identity['processor'],
            'Actual saved tokenizer/CLIP processor metadata differs')
        from experiments.root_native_trained_qual_review_v5_20261027.cpu_inputs import primitives
        with primitives(subject['source_root'],inputs.c.LLAVA,subject['sources'])as(native_modules,import_proof):
            conv_templates=native_modules['conversation'].conv_templates
            process_images=native_modules['mm_utils'].process_images
            tokenizer_image_token=native_modules['mm_utils'].tokenizer_image_token
            IMAGE_TOKEN_INDEX=native_modules['constants'].IMAGE_TOKEN_INDEX
            DEFAULT_IMAGE_TOKEN=native_modules['constants'].DEFAULT_IMAGE_TOKEN
            from types import SimpleNamespace
            modelconfig=SimpleNamespace(**identity['native_config'])
            require(not modelconfig.mm_use_im_start_end and not modelconfig.mm_use_im_patch_token,'Native image token policy differs')
            store=inputs.media.Store(dict(base,output=str(work)),roster['records'],inventory,snapshot);commitments=[]
            try:
                for row in roster['records'][:2]:
                    conversation=conv_templates['vicuna_v1'].copy();conversation.append_message(conversation.roles[0],DEFAULT_IMAGE_TOKEN+'\n'+row['native_input']['text']);conversation.append_message(conversation.roles[1],None)
                    prompt=conversation.get_prompt();tokens=tokenizer_image_token(prompt,tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0)
                    image=store.rgb(row['native_input']['image'])
                    try:pixels=process_images([image],processor,modelconfig)[0].unsqueeze(0).to(torch.float16);size=list(image.size)
                    finally:image.close()
                    commitments.append(dict(prompt_utf8_sha256=hashlib.sha256(prompt.encode()).hexdigest(),input_ids_sha256=tensor_commitment(tokens),input_token_count=tokens.shape[-1],pixel_values_sha256=tensor_commitment(pixels),image_size=size))
            finally:store.close()
    publish(work/'CPU_INPUT_IMPORT_ROUTE.json',import_proof)
    require(not torch.cuda.is_initialized(),'Reviewer unexpectedly initialized CUDA')
    return roster,commitments,tokenizer

def configuration(config_ref,peer_ref):
    cfg=checked(config_ref);peer=checked(peer_ref)
    require(cfg['status']=='ROOT_CONFIGURED_INDEPENDENT_TRAINED_QUAL_REGISTERED_STEP_V6'
        and cfg['owner']=='Root' and cfg['scope']==SCOPE and cfg['source_root']==str(ROOT)
        and cfg['subject_job_id']=='10946795' and cfg['sources']==sources()
        and cfg['review_output_path']==str(REVIEW_OUTPUT)
        and cfg['parent_job_id']=='10942711' and cfg['resources']==dict(step_cpus=4,parent_cpus=8,parent_memory_mb=65536,parent_gpus=1,shared_parent_memory=True,minutes=15)
        and cfg['new_neural_generations']==cfg['models']==0,'Exact registered-step configuration required')
    require(peer['status']==SOURCE_REVIEW and peer['independent']is True and peer['reviewer']not in('',AUTHOR,REVIEWER)
        and peer['configuration']==config_ref and peer['sources']==sources(),'Independent exact V6 source/config review required')
    return cfg

def documents(binding):
    cfg=configuration(binding['configuration'],binding['independent_source_review'])
    subject=checked(cfg['subject_configuration']);sb=checked(cfg['subject_binding']);sr=checked(sb['independent_source_config_review']);job=cfg['subject_job_id']
    require(subject['status']=='ROOT_CONFIGURED_NATIVE_TRAINED_EVAL_QUALIFICATION_V4'
        and subject['scope']=='PAID_CANARY28_TRAINED_ADAPTER_TWO_POPE_QA_FRESH_RELOAD_NOT_FORMAL_SCORE'
        and subject['banks']==['POPE']and subject['questions_per_bank']==2 and subject['fresh_model_processes']==2
        and sb['status']=='ROOT_BOUND_NATIVE_TRAINED_EVAL_QUALIFICATION_V4'
        and sb['job_id']==job and sb['configuration']==cfg['subject_configuration']and sb['nonce']==subject['nonce']
        and sr['status']=='PASS_INDEPENDENT_NATIVE_TRAINED_EVAL_QUAL_SOURCE_CONFIG_V4'and sr['independent']is True
        and sr['configuration']==cfg['subject_configuration']and sr['sources']==subject['sources'], 'Exact real subject source/config/binding required')
    require(subject['source_root']==cfg['subject_source_root']and subject['sources']==cfg['subject_sources'],'Actual subject source closure differs')
    for name,sha in subject['sources'].items():
        require(not Path(name).is_absolute()and '..'not in Path(name).parts,'Literal relative subject source required')
        require(descriptor(Path(subject['source_root'])/name)['sha256']==sha,'Actual executed subject source changed')
    owner=checked(sb['ownership']);require(owner['owner']=='Root'and owner['job_id']==job and owner['nonce']==subject['nonce']
        and owner['configuration']==cfg['subject_configuration']and owner['accepted_checkpoint']==subject['accepted_checkpoint']
        and owner['no_live_duplicate_qualification']is True,'Subject actual exclusive ownership differs')
    require(binding['status']=='ROOT_BOUND_INDEPENDENT_TRAINED_QUAL_REGISTERED_STEP_V6' and binding['owner']=='Root'
        and binding['nonce']==cfg['nonce'] and binding['parent_job_id']==cfg['parent_job_id'],'Exact genuine registered-step binding required')
    owner=checked(binding['ownership']);require(owner['owner']=='Root' and owner['configuration']==binding['configuration']
        and owner['registered_step']==binding['registered_step'] and owner['subject_binding']==cfg['subject_binding']
        and owner['scope']==SCOPE and owner['no_competing_actual_readout_reviewer']is True,'Exclusive registered readout ownership required')
    return cfg,subject,sb

def inspect_parent(raw,cfg,parent):
    f=fields(raw);wanted=dict(JobId='10942711',JobState='RUNNING',UserId='luyangz(2708899)',Account='cx',Partition='general',QOS='normal',
        JobName='dsmm-fulltrajectory-babel-production-student-v1-10-source_sqrt-s17',NumCPUs='8',NumTasks='1',NumNodes='1',Restarts='0',
        TresPerNode='gres/gpu:L40S:1',WorkDir=cfg['parent_source_root'],Command=cfg['parent_command'],Comment='dsmm-fulltrajectory:'+parent['nonce'])
    require(parent['schema']=='root-fulltrajectory-held-binding-v2' and parent['owner']=='Root' and parent['job_id']=='10942711' and parent['nonce']==cfg['parent_nonce']
        and parent['spool']['sha256']==parent['command']['sha256']==cfg['parent_spool_sha256'] and parent['command']['path']==cfg['parent_command'] and all(f.get(k)==v for k,v in wanted.items()),'Actual owned parent identity/source differs')
    tres=dict(x.split('=',1)for x in f['AllocTRES'].split(',')if'='in x)
    require(tres.get('cpu')=='8' and tres.get('gres/gpu')=='1' and tres.get('mem')in('64G','65536M') and f['NodeList']=='babel-o5-20','Actual owned parent allocation differs')
    require(hashlib.sha256(subprocess.check_output(['scontrol','write','batch_script','10942711','-'],timeout=30)).hexdigest()==cfg['parent_spool_sha256'],'Immutable parent spool differs')
    return f

def inspect_step(raw,identifier,cpus,node):
    f=fields(raw)
    require(f.get('StepId')==identifier and f.get('State')=='RUNNING' and f.get('UserId')=='2708899'
        and f.get('CPUs')==str(cpus) and f.get('NodeList')==node and f.get('Nodes')=='1' and f.get('Tasks')=='1'
        and f.get('Partition')=='general' and f.get('TresPerStep')=='cpu:4','Actual registered step identity/CPU/node differs')
    tres=dict(x.split('=',1)for x in f['TRES'].split(',')if'='in x)
    require(tres.get('cpu')==str(cpus) and tres.get('mem')in('64G','65536M') and tres.get('gres/gpu')=='1', 'Actual shared parent-backed step TRES differs')
    return f

def actor(binding,cfg):
    job=os.environ.get('SLURM_JOB_ID','');step=os.environ.get('SLURM_STEP_ID','')
    require(job==cfg['parent_job_id']=='10942711' and step.isdecimal() and int(step)>0 and os.getuid()==2708899
        and os.environ.get('SLURM_CPUS_PER_TASK')=='4' and not os.environ.get('SLURM_ARRAY_JOB_ID')
        and os.environ.get('SLURM_RESTART_COUNT','0')=='0','Genuine fresh registered four-CPU step required')
    identifier=job+'.'+step;require(binding['registered_step']==identifier,'Step registration differs')
    parent=checked(cfg['parent_binding']);raw=query(['scontrol','show','job','-o',job]);f=inspect_parent(raw,cfg,parent)
    sr=query(['scontrol','show','step','-o',identifier]);inspect_step(sr,identifier,4,f['NodeList'])
    return dict(route='REGISTERED_CPU_STEP_WITHIN_EXISTING_OWNED_GPU_ALLOCATION_V6',job_id=job,step_id=step,registered_step=identifier,
        node=f['NodeList'],scheduler=raw,step_scheduler=sr,parent_binding=cfg['parent_binding'],models=0,neural_generations=0,
        step_CPUs=4,parent_allocated_CPUs=8,parent_allocated_memory_mb=65536,parent_allocated_GPUs=1,memory_is_shared_parent=True,
        no_separate_GPU_allocation_requested=True)

def register(config_path,peer_path,intent_path):
    cr=descriptor(config_path);pr=descriptor(peer_path);cfg=configuration(cr,pr);ir=descriptor(intent_path);intent=checked(ir)
    require(intent['status']=='ROOT_REGISTERED_STEP_LAUNCH_INTENT_V6' and intent['owner']=='Root' and intent['configuration']==cr
        and intent['independent_source_review']==pr and intent['parent_binding']==cfg['parent_binding'] and intent['nonce']==cfg['nonce']
        and intent['step_cpus']==4 and intent['no_competing_actual_readout_reviewer']is True,'Exact pre-existing launch intent required')
    step=os.environ.get('SLURM_STEP_ID','');require(step.isdecimal() and int(step)>0,'Actual numeric step required before registration')
    binding=dict(status='ROOT_BOUND_INDEPENDENT_TRAINED_QUAL_REGISTERED_STEP_V6',owner='Root',configuration=cr,independent_source_review=pr,
        parent_job_id='10942711',registered_step='10942711.'+step,nonce=cfg['nonce'],launch_intent=ir)
    actual=actor(binding,cfg)
    with literal(LOCK).open('r+b')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not (CONTROL/'STEP_BINDING.json').exists() and not (CONTROL/'ACTUAL_REVIEW_OWNERSHIP.json').exists(),'No prior registered review owner may exist')
        ownership=publish(CONTROL/'STEP_OWNERSHIP.json',dict(owner='Root',configuration=cr,registered_step=binding['registered_step'],
            subject_binding=cfg['subject_binding'],scope=SCOPE,no_competing_actual_readout_reviewer=True,launch_intent=ir,actual=actual))
        binding['ownership']=ownership;return publish(CONTROL/'STEP_BINDING.json',binding)

def run(binding_path):
    began=time.monotonic();binding=checked(descriptor(literal(binding_path)));cfg,subject,sb=documents(binding);actual=actor(binding,cfg);job=cfg['subject_job_id']
    out=literal(cfg['output']);require(out.parent==CONTROL and not out.exists(),'New exclusive independent review attempt required');out.mkdir()
    with literal(LOCK).open('r+b')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        publish(CONTROL/'ACTUAL_REVIEW_OWNERSHIP.json',dict(configuration=binding['configuration'],actual=actual,subject_binding=cfg['subject_binding'],author=AUTHOR))
    publish(out/'START.json',dict(actual=actual,configuration=binding['configuration']))
    try:
        require(not REVIEW_OUTPUT.exists(),'Existing actual review requires explicit reconciliation, not overwrite')
        acct=actual_accounting(query(['sacct','-D','-X','-n','-P','-j',job,'--format=JobID,State,ExitCode,User,Account,Partition,QOS,AllocCPUS,ElapsedRaw,AllocTRES,NodeList']),job)
        parent=Path(cfg['subject_binding']['path']).parent;held=checked(descriptor(parent/'VERIFIED_HELD.json'));intent=checked(held['intent']);ack=checked(descriptor(parent/'ACK.json'))
        held_subject(held,cfg['subject_binding'],subject,job)
        require(held['job_id']==job and held['binding']==cfg['subject_binding']and intent['configuration']==cfg['subject_configuration']
            and ack['returncode']==0 and re.fullmatch(job+r'(?:;[^\s]+)?\s*',ack['stdout'])
            and held['spool_sha256']==descriptor(Path(subject['source_root'])/'experiments/root_native_trained_eval_qual_v4_20261026/run.sbatch')['sha256'],
            'Actual held intent/ACK/spool subject custody differs')
        resultref=descriptor(Path(subject['output'])/'RESULT.json');result=checked(resultref)
        require(result['status']=='PASS_ACTUAL_NATIVE_TRAINED_ADAPTER_TWO_QA_FRESH_RELOAD'and result['configuration']==cfg['subject_configuration']
            and result['sources']==subject['sources']and result['neural_outputs']==4 and result['distinct_QAs']==2
            and result['adapter_active']is True and result['adapter_merged']is False and result['full_bank_scored']is False
            and result['formal_student_evaluation_complete']is False and result['scientific_improvement_demonstrated']is False
            and result['engineering_canary_only']is True and result['accepted_checkpoint']==subject['accepted_checkpoint'],
            'Actual full qualifier output differs, partial/START is not eligible')
        captured=inspect_actor(result['actual']['scheduler'],job,subject['nonce'],subject['source_root'])
        require(result['actual']['job_id']==job and result['actual']['nonce']==subject['nonce']and result['actual']['restart']==0
            and result['actual']['node']==captured['NodeList']==acct['node'],'Final/captured/accounted subject actor differs')
        refs=result['child_results'];require([r['path']for r in refs]==[str(Path(subject['output'])/p/'RESULT.json')for p in('first','repeat')], 'Exact two child-result paths required')
        children=[checked(r)for r in refs]
        for phase,c in zip(('first','repeat'),children):
            require(c['model_identity']['path']==str(Path(subject['output'])/phase/'MODEL_IDENTITY.json')and c['reconstruction']['path']==str(Path(subject['output'])/phase/'MODEL_RECONSTRUCTION.json'),
                'Actual child identity/reconstruction paths differ')
            inspect_bridge_paths(c,subject,phase)
        identities=[checked(c['model_identity'])for c in children];proofs=[checked(c['reconstruction'])for c in children]
        bridge_refs=[c['generation_bridge_readout']for c in children]
        bridges=[checked(r)for r in bridge_refs]
        post_refs=[c['post_generation_model_identity']for c in children]
        post_identities=[checked(r)for r in post_refs]
        for before,after in zip(identities,post_identities):inspect_post_generation_identity(before,after)
        for identity,proof,bridge in zip(identities,proofs,bridges):
            inspect_generation_bridge(bridge,subject)
            require(identity['generation_API_bridge']==proof['generation_API_bridge']==bridge['runtime_API_proof'],
                'Identity/reconstruction/actual bridge API source and policy differ')
        accepted=checked(subject['accepted_checkpoint']);training=checked(subject['training_configuration']);checkpoint=Path(subject['accepted_checkpoint']['path']).parent
        for name,pin in accepted['files'].items():
            require(not Path(name).is_absolute()and '..'not in Path(name).parts,'Unsafe checkpoint member')
            require(descriptor(checkpoint/name)==dict(path=str(checkpoint/name),**pin),'Accepted state payload changed')
        reconstruction=checked(descriptor(checkpoint/'FROZEN_RECONSTRUCTION.json'))
        import torch
        model_files=[n for n in accepted['files']if n.endswith('model_states.pt')];require(len(model_files)==1,'One saved DS buffer archive required')
        state=torch.load(checkpoint/model_files[0],map_location='cpu',weights_only=False);ds_buffers=state['buffer_names']
        require(descriptor(checkpoint/model_files[0])==dict(path=str(checkpoint/model_files[0]),**accepted['files'][model_files[0]]),'Saved buffer archive changed while reading');del state
        identity=inspect_models(subject,*children,identities,proofs,accepted,reconstruction,training,ds_buffers)
        inventory_refs=[];injection=expected_injection(reconstruction)
        for phase in ('first','repeat'):
            inventory_ref=descriptor(Path(subject['output'])/phase/'LOADED_PARAMETER_INVENTORY.json')
            inspect_loaded_inventory(checked(inventory_ref),reconstruction,injection);inventory_refs.append(inventory_ref)
        roster,commitments,tokenizer=recompute_inputs(subject,identity,out)
        fn=expected_function(subject,identity,roster);inspect_answers(*children,fn,roster['records'][:2],commitments,identity,subject,job)
        for child in children:
            for e in child['entries']:
                ids=e['audit']['native_generation']['output_ids'];require(all(i<len(tokenizer)for i in ids),'Generated ID outside saved tokenizer vocabulary')
                require(tokenizer.decode(ids,skip_special_tokens=True).strip()==e['raw']['text'],'Actual decoded text differs from generated IDs')
        require(descriptor(subject['accepted_checkpoint']['path'])==subject['accepted_checkpoint'],'Accepted state header changed after review')
        review=dict(status=PASS,independent=True,reviewer=REVIEWER,configuration=cfg['subject_configuration'],readout=resultref,
            subject_binding=cfg['subject_binding'],subject_actual_accounting=acct,subject_held_custody=descriptor(parent/'VERIFIED_HELD.json'),
            child_results=refs,actual_independent_reviewer=actual,independent_review_configuration=binding['configuration'],
            models_constructed=0,new_neural_generations=0,subject_neural_generations_verified=4,distinct_QAs=2,
            independent_prompt_token_RGB_processor_recomputation=True,all_saved_checkpoint_payload_SHA_verified=True,
            actual_trained_tensor_and_persistent_buffer_guards_verified=True,no_full_bank_or_student_improvement_claim=True,
            actual_loaded_parameter_inventories=inventory_refs,native_adapter_construction_verified=injection,actual_pinned_PEFT071_bridge_four_calls_verified=True,
            generation_bridge_readouts=bridge_refs,post_generation_model_identities=post_refs,
            cpu_input_import_route=descriptor(out/'CPU_INPUT_IMPORT_ROUTE.json'),
            source_only_repair_preserves_V4_protocol=True,repair_author=REPAIR_AUTHOR,
            independent_verification_design_author=AUTHOR,
            registered_step_route_version='V6',registered_step_binding=descriptor(binding_path),
            completion_requires_separate_actual_sacct_step_receipt=True,
            all_prior_and_current_costs_count=True,elapsed_seconds=time.monotonic()-began)
        rr=publish(REVIEW_OUTPUT,review);publish(out/'RESULT.json',dict(status='ACTUAL_INDEPENDENT_TRAINED_QUALIFICATION_REVIEW_PUBLISHED',review=rr,actual=actual))
        print(json.dumps(rr))
    except BaseException as error:
        publish(out/'FAILURE.json',dict(status='INDEPENDENT_ACTUAL_QUALIFICATION_REJECTED_NO_PROMOTION',actual=actual,error=repr(error),elapsed_seconds=time.monotonic()-began));raise

if __name__=='__main__':
    require(__debug__,'Optimized execution forbidden')
    p=argparse.ArgumentParser();p.add_argument('--configuration',required=True);p.add_argument('--independent-source-review',required=True);p.add_argument('--launch-intent',required=True)
    a=p.parse_args();binding=register(a.configuration,a.independent_source_review,a.launch_intent);run(binding['path'])
