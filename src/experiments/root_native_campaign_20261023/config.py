"""Fail-closed common scientific recipe and native CLI assembly."""
from pathlib import Path
from datetime import datetime, timezone
from . import common as c
from .selection import validate_selection, validate_pool

RECIPE = dict(global_batch=128, epochs=1, lora_r=128, lora_alpha=256, lora_dropout=0.05,
    learning_rate=2e-4, projector_learning_rate=2e-5, weight_decay=0, warmup_ratio=.03,
    scheduler='cosine', model_max_length=2048, masking='original_c121_native_v1_multiturn',
    precision='fp16', tf32=False, attention='xformers', optimizer='adamw_torch', deepspeed='zero3')
ASSET_PINS = {
    'vicuna': ('lmsys/vicuna-7b-v1.5','3321f76e3f527bd14065daf69dad9344000a201d'),
    'clip336': ('openai/clip-vit-large-patch14-336','ce19dc912ca5cd21c8a653c79e251e808ccabcd1'),
    'stage1_projector': ('liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5','5414da88308e4287a29f2e9609256458afb0a981'),
    'LLaVA665K': ('liuhaotian/LLaVA-Instruct-150K','9d451dc7629cfe0469f6ae4432b765cd603d5fcb')}

def accumulation(gpus, microbatch):
    c.require(gpus in (1,2,4) and microbatch >= 1 and 128 % (gpus*microbatch) == 0, 'Global128 requires divisible 1/2/4-GPU microbatch')
    return 128//(gpus*microbatch)

def validate_profile(profile):
    c.require(profile['schema'] == 'native-site-profile-v1', 'Wrong site profile')
    evidence = c.checked(profile['inventory'])
    c.require(evidence['status'] == 'PASS_CURRENT_SITE_PATHS_AND_SCHEDULER' and evidence['site'] == profile['site'], 'Fresh site inventory is required; templates are not ready')
    c.require(datetime.fromisoformat(evidence['valid_until'])>datetime.now(timezone.utc), 'Site inventory expired; fresh read-only review required')
    for field in ('partition','qos','account','gpu_type','slurm_user','python','cache_root','code_root','output_root','registry_root'):
        c.require(isinstance(profile[field], str) and profile[field] and '<' not in profile[field], 'Unresolved site field: '+field)
        c.require(evidence[field] == profile[field], 'Profile differs from checked inventory: '+field)
    if profile['site'] == 'babel':
        c.require(profile['gpu_type'] == 'L40S' and profile['account'] == 'cx' and profile['partition'] in ('general','preempt'), 'Babel route/type mismatch')
        c.require(profile['qos'] == ('normal' if profile['partition']=='general' else 'preempt_qos'), 'Babel QoS mismatch')
        c.require(profile['cache_root'] == '/data/hf_cache', 'Babel base/dataset cache must be shared')
    else:
        c.require(profile['site'] == 'aces' and profile['gpu_type'] == 'h100', 'Only inventoried ACES H100 route supported')
        c.require(profile['output_root'].startswith('/scratch/user/'), 'New ACES output must use approved user scratch')
        c.require(profile['code_root'].startswith('/scratch/user/'), 'New ACES code must use approved user scratch')
    for name in ('python','cache_root','code_root','output_root','registry_root'):
        c.require(Path(profile[name]).is_absolute(), 'Absolute site path required')
    for name in ('max_minutes','max_gpus_per_node','max_cpus_per_request','max_memory_mb'):
        c.require(isinstance(profile[name],int) and profile[name]>0 and evidence[name]==profile[name], 'Site resource limit not independently inventoried: '+name)
    return profile

def validate(config, gpu_check=False):
    c.require(config['schema'] == 'native-selection-run-v1' and config['recipe'] == RECIPE, 'Recipe differs from frozen shared recipe')
    c.require(config['official_commit'] == c.PIN and config['owner'] == 'Root', 'Wrong source or owner')
    c.require(config['role'] in ('qualify','student') and config['training_seed'] in (17,29,43), 'Unsupported role/paired seed')
    c.verify_sources(config['sources'])
    profile = validate_profile(c.checked(config['site_profile']))
    c.require(Path(profile['code_root']).resolve() == c.ROOT.resolve(), 'Execute exact installed source root')
    allocation = config['allocation']
    accumulation(allocation['gpus'],allocation['microbatch'])
    c.require(allocation['cpus'] >= allocation['gpus'] and allocation['memory_mb'] >= 32768, 'Explicit CPU+GPU and sufficient reviewed RAM required')
    c.require(0 < allocation['reserve_seconds'] < allocation['minutes']*60, 'Checkpoint reserve invalid')
    c.require(allocation['minutes'] <= profile['max_minutes'] and allocation['gpus'] <= profile['max_gpus_per_node'], 'Site per-request limit exceeded')
    c.require(allocation['cpus']<=profile['max_cpus_per_request'] and allocation['memory_mb']<=profile['max_memory_mb'],'Site CPU/RAM limit exceeded')
    c.require(c.within(config['output'], profile['output_root']) and c.within(config['report'], profile['registry_root']), 'Outputs outside approved owned roots')
    providers = {name:c.checked(value) for name,value in config['providers'].items()}
    for required in ('environment','assets','pool','selection','eligibility','ownership'):
        c.require(required in providers, 'Unready provider: '+required)
    env = providers['environment']
    c.require(env['status'] == 'PASS_ENVIRONMENT' and env['python_executable'] == profile['python'], 'Exact native environment required')
    c.require(env['versions']['transformers'] == '4.37.2' and env['versions']['torch'].startswith('2.1.2') and env['versions']['peft'] == '0.7.1', 'Native runtime API versions differ')
    c.require(env['attention'] == RECIPE['attention'] and env['precision'] == RECIPE['precision'], 'FP16/xFormers environment must be qualified explicitly; old BF16/Flash receipt is not enough')
    assets = providers['assets']
    c.require(assets['status'] == 'PASS_ASSETS' and assets['official_commit'] == c.PIN, 'Original exact assets required')
    for name, pin in ASSET_PINS.items():
        asset = assets['assets'][name]
        c.require((asset['repository'],asset['revision']) == pin, 'Wrong asset revision: '+name)
        c.require(c.within(asset['snapshot'],profile['cache_root']), 'Base model/dataset outside approved cache')
        for descriptor in asset['files']:
            c.require(c.desc(descriptor['path']) == {k:descriptor[k] for k in ('path','bytes','sha256')}, 'Asset bytes changed')
    pool, _, _ = validate_pool(config['providers']['pool'],allow_engineering=config['role']=='qualify')
    selected = validate_selection(config['providers']['selection'],config['providers']['pool'],config['method'],config['fraction'],allow_engineering=config['role']=='qualify')
    c.require(selected['selection_seed'] == config['selection_seed'], 'Fixed selection seed differs')
    c.require(config['replication_block']['replicate_site']==profile['site'],'Registered hardware replication site differs')
    function=c.key(dict(recipe=config['recipe'],pool_records=pool['records_json']['sha256'],
        selected_records=selected['records_json']['sha256'],training_seed=config['training_seed'],
        role=config['role'],qualification_max_steps=config.get('qualification_max_steps'),
        replicate_site=config['replication_block']['replicate_site']))
    ownership=providers['ownership']
    if ownership['status']=='ACTIVE_EXCLUSIVE_NATIVE_RUN_CLAIM':
        c.require(ownership['function_identity']==function,'Exact native function claim differs')
    else:
        c.require(ownership['status']=='ACTIVE_EXCLUSIVE_NATIVE_PLAN_CELL_CLAIM' and ownership['plan']==config['plan']
            and ownership['cell_id']==config['id'] and ownership['pool']==config['providers']['pool']
            and ownership['replicate_site']==config['replication_block']['replicate_site'],
            'Root plan-cell ownership claim differs')
    c.require(ownership['site_profile']==config['site_profile'] and ownership['owner']=='Root',
        'Root must register exact cross-site ownership before launch')
    reservation=c.checked(config['storage_reservation'])
    c.require(reservation['status']=='PASS_EXCLUSIVE_MEASURED_STORAGE_RESERVATION' and reservation['output']==config['output']
        and (reservation.get('function_identity')==function or reservation.get('cell_id')==config['id']), 'Measured collective storage reservation required')
    eligibility = providers['eligibility']
    c.require(eligibility['status'] == 'PASS_FROZEN_POOL_EVALUATION_ELIGIBILITY' and eligibility['pool'] == config['providers']['pool'], 'No explicit pool/evaluation eligibility review')
    c.require(eligibility['protected_QA_image_RGB_overlap_count'] == 0 and eligibility['missing_media_count'] == 0, 'Missing media/protected overlap must be resolved')
    c.checked(eligibility['evaluation_protocol'])
    image_manifest = c.checked(pool['image_manifest'])
    c.require(all(row.get('image') is None or row['image'] in image_manifest['assets'] for row in c.checked(pool['records_json'])), 'Missing image manifest entries')
    if config['role'] == 'student':
        c.require('qualification' in providers,'Student blocked: actual native GPU/restart qualification absent')
        qualification = providers['qualification']
        c.require(qualification['status'] == 'PASS_ACTUAL_NATIVE_GPU_MASKING_NUMERICS_RESTART', 'No actual new-runtime GPU/restart qualification')
        for name in ('recipe','sources'):
            c.require(qualification[name] == config[name], 'Qualification differs in '+name)
        for name in ('gpus','microbatch','cpus','memory_mb','loader_workers'):
            c.require(qualification['allocation'].get(name)==allocation.get(name), 'Qualification allocation differs: '+name)
        for name in ('environment','assets'):
            c.require(qualification['providers'][name] == config['providers'][name], 'Qualification provider differs: '+name)
        c.require(qualification['site_profile'] == config['site_profile'], 'Qualification hardware/site differs')
        for name in ('forward_backward','masking','restart','optimizer_groups','throughput'):
            evidence=c.checked(qualification['evidence'][name])
            c.require(evidence['status']=='PASS_ACTUAL_GPU_'+name.upper() and evidence['sources']==config['sources']
                and evidence['recipe']==config['recipe'], 'Missing exact raw GPU qualification evidence: '+name)
    else:
        c.require(0 < config['qualification_max_steps'] <= 20, 'Qualification must be bounded engineering work, not a scientific epoch')
    review = c.checked(config['review'])
    review_body = {k:v for k,v in config.items() if k != 'review'}
    if review['status']=='PASS_INDEPENDENT_EXACT_SOURCE_AND_CONFIG':
        c.require(review['config_identity']==c.key(review_body) and review['sources']==config['sources'],'Independent exact-source/config review differs')
    else:
        from .plan import get_cell
        plan=c.checked(config['plan']);cell=get_cell(plan,config['id'])
        c.require(review['status']=='PASS_INDEPENDENT_EXACT_PLAN_AND_SOURCE' and review['plan_identity_sha256']==c.key(plan)
            and review['sources']==config['sources']==plan['sources'],'Exact reviewed finite plan/source required for late binding')
        for field in ('method','fraction','training_seed','selection_seed','role'):
            c.require(config[field]==cell[field],'Materialized cell differs in '+field)
        for field in ('gpus','microbatch'):
            c.require(config['allocation'][field]==cell[field],'Materialized allocation differs in '+field)
        c.require(profile['site']==cell['site'] and config['recipe']==plan['recipe'],'Site/recipe differs from reviewed plan')
        readiness=c.checked(config['readiness'])
        c.require(readiness['status']=='PASS_READY_NATIVE_CAMPAIGN_INPUTS' and readiness['providers']['pool']==config['providers']['pool'],'Late input pool not bound to actual ready receipt')
    if gpu_check:
        import os, sys
        c.require(Path(sys.executable).absolute() == Path(profile['python']).absolute(), 'Wrong Python')
        c.require(int(os.environ.get('WORLD_SIZE','0')) == allocation['gpus'], 'Real distributed GPU world size differs')
    return profile, providers

def cli(config, profile, providers):
    assets = providers['assets']['assets']; selection = providers['selection']; pool=providers['pool']
    a=config['allocation']
    values = dict(model_name_or_path=assets['vicuna']['snapshot'], version='v1',
        data_path=selection['records_json']['path'], image_folder=pool['image_root'],
        vision_tower=assets['clip336']['snapshot'], pretrain_mm_mlp_adapter=str(Path(assets['stage1_projector']['snapshot'])/'mm_projector.bin'),
        mm_projector_type='mlp2x_gelu',mm_vision_select_layer=-2,mm_use_im_start_end=False,
        mm_use_im_patch_token=False,image_aspect_ratio='pad',group_by_modality_length=True,
        lora_enable=True,lora_r=128,lora_alpha=256,lora_dropout=.05,lora_bias='none',
        mm_projector_lr=2e-5,fp16=True,bf16=False,tf32=False,
        num_train_epochs=1,per_device_train_batch_size=a['microbatch'],per_device_eval_batch_size=1,
        gradient_accumulation_steps=accumulation(a['gpus'],a['microbatch']),evaluation_strategy='no',
        save_strategy='no',learning_rate=2e-4,weight_decay=0,warmup_ratio=.03,lr_scheduler_type='cosine',
        model_max_length=2048,gradient_checkpointing=True,dataloader_num_workers=a.get('loader_workers',0),
        lazy_preprocess=True,output_dir=str(Path(config['output'])/'training'),seed=config['training_seed'],
        data_seed=config['training_seed'],optim='adamw_torch',report_to='none',disable_tqdm=True,
        logging_steps=10,save_safetensors=True,deepspeed=str(c.HERE/'zero3.json'),save_total_limit=None)
    values.pop('save_total_limit')  # own receipt-aware rotation, not HF pre-acceptance rotation
    if config['role']=='qualify': values['max_steps']=config['qualification_max_steps']
    return [item for name,value in values.items() for item in ('--'+name,str(value))]
