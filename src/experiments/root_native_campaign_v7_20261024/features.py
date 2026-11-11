"""Paid frozen-Stage1 assistant loss + visual features, one durable row per record.

No benchmark labels or trained/Qwen proxy is used. Actual runtime must first be
qualified; these are upstream signals, not established downstream utilities.
"""
import argparse
import importlib
import os
from pathlib import Path
import signal
import sqlite3
import sys
import time
from . import common as c
from .config import validate_profile,ASSET_PINS,cli
from .selection import validate_pool

class Captured(Exception):
    def __init__(self,trainer):self.trainer=trainer

def shard_ids(ids,index,total):
    c.require(0<=index<total,'Invalid feature shard')
    return [ident for ident in ids if int(c.key(['stage1-feature-shard-v1',ident]),16)%total==index]

def feature_identity(config):
    return c.key(dict(stage='before_visual_instruction_tuning',feature_version='native_loss_clip12-v1',
        pool=config['providers']['pool'],assets=config['providers']['assets'],environment=config['providers']['environment'],
        sources=config['sources'],visual_hash_seed=1234,visual_hash_bits=12,record_unit='whole_native_conversation'))

def validate(config):
    c.require(config['schema']=='native-stage1-feature-job-v1' and config['owner']=='Root','Wrong feature job')
    c.verify_sources(config['sources']);profile=validate_profile(c.checked(config['site_profile']))
    c.require(Path(profile['code_root']).resolve()==c.ROOT.resolve(),'Wrong staged feature source root')
    providers={name:c.checked(value) for name,value in config['providers'].items()}
    c.require(providers['environment']['status']=='PASS_ENVIRONMENT' and providers['environment']['python_executable']==profile['python']
        and providers['environment']['precision']=='fp16' and providers['environment']['attention']=='xformers','Actual native environment required')
    assets=providers['assets']
    c.require(assets['status']=='PASS_ASSETS' and assets['official_commit']==c.PIN,'Wrong original Stage1 components')
    for name,pin in ASSET_PINS.items():
        asset=assets['assets'][name]
        c.require((asset['repository'],asset['revision'])==pin and c.within(asset['snapshot'],profile['cache_root']),'Wrong cached model/data asset')
        for descriptor in asset['files']:
            c.require(c.desc(descriptor['path'])=={k:descriptor[k] for k in ('path','bytes','sha256')},'Asset bytes changed')
    pool,rows,ids=validate_pool(config['providers']['pool'],eligibility_desc=config['providers']['eligibility'])
    c.require(pool.get('pool_kind')=='all_source_protected_bridge' and 'input_authority' in pool,
        'Feature acquisition requires authenticated complete all-source membership')
    eligibility=providers['eligibility']
    c.require(eligibility['status']=='PASS_FROZEN_POOL_EVALUATION_ELIGIBILITY' and eligibility['pool']==config['providers']['pool']
        and eligibility['protected_QA_image_RGB_overlap_count']==0 and eligibility['missing_media_count']==0,'Unready broad pool/media/evaluation boundary')
    claim=providers['ownership'];identity=feature_identity(config)
    c.require(claim['status']=='ACTIVE_EXCLUSIVE_NATIVE_FEATURE_SHARD_CLAIM' and claim['feature_identity']==identity
        and claim['shard_index']==config['shard_index'] and claim['shard_count']==config['shard_count'],'Native feature acquisition not exclusively owned')
    review=c.checked(config['review'])
    c.require(review['status']=='PASS_INDEPENDENT_EXACT_FEATURE_ACQUISITION' and review['sources']==config['sources']
        and review['feature_identity']==identity,'Actual reviewed feature acquisition required')
    qualification=providers['qualification']
    c.require(qualification['status']=='PASS_ACTUAL_STAGE1_FEATURE_FORWARD_PARITY' and qualification['sources']==config['sources']
        and qualification['assets']==config['providers']['assets'] and qualification['environment']==config['providers']['environment'],
        'Actual feature/masking/RGB forward parity qualification required')
    c.require(config['allocation']['gpus']==1,'Frozen feature shards request one actual GPU')
    c.require(c.within(config['output'],profile['output_root']),'Unapproved feature output')
    return profile,providers,pool,rows,ids

def compute(config):
    profile,providers,pool,rows,ids=validate(config)
    c.require(Path(sys.executable).absolute()==Path(profile['python']).absolute(),'Run feature code with actual native Python')
    import torch
    import numpy as np
    from PIL import Image
    import hashlib
    c.require(torch.cuda.is_available() and torch.cuda.device_count()==1,'One real visible GPU required')
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    identity=feature_identity(config);roster=shard_ids(ids,config['shard_index'],config['shard_count'])
    index={ident:i for i,ident in enumerate(ids)}
    image_manifest=c.checked(pool['image_manifest'])['assets'];image_root=pool['image_root']
    runtime_images=None
    if 'media_transport' in config['providers']:
        from experiments.root_native_media_transport_20261023.transport import prepare,validate as validate_alias
        from .media import staging_boundary,require_launch_time
        shard_records=c.seal(output/'SHARD_RECORDS.json',[rows[index[ident]] for ident in roster])
        media_report=output/f"media-{os.environ['SLURM_JOB_ID']}-r{os.environ.get('SLURM_RESTART_COUNT','0')}"
        with staging_boundary():
            alias_desc=prepare(config['providers']['pool'],shard_records,config['providers']['media_transport'],media_report)
            require_launch_time(config['allocation'])
        runtime_images=validate_alias(alias_desc,config['providers']['pool'],shard_records)
        c.require(runtime_images['scope']=='ALL_SOURCE' and runtime_images['transport']==config['providers']['media_transport'],
            'Feature transport is not the exact broad-pool logical input')
        image_manifest=runtime_images['assets'];image_root=runtime_images['image_root']
    source=str(c.ROOT/c.NATIVE);sys.path.insert(0,source)
    patch=importlib.import_module('llava.train.llama_xformers_attn_monkey_patch');patch.replace_llama_attn_with_xformers_attn()
    module=importlib.import_module('llava.train.train')
    c.require(Path(module.__file__).resolve()==Path(source)/'llava/train/train.py','Wrong native constructor')
    from .structured_preprocess import install as install_structured_preprocess
    preprocessing_audit=install_structured_preprocess(module)
    Parent=module.LLaVATrainer
    class CaptureTrainer(Parent):
        def train(self,*a,**k):raise Captured(self)
    module.LLaVATrainer=CaptureTrainer
    # Construct the SAME Stage1 inputs/tokenizer/native lazy data. No LoRA, update,
    # optimizer, DeepSpeed checkpoint or training occurs in feature acquisition.
    fake=dict(allocation=dict(gpus=1,microbatch=1,loader_workers=0),training_seed=1234,
        role='student',output=config['output'])
    ps=dict(assets=providers['assets'],pool=pool,selection=dict(records_json=pool['records_json']))
    if runtime_images is not None:ps['runtime_images']=runtime_images
    args=cli(fake,profile,ps);values=dict(zip(args[::2],args[1::2]))
    values.pop('--deepspeed');values.update({'--lora_enable':'False','--gradient_checkpointing':'False',
        '--gradient_accumulation_steps':'1','--save_strategy':'no'})
    sys.argv=['frozen-stage1-native']+[item for k,v in values.items() for item in (k,v)]
    try:
        module.train(attn_implementation='eager');raise ValueError('Native constructor did not enter capture')
    except Captured as captured:trainer=captured.trainer
    finally:module.LLaVATrainer=Parent
    model=trainer.model.requires_grad_(False).eval().to(device='cuda',dtype=torch.float16)
    from .resume_adapter import assert_attention
    attention_audit=assert_attention(model,patch)
    c.seal(output/'FEATURE_OWNER.json',dict(feature_identity=identity,shard_index=config['shard_index'],shard_count=config['shard_count'],record_ids_sha256=c.key(roster)))
    c.seal(output/'ATTENTION_IMPLEMENTATION.json',attention_audit)
    c.seal(output/'PREPROCESSING_IMPLEMENTATION.json',preprocessing_audit)
    db_path=output/'features.sqlite3';db=sqlite3.connect(db_path)
    db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS features(record_id TEXT PRIMARY KEY,loss REAL NOT NULL,targets INTEGER NOT NULL,vision_bucket INTEGER,embedding BLOB,feature_identity TEXT NOT NULL)')
    db.commit()
    existing={ident for ident, in db.execute('SELECT record_id FROM features')}
    c.require(existing<=set(roster),'Foreign record in owned feature shard')
    c.require(all(row[0]==identity for row in db.execute('SELECT DISTINCT feature_identity FROM features')),'Feature runtime identity differs')
    requested=[False]
    def stop(*_):requested[0]=True
    signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
    stop_epoch=int(os.environ['SLURM_JOB_END_TIME'])-config['allocation']['reserve_seconds']
    capture={}
    def vision_hook(module,inputs,value):
        if isinstance(value,list):value=torch.cat(value,dim=0)
        capture['pooled']=value.detach().float().mean(dim=1).mean(dim=0).cpu()
    hook=model.get_vision_tower().register_forward_hook(vision_hook)
    projection=None
    try:
        for ident in roster:
            if ident in existing:continue
            if requested[0] or time.time()>=stop_epoch:break
            i=index[ident];row=rows[i]
            if row.get('image'):
                path=(Path(image_root)/row['image']).absolute();expected=image_manifest[row['image']]
                c.require(str(path)==expected['path'] and c.within(path,image_root) and c.sha(path)==expected['sha256'],'Encoded image identity differs')
                with Image.open(path) as image:
                    rgb=image.convert('RGB')
                    c.require(rgb.size==(expected['width'],expected['height']) and hashlib.sha256(rgb.tobytes()).hexdigest()==expected['rgb_sha256'],'DecodedRGB identity differs')
            batch=trainer.data_collator([trainer.train_dataset[i]])
            batch={k:([v.to('cuda') for v in value] if isinstance(value,list) else value.to('cuda')) for k,value in batch.items()}
            capture.clear()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                prepared=model.prepare_inputs_labels_for_multimodal(batch['input_ids'],None,batch.get('attention_mask'),None,batch['labels'],batch.get('images'),None)
                input_ids,position_ids,attention_mask,past_key_values,embeds,labels=prepared
                targets=int((labels[...,1:]!=-100).sum())
                c.require(targets>0,'Native expansion/truncation left zero assistant targets; preserve failure, do not fabricate signal')
                result=model(input_ids=input_ids,position_ids=position_ids,attention_mask=attention_mask,
                    past_key_values=past_key_values,inputs_embeds=embeds,labels=labels,use_cache=False,return_dict=True)
                loss=float(result.loss.detach().cpu())
            c.require(np.isfinite(loss),'Nonfinite native assistant loss')
            bucket=None;embedding=None
            if row.get('image'):
                c.require('pooled' in capture,'Native visual feature hook missing')
                vector=capture['pooled'].numpy()
                if projection is None:
                    rng=np.random.default_rng(1234);projection=rng.standard_normal((12,len(vector)),dtype=np.float32)
                bucket=sum(int(value>=0)<<j for j,value in enumerate(projection@vector))
                embedding=vector.astype('<f2').tobytes()
            db.execute('INSERT INTO features VALUES(?,?,?,?,?,?)',(ident,loss,targets,bucket,embedding,identity));db.commit()
            existing.add(ident)
    except Exception as error:
        c.seal(output/f"FAILURE-{os.environ.get('SLURM_JOB_ID','nojob')}-r{os.environ.get('SLURM_RESTART_COUNT','0')}.json",
            dict(feature_identity=identity,record_id=ident,eligible_position=i,
                original_source_index=pool['source_indices'][i],records_complete=len(existing),
                error_type=type(error).__name__,error=str(error),failed_attempt_preserved=True))
        raise
    finally:
        hook.remove();db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close()
    completed=len(existing)==len(roster)
    receipt=dict(status='COMPLETE_NATIVE_FEATURE_SHARD' if completed else 'DURABLE_MISSING_ONLY_CONTINUATION',
        feature_identity=identity,shard_index=config['shard_index'],shard_count=config['shard_count'],
        records_complete=len(existing),records_required=len(roster),record_ids_sha256=c.key(roster),
        database=c.desc(db_path),pool=config['providers']['pool'],assets=config['providers']['assets'],
        eligibility=config['providers']['eligibility'],
        actual_GPU_forward=True,downstream_utility_established=False)
    if completed:c.seal(output/'COMPLETE.json',receipt)
    return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);a=p.parse_args()
    binding=c.read(a.binding);config=c.checked(binding['configuration'])
    profile,_,_,_,_=validate(config)
    if Path(sys.executable).absolute()!=Path(profile['python']).absolute():
        os.execv(profile['python'],[profile['python'],'-B','-m','experiments.root_native_campaign_v7_20261024.features','--binding',a.binding])
    from .worker import actual
    actual(binding,config,profile)
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    import fcntl
    execution_lock=open(output/'EXECUTION.lock','a+')
    fcntl.flock(execution_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'COMPLETE.json').exists():
        complete=c.read(output/'COMPLETE.json')
        c.require(complete['feature_identity']==feature_identity(config) and c.desc(complete['database']['path'])==complete['database'],'Completed feature shard changed')
        print(c.canonical(dict(status='ALREADY_COMPLETE_NO_ACQUISITION_REPLAY',complete=c.desc(output/'COMPLETE.json'))).decode())
        return 0
    attempt=output/f"attempt-{binding['job_id']}-r{os.environ.get('SLURM_RESTART_COUNT','0')}"
    started=time.monotonic()
    from .media import StagingContinuation,requeue
    try:result=compute(config)
    except StagingContinuation as error:
        requeue(binding,attempt,error)
        return 0
    c.seal(attempt/'RESULT.json',dict(result=result,elapsed_seconds=time.monotonic()-started,
        supervisor_GPU_hours=(time.monotonic()-started)/3600,setup_and_failed_costs_also_count=True))
    print(c.canonical(result).decode())
    if result['status']=='COMPLETE_NATIVE_FEATURE_SHARD':return 0
    c.seal(attempt/'REQUEUE_INTENT.json',dict(job_id=binding['job_id'],accepted_database=result['database']))
    import subprocess
    requeue=subprocess.run(['scontrol','requeue',binding['job_id']],capture_output=True,text=True)
    c.seal(attempt/'REQUEUE_RESPONSE.json',dict(rc=requeue.returncode,stdout=requeue.stdout,stderr=requeue.stderr))
    c.require(requeue.returncode==0,'Feature requeue unacknowledged; no replacement submission')
    return 0

def collect(pool_desc,assets_desc,shard_receipt_descs,output):
    receipts=[c.checked(d) for d in shard_receipt_descs]
    c.require(receipts and all(r['eligibility']==receipts[0]['eligibility'] for r in receipts),'Mixed feature pool eligibility')
    pool,_,ids=validate_pool(pool_desc,eligibility_desc=receipts[0]['eligibility'])
    c.require(receipts and len({r['feature_identity'] for r in receipts})==1,'Mixed feature functions')
    total=receipts[0]['shard_count']
    c.require(len(receipts)==total and {r['shard_index'] for r in receipts}==set(range(total)),'Missing/duplicate feature shards')
    for r in receipts:
        roster=shard_ids(ids,r['shard_index'],total)
        c.require(r['status']=='COMPLETE_NATIVE_FEATURE_SHARD' and r['pool']==pool_desc and r['assets']==assets_desc
            and r['record_ids_sha256']==c.key(roster) and r['records_complete']==r['records_required']==len(roster),'Incomplete native feature scope')
        c.require(c.desc(r['database']['path'])==r['database'],'Feature DB changed')
    result=dict(schema='native-stage1-record-features-v1',status='PASS_COMPLETE_NATIVE_RECORD_FEATURES',
        pool=pool_desc,assets=assets_desc,stage='stage1_before_visual_instruction_tuning',count=len(ids),record_ids_sha256=c.key(ids),
        feature_identity=receipts[0]['feature_identity'],databases=[r['database'] for r in sorted(receipts,key=lambda r:r['shard_index'])],
        shard_receipts=shard_receipt_descs,downstream_learning_benefit_proven=False)
    # Read every actual measurement and enforce exact coverage before promotion.
    from .selection import load_features
    temporary=c.seal(Path(output)/'FEATURES_CANDIDATE.json',result)
    load_features(temporary,pool_desc,assets_desc,ids)
    return c.seal(Path(output)/'FEATURES.json',result)

if __name__=='__main__':raise SystemExit(main())
