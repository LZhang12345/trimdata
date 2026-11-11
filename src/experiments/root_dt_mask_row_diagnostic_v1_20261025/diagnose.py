"""One pinned training row, real native preprocessing, no models/CUDA/network."""
if not __debug__:
    raise RuntimeError("OptimizedPythonForbiddenForExactRowDiagnostic")
import argparse
from contextlib import contextmanager
import copy
import hashlib
import importlib
import json
import mmap
import os
from pathlib import Path
import pwd
import re
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

from experiments.root_native_campaign_v7_20261024 import common as c

ROOT=Path(__file__).resolve().parents[2]
CODE=Path('/home/luyangz/data_selection_multimodal/native_campaign_v3_20261023/src')
OWNER='Root_exact_DT_mask_row_diagnostic_v1'
SCOPE='TOKENIZER_ONLY_EXACT_G0_628943'
ROW_ID='628943:elqCtqo_0'
TYPED='37663c1ac8194fb50fd143dee57b742a64fdd043f7992de8e10ab8d52f0bdfe0'
TOKEN=re.compile(rb'"(?:[^"\\]|\\.)*"|[{}\[\]]')
COUNTERS={}

def selected_row(ref,index):
    """Byte hash whole pinned file, decode only one object in its top-level array."""
    path=Path(ref['path'])
    with path.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as view:
        assert len(view)==ref['bytes'] and hashlib.sha256(view).hexdigest()==ref['sha256'],'Exact record source differs'
        assert view[:64].lstrip().startswith(b'['),'Top-level record array required'
        depth=0;ordinal=-1;start=None
        for token in TOKEN.finditer(view):
            first=view[token.start()]
            if first==34:continue  # Never decode or copy an unselected string.
            if first in (91,123):
                if depth==1:
                    assert first==123,'Every selected top-level row must be an object'
                    ordinal+=1
                    if ordinal==index:start=token.start()
                depth+=1
            else:
                depth-=1
                assert depth>=0,'Malformed pinned record structure'
                if depth==1 and start is not None:
                    raw=view[start:token.end()]
                    assert len(raw)<=4*1024**2,'Selected original row exceeds bounded scope'
                    return json.loads(raw),hashlib.sha256(raw).hexdigest()
        raise ValueError('Exact eligible row absent')

def guard_actual(reg,cfg):
    assert pwd.getpwuid(os.getuid()).pw_name=='luyangz' and os.getuid()==2708899 and ROOT==CODE and Path.cwd()==CODE
    assert Path(sys.executable).absolute()==Path(cfg['python']), 'Exact NativePy executable required'
    assert reg['owner']==OWNER and reg['scope']==SCOPE and reg['configuration']==cfg['_reference']
    review=c.checked(reg['review'])
    assert review['status']=='PASS_INDEPENDENT_EXACT_DT_ROW_TOKENIZER_DIAG_SOURCE_CONFIG_V1' and review['configuration']==reg['configuration'] and review['sources']==cfg['sources']
    assert review['reviewer'] not in ('/root/scientific_packet_review','scientific_packet_review')
    parent=c.checked(reg['parent_binding']);assert parent['owner']=='Root' and parent['job_id']==os.environ['SLURM_JOB_ID']=='10922850'
    assert reg['parent_binding']==cfg['parent_binding'] and os.environ.get('SLURM_STEP_ID','').isdecimal()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('SLURM_CPUS_PER_TASK')=='1'
    raw=subprocess.check_output(['scontrol','show','job','-o',parent['job_id']],text=True,timeout=20)
    assert len(raw.strip().splitlines())==1
    v=dict(re.findall(r'(?:^|\s)([^\s=]+)=(\S+)',raw))
    expected=dict(JobId='10922850',UserId='luyangz(2708899)',JobState='RUNNING',Account='cx',Partition='general',QOS='normal',
      JobName='dsmm-root-s7-composite-g3-v1',Comment='root-s7-dt-composite-v1:'+parent['nonce'],
      Command='/home/luyangz/data_selection_multimodal/session7_fullpool_published_src_v2/experiments/root_s7_dt_composite_general_v1_20261025/run.sbatch',
      WorkDir='/home/luyangz/data_selection_multimodal/session7_fullpool_published_src_v2',NumCPUs='8',TimeLimit='02:00:00',Restarts='0',TresPerNode='gres/gpu:L40S:1',NodeList=socket.gethostname().split('.')[0])
    assert all(v.get(k)==value for k,value in expected.items()),'Current truthful parent route differs'
    clock=list(map(int,v['RunTime'].split(':')));assert len(clock)==3 and 7200-(clock[0]*3600+clock[1]*60+clock[2])>=180,'Parent lacks bounded diagnostic lifetime'
    spool=subprocess.check_output(['scontrol','write','batch_script',parent['job_id'],'-'],timeout=20)
    assert hashlib.sha256(spool).hexdigest()==cfg['parent_wrapper_sha256'],'Actual parent spool changed'
    return dict(job_id=parent['job_id'],step_id=os.environ['SLURM_STEP_ID'],node=v['NodeList'],scheduler=raw,registered_resources=reg['resources'])

def validate(cfg):
    assert cfg['python']=='/data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python'
    assert cfg['schema']=='root-exact-dt-mask-row-diagnostic-v1' and cfg['record_id']==ROW_ID
    assert cfg['eligible_position']==624408 and cfg['source_index']==628943 and cfg['typed_sources_sha256']==TYPED
    assert cfg['tokenizer_policy']==dict(model_max_length=2048,padding_side='right',use_fast=False,pad_token='unk_token',template='v1',legacy=False)
    assert cfg['flags']==dict(is_multimodal=True,mm_use_im_start_end=False,mm_use_im_patch_token=False,image_aspect_ratio='pad')
    proof=c.checked(cfg['exact_row_proof']);assert proof['actual_job_id']=='10922850' and proof['actual_step_id']=='2'
    assert proof['first_missing']==dict(shard_ordinal=39300,record_id=ROW_ID,eligible_position=624408,source_index=628943)
    assert proof['accepted_prefix_exact'] is True and proof['accepted_count']==39300 and proof['roster_count']==41538
    measured=c.checked(cfg['runtime_provenance']);assert cfg['runtime_module_bindings']=={k:measured[k] for k in ('modules','torch_C')}
    for relative,digest in cfg['sources'].items():assert c.sha(ROOT/relative)==digest,relative

def tokenizer_metadata(cfg):
    receipt=c.checked(cfg['assets']);asset=receipt['assets']['vicuna']
    assert asset['repository']=='lmsys/vicuna-7b-v1.5' and asset['revision']=='3321f76e3f527bd14065daf69dad9344000a201d'
    names={'config.json','generation_config.json','special_tokens_map.json','tokenizer_config.json','tokenizer.model'}
    refs=[r for r in asset['files'] if Path(r['path']).name in names]
    assert len(refs)==5 and {Path(r['path']).name for r in refs}==names
    for ref in refs:assert c.desc(ref['path'])=={k:ref[k] for k in ('path','bytes','sha256')},'Pinned small tokenizer bytes changed'
    assert Path(asset['snapshot']).resolve().is_relative_to(Path('/data/hf_cache').resolve())
    return asset,refs

@contextmanager
def no_neural():
    import torch
    import transformers
    assert not torch.cuda.is_initialized(),'Child inherited CUDA context'
    changes=[]
    def install(obj,name,label):
        old=getattr(obj,name)
        def fail(*args,**kwargs):
            COUNTERS[label]=COUNTERS.get(label,0)+1
            raise RuntimeError('ForbiddenDiagnosticOperation:'+label)
        changes.append((obj,name,old));setattr(obj,name,fail)
    for obj,name,label in [(socket.socket,'connect','network'),(torch,'load','torch_load'),(torch,'save','torch_save'),
        (torch.cuda,'_lazy_init','CUDA_init'),(torch.cuda,'init','CUDA_init'),(torch.cuda,'set_device','CUDA_set_device'),
        (transformers.PreTrainedModel,'__init__','pretrained_model_constructor'),(torch.nn.Parameter,'__new__','parameter_constructor')]:
        install(obj,name,label)
    try:yield
    finally:
        for obj,name,old in reversed(changes):setattr(obj,name,old)
        assert not torch.cuda.is_initialized(),'Unexpected CUDA initialization'

def digest(tensor):return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def compare(row,cfg,tokenizer,module):
    import torch
    from experiments.copilot7_datatailor_mask_repair_20261024 import hook_v1
    from experiments.root_native_campaign_v5_20261024 import structured_preprocess as s7
    from experiments.root_native_campaign_v7_20261024 import structured_preprocess as native91
    from experiments.root_native_qualification_v6_20261024.masking import author_v1
    assert row['id']=='elqCtqo_0'
    module.conversation_lib.default_conversation=module.conversation_lib.conv_templates['v1']
    template=module.conversation_lib.default_conversation
    sources=copy.deepcopy([row['conversations']]);has_image=bool(row.get('image'))
    if has_image:sources=module.preprocess_multimodal(sources,SimpleNamespace(**cfg['flags']))
    typed=hook_v1.c.key(sources)
    old=module.preprocess_v1(copy.deepcopy(sources),tokenizer,has_image=has_image)
    repaired=s7.preprocess_structured(module,copy.deepcopy(sources),tokenizer,has_image=has_image)
    n91=native91.preprocess_structured(module,copy.deepcopy(sources),tokenizer,has_image=has_image)
    oracle_error=None
    try:oracle=author_v1(row,tokenizer,template.copy())
    except ValueError as error:
        oracle=None;oracle_error=dict(error_class=type(error).__name__,message_sha256=hashlib.sha256(str(error).encode()).hexdigest())
    ids=repaired['input_ids'][0].tolist();labels=repaired['labels'][0].tolist()
    original_count=int((old['labels'][:,1:]!=-100).sum())
    repaired_count=int((repaired['labels'][:,1:]!=-100).sum())
    expanded=[]
    for token,target in zip(ids[:2048],labels[:2048]):expanded.extend([-100]*576 if token==-200 else [target])
    expanded=expanded[:2048]
    checks=dict(typed_sources_match=typed==TYPED,original_input_ids_equal=torch.equal(old['input_ids'],repaired['input_ids']),
      native91_input_ids_equal=torch.equal(n91['input_ids'],repaired['input_ids']),native91_labels_equal=torch.equal(n91['labels'],repaired['labels']),
      oracle_input_ids_equal=oracle is not None and ids==oracle['input_ids'],oracle_labels_equal=oracle is not None and labels==oracle['labels'],
      original_zero_targets=original_count==0,actual_repaired1175=repaired_count==1175,
      dtype_shapes_unchanged=old['input_ids'].dtype==repaired['input_ids'].dtype and old['labels'].dtype==repaired['labels'].dtype and old['input_ids'].shape==repaired['input_ids'].shape==old['labels'].shape==repaired['labels'].shape,
      positive_label_only_expansion=sum(x!=-100 for x in expanded[1:])>0)
    return dict(record_id=ROW_ID,typed_sources_sha256=typed,checks=checks,
      native_shifted_targets=original_count,repaired_shifted_targets=repaired_count,
      input_ids_sha256=digest(repaired['input_ids']),old_labels_sha256=digest(old['labels']),
      repaired_labels_sha256=digest(repaired['labels']),native91_labels_sha256=digest(n91['labels']),
      oracle_input_ids_sha256=hashlib.sha256(c.canonical(oracle['input_ids'])).hexdigest() if oracle else None,
      oracle_labels_sha256=hashlib.sha256(c.canonical(oracle['labels'])).hexdigest() if oracle else None,oracle_error=oracle_error,
      full_sequence_length=len(ids),truncated_length=min(2048,len(ids)),image_record=has_image,
      image_token_count=ids.count(-200),label_only_expanded_length=len(expanded),
      label_only_expanded_targets=sum(x!=-100 for x in expanded[1:]),actual_multimodal_model_executed=False,
      expansion_scope='Derived576-patch label projection only, not actualmodel/preparation proof',
      role_delimiter_occurrences={key:sum(t['value'].count(key) for t in row['conversations']) for key in ['USER:','ASSISTANT:','</s>']},
      template=dict(version=template.version,roles=list(template.roles),sep_style=template.sep_style.name,sha256=c.key(dict(system=template.system,roles=template.roles,sep=template.sep,sep2=template.sep2))),
      raw_QA_tokens_or_targets_exported=False,automatic_mask_permit=False,wholepool_or_method_PASS=False)

def run(registration):
    start=time.monotonic();reg=c.read(registration);cfg=c.checked(reg['configuration']);validate(cfg)
    cfg=dict(cfg,_reference=reg['configuration']);actual=guard_actual(reg,cfg)
    assert reg['resources']==dict(cpus=1,memory_mb=8192,maximum_seconds=180,gpus=0)
    out=Path(reg['output']);allowed=CODE/'collaboration/root-babel-queue-expansion-20261025/g0-tokenizer-row-diagnostic-v1'
    assert out.parent==allowed and out.name.startswith('actual-10922850-step') and '..' not in out.parts
    out.mkdir(exist_ok=False)
    c.seal(out/'ACTUAL.json',dict(actual=actual,registration=c.desc(registration)))
    try:
        pool=c.checked(cfg['pool']);assert pool['count']==660763 and pool['record_ids'][624408]==ROW_ID and pool['source_indices'][624408]==628943
        row,rowsha=selected_row(pool['records_json'],624408)
        assert row['id']=='elqCtqo_0'
        asset,refs=tokenizer_metadata(cfg)
        with no_neural():
            from experiments.root_babel_start_evaluation_v2_20261025.worker import module_provenance,verify_module_provenance
            provenance=module_provenance();c.seal(out/'ACTUAL_MODULE_PROVENANCE.json',provenance);verify_module_provenance(provenance,cfg)
            import torch
            torch.set_num_threads(1)
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(asset['snapshot'],use_fast=False,model_max_length=2048,padding_side='right',local_files_only=True)
            tokenizer.pad_token=tokenizer.unk_token
            assert tokenizer.legacy is False and len(tokenizer)==32000 and tokenizer.pad_token_id==0 and tokenizer.bos_token_id==1 and tokenizer.eos_token_id==2
            assert tokenizer.model_max_length==2048 and tokenizer.padding_side=='right'
            sys.path.insert(0,str(ROOT/c.NATIVE))
            module=importlib.import_module('llava.train.train')
            assert Path(module.__file__).resolve()==(ROOT/c.NATIVE/'llava/train/train.py').resolve()
            assert Path(module.preprocess_v1.__code__.co_filename).resolve()==Path(module.__file__).resolve() and module.preprocess_v1.__name__=='preprocess_v1','Native original preprocessor was already replaced'
            result=compare(row,cfg,tokenizer,module)
        result.update(actual=actual,row_serialized_sha256=rowsha,tokenizer_files=refs,
          tokenizer_actual=dict(legacy=tokenizer.legacy,pad_token_id=tokenizer.pad_token_id,model_max_length=tokenizer.model_max_length,padding_side=tokenizer.padding_side),
          guarded_operation_attempts=COUNTERS,elapsed_seconds=time.monotonic()-start,
          status='PASS_ONE_ACTUAL_ROW_TOKENIZER_COMPARISON_NOT_REPAIR_AUTHORITY' if all(result['checks'].values()) else 'VALID_NEGATIVE_ROW_COMPARISON_DIAGNOSIS')
        c.seal(out/'ACTUAL_ROW_COMPARISON.json',result)
        assert not any(COUNTERS.values()),'Forbidden operation attempted; actual result preserved'
        return c.desc(out/'ACTUAL_ROW_COMPARISON.json')
    except BaseException as error:
        c.seal(out/'FAILURE.json',dict(actual=actual,error_class=type(error).__name__,message_sha256=hashlib.sha256(str(error).encode()).hexdigest(),guarded_operation_attempts=COUNTERS,elapsed_seconds=time.monotonic()-start,no_automatic_case_approval=True))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--registration',required=True)
    print(json.dumps(run(parser.parse_args().registration)))
