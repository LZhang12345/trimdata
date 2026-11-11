"""Actual pinned-tokenizer repair qualification, inside one owned CPU+GPU job.

No model weights/images/forward/training; full token/mask records stay remote.
No record filtering, replacement, synthetic supervision or runtime promotion.
"""
import argparse
from contextlib import redirect_stdout
import copy
import importlib
import io
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from . import common as c
from .structured_preprocess import preprocess_structured


def tokens(result):
    return result['input_ids'][0].tolist()[:2048],result['labels'][0].tolist()[:2048]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--intent',required=True);a=parser.parse_args()
    intent=c.read(a.intent);started=time.monotonic()
    c.require(intent['status']=='ROOT_REGISTERED_STRUCTURED_NATIVE_TOKENIZER_PARITY'
        and intent['parent_job']==os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') is not None,
        'Exact registered owned CPU+GPU job/step required')
    c.require(intent['source_sha256']==c.sha(__file__),'Parity source changed')
    c.verify_sources(intent['repair_sources'])
    from experiments.root_native_qualification_v6_20261024 import common as oracle_common
    c.require(oracle_common.sources()==intent['oracle_sources'],'Independent oracle/driver source changed')
    actual=subprocess.check_output(['scontrol','show','job',intent['parent_job'],'-o'],text=True,timeout=20)
    fields=dict(re.findall(r'(?:^|\s)([^\s=]+)=(\S+)',actual));tres=dict(s.split('=',1) for s in fields.get('AllocTRES','').split(',') if '=' in s)
    c.require(fields.get('JobState')=='RUNNING' and fields.get('Account')=='cx'
        and fields.get('UserId','').split('(')[0]=='luyangz' and int(tres.get('gres/gpu','0'))>=1
        and fields.get('NodeList')==socket.gethostname().split('.')[0],
        'Actual Root Babel GPU allocation required')
    cfg=c.checked(intent['native_configuration']);env=c.checked(cfg['providers']['environment'])
    c.require(Path(sys.executable).absolute()==Path(env['python_executable']).absolute(),'Use exact failed native tokenizer environment')
    old=importlib.import_module('experiments.root_native_campaign_v3_20261023.common');old.verify_sources(cfg['sources'])
    pool=c.checked(cfg['providers']['pool']);rows=c.checked(pool['records_json']);ids=pool['record_ids']
    c.require(len(rows)==len(ids)==1024 and pool['engineering_only'] is True,'This token parity scope is exact1024 original records')
    assets=c.checked(cfg['providers']['assets'])['assets'];snapshot=assets['vicuna']['snapshot'];token_files=[]
    for d in assets['vicuna']['files']:
        if Path(d['path']).name in ('tokenizer.model','tokenizer_config.json','special_tokens_map.json','added_tokens.json','config.json'):
            c.require(c.desc(d['path'])=={k:d[k] for k in ('path','bytes','sha256')},'Tokenizer bytes changed');token_files.append(d)
    c.require(any(Path(d['path']).name=='tokenizer.model' for d in token_files),'Exact tokenizer model descriptor required')
    from transformers import AutoTokenizer
    import transformers,tokenizers
    c.require(transformers.__version__=='4.37.2' and tokenizers.__version__=='0.15.1','Exact failed runtime APIs required')
    tokenizer=AutoTokenizer.from_pretrained(snapshot,use_fast=False,model_max_length=2048,padding_side='right',local_files_only=True)
    tokenizer.pad_token=tokenizer.unk_token
    sys.path.insert(0,str(c.ROOT/c.NATIVE));native=importlib.import_module('llava.train.train')
    c.require(Path(native.__file__).resolve()==(c.ROOT/c.NATIVE/'llava/train/train.py').resolve(),'Foreign native preprocessor')
    native.conversation_lib.default_conversation=native.conversation_lib.conv_templates['v1']
    from experiments.root_native_qualification_v6_20261024.masking import author_v1
    from experiments.root_native_qualification_v6_20261024.observer import expanded_labels
    results=[];native_bad=[];new_bad=[];changed=[];violations=[]
    for position,(ident,row) in enumerate(zip(ids,rows)):
        sources=[copy.deepcopy(row['conversations'])];has_image='image' in row
        if has_image:sources=native.preprocess_multimodal(sources,SimpleNamespace(is_multimodal=True,mm_use_im_start_end=False))
        before_warning=io.StringIO();after_warning=io.StringIO()
        try:
            with redirect_stdout(before_warning):old_result=native.preprocess(sources,tokenizer,has_image=has_image)
            with redirect_stdout(after_warning):new_result=preprocess_structured(native,sources,tokenizer,has_image)
            old_ids,old_labels=tokens(old_result);new_ids,new_labels=tokens(new_result)
            mask=[token!=tokenizer.pad_token_id for token in new_ids]
            old_expanded=expanded_labels([old_ids],[old_labels],[mask],-200,576,2048)[0]
            new_expanded=expanded_labels([new_ids],[new_labels],[mask],-200,576,2048)[0]
            old_count=sum(v!=-100 for v in old_labels[1:]);old_post=sum(v!=-100 for v in old_expanded[1:])
            new_count=sum(v!=-100 for v in new_labels[1:]);new_post=sum(v!=-100 for v in new_expanded[1:])
            oracle=author_v1(row,tokenizer,native.conversation_lib.default_conversation)
            oracle_equal=new_ids==oracle['input_ids'][:2048] and new_labels==oracle['labels'][:2048]
            valid_old=old_count>0 and old_post>0;valid_new=new_count>0 and new_post>0
            if not valid_old:native_bad.append(ident)
            if not valid_new:new_bad.append(ident)
            if old_labels!=new_labels:changed.append(ident)
            ordinary_equal=(not valid_old) or (old_labels==new_labels and old_expanded==new_expanded)
            record=dict(position=position,record_id=ident,input_ids_identical=old_ids==new_ids,ordinary_labels_identical=ordinary_equal,
                independent_prefix_oracle_identical=oracle_equal,native_unexpanded_targets=old_count,native_expanded_targets=old_post,
                repaired_unexpanded_targets=new_count,repaired_expanded_targets=new_post,
                native_warning=before_warning.getvalue(),repair_warning=after_warning.getvalue(),
                input_ids=new_ids,original_labels=old_labels,repaired_labels=new_labels,expanded_labels=new_expanded)
            if not valid_old:
                selected=[i for i,v in enumerate(new_labels) if v!=-100]
                start=selected[0] if selected else 0
                record['boundary_audit']=dict(first_supervised_index=start,last_supervised_index=selected[-1] if selected else None,
                    decoded_assistant_targets=tokenizer.decode([v for v in new_labels if v!=-100]),
                    boundary_token_ids=new_ids[max(0,start-12):start+12],boundary_labels=new_labels[max(0,start-12):start+12],
                    original_conversations=row['conversations'],independent_spans=oracle['rounds'])
            if not (record['input_ids_identical'] and ordinary_equal and oracle_equal and valid_new):violations.append(ident)
        except (ValueError,AssertionError,KeyError) as error:
            record=dict(position=position,record_id=ident,error=repr(error),native_warning=before_warning.getvalue(),repair_warning=after_warning.getvalue())
            violations.append(ident)
        results.append(record)
    expected=intent['expected_original_bad_record_ids'];passed=not violations and not new_bad and set(native_bad)==set(expected) and set(changed)==set(expected)
    report=dict(schema='actual-structured-native-tokenizer-parity-v1',status='PASS_ACTUAL_TOKENIZER_PARITY_ONLY' if passed else 'FAIL_ACTUAL_TOKENIZER_PARITY',
        actual_parent=actual,actual_step=os.environ['SLURM_STEP_ID'],node=socket.gethostname(),native_configuration=intent['native_configuration'],
        repair_sources=intent['repair_sources'],oracle_sources=intent['oracle_sources'],tokenizer_files=token_files,tokenizer_class=type(tokenizer).__module__+'.'+type(tokenizer).__qualname__,
        tokenizer_legacy=tokenizer.legacy,transformers_version=transformers.__version__,tokenizers_version=tokenizers.__version__,
        pool=cfg['providers']['pool'],records_checked=len(results),native_bad_record_ids=native_bad,repaired_bad_record_ids=new_bad,
        changed_label_record_ids=changed,violations=violations,expected_original_bad_record_ids=expected,records=results,
        elapsed_seconds=time.monotonic()-started,images_or_model_weights_loaded=False,new_training_or_inference=False,
        no_record_filter_or_replacement=True,native_GPU_restart_qualified=False,raw_token_masks_not_for_Git=True)
    output=c.seal(intent['output'],report)
    print(c.canonical(dict(status=report['status'],raw_remote_output=output,records_checked=len(results),original_bad=native_bad,
        repaired_bad=new_bad,changed=changed,violations=violations,elapsed_seconds=report['elapsed_seconds'])).decode())
    return 0 if passed else 2

if __name__=='__main__':raise SystemExit(main())
