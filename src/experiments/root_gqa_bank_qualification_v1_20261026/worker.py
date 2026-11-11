"""Bounded public GT acquisition + exact native QA/GT/RGB/scorer qualification.

Old inventory V4 remains immutable. This explicitly new function reads ONLY the
one independently inventoried public EVAL member, never selector references.
"""
import argparse
import ast
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import struct
import subprocess
import time
import urllib.error
import urllib.request
import warnings
import zipfile
import zlib

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
URL='https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip'
SIZE=1498616372
ETAG='"5c9a1875-59531234"'
MEMBER='testdev_balanced_questions.json'
SCOPE='PUBLIC_NATIVE_EVALUATION_ONLY;NO_SELECTOR_OR_PROBING_REFERENCE'
SUFFIX='\nAnswer the question using a single word or phrase.'
ATTEMPTS=[]

def require(ok,message):
    if not ok: raise ValueError(message)

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
def sha(v):return hashlib.sha256(v).hexdigest()
def desc(p):
    p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p.read_bytes()))

def literal(p):
    p=Path(p)
    require(p.is_absolute() and '..' not in p.parts and not any(x.is_symlink() for x in (p,*p.parents)), 'Literal non-symlink path required')
    return p

def read(ref,limit=32<<20):
    require(set(ref)=={'path','bytes','sha256'} and type(ref['bytes']) is int and 0<ref['bytes']<=limit
        and re.fullmatch('[0-9a-f]{64}',ref['sha256']) is not None,'Bounded exact descriptor required')
    p=literal(ref['path'])
    with p.open('rb') as f:
        before=os.fstat(f.fileno());data=f.read(limit+1);after=os.fstat(f.fileno())
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    require(stat.S_ISREG(before.st_mode) and identity(before)==identity(after)
        and len(data)==ref['bytes'] and sha(data)==ref['sha256'],'Input bytes/stat changed: '+str(p))
    return data

def checked(ref,limit=32<<20):return json.loads(read(ref,limit))
def publish(p,v):
    p=literal(p);data=canonical(v)+b'\n'
    require(len(data)<32<<20,'Bounded output required')
    with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    fd=os.open(str(p.parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return desc(p)

def sources():
    paths=[HERE/'__init__.py',HERE/'worker.py',HERE/'run.sbatch',
        ROOT/'collaboration/session6-official-eval-v2-20261024/reference_sources/GQA_c121_recommended_eval.py']
    return {str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in paths}

def fields(raw):
    rows=[r for r in raw.splitlines() if r.strip()]
    require(len(rows)==1,'One actual scheduler row required')
    matches=list(re.finditer(r'(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=',rows[0]));out={}
    require(matches and matches[0].start()==0,'Malformed scheduler row')
    for i,m in enumerate(matches):
        require(m[1] not in out,'Duplicate scheduler key')
        out[m[1]]=rows[0][m.end():matches[i+1].start() if i+1<len(matches) else len(rows[0])].strip()
    return out

def documents(binding,load=checked):
    cfg=load(binding['configuration']);review=load(binding['independent_source_config_review'])
    require(cfg['status']=='ROOT_CONFIGURED_PUBLIC_GQA_BANK_QUALIFICATION_V1'
        and cfg['owner']=='Root' and cfg['scope']==SCOPE and cfg['source_root']==str(ROOT)
        and cfg['sources']==sources() and cfg['GT_body_read_authorized'] is True
        and cfg['whole_archive_download_authorized'] is False and cfg['selector_reference_access'] is False
        and cfg['models']==cfg['forwards']==0 and cfg['output_limit_bytes']==64<<20
        and cfg['resources']==dict(cpus=8,memory_mb=16384,gpus=1,minutes=45), 'Exact new public EVAL qualification scope required')
    require(review['status']=='PASS_INDEPENDENT_PUBLIC_GQA_BANK_SOURCE_CONFIG_V1'
        and review['independent'] is True and review['reviewer'] not in ('Root','/root/telemetry_finish','')
        and review['configuration']==binding['configuration'] and review['sources']==cfg['sources']
        and review['actual_inventory']==cfg['inventory_result']
        and review['GT_member_extraction_scope_approved'] is True,'Independent actual inventory and source/config review required')
    require(binding['status']=='ROOT_BOUND_PUBLIC_GQA_BANK_QUALIFICATION_V1' and binding['owner']=='Root'
        and binding['nonce']==cfg['nonce'] and re.fullmatch('[0-9a-f]{32}',cfg['nonce']) is not None,'Exact Root binding/nonce required')
    claim=load(binding['ownership'])
    require(claim['owner']=='Root' and claim['scope']==SCOPE and claim['job_id']==binding['job_id']
        and claim['nonce']==cfg['nonce'] and claim['configuration']==binding['configuration']
        and claim['no_live_duplicate_provider'] is True,'Exact exclusive public bank ownership required')
    grant=load(binding['storage_reservation'])
    require(grant['output']==cfg['output'] and grant['configuration']==binding['configuration']
        and grant['reserved_bytes']==cfg['output_limit_bytes'],'Actual bounded output grant required')
    out=literal(cfg['output'])
    require(out.is_relative_to(Path('/data/hf_cache/project_features/DS-MultiModal/gqa_bank_qualification_20261026')),
        'Separate shared project-derived output required')
    return cfg

def actor(binding,cfg):
    job=binding['job_id'];require(job.isdecimal() and os.environ.get('SLURM_JOB_ID')==job and os.getuid()==2708899
        and not os.environ.get('SLURM_ARRAY_JOB_ID') and os.environ.get('SLURM_RESTART_COUNT','0')=='0','Exact fresh scalar actor required')
    raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=30);f=fields(raw)
    wanted=dict(JobId=job,JobName='dsmm-root-gqa-bank-v1',UserId='luyangz(2708899)',Account='cx',Partition='general',QOS='normal',
        JobState='RUNNING',NumCPUs='8',NumTasks='1',NumNodes='1',Restarts='0',TimeLimit='00:45:00',
        TresPerNode='gres/gpu:L40S:1',Command=str(HERE/'run.sbatch'),WorkDir=str(ROOT),
        Comment='dsmm-root-gqa-bank:'+binding['nonce'],NodeList=socket.gethostname().split('.')[0])
    require(all(f.get(k)==v for k,v in wanted.items()),'Actual owned General GPU/CPU route differs')
    tres=dict(x.split('=',1) for x in f['AllocTRES'].split(',') if '=' in x)
    require(tres.get('cpu')=='8' and tres.get('mem') in ('16G','16384M') and tres.get('gres/gpu')=='1','Actual resource allocation differs')
    require(subprocess.check_output(['scontrol','write','batch_script',job,'-'],timeout=30)==(HERE/'run.sbatch').read_bytes(), 'Actual wrapper spool differs')
    return dict(job_id=job,scheduler=raw,node=f['NodeList'],GPU_computation=False)

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise ValueError('Publisher redirect forbidden')

def remote_range(start,end,opener=None):
    require(type(start) is int and type(end) is int and 0<=start<=end<SIZE and end-start+1<=2<<20,'Bounded single-member HTTP range required')
    request=urllib.request.Request(URL,headers={'Accept-Encoding':'identity','If-Match':ETAG,'Range':'bytes=%d-%d'%(start,end)})
    open_url=opener or urllib.request.build_opener(NoRedirect()).open
    for attempt in range(3):
        try:
            with open_url(request,timeout=30) as r:
                ATTEMPTS.append(dict(start=start,end=end,attempt=attempt,status=r.status))
                require(r.geturl()==URL and r.status==206 and r.headers.get('ETag')==ETAG
                    and r.headers.get('Content-Range')=='bytes %d-%d/%d'%(start,end,SIZE)
                    and r.headers.get('Content-Length')==str(end-start+1),'Exact If-Match206 range required; no whole-download fallback')
                data=r.read(end-start+2)
            require(len(data)==end-start+1,'Truncated or excess range bytes')
            return data
        except urllib.error.HTTPError as e:
            ATTEMPTS.append(dict(start=start,end=end,attempt=attempt,status=e.code));retry=e.code in (429,502,503,504) and attempt<2;e.close()
            if not retry:raise
            time.sleep(attempt+1)
    raise RuntimeError('Bounded retry exhausted')

def extract_member(inventory,fetch=remote_range):
    require(inventory['status']=='PUBLIC_GQA_GT_ARCHIVE_MEMBER_INVENTORIED_BODY_NOT_READ'
        and inventory['job_id']=='10937126' and inventory['url']==URL and inventory['etag']==ETAG
        and inventory['archive_bytes']==SIZE and inventory['archive_members']==22
        and inventory['central_directory_sha256']=='2367fddf2b3924b267b3073dce02887b6677bdc3054bc23785887b7ba8c023d2','Exact paid member inventory required')
    m=inventory['exact_member']
    require(m==dict(name=MEMBER,compressed_bytes=1427737,uncompressed_bytes=10848046,compression=8,crc32=2474110405,
        filename_bytes=31,flags=0,local_header_offset=119369412,version_needed=20),'Inventoried member metadata differs')
    h=fetch(m['local_header_offset'],m['local_header_offset']+29)
    require(len(h)==30,'Exact local header length required')
    f=struct.unpack('<4s5H3I2H',h)
    require(f[0]==b'PK\x03\x04' and f[1]==20 and f[2]==0 and f[3]==8 and f[6:9]==(m['crc32'],m['compressed_bytes'],m['uncompressed_bytes'])
        and f[9]==31 and f[10]<=65504,'Local/central header mismatch; no descriptor or ZIP64 permitted')
    variable=fetch(m['local_header_offset']+30,m['local_header_offset']+29+f[9]+f[10])
    require(variable[:f[9]]==MEMBER.encode(),'Local member name changed')
    extra=variable[f[9]:];offset=0
    while offset<len(extra):
        require(offset+4<=len(extra),'Truncated extra field');kind,n=struct.unpack_from('<HH',extra,offset);offset+=4
        require(kind!=1 and offset+n<=len(extra),'ZIP64/truncated extra field');offset+=n
    start=m['local_header_offset']+30+len(variable);compressed=fetch(start,start+m['compressed_bytes']-1)
    data=decode_member(compressed,m)
    return data,dict(payload_start=start,payload_end_exclusive=start+len(compressed),compressed_sha256=sha(compressed),
        GT_bytes=len(data),GT_sha256=sha(data),whole_archive_downloaded=False)

def decode_member(compressed,m):
    require(len(compressed)==m['compressed_bytes'] and 0<m['uncompressed_bytes']<=16<<20,'Bounded exact compressed/uncompressed sizes required')
    decoder=zlib.decompressobj(-15);data=decoder.decompress(compressed,m['uncompressed_bytes']+1)
    require(len(data)==m['uncompressed_bytes'] and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
        and zlib.crc32(data)&0xffffffff==m['crc32'],'Exact bounded DEFLATE/size/CRC required')
    return data

def json_unique(data):
    def pairs(items):
        out={}
        for k,v in items:require(k not in out,'Duplicate JSON key');out[k]=v
        return out
    return json.loads(data,object_pairs_hook=pairs)

def join(native_data,gt_data):
    native=[json_unique(x) for x in native_data.splitlines() if x.strip()];gt=json_unique(gt_data)
    require(type(gt) is dict and len(native)==len(gt)==12578,'Full exact12578bank required')
    ids=set();joined=[]
    for row in native:
        qid=row['question_id'];require(type(qid) in (str,int) and str(qid) not in ids,'Unique typed native QA required');ids.add(str(qid))
        target=gt.get(str(qid));require(type(target) is dict,'Native question missing from publicGT')
        image=PurePosixPath(row['image']);require(not image.is_absolute() and '..' not in image.parts and image.suffix=='.jpg','Safe exact native JPEG path required')
        require(type(target['imageId']) in (str,int) and image.stem==str(target['imageId'])
            and row['text']==target['question']+SUFFIX and type(target['answer']) is str
            and target['isBalanced'] is True,'Exact image/question/prompt/balanced GT join required')
        joined.append(dict(question_id=qid,native_input=row,answer=target['answer'],image_id=image.stem,
            source_row_sha256=sha(canonical(row)),GT_row_sha256=sha(canonical(target))))
    require(ids==set(gt),'Extra/foreign GT key')
    return joined

def score(rows,predictions):
    require(set(predictions)=={str(r['question_id']) for r in rows} and all(type(v) is str for v in predictions.values()), 'Exact full prediction membership required')
    return 100.0*sum(predictions[str(r['question_id'])]==r['answer'] for r in rows)/len(rows)

def scorer_qualification(source,rows):
    tree=ast.parse(source)
    candidates=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='correct' for t in n.targets)]
    require(len(candidates)==1 and ast.dump(candidates[0].value)==ast.dump(ast.parse('predicted == gold',mode='eval').body), 'Pinned official exact-match predicate changed')
    gold={str(r['question_id']):r['answer'] for r in rows}
    wrong={k:v+'__guaranteed_wrong' for k,v in gold.items()}
    require(score(rows,gold)==100.0 and score(rows,wrong)==0.0,'Actual full-bank exact scoring controls failed')
    return dict(metric='Official balanced test-dev exact answer match',normalization='none',
        official_predicate_AST_verified=True,all_questions_balanced=True,identity_predictions_percent=100.0,
        deliberately_wrong_predictions_percent=0.0,model_score_produced=False)

def rgb_qualification(rows,inventory,archive_ref):
    from PIL import Image
    p=literal(archive_ref['path']);before=p.stat()
    require(before.st_size==archive_ref['bytes'] and str(p).startswith('/data/hf_cache/'),'Existing full publisher image archive required')
    wanted={r['native_input']['image'] for r in rows};images=inventory['images'];results=[]
    require(len(wanted)==len(images)==398,'Exact existing398image membership required')
    with zipfile.ZipFile(p) as z:
        names=Counter(z.namelist())
        for native in sorted(wanted):
            choices=[n for n in (native,'images/'+native) if names[n]==1]
            require(len(choices)==1,'Exact native or images/ member mapping absent/ambiguous')
            member=choices[0];info=z.getinfo(member);expected=images['native:'+PurePosixPath(native).stem]
            require(not info.is_dir() and (info.external_attr>>16)&0o170000!=0o120000
                and 0<info.file_size==expected['encoded_bytes']<=64<<20,'Expected regular bounded image member required')
            raw=z.read(info);require(len(raw)==expected['encoded_bytes'] and sha(raw)==expected['encoded_sha256'],'Existing original image bytes differ')
            with warnings.catch_warnings():
                warnings.simplefilter('error',Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as im:
                    require(im.width*im.height<=128000000,'Bounded RGB decode required');rgb=im.convert('RGB');rgb.load()
            require(rgb.size==(expected['width'],expected['height']) and sha(rgb.tobytes())==expected['rgb_sha256'],'Existing original RGB/dimensions differ')
            results.append(dict(native_image=native,member=member,encoded_sha256=expected['encoded_sha256'],rgb_sha256=expected['rgb_sha256'],width=rgb.width,height=rgb.height))
    after=p.stat();identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    require(identity(before)==identity(after),'Image archive changed during398reads')
    return results,dict(path=str(p),bytes=after.st_size,inode=after.st_ino,mtime_ns=after.st_mtime_ns,
        ctime_ns=after.st_ctime_ns,whole_archive_sha256_recomputed=False,paid_archive_descriptor=archive_ref)

def run(binding_path):
    binding=checked(desc(literal(binding_path)));cfg=documents(binding);actual=actor(binding,cfg)
    out=literal(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    publish(out/'START.json',dict(actual=actual,configuration=binding['configuration']))
    try:
        protocol=checked(cfg['protocol']);qa=checked(cfg['native_QA_result']);inventory=checked(cfg['inventory_result'])
        require(protocol['scope']==SCOPE and protocol['selection_reference_access'] is False
            and qa['job_id']=='10935465' and qa['member_sha256']=='515aab3142889269480c71212d3b5b8118704d8da6637d767a0cb5bdf80a3d35'
            and qa['questions']==12578 and qa['archive']==protocol['native_input_archive'],'Exact new EVAL authorization and paid native QA required')
        archive=read(protocol['native_input_archive'])
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            members=[i for i in z.infolist() if i.filename=='gqa/llava_gqa_testdev_balanced.jsonl']
            require(len(members)==1,'Unique authorized native member required');native=z.read(members[0])
        require(sha(native)==qa['member_sha256'] and len(native)==qa['member_bytes'],'Paid native QA bytes changed')
        gt,evidence=extract_member(inventory)
        with (out/'testdev_balanced_questions.json').open('xb') as f:f.write(gt);f.flush();os.fsync(f.fileno())
        rows=join(native,gt);roster=publish(out/'ROSTER.json',dict(scope=SCOPE,records=rows))
        rgb,archive_observation=rgb_qualification(rows,checked(protocol['existing398_media']),protocol['existing_full_publisher_image_archive'])
        scoring=scorer_qualification(read(cfg['official_scorer_source']).decode(),rows)
        result=dict(status='QUALIFIED_PUBLIC_GQA12578_QA_GT_RGB_AND_EXACT_MATCH_SCORE_INPUTS',actual=actual,
            questions=len(rows),images=len(rgb),roster=roster,GT=desc(out/'testdev_balanced_questions.json'),
            GT_transport=evidence,RGB=publish(out/'RGB.json',dict(images=rgb,archive_observation=archive_observation)),
            scoring=scoring,scope=SCOPE,selector_reference_access=False,models=0,forwards=0,new_image_downloads=0,
            public_GT_member_downloaded_bytes=evidence['payload_end_exclusive']-evidence['payload_start'],
            transport_attempts=ATTEMPTS,model_generation_or_student_READY=False,elapsed_seconds=time.monotonic()-start)
        require(sum(p.stat().st_size for p in out.iterdir() if p.is_file())<cfg['output_limit_bytes'],'Output envelope exceeded')
        publish(out/'RESULT.json',result);print(json.dumps(desc(out/'RESULT.json')),flush=True)
    except BaseException as e:
        publish(out/'FAILURE.json',dict(status='GQA_BANK_QUALIFICATION_FAILED_NO_AUTO_RETRY',actual=actual,error=repr(e),
            transport_attempts=ATTEMPTS,elapsed_seconds=time.monotonic()-start));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--binding',required=True);run(parser.parse_args().binding)
