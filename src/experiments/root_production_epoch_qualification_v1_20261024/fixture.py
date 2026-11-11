"""Freeze1153 real eligible original records; selected-only encoded/RGB checks.

Called after the combined Root prep worker authenticates its current GPU job.
No model, network, scorer, opaque QA bank or training work is performed here.
"""
import argparse,collections,hashlib,json,os,pwd,re,socket,time
from pathlib import Path
from experiments.root_native_input_authority_20261023 import pool_authority_v1 as authority
c=authority.c;ROOT=Path(__file__).resolve().parents[2]
PACKAGE='experiments/root_production_epoch_qualification_v1_20261024'
COUNTS=dict(coco=384,gqa=192,ocr_vqa=192,vg=192,textvqa=128,text_only=65)
HOME='/home/luyangz/data_selection_multimodal'
OUTPUT='/data/user_data/luyangz/native_campaign_20261023/natural-epoch-qualification-v1/fixture-1153'
REPORT=HOME+'/collaboration/root-production-epoch-qualification-v1-20261024/fixture-1153'
POOL=dict(path='/data/user_data/luyangz/native_campaign_20261023/root-wholepool-authority-v1/authority/POOL.json',bytes=18754787,sha256='c76040a17ca0dc1e99fea14514618edade3171a9b05f6ccc3e4751a5939e4964')
ELIGIBILITY=dict(path='/data/user_data/luyangz/native_campaign_20261023/root-wholepool-eligibility-v1/ELIGIBILITY.json',bytes=11168,sha256='3e0459fd6c6fa2ab694b44c60d423fa7b0e662dcb24a54bbcf4082a7a88ec66e')

def sources():
 names=[PACKAGE+'/__init__.py',PACKAGE+'/fixture.py','experiments/root_native_input_authority_20261023/__init__.py','experiments/root_native_input_authority_20261023/pool_authority_v1.py','experiments/root_native_campaign_20261023/__init__.py','experiments/root_native_campaign_20261023/common.py']
 return {name:c.sha(ROOT/name) for name in sorted(names)}

def choose(pool,rows,*,counts=COUNTS):
 ids=pool['record_ids'];indices=pool['source_indices']
 c.require(indices==sorted(indices),'Original source order differs')
 c.require(len(rows)==len(ids)==len(indices)==pool['count'] and len(set(ids))==len(ids) and len(set(indices))==len(indices),'Exact eligible original whole-record membership required')
 bins=collections.defaultdict(list)
 for position,(row,ident,index) in enumerate(zip(rows,ids,indices)):
  c.require(ident==authority.record_id(index,row),'Original index/record ID changed')
  authority.turns(row);bins[authority.source(row)].append(position)
 selected=set()
 for name,amount in counts.items():
  c.require(type(amount) is int and amount>0 and len(bins[name])>=amount,'Missing source quota '+name)
  ranked=sorted(bins[name],key=lambda i:(c.key(['all-source-natural-epoch-fixture-v1',1234,ids[i]]),ids[i]))
  selected.update(ranked[:amount])
 positions=[i for i in range(len(rows)) if i in selected]
 c.require(len(positions)==sum(counts.values()) and positions==sorted(positions),'Whole original order/count changed')
 return positions

def coverage(rows):
 return dict(source_counts=dict(collections.Counter(authority.source(r) for r in rows)),long=sum(sum(len(t['value']) for t in r['conversations'])>=2000 for r in rows),multi_turn=sum(len(r['conversations'])>=4 for r in rows),text_only=sum(not r.get('image') for r in rows),image=sum(bool(r.get('image')) for r in rows))

def check_image(name,expected,root):
 from PIL import Image
 path=Path(root)/name
 c.require(not Path(name).is_absolute() and '..' not in Path(name).parts and c.within(path,root) and str(path)==expected['path'],'Unsafe or mismatched selected media locator')
 c.require(path.is_file() and path.stat().st_size==expected['bytes'] and c.sha(path)==expected['sha256'],'Selected encoded image identity differs')
 with Image.open(path) as image:
  rgb=image.convert('RGB');c.require(rgb.size==(expected['width'],expected['height']) and hashlib.sha256(rgb.tobytes()).hexdigest()==expected['rgb_sha256'],'Selected decoded RGB identity differs')
 return expected

def parent_guard(actual,binding):
 c.require(pwd.getpwuid(os.getuid()).pw_name=='luyangz' and os.getuid()==2708899 and Path.cwd()==Path(HOME),'Actual Root HOME metadata worker only')
 job=os.environ.get('SLURM_JOB_ID','');c.require(job.isdecimal() and actual['job_id']==binding['job_id']==job and not os.environ.get('SLURM_ARRAY_JOB_ID'),'Actual combined scalar prep parent required before DATA')
 f=dict(re.findall(r'(?:^|\s)([^\s=]+)=(\S+)',actual['scheduler']))
 c.require(f.get('JobId')==job and f.get('JobState')=='RUNNING' and f.get('UserId')=='luyangz(2708899)' and f.get('JobName')=='dsmm-root-s7-production-prep-v1' and f.get('NodeList')==socket.gethostname().split('.')[0] and f.get('Partition')=='general' and f.get('QOS')=='normal' and f.get('TresPerNode')=='gres/gpu:L40S:1','Registered owned General GPU parent differs')
 return job

def materialize(request,actual,binding):
 job=parent_guard(actual,binding)
 import PIL
 c.require(PIL.__version__=='10.2.0','Exact native Pillow10.2.0 selected-image decoder required')
 c.require(request['schema']=='root-all-source-natural-epoch-fixture-v1' and request['sources']==sources() and request['parent_binding']['path']==HOME+'/collaboration/root-s7-production-prep-v1-20261024/BINDING.json' and request['pool']==POOL and request['eligibility']==ELIGIBILITY and request['counts']==COUNTS and request['selection_seed']==1234 and request['count']==1153 and request['output']==OUTPUT and request['report']==REPORT,'Exact reviewed1153 fixture request required')
 for target in (OUTPUT,REPORT):c.require(all(not x.is_symlink() for x in (Path(target),*Path(target).parents)),'Symlink destination refused')
 output=Path(OUTPUT);report=Path(REPORT)
 # Existing genuine authority defaults to logical validation; this explicitly
 # replays the exact complete partition, NOT all348055 image payloads again.
 started=time.monotonic();pool,rows,ids=authority.validate_pool(POOL,ELIGIBILITY,full_media_revalidation=False)
 positions=choose(pool,rows);chosen=[rows[i] for i in positions];chosen_ids=[ids[i] for i in positions];indices=[pool['source_indices'][i] for i in positions];types=coverage(chosen)
 c.require(types['source_counts']==COUNTS and types['long']>=32 and types['multi_turn']>=128 and types['text_only']==65 and types['image']==1088,'Frozen broad-source/type coverage failed; do not alter/drop/top-up rows')
 images=c.checked(pool['image_manifest']);selected_images={name:images['assets'][name] for name in sorted({r['image'] for r in chosen if r.get('image')})}
 for name,value in selected_images.items():
  c.require(time.time()<actual['end_epoch']-300,'Actual prep deadline reserve reached; no fixture promotion')
  check_image(name,value,pool['image_root'])
 # No pixels are written/copied. All original image locators/encoded bytes stay.
 output.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
 records=c.seal(output/'records.json',chosen)
 image_ref=c.seal(output/'IMAGES.json',dict(schema='native-engineering-image-manifest-v1',image_root=pool['image_root'],assets=selected_images))
 protected=c.seal(output/'PROTECTED_EXCLUSIONS.json',dict(status='DERIVED_EXACT_SUBSET_OF_INDEPENDENT_ELIGIBLE_AUTHORITY',parent_pool=POOL,parent_eligibility=ELIGIBILITY,parent_authority=pool['input_authority'],original_indices=indices,original_row_sha256=[authority.row_hash(r) for r in chosen],protected_QA_image_RGB_overlap_count=0,new_exclusions_added=False))
 fixture=dict(schema='native-engineering-pool-v1',status='PASS_ACTUAL_NATIVE_ENGINEERING_INPUTS',engineering_only=True,count=1153,record_ids=chosen_ids,source_indices=indices,records_json=records,image_manifest=image_ref,image_root=pool['image_root'],protected_exclusions=protected,source_json=pool['source_json'],parent_pool=POOL,parent_eligibility=ELIGIBILITY,source_counts=COUNTS)
 fixture_ref=c.seal(output/'ENGINEERING_POOL.json',fixture)
 selected=c.seal(output/'SELECTION.json',dict(schema='native-record-selection-v1',status='PASS_SELECTION',pool=fixture_ref,method='full',fraction='1',selection_seed=1234,count=1153,record_ids=chosen_ids,records_json=records,groups=None,features=None,assets=None,algorithm_sources=sources(),budget=dict(unit='whole_conversation_count',value=1153,fraction='1',eligible_pool_count=1153)))
 # Deliberately a DRAFT. A genuine independent actual metadata reviewer must
 # authenticate this subset before sealing the Native config eligibility.
 eligibility=c.seal(output/'ELIGIBILITY_DRAFT.json',dict(status='DRAFT_DERIVED_ENGINEERING_ELIGIBILITY_PENDING_INDEPENDENT_REVIEW',pool=fixture_ref,parent_pool=POOL,parent_eligibility=ELIGIBILITY,protected_QA_image_RGB_overlap_count=0,missing_media_count=0,evaluation_protocol=c.checked(ELIGIBILITY)['evaluation_protocol'],subset_complete_exact=True))
 result=dict(status='COMPLETE_ACTUAL_ENGINEERING_FIXTURE_PENDING_INDEPENDENT_METADATA_REVIEW',actual=actual,parent_binding=request['parent_binding'],pool=fixture_ref,selection=selected,eligibility_draft=eligibility,records=records,image_manifest=image_ref,protected_exclusions=protected,count=1153,source_counts=COUNTS,type_coverage=types,unique_images=len(selected_images),selected_images_encoded_RGB_checked=len(selected_images),native_Pillow=PIL.__version__,parent_complete_partition_logically_replayed=True,full_media_revalidation=False,original_order_preserved=True,network_calls=0,models_opened=0,trainer_qualified=False,source_map=sources(),elapsed_seconds=time.monotonic()-started)
 c.seal(output/'COMPLETE.json',result);return c.seal(report/'COMPLETE.json',result)

def main():
 a=argparse.ArgumentParser();a.add_argument('--request',required=True);a.add_argument('--parent-binding',required=True);a.add_argument('--actual-parent',required=True);v=a.parse_args();request=c.checked(c.desc(v.request));binding=c.checked(c.desc(v.parent_binding));actual=c.checked(c.desc(v.actual_parent));print(json.dumps(materialize(request,actual,binding),sort_keys=True))
if __name__=='__main__':main()
