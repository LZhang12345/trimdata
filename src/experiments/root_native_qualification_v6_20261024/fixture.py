"""Complete actual-record engineering fixture while broad-media providers finish."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path,PurePosixPath
from . import common as c

def protected_keys(export_descriptor,image_descriptor,repair_descriptor=None):
    export=c.checked(export_descriptor)
    c.require(export['schema']=='aces-full-large-training-inputs-v1' and export['status']=='COMPLETE_SOURCE_EXPORT_NOT_ACES_READY','Existing exact protected export required')
    ids=set();rgb={row['rgb_sha256'] for row in export['public']}
    for name,role in export['roles']['banks'].items():
        if name!='train':ids.update(role['image_ids'])
    ids.update(export['roles']['confirmation']['image_ids'])
    for rows in export['banks'].values():
        for row in rows:rgb.add(export['native_images'][str(row['image_id'])]['original']['rgb_sha256'])
    images=c.checked(image_descriptor);missing=[]
    for ident in ids:
        item=images.get(str(ident))
        original=export['native_images'].get(str(ident),{}).get('original',{})
        if item and isinstance(item.get('rgb_sha256'),str):rgb.add(item['rgb_sha256'])
        elif isinstance(original.get('rgb_sha256'),str):rgb.add(original['rgb_sha256'])
        else:missing.append(ident)
    if missing and repair_descriptor:
        from PIL import Image
        repair=c.checked(repair_descriptor)
        c.require(repair['status']=='PASS_ACTUAL_PROTECTED_RGB_COMPLETION' and repair['protected_export']==export_descriptor and repair['protected_images']==image_descriptor,'Wrong protected RGB repair provenance')
        for ident in missing[:]:
            item=repair['images'].get(str(ident))
            if item is None:continue
            c.require(c.desc(item['path'])=={k:item[k] for k in ('path','bytes','sha256')},'Protected image repair bytes differ')
            with Image.open(item['path']) as image:decoded=image.convert('RGB')
            c.require(decoded.size==(item['width'],item['height']) and hashlib.sha256(decoded.tobytes()).hexdigest()==item['rgb_sha256'],'Protected image repair RGB differs')
            rgb.add(item['rgb_sha256']);missing.remove(ident)
    c.require(not missing,'Protected/confirmation RGB coverage incomplete; no cross-source engineering PASS')
    return ids,rgb

def choose(records,count,seed):
    """Predeclared type/length coverage only; no loss or benchmark outcome."""
    c.require(count>=1024,'Need1024records to exercise true global128 accumulation')
    ranked=sorted(records,key=lambda i:(c.key(['engineering-fixture',seed,i]),i));selected=set()
    specs=[('text_only',32,lambda r:not r.get('image')),
        ('image',512,lambda r:bool(r.get('image'))),
        ('multi_turn',128,lambda r:len(r['conversations'])>=4),
        ('long',32,lambda r:sum(len(t['value']) for t in r['conversations'])>=2000)]
    coverage={}
    for name,minimum,predicate in specs:
        candidates=[i for i in ranked if predicate(records[i])]
        c.require(len(candidates)>=minimum,'Insufficient actual engineering coverage: '+name)
        selected.update(candidates[:minimum]);coverage[name]=minimum
    for ident in ranked:
        if len(selected)>=count:break
        selected.add(ident)
    c.require(len(selected)==count,'Insufficient complete eligible records')
    return sorted(selected,key=lambda i:int(i.split(':',1)[0])),coverage

def build(config):
    from PIL import Image
    c.require(config['schema']=='native-engineering-fixture-config-v1' and config['owner']=='Root','Wrong fixture configuration')
    c.require(config['driver_sources']==c.fixture_sources(),'Fixture source version changed')
    claim=c.checked(config['ownership'])
    c.require(claim['status']=='ACTIVE_EXCLUSIVE_ENGINEERING_FIXTURE_CLAIM' and claim['original_metadata']==config['original_metadata']
        and claim['output']==config['output'] and claim['count']==config['count'] and claim['seed']==config['seed'],'Fixture ownership/readiness differs')
    c.require(config['original_metadata']['sha256']=='ce959ce6e23073ee1cd1a8a2ef1c633768c10d4174327b8b2dc7113b91af6cf8','Original pinned metadata SHA differs')
    rows=c.checked(config['original_metadata']);c.require(len(rows)==665298,'Complete original pinned665K metadata required')
    protected_ids,protected_rgb=protected_keys(config['protected_export'],config['protected_images'],config.get('protected_RGB_repair'))
    inventory={};roots=set()
    for provider in config['media_providers']:
        receipt=c.checked(provider)
        c.require(receipt['status']=='PASS_MEDIA_ARCHIVE','Actual complete shared media archive provider required')
        roots.add(receipt['image_root'])
        for name,value in c.checked(receipt['inventory'])['assets'].items():
            c.require(name not in inventory or inventory[name]==value,'Conflicting media inventory locator: '+name)
            inventory[name]=value
    c.require(len(roots)==1,'Explicit coherent image root required; no guessed media paths')
    image_root=Path(next(iter(roots)))
    eligible={};indices={};exclusions=Counter()
    for index,row in enumerate(rows):
        ident=f'{index:06d}:{row.get("id", "")}';relative=row.get('image')
        if relative:
            path=PurePosixPath(relative)
            c.require(not path.is_absolute() and '..' not in path.parts,'Unsafe original media locator')
            if relative not in inventory:exclusions['media_not_yet_available']+=1;continue
            if path.parts[0]=='coco' and Path(relative).stem.isdecimal() and int(Path(relative).stem) in protected_ids:
                exclusions['protected_coco_image_id']+=1;continue
        turns=row.get('conversations',[])
        if not (len(turns)>=2 and len(turns)%2==0 and all(t.get('from')==('human' if i%2==0 else 'gpt') and isinstance(t.get('value'),str) for i,t in enumerate(turns))):
            exclusions['invalid_original_turns']+=1;continue
        eligible[ident]=row;indices[ident]=index
    # Resolve data-only RGB eligibility BEFORE freezing the fixture. Removing all
    # records on a protected image is allowed; no model score/outcome is observed.
    qualified_images={};forbidden_images=set()
    while True:
        chosen,coverage=choose(eligible,config['count'],config['seed']);new_forbidden=set()
        for ident in chosen:
            row=eligible[ident];relative=row.get('image')
            if not relative or relative in qualified_images:continue
            expected=inventory[relative];path=(image_root/relative).absolute()
            c.require(c.within(path,image_root) and str(path)==expected['path'] and c.desc(path)=={k:expected[k] for k in ('path','bytes','sha256')},'Engineering actual image bytes differ')
            with Image.open(path) as image:rgb=image.convert('RGB')
            rgb_sha=hashlib.sha256(rgb.tobytes()).hexdigest()
            if rgb_sha in protected_rgb:new_forbidden.add(relative);continue
            qualified_images[relative]=dict(expected,rgb_sha256=rgb_sha,width=rgb.width,height=rgb.height)
        if not new_forbidden:break
        forbidden_images.update(new_forbidden)
        removed=[ident for ident,row in eligible.items() if row.get('image') in new_forbidden]
        exclusions['protected_RGB']+=len(removed)
        for ident in removed:eligible.pop(ident)
    used={eligible[ident]['image'] for ident in chosen if eligible[ident].get('image')}
    qualified_images={name:value for name,value in qualified_images.items() if name in used}
    output=Path(config['output']);record_desc=c.seal(output/'records.json',[eligible[i] for i in chosen])
    images_desc=c.seal(output/'images.json',dict(assets=qualified_images))
    excluded_desc=c.seal(output/'PROTECTED_EXCLUSIONS.json',dict(status='PASS_ACTUAL_ENGINEERING_PROTECTED_BOUNDARIES',
        protected_export=config['protected_export'],protected_images=config['protected_images'],protected_RGB_repair=config.get('protected_RGB_repair'),counts=dict(exclusions),
        protected_coco_ids_sha256=c.key(sorted(protected_ids)),protected_rgb_sha256=c.key(sorted(protected_rgb)),
        actual_original_metadata=config['original_metadata'],selected_zero_protected_QA_image_RGB_overlap=True,
        missing_protected_RGB_count=0,engineering_only=True,not_a_full_pool_qualification=True))
    pool=c.seal(output/'ENGINEERING_POOL.json',dict(schema='native-engineering-pool-v1',status='PASS_ACTUAL_NATIVE_ENGINEERING_INPUTS',
        engineering_only=True,count=len(chosen),record_ids=chosen,source_indices=[indices[i] for i in chosen],
        records_json=record_desc,image_root=str(image_root),image_manifest=images_desc,protected_exclusions=excluded_desc,
        original_metadata=config['original_metadata'],declared_available_sources=sorted({r.get('image','text').split('/')[0] for r in [eligible[i] for i in chosen]}),
        type_length_coverage=coverage,all_assistant_turns_preserved=True,full_broad_pool_qualified=False))
    eligibility=c.seal(output/'ENGINEERING_ELIGIBILITY.json',dict(status='PASS_FROZEN_POOL_EVALUATION_ELIGIBILITY',pool=pool,
        scope='ENGINEERING_ONLY_NOT_BENCHMARK',protected_QA_image_RGB_overlap_count=0,missing_media_count=0,
        evaluation_protocol=config['evaluation_protocol']))
    return dict(status='COMPLETE_ACTUAL_ENGINEERING_FIXTURE_ONLY',pool=pool,eligibility=eligibility,
        counts=dict(records=len(chosen),unique_images=len(qualified_images)),scientific_pool_qualified=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    print(c.canonical(build(c.read(a.config))).decode())

if __name__=='__main__':main()
