"""Answer-free inference rosters with distinct original official GT joins."""
import copy
from pathlib import Path, PurePosixPath

from experiments.root_gqa_bank_qualification_v1_20261026 import worker as u
from experiments.root_babel_start_evaluation_v2_20261025 import worker as original
from experiments.session6_official_eval_v2_20261024 import scoring
from experiments.session6_official_eval_v2_20261024.journal import semantic_key
from experiments.session6_start_evaluation_v1_20261024.bank import build_jsonl
from experiments.root_scienceqa_native_bank_v1_20261025 import bank as science
from experiments.session6_scienceqa_bank_v1_20261024.scoring import score_complete

COUNTS={'POPE':8910,'TextVQA':5000,'GQA':12578,'ScienceQA_image':2017}
IMAGE_COUNTS={'POPE':500,'TextVQA':3166,'GQA':398,'ScienceQA_image':2017}
GQA_SOURCE_SHA='515aab3142889269480c71212d3b5b8118704d8da6637d767a0cb5bdf80a3d35'

def textvqa_targets(rows,annotation):
    data=annotation['data'];u.require(len(rows)==len(data)==5000,'Full5000 TextVQA bank required')
    typed=lambda v:(type(v).__name__,v)
    targets={typed(r['question_id']):r for r in data}
    u.require(len(targets)==len(data) and set(targets)=={typed(r['native_id']) for r in rows},'Exact typed TextVQA GT membership required')
    result={}
    for r in rows:
        t=targets[typed(r['native_id'])]
        u.require(str(t['image_id'])==PurePosixPath(r['native_input']['image']).stem and
            type(t['answers']) is list and len(t['answers'])==10 and all(type(a) is str for a in t['answers']),
            'TextVQA native image/ten official human answers differ')
        result[r['semantic_key']]=list(t['answers'])
    return result

def gqa_row(r,index):
        native=r['native_input'];qid=r['question_id']
        u.require(set(native)=={'category','question_id','image','text'} and native['category']=='default'
            and type(qid) in (str,int) and native['question_id']==qid
            and type(native['question_id']) is type(qid) and u.sha(u.canonical(native))==r['source_row_sha256']
            and native['text'].endswith(u.SUFFIX) and type(r['answer']) is str
            and PurePosixPath(native['image']).stem==r['image_id'],'Qualified GQA native input/typed ID changed')
        key=semantic_key('GQA',GQA_SOURCE_SHA,index,qid)
        return dict(bank='GQA',source_sha256=GQA_SOURCE_SHA,original_row_index=index,native_id=qid,semantic_key=key,
            image_reference='native:'+r['image_id'],native_input=copy.deepcopy(native)),r['answer']

def gqa_roster(joined):
    u.require(len(joined)==12578,'Full12578 GQA required')
    rows=[];targets={}
    for index,r in enumerate(joined):
        row,answer=gqa_row(r,index);rows.append(row);targets[row['semantic_key']]=answer
    u.require(len(targets)==len(rows) and len({str(r['native_id']) for r in rows})==len(rows),'GQA duplicate semantic/native QA')
    return dict(schema='qualified-native-full-GQA12578-answer-free-roster-v1',bank='GQA',records=rows,expected_questions=len(rows),
        inference_answers_excluded=True,full_declared_scope=True),targets

def proof_media(inventory):
    refs={original.c.key(v['source_proof']):v['source_proof'] for v in inventory['images'].values()}
    proofs={k:original.c.checked(ref) for k,ref in refs.items()}
    for v in inventory['images'].values():
        expected=proofs[original.c.key(v['source_proof'])]['images'][v['source_proof_key']]
        u.require(all(expected.get(k)==v[k] for k in ('encoded_bytes','encoded_sha256','rgb_sha256','width','height')),
            'Historical complete encoded/RGB proof differs')

def load(bank,cfg):
    spec=cfg['bank_inputs'][bank];scope=original.c.checked(cfg['frozen_scope'])['banks']
    if bank=='POPE':
        base=original.c.checked(cfg['official_POPE_configuration'])
        roster,inventory,targets,snapshot=original.bound_inputs(base)
        return roster,inventory,targets,dict(base,output=str(Path(cfg['output'])/bank)),snapshot
    if bank in ('TextVQA','ScienceQA_image'):
        frozen=scope[bank];u.require(frozen['media']==spec['media'],'Exact frozen bank media descriptor required')
        if bank=='TextVQA':
            component=frozen['components'][0]
            roster=build_jsonl(bank,frozen,component['source'],component['support'])
            targets=textvqa_targets(roster['records'],original.c.checked(spec['official_annotations']))
        else:
            built=science.build(frozen,read_aliases=None);roster,targets=built['roster'],built['GT']
        inventory=original.c.checked(spec['media']);proof_media(inventory)
        if bank=='TextVQA':
            root=u.literal(spec['image_root'])
            u.require(root==Path('/data/hf_cache/datasets/dsmm_llava_665k_v1/media/textvqa/train_images'),
                'Exact actual native fullpool TextVQA file root required')
            inventory=copy.deepcopy(inventory)
            for row in roster['records']:
                name=PurePosixPath(row['native_input']['image'])
                u.require(len(name.parts)==1 and name.suffix=='.jpg','Only original native TextVQA basename supported')
                expected=inventory['images'][row['image_reference']];path=u.literal(root/str(name))
                ref=dict(path=str(path),bytes=expected['encoded_bytes'],sha256=expected['encoded_sha256'])
                u.require(original.c.desc(path)==ref,'Existing TextVQA original file differs from exact historical media')
                expected['storage_locator']=dict(kind='shared_original_file',file=ref)
            physical=dict(output=str(Path(cfg['output'])/bank),cache_root='/data/hf_cache');snapshot=None
        else:
        # No guess from MEDIA namespaces. Root pins an actual existing exact
        # archive plus a qualified member mapping before this consumer releases.
            mapping=spec['archive_mapping'];u.require(mapping['kind']=='prefix','Reviewed original prefix mapping required')
            u.require(original.c.desc(mapping['archive']['path'])==mapping['archive'],'Exact original shared image archive changed')
            physical=dict(output=str(Path(cfg['output'])/bank),cache_root='/data/hf_cache',archive_mapping=mapping)
            snapshot=original.media.fingerprint(mapping['archive']['path'])
    else:
        u.require(bank=='GQA','Unknown bank')
        result=original.c.checked(spec['qualified_result'])
        u.require(result['status']=='QUALIFIED_PUBLIC_GQA12578_QA_GT_RGB_AND_EXACT_MATCH_SCORE_INPUTS'
            and result['questions']==12578 and result['images']==398 and result['selector_reference_access'] is False,
            'Actual complete GQA provider required')
        roster,targets=gqa_roster(original.c.checked(result['roster'])['records'])
        qcfg=original.c.checked(spec['qualification_configuration']);protocol=original.c.checked(qcfg['protocol'])
        u.require(qcfg['scope']==u.SCOPE and qcfg['selector_reference_access'] is False and protocol['selection_reference_access'] is False,
            'Original GQA public-evaluation-only provider scope required')
        rgb=original.c.checked(result['RGB']);inventory=original.c.checked(protocol['existing398_media']);images=inventory['images']
        u.require(len(rgb['images'])==len(images)==398,'Exact398 image provider required')
        members={};observed=rgb['archive_observation'];archive=observed['paid_archive_descriptor']
        snapshot={k:observed[k] for k in ('path','bytes','inode','mtime_ns','ctime_ns')}
        u.require(original.media.fingerprint(archive['path'])==snapshot,'Qualified GQA archive changed')
        for entry in rgb['images']:
            key='native:'+PurePosixPath(entry['native_image']).stem;expected=images[key]
            u.require(all(expected[k]==entry[k] for k in ('encoded_sha256','rgb_sha256','width','height')),'Actual GQA RGB proof changed')
            u.require(entry['native_image'] not in members,'Duplicate GQA member binding');members[entry['native_image']]=entry['member']
        physical=dict(output=str(Path(cfg['output'])/bank),cache_root='/data/hf_cache',explicit_members=members,archive=archive)
    rows=roster['records']
    u.require(len(rows)==COUNTS[bank] and len(inventory['images'])==IMAGE_COUNTS[bank] and {r['image_reference'] for r in rows}==set(inventory['images'])
        and all('answer' not in r['native_input'] and 'answers' not in r['native_input'] for r in rows),
        'Complete answer-free QA/media membership required')
    return roster,inventory,targets,physical,snapshot

def score(bank,roster,journal,targets):
    rows=roster['records'];entries=original.entries(journal,rows)
    u.require(len(entries)==len(rows)==COUNTS[bank] and not journal.missing(),'Partial bank cannot produce final score')
    if bank=='POPE':
        value=original.result_score({'bank':bank},rows,journal,targets)
        if value['status']=='COMPLETE_OFFICIAL_POPE8910_START':value['status']='COMPLETE_OFFICIAL_POPE8910_TRAINED'
        return value
    if bank=='ScienceQA_image':return score_complete(roster,targets,entries)
    if bank=='TextVQA':return scoring.textvqa([dict(prediction=e['raw']['text'],answers=targets[r['semantic_key']]) for r,e in zip(rows,entries)])
    u.require(bank=='GQA','Unknown scorer')
    return dict(metric='Official balanced test-dev exact answer match',normalization='none',questions=len(rows),official_algorithm=True,
        score=100.0*sum(e['raw']['text']==targets[r['semantic_key']] for r,e in zip(rows,entries))/len(rows),
        scorer=original.c.desc(original.c.ROOT/'collaboration/session6-official-eval-v2-20261024/reference_sources/GQA_c121_recommended_eval.py'))
