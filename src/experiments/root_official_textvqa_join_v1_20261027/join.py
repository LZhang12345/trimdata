"""Pinned author's image-ID/processed-question join; never rewrite native QA IDs."""
import ast,hashlib,json,re
from pathlib import Path,PurePosixPath

SCORER='collaboration/published-foundation-20261022/reference_sources/LLaVA/c121f0432da27facab705978f83c4ada465e46fd/llava/eval/eval_textvqa.py'
SCORER_SHA='e78fdd1ae8fdf94c46053c14f7d65f20707b6a0c324c632b841dde5ef04783a0'

def require(ok,message):
    if not ok:raise ValueError(message)

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()

def author_prompt_processor(source_root):
    path=Path(source_root)/SCORER;raw=path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==SCORER_SHA,'Exact pinned author TextVQA scorer changed')
    tree=ast.parse(raw);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='prompt_processor']
    require(len(nodes)==1,'Sole pinned author prompt processor required')
    namespace={'re':re};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['prompt_processor']

def join(rows,annotation,*,source_root,expected_count=5000):
    """Pure scorer-side targets + receipt; caller's native roster is never edited.

    expected_count is5000 in production. Small explicit values are for unit
    fixtures only and cannot establish actual full-bank qualification.
    """
    require(type(expected_count)is int and expected_count>0 and type(rows)is list and type(annotation)is dict
        and type(annotation.get('data'))is list and len(rows)==len(annotation['data'])==expected_count,'Exact full declared TextVQA QA/GT counts')
    process=author_prompt_processor(source_root)
    from experiments.session6_official_eval_v2_20261024.journal import semantic_key
    originals=digest(rows);gt_original=digest(annotation);official={};official_ids=set()
    for index,item in enumerate(annotation['data']):
        require(type(item)is dict and type(item.get('image_id'))is str and item['image_id']!=''
            and type(item.get('question'))is str and item['question']!='','Original official image/string question required')
        qid=item.get('question_id');require(type(qid)in(str,int),'Original typed official question ID required')
        typed=(type(qid).__name__,qid);require(typed not in official_ids,'Duplicate official typed QA ID');official_ids.add(typed)
        require(type(item.get('answers'))is list and len(item['answers'])==10 and all(type(a)is str for a in item['answers']),
            'Exactly ten unmodified official human answer strings required')
        key=(item['image_id'],item['question'].lower())
        require(key not in official,'Ambiguous duplicate official image/processed-question key');official[key]=(index,item)
    targets={};mapping=[];used=set();native_ids=[];keys=set()
    for row in rows:
        native=row['native_input'];nid=row['native_id'];key=row['semantic_key'];index=row['original_row_index']
        require(type(native)is dict and set(native)=={'question_id','image','text'}and type(nid)is str
            and type(native['question_id'])is type(nid)and native['question_id']==nid
            and type(native['text'])is str and type(native['image'])is str,'Exact original typed native QA/input required')
        require(row['bank']=='TextVQA'and key==semantic_key('TextVQA',row['source_sha256'],index,nid)
            and key not in keys,'Unique unchanged original semantic QA key required');keys.add(key)
        image=PurePosixPath(native['image'])
        require(len(image.parts)==1 and image.suffix=='.jpg'and image.stem==nid,'Native image basename must equal original typed image/question ID')
        try:question=process(native['text'])
        except (AssertionError,AttributeError,IndexError,TypeError)as error:raise ValueError('Native prompt rejected by exact author processor')from error
        composite=(nid,question)
        require(composite in official and composite not in used,'Missing or duplicate native author image/processed-question membership')
        used.add(composite);official_index,item=official[composite]
        require(type(item['image_id'])is type(nid)and item['image_id']==nid,'Exact typed image identity, never ID coercion')
        targets[key]=list(item['answers']);native_ids.append(nid)
        mapping.append(dict(semantic_key=key,original_row_index=index,native_id=nid,native_id_type=type(nid).__name__,
            native_input_sha256=digest(native),official_row_index=official_index,official_question_id=item['question_id'],
            official_question_id_type=type(item['question_id']).__name__,image_id=item['image_id'],
            processed_question_sha256=hashlib.sha256(question.encode()).hexdigest(),official_answers_sha256=digest(item['answers'])))
    require(used==set(official)and len(targets)==expected_count and digest(rows)==originals and digest(annotation)==gt_original,
        'Full bijective author-key coverage and unchanged native/GT inputs required')
    receipt=dict(schema='official-author-textvqa-image-question-bijection-v1',questions=expected_count,
        official_questions=len(official),unique_native_image_ids=len(set(native_ids)),repeated_native_image_ids=expected_count-len(set(native_ids)),
        unique_semantic_QAs=len(keys),unique_author_join_keys=len(used),all_native_and_official_rows_joined_once=True,
        mapping_sha256=digest(mapping),native_roster_sha256=originals,official_annotations_sha256=gt_original,
        target_answer_lists_sha256=digest(targets),prompt_processor_source=SCORER,prompt_processor_file_sha256=SCORER_SHA,
        native_IDs_prompts_semantic_keys_and_answers_unchanged=True,answer_targets_are_scorer_side_only=True,
        model_or_inference_or_score_performed=False)
    return targets,receipt,mapping
