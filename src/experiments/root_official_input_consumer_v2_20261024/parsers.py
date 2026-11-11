"""Restricted input parsers. Skipped answers/assistant strings are never decoded."""
from collections import Counter,defaultdict
import base64
import hashlib
import io
from pathlib import PurePosixPath
import re
from experiments.root_official_protected_inputs_v1_20261024 import parsers as primitive
from .lexical import Reader

require=primitive.require
identity=primitive.identity
image_ref=primitive.image_ref
question=primitive.question
sha=lambda raw:hashlib.sha256(raw).hexdigest()


def scalar(reader):
    reader.ws();require(reader.peek() not in (123,91),'Input field must be scalar')
    return reader.string(True) if reader.peek()==34 else reader.atom(True)


def array(reader,callback):
    reader.expect('[');reader.ws()
    if reader.peek()==93:reader.take();return
    index=0
    while True:
        callback(index);index+=1;reader.ws();token=reader.take()
        if token==93:return
        require(token==44,'Malformed restricted input array')


def end(reader):reader.ws();require(reader.peek() is None,'Trailing restricted input bytes')


def author_jsonl(raw,options):
    require(options.get('qa_access')=='authorized','Explicit authorized QA access required')
    text_field=options.get('text_field','text');require(text_field in ('text','question'),'Unknown native input field')
    policy=options.get('row_identity','native');require(policy in ('native','source-row'),'Unknown QA key policy')
    result=[]
    for index,line in enumerate(raw.splitlines()):
        require(line.strip() and len(line)<=1<<20,'Blank/unbounded exact native row')
        reader=Reader(io.BytesIO(line),len(line));fields={};seen=set()
        def field(name):
            require(name not in seen,'Duplicate original native field');seen.add(name)
            if name in ('question_id','image',text_field):fields[name]=scalar(reader)
            else:reader.skip_value()
        reader.object(field);end(reader)
        require(set(fields)=={'question_id','image',text_field},'Missing native input identity')
        native=identity(fields['question_id']);key='source-row/'+str(index)+'/'+native if policy=='source-row' else native
        result.append(dict(id=key,native_question_id=native,original_row_index=index,
          image=image_ref(fields['image']),question=question(fields[text_field])))
    require(result,'Empty native input source');return result


def projection(raw,array_name,allowed):
    reader=Reader(io.BytesIO(raw),len(raw));result=[];found=[]
    def root(name):
        if name!=array_name:reader.skip_value();return
        found.append(name)
        def row(_):
            value={};seen=set()
            def field(key):
                require(key not in seen,'Duplicate original input field');seen.add(key)
                if key in allowed:value[key]=scalar(reader)
                else:reader.skip_value()
            reader.object(field);result.append(value)
        array(reader,row)
    reader.object(root);end(reader);require(found==[array_name] and result,'Original input array missing/empty')
    return result


def seed_image_rows(rows,original):
    require(all(r.get('data_type') in ('image','video') for r in original),'Original image/video membership required')
    image_original=[r for r in original if r['data_type']=='image'];remaining=Counter();questions=defaultdict(set)
    for row in image_original:
        native=identity(row['question_id']);image=identity(row['data_id']);q=question(row['question'])
        remaining[(native,image,sha(q.encode()))]+=1;questions[(native,image)].add(q)
    selected=[]
    for row in rows:
        path=PurePosixPath(row['image'])
        require(len(path.parts)==2 and path.parts[0] in ('SEED-Bench-image','SEED-Bench-video-image'),'Exact image/video prefix required')
        if path.parts[0]!='SEED-Bench-image':continue
        pair=(row['native_question_id'],path.name)
        matches=[q for q in questions.get(pair,()) if row['question'].startswith(q+'\n')]
        require(len(matches)==1,'Native prompt lacks exact original question prefix')
        key=(*pair,sha(matches[0].encode()));require(remaining[key]>0,'Original image multiplicity exceeded')
        remaining[key]-=1;selected.append(row)
    require(selected and len(selected)==len(image_original) and not any(remaining.values()),
      'Complete original image-question multiset differs')
    return selected


def prompt_processor(prompt):
    # Exact author c121 eval_textvqa join; do not fuzzy-match or rewrite prompts.
    if prompt.startswith('OCR tokens: '):
        match=re.search(r'Question: (.*?) Short answer:',prompt,re.DOTALL)
        require(match is not None,'Original OCR prompt pattern absent');value=match.group(1)
    elif 'Reference OCR token: ' in prompt and len(prompt.split('\n'))==3:
        value=prompt.split('\n')[1] if prompt.startswith('Reference OCR token:') else prompt.split('\n')[0]
    elif len(prompt.split('\n'))==2:value=prompt.split('\n')[0]
    else:raise ValueError('Original author OCR prompt differs')
    return value.lower()


def textvqa_rows(rows,original):
    wanted={};joins={}
    for row in original:
        key=identity(row['question_id']);image=identity(row['image_id']);q=question(row['question'])
        pair=(image,q.lower());require(row.get('set_name')=='val' and key not in wanted and pair not in joins,
          'Original val question membership/author join ambiguous')
        wanted[key]=image;joins[pair]=key
    matched=[]
    for row in rows:
        pair=(row['native_question_id'],prompt_processor(row['question']))
        require(pair in joins,'Native image-ID/prompt lacks original val counterpart')
        matched.append(joins[pair]);path=PurePosixPath(row['image'])
        require(path.name==pair[0]+'.jpg' and (len(path.parts)==1 or
          (len(path.parts)==2 and path.parts[0]=='train_images')),'Original TextVQA literal image differs')
    require(rows and Counter(matched)==Counter(wanted.keys()),'Complete original validation multiset differs')
    return rows


PERCEPTION=('existence','count','position','color','posters','celebrity','scene','landmark','artwork','OCR')
COGNITION=('commonsense_reasoning','numerical_calculation','text_translation','code_reasoning')


def mme_alias(name):
    parts=PurePosixPath(image_ref(name)).parts;positions=[i for i,v in enumerate(parts) if v in PERCEPTION+COGNITION]
    require(len(positions)==1,'Ambiguous MME category')
    tail=parts[positions[0]+1:]
    require(len(tail)==1 or (len(tail)==2 and tail[0]=='images'),'Unreviewed original MME image layout')
    return parts[positions[0]]+'/'+tail[-1]


def mme_rows(rows,report,categories):
    aliases={r['image']:mme_alias(r['image']) for r in rows}
    require(aliases==report['native_image_aliases'] and report['native_original_question_identity_via_exact_author_conversion'] is True,
      'Authenticated original native/category aliases differ')
    current=[{k:r[k] for k in ('id','native_question_id','original_row_index','image')} for r in rows]
    require(current==report['native_QA_identity_map'],'Original native row identity map differs')
    by_image=defaultdict(list)
    for row in rows:by_image[aliases[row['image']]].append(row)
    require(set(by_image)==set(report['original_question_hashes']) and all(len(v)==2 for v in by_image.values()),
      'Original paired MME image membership differs')
    for image,pair in by_image.items():
        expected=Counter(report['original_question_hashes'][image]);observed=[]
        for row in pair:
            prompt=row['question']
            if 'Answer the question using a single word or phrase.' in prompt:
                prompt=prompt.replace('Answer the question using a single word or phrase.','').strip()
            if 'Please answer yes or no.' not in prompt:
                candidate=prompt+' Please answer yes or no.'
                prompt=candidate if sha(candidate.encode()) in expected else prompt+'  Please answer yes or no.'
            observed.append(sha(prompt.encode()))
        require(Counter(observed)==expected,'Author-converted paired question identities differ')
    require(categories and len(categories)==len(set(categories)) and set(categories)<=set(PERCEPTION+COGNITION),
      'Explicit original category partition required')
    return [r for r in rows if aliases[r['image']].split('/')[0] in categories]


def scienceqa_image(raw,options):
    require(options.get('qa_access')=='authorized','ScienceQA question access must be authorized')
    # This exact reviewed projector skips every later conversation turn and all
    # unrelated fields. It returns the original first-human input, not GT.
    from experiments.root_scienceqa_input_qualification_v3_20261024.formats import native
    result=[]
    for row in native(raw):
        if 'image' not in row:continue
        result.append(dict(id=row['id'],image=image_ref(row['image']),
          question=question(row['human'].replace('<image>','').strip()),original_row_index=row['source_row']))
    return result


def decode_tsv_field(raw):return raw.decode('utf-8')


def tsv_projection(raw,allowed):
    """RFC-style quoted TSV bytes; unselected fields never become strings."""
    # Parse structural bytes without decoding a whole row/file (answer columns
    # may contain strings that are expressly outside the input boundary).
    if raw.startswith(b'\xef\xbb\xbf'):raw=raw[3:]
    rows=[];fields=[];value=bytearray();quoted=False;closed=False;start=True;column=0;names=None;keep=True;i=0
    def finish_field():
        nonlocal value,column,keep,start,closed
        fields.append(decode_tsv_field(bytes(value)) if keep else None)
        value=bytearray();column+=1;start=True;closed=False
        keep=names is None or (column<len(names) and names[column] in allowed)
    def finish_row():
        nonlocal fields,names,column,keep
        if names is None:
            names=fields
            require(names and len(names)==len(set(names)) and set(allowed)<=set(names),'Exact unique TSV headers required')
        else:
            require(len(fields)==len(names),'TSV column count differs')
            rows.append({name:fields[j] for j,name in enumerate(names) if name in allowed})
        fields=[];column=0;keep=names[0] in allowed
    while i<len(raw):
        b=raw[i];i+=1
        if quoted:
            if b==34:
                if i<len(raw) and raw[i]==34:
                    i+=1
                    if keep:value.append(34)
                else:quoted=False;closed=True
            elif keep:value.append(b)
        elif b==34 and start:
            quoted=True;start=False
        elif b in (9,10,13):
            if b in (10,13) and not fields and not value and start and column==0:
                if b==13 and i<len(raw) and raw[i]==10:i+=1
                continue # csv.DictReader ignores physically empty lines, not rows of empty fields
            finish_field()
            if b!=9:
                if b==13 and i<len(raw) and raw[i]==10:i+=1
                finish_row()
        else:
            require(not closed,'Unquoted suffix after closed TSV field')
            start=False
            if keep:value.append(b)
        require(len(value)<=32<<20,'Selected TSV input field exceeds bound')
    require(not quoted,'Unterminated quoted TSV field')
    if fields or value or not start:finish_field();finish_row()
    require(rows,'Empty original TSV');return rows


NA={'','nan','none','NaN','NA','N/A','n/a','NULL','null','#N/A','#NA','<NA>','-NaN','-nan','1.#IND','1.#QNAN','-1.#IND','-1.#QNAN'}
def none(value):return value is None or value in NA or value.lower() in ('nan','none')


def mmbench(raw,options):
    require(options.get('qa_access')=='authorized','MMBench question access must be authorized')
    allowed=('index','image','question','hint','A','B','C','D');rows=tsv_projection(raw,allowed)
    mapping={identity(r['index']):r['image'] for r in rows}
    require(len(mapping)==len(rows),'Duplicate original TSV index')
    result=[]
    for row in rows:
        ident=identity(row['index']);encoded=row['image']
        if 0<len(encoded)<=64:
            require(encoded in mapping and len(mapping[encoded])>64,'Missing/chained/cyclic original image pointer')
            encoded=mapping[encoded]
        require(len(encoded)>64,'Original inline image absent')
        payload=base64.b64decode(encoded,validate=True);require(payload,'Empty original inline image')
        prompt=question(row['question'])
        if not none(row['hint']):prompt=row['hint']+'\n'+prompt
        count=0
        for option in 'ABCD':
            if none(row[option]):break
            count+=1;prompt+='\n'+option+'. '+row[option]
        require(1<=count<=4,'Exact c121 nonempty option support required')
        result.append(dict(id=ident,image='tsv-index/'+ident,question=prompt,
          inline_encoded_sha256=sha(payload),inline_encoded_bytes=len(payload)))
    return result


def parse(raw,kind,options):
    if kind=='c121-author-jsonl-v1':return author_jsonl(raw,dict(options,row_identity='native'))
    if kind=='c121-author-source-row-jsonl-v2':return author_jsonl(raw,dict(options,row_identity='source-row'))
    if kind=='c121-scienceqa-image-v2':return scienceqa_image(raw,options)
    if kind=='c121-mmbench-tsv-one-hop-v2':return mmbench(raw,options)
    if kind=='reserved-id-image-projection-v1':return primitive.reserved_id_projection(raw,options)
    if kind=='nocaps-validation-images-v1':
        require(options.get('qa_access')=='image_only','NoCaps remains image-only')
        return [dict(id=identity(r['id']),image=identity(r['id'])) for r in projection(raw,'images',{'id'})]
    raise ValueError('Unreviewed safe input parser')
