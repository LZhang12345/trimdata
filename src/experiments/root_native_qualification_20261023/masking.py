"""Independent reconstruction of declared author-v1 assistant target intervals.

This uses the actual pinned tokenizer but never calls native preprocess/masking.
It preserves the author's documented token-boundary/legacy convention rather
than silently replacing its objective with a convenient new answer-only loss.
"""
from . import common as c

def image_tokens(text,tokenizer):
    chunks=[tokenizer(chunk).input_ids for chunk in text.split('<image>')]
    offset=int(bool(chunks and chunks[0] and chunks[0][0]==tokenizer.bos_token_id))
    output=[tokenizer.bos_token_id] if offset else []
    for index,chunk in enumerate(chunks):
        if index:output.append(-200)
        output.extend(chunk[offset:])
    return output

def author_v1(row,tokenizer,template):
    """Pure Python interval oracle; returns unpadded ids+labels and provenance."""
    c.require(template.version.startswith('v1') and template.sep_style.name=='TWO','Only declared native v1 template')
    c.require(template.roles==('USER','ASSISTANT') or template.roles==['USER','ASSISTANT'],'Wrong native conversation roles')
    has_image=bool(row.get('image'));turns=[]
    for index,turn in enumerate(row['conversations']):
        c.require(turn['from']==('human' if index%2==0 else 'gpt'),'Original turns not alternating')
        value=turn['value']
        if has_image and '<image>' in value:value=('<image>\n'+value.replace('<image>','').strip()).strip()
        turns.append(value)
    c.require(len(turns)>=2 and len(turns)%2==0,'Complete original assistant turns required')
    prompt=template.system+template.sep
    for index,value in enumerate(turns):
        prompt+=template.roles[index%2]+(': '+value+([template.sep,template.sep2][index%2]) if value else ':')
    tokenize=(lambda text:image_tokens(text,tokenizer)) if has_image else (lambda text:tokenizer(text).input_ids)
    original_ids=tokenize(prompt)
    ids=original_ids if has_image else original_ids[:tokenizer.model_max_length]
    labels=ids[:];cursor=1;labels[:1]=[-100]
    separator=template.sep+template.roles[1]+': ';rounds=[]
    for index,part in enumerate(prompt.split(template.sep2)):
        if not part:break
        pieces=part.split(separator)
        if len(pieces)!=2:break
        instruction=pieces[0]+separator
        round_length=len(tokenize(part));instruction_length=len(tokenize(instruction))-2
        if index and not tokenizer.legacy:
            # Native pinned tokenizers0.15.1 follows the >0.14 branch.
            round_length-=1;instruction_length-=1
        start=cursor;end=min(len(labels),cursor+instruction_length)
        if start<len(labels):labels[start:end]=[-100]*(end-start)
        rounds.append(dict(round=index,instruction_start=start,instruction_end=end,round_length=round_length))
        cursor+=round_length
    labels[cursor:]=[-100]*max(0,len(labels)-cursor)
    total=sum(token!=tokenizer.pad_token_id for token in ids)
    mismatch=cursor<tokenizer.model_max_length and cursor!=total
    if mismatch:labels=[-100]*len(ids)
    return dict(input_ids=ids,labels=labels,rounds=rounds,tokenization_mismatch=mismatch,
        prompt_sha256=c.key(prompt),image_record=has_image,original_turn_count=len(turns),
        original_assistant_values=turns[1::2],oracle='Independent author-v1 interval reconstruction with native legacy-boundary convention')
