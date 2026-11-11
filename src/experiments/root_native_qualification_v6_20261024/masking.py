"""Independent full-prefix token-boundary oracle; never calls native masking.

The repaired operator sums tokenized structured rounds. This oracle instead
locates each assistant interval in the full serialized-prefix token stream and
checks prefix alignment against full input IDs. Body role/EOS strings are data.
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
    c.require(template.version.startswith('v1') and template.sep_style.name=='TWO','Only native v1 TWO template')
    c.require(tuple(template.roles)==('USER','ASSISTANT'),'Wrong native conversation roles')
    has_image=bool(row.get('image'));turns=[]
    for index,turn in enumerate(row['conversations']):
        c.require(turn['from']==('human' if index%2==0 else 'gpt'),'Original turns not alternating')
        value=turn['value']
        if has_image and '<image>' in value:value=('<image>\n'+value.replace('<image>','').strip()).strip()
        turns.append(value)
    c.require(len(turns)>=2 and len(turns)%2==0,'Complete original assistant turns required')
    tokenize=(lambda text:image_tokens(text,tokenizer)) if has_image else (lambda text:tokenizer(text).input_ids)
    prompt=template.system+template.sep;spans=[]
    for index,value in enumerate(turns):
        role=template.roles[index%2]
        if not value:
            prompt+=role+':'
            if index%2:spans.append(None)
            continue
        prompt+=role+': '
        if index%2:
            before=tokenize(prompt);start=len(before)-1
            prompt+=value+template.sep2
            after=tokenize(prompt)
            spans.append(dict(start=start,end=len(after),prefix_ids=before[:-1],complete_ids=after,
                assistant_turn=index,assistant_value=value))
        else:prompt+=value+template.sep
    original_ids=tokenize(prompt)
    ids=original_ids if has_image else original_ids[:tokenizer.model_max_length]
    labels=[-100]*len(ids);rounds=[]
    # A genuinely empty assistant is a common-pool input-policy issue, never
    # justification for manufacturing EOS targets or deleting the record.
    if all(span is not None for span in spans):
        for span in spans:
            c.require(original_ids[:span['start']]==span['prefix_ids'],'Assistant prefix tokenization is not aligned')
            c.require(original_ids[:span['end']]==span['complete_ids'],'Assistant suffix/EOS tokenization is not aligned')
            begin=min(len(ids),span['start']);end=min(len(ids),span['end'])
            labels[begin:end]=ids[begin:end]
            rounds.append(dict(assistant_turn=span['assistant_turn'],assistant_start=begin,assistant_end=end,
                boundary_method='full_serialized_prefix_token_alignment'))
    return dict(input_ids=ids,labels=labels,rounds=rounds,tokenization_mismatch=False,
        prompt_sha256=c.key(prompt),image_record=has_image,original_turn_count=len(turns),
        original_assistant_values=turns[1::2],oracle='Independent full-prefix token-alignment assistant spans; no rendered-body delimiter split')
