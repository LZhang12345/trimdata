"""Typed-boundary repair of pinned c121 v1 masks; original prompt/IDs unchanged.

Rendered user/assistant content may itself contain a role delimiter. Boundaries
are obtained from original turn structure, never by splitting arbitrary content.
The author's token-length, legacy adjustment, truncation and mismatch rejection
are retained. No row is removed, text escaped, target invented or guard relaxed.
"""
from pathlib import Path
from . import common as c


def structured_rounds(source, template):
    conv=template.copy();roles={'human':conv.roles[0],'gpt':conv.roles[1]}
    c.require(conv.version.startswith('v1') and conv.sep_style.name=='TWO','Only exact native v1 TWO template')
    c.require(source and all(isinstance(t.get('value'),str) for t in source),'Original string conversations required')
    # Preserve the native author convention for a leading assistant message.
    first_skipped=roles[source[0]['from']]!=conv.roles[0]
    turns=source[1:] if first_skipped else source
    c.require(turns and len(turns)%2==0,'Incomplete conversation requires separate common-pool policy, not fabricated targets')
    conv.messages=[]
    for index,turn in enumerate(turns):
        role=roles[turn['from']]
        c.require(role==conv.roles[index%2],'Original author role alternation violated')
        conv.append_message(role,turn['value'])
    prompt=conv.get_prompt();rounds=[];assembled=''
    for index in range(0,len(turns),2):
        user,assistant=turns[index]['value'],turns[index+1]['value']
        prefix=conv.system+conv.sep if index==0 else ''
        user_piece=conv.roles[0]+(': '+user+conv.sep if user else ':')
        assistant_head=conv.roles[1]+(': ' if assistant else ':')
        instruction=prefix+user_piece+assistant_head
        round_text=instruction+assistant
        assembled+=round_text+(conv.sep2 if assistant else '')
        rounds.append(dict(round=round_text,instruction=instruction,assistant=assistant,
            original_user_turn=index+int(first_skipped),original_assistant_turn=index+1+int(first_skipped)))
    c.require(assembled==prompt,'Structured boundaries changed original author prompt bytes')
    return prompt,rounds


def preprocess_structured(module,sources,tokenizer,has_image=False):
    """Same native interface/math, replacing only content-ambiguous splitting."""
    template=module.conversation_lib.default_conversation
    bodies=[structured_rounds(source,template) for source in sources]
    conversations=[body[0] for body in bodies]
    # Exact original author input-tokenization branches.
    if has_image:
        input_ids=module.torch.stack([module.tokenizer_image_token(prompt,tokenizer,return_tensors='pt') for prompt in conversations],dim=0)
    else:
        input_ids=tokenizer(conversations,return_tensors='pt',padding='longest',
            max_length=tokenizer.model_max_length,truncation=True).input_ids
    targets=input_ids.clone()
    for (conversation,rounds),target in zip(bodies,targets):
        total_len=int(target.ne(tokenizer.pad_token_id).sum())
        cur_len=1;target[:cur_len]=module.IGNORE_INDEX
        for index,record in enumerate(rounds):
            # An actually empty assistant must never be repaired into EOS-only
            # supervision. Preserve strict rejection for a common-policy audit.
            if not record['assistant']:
                target[:]=module.IGNORE_INDEX;cur_len=None;break
            if has_image:
                round_len=len(module.tokenizer_image_token(record['round'],tokenizer))
                instruction_len=len(module.tokenizer_image_token(record['instruction'],tokenizer))-2
            else:
                round_len=len(tokenizer(record['round']).input_ids)
                instruction_len=len(tokenizer(record['instruction']).input_ids)-2
            if index!=0 and not tokenizer.legacy and module.IS_TOKENIZER_GREATER_THAN_0_14:
                round_len-=1;instruction_len-=1
            target[cur_len:cur_len+instruction_len]=module.IGNORE_INDEX
            cur_len+=round_len
        if cur_len is None:continue
        target[cur_len:]=module.IGNORE_INDEX
        if cur_len<tokenizer.model_max_length and cur_len!=total_len:
            target[:]=module.IGNORE_INDEX
            print(f'WARNING: structured tokenization mismatch: {cur_len} vs. {total_len}. (ignored)')
    return dict(input_ids=input_ids,labels=targets)


def install(module):
    """Install only in a NEW reviewed runtime; no author source file is edited."""
    expected=c.ROOT/c.NATIVE/'llava/train/train.py'
    c.require(Path(module.__file__).resolve()==expected.resolve(),'Wrong pinned author preprocessing module')
    original=module.preprocess_v1
    c.require(Path(original.__code__.co_filename).resolve()==expected.resolve(),'Existing foreign preprocessing hook')
    def repaired(sources,tokenizer,has_image=False):
        return preprocess_structured(module,sources,tokenizer,has_image)
    module.preprocess_v1=repaired
    return dict(status='INSTALLED_STRUCTURED_NATIVE_V1_BOUNDARIES_NOT_RUNTIME_QUALIFIED',
        original_native_source=c.desc(expected),repair_source=c.desc(__file__),
        prompt_bytes_unchanged=True,original_token_length_legacy_truncation_math_preserved=True,
        no_records_removed=True,no_targets_fabricated=True,allmasked_guards_remain=True)
