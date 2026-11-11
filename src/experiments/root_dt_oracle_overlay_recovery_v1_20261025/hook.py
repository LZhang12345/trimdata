"""Reversible preprocessing gate; unchanged native numerical feature computation."""
from contextlib import contextmanager
import copy
import hashlib
from pathlib import Path
from experiments.root_native_campaign_20261023 import common as c
from experiments.root_native_input_authority_20261023 import pool_authority_v1 as authority
from experiments.root_native_campaign_v5_20261024 import structured_preprocess as v5
from experiments.root_native_campaign_v7_20261024 import structured_preprocess as native91
from experiments.root_native_qualification_v6_20261024.masking import author_v1 as oracle

CONTRACT='root-dt-exact-row-native91-independent-prefix-before-forward-v1'

def sources():
    return dict(hook=c.sha(__file__),structured_v5=c.sha(v5.__file__),
        structured_native91=c.sha(native91.__file__),
        independent_prefix=c.sha(__import__(oracle.__module__,fromlist=['x']).__file__))

def digest(value):
    import torch
    value=value.detach().cpu().contiguous()
    h=hashlib.sha256(c.canonical([str(value.dtype),list(value.shape)]))
    h.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()).cast('B'))
    return h.hexdigest()

def tokenizer_policy(tokenizer):
    return dict(class_name=type(tokenizer).__module__+'.'+type(tokenizer).__name__,
        vocabulary_sha256=c.key(tokenizer.get_vocab()),length=len(tokenizer),
        legacy=getattr(tokenizer,'legacy',None),model_max_length=tokenizer.model_max_length,
        padding_side=tokenizer.padding_side,pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,eos_token_id=tokenizer.eos_token_id)

def compare(module,original,raw_row,row_context,typed_sources,tokenizer,has_image):
    """Called while an authenticated Dataset row is active, before it returns."""
    import torch
    c.require(has_image==bool(raw_row.get('image')) and len(typed_sources)==1,'Exact native B1 row modality required')
    old=original(copy.deepcopy(typed_sources),tokenizer,has_image=has_image)
    new=v5.preprocess_structured(module,copy.deepcopy(typed_sources),tokenizer,has_image=has_image)
    current=native91.preprocess_structured(module,copy.deepcopy(typed_sources),tokenizer,has_image=has_image)
    c.require(set(old)==set(new)==set(current)=={'input_ids','labels'},'Native interface changed')
    c.require(all(isinstance(value,torch.Tensor) and value.device.type=='cpu' for result in (old,new,current) for value in result.values()),
        'Preprocessing input IDs and labels must remain CPU tensors')
    c.require(old['input_ids'].shape==new['input_ids'].shape==old['labels'].shape==new['labels'].shape
        and old['input_ids'].dtype==new['input_ids'].dtype and old['labels'].dtype==new['labels'].dtype
        and new['input_ids'].shape[0]==1 and torch.equal(old['input_ids'],new['input_ids']),
        'Original input IDs, dtype, B1 or shape changed')
    c.require(torch.equal(current['input_ids'],new['input_ids']) and torch.equal(current['labels'],new['labels']),
        'Frozen Native91 and V5 structured paths disagree')
    native_count=int((old['labels'][:,1:]!=-100).sum());new_count=int((new['labels'][:,1:]!=-100).sum())
    changed=not torch.equal(old['labels'],new['labels'])
    if native_count>0:c.require(not changed,'An ordinary valid native assistant mask changed')
    c.require(new_count>0,'No genuine assistant targets; preserve row, do not manufacture or exclude')
    proof=None
    if changed:
        c.require(native_count==0,'Only actual allmasked native rows may differ')
        proof=oracle(raw_row,tokenizer,module.conversation_lib.default_conversation.copy())
        c.require(new['input_ids'][0].tolist()==proof['input_ids'] and new['labels'][0].tolist()==proof['labels'],
            'Changed row fails independent full-prefix oracle before any forward')
    event=dict(contract=CONTRACT,**row_context,typed_sources_sha256=c.key(typed_sources),
        hook_sources=sources(),all_input_and_label_tensors_CPU=True,input_ids_identical=True,ordinary_masks_identical=True,
        native91_input_ids_equal=True,native91_labels_equal=True,
        labels_changed=changed,native_shifted_targets=native_count,repaired_shifted_targets=new_count,
        independent_prefix_oracle_equal=True if changed else None,
        proof_scope='Exact full IDs/labels before Dataset return; no model or expandedNN fixture',
        input_ids_sha256=digest(new['input_ids']),original_labels_sha256=digest(old['labels']),
        labels_sha256=digest(new['labels']),full_sequence_length=int(new['input_ids'].shape[1]),
        raw_QA_token_or_label_arrays_exported=False,numerical_feature_function_changed=False,
        positive_actual_expanded_target_guard_unchanged=True)
    return new,event

@contextmanager
def installed(module,records,ids,pool,observe,refused=None,expected_tokenizer=None):
    """Attach row identity before the exact lazy Dataset implementation executes."""
    expected=c.ROOT/c.NATIVE/'llava/train/train.py'
    c.require(Path(module.__file__).resolve()==expected.resolve(),'Wrong native module')
    original=module.preprocess_v1;dataset=module.LazySupervisedDataset;getitem=dataset.__getitem__
    c.require(Path(original.__code__.co_filename).resolve()==expected.resolve()
        and Path(getitem.__code__.co_filename).resolve()==expected.resolve(),
        'Foreign or already hooked native preprocessing/Dataset')
    c.require(len(records)==len(ids)==len(pool['source_indices']) and ids==pool['record_ids'],'Exact authority row roster')
    active=[None];checked_tokenizers=set()
    def row_item(self,index):
        c.require(type(index) is int and 0<=index<len(ids) and active[0] is None,'Only one exact native row may be active')
        raw=self.list_data_dict[index];expected_row=records[index]
        row_hash=authority.row_hash(expected_row)
        c.require(authority.row_hash(raw)==row_hash and authority.record_id(pool['source_indices'][index],raw)==ids[index],
            'Native Dataset row differs from immutable eligible authority')
        active[0]=(raw,dict(record_id=ids[index],eligible_position=index,source_index=pool['source_indices'][index],
            row_sha256=row_hash,records_json=pool['records_json']),self.data_args)
        try:return getitem(self,index)
        finally:active[0]=None
    def repaired(typed_sources,tokenizer,has_image=False):
        c.require(active[0] is not None,'Preprocessing outside an authenticated authority row')
        if expected_tokenizer is not None and id(tokenizer) not in checked_tokenizers:
            c.require(tokenizer_policy(tokenizer)==expected_tokenizer,'Constructor tokenizer differs from pre-model proof')
            checked_tokenizers.add(id(tokenizer))
        raw,context,args=active[0]
        expected_sources=copy.deepcopy([raw['conversations']])
        if has_image:expected_sources=module.preprocess_multimodal(expected_sources,args)
        c.require(typed_sources==expected_sources,'Typed conversations differ from the original active row')
        try:result,event=compare(module,original,raw,context,typed_sources,tokenizer,has_image)
        except BaseException as error:
            if refused is not None:refused(dict(**context,typed_sources_sha256=c.key(typed_sources),
                error_class=type(error).__name__,message_sha256=hashlib.sha256(str(error).encode()).hexdigest(),
                phase='BEFORE_DATASET_RETURN_OR_ANY_FORWARD',automatic_permit=False))
            raise
        observe(event)  # Durable new-store proof precedes Dataset return/vision/LM forward.
        return result
    module.preprocess_v1=repaired;dataset.__getitem__=row_item
    try:yield dict(contract=CONTRACT,sources=sources(),native_preprocessing=c.desc(expected))
    finally:module.preprocess_v1=original;dataset.__getitem__=getitem
