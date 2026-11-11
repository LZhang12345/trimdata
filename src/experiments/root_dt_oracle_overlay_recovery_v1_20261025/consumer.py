"""Typed immutable custody, correct writer locks and atomic new-row evidence."""
from contextlib import contextmanager,ExitStack
import fcntl
import json
from pathlib import Path
import sqlite3
import time
import numpy as np
from experiments.root_native_campaign_20261023 import common as c
from experiments.copilot7_datatailor_composite_consumer_20261025 import consumer_v3 as previous
from . import hook

native=previous.engine.native
BaseStore=previous.engine.prior.CompatibleStore
SCHEMA='root-dt-independent-mask-row-composite-v1'

def identity_components(original):
    return dict(original_measurement=original,preprocess_contract=hook.CONTRACT,
        hook_sources=hook.sources(),consumer_source=c.sha(__file__),
        unchanged_numerical_source=c.sha(native.__file__),NN33_scope='Preserved original qualification, not an expanded NN fixture')

def identity(original):return c.key(identity_components(original))

def valid_event(event,ident,records_ref):
    c.require(event['contract']==hook.CONTRACT and event['hook_sources']==hook.sources()
        and event['record_id']==ident and event['records_json']==records_ref
        and type(event['eligible_position']) is int and type(event['source_index']) is int
        and ident.split(':',1)[0]==str(event['source_index']).zfill(6) and len(event['row_sha256'])==64,
        'Exact row/source/hook preprocessing custody differs')
    c.require(event['all_input_and_label_tensors_CPU'] is True and event['input_ids_identical'] is True and event['ordinary_masks_identical'] is True
        and event['native91_input_ids_equal'] is True and event['native91_labels_equal'] is True
        and event['positive_actual_expanded_target_guard_unchanged'] is True
        and event['numerical_feature_function_changed'] is False
        and event['raw_QA_token_or_label_arrays_exported'] is False
        and type(event['repaired_shifted_targets']) is int and event['repaired_shifted_targets']>0,
        'Native identity/ordinary-mask/target/feature guard differs')
    if event['labels_changed']:
        c.require(event['native_shifted_targets']==0 and event['independent_prefix_oracle_equal'] is True,
            'Every changed mask needs exact independent full-prefix proof before forward')
    else:c.require(event['original_labels_sha256']==event['labels_sha256'],'Ordinary labels changed')

@contextmanager
def retained_locks(entries):
    """Original writers use EXECUTION; composite writers use COMPOSITE_EXECUTION."""
    locks={}
    for entry in entries:
        c.require(entry['kind'] in ('original','composite_v3','oracle_overlay_v1'),'Unknown retained ledger kind')
        name='EXECUTION.lock' if entry['kind']=='original' else 'COMPOSITE_EXECUTION.lock'
        path=Path(entry['database']['path']).parent/name
        c.require(path.is_file() and not any(v.is_symlink() for v in (path,*path.parents)),
            'Exact existing source writer lock required')
        locks[str(path)]=path
    with ExitStack() as stack:
        for path in [locks[k] for k in sorted(locks)]:
            stream=stack.enter_context(path.open('rb'));fcntl.flock(stream,fcntl.LOCK_SH|fcntl.LOCK_NB)
        yield

def read_new(ref,expected,roster,records_ref):
    values=previous.rows(ref,expected,roster)
    path=Path(ref['path']);db=sqlite3.connect(path.absolute().as_uri()+'?mode=ro&immutable=1',uri=True)
    try:
        audits={}
        for ident,body,proof_key in db.execute('SELECT record_id,event,proof_key FROM preprocess_audit'):
            event=json.loads(body);valid_event(event,ident,records_ref)
            row=db.execute('SELECT event FROM pre_forward_proofs WHERE proof_key=?',(proof_key,)).fetchone()
            c.require(row is not None and json.loads(row[0])==event and ident in values and ident not in audits,
                'Accepted row lacks its durable pre-forward exact-row proof')
            audits[ident]=event
        c.require(set(audits)==set(values),'Missing/foreign atomic feature custody')
    finally:db.close()
    c.require(c.desc(path)==ref,'Immutable overlay changed while validating custody')
    return values

def read_retained(entry,original,roster,records_ref,permitted):
    if entry['kind']=='original':
        c.require(entry['identity']==original,'Original measurement was reidentified')
        return previous.rows(entry['database'],original,roster)
    if entry['kind']=='composite_v3':
        expected=previous.overlay_identity(original)
        c.require(entry['identity']==expected,'Prior composite measurement was reidentified')
        return previous.rows(entry['database'],expected,roster,overlay=True,permitted_repairs=permitted)
    c.require(entry['kind']=='oracle_overlay_v1' and entry['identity']==identity(original),
        'Wrong new overlay identity')
    return read_new(entry['database'],entry['identity'],roster,records_ref)

class Store(BaseStore):
    def __init__(self,path,measurement,roster,legacy,records_ref,attempt_id):
        super().__init__(path,measurement,roster,legacy)
        self.records_ref=records_ref;self.attempt_id=attempt_id;self.pending=None
        self.new.db.execute('CREATE TABLE IF NOT EXISTS pre_forward_proofs(proof_key TEXT PRIMARY KEY,event TEXT NOT NULL)')
        self.new.db.execute('CREATE TABLE IF NOT EXISTS preprocess_audit(record_id TEXT PRIMARY KEY,event TEXT NOT NULL,proof_key TEXT NOT NULL)')
        self.new.db.execute('CREATE TABLE IF NOT EXISTS tokenizer_preflight(record_id TEXT PRIMARY KEY,event TEXT NOT NULL,policy TEXT NOT NULL)')
        self.new.db.commit()
        for ident,blob,info,tau,length,nans,actual in self.new.db.execute('SELECT * FROM features'):
            vector=np.frombuffer(blob,dtype='<f4')
            c.require(ident in self.allowed and actual==measurement and vector.shape==(4096,)
                and np.isfinite(vector).all() and info is not None and np.isfinite(info)
                and tau is not None and np.isfinite(tau) and tau>0 and type(length) is int and length>0 and nans==0,
                'Existing accepted overlay vector/scalar state is corrupt')
        audited=set()
        for ident,body,proof_key in self.new.db.execute('SELECT record_id,event,proof_key FROM preprocess_audit'):
            event=json.loads(body);valid_event(event,ident,records_ref)
            proof=self.new.db.execute('SELECT event FROM pre_forward_proofs WHERE proof_key=?',(proof_key,)).fetchone()
            c.require(proof is not None and json.loads(proof[0])==event,'Current accepted state lost pre-forward proof')
            audited.add(ident)
        c.require(audited==self.new.existing,'Current accepted features/custody disagree')

    def observe(self,event):
        ident=event['record_id'];valid_event(event,ident,self.records_ref)
        c.require(self.pending is None and self.missing_batch([ident]),'Foreign/repeated preprocessing before forward')
        key=c.key([self.attempt_id,ident]);body=c.canonical(event).decode()
        prior=self.new.db.execute('SELECT event FROM pre_forward_proofs WHERE proof_key=?',(key,)).fetchone()
        c.require(prior is None or prior[0]==body,'Pre-forward proof cannot be rewritten')
        with self.new.db:
            if prior is None:self.new.db.execute('INSERT INTO pre_forward_proofs VALUES(?,?)',(key,body))
        self.pending=(ident,event,key)  # synchronous FULL transaction completed before Dataset returns.

    def commit(self,measurements):
        c.require(len(measurements)==1 and self.pending is not None and measurements[0]['record_id']==self.pending[0],
            'Feature commit does not match its actual pre-forward row')
        ident,event,key=self.pending
        try:
            self.new.db.execute('INSERT INTO preprocess_audit VALUES(?,?,?)',(ident,c.canonical(event).decode(),key))
            super().commit(measurements)  # Same SQLite transaction includes feature and accepted custody.
        except BaseException:self.new.db.rollback();raise
        self.pending=None

def preflight(store,config,pool,records,ids,roster,module,attempt,should_stop):
    """Missing rows only, no parameters/models/weights/images; actual online gate remains.

    Durable cached token checks may avoid repeating CPU work after interruption.
    They never replace the authenticated Dataset gate before an actual forward.
    """
    from .pre_model_guard import pre_model_tokenizer_guard
    import copy
    assets=c.checked(config['feature_assets'])
    context_report={}
    try:
        with pre_model_tokenizer_guard(context_report):
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(assets['assets']['vicuna']['snapshot'],use_fast=False,
                model_max_length=2048,padding_side='right',local_files_only=True)
            tokenizer.pad_token=tokenizer.unk_token;policy=hook.tokenizer_policy(tokenizer)
            previous_template=module.conversation_lib.default_conversation
            module.conversation_lib.default_conversation=module.conversation_lib.conv_templates['v1']
            positions={ident:i for i,ident in enumerate(ids)};checked=0;reused=0;changed=0
            try:
                for ident in roster:
                    if ident in store.existing:continue
                    if should_stop():return None
                    i=positions[ident];raw=records[i]
                    context=dict(record_id=ident,eligible_position=i,source_index=pool['source_indices'][i],
                        row_sha256=hook.authority.row_hash(raw),records_json=pool['records_json'])
                    cached=store.new.db.execute('SELECT event,policy FROM tokenizer_preflight WHERE record_id=?',(ident,)).fetchone()
                    if cached is not None:
                        event=json.loads(cached[0]);valid_event(event,ident,pool['records_json'])
                        c.require(json.loads(cached[1])==policy and all(event[k]==v for k,v in context.items())
                            and type(event.get('derived_positive_expanded_targets')) is int and event['derived_positive_expanded_targets']>0,'Foreign pre-model tokenizer cache')
                        reused+=1
                    else:
                        typed=copy.deepcopy([raw['conversations']]);has_image=bool(raw.get('image'))
                        if has_image:typed=module.preprocess_multimodal(typed,type('Args',(),dict(is_multimodal=True,mm_use_im_start_end=False))())
                        try:result,event=hook.compare(module,module.preprocess_v1,raw,context,typed,tokenizer,has_image)
                        except BaseException as error:
                            c.seal(attempt/('PREFLIGHT-REFUSED-'+c.key(ident)+'.json'),dict(**context,
                                typed_sources_sha256=c.key(typed),error_class=type(error).__name__,message_sha256=c.key(str(error)),
                                phase='BEFORE_MODEL_CONSTRUCTION',automatic_permit=False))
                            raise
                        labels=[]
                        for token,target in zip(result['input_ids'][0].tolist(),result['labels'][0].tolist()):
                            labels.extend([-100]*576 if token==-200 else [target])
                            if len(labels)>=2048:break
                        expanded_targets=sum(v!=-100 for v in labels[:2048][1:])
                        if expanded_targets<=0:
                            c.seal(attempt/('PREFLIGHT-ZERO-'+c.key(ident)+'.json'),dict(**event,
                                derived_expanded_targets=expanded_targets,actual_multimodal_model_executed=False,
                                population_or_guard_change_authorized=False))
                            raise ValueError('Missing row has no targets after derived576patch/truncation; no policy relaxation')
                        event=dict(event,derived_positive_expanded_targets=expanded_targets)
                        with store.new.db:store.new.db.execute('INSERT INTO tokenizer_preflight VALUES(?,?,?)',(ident,c.canonical(event).decode(),c.canonical(policy).decode()))
                    checked+=1;changed+=int(event['labels_changed'])
            finally:module.conversation_lib.default_conversation=previous_template
    finally:c.seal(attempt/'PRE_MODEL_TOKENIZER_CONTEXT.json',context_report)
    c.seal(attempt/'MISSING_TOKENIZER_PREFLIGHT.json',dict(count=checked,cached_reused=reused,changed_rows=changed,
        policy=policy,actual_model_or_forward=False,no_missing_rows_dropped=True,independent_mask_checks_not_expandedNN_fixture=True))
    return policy

def acquire(config,pool,records,ids,entries,parity_ref,parity_review,module,output,attempt,should_stop,*,bound_configuration):
    started=time.monotonic()
    c.require(config['role']=='features' and config['method']=='datatailor' and config['shard_count']==16
        and len(records)==len(ids)==pool['count']==660763 and ids==pool['record_ids'],'Exact complete eligible native pool')
    parity=previous.engine.reviewed_parity(parity_ref,parity_review)
    qualified=c.checked(parity['configuration'])
    c.require(qualified['inputs']['assets']==config['feature_assets'] and qualified['inputs']['environment']==config['environment']
        and qualified['native_sources']==config['native_sources']
        and all(qualified['sources'].get(k)==v for k,v in config['sources'].items()),'Preserved original NN33 numerical function differs')
    roster=native.roster(config,records,ids);original=native.identity(config,pool)
    output=Path(output);attempt=Path(attempt)
    c.require(output.resolve() in attempt.resolve().parents,'Owned distinct attempt required')
    for entry in entries:
        path=Path(entry['database']['path']).parent
        c.require(output.resolve()!=path.resolve() and output.resolve() not in path.resolve().parents
            and path.resolve() not in output.resolve().parents,'New output must not alias/nest retained stores')
    output.mkdir(parents=True,exist_ok=True)
    with previous.engine.exclusive(output),retained_locks(entries):
        attempt.mkdir(parents=True,exist_ok=False)
        retained={}
        for entry in entries:
            part=read_retained(entry,original,roster,pool['records_json'],parity['permitted_changed_typed_sources'])
            c.require(not retained.keys() & part.keys(),'Retained ledgers overlap; no relabel/deduplication')
            retained.update(part)
        store=Store(output/'features.sqlite3',identity(original),roster,retained,pool['records_json'],attempt.name)
        before=len(store.existing)
        try:
            def refused(event):c.seal(attempt/('REFUSED-'+c.key(event['record_id'])+'.json'),event)
            policy=preflight(store,config,pool,records,ids,roster,module,attempt,should_stop) if before<len(roster) else None
            with hook.installed(module,records,ids,pool,store.observe,refused,policy):
                if before<len(roster) and policy is not None and not should_stop():
                    native.compute(dict(config,output=str(output)),pool,records,ids,store,should_stop,False,attempt)
            after=len(store.existing)
        finally:store.close()
        for entry in entries:c.require(c.desc(entry['database']['path'])==entry['database'],'Retained paid bytes changed')
        if after==len(roster):
            actual_new=read_new(c.desc(output/'features.sqlite3'),identity(original),roster,pool['records_json'])
            c.require(not retained.keys() & actual_new.keys() and set(retained)|set(actual_new)==set(roster),
                'Complete physical feature/audit union differs from exact shard')
        return c.seal(attempt/'COMPOSITE_PROGRESS.json',dict(schema=SCHEMA,
            status='COMPLETE_COMPATIBLE_COMPOSITE_FEATURE_SHARD' if after==len(roster) else 'MISSING_ONLY_COMPATIBLE_FEATURE_PROGRESS',
            pool=config['pool'],records_json=pool['records_json'],shard_index=config['shard_index'],shard_count=16,
            original_measurement=original,overlay_measurement=identity(original),overlay_identity_components=identity_components(original),
            retained_databases=entries,overlay_database=c.desc(output/'features.sqlite3'),bound_configuration=bound_configuration,
            before=before,after=after,required=len(roster),new_forwards=after-before,retained_records=len(retained),
            roster_sha256=c.key(roster),actual_parity=parity_ref,independent_review=parity_review,
            all_changed_rows_native91_and_independent_prefix_before_forward=True,expanded_NN_fixture_qualified=False,
            no_paid_copy_relabel_reacquisition=True,fullpool_student_or_scientific_PASS=False,elapsed_seconds=time.monotonic()-started))

def load_complete(ref,roster,expected_original):
    manifest=c.checked(ref)
    c.require(manifest['schema']==SCHEMA and manifest['status']=='COMPLETE_COMPATIBLE_COMPOSITE_FEATURE_SHARD'
        and manifest['original_measurement']==expected_original and manifest['overlay_measurement']==identity(expected_original)
        and manifest['overlay_identity_components']==identity_components(expected_original)
        and manifest['roster_sha256']==c.key(roster) and manifest['required']==len(roster),'Wrong or partial composite')
    parity=previous.engine.reviewed_parity(manifest['actual_parity'],manifest['independent_review']);values={}
    current=dict(kind='oracle_overlay_v1',database=manifest['overlay_database'],identity=manifest['overlay_measurement'])
    with retained_locks(manifest['retained_databases']+[current]):
        for entry in manifest['retained_databases']:
            part=read_retained(entry,expected_original,roster,manifest['records_json'],parity['permitted_changed_typed_sources'])
            c.require(not values.keys() & part.keys(),'Duplicate retained feature');values.update(part)
        part=read_new(manifest['overlay_database'],manifest['overlay_measurement'],roster,manifest['records_json'])
        c.require(not values.keys() & part.keys(),'Repeated paid/new feature');values.update(part)
        c.require(set(values)==set(roster),'Incomplete true composite coverage')
    return values

def load_full_pool(entries,ids,pool_ref,expected_original,parity_ref,parity_review):
    """Drop-in values API for an owned S7 selector after all16 genuine completions.

    Each shard declares original immutable DB, prior composite_v3 completion or
    this new completion. No old writer/controller/manifest is rewritten.
    """
    c.require(len(entries)==16 and len(ids)==len(set(ids))==660763,'All16 full eligible shards required')
    permitted=previous.engine.reviewed_parity(parity_ref,parity_review)['permitted_changed_typed_sources']
    seen=set();values={}
    for entry in entries:
        shard=entry['shard_index'];c.require(type(shard) is int and 0<=shard<16 and shard not in seen,'Foreign/duplicate global shard')
        roster=native.roster(dict(role='features',shard_count=16,shard_index=shard),[],ids)
        if entry['kind']=='original':
            with retained_locks([entry]):part=read_retained(entry,expected_original,roster,None,permitted)
        elif entry['kind']=='composite_v3':
            manifest=c.checked(entry['manifest']);c.require(manifest['pool']==pool_ref and manifest['shard_index']==shard,'Prior composite scope differs')
            # Old reader remains unchanged; acquire its actual old writer locks.
            locks=[dict(kind='original',database=ref) for ref in manifest['legacy_databases']]
            locks.append(dict(kind='composite_v3',database=manifest['overlay_database']))
            with retained_locks(locks):part=previous.load_complete(entry['manifest'],roster,expected_original)
        else:
            c.require(entry['kind']=='oracle_overlay_v1','Unsupported complete shard kind')
            manifest=c.checked(entry['manifest']);c.require(manifest['pool']==pool_ref and manifest['shard_index']==shard,'New composite scope differs')
            part=load_complete(entry['manifest'],roster,expected_original)
        c.require(set(part)==set(roster) and not values.keys() & part.keys(),'Partial or repeated fullpool shard')
        values.update(part);seen.add(shard)
    c.require(seen==set(range(16)) and set(values)==set(ids),'Incomplete fullpool union')
    return values
