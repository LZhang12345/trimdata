"""Full-pool collection, pinned operators, and independently gated selection output."""
import os
from pathlib import Path
import tempfile
import numpy as np
from experiments.root_native_campaign_20261023 import common as c
from experiments.root_native_campaign_20261023.selection import validate_pool,budget_size
from . import pins,features,operators
from .store import load_complete


def save_array(path,value):
    """Never overwrite a partial/foreign array; arrays are immutable evidence."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.array-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:np.save(f,value,allow_pickle=False);f.flush();os.fsync(f.fileno())
        if path.exists():c.require(c.sha(path)==c.sha(temp),'Immutable native array conflict')
        else:os.link(temp,path)
        directory=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:os.unlink(temp)
    return c.desc(path)


def collect(pool_desc,method,receipt_descs,output,transfer=None):
    """Promote measured full scope, never engineering fixtures, to an operator input."""
    pool,_,ids=validate_pool(pool_desc);receipts=[c.checked(d) for d in receipt_descs]
    c.require(receipts and method in features.CONTRACTS,'Published measurement shards absent')
    total=receipts[0]['shard_count'];identity=receipts[0]['measurement_identity']
    c.require(len(receipts)==total and {r['shard_index'] for r in receipts}==set(range(total)), 'Missing/duplicate acquisition shards')
    databases=[]
    for r in sorted(receipts,key=lambda r:r['shard_index']):
        roster=[ids[i] for b in features.batches(ids,method,r['shard_index'],total) for i in b]
        c.require(r['status']=='COMPLETE_PUBLISHED_FEATURE_SHARD' and r['engineering_only'] is False
            and r['method']==method and r['measurement_identity']==identity
            and r['logical_pool']==features.logical_pool(pool) and r['measurement_contract']==features.CONTRACTS[method]
            and r['roster_sha256']==c.key(roster) and r['counts']['after']==r['counts']['required']==len(roster),
            'Feature receipt does not cover the full exact declared shard')
        pins.verify(r['sources']);c.verify_sources(r['native_sources'])
        if transfer is None:
            database=r['database'];c.require(r['pool']==pool_desc,'Cross-site feature reuse requires an explicit exact-byte transfer receipt')
        else:
            moved=c.checked(transfer)
            c.require(moved['status']=='PASS_EXACT_FEATURE_TRANSFER_AND_LOGICAL_POOL'
                and moved['source_measurement_identity']==identity and moved['destination_pool']==pool_desc
                and moved['logical_pool']==features.logical_pool(pool),'Feature transfer/pool boundary not independently qualified')
            pairs=[p for p in moved['databases'] if p['source']==r['database']]
            c.require(len(pairs)==1,'Missing/duplicate feature transfer mapping');database=pairs[0]['destination']
            c.require(all(database[k]==r['database'][k] for k in ('bytes','sha256')),'Feature transfer changed bytes')
        databases.append(database)
    load_complete(databases,method,identity,ids)
    return c.seal(Path(output)/'FULL_FEATURES.json',dict(status='PASS_COMPLETE_PUBLISHED_NATIVE_FEATURES',method=method,
        pool=pool_desc,logical_pool=features.logical_pool(pool),record_ids_sha256=c.key(ids),count=len(ids),
        measurement_identity=identity,measurement_contract=features.CONTRACTS[method],databases=databases,
        source_receipts=receipt_descs,transfer=transfer,operator_values_not_yet_computed=True))


def materialize_values(feature_desc,pool_desc,output,should_stop=lambda:False,operator_context=None):
    """Actual features feed the author's expensive full geometry, not small fixtures."""
    import torch
    pool,_,ids=validate_pool(pool_desc);receipt=c.checked(feature_desc);method=receipt['method']
    c.require(receipt['status']=='PASS_COMPLETE_PUBLISHED_NATIVE_FEATURES' and receipt['pool']==pool_desc
        and receipt['logical_pool']==features.logical_pool(pool) and receipt['count']==len(ids),'Incomplete or incompatible feature function')
    values=load_complete(receipt['databases'],method,receipt['measurement_identity'],ids)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    c.require(operator_context is not None,'Actual operator environment/hardware/thread context required')
    owner=dict(features=feature_desc,pool=pool_desc,sources=pins.sources(),method=method,operator_context=operator_context)
    c.seal(output/'OWNER.json',owner)
    vector=np.stack([values[i]['vector'] for i in ids])
    hidden_desc=save_array(output/'hidden.npy',vector)
    if method=='coincide':
        cluster_dir=output/'native-faiss'
        ready=output/'CLUSTER_COMPLETE.json'
        if ready.exists():
            completed=c.read(ready)
            c.require(completed['owner']==owner,'Native clustering owner differs')
            for d in completed['files']:c.require(c.desc(d['path'])==d,'Native clustering changed')
        else:
            c.require(not cluster_dir.exists(),'Unaccepted FAISS directory exists; preserve failed attempt, no silent reuse')
            operators.coincide_cluster(vector,cluster_dir)
            c.seal(ready,dict(owner=owner,files=[c.desc(p) for p in sorted(cluster_dir.iterdir()) if p.is_file()]))
        centroids=np.load(cluster_dir/'kmeans_centroids.npy',allow_pickle=False)
        transfer=operators.coincide_transfer(centroids)
        result=dict(status='COMPLETE_PUBLISHED_OPERATOR_INPUTS',owner=owner,method=method,
            hidden=hidden_desc,clusters=c.desc(cluster_dir/'nearest_cent.npy'),transfer=save_array(output/'transfer.npy',transfer),
            sources=pins.sources(),feature_desc=feature_desc)
    else:
        hidden=torch.from_numpy(vector.astype(np.float32,copy=False))
        info=torch.tensor([values[i]['info'] for i in ids],dtype=torch.float32)
        tau=torch.tensor([values[i]['tau'] for i in ids],dtype=torch.float32)
        blocks=operators.source_blocks(pool['source_indices']);parts=[];offset=0
        for index,block in enumerate(blocks):
            if not block:continue
            if should_stop():return dict(status='DURABLE_OPERATOR_CONTINUATION',owner=owner)
            receipt_path=output/f'WARD-{index:02d}.json'
            if receipt_path.exists():
                r=c.read(receipt_path);c.require(r['owner']==owner and r['positions']==block,'Saved Ward source scope differs')
                c.require(c.desc(r['labels']['path'])==r['labels'],'Saved Ward labels changed')
                labels=torch.from_numpy(np.load(r['labels']['path'],allow_pickle=False))
            else:
                labels=operators.datatailor_ward_block(hidden[block])
                d=save_array(output/f'ward-{index:02d}.npy',labels.numpy())
                c.seal(receipt_path,dict(owner=owner,positions=block,labels=d))
            parts.append(labels+offset);offset=int(parts[-1].max())
        labels=torch.cat(parts)
        already={}
        for path in sorted(output.glob('RELATION-*.json')):
            r=c.read(path);c.require(r['owner']==owner,'Saved relation function differs')
            for name in ('representativeness','uniqueness'):
                c.require(c.desc(r[name]['path'])==r[name],'Saved relation array changed')
            already[r['cluster']]=(torch.from_numpy(np.load(r['representativeness']['path'],allow_pickle=False)),
                torch.from_numpy(np.load(r['uniqueness']['path'],allow_pickle=False)))
        class Continuation(Exception):pass
        def accept(group,positions,represent,unique):
            c.seal(output/f'RELATION-{group:06d}.json',dict(owner=owner,cluster=group,positions=positions,
                representativeness=save_array(output/f'represent-{group:06d}.npy',represent.numpy()),
                uniqueness=save_array(output/f'unique-{group:06d}.npy',unique.numpy())))
            if should_stop():raise Continuation()
        try:unique,represent=operators.datatailor_relationships(hidden,info,labels,pool['source_indices'],accept,already)
        except Continuation:return dict(status='DURABLE_OPERATOR_CONTINUATION',owner=owner)
        result=dict(status='COMPLETE_PUBLISHED_OPERATOR_INPUTS',owner=owner,method=method,hidden=hidden_desc,
            informativeness=save_array(output/'informativeness.npy',info.numpy()),tau=save_array(output/'tau.npy',tau.numpy()),
            uniqueness=save_array(output/'uniqueness.npy',unique.numpy()),representativeness=save_array(output/'representativeness.npy',represent.numpy()),
            clusters=save_array(output/'clusters.npy',labels.numpy()),sources=pins.sources(),feature_desc=feature_desc)
    return c.seal(output/'OPERATOR_INPUTS.json',result)


def candidate(pool_desc,operator_desc,fraction,selection_seed,output):
    import torch
    c.require(selection_seed==1234,'Published frozen comparison selection seed must be1234')
    pool,records,ids=validate_pool(pool_desc);operator=c.checked(operator_desc)
    c.require(operator['status']=='COMPLETE_PUBLISHED_OPERATOR_INPUTS' and operator['owner']['pool']==pool_desc,'Actual complete operator inputs required')
    pins.verify(operator['sources']);method=operator['method'];output=Path(output)
    arrays={}
    for name in ('hidden','clusters','transfer','informativeness','tau','uniqueness','representativeness'):
        if name in operator:
            descriptor=operator[name];c.require(c.desc(descriptor['path'])==descriptor,'Operator values changed')
            arrays[name]=np.load(descriptor['path'],allow_pickle=False)
    if method=='coincide':
        result=operators.coincide_select(arrays['hidden'],arrays['clusters'],arrays['transfer'],fraction)
    else:
        result=operators.datatailor_select(records,pool['source_indices'],
            *[torch.from_numpy(arrays[n]) for n in ('informativeness','tau','uniqueness','representativeness')],fraction)
    positions=sorted(result['positions']);actual=len(positions);requested=budget_size(len(ids),fraction)
    c.require(0<actual<=requested and len(positions)==len(set(positions)),'Author native floor output violates count budget')
    selected_ids=[ids[i] for i in positions]
    rows=c.seal(output/'records.json',[records[i] for i in positions])
    result_desc=c.seal(output/'AUTHOR_OPERATOR_RESULT.json',result)
    return c.seal(output/'CANDIDATE_SELECTION.json',dict(schema='native-record-selection-v1',
        status='PENDING_INDEPENDENT_SELECTOR_IMPLEMENTATION_AND_OUTPUT',pool=pool_desc,method=method,
        fraction=str(fraction),selection_seed=selection_seed,count=actual,record_ids=selected_ids,records_json=rows,
        requested_K=requested,author_operator_result=result_desc,operator_inputs=operator_desc,operator_sources=pins.sources(),
        author_seed_semantics='COINCIDE Kmeans1234; DataTailor deterministic given native measurements',
        author_floor_not_topped_up=True,matched_actual_K_uniform_required=actual!=requested,
        budget=dict(unit='whole_conversation_count',value=actual,fraction=str(fraction),eligible_pool_count=len(ids))))


def finalize(candidate_desc,independent_review_desc,output):
    value=c.checked(candidate_desc);review=c.checked(independent_review_desc)
    c.require(value['status']=='PENDING_INDEPENDENT_SELECTOR_IMPLEMENTATION_AND_OUTPUT','Not a pending actual native output')
    pins.verify(value['operator_sources'])
    expected=dict(method=value['method'],pool=value['pool'],records_json=value['records_json'],
        selection_seed=value['selection_seed'],fraction=value['fraction'],requested_K=value['requested_K'],actual_K=value['count'])
    c.require(review['status']=='PASS_EXACT_SELECTOR_IMPLEMENTATION_AND_OUTPUT'
        and all(review.get(k)==v for k,v in expected.items()) and review['candidate']==candidate_desc
        and review['sources']==value['operator_sources'],'Exact independent operator/output review differs')
    value.update(status='PASS_SELECTION',method_review=independent_review_desc)
    return c.seal(Path(output)/'SELECTION.json',value)
