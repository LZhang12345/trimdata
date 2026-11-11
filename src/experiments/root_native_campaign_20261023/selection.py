"""Four transparent candidate variants and uniform, using real TRAIN-only inputs."""
import argparse
from collections import defaultdict
from decimal import Decimal,ROUND_FLOOR,localcontext
import heapq
import math
from pathlib import Path
import sqlite3
from . import common as c
from experiments.root_native_selection_20261023.selection import budget_size,validate_pool as validate_broad_pool

METHODS=('uniform','source_proportional','source_sqrt','source_loss_coverage','source_visual_coverage','full')

def validate_pool(descriptor,allow_engineering=False):
    pool=c.checked(descriptor)
    if pool.get('schema')!='native-engineering-pool-v1':return validate_broad_pool(descriptor)
    c.require(allow_engineering and pool['status']=='PASS_ACTUAL_NATIVE_ENGINEERING_INPUTS' and pool['engineering_only'] is True,
        'Engineering sample cannot satisfy scientific broad-pool readiness')
    rows=c.checked(pool['records_json']);ids=pool['record_ids'];indices=pool['source_indices']
    c.require(len(rows)==len(ids)==len(set(ids))==len(indices)==len(set(indices))==pool['count']>=1024,'Engineering fixture needs1024distinct actual records for genuine global128 accumulation')
    c.require(ids==[f'{i:06d}:{r.get("id", "")}' for i,r in zip(indices,rows)],'Engineering original record IDs differ')
    c.checked(pool['image_manifest']);c.checked(pool['protected_exclusions'])
    c.require(all(len(r['conversations'])>=2 for r in rows),'Whole native engineering conversations required')
    return pool,rows,ids

def groups_for(descriptor,pool_desc,ids):
    group=c.checked(descriptor)
    c.require(group['status']=='PASS_FROZEN_TRAIN_SOURCE_LABELS' and group['pool']==pool_desc,'Exact frozen TRAIN source labels required')
    c.require(set(group['groups'])==set(ids),'Source labels must cover every exact eligible ID')
    c.require(all(isinstance(g,str) and g for g in group['groups'].values()),'Empty source label')
    return group['groups']

def allocation(sizes,count,sqrt=False):
    """Capped Hamilton allocation; never oversubscribes a rare source."""
    c.require(0<count<=sum(sizes.values()) and all(n>0 for n in sizes.values()),'Invalid allocation')
    with localcontext() as context:
        context.prec=40
        weights={g:Decimal(n).sqrt() if sqrt else Decimal(n) for g,n in sizes.items()}
        result={g:0 for g in sizes};active=set(sizes);remaining=count
        while active:
            total=sum(weights[g] for g in active)
            shares={g:Decimal(remaining)*weights[g]/total for g in active}
            capped=[g for g in active if shares[g]>=sizes[g]]
            if capped:
                for g in capped:
                    result[g]=sizes[g];remaining-=sizes[g];active.remove(g)
                continue
            floors={g:int(shares[g].to_integral_value(rounding=ROUND_FLOOR)) for g in active}
            for g in active:result[g]=floors[g]
            spare=remaining-sum(floors.values())
            for g in sorted(active,key=lambda g:(-(shares[g]-floors[g]),g))[:spare]:result[g]+=1
            break
    c.require(sum(result.values())==count and all(0<=result[g]<=sizes[g] for g in sizes),'Allocation budget/cap error')
    return result

def load_features(descriptor,pool_desc,assets_desc,ids):
    receipt=c.checked(descriptor)
    c.require(receipt['schema']=='native-stage1-record-features-v1' and receipt['status']=='PASS_COMPLETE_NATIVE_RECORD_FEATURES','Actual complete native features required')
    c.require(receipt['pool']==pool_desc and receipt['assets']==assets_desc and receipt['stage']=='stage1_before_visual_instruction_tuning','Features model/pool/stage differs')
    c.require(receipt['count']==len(ids) and receipt['record_ids_sha256']==c.key(ids),'Feature roster differs')
    values={};allowed=set(ids)
    for db_desc in receipt['databases']:
        c.require(c.desc(db_desc['path'])==db_desc,'Feature database bytes changed')
        connection=sqlite3.connect('file:'+str(Path(db_desc['path']).absolute())+'?mode=ro',uri=True)
        try:
            for ident,loss,targets,bucket in connection.execute('SELECT record_id,loss,targets,vision_bucket FROM features'):
                c.require(ident not in values and ident in allowed,'Foreign/duplicate feature')
                c.require(loss is not None and math.isfinite(loss) and targets>0,'Invalid native answer loss/target measurement')
                c.require(bucket is None or 0<=bucket<4096,'Visual bucket out of range')
                values[ident]=dict(loss=loss,targets=targets,vision_bucket=bucket)
        finally:connection.close()
    c.require(set(values)==set(ids),'Incomplete features cannot select the full pool')
    return values

def round_robin(roster,count,bucket,seed):
    bins=defaultdict(list)
    for ident in roster:bins[bucket[ident]].append(ident)
    for key in bins:bins[key].sort(key=lambda ident:(c.key(['within',seed,ident]),ident))
    queue=[(0,c.key(['bucket',seed,str(key)]),str(key),key) for key in bins]
    heapq.heapify(queue);out=[]
    while len(out)<count:
        position,tie,label,key=heapq.heappop(queue)
        out.append(bins[key][position]);position+=1
        if position<len(bins[key]):heapq.heappush(queue,(position,tie,label,key))
    return out

def choose(ids,method,count,seed,groups=None,features=None):
    c.require(method in METHODS and 0<count<=len(ids)==len(set(ids)),'Invalid method/record budget')
    ranked=lambda roster:sorted(roster,key=lambda ident:(c.key(['native-campaign-v1',seed,ident]),ident))
    if method=='full':
        c.require(count==len(ids),'Full must select the complete pool');return ids[:]
    if method=='uniform':selected=set(ranked(ids)[:count])
    else:
        c.require(groups is not None and set(groups)==set(ids),'Complete source groups required')
        sources=defaultdict(list)
        for ident in ids:sources[groups[ident]].append(ident)
        quotas=allocation({g:len(v) for g,v in sources.items()},count,method=='source_sqrt')
        selected=set()
        for source,roster in sorted(sources.items()):
            k=quotas[source]
            if not k:continue
            if method in ('source_proportional','source_sqrt'):chosen=ranked(roster)[:k]
            elif method=='source_loss_coverage':
                c.require(features is not None and set(features)==set(ids),'Complete real loss features required')
                ordered=sorted(roster,key=lambda ident:(features[ident]['loss'],ident))
                # Cover the FULL loss distribution: no outcome-tuned easy/hard threshold.
                bins={ident:min(9,10*i//len(ordered)) for i,ident in enumerate(ordered)}
                chosen=round_robin(roster,k,bins,seed)
            else:
                c.require(features is not None and set(features)==set(ids),'Complete real visual features required')
                bins={ident:features[ident]['vision_bucket'] if features[ident]['vision_bucket'] is not None else 'text-only' for ident in roster}
                chosen=round_robin(roster,k,bins,seed)
            selected.update(chosen)
    c.require(len(selected)==count,'Exact selected-data count lost')
    return [ident for ident in ids if ident in selected]

def validate_selection(descriptor,pool_desc,expected_method=None,expected_fraction=None,allow_engineering=False):
    selected=c.checked(descriptor);pool,rows,ids=validate_pool(pool_desc,allow_engineering)
    c.require(selected['schema']=='native-record-selection-v1' and selected['status']=='PASS_SELECTION' and selected['pool']==pool_desc,'Wrong exact selection')
    chosen=selected['record_ids'];original=dict(zip(ids,rows))
    requested=budget_size(len(ids),selected['fraction'])
    c.require(0<len(chosen)==len(set(chosen))==selected['count']<=requested,'Wrong selection count/duplicates')
    c.require(set(chosen)<=set(ids) and c.checked(selected['records_json'])==[original[i] for i in chosen],'Altered original conversation')
    c.require(selected['budget']==dict(unit='whole_conversation_count',value=len(chosen),fraction=selected['fraction'],eligible_pool_count=len(ids)),'Wrong budget unit/count')
    if expected_method is not None:c.require(expected_method==selected['method'],'Wrong selected method')
    if expected_fraction is not None:c.require(Decimal(str(expected_fraction))==Decimal(selected['fraction']),'Wrong selected percentage')
    if selected['method'] in METHODS:
        c.require(selected['count']==requested,'Own/control methods must spend exact declared record budget')
        groups=groups_for(selected['groups'],pool_desc,ids) if selected['method'] not in ('uniform','full') else None
        features=None
        if selected['method'] in ('source_loss_coverage','source_visual_coverage'):
            features=load_features(selected['features'],pool_desc,selected['assets'],ids)
        c.require(chosen==choose(ids,selected['method'],len(chosen),selected['selection_seed'],groups,features),'Subset does not match declared algorithm/seed')
    else:
        review=c.checked(selected['method_review'])
        c.require(review['status']=='PASS_EXACT_SELECTOR_IMPLEMENTATION_AND_OUTPUT' and review['pool']==pool_desc and review['records_json']==selected['records_json']
            and review['method']==selected['method'] and review['selection_seed']==selected['selection_seed']
            and review['fraction']==selected['fraction'] and review['requested_K']==requested and review['actual_K']==len(chosen),'Published selection needs actual reviewed operator/output')
    return selected

def build(pool_desc,method,fraction,seed,output,group_desc=None,feature_desc=None,assets_desc=None,allow_engineering=False):
    pool,rows,ids=validate_pool(pool_desc,allow_engineering)
    groups=groups_for(group_desc,pool_desc,ids) if method not in ('uniform','full') else None
    features=None
    if method in ('source_loss_coverage','source_visual_coverage'):
        features=load_features(feature_desc,pool_desc,assets_desc,ids)
    chosen=choose(ids,method,budget_size(len(ids),fraction),seed,groups,features)
    original=dict(zip(ids,rows));output=Path(output)
    records=c.seal(output/'records.json',[original[i] for i in chosen])
    return c.seal(output/'SELECTION.json',dict(schema='native-record-selection-v1',status='PASS_SELECTION',pool=pool_desc,
        method=method,fraction=str(fraction),selection_seed=seed,count=len(chosen),record_ids=chosen,records_json=records,
        groups=group_desc,features=feature_desc,assets=assets_desc,algorithm_sources=c.sources(),
        budget=dict(unit='whole_conversation_count',value=len(chosen),fraction=str(fraction),eligible_pool_count=len(ids))))
