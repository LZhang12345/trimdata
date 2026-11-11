"""Pure HOME metadata constructor for six terminal scopes; creates no approval/claim."""
import argparse
import copy
import json
from pathlib import Path
from . import worker as w
from .sources import sources
c=w.c
ROOT=Path(__file__).resolve().parents[2]
TURN=ROOT/'collaboration/root-next-wave-and-capacity-20261025'
HOME=Path('/home/luyangz/data_selection_multimodal')
INVENTORY=TURN/'actual-owned-inventory-and-scienceqa-v1/collaboration/root-next-wave-and-capacity-20261025/actual-owned-inventory-v1/INVENTORY.json'
HASHES=TURN/'actual-seven-db-sha-and-start-evaluations-v1/collaboration/root-next-wave-and-capacity-20261025/ACTUAL_SEVEN_DB_SHA_AND_START_EVALUATIONS_V1.json'
ARRAY=TURN/'ACTUAL_ARRAY_MAPPING_PARENT_AND_LIMITS_V1_RESULT.json'
CENSUS=TURN/'FRESH_BABEL_CENSUS_V1_RESULT.json'
ALIASES={0:['10919220_0','10919240','10922849'],2:['10919402_1','10919404'],11:['10922051_10','10922056'],
    13:['10922051_12','10922058'],14:['10922051_13','10922059'],15:['10922051_14','10922051']}
REQUIRED={0:41538,2:41289,11:41779,13:41477,14:41074,15:41452}

def exported(local,remote):return dict(c.desc(local),path=str(remote))

def build(output):
    output=Path(output);inventory=c.read(INVENTORY);hashes=c.read(HASHES)
    mapped=json.loads(c.read(ARRAY)['stdout']);current=json.loads(c.read(CENSUS)['stdout'])
    c.require(set(inventory['failed_global_scopes'])=={str(v) for v in w.CANDIDATES},'Actual six-shard inventory differs')
    c.require(len(hashes['databases'])==7 and all(v['WAL_bytes']==0 and v['stat_unchanged'] is True
        and v['shared_lock']['shared_lock_acquired'] is True for v in hashes['databases']),'Actual shared-lock seven DB hash evidence required')
    by_path={v['descriptor']['path']:v for v in hashes['databases']};c.require(len(by_path)==7,'Duplicate DB hash evidence')
    review_path=ROOT/'collaboration/copilot7-datatailor-mask-repair-20261024/qualification-v3/independent-actual-review-v1/INDEPENDENT_ACTUAL_REVIEW.json'
    parity_review=exported(review_path,HOME/review_path.relative_to(ROOT));parity=c.read(review_path)['result']
    c.require(parity['sha256']==w.base.PARITY_SHA and parity_review['sha256']==w.base.PARITY_REVIEW_SHA,'Exact actual NN33 evidence differs')
    refs={};mappings={};counts={}
    for shard in w.CANDIDATES:
        old_path=ROOT/('collaboration/copilot7-fullpool-published-20261024/datatailor-preempt-v1/CONFIG.json' if shard==0 else
            'collaboration/root-s7-datatailor-general-takeover-v1-20261024/actual-Root/remote-registered-state-v1/CONFIG.json')
        old=c.read(old_path);old_remote=HOME/('collaboration/copilot7-fullpool-published-20261024/datatailor-preempt-v1/CONFIG.json' if shard==0 else
            'collaboration/root-s7-datatailor-general-takeover-v1-20261024/CONFIG.json')
        original=exported(old_path,old_remote);entries=[];paid=0
        for snapshot in inventory['failed_global_scopes'][str(shard)]:
            raw=snapshot['database'];actual=by_path[raw['path']]
            c.require(actual['global_index']==shard and actual['features']==snapshot['features']
                and actual['descriptor']['bytes']==raw['bytes'] and raw['wal_bytes']==0
                and actual['shared_lock']['path']==snapshot['lock']['path'],'Inventory/hash scope mismatch')
            kind='original' if Path(raw['path']).parent.parent.name=='datatailor' else 'composite_v3'
            c.require(Path(actual['shared_lock']['path']).name==('EXECUTION.lock' if kind=='original' else 'COMPOSITE_EXECUTION.lock'),'Wrong actual writer lock')
            entries.append(dict(kind=kind,database=actual['descriptor'],identity=snapshot['metadata']['identity']))
            paid+=actual['features']
        aliases=ALIASES[shard]
        terminal=[]
        for row in (mapped['map']['stdout']+'\n'+current['accounting']['stdout']).splitlines():
            fields=row.split('|')
            if fields[0] in aliases and 'FAILED' in fields:terminal.append(row)
        c.require(terminal,'No actual terminal failure for proposed scope')
        mapping=dict(status='EXACT_DT_SHARD_MAPPING_FROM_ACTUAL_SACCT',author=w.AUTHOR,
            global_shard_index=shard,original_configuration=original,producer_aliases=aliases,
            original_global_shard_indices=old['global_shard_indices'],actual_terminal_rows=terminal,
            array_readback=exported(ARRAY,w.CONTROL/'evidence'/ARRAY.name),fresh_census=exported(CENSUS,w.CONTROL/'evidence'/CENSUS.name),inventory=exported(INVENTORY,w.CONTROL/'evidence'/INVENTORY.name),database_SHA_receipt=exported(HASHES,w.CONTROL/'evidence'/HASHES.name),
            physical_id_order_not_used=True,live_global3_excluded=True,Root_custody_grant_still_required=True)
        mapping_local=output/('g%d'%shard)/'MAPPING.json';c.seal(mapping_local,mapping)
        mapping_ref=exported(mapping_local,w.CONTROL/('g%d'%shard)/'MAPPING.json')
        cfg=copy.deepcopy(old)
        cfg.update(schema='root-dt-oracle-overlay-recovery-v1',owner='Root',scope=w.SCOPE,source_root=str(w.SOURCE_ROOT),
            global_shard_indices=[shard],site=dict(name='babel',partition='general',qos='normal',account='cx',user='luyangz',gpu_type='L40S'),config_revision=3,allocation=dict(cpus=8,gpus=1,memory_mb=65536,minutes=120,reserve_seconds=300),
            maximum_restarts=3,required_storage_bytes=2<<30,sources=sources(old),
            original_spec=original,retained_databases=entries,output=str(w.OUTPUT_ROOT/('shard-%04d'%shard)),
            actual_parity=parity,actual_parity_review=parity_review,actual_mapping=mapping_ref,predecessor_aliases=aliases,
            wrapper=str(w.HERE/'run.sbatch').replace(str(ROOT),str(w.SOURCE_ROOT)),
            job_name='dsmm-root-dt-oracle-g%d-v1'%shard,expected_shard_records=REQUIRED[shard],observed_retained_records=paid,
            observed_missing_records=REQUIRED[shard]-paid,
            logical_claims='No expanded NN fixture, feature efficacy, fullpool or student PASS',
            route_disclosure='New Root General scalar; original measurement198/native85 and actualNN33 retained; Native91 only preprocessing oracle',
            actual_runtime_or_storage_approved=False)
        cfg_local=output/('g%d'%shard)/'CONFIG_V3.json';c.seal(cfg_local,cfg)
        refs[str(shard)]=exported(cfg_local,w.CONTROL/('g%d'%shard)/'CONFIG_V3.json');mappings[str(shard)]=mapping_ref
        counts[str(shard)]=dict(retained=paid,missing=REQUIRED[shard]-paid,required=REQUIRED[shard])
    return c.seal(output/'CONFIGURATION_INDEX_V1.json',dict(configurations=refs,mappings=mappings,counts=counts,
        Root_review_custody_storage_and_actual_admission_still_required=True,no_GPU_or_RPC_performed=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    print(json.dumps(build(parser.parse_args().output)))
