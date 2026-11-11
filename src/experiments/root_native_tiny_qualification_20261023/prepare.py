"""Freeze one original author batch8, with mixed modalities and padding peers."""
import argparse
from pathlib import Path
from experiments.root_native_campaign_20261023 import common as c
from experiments.root_native_campaign_20261023.selection import validate_pool
from experiments.root_native_baselines_20261023 import features,pins


def sources():
    root=Path(__file__).parent
    return {str(p.relative_to(c.ROOT)):c.sha(p) for p in sorted(list(root.glob('*.py'))+list(root.glob('*.sbatch')))
            if not p.name.startswith('test_')}


def one_batch(ids,rows):
    c.require(len(ids)==len(rows)>=1024,'Use a real qualified engineering construction, not a new8-row dataset')
    lanes=((len(ids)+3)//4+7)//8
    for index in range(4*lanes):
        batches=features.batches(ids,'coincide',index,4*lanes)
        if len(batches)!=1 or len(batches[0])!=8:continue
        positions=batches[0];selected=[rows[i] for i in positions]
        sizes=[sum(len(t['value']) for t in r['conversations']) for r in selected]
        if {bool(r.get('image')) for r in selected}!={False,True}:continue
        if len(set(sizes))<2 or not any(len(r['conversations'])>=4 for r in selected):continue
        return dict(shard_index=index,shard_count=4*lanes,positions=positions,
            record_ids=[ids[i] for i in positions],character_lengths=sizes,
            original_four_chunks_and_batch8_preserved=True)
    raise ValueError('No complete author batch8 has mixed text/image, multiple lengths and multiturn; repair engineering construction, never change its padding peers')


def prepare(spec,output):
    c.require(spec['schema']=='tiny-native-batch8-qualification-preparation-v1' and spec['owner']=='Root','Wrong qualifier preparation')
    missing=[name for name in ('pool','feature_assets','environment','site_profile','eligibility')
        if spec.get(name) is None or not Path(spec[name]['path']).is_file()]
    if missing:return dict(status='NOT_READY',missing=missing,actual_forward=False)
    pool,rows,ids=validate_pool(spec['pool'],allow_engineering=True)
    assets=c.checked(spec['feature_assets'])
    c.require(assets['status']=='PASS_PINNED_TINY_PROXY_ASSETS','Actual pinned Tiny assets required')
    for role in ('tiny_proxy','tiny_siglip'):pins.verify_tiny_snapshot(assets['assets'][role],role)
    environment=c.checked(spec['environment'])
    c.require(environment['versions']['torch'].startswith('2.2.0') and environment['versions']['transformers']=='4.37.2'
        and environment['versions']['tiktoken']=='0.5.2','Separate author-Torch2.2 tokenizer-qualified environment required')
    c.require('tiny' in environment['python_executable'] and 'native-py310-cu121-v1' not in environment['python_executable'],'Do not repurpose the native trainer environment')
    batch=one_batch(ids,rows)
    config=dict(schema='tiny-native-batch8-qualification-config-v1',owner='Root',id='coincide-native-qual8-v1',
        method='coincide',role='qualify_features',measurement_contract=features.CONTRACTS['coincide'],
        pool=spec['pool'],feature_assets=spec['feature_assets'],environment=spec['environment'],
        site_profile=spec['site_profile'],eligibility=spec['eligibility'],sources=pins.sources(),
        native_sources=c.sources(),qualifier_sources=sources(),output=spec['output'],
        allocation=dict(gpus=1,cpus=8,memory_mb=65536,minutes=55,reserve_seconds=300),
        exact_batch=batch,shard_index=batch['shard_index'],shard_count=batch['shard_count'],
        engineering_only=True,full_pool_qualified=False,ownership=spec.get('ownership'),storage_reservation=spec.get('storage_reservation'))
    config['measurement_identity']=features.measurement_identity(config,pool)
    return dict(status='PROPOSED_CONFIG_REQUIRES_REAL_OWNERSHIP_STORAGE_AND_INDEPENDENT_REVIEW',
        configuration_without_review=c.seal(Path(output)/'PROPOSED_CONFIG_WITHOUT_REVIEW.json',config),
        exact_batch=batch,actual_forward=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(c.canonical(prepare(c.read(a.spec),a.output)).decode())
