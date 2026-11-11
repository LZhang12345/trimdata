"""Exact primary sources, independent of old executed feature workers."""
from pathlib import Path
import hashlib
from experiments.root_native_campaign_20261023 import common as c

COIN_PIN = '629f7760a94c70740b1138eed080cb42eb34c69c'
TAILOR_PIN = 'd6fa5eb6bc2920a561c614abb6b841a128affb92'
COIN_ROOT = c.ROOT / 'collaboration/root-native-baselines-20261023/reference_sources/COINCIDE' / COIN_PIN
COIN_CODE = COIN_ROOT / 'COINCIDE_cluster'
COIN_SCORE = COIN_CODE / 'tinyllava/eval/score/coincide'
TAILOR_ROOT = c.ROOT / 'collaboration/published-foundation-20261022/reference_sources/DataTailor' / TAILOR_PIN
TAILOR_MODEL = c.ROOT / 'collaboration/published-foundation-20261022/session7/reference_additions/DataTailor' / TAILOR_PIN / 'llava/model/modeling_llama.py'
TINY_REPOSITORY = 'bczhou/TinyLLaVA-2.0B'
TINY_REVISION = 'ce2d6afe3749d5623dc0ac62f2d760527482d45d'
TINY_WEIGHTS_SHA = '6ea6d36d10e10fdae9b90f56a961fdde5cb8795ebc26408f5351877d129d18dd'
TINY_WEIGHTS_BYTES = 4097746976
CATALOG_ROOT=c.ROOT/'collaboration/root-native-baselines-20261023'


def sources():
    local = Path(__file__).parent
    files = list(local.glob('*.py')) + list(local.glob('*.sbatch')) + list(COIN_CODE.rglob('*.py'))
    files += [TAILOR_ROOT/name for name in ('datatailor_selector.py', 'make_clustering.py', 'relationship_value_calculation.py')]
    files += [TAILOR_MODEL]
    files += [CATALOG_ROOT/name for name in ('TINY_PROXY_PINNED_TREE_V1.json','TINY_SIGLIP_PINNED_TREE_V1.json')]
    files += [c.ROOT/'experiments/copilot7_published_foundation_20261022'/name
              for name in ('datatailor_v1.py', 'datatailor_v2.py')]
    return {str(p.relative_to(c.ROOT)): c.sha(p) for p in sorted(set(files))
            if not p.name.startswith('test_')}


def verify(packet):
    c.require(packet == sources(), 'Published operator execution sources differ')


def verify_tiny_snapshot(asset,role):
    catalog=c.read(CATALOG_ROOT/('TINY_PROXY_PINNED_TREE_V1.json' if role=='tiny_proxy' else 'TINY_SIGLIP_PINNED_TREE_V1.json'))
    c.require((asset['repository'],asset['revision'])==(catalog['repository'],catalog['revision']),'Tiny catalog revision differs')
    entries={row['path']:row for row in catalog['entries']}
    actual_names={p.name for p in Path(asset['snapshot']).iterdir() if p.is_file()}
    c.require(actual_names<=set(entries),'Unpinned snapshot sidecar could alter model/tokenizer loading')
    required={'config.json','model.safetensors'}
    if role=='tiny_proxy':required.update(('generation_config.json','arcade100k.tiktoken','special_tokens_map.json','tokenizer_config.json'))
    actual={Path(d['path']).name:d for d in asset['files']}
    c.require(required<=set(actual),'Tiny snapshot missing pinned tokenizer/configuration/weights')
    for name in required:
        descriptor=actual[name];path=Path(descriptor['path']);entry=entries[name]
        c.require(path.parent.resolve()==Path(asset['snapshot']).resolve() and c.desc(path)=={k:descriptor[k] for k in ('path','bytes','sha256')},'Tiny exact snapshot locator/bytes differ')
        c.require(descriptor['bytes']==entry['size'],'Tiny file size differs from pinned primary tree')
        if 'lfs' in entry:c.require(descriptor['sha256']==entry['lfs']['oid'],'Tiny LFS SHA256 differs')
        else:
            raw=path.read_bytes();digest=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            c.require(digest==entry['oid'],'Tiny tokenizer/config Git blob differs')
