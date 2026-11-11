"""Explicit new loader adapter; untouched other three Source5 bank loaders."""
import copy
from pathlib import Path,PurePosixPath
from experiments.root_native_trained_evaluation_v3_20261026 import banks as old
from . import join


def load(bank,cfg,*,source_root):
    if bank!='TextVQA':return old.load(bank,cfg),None,None
    spec=cfg['bank_inputs'][bank];frozen=old.original.c.checked(cfg['frozen_scope'])['banks'][bank]
    old.u.require(frozen['media']==spec['media'],'Exact frozen bank media descriptor required')
    component=frozen['components'][0];roster=old.build_jsonl(bank,frozen,component['source'],component['support'])
    targets,receipt,mapping=join.join(roster['records'],old.original.c.checked(spec['official_annotations']),source_root=source_root)
    inventory=old.original.c.checked(spec['media']);old.proof_media(inventory)
    root=old.u.literal(spec['image_root'])
    old.u.require(root==Path('/data/hf_cache/datasets/dsmm_llava_665k_v1/media/textvqa/train_images'),'Exact existing original TextVQA image root')
    inventory=copy.deepcopy(inventory)
    for row in roster['records']:
        name=PurePosixPath(row['native_input']['image'])
        old.u.require(len(name.parts)==1 and name.suffix=='.jpg','Original native TextVQA basename required')
        expected=inventory['images'][row['image_reference']];path=old.u.literal(root/str(name))
        ref=dict(path=str(path),bytes=expected['encoded_bytes'],sha256=expected['encoded_sha256'])
        old.u.require(old.original.c.desc(path)==ref,'Existing TextVQA original file differs from exact historical media')
        expected['storage_locator']=dict(kind='shared_original_file',file=ref)
    rows=roster['records']
    old.u.require(len(rows)==5000 and len(inventory['images'])==3166 and {r['image_reference']for r in rows}==set(inventory['images'])
        and all('answer'not in r['native_input']and'answers'not in r['native_input']for r in rows),'Full answer-free TextVQA/media bank required')
    physical=dict(output=str(Path(cfg['output'])/bank),cache_root='/data/hf_cache')
    return (roster,inventory,targets,physical,None),receipt,mapping
