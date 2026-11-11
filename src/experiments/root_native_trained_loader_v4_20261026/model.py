"""Reuse every frozen V3 state/injection guard, then bridge only generation API."""
from pathlib import Path
from experiments.root_native_trained_loader_v3_20261026 import model as original
from experiments.root_native_campaign_v7_20261024 import common as n
from . import generation

def sources():
    paths=[p for directory in(Path(__file__).parent,Path(original.__file__).parent)for p in directory.iterdir()
        if p.suffix=='.py'and not p.name.startswith('test_')]
    return {str(p.relative_to(n.ROOT)):n.sha(p)for p in paths}

def compose(accepted_ref,training_ref,diagnostic_path=None):
    loaded,proof=original.compose(accepted_ref,training_ref,diagnostic_path)
    proof=dict(proof,generation_API_bridge=generation.install(loaded[1]))
    return loaded,proof

def identity(loaded,proof,runtime):
    value=original.identity(loaded,proof,runtime)
    value['generation_API_bridge']=proof['generation_API_bridge']
    return value

# Exact unchanged APIs remain available to existing semantic-state test clients.
portable_support=original.portable_support
accepted_checkpoint=original.accepted_checkpoint
verify_parameter_layout=original.verify_parameter_layout
verify_loaded_parameters=original.verify_loaded_parameters
native_adapter_targets=original.native_adapter_targets
attach_native_adapter=original.attach_native_adapter
restore_persistent_buffers=original.restore_persistent_buffers
authenticated_archive=original.authenticated_archive
semantic=original.semantic
