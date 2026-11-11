"""Finite COINCIDE metadata specialization of the qualified Stage1 native engine."""
import ast
from pathlib import Path
import re

from experiments.root_native_campaign_v7_20261024 import common as n, config as native_config
from experiments.root_fulltrajectory_student_v4_20261025 import protocol as anchor
from experiments.root_production_epoch_qualification_v1_20261024 import protocol as qualification
from experiments.root_native_deterministic_attention_v2_20261025 import bridge
from experiments.meta_probe_v1_20261028.contracts import canonical
from experiments.copilot7_coincide_actual_review_v2_20261028 import bridge as selection_bridge

ROOT = n.ROOT
HERE = Path(__file__).resolve().parent
PACKAGE = "experiments.copilot7_coincide_native_exec_v2_20261028"
SOURCE_ROOT = Path("/home/luyangz/data_selection_multimodal/coincide_native_exec_src_v2_20261028")
HOME = Path("/home/luyangz/data_selection_multimodal")
CHECKPOINT_ROOT = "/data/hf_cache/project_checkpoints/DS-MultiModal/coincide_native_exec_v2_20261028"
REPORT_ROOT = HOME / "collaboration/session7-coincide-actual-output-review-20261028/v2/native"
AUTHOR = "Session7"
SEEDS = (17, 29, 43)
KEYS = ("coincide",)
N, K = 660763, 66076
POOL = selection_bridge.POOL
CANDIDATE = selection_bridge.CANDIDATE
ADAPTATION = selection_bridge.ADAPTATION
RUNTIME = anchor.RUNTIME
CAPSULE_BYTES, TRANSIENT_BYTES = 5 << 30, 6 << 30
GATE = "REQUIRES_REVIEWED_ACTUAL_COINCIDE_NATIVE_EXECUTION_V2"
LOCK = HOME / "collaboration/babel-two-track-20261019/ADMISSION.lock"
COMMENT = "dsmm-coincide-native:"
RESOURCES = dict(gpus=1, cpus=8, memory_mb=65536, minutes=720)
NATURAL_ANCHOR_SEED = anchor.NATURAL_ANCHOR_SEED
NATURAL_ANCHOR_READOUT = anchor.NATURAL_ANCHOR_READOUT
SEED_POLICY = anchor.SEED_POLICY
require, key, now, sha, desc, checked, read, seal = (
    n.require, n.key, n.now, n.sha, n.desc, n.checked, n.read, n.seal)
horizon = anchor.horizon


def independent(value):
    name = value.get("reviewer")
    require(type(name) is str and name.strip() and name.isascii(),
            "Named genuine independent exact reviewer required")
    normalized = re.sub("[^a-z0-9]", "", name.casefold())
    require(normalized not in ("session7", "s7", "copilot7", "sourceauthor")
            and "session7" not in normalized and not normalized.startswith("copilot7"),
            "Implementer cannot accept its own execution")


def cell_id(method, seed):
    require(method == "coincide" and type(seed) is int and seed in SEEDS,
            "Only fixed10 COINCIDE paired seeds17/29/43")
    return "coincide-native10-s" + str(seed)


CANARY_ID = cell_id("coincide", 17)


def sources():
    result = dict(anchor.sources())
    for name, digest in bridge.sources().items():
        require(name not in result or result[name] == digest, "Conflicting unchanged runtime dependency")
        result[name] = digest
    own = [path for path in HERE.iterdir() if path.suffix in (".py", ".sbatch")
           and not path.name.startswith("test_")]
    # Selection validation is an explicit new dependency, never an old source alias.
    own.extend([Path(selection_bridge.__file__), Path(selection_bridge.__file__).with_name("data_io.py")])
    for path in list(own):
        for parent in path.parents:
            if parent == ROOT:
                break
            init = parent / "__init__.py"
            if init.is_file():
                own.append(init)
    for path in own:
        result[str(path.relative_to(ROOT))] = sha(path)
    return dict(sorted(result.items()))


def supported(cfg):
    require(cfg["id"] == cell_id(cfg["method"], cfg["training_seed"])
            and cfg["fraction"] == "0.10" and type(cfg["selection_seed"]) is int
            and cfg["selection_seed"] == 1234
            and cfg["expected_horizon"] == horizon(K),
            "Only exact finite COINCIDE K66076 native516, not an old uniform/A identity")
    return True


def bare(value):
    return {k: v for k, v in value.items() if k != "review"}


def numerical_bodies():
    result = {}
    for name in ("observer.py", "state_audit.py", "collect.py", "capsule.py", "canary.py"):
        before = (anchor.HERE / name).read_text()
        after = (HERE / name).read_text().replace(PACKAGE, anchor.PACKAGE)
        require(before == after and ast.dump(ast.parse(before), include_attributes=False)
                == ast.dump(ast.parse(after), include_attributes=False),
                "Qualified observational/lifecycle body changed: " + name)
        result[name] = dict(anchor_sha256=sha(anchor.HERE / name), target_sha256=sha(HERE / name))
    from .student_config import cli
    require(cli is native_config.cli, "Exact qualified nativev7 CLI function object required")
    return result
