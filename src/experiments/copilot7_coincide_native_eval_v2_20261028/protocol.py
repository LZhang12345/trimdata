"""Finite fixed10 final-checkpoint consumer; banks and decoding remain frozen."""
from copy import deepcopy
from pathlib import Path
import re

from experiments.copilot7_coincide_native_exec_v2_20261028 import protocol as native
from experiments.root_current_four_bank_eval_v1_20261027 import contract as basis_contract
from . import model

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PACKAGE = "experiments.copilot7_coincide_native_eval_v2_20261028"
SOURCE_ROOT = native.HOME / "coincide_native_eval_src_v2_20261028"
OUTPUT = Path("/data/hf_cache/project_features/DS-MultiModal/coincide_native_evaluation_20261028/v2")
NAME, COMMENT = "dsmm-coincide-fourbank-v2", "dsmm-coincide-fourbank:"
SCOPE = "FINAL_COINCIDE_FIXED10_THREE_SEEDS_FOUR_COMPLETE_OFFICIAL_BANKS_V2"
STATUS = "ROOT_CONFIGURED_COINCIDE_COMPLETED_PARENT_FOUR_BANKS_V2"
BINDING = "ROOT_BOUND_COINCIDE_COMPLETED_PARENT_FOUR_BANKS_V2"
REVIEW = "PASS_INDEPENDENT_EXACT_COINCIDE_FOUR_BANK_SOURCE_CONFIG_V2"
RESOURCES = dict(cpus=8, memory_mb=65536, gpus=1, minutes=360)
PYTHON = "/data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python"
COUNTS = dict(POPE=8910, GQA=12578, TextVQA=5000, ScienceQA_image=2017)
BANK_FIELDS = basis_contract.BANK_FIELDS
require, checked, desc, canonical, key = native.require, native.checked, native.desc, native.canonical, native.key


def sources(cfg):
    result = dict(cfg["qualified_basis_sources"])
    for name, digest in result.items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts
                and native.sha(ROOT / name) == digest, "Frozen qualified evaluator source drift")
    for name, digest in native.sources().items():
        require(name not in result or result[name] == digest, "Conflicting unchanged native dependency")
        result[name] = digest
    for path in HERE.iterdir():
        if path.suffix in (".py", ".sbatch") and not path.name.startswith("test_"):
            result[str(path.relative_to(ROOT))] = native.sha(path)
    return dict(sorted(result.items()))


def validate_science(cfg, basis, training):
    require(cfg["status"] == STATUS and cfg["owner"] == "Root" and cfg["scope"] == SCOPE
        and cfg["student_id"] == native.cell_id("coincide", training["training_seed"])
        and cfg["student_id"] == training["id"] and cfg["method"] == "coincide"
        and cfg["fraction"] == "0.10" and cfg["training_seed"] == training["training_seed"]
        and cfg["source_root"] == str(SOURCE_ROOT) and cfg["python"] == PYTHON
        and cfg["resources"] == RESOURCES and cfg["banks"] == list(COUNTS) and cfg["counts"] == COUNTS
        and cfg["max_physical_attempts"] == sum(COUNTS.values()) + 1
        and cfg["output_limit_bytes"] == 512 << 20 and re.fullmatch("[a-f0-9]{32}", cfg["nonce"])
        and cfg["output"] == str(OUTPUT / training["id"] / cfg["nonce"]),
        "Only finite COINCIDE fixed10 seeds and four complete native banks")
    native.supported(training)
    require(all(cfg[field] == basis[field] for field in BANK_FIELDS)
        and cfg["qualified_basis_sources"] == basis["sources"] and cfg["sources"] == sources(cfg)
        and cfg["student_completion_path"] == str(Path(training["report"]) / "ROOT_LONG/COMPLETE.json")
        and cfg["adaptation_label"] == native.ADAPTATION
        and cfg["GQA_conversion"] == "prediction.rstrip('.').lower(); original target exact equality",
        "Same actual bank/media/processor/decoder inputs; declared native GQA conversion only")
    require(cfg["no_model_downloads"] is True and cfg["no_base_copies"] is True
        and cfg["no_START_journal_reuse"] is True and cfg["no_evaluation_targets_for_selection"] is True,
        "No reacquisition, foreign journals, target tuning or START/retention relabel")
    return training


def configuration(basis_ref, training_ref, training_binding_ref, nonce, load=checked):
    basis, training, binding = load(basis_ref), load(training_ref), load(training_binding_ref)
    native.supported(training)
    require(binding["schema"] == "coincide-fixed10-native-v2-binding-v1" and binding["owner"] == "Root"
        and binding["configuration"] == training_ref and binding["sources"] == native.sources()
        and binding["job_id"].isdecimal(), "Actual admitted original COINCIDE native parent")
    cfg = deepcopy(basis)
    cfg.update(status=STATUS, owner="Root", scope=SCOPE, student_id=training["id"], method="coincide",
        fraction="0.10", training_seed=training["training_seed"], nonce=nonce,
        source_root=str(SOURCE_ROOT), python=PYTHON, resources=RESOURCES,
        banks=list(COUNTS), counts=COUNTS, max_physical_attempts=sum(COUNTS.values()) + 1,
        output_limit_bytes=512 << 20, output=str(OUTPUT / training["id"] / nonce),
        training_configuration=training_ref, training_binding=training_binding_ref,
        required_training_job_id=binding["job_id"],
        student_completion_path=str(Path(training["report"]) / "ROOT_LONG/COMPLETE.json"),
        qualified_basis=basis_ref, qualified_basis_sources=basis["sources"],
        adaptation_label=native.ADAPTATION, no_evaluation_targets_for_selection=True,
        GQA_conversion="prediction.rstrip('.').lower(); original target exact equality")
    for name in ("evaluation_roster", "training_cell"):
        cfg.pop(name, None)
    cfg["sources"] = sources(cfg)
    validate_science(cfg, basis, training)
    return cfg
