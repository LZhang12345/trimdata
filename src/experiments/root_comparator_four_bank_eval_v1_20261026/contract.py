"""Finite Root comparator identities and metadata-only configuration contract."""
import hashlib
import json
from pathlib import Path

from experiments.root_gqa_bank_qualification_v1_20261026 import worker as u
from experiments.root_native_trained_evaluation_v3_20261026 import banks

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
HOME = Path("/home/luyangz/data_selection_multimodal")
CONTROL = HOME / "collaboration/root-comparator-four-bank-eval-v1-20261026"
SOURCE_ROOT = HOME / "root_comparator_four_bank_eval_src_v1"
OUTPUT = Path("/data/hf_cache/project_features/DS-MultiModal/root_comparator_four_bank_eval_20261026")
LOCK = HOME / "collaboration/babel-two-track-20261019/ADMISSION.lock"
PACKAGE = "experiments.root_comparator_four_bank_eval_v1_20261026"
NAME = "dsmm-root-comparator-eval-v1"
COMMENT = "dsmm-root-comparator-eval:"
STATUS = "ROOT_CONFIGURED_ROOT264_COMPARATOR_FOUR_BANK_EVALUATION_V1"
BINDING_STATUS = "ROOT_BOUND_ROOT264_COMPARATOR_FOUR_BANK_EVALUATION_V1"
REVIEW_STATUS = "PASS_INDEPENDENT_ROOT_COMPARATOR_SIX_FOUR_BANK_SOURCE_CONFIG_V1"
SCOPE = "FINAL_ROOT264_COMPARATOR_FOUR_COMPLETE_OFFICIAL_BANKS_NOT_S1_OR_START"
RESOURCES = dict(cpus=8, memory_mb=65536, gpus=1, minutes=360)
JOBS = {
    ("source_sqrt", 17): "10942711", ("source_sqrt", 29): "10942714",
    ("source_sqrt", 43): "10942715", ("uniform", 17): "10942716",
    ("uniform", 29): "10942717", ("uniform", 43): "10942719",
}
QUALIFIER = "10942687"
REVIEWER = "10942858"
OFFICIAL_BANK_TEMPLATE = dict(
    path=str(HOME / "collaboration/root-native-trained-evaluation-v1-20261026/formal-configs-v3/CONFIG_s1-student-v1-10-0R-s17.json"),
    bytes=65250, sha256="be770d24d2355e5d2cf51092a0a641c26e1b4c4f7896aa9851bf93993c6979a1")
require = u.require
checked = u.checked
desc = u.desc
literal = u.literal


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def key(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def validate_configuration(cfg, load=checked):
    require(cfg["status"] == STATUS and cfg["owner"] == "Root" and cfg["scope"] == SCOPE,
            "Distinct Root comparator configuration required")
    cell = cfg["training_cell"]
    pair = cell["method"], cell["training_seed"]
    require(pair in JOBS and cell["training_job_id"] == JOBS[pair]
            and cell["fraction"] == "0.10" and cell["selected_whole_conversations"] == 66076
            and cell["expected_update_horizon"] == 516
            and cell["required_logical_sources"] == 264 and cell["required_physical_sources"] == 272,
            "Exactly the frozen six Root comparator cells")
    require(cfg["student_id"] == cell["student_id"]
            and cfg["training_configuration"] == cell["training_configuration"]
            and cfg["student_completion_path"] == cell["student_completion_path"]
            and cfg["source_root"] == str(SOURCE_ROOT)
            and cfg["resources"] == RESOURCES and cfg["counts"] == banks.COUNTS
            and cfg["banks"] == ["POPE", "GQA", "TextVQA", "ScienceQA_image"]
            and cfg["max_physical_attempts"] == 28506 and cfg["output_limit_bytes"] == 512 << 20
            and literal(cfg["output"]) == OUTPUT / cfg["student_id"],
            "Frozen bank, result, resource and late-bound completion scope")
    roster = load(cfg["root_comparator_roster"])
    require(len(roster["entries"]) == 6 and
            [entry for entry in roster["entries"] if entry["student_id"] == cfg["student_id"]] == [cell],
            "Cell must be byte-authenticated by the genuine Root roster")
    training = load(cfg["training_configuration"])
    require(training["id"] == cell["student_id"] and training["method"] == pair[0]
            and training["training_seed"] == pair[1] and training["fraction"] == "0.10",
            "Actual Root training configuration differs")
    require(cfg["official_bank_contract_configuration"] == OFFICIAL_BANK_TEMPLATE,
            "Exact existing official bank-contract provenance required, not S1 model/predictions")
    template = load(OFFICIAL_BANK_TEMPLATE)
    require(all(cfg[field] == template[field] for field in
              ("bank_inputs", "banks", "counts", "decoding", "scienceqa_decoding", "frozen_scope",
               "official_POPE_configuration", "phase3_configuration", "actual_media_observation",
               "qualified_loader_configuration", "required_actual_qualification_result_path",
               "required_independent_qualification_review_path")),
            "Official bank/generation/scorer/qualification identities cannot drift")
    require(cfg["required_training_job_id"] == JOBS[pair]
            and cfg["required_loader_qualifier_job_id"] == QUALIFIER
            and cfg["required_independent_reviewer_job_id"] == REVIEWER,
            "Exact matching training/loader/reviewer prerequisites")
    return training
