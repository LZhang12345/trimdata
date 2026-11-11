"""Frozen native recipe, evidence identities and deliberately closed dispatch."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Callable

from experiments.copilot7_coincide_actual_review_20261028.bridge import ADAPTATION
from experiments.root_fulltrajectory_student_v4_20261025 import protocol as native
from experiments.root_fulltrajectory_student_v4_20261025 import student_config

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PACKAGE = "experiments.copilot7_coincide_native_consumer_20261028"
CONTROL = ROOT / "collaboration/session7-coincide-actual-output-review-20261028/v1/native-consumer"
SCOPE = "COINCIDE_FIXED10_NATIVE_THREE_PAIRED_SEEDS_V1"
NOT_READY = "NOT_READY_SOURCE_PREPARATION_ONLY"
N, K, SELECTION_SEED = 660763, 66076, 1234
SEEDS = (17, 29, 43)
FUNCTION = "bbde85dd4c12b6321399d553d3333ec9c2e855d90cc7ca73600b3e32413be212"
OPERATOR_JOB = "10961650"
POOL = dict(
    path="/data/user_data/luyangz/native_campaign_20261023/root-wholepool-authority-v1/authority/POOL.json",
    bytes=18754787,
    sha256="c76040a17ca0dc1e99fea14514618edade3171a9b05f6ccc3e4751a5939e4964",
)
CANDIDATE = dict(
    path="/data/hf_cache/project_features/DS-MultiModal/coincide_native_operator_20261027/v3/budget-0.10/CANDIDATE.json",
    bytes=1856449,
    sha256="e704324dc5e6841db1d23f3b990829b54b2b8d564b946c58da8ccec67a6adb24",
)
OPERATOR_RESULT = dict(
    path="/data/hf_cache/project_features/DS-MultiModal/coincide_native_operator_20261027/v3/RESULT.json",
    bytes=13493,
    sha256="170577dfa98a9e443e3de1a2d8af9dc272c055ad36a8151550edbb886cef49c2",
)
SOURCE_SQRT_REFERENCE_ONLY = dict(
    path="/home/luyangz/data_selection_multimodal/native_campaign_v3_20261023/registry/production-students/production-student-v1-10-source_sqrt-s17-control10-shared-v1/CONFIG_FINAL.json",
    bytes=59718,
    sha256="bbafda365fcfbf56435d0ab7fc553a0f54386da674003c2bdab40214ac992403",
)
TRAINING_SEAM = "ROOT_OWNED_ADDITIVE_COINCIDE_NATIVE_TRAINING_SEAM_V1_UNAVAILABLE"
EVALUATION_SEAM = "ROOT_OWNED_ADDITIVE_COINCIDE_FOUR_BANK_EVALUATION_SEAM_V1_UNAVAILABLE"
TRAINING_INPUTS = (
    "accepted_selection", "materialized_candidate", "actual_output_review",
    "independent_method_review", "environment", "assets", "eligibility",
    "media_processor_RGB", "cache19_exact_receipt", "site_profile",
    "cold_native_and_restart_qualification", "independent_source_config_review",
    "measured_storage_reservation", "exclusive_function_ownership",
    "Root_execution_authorization", "P2_durable_continuation_review",
)
EVALUATION_INPUTS = (
    "training_configuration", "full516_completion", "accepted_model",
    "actual_parent_completion_and_all_in_cost", "independent_parent_review",
    "qualified_trained_loader", "four_bank_contract", "independent_source_config_review",
    "measured_storage_reservation", "exclusive_function_ownership",
    "Root_execution_authorization", "P2_durable_continuation_review",
)


class BindingError(ValueError):
    """Unbound or inconsistent evidence, never permission to execute."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BindingError(message)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def key(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def descriptor(value) -> dict:
    require(type(value) is dict and set(value) == {"path", "bytes", "sha256"},
            "Actual path/bytes/sha256 descriptor required; future identities stay null")
    path = value["path"]
    require(type(path) is str and path.startswith("/") and "<" not in path
            and "\x00" not in path and str(PurePosixPath(path)) == path
            and ".." not in PurePosixPath(path).parts, "Literal absolute artifact path required")
    require(type(value["bytes"]) is int and value["bytes"] > 0,
            "Positive actual byte count required")
    require(type(value["sha256"]) is str
            and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None
            and value["sha256"] != "0" * 64, "Actual SHA256 required")
    return deepcopy(value)


def independent(value: dict) -> None:
    reviewer = value.get("reviewer")
    require(value.get("independent") is True and type(reviewer) is str and reviewer.strip(),
            "Detached named independent review required")
    normalized = re.sub(r"[^a-z0-9]", "", reviewer.casefold())
    require(normalized not in {"s7", "copilot7"} and "session7" not in normalized
            and not normalized.startswith("copilot7"), "Implementer cannot accept its own evidence")


class Documents:
    """Hash bytes supplied by Root's scoped actor; this class performs no I/O.

    The caller owns custody, path authorization and actual allocation checks.
    A byte-valid document is not a GPU/runtime/independent qualification.
    """

    def __init__(self, read_bytes: Callable[[dict], bytes]):
        self.read_bytes = read_bytes

    def load(self, ref: dict):
        ref = descriptor(ref)
        try:
            raw = self.read_bytes(deepcopy(ref))
        except (KeyError, OSError) as error:
            raise BindingError("Missing or unavailable exact artifact: " + ref["path"]) from error
        require(type(raw) is bytes and len(raw) == ref["bytes"]
                and hashlib.sha256(raw).hexdigest() == ref["sha256"],
                "Evidence bytes differ from the actual descriptor")

        def pairs(items):
            result = {}
            for name, value in items:
                require(name not in result, "Duplicate JSON key")
                result[name] = value
            return result

        def nonfinite(_value):
            raise BindingError("Nonfinite JSON is not evidence")

        try:
            return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise BindingError("Malformed evidence JSON") from error

    def object(self, ref: dict) -> dict:
        value = self.load(ref)
        require(type(value) is dict, "Evidence object required")
        return value


def cell_id(seed: int) -> str:
    require(type(seed) is int and seed in SEEDS, "Only paired training seeds17/29/43")
    return "coincide-native10-s" + str(seed)


def horizon() -> dict:
    value = native.horizon(K)
    require((value["loader_batches"], value["natural_updates"], value["optimized_records"],
             value["unoptimized_tail_count"]) == (16519, 516, 66048, 28),
            "Native horizon changed; no local topup or rescaling repair")
    return value


def frozen_native() -> dict:
    expected_recipe = dict(global_batch=128, epochs=1, lora_r=128, lora_alpha=256, lora_dropout=.05,
        learning_rate=2e-4, projector_learning_rate=2e-5, weight_decay=0, warmup_ratio=.03,
        scheduler="cosine", model_max_length=2048, masking="c121_structured_turn_boundaries_v1",
        precision="fp16", tf32=False, attention="xformers", optimizer="adamw_torch", deepspeed="zero3")
    require(student_config.RECIPE == expected_recipe
            and native.RUNTIME == "explicit_xformers_cutlass_fw_bw_deterministic_v1"
            and native.n.PIN == "c121f0432da27facab705978f83c4ada465e46fd",
            "Frozen qualified native2048 source recipe/runtime changed")
    require({name: pin[1] for name, pin in student_config.ASSET_PINS.items() if name in
             ("vicuna", "clip336", "stage1_projector")} == dict(
                 vicuna="3321f76e3f527bd14065daf69dad9344000a201d",
                 clip336="ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
                 stage1_projector="5414da88308e4287a29f2e9609256458afb0a981"),
            "Frozen native asset revisions changed")
    return dict(
        official_commit=native.n.PIN,
        recipe=deepcopy(student_config.RECIPE),
        asset_pins={name: dict(repository=pin[0], revision=pin[1])
                    for name, pin in student_config.ASSET_PINS.items()},
        architecture="Stage1 LLaVA-1.5-7B Vicuna7B + CLIP336 + pretrain mlp2x projector",
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        trainable=["LoRA", "mm_projector"],
        frozen=["vision_tower", "lm_head", "non_LoRA_language_weights"],
        runtime=native.RUNTIME,
        versions=dict(torch="2.1.2+cu121", peft="0.7.1", transformers="4.37.2"),
        operator_faiss_torch220_environment_forbidden=True,
        expected_horizon=horizon(),
        no_extra_target_specific_tuning=True,
        no_selection_access_to_evaluation_labels=True,
    )


def training_template(seed: int, source_manifest: dict) -> dict:
    sid = cell_id(seed)
    return dict(
        schema="s7-coincide-native-training-binding-template-v1",
        status=NOT_READY, runtime_qualified=False, owner="Root", implementer="Session7",
        scope=SCOPE, id=sid, role="TRAIN_CANDIDATE", method="coincide",
        adaptation_label=ADAPTATION, fraction="0.10", pool=deepcopy(POOL),
        candidate=deepcopy(CANDIDATE), operator_result=deepcopy(OPERATOR_RESULT),
        eligible_pool_count=N, selected_whole_conversations=K,
        selection_seed=SELECTION_SEED, training_seed=seed,
        native=frozen_native(), source_manifest=descriptor(source_manifest),
        proposed_resources=dict(site="babel", partition="general", qos="normal", account="cx",
            gres="gpu:L40S:1", gpus=1, cpus=8, memory_mb=65536, microbatch=4,
            gradient_accumulation=32, loader_workers=0, minutes=None),
        checkpoint_policy=dict(checkpoint_seconds=1200, retained_accepted_states=2,
            transaction_bytes=5 << 30, transient_bytes=6 << 30, total_envelope_bytes=21 << 30,
            preserve_final_and_current_resume=True, cleanup_authorized=False),
        proposed_output="/data/hf_cache/project_checkpoints/DS-MultiModal/native_coincide_fixed10_20261028/" + sid,
        actual_configuration=None, actual_job=None, function_identity=None,
        bindings={name: None for name in TRAINING_INPUTS},
        runtime_seam=None, runtime_gate=TRAINING_SEAM,
        direct_legacy_wrapper_or_method_relabel_forbidden=True,
        P2_policy="STRICT_CONDITIONAL_ONLY_AFTER_ROOT_PROVES_DURABLE_ACCEPTED_PROGRESS_AND_ACK",
    )


def validate_training_template(value: dict) -> None:
    require(type(value) is dict, "Training binding template required")
    expected = training_template(value.get("training_seed"), value.get("source_manifest"))
    require(set(value) == set(expected), "Unknown training override or missing frozen field")
    variable = {"bindings", "function_identity"}
    require(all(value[name] == expected[name] for name in expected if name not in variable),
            "Frozen COINCIDE/native recipe, method, geometry, resources or source scope changed")
    require(type(value["bindings"]) is dict and set(value["bindings"]) == set(TRAINING_INPUTS),
            "Exact training-input binding API required")
    for ref in value["bindings"].values():
        if ref is not None:
            descriptor(ref)
    require(value["function_identity"] is None or
            re.fullmatch(r"[0-9a-f]{64}", str(value["function_identity"])) is not None,
            "Malformed function identity")


def readiness(value: dict) -> dict:
    validate_training_template(value)
    return dict(status=NOT_READY, runtime_qualified=False,
        missing=[TRAINING_SEAM] + [name for name, ref in value["bindings"].items() if ref is None],
        actual_configuration=None, actual_job=None, admission_owner="Root")


def require_training_dispatch(value: dict) -> None:
    validate_training_template(value)
    raise BindingError("NOT_READY: " + TRAINING_SEAM
        + "; native V2 KEYS/cell_id and V4 supported/shared_runtime.target reject 10:coincide; "
        "Root must provide an independently reviewed additive authorization/materialization/driver seam "
        "reusing unchanged native CLI, LoRA/projector, StudentAudit, state audit and accepted capsules. "
        "Provider PASS strings, source tests or a filled seam field cannot enable this version.")
