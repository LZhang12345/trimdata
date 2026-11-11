"""Completed-parent-only four-bank bindings, not the legacy current12 launcher."""
from __future__ import annotations

from copy import deepcopy
from pathlib import PurePosixPath
import re

from . import contract as c
from .binding import bind_accepted_selection

COUNTS = dict(POPE=8910, GQA=12578, TextVQA=5000, ScienceQA_image=2017)
METRICS = dict(
    POPE="native category accuracy; primary metric is not F1",
    GQA='native prediction.rstrip(".").lower() then exact original answer match',
    TextVQA="official TextVQA ten-answer leave-one-out credit",
    ScienceQA_image="official native image-subset choice accuracy",
)


def gqa_native_prediction(text: str) -> str:
    c.require(type(text) is str, "Raw GQA prediction must be text")
    return text.rstrip(".").lower()


def evaluation_template(seed: int, training_template_ref: dict, source_manifest: dict) -> dict:
    sid = c.cell_id(seed)
    return dict(
        schema="s7-coincide-native-four-bank-binding-template-v1",
        status=c.NOT_READY, runtime_qualified=False, owner="Root", scope=c.SCOPE,
        id=sid + "-fourbank", role="EVAL_COMPLETED_PARENT_ONLY",
        parent_id=sid, training_seed=seed, method="coincide", fraction="0.10",
        adaptation_label=c.ADAPTATION, training_template=c.descriptor(training_template_ref),
        source_manifest=c.descriptor(source_manifest), expected_parent_horizon=c.horizon(),
        banks={bank: dict(questions=count, metric=METRICS[bank], bank_inputs=None,
                    semantic_QA_keys=None, encoded_and_RGB_media=None, processor=None,
                    official_scorer=None) for bank, count in COUNTS.items()},
        full_panel_QA=sum(COUNTS.values()),
        generation_policy="PINNED_PEFT071_NATIVE_POSITIONAL_INPUTS_BRIDGE_V1",
        loader_policy="Exact qualified native loader; active unmerged LoRA + trained projector",
        no_evaluation_labels_for_selection_or_tuning=True,
        no_START_journal_reuse=True, retention_claim_requires_genuine_matched_START=True,
        ordinary_terminal_scores_are_not_retention=True,
        dependency_policy="STRICT_AFTEROK_EXACT_FULL516_COMPLETED_PARENT_ONLY",
        P2_policy="BLOCKED_UNTIL_ROOT_PROVES_DURABLE_PER_QA_PROGRESS_AND_CONTINUATION_ACK",
        self_requeue_ACK_or_partial_headers_are_not_completion=True,
        bindings={name: None for name in c.EVALUATION_INPUTS},
        runtime_seam=None, runtime_gate=c.EVALUATION_SEAM,
        proposed_resources=dict(site="babel", partition="general", gres="gpu:L40S:1",
            gpus=1, cpus=8, memory_mb=65536, minutes=None),
        actual_configuration=None, actual_job=None,
    )


def validate_evaluation_template(value: dict) -> None:
    c.require(type(value) is dict, "Evaluation binding template required")
    expected = evaluation_template(value.get("training_seed"), value.get("training_template"),
                                   value.get("source_manifest"))
    c.require(set(value) == set(expected)
              and all(value[name] == expected[name] for name in expected if name != "bindings"),
              "Frozen complete four-bank policy, counts, metrics or parent scope changed")
    c.require(type(value["bindings"]) is dict and set(value["bindings"]) == set(c.EVALUATION_INPUTS),
              "Exact evaluation-input binding API required")
    for ref in value["bindings"].values():
        if ref is not None:
            c.descriptor(ref)


def bind_completed_parent(template: dict, training_ref: dict, completion_ref: dict,
                          actual_completion_and_cost_ref: dict, independent_review_ref: dict,
                          documents: c.Documents) -> dict:
    """Validate metadata; tensor/media/loader and Root dispatch gates stay closed.

    The independent review must verify actual capsule payloads and all accepted
    exposure, including restart history. An ACCEPTED.json header alone is not it.
    """
    validate_evaluation_template(template)
    training = documents.object(training_ref)
    c.require(training.get("schema") == "native-selection-run-v1"
              and training.get("owner") == "Root" and training.get("role") == "student"
              and training.get("id") == template["parent_id"]
              and training.get("method") == "coincide" and training.get("fraction") == "0.10"
              and training.get("training_seed") == template["training_seed"]
              and training.get("selection_seed") == c.SELECTION_SEED
              and training.get("official_commit") == c.native.n.PIN
              and training.get("recipe") == c.frozen_native()["recipe"]
              and training.get("expected_horizon") == c.horizon()
              and training.get("qualification_max_steps") is None,
              "Real frozen COINCIDE native parent configuration required, never a SourceSqrt relabel")
    allocation = training.get("allocation", {})
    c.require(type(allocation) is dict and all(allocation.get(name) == value for name, value in
                  dict(gpus=1, cpus=8, memory_mb=65536, microbatch=4,
                       loader_workers=0, reserve_seconds=300).items())
              and allocation.get("minutes") in (360, 720, 1440)
              and training.get("checkpoint_seconds") == 1200,
              "Exact native world1/micro4/GA32 qualified allocation class required")
    adapter = training.get("execution_adapter", {})
    c.require(type(adapter) is dict and type(adapter.get("sources")) is dict
              and bool(adapter["sources"]) and adapter.get("runtime") == c.native.RUNTIME
              and adapter.get("package") not in (
                  None, "", "experiments.root_production_student_adapter_v2_20261025",
                  "experiments.root_fulltrajectory_student_v4_20261025")
              and adapter.get("ordinary_native_launch_forbidden") is True,
              "Root additive COINCIDE runtime/source seam required; legacy wrappers reject this method")
    binding_ref = c.descriptor(training.get("coincide_consumer_binding"))
    prepared = documents.object(binding_ref)
    c.validate_training_template(prepared)
    selection_ref = prepared["bindings"]["accepted_selection"]
    c.require(prepared["id"] == template["parent_id"] and selection_ref is not None
              and training.get("providers", {}).get("selection") == selection_ref
              and prepared["source_manifest"] == template["source_manifest"],
              "Actual native parent must bind this consumer's accepted selection/source manifest")
    bind_accepted_selection(selection_ref, prepared["bindings"]["materialized_candidate"],
                            prepared["bindings"]["actual_output_review"], documents)
    selected = documents.object(selection_ref)
    identity = c.native.n.run_identity(training)
    complete = documents.object(completion_ref)
    tail = complete.get("actual_unoptimized_tail_record_ids")
    c.require(type(tail) is list and all(type(ident) is str for ident in tail)
              and len(tail) == len(set(tail)) == 28
              and set(tail) <= set(selected["record_ids"]), "Actual28 distinct unoptimized selected records required")
    c.require(complete.get("status") == "COMPLETE_ACTUAL_NATURAL_STUDENT_TRAJECTORY_NOT_EVALUATION"
              and complete.get("configuration") == training_ref and complete.get("config_identity") == identity
              and complete.get("adapter_sources") == adapter["sources"]
              and complete.get("runtime") == c.native.RUNTIME and complete.get("expected") == c.horizon()
              and complete.get("actual_optimizer_updates") == 516
              and complete.get("actual_optimized_records") == 66048
              and complete.get("native_objective_scaling_unchanged") is True
              and complete.get("actual_no_skipped_optimizer_updates") is True
              and complete.get("downstream_evaluation_complete") is False
              and type(complete.get("attempts")) is list and bool(complete["attempts"]),
              "Full actual516/66048 completion required, not an accepted partial prefix or exit0")
    for attempt in complete["attempts"]:
        c.descriptor(attempt)
    accepted_ref = c.descriptor(complete.get("accepted"))
    accepted = documents.object(accepted_ref)
    c.require(accepted.get("status") == "ACCEPTED_FULL_NATIVE_STATE"
              and accepted.get("identity") == identity and accepted.get("global_step") == 516
              and accepted.get("world_size") == 1
              and accepted_ref["path"] == training["output"] + "/training/checkpoint-516/ACCEPTED.json",
              "Exact final native model/capsule identity required")
    files = accepted.get("files")
    c.require(type(files) is dict and {"non_lora_trainables.bin", "FROZEN_RECONSTRUCTION.json",
                  "config.json", "trainer_state.json", "rng_state.pth"} <= set(files)
              and bool({"adapter_model.bin", "adapter_model.safetensors"} & set(files))
              and len([name for name in files if name.endswith("optim_states.pt")]) == 1
              and len([name for name in files if name.endswith("model_states.pt")]) == 1,
              "Full world1 LoRA/projector/reconstruction/RNG/optimizer payload inventory required")
    for name, ref in files.items():
        c.require(type(name) is str and not PurePosixPath(name).is_absolute()
                  and ".." not in PurePosixPath(name).parts, "Relative accepted payload name required")
        c.require(type(ref) is dict and set(ref) == {"bytes", "sha256"},
                  "Exact accepted payload byte inventory required")
        c.descriptor(dict(path=str(PurePosixPath(accepted_ref["path"]).parent / name), **ref))
    semantic = documents.object(accepted.get("semantic_validation"))
    c.require(semantic.get("global_steps") == 516
              and semantic.get("files_verified_before_and_after") is True,
              "Actual final pre-acceptance semantic verification required")

    actual = documents.object(actual_completion_and_cost_ref)
    c.require(actual.get("state") == "COMPLETED" and actual.get("exit_code") == "0:0"
              and actual.get("configuration") == training_ref
              and type(actual.get("job_id")) is str
              and re.fullmatch(r"[1-9]\d*", actual["job_id"]) is not None
              and actual.get("all_setup_failed_training_native_attempts_accounted") is True,
              "Actual completed parent and all-in attempt cost, not a pending afterok ACK")
    attempts = actual.get("attempts")
    c.require(type(attempts) is list and bool(attempts), "All physical attempt costs required")
    physical = set()
    for attempt in attempts:
        c.require(type(attempt) is dict and type(attempt.get("job_id")) is str
                  and re.fullmatch(r"[1-9]\d*", attempt["job_id"]) is not None
                  and type(attempt.get("restart")) is int and attempt["restart"] >= 0,
                  "Immutable job/restart cost identity required")
        unit = (attempt["job_id"], attempt["restart"])
        c.require(unit not in physical, "Duplicate physical cost identity")
        physical.add(unit)
        for field in ("allocated_gpu_seconds", "allocated_CPU_core_seconds"):
            cost = attempt.get(field)
            c.require(type(cost) in (int, float) and 0 <= cost < float("inf"),
                      "Finite nonnegative actual allocated costs required")
    c.require(actual["job_id"] in {job for job, _ in physical},
              "Final actual parent missing from all-in costs")
    c.require(sum(attempt["allocated_gpu_seconds"] for attempt in attempts) > 0,
              "Full516 parent cannot have zero allocated GPU cost")
    review = documents.object(independent_review_ref)
    c.independent(review)
    c.require(review.get("status") == "PASS_INDEPENDENT_FULL516_COINCIDE_PARENT_AND_MODEL_V1"
              and review.get("configuration") == training_ref
              and review.get("completion") == completion_ref
              and review.get("accepted_model") == accepted_ref
              and review.get("actual_completion_and_cost") == actual_completion_and_cost_ref
              and review.get("all_capsule_payloads_and_full_exposure_verified") is True,
              "Detached actual full-model/exposure/cost review required; headers are insufficient")
    result = deepcopy(template)
    result["bindings"].update(training_configuration=c.descriptor(training_ref),
        full516_completion=c.descriptor(completion_ref), accepted_model=accepted_ref,
        actual_parent_completion_and_all_in_cost=c.descriptor(actual_completion_and_cost_ref),
        independent_parent_review=c.descriptor(independent_review_ref))
    validate_evaluation_template(result)
    return result


def validate_full_coverage(keys_by_bank: dict, actual_keys_by_bank: dict) -> None:
    c.require(type(keys_by_bank) is dict and type(actual_keys_by_bank) is dict
              and set(keys_by_bank) == set(actual_keys_by_bank) == set(COUNTS),
              "All four exact banks required")
    for bank, count in COUNTS.items():
        expected, actual = keys_by_bank[bank], actual_keys_by_bank[bank]
        c.require(type(expected) is list and type(actual) is list
                  and all(type(key) is str and key for key in expected + actual)
                  and len(expected) == len(set(expected)) == len(actual) == len(set(actual)) == count
                  and set(expected) == set(actual), "Partial/duplicate/foreign semantic QA keys: " + bank)


def require_evaluation_dispatch(value: dict) -> None:
    validate_evaluation_template(value)
    raise c.BindingError("NOT_READY: " + c.EVALUATION_SEAM
        + "; current12 consumer excludes COINCIDE. Root must bind actual full516 parent/model/cost, "
        "qualified trained loader, complete28505 QA+RGB+processor+official scorers, storage/ownership "
        "and reviewed strict-afterok/P2 durability. A header or self-requeue ACK is insufficient.")
