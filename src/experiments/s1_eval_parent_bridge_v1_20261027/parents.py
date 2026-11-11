import hashlib
import json
from pathlib import Path
import subprocess

from experiments.root_native_campaign_v7_20261024 import common as c

SCHEMA = "s1-audited-failed-parent-completion-contract-v1"
QUALIFIED = "PASS_ACTUAL_AUDITED_NATIVE_COMPLETION_PARENT_INTERFACE_V1"
PEER = "PASS_INDEPENDENT_ACTUAL_AUDITED_NATIVE_PARENT_INTERFACE_V1"
CASES = {
    "production-student-v1-10-uniform-s29": ("10942717", None, "Root"),
    "production-student-v1-10-uniform-s43": ("10942719", "10958118", "Root"),
    "production-student-v1-10-source_sqrt-s17": ("10942711", "10958150", "Root"),
    "production-student-v1-10-source_sqrt-s29": ("10942714", "10958151", "Root"),
    "s1-student-v1-10-H0-s17": ("10937439", "10958152", "S1"),
}
COLLECTOR_RESULT_SHA = "b6ff2fb0931a1314b19ecd48cb5b0cc6a6923bc4141bd825d215a64cd9d5e622"
COLLECTOR_COMPLETE_SHA = "dd889d41ef540be5e9f71640dee2d88be8002789a8d25c9296ea6a2bbc9794a2"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def equal(a, b):
    return canonical(a) == canonical(b)


def query(argv):
    return subprocess.check_output(argv, text=True, timeout=30)


def accounting(job, expected_state, expected_exit, runner=query):
    raw = runner(["sacct", "-X", "-nP", "-j", job,
                  "--format=JobIDRaw,State,ExitCode,User,Account,Partition,QOS,ElapsedRaw,AllocTRES"])
    rows = [line.split("|") for line in raw.splitlines() if line.strip()]
    own = [row for row in rows if row[0] == job]
    c.require(len(own) == 1 and len(own[0]) == 9
              and own[0][1:7] == [expected_state, expected_exit, "luyangz", "cx", "general", "normal"]
              and own[0][7].isdecimal() and int(own[0][7]) > 0,
              "Exact original failed or completed owned physical accounting")
    resources = dict(part.split("=", 1) for part in own[0][8].split(",") if "=" in part)
    c.require(resources.get("cpu") == "8" and resources.get("gres/gpu") == "1"
              and resources.get("mem") in ("64G", "65536M"), "Original physical allocation identity")
    return raw


def declaration(evaluation, load=c.checked):
    contract_ref = evaluation["audited_training_parent"]
    value = load(contract_ref)
    sid = evaluation["student_id"]
    c.require(sid in CASES and value["schema"] == SCHEMA
              and value["owner"] == "Session1-author-delegated"
              and value["student_id"] == sid and value["family"] == CASES[sid][2]
              and value["original_job_id"] == CASES[sid][0]
              and value["original_binding"] == evaluation["training_cell"]["training_binding"]
              and value["original_configuration"] == evaluation["training_configuration"]
              and value["completion_path"] == evaluation["student_completion_path"]
              and value["old_failed_accounting_must_remain_failed"] is True,
              "Exact five-case additive failed-parent bridge declaration")
    training = load(value["original_configuration"])
    original = load(value["original_binding"])
    c.require(original["owner"] == "Root" and original["job_id"] == value["original_job_id"]
              and training["id"] == sid and training["owner"] == "Root"
              and training["role"] == "student" and training["fraction"] == "0.10"
              and training["training_seed"] in (17, 29, 43)
              and c.run_identity(training) == value["original_native_identity"]
              and value["expected_selected_count"] == 66076
              and value["expected_optimizer_updates"] == 516
              and Path(value["completion_path"]) == Path(training["report"]) / "ROOT_LONG/COMPLETE.json",
              "Unchanged native configuration, budget, identity and canonical completion")
    if CASES[sid][1] is None:
        c.require(value["origin_kind"] == "completed_collector"
                  and value["collector_result"]["sha256"] == COLLECTOR_RESULT_SHA,
                  "Only the genuine already-executed uniform29 collector")
    else:
        c.require(value["origin_kind"] == "paid_recovery"
                  and value["recovery_job_id"] == CASES[sid][1]
                  and value["future_recovery_result_SHA_bound"] is False,
                  "Real existing recovery, never a future fabricated result")
        binding = load(value["recovery_binding"])
        recovery = load(value["recovery_configuration"])
        c.require(binding["owner"] == "Root" and binding["job_id"] == CASES[sid][1]
                  and binding["recovery_configuration"] == value["recovery_configuration"]
                  and recovery["original_binding"] == value["original_binding"]
                  and recovery["configuration"] == value["original_configuration"]
                  and recovery["scientific_config_identity"] == value["original_native_identity"]
                  and recovery["full_restart_forbidden"] is True,
                  "Exact existing new physical recovery and unchanged logical state")
    return contract_ref, value, training


def complete_fields(contract, training, ref, value, protocol, load=c.checked):
    expected = protocol.horizon(66076)
    c.require(ref["path"] == contract["completion_path"]
              and value["status"] == "COMPLETE_ACTUAL_NATURAL_STUDENT_TRAJECTORY_NOT_EVALUATION"
              and value["configuration"] == contract["original_configuration"]
              and value["config_identity"] == contract["original_native_identity"] == c.run_identity(training)
              and value["adapter_sources"] == protocol.sources() and value["runtime"] == protocol.RUNTIME
              and value["expected"] == expected and value["actual_optimizer_updates"] == 516
              and value["actual_optimized_records"] == 66048
              and len(value["actual_unoptimized_tail_record_ids"])
              == len(set(value["actual_unoptimized_tail_record_ids"])) == 28
              and value["native_objective_scaling_unchanged"] is True
              and value["actual_no_skipped_optimizer_updates"] is True
              and value["downstream_evaluation_complete"] is False and bool(value["attempts"]),
              "Full natural epoch, accepted exposure, LR/objective and tail accounting required")
    accepted = load(value["accepted"])
    c.require(accepted["status"] == "ACCEPTED_FULL_NATIVE_STATE"
              and accepted["identity"] == value["config_identity"] and accepted["global_step"] == 516
              and accepted["world_size"] == 1 and len(accepted["files"]) == 17
              and Path(value["accepted"]["path"])
              == Path(training["output"]) / "training/checkpoint-516/ACCEPTED.json",
              "Actual original final checkpoint/state, not a compatible or partial model")
    return accepted


def collector_origin(contract, complete_ref, complete, load=c.checked, runner=query):
    result = load(contract["collector_result"])
    cfg = load(result["configuration"])
    registration = load(result["registration"])
    c.require(complete_ref["sha256"] == COLLECTOR_COMPLETE_SHA
              and result["status"] == "ACTUAL_UNIFORM29_PAID_FULL_TRAJECTORY_COLLECTED_NOT_EVALUATION"
              and result["full_trajectory"] == complete_ref
              and result["original_configuration"] == contract["original_configuration"]
              and result["original_native_identity"] == contract["original_native_identity"]
              and result["accepted"] == complete["accepted"] and result["old_job_state"] == "FAILED"
              and result["original_source_or_recipe_or_identity_changes"] is False
              and result["neural_forwards"] == result["checkpoint_or_state_writes"] == 0
              and result["scheduler_mutations"] is False and result["downstream_evaluation_complete"] is False
              and result["whole_epoch_collector"] == "UNCHANGED_LOGICAL_ROOT264"
              and cfg["configuration"] == contract["original_configuration"]
              and cfg["student_complete"] == result["old_child_complete"]
              and result["actual"]["registration"] == result["registration"],
              "Genuine completed metadata collector is provenance, not a new successful training scalar")
    audit = load(result["actual_capsule_audit"])
    peer = load(result["independent_actual_audit_review"])
    c.require(audit["cursor"] == audit["successful_updates"] == 516 and audit["skipped_steps"] == 0
              and audit["accepted"] == complete["accepted"]
              and peer["status"] == "PASS_INDEPENDENT_ACTUAL_FIVE_FAILED_STUDENT_CAPSULE_VALIDITY_AUDIT_V1"
              and peer["independent"] is True
              and any(row["student_id"] == contract["student_id"] and row["cursor"] == 516
                      and row["readout"] == result["actual_capsule_audit"] for row in peer["cases"]),
              "Independent actual final-capsule validity, not just source approval")
    return dict(kind="completed_collector", result=contract["collector_result"],
                registration=result["registration"], captured_registration=registration,
                original_failed_accounting=accounting(contract["original_job_id"], "FAILED", "1:0", runner),
                original_training_state_preserved="FAILED", no_original_failed_afterok=True)


def recovery_origin(contract, complete_ref, complete, load=c.checked, describe=c.desc, runner=query):
    binding = load(contract["recovery_binding"])
    cfg = load(contract["recovery_configuration"])
    result_ref = describe(contract["recovery_result_path"])
    result = load(result_ref)
    expected_status = ("ACTUAL_S1_H017_PAID486_RESTORED_AND_PROGRESS_REQUIRES_INDEPENDENT_REVIEW"
                       if contract["family"] == "S1" else
                       "ACTUAL_ROOT10_PAID_PREFIX_RESTORED_AND_PROGRESS_REQUIRES_INDEPENDENT_REVIEW")
    c.require(result["status"] == expected_status and result["binding"] == contract["recovery_binding"]
              and result["original_configuration"] == contract["original_configuration"]
              and result["original_scientific_identity"] == contract["original_native_identity"]
              and result["previous_failed_attempt_preserved"] is True
              and result["paid_state_payloads_rewritten"] is False
              and result["full_native_endpoint_observed"] is True
              and result["whole_trajectory_exposure_reconciliation_complete"] is True
              and result["final_accepted_step"] == 516 and result["student_completion"] == complete_ref
              and result["actual_full_accepted_history"] == complete["attempts"]
              and result["original_accepted_history"] == cfg["accepted_history"],
              "A physically successful partial recovery is not a completed native epoch")
    gate = load(result["restored_gate"])
    saved = load(cfg["raw_SAVE486"] if contract["family"] == "S1" else cfg["raw_saved_state"])
    cursor = 486 if contract["family"] == "S1" else load(cfg["accepted_history"][-1])["global_step"]
    status = ("PASS_EXACT_RESTORED_BEGIN_EQUALS_ORIGINAL_SAVE486_BEFORE_MICROBATCH"
              if contract["family"] == "S1" else
              "PASS_EXACT_RESTORED_BEGIN_EQUALS_ORIGINAL_SAVE_ACCEPTED_PREFIX_BEFORE_MICROBATCH")
    c.require(gate["status"] == status and gate["before_any_original_training_step"] is True
              and gate["original_scientific_identity"] == contract["original_native_identity"]
              and equal(gate["expected"], gate["actual"]) and equal(gate["expected"], saved["state"])
              and gate["actual"]["global_step"] == gate["actual"]["actual_DS_global_steps"] == cursor
              and gate["actual"]["DS_skipped_steps"] == 0
              and gate["actual"]["all_weights_finite"] is True, "Exact actual saved/restored BEGIN state gate")
    before, after = load(result["before_guard"]), load(result["after_guard"])
    outcome, proof = load(result["actual_child_exit"]), load(result["physical_execution_proof"])
    packet = load(binding["source_packet"])
    ownership = load(binding["runtime_ownership"])
    c.require(before["latest_step"] == cursor and after["latest_step"] == 516
              and outcome["configuration"] == contract["original_configuration"]
              and outcome["config_identity"] == contract["original_native_identity"]
              and outcome["torchrun_process_exit_code"] == outcome["logical_exit_code"] == 0
              and outcome["receipt"] == result["actual_child_receipt"]
              and proof["physical_binding"] == contract["recovery_binding"]
              and proof["executed_sources"] == packet["executed_sources"]
              and proof["child_receipt"] == outcome["receipt"] and proof["logical_exit_code"] == 0
              and ownership["job_id"] == binding["job_id"] == contract["recovery_job_id"]
              and ownership["scientific_config_identity"] == contract["original_native_identity"]
              and ownership["no_live_duplicate_function"] is True
              and ownership["superseded_controllers_inactive"] is True,
              "Actual accepted progression, source, child and physical recovery ownership")
    return dict(kind="paid_recovery", result=result_ref, restored_gate=result["restored_gate"],
                physical_execution_proof=result["physical_execution_proof"],
                original_failed_accounting=accounting(contract["original_job_id"], "FAILED", "1:0", runner),
                completed_recovery_accounting=accounting(contract["recovery_job_id"], "COMPLETED", "0:0", runner),
                original_training_state_preserved="FAILED", no_original_failed_afterok=True)


def resolved(evaluation, load=c.checked, describe=c.desc):
    from .qualify import sources
    contract_ref, contract, training = declaration(evaluation, load)
    qualification_ref = describe(contract["qualification_result_path"])
    qualification = load(qualification_ref)
    review_ref = describe(contract["independent_actual_review_path"])
    review = load(review_ref)
    complete_ref = describe(contract["completion_path"])
    c.require(qualification["status"] == QUALIFIED
              and qualification["parent_contract"] == contract_ref
              and qualification["training_configuration"] == contract["original_configuration"]
              and qualification["student_completion"] == complete_ref
              and qualification["original_native_identity"] == contract["original_native_identity"]
              and qualification["old_failed_accounting_preserved"] is True
              and qualification["full_epoch_exposure_LR_objective_verified"] is True
              and qualification["bridge_sources"] == sources()
              and qualification["new_NN"] == 0 and qualification["checkpoint_writes"] is False
              and review["status"] == PEER and review["independent"] is True
              and review["readout"] == qualification_ref and review["parent_contract"] == contract_ref
              and review["bridge_sources"] == sources()
              and review["original_native_identity_verified"] is True
              and review["full_epoch_exposure_LR_objective_verified"] is True
              and review["old_failed_accounting_preserved"] is True
              and review["reviewer"] not in (None, "", "Session1", "S1", "Session1-author-delegated"),
              "Actual interface qualification and independent readout acceptance required")
    registration = load(qualification["registration"])
    c.require(registration["status"] == "ACTUAL_S1_CURRENT_OWNED_CPU_STEP_REGISTERED"
              and registration["actual"] == qualification["actual"]
              and qualification["actual"]["numeric_step"].isdecimal()
              and int(qualification["actual"]["numeric_step"]) < 0xfffffff0
              and review["actual_registration"] == qualification["registration"],
              "Genuine current-parent numeric CPU qualification, never an old batch actor")
    return training, complete_ref, load(complete_ref), qualification["historical_proofs"], dict(
        kind=contract["origin_kind"], contract=contract_ref, actual_qualification=qualification_ref,
        independent_actual_review=review_ref, original_failed_afterok_used=False)
