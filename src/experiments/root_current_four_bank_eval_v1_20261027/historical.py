"""Root264 completion and per-segment physical272 checks, without live training gates."""
import json
import math
import os
from pathlib import Path

from experiments.root_control10_telemetry_physical_v1_20261026 import route
from experiments.root_fulltrajectory_student_v4_20261025 import protocol as student
from experiments.root_native_qualification_v10_20261024.observer import validate_lr_trajectory
from . import contract as c


def jsonl(path):
    ref = c.desc(path)
    return [json.loads(line) for line in c.u.read(ref).decode().splitlines()]


def segment_paths(training, binding, restart):
    c.require(type(restart) is int and restart >= 0, "Original physical restart required")
    name = binding["job_id"] + "-r" + str(restart) + "-" + binding["nonce"][:8]
    return (Path(training["report"]) / "attempts" / name / "child",
            Path(training["report"]) / "ROOT_LONG/requests" / ("restart-" + str(restart) + ".json"))


def export_evidence(export, storage, actual):
    after = export["after_segment_observation"]
    c.require(export["status"] == "ACTUAL_ROOT_CONTROL10_PHYSICAL_SHARED_EXPORT_V1"
              and export["checkpoint_root"] == student.CHECKPOINT_ROOT
              and export["export_identity"] == after["export_identity"] == route.EXPORT,
              "Genuine exact checkpoint NFS export required")
    for value in (export, after):
        device = value["observed_device"]
        c.require(type(device) is int and device >= 0 and value["device_number_is_namespace_local"] is True
                  and value["mount"]["major_minor"] == "%d:%d" % (os.major(device), os.minor(device))
                  and all(value["mount"][key] == item for key, item in route.EXPORT.items())
                  and value["actual_job_id"] == actual["job_id"]
                  and value["actual_restart"] == actual["restart"]
                  and value["actual_node"] == actual["node"],
                  "Same recorded process mount/device/job/restart/node linkage required")
    c.require(export["observed_device"] == after["observed_device"] == storage["checkpoint_root_device"]
              and export["mount_namespace"] == after["mount_namespace"]
              and type(export["mount_namespace"]) is str and export["mount_namespace"].startswith("mnt:[")
              and storage["actual_job_id"] == actual["job_id"],
              "Never normalize or mix segment namespace/device observations")


def physical_segment(ref, receipt, cfg, training, load=c.checked, describe=c.desc, history=()):
    cell = cfg["training_cell"]
    binding_ref, activation_ref = cell["training_binding"], cell["training_activation"]
    binding, activation = load(binding_ref), load(activation_ref)
    intent, ack = load(binding["intent"]), load(binding["ack"])
    c.require(intent["nonce"] == binding["nonce"] and intent["request"] == binding["request"]
              and ack["job_id"] == binding["job_id"] and ack["intent"] == binding["intent"],
              "Original intent-before-ACK custody must remain authentic")
    restart = receipt["actual_restart_count"]
    attempt, execution_path = segment_paths(training, binding, restart)
    c.require(Path(ref["path"]).parent == attempt
              and receipt["actual_job_id"] == binding["job_id"] == cell["training_job_id"]
              and binding["owner"] == activation["owner"] == "Root"
              and binding["schema"] == "root-fulltrajectory-held-binding-v2"
              and activation["binding"] == binding_ref
              and activation["configuration"] == cfg["training_configuration"],
              "Exact historical Root cell/binding/activation and child path")
    execution_ref = describe(execution_path)
    if restart == 0:
        c.require(execution_ref == cell["training_outer_execution"], "Original physical execution changed")
    execution = load(execution_ref)
    c.require(execution["actual_restart"] == restart
              and execution["config_identity"] == student.n.run_identity(training)
              and execution["sources"] == student.sources()
              and execution["accepted_history"] == list(history),
              "Historical logical execution/accepted-chain identity differs")
    activation_review, execution_review = load(activation["review"]), load(execution["review"])
    c.require(activation_review["status"] == "PASS_INDEPENDENT_ACTUAL_FULLTRAJECTORY_JOB_ACTIVATION_V2"
              and activation_review["independent"] is True
              and activation_review["activation_identity"] == student.key(
                  {key: value for key, value in activation.items() if key != "review"})
              and activation_review["binding"] == binding_ref
              and activation_review["target_sources"] == activation["target_sources"]
              and activation_review["actual_gate_descriptors"] == activation["gates"],
              "Historically reviewed exact activation, without current training profile gates")
    c.require(execution_review["status"] == "PASS_INDEPENDENT_EXACT_BABEL_SHARED_OUTER_EXECUTION_V4"
              and execution_review["independent"] is True
              and execution_review["execution_policy_identity"] == student.key(
                  {key: value for key, value in execution.items() if key not in
                   ("review", "actual_restart", "accepted_history", "predecessor_transition")})
              and type(execution["max_physical_attempts"]) is int
              and restart < execution["max_physical_attempts"] <= 516,
              "Historical exact outer-execution review and finite original attempt allowance")
    if restart:
        previous = load(execution["predecessor_transition"])
        c.require(previous["binding"] == binding_ref and previous["to_restart"] == restart
                  and previous["continuation"] == list(history)[-1],
                  "Recorded predecessor transition must bind this accepted chain")
    permit = load(execution["authorization"])
    c.require(permit["configuration"] == cfg["training_configuration"]
              and permit["physical_binding"] == binding_ref and permit["phase"] == "production"
              and permit["runtime_certificate"] == execution["runtime_certificate"],
              "Historical original permit/certificate linkage, not live authorization")
    # This is the authenticated policy primitive, not current()/execution() or
    # verify_physical_receipt(), which revalidate a live training authorization.
    extension = route.physical_policy(binding_ref, activation_ref, execution_ref,
                                      execution, training, load=load)
    proof_ref = describe(attempt / "PHYSICAL_EXECUTION_PROOF.json")
    proof = load(proof_ref)
    actual = proof["actual"]
    c.require(proof["physical_extension"] == actual["physical_extension"] == extension
              and proof["child_receipt"] == ref and proof["configuration"] == cfg["training_configuration"]
              and proof["config_identity"] == student.n.run_identity(training)
              and proof["logical_sources"] == student.sources()
              and proof["executed_sources"] == route.sources()
              and len(proof["logical_sources"]) == 264 and len(proof["executed_sources"]) == 272
              and actual["job_id"] == binding["job_id"] and actual["restart"] == restart
              and proof["operators"] == receipt["operators"],
              "Exact same-segment physical272 receipt and observational extension")
    recorded = route.transport.fields(actual["scheduler"])
    c.require(recorded["JobId"] == binding["job_id"] and recorded["Restarts"] == str(restart)
              and recorded["Comment"] == "dsmm-fulltrajectory:" + binding["nonce"]
              and recorded["NodeList"] == actual["node"],
              "Original captured historical job/restart/nonce/node, not a current actor")
    operators = load(proof["operators"])
    c.require(operators["observational_execution_extension"] == extension,
              "Actual executed operator telemetry extension is mandatory")
    outcome = load(describe(attempt / "CHILD_EXIT.json"))
    expected_exit = 0 if receipt["status"] == "COMPLETE_STUDENT_TRAINING_NOT_EVALUATION" else 75
    c.require(proof["logical_exit_code"] == outcome["logical_exit_code"] == expected_exit
              and outcome["receipt"] == ref and outcome["configuration"] == cfg["training_configuration"],
              "Original actual child exit/receipt differs")
    storage = load(describe(attempt / "ACTUAL_STORAGE.json"))
    c.require(storage["reservation"] == training["storage_reservation"],
              "Recorded segment storage must belong to this training")
    export_evidence(load(proof["checkpoint_export"]), storage, actual)
    return proof_ref


def complete_student(cfg, load=c.checked, describe=c.desc, read_jsonl=jsonl):
    training = c.validate_configuration(cfg, load)
    c.require(len(student.sources()) == 264 and len(route.sources()) == 272,
              "Exact genuine Root264/physical272 implementation required")
    c.require(training["sources"] == student.n.sources() and len(training["sources"]) == 91,
              "Unchanged Native91 state identity required")
    path = c.literal(cfg["student_completion_path"])
    c.require(path == Path(training["report"]) / "ROOT_LONG/COMPLETE.json",
              "Original Root collector location required")
    complete_ref = describe(path)
    value = load(complete_ref)
    identity = student.n.run_identity(training)
    horizon = student.horizon(66076)
    c.require(value["status"] == "COMPLETE_ACTUAL_NATURAL_STUDENT_TRAJECTORY_NOT_EVALUATION"
              and value["configuration"] == cfg["training_configuration"]
              and value["config_identity"] == identity and value["adapter_sources"] == student.sources()
              and value["runtime"] == student.RUNTIME and value["expected"] == horizon
              and training["expected_horizon"] == horizon
              and value["native_objective_scaling_unchanged"] is True
              and value["actual_no_skipped_optimizer_updates"] is True
              and value["downstream_evaluation_complete"] is False
              and value["scientific_improvement_demonstrated"] is False,
              "Root completion only: never S1, partial, skipped or relabeled START")
    selected = load(training["providers"]["selection"])
    records = selected["record_ids"]
    c.require(type(selected["count"]) is int and selected["count"] == len(records) == 66076
              and len(set(records)) == len(records), "Exact unique frozen whole-conversation selection")
    allowed = set(records)
    seen, bound, physical = [], [], []
    cursor, last_lr, last_accepted = 0, None, None
    c.require(value["attempts"], "Full historical accepted chain required")
    for ref in value["attempts"]:
        receipt = load(ref)
        c.require(receipt["configuration"] == cfg["training_configuration"]
                  and receipt["config_identity"] == identity
                  and receipt["adapter_sources"] == student.sources()
                  and receipt["runtime"] == student.RUNTIME, "Foreign accepted segment")
        status = receipt["status"]
        c.require(status in ("COMPLETE_STUDENT_TRAINING_NOT_EVALUATION",
                             "ACCEPTED_STUDENT_CONTINUATION_REQUIRES_ROOT_SAME_JOB_ACTION"),
                  "Failed or unaccepted segment cannot advance completion")
        attempt = Path(receipt["exposure"]["path"]).parent
        h = load(describe(attempt / "ACTUAL_HORIZON.json"))
        c.require(h["actual_resume_step"] == cursor and h["args_max_steps"] == -1
                  and h["native_max_steps"] == 516 and h["selected_count"] == 66076,
                  "Missing/repeated cursor or changed epoch")
        if status == "COMPLETE_STUDENT_TRAINING_NOT_EVALUATION":
            native = load(receipt["native_complete"])
            end, accepted_ref = native["global_step"], native["accepted"]
        else:
            end, accepted_ref = receipt["global_step"], receipt["accepted"]
        c.require(type(end) is int and cursor < end <= 516, "Nonincreasing accepted boundary")
        archived = receipt["accepted_archive"]
        c.require(all(archived[key] == accepted_ref[key] for key in ("bytes", "sha256")),
                  "Archived accepted metadata differs from original")
        accepted = load(archived)
        c.require(accepted["status"] == "ACCEPTED_FULL_NATIVE_STATE"
                  and accepted["identity"] == identity and accepted["global_step"] == end
                  and accepted["world_size"] == 1, "Exact unskipped native world1 accepted state")
        exposure = load(receipt["exposure"])
        ids = exposure["ordered_record_ids"]
        c.require(exposure["record_count"] == len(ids) == (end - cursor) * 128
                  and exposure["microbatches"] == (end - cursor) * 32
                  and len(set(ids)) == len(ids) and set(ids) <= allowed, "Exact unique segment exposure")
        lr = read_jsonl(attempt / "LEARNING_RATES.jsonl")
        validate_lr_trajectory(lr, cursor, end, 516)
        c.require(all(row["skipped_steps"] == 0 for row in lr)
                  and (last_lr is None or lr[0] == last_lr), "Unskipped continuous scheduler chain")
        objective = read_jsonl(attempt / "OBJECTIVE_UNITS.jsonl")
        c.require(len(objective) == (end - cursor) * 32
                  and [v for row in objective for v in row["record_ids"]] == ids
                  and all(row["microbatch_size"] == 4 and len(row["per_record_shifted_targets"]) == 4
                          and min(row["per_record_shifted_targets"]) > 0
                          and sum(row["per_record_shifted_targets"]) == row["expanded_shifted_targets"]
                          and row["declared_accumulation"] == 32
                          and math.isfinite(row["native_unscaled_loss"])
                          and math.isfinite(row["returned_native_scaled_loss"]) for row in objective),
                  "Original native objective and accumulation accounting")
        physical.append(physical_segment(ref, receipt, cfg, training, load, describe, bound))
        seen += ids
        cursor, last_lr, last_accepted = end, lr[-1], accepted_ref
        bound.append(ref)
    optimized = set(seen)
    tail = [record for record in records if record not in optimized]
    c.require(cursor == 516 and len(seen) == len(set(seen)) == 66048 and len(tail) == 28
              and value["actual_optimized_records"] == 66048
              and value["actual_optimizer_updates"] == 516
              and value["actual_unoptimized_tail_record_ids"] == tail
              and value["ordered_record_ids_sha256"] == student.key(seen)
              and value["accepted"] == last_accepted, "Full unique epoch, ordered exposure and exact native tail")
    final = load(value["accepted"])
    c.require(final["global_step"] == 516 and final["identity"] == identity
              and final["status"] == "ACCEPTED_FULL_NATIVE_STATE"
              and Path(value["accepted"]["path"]) == Path(training["output"]) / "training/checkpoint-516/ACCEPTED.json",
              "Late-bound exact final accepted checkpoint516")
    return training, complete_ref, value, physical
