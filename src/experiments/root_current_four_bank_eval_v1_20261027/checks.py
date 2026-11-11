"""V4 runtime gates stay stricter than pending afterok admission dependencies."""
from pathlib import Path
import re

from experiments.root_native_trained_loader_v4_20261026 import generation as bridge
from . import contract as c


def qualification_result(cfg, value, load):
    roster, qcfg = c.prerequisite_roster(cfg, load)
    c.require(value["status"] == "PASS_ACTUAL_NATIVE_TRAINED_ADAPTER_TWO_QA_FRESH_RELOAD"
              and value["configuration"] == cfg["qualified_loader_configuration"]
              and value["actual"]["job_id"] == roster["qualifier"]["job_id"]
              and value["sources"] == qcfg["sources"]
              and value["accepted_checkpoint"] == qcfg["accepted_checkpoint"]
              and value["adapter_active"] is True and value["adapter_merged"] is False
              and value["distinct_QAs"] == 2 and value["neural_outputs"] == 4
              and value["full_bank_scored"] is False,
              "Actual new V4 trained-loader PASS required, never a pending afterok or old failure")


def independent_result(cfg,value,qref,load):
    c.prerequisite_roster(cfg,load)
    contract,review,recorded,proof=c.materialized_reviewer(cfg,load)
    c.require(review==value and recorded==qref,'Runtime actual materialized review/readout differs from qualified inputs')
    return proof


def training_result(cfg, value, training):
    from experiments.root_fulltrajectory_student_v4_20261025 import protocol as root_training
    from experiments.root_s1_shared_student_v1_20261025 import protocol as s1_training

    protocol = s1_training if cfg["training_cell"]["family"] == "S1" else root_training
    c.require(value["status"] == "COMPLETE_ACTUAL_NATURAL_STUDENT_TRAJECTORY_NOT_EVALUATION"
              and value["configuration"] == cfg["training_configuration"]
              and value["config_identity"] == protocol.n.run_identity(training)
              and value["adapter_sources"] == protocol.sources()
              and value["runtime"] == protocol.RUNTIME
              and value["expected"] == protocol.horizon(66076)
              and value["actual_optimizer_updates"] == 516
              and value["actual_optimized_records"] == 66048
              and len(value["actual_unoptimized_tail_record_ids"])
              == len(set(value["actual_unoptimized_tail_record_ids"])) == 28
              and value["native_objective_scaling_unchanged"] is True
              and value["actual_no_skipped_optimizer_updates"] is True
              and value["downstream_evaluation_complete"] is False,
              "Exact original family, full native epoch and accepted exposure required")


def complete_student(cfg, load=c.checked, describe=c.desc):
    if cfg["training_cell"]["family"] == "Root":
        from . import historical
        return historical.complete_student(cfg, load=load, describe=describe)
    training = c.validate_configuration(cfg, load)
    path = c.literal(cfg["student_completion_path"])
    c.require(path == Path(training["report"]) / "ROOT_LONG/COMPLETE.json", "Original S1 collector location")
    ref = describe(path)
    value = load(ref)
    training_result(cfg, value, training)
    accepted = load(value["accepted"])
    c.require(accepted["status"] == "ACCEPTED_FULL_NATIVE_STATE"
              and accepted["global_step"] == 516 and accepted["identity"] == value["config_identity"]
              and Path(value["accepted"]["path"]) == Path(training["output"]) / "training/checkpoint-516/ACCEPTED.json",
              "Actual final S1 checkpoint516; paid partial5% states are not final models")
    return training, ref, value, []


def documents(binding, load, sources):
    cfg = load(binding["configuration"])
    training = c.validate_configuration(cfg, load)
    base, phase3 = load(cfg["official_POPE_configuration"]), load(cfg["phase3_configuration"])
    c.require(cfg["sources"] == c.augmented_sources(sources(base, phase3))
              and cfg["python"] == "/data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python"
              and all(cfg[field] is True for field in ("no_model_downloads", "no_base_copies", "no_START_journal_reuse")),
              "Exact new separately staged V4 consumer/source map and unchanged worker environment")
    review = load(binding["independent_source_config_review"])
    c.require(review["status"] == c.REVIEW_STATUS and review["independent"] is True
              and review["reviewer"] not in (None, "", "Session5")
              and review["configurations"][cfg["student_id"]] == binding["configuration"]
              and review["sources"] == cfg["sources"], "Root personal exact twelve-source/config review")
    for path, digest in cfg["sources"].items():
        c.require(c.u.sha((c.ROOT / path).read_bytes()) == digest, "Frozen consumer source changed: " + path)
    qref = c.desc(c.literal(cfg["required_actual_qualification_result_path"]))
    q = load(qref)
    qualification_result(cfg, q, load)
    peer = load(c.desc(c.literal(cfg["required_independent_qualification_review_path"])))
    independent_result(cfg, peer, qref, load)
    _, qcfg = c.prerequisite_roster(cfg, load)
    anchor = load(qcfg["training_configuration"])
    c.require(all(training["providers"][key] == anchor["providers"][key] for key in ("assets", "environment"))
              and training["recipe"] == anchor["recipe"], "Exact qualified model/processor/native recipe")
    claim, grant = load(binding["ownership"]), load(binding["storage_reservation"])
    c.require(binding["status"] == c.BINDING_STATUS and binding["owner"] == "Root"
              and binding["nonce"] == cfg["nonce"] == claim["nonce"]
              and claim["job_id"] == binding["job_id"] and claim["owner"] == "Root"
              and claim["scope"] == c.SCOPE and claim["configuration"] == binding["configuration"]
              and claim["training_configuration"] == cfg["training_configuration"]
              and claim["no_live_duplicate_checkpoint_evaluation"] is True,
              "Exact new sole twelve-consumer ownership")
    c.require(grant["configuration"] == binding["configuration"] and grant["output"] == cfg["output"]
              and grant["reserved_bytes"] == cfg["output_limit_bytes"]
              and grant["final_checkpoint_retained_for_evaluation"] is True
              and grant["checkpoint_retention"] == cfg["checkpoint_retention"],
              "Measured small output reservation and final-state retention")
    return cfg, base


def function(cfg, identity, roster, complete, bank):
    from experiments.root_native_trained_evaluation_v3_20261026 import worker as s1
    from experiments.root_fulltrajectory_student_v4_20261025 import protocol as root_training
    from experiments.root_s1_shared_student_v1_20261025 import protocol as s1_training

    value = s1.function(cfg, identity, roster, complete, bank)
    family = cfg["training_cell"]["family"]
    return dict(value, schema="current12-v4-final-four-bank-function-v1", role=c.SCOPE,
        training_family=family, generation_API_policy=bridge.POLICY,
        training_logical_protocol=s1_training.PACKAGE if family == "S1" else root_training.PACKAGE,
        physical_extension_protocol="UNCHANGED_ORIGINAL_S1" if family == "S1"
            else "experiments.root_control10_telemetry_physical_v1_20261026")


def acked_prerequisite(runtime, binding_ref, binding, cfg, role):
    folder = Path(binding_ref["path"]).parent
    ack_ref, ack = runtime.document(folder / "ACK.json")
    held_ref, held = runtime.document(folder / "VERIFIED_HELD.json")
    intent = runtime.checked(held["intent"])
    match = re.fullmatch(r"([1-9]\d*)(?:;[A-Za-z0-9_.-]+)?\s*", ack["stdout"])
    c.require(ack["returncode"] == 0 and match is not None and match[1] == binding["job_id"]
              and held["binding"] == binding_ref and held["job_id"] == binding["job_id"]
              and intent["owner"] == "Root" and intent["configuration"] == binding["configuration"]
              and intent["nonce"] == binding["nonce"],
              "Actual durable intent, unambiguous sbatch ACK and exact held physical binding required")
    configuration = runtime.checked(binding["configuration"])
    prefix = "dsmm-trained-eval-qual:" if role == "qualifier" else "dsmm-trained-qual-review:"
    name = "dsmm-native-trained-eval-qual-v4" if role == "qualifier" else "dsmm-trained-qual-review-v4"
    expected_comment = prefix + binding["nonce"]
    c.require("--hold" in intent["argv"] and "--partition=general" in intent["argv"]
              and "--gres=gpu:L40S:1" in intent["argv"]
              and "--comment=" + expected_comment in intent["argv"]
              and "--job-name=" + name in intent["argv"]
              and intent["argv"][-1] == binding_ref["path"],
              "Exact original Root-held General source/nonce request, not invented future afterok")
    recorded = c.u.fields(held["scheduler"])
    c.require(recorded["JobId"] == binding["job_id"] and recorded["JobName"] == name
              and recorded["UserId"] == "luyangz(2708899)" and recorded["Comment"] == expected_comment
              and recorded["WorkDir"] == configuration["source_root"]
              and recorded["JobState"] == "PENDING" and recorded["Reason"] == "JobHeldUser"
              and recorded["Restarts"] == "0" and recorded["RunTime"] == "00:00:00",
              "Captured original held actor does not match the real prerequisite binding")
    if role == "reviewer":
        dependencies = [arg for arg in intent["argv"] if arg.startswith("--dependency=")]
        c.require(dependencies in ([], ["--dependency=afterok:" + cfg["required_loader_qualifier_job_id"]]),
                  "Reviewer cannot consume a foreign or failed subject dependency")
    return dict(ack=ack_ref, verified_held=held_ref, intent=held["intent"],
                pending_dependency_is_not_runtime_PASS=True)
