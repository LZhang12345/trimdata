"""Full516 actual parent and all accepted exposure; afterok alone is insufficient."""
from pathlib import Path
from . import protocol as p
from experiments.copilot7_coincide_native_exec_v2_20261028 import collect
from experiments.s1_eval_parent_bridge_v1_20261027.parents import accounting


def complete(cfg, binding, output, load=p.checked, describe=p.desc, account=accounting, publish=p.native.seal):
    training = load(cfg["training_configuration"])
    p.native.supported(training)
    physical = load(cfg["training_binding"])
    p.require(physical["owner"] == "Root" and physical["configuration"] == cfg["training_configuration"]
        and physical["sources"] == p.native.sources()
        and physical["job_id"] == cfg["required_training_job_id"],
        "Exact actual original COINCIDE parent binding")
    history = account(physical["job_id"], "COMPLETED", "0:0")
    ref = describe(cfg["student_completion_path"])
    value = load(ref)
    tail = value["actual_unoptimized_tail_record_ids"]
    p.require(value["status"] == "COMPLETE_ACTUAL_NATURAL_STUDENT_TRAJECTORY_NOT_EVALUATION"
        and value["configuration"] == cfg["training_configuration"]
        and value["config_identity"] == p.native.n.run_identity(training)
        and value["adapter_sources"] == p.native.sources() and value["runtime"] == p.native.RUNTIME
        and value["expected"] == p.native.horizon(66076) and value["actual_optimizer_updates"] == 516
        and value["actual_optimized_records"] == 66048 and len(tail) == len(set(tail)) == 28
        and value["native_objective_scaling_unchanged"] is True
        and value["actual_no_skipped_optimizer_updates"] is True
        and value["downstream_evaluation_complete"] is False,
        "Actual full natural516/66048/28 completion; no partial75/header-only or afterok inference")
    accepted = load(value["accepted"])
    p.require(accepted["status"] == "ACCEPTED_FULL_NATIVE_STATE" and accepted["global_step"] == 516
        and accepted["identity"] == value["config_identity"]
        and Path(value["accepted"]["path"]) == Path(training["output"]) / "training/checkpoint-516/ACCEPTED.json",
        "Exact final native capsule; actual payloads are revalidated by the trained loader")
    peer = load(binding["independent_completed_parent_review"])
    p.native.independent(peer)
    p.require(peer["status"] == "PASS_INDEPENDENT_ACTUAL_COINCIDE_COMPLETE516_PARENT_V2"
        and peer["training_configuration"] == cfg["training_configuration"]
        and peer["student_completion"] == ref and peer["actual_job_id"] == physical["job_id"]
        and peer["adapter_sources"] == p.native.sources()
        and peer["actual_full_accepted_history_verified"] is True,
        "Independent actual completed-parent acceptance, not only an exit code")
    recheck = collect.collect(cfg["training_configuration"], value["attempts"],
                              Path(output) / "RECOMPUTED_COMPLETION.json", load=load)
    observed = load(recheck)
    p.require(observed == value, "Actual unchanged collector/exposure/LR/objective readout differs")
    proof = publish(Path(output) / "PARENT_ACCEPTANCE.json", dict(
        status="ACTUAL_COINCIDE_FULL516_PARENT_VERIFIED_NOT_MODEL_FORWARD",
        configuration=cfg["training_configuration"], completion=ref, accepted=value["accepted"],
        independent_parent_review=binding["independent_completed_parent_review"], accounting=history,
        recomputed_completion=recheck, no_parent_state_or_checkpoint_changes=True, new_NN=0))
    return training, ref, value, [proof]
