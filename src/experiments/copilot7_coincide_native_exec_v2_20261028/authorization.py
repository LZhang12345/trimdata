"""Exact COINCIDE physical scope around the unchanged, already paid native engine."""
from contextlib import contextmanager
from contextvars import ContextVar
from . import protocol as p
from .registry import registered_plan
from experiments.root_fulltrajectory_student_v4_20261025 import shared_runtime as prior
from experiments.root_native_deterministic_attention_v2_20261025 import bridge

_CURRENT = ContextVar("reviewed_coincide_native_authorization_v2", default=None)
independent = p.independent


def natural_evidence(ref, cfg, load=p.checked):
    bundle = load(ref)
    raw, study = load(bundle["raw_readout"]), load(bundle["study"])
    peer = load(bundle["independent_review"])
    independent(peer)
    p.require(bundle["raw_readout"] == p.NATURAL_ANCHOR_READOUT
        and raw["status"] == "PASS_ACTUAL_DECLARED_EPOCH_RUNTIME_UNREVIEWED"
        and raw["driver_sources"] == p.qualification.sources() and raw["native_sources"] == cfg["sources"]
        and raw["study_identity"] == p.key(p.bare(study))
        and raw["paired_initialization_seed"] == study["seed"] == 17
        and raw["actual_same_job_requeue"] is True
        and peer["status"] == "PASS_INDEPENDENT_ACTUAL_DECLARED_EPOCH_RUNTIME_FOR_STUDENT_PROBE"
        and peer["raw_readout"] == bundle["raw_readout"] and peer["study"] == bundle["study"]
        and peer["driver_sources"] == p.qualification.sources(),
        "Genuine paid natural-epoch restart evidence; no new COINCIDE PASS fabricated")
    units = raw["epoch_and_objective_units"]
    p.require(units["passed"] is True and units["natural_epoch"] is True
        and units["selected_count"] == 1153 and units["actual_optimized_count"] == 1152
        and units["tail_count"] == 1 and units["no_tail_replacement_or_topup"] is True
        and set(units["actual_forwarded_sources"]) == {"coco", "gqa", "ocr_vqa", "vg", "textvqa", "text_only"}
        and set(raw["numerical_comparisons"]) == {"portable_weights", "Adam_FP32_master_scaler",
            "saved_RNG", "scheduler", "microbatch_loss_trajectory"}
        and all(row["pass_numeric"] is True for row in raw["numerical_comparisons"].values())
        and raw["pre_restart_reference_cut4"]["passed"] is True
        and raw["actual_update_cursor_checks"]["passed"] is True
        and all(row["all_pass"] is True for row in raw["criteria"].values())
        and raw["optimizer_groups_equal"] is True
        and raw["semantic_learning_rate_trajectories"]["passed"] is True,
        "Complete qualified mask/objective/LR/optimizer/RNG/restart proof required")
    reference = load(study["native_configurations"]["reference"])
    p.require(reference["sources"] == cfg["sources"] and reference["recipe"] == cfg["recipe"]
        and all(reference["providers"][name] == cfg["providers"][name] for name in ("environment", "assets"))
        and all(reference["allocation"][name] == cfg["allocation"][name]
                for name in ("gpus", "microbatch", "cpus", "memory_mb", "loader_workers")),
        "Identical qualified engine, original Stage1 assets, runtime and numerical allocation")
    return bundle


def verify(cfg, providers, profile, permit_ref, load=p.checked):
    p.supported(cfg)
    permit, gate = load(permit_ref), providers["qualification"]
    registry = registered_plan(cfg["student_candidate_plan"], load)
    rows = [row for row in registry["cells"] if row["id"] == cfg["id"]]
    p.require(len(rows) == 1 and rows[0]["new_training_required"] is True
        and rows[0]["duplicate_of"] is None and rows[0]["selection"] == cfg["providers"]["selection"],
        "Actual accepted missing-only COINCIDE selection and seed")
    p.require(profile["site"] == "babel" and profile["partition"] == "general" and profile["qos"] == "normal"
        and profile["account"] == "cx" and profile["gpu_type"] == "L40S"
        and profile["code_root"] == str(p.ROOT) and profile["output_root"] == p.CHECKPOINT_ROOT,
        "Actual General L40S code/checkpoint profile")
    p.require(gate["status"] == p.GATE and gate["ordinary_native_launch_forbidden"] is True
        and gate["requires_separate_outer_authorization"] is True and gate["native_sources"] == cfg["sources"]
        and gate["recipe"] == cfg["recipe"] and gate["runtime"] == p.RUNTIME
        and gate["student_adapter"] == dict(package=p.PACKAGE, sources=p.sources())
        and cfg["execution_adapter"] == dict(package=p.PACKAGE, sources=p.sources(), runtime=p.RUNTIME,
            natural_epoch=True, ordinary_native_launch_forbidden=True),
        "Exact new COINCIDE execution gate; old21/18/A physical bindings do not authorize it")
    natural_evidence(gate["qualification_evidence"], cfg, load)
    prior.anchor(load)
    reuse = load(gate["runtime_reuse_review"])
    independent(reuse)
    p.require(reuse["status"] == "PASS_INDEPENDENT_EXACT_COINCIDE_NATIVE_PAID_RUNTIME_REUSE_V2"
        and reuse["anchor_sources"] == p.anchor.sources() and reuse["target_sources"] == p.sources()
        and reuse["native_sources"] == cfg["sources"] and reuse["numerical_bodies"] == p.numerical_bodies()
        and reuse["longwall_readout"] == prior.ANCHOR_READOUT and reuse["longwall_review"] == prior.ANCHOR_REVIEW
        and reuse["no_old_checkpoint_identity_migration"] is True,
        "Independent exact-source paid-runtime reuse; no checkpoint/qualification identity migration")
    gpu = load(reuse["actual_GPU_operator_qualification"])
    peer = load(reuse["actual_GPU_operator_review"])
    independent(peer)
    packet = load(gpu["binding"]["source_packet"])
    p.require(gpu["status"] == "PASS_ACTUAL_GPU_LAYOUT_TELEMETRY_EQUIVALENCE_V1"
        and gpu["distinct_planned_layouts"] > 512 and gpu["late_unsupported_rejected"] is True
        and gpu["deterministic_flag_reset_rejected"] is True
        and all(packet["sources"].get(name) == digest for name, digest in bridge.sources().items())
        and peer["status"] == "PASS_INDEPENDENT_ACTUAL_LAYOUT_GPU_QUALIFICATION_V1"
        and peer["readout"] == reuse["actual_GPU_operator_qualification"],
        "Existing exact GPU attention/layout qualification, not another model/operator acquisition")
    p.require(permit["schema"] == "coincide-fixed10-native-v2-authorization-v1" and permit["owner"] == "Root"
        and load(permit["configuration"]) == cfg and permit["config_identity"] == p.n.run_identity(cfg)
        and permit["adapter_sources"] == p.sources() and permit["phase"] in ("shared_canary", "production")
        and permit["runtime"] == p.RUNTIME and permit["expected_horizon"] == p.horizon(p.K)
        and cfg.get("qualification_max_steps") is None,
        "Actual new Root-owned permit: full natural516, no scientific step cap")
    review = load(permit["review"])
    independent(review)
    p.require(review["status"] == "PASS_INDEPENDENT_EXACT_COINCIDE_NATIVE_EXECUTION_POLICY_V2"
        and review["policy_identity"] == permit["policy_identity"] and review["sources"] == p.sources()
        and review["configuration"] == permit["configuration"],
        "Genuine independent exact-source/config/policy review")
    return permit


@contextmanager
def scope(config, providers, profile, permit_ref, load=p.checked):
    permit = verify(config, providers, profile, permit_ref, load)
    token = _CURRENT.set((p.n.run_identity(config), permit))
    from . import runtime
    proof = dict(schema="REVIEWED_COINCIDE_OBSERVATIONAL_NATIVE_RUNTIME_V2",
        configuration=permit["configuration"], native_identity=p.n.run_identity(config),
        physical_binding=permit["physical_binding"],
        runtime_reuse_review=providers["qualification"]["runtime_reuse_review"],
        executed_sources=p.sources(), native_sources=config["sources"], runtime=p.RUNTIME,
        checkpoint_identity_migration=False, math_body_changed=False)
    try:
        with runtime.extension(proof):
            yield permit
    finally:
        _CURRENT.reset(token)


def require_current(config, providers, profile):
    value = _CURRENT.get()
    p.require(value is not None and value[0] == p.n.run_identity(config)
              and providers["qualification"]["status"] == p.GATE,
              "No exact new COINCIDE execution scope; legacy native launch stays closed")
    return value[1]
