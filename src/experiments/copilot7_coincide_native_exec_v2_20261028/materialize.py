"""Executable finite native config construction; Root binds real gates, then admits."""
import argparse
import os
from pathlib import Path

from . import protocol as p, plan as plans, reservations
from .registry import registered_plan

REPORT_ROOT = p.REPORT_ROOT
MINUTES = 720


def make_plan(registry_ref, profile_ref, load=p.checked):
    registry = registered_plan(registry_ref, load)
    profile = load(profile_ref)
    p.require(profile["site"] == "babel" and profile["partition"] == "general"
              and profile["qos"] == "normal" and profile["account"] == "cx"
              and profile["gpu_type"] == "L40S" and profile["code_root"] == str(p.ROOT)
              and profile["output_root"] == p.CHECKPOINT_ROOT,
              "Actual General-only qualified native profile, no old output/root alias")
    cells = [dict(row, site="babel", role="student", fraction="0.10", selection_seed=1234,
        gpus=1, microbatch=4, providers=dict(pool=registry["pool"], eligibility=registry["eligibility"],
            selection=row["selection"], **registry["initialization_providers"])) for row in registry["cells"]]
    return dict(schema="coincide-fixed10-native-v2-plan-v1", owner="Root",
        accepted_registry=registry_ref, sources=p.n.sources(), adapter_sources=p.sources(),
        recipe=p.native_config.RECIPE, site_profile=profile_ref, cells=cells,
        selected_K=p.K, selection_seed=1234, proposed_single_attempt_GPU_hours=36,
        actual_jobs_submitted=0, adaptation_label=p.ADAPTATION,
        timing_policy="Same reviewed native10percent720min envelope; not an observed COINCIDE fit time",
        no_padding_to_GPU_hour_target=True)


def configuration(plan_ref, identifier, ownership, storage, qualification, compatibility, load=p.checked):
    plan = load(plan_ref)
    row = plans.get_cell(plan, identifier, load)
    for ref in (ownership, storage, qualification, compatibility):
        p.require(ref is not None, "Actual existing ownership/storage/runtime evidence required")
        load(ref)
    cfg = dict(schema="native-selection-run-v1", id=identifier, owner="Root", role="student",
        method="coincide", fraction="0.10", training_seed=row["training_seed"], selection_seed=1234,
        official_commit=p.n.PIN, sources=p.n.sources(), recipe=p.native_config.RECIPE,
        plan=plan_ref, student_candidate_plan=plan["accepted_registry"], site_profile=plan["site_profile"],
        providers=dict(row["providers"], ownership=ownership, qualification=qualification),
        allocation=dict(gpus=1, cpus=8, memory_mb=65536, microbatch=4, loader_workers=0,
                        minutes=MINUTES, reserve_seconds=300),
        output=str(Path(p.CHECKPOINT_ROOT) / identifier), report=str(REPORT_ROOT / identifier),
        checkpoint_seconds=1200, retained_accepted_checkpoints=2, storage_reservation=storage,
        replication_block=dict(replicate_site="babel"), qualification_max_steps=None,
        runtime_compatibility=compatibility, expected_horizon=p.horizon(p.K),
        execution_adapter=dict(package=p.PACKAGE, sources=p.sources(), runtime=p.RUNTIME,
                               natural_epoch=True, ordinary_native_launch_forbidden=True),
        adaptation_label=p.ADAPTATION, review=None)
    p.supported(cfg)
    reservations.validate(load(storage), cfg, load(cfg["providers"]["pool"]), load(cfg["providers"]["selection"]))
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("registry", "profile", "plan", "ownership", "storage", "qualification", "compatibility", "review"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--execution-registration", type=Path, required=True)
    parser.add_argument("--cell")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    p.require(os.getuid() == 2708899, "Actual Root materialization user")
    from experiments.root_production_student_adapter_v1_20261024.launch import require_metadata_registration
    require_metadata_registration(p.desc(args.execution_registration))
    if args.cell:
        p.require(all(getattr(args, name) is not None for name in
                      ("plan", "ownership", "storage", "qualification", "compatibility", "review")),
                  "Existing actual cell inputs and independent review required")
        cfg = configuration(p.desc(args.plan), args.cell, p.desc(args.ownership), p.desc(args.storage),
                            p.desc(args.qualification), p.desc(args.compatibility))
        review_ref = p.desc(args.review)
        review = p.checked(review_ref)
        p.independent(review)
        p.require(review["status"] == "PASS_INDEPENDENT_EXACT_SOURCE_AND_CONFIG"
                  and review["sources"] == cfg["sources"]
                  and review["config_identity"] == p.n.run_identity(cfg),
                  "Actual exact config/source review; review excluded from native identity")
        cfg["review"] = review_ref
        value = cfg
    else:
        p.require(args.registry is not None and args.profile is not None, "Actual accepted registry and profile")
        value = make_plan(p.desc(args.registry), p.desc(args.profile))
    print(p.canonical(p.seal(args.output, value)).decode())


if __name__ == "__main__":
    main()
