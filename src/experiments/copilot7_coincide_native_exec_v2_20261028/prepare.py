"""Root-owned registry CLI; source-only packet mode never accesses DATA."""
import argparse
import json
import os
from pathlib import Path
import tarfile

from . import protocol as p, registry, paid
from experiments.copilot7_coincide_actual_review_v2_20261028.prepare import source_closure

CONTROL = Path(__file__).resolve().parents[2] / "collaboration/session7-coincide-actual-output-review-20261028/v2/native-consumer"


def packet(output=CONTROL):
    output = Path(output)
    logical = p.sources()
    full = source_closure([p.ROOT / name for name in logical])
    archive = output / "NATIVE_EXECUTABLE_SOURCE_V2.tar.gz"
    output.mkdir(parents=True, exist_ok=True)
    p.require(not archive.exists(), "Never replace an existing frozen native archive")
    with tarfile.open(archive, "w:gz") as sink:
        for name in full:
            sink.add(p.ROOT / name, arcname=name, recursive=False)
    configs = []
    for seed in p.SEEDS:
        configs.append(dict(id=p.cell_id("coincide", seed), method="coincide", fraction="0.10",
            training_seed=seed, selection_seed=1234, K=p.K, expected_horizon=p.horizon(p.K),
            output=str(Path(p.CHECKPOINT_ROOT) / p.cell_id("coincide", seed)),
            report=str(p.REPORT_ROOT / p.cell_id("coincide", seed)),
            resources=p.RESOURCES, proposed_request_GPUh=12, actual_job=None,
            entry_module=p.PACKAGE + ".driver", exact_native_cli=p.PACKAGE + ".student_config.cli",
            training_engine="experiments.root_native_campaign_v7_20261024.native.main",
            Root_admission_only=True))
    roster = p.seal(output / "FINITE_THREE_NATIVE_CELLS_V2.json", dict(
        scope="FIXED10_COINCIDE_ONLY", cells=configs, paired_seeds=list(p.SEEDS),
        requested_GPUh_if_Root_admits=36, actual_submissions=0, actual_GPU_seconds=0,
        adaptive_budget_or_target_tuning=False, no_padding=True,
        adaptation_label=p.ADAPTATION))
    source = p.seal(output / "NATIVE_EXECUTABLE_SOURCE_MANIFEST_V2.json", dict(
        schema="s7-native-executable-source-v2", package=p.PACKAGE,
        adapter_sources=logical, full_static_closure=full, native_math_sources=p.n.sources(),
        exact_five_qualified_observer_bodies=p.numerical_bodies(),
        source_root=str(p.SOURCE_ROOT), source_archive=p.desc(archive),
        uses_same_native_cli_function_object=True, existing_paid_inputs=paid.inventory(),
        finite_roster=roster, all_actual_IO_still_requires_Root_owned_registration=True,
        mathematical_training_engine_changed=False, new_model_loader_for_training=False,
        executed_General_jobs=0, actual_new_GPU_or_training_result=None))
    return p.seal(output / "NATIVE_EXECUTABLE_HANDOFF_V2.json", dict(
        status="IMPLEMENTED_EXECUTABLE_NATIVE_SOURCE_REQUIRES_REAL_SELECTION_STORAGE_REVIEW_BINDINGS",
        source=source, source_archive=p.desc(archive), finite_roster=roster,
        registry_command=["-m", p.PACKAGE + ".prepare", "--registry-output", "REAL_NEW_HOME_REGISTRY_PATH",
            "--execution-registration", "REAL_ROOT_METADATA_REGISTRATION",
            "--selection", "REAL_V2_ACCEPTED_SELECTION", "--materialized", "REAL_MATERIALIZED_CANDIDATE",
            "--actual-output-review", "REAL_REGISTERED_OUTPUT_REVIEW", "--registration", "REAL_ROOT_REGISTRY_ACCEPTANCE",
            "--eligibility", "REAL_FROZEN_ELIGIBILITY", "--environment", paid.ENVIRONMENT["path"],
            "--assets", paid.ASSETS["path"]],
        source_only_execution_command=["python3", "-B", "-m", p.PACKAGE + ".driver", "--entry-contract"],
        exact_Root_materialization_entry=p.PACKAGE + ".materialize",
        exact_Root_request_and_postACK_entry=p.PACKAGE + ".bind",
        actual_runtime_child=p.PACKAGE + ".child",
        genuine_missing_inputs=["actual independently accepted selection/materialization", "current native profile/cache19 cold readiness",
            "Root genuine three separate21GiB storage grants and current collective ledger", "Root exclusive three function claims",
            "independent exact native source/config/policy and paid-runtime reuse acceptance",
            "Root actual admission lock/census/intent/ACK/spool/release"],
        old_bound_sources_changed=False, no_operator_or_feature_reacquisition=True,
        accepted_partial_is_not_full516=True,
        first36GPUh_is_requested_not_consumed=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-only", action="store_true")
    parser.add_argument("--output", type=Path, default=CONTROL)
    parser.add_argument("--registry-output", type=Path)
    parser.add_argument("--execution-registration", type=Path)
    for name in ("selection", "materialized", "actual-output-review", "registration", "environment", "assets", "eligibility"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.packet_only:
        value = packet(args.output)
    else:
        names = ("selection", "materialized", "actual_output_review", "registration", "environment", "assets", "eligibility")
        p.require(os.getuid() == 2708899 and args.registry_output is not None
                  and args.execution_registration is not None and all(getattr(args, name) for name in names),
                  "Actual existing registry inputs and owned metadata registration")
        from experiments.root_production_student_adapter_v1_20261024.launch import require_metadata_registration
        require_metadata_registration(p.desc(args.execution_registration))
        value = p.seal(args.registry_output, registry.prepare(
            *(p.desc(getattr(args, name)) for name in names)))
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
