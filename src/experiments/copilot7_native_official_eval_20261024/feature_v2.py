"""Exact DT32 engineering dependencies; no change to the native numerical kernel."""
import argparse
from pathlib import Path
import sys
import types

from . import common as c
from . import feature_v1 as prior


MODEL_ROLES = ("vicuna", "clip336", "stage1_projector")
ENGINEERING_POOL_SHA256 = "fa016ff3347117432ee7034633f67420c7760f4ad79f0bdd61bc8f5a0f318e47"
ENGINEERING_RECORDS_SHA256 = "3cf04ac5cb0b74c39de779ef94250bb4a3fb7d27fc09eb36ff6f16d4a9db3e90"
PREPARATION_OUTPUT = Path("/data/user_data/luyangz/native_eval_session7_20261024/native15-prep-v2")
CONTROL_RELATIVE = Path("collaboration/session7-native-official-execution-20261024/datatailor32-v2")
DATASET_REFERENCE = dict(
    repository="liuhaotian/LLaVA-Instruct-150K",
    revision="9d451dc7629cfe0469f6ae4432b765cd603d5fcb",
    filename="llava_v1_5_mix665k.json",
    bytes=1029887963,
    sha256="ce959ce6e23073ee1cd1a8a2ef1c633768c10d4174327b8b2dc7113b91af6cf8",
)
INPUT_SCOPE = dict(
    required_physical_model_roles=list(MODEL_ROLES),
    required_physical_dataset="EXACT_ROOT_ENGINEERING1024",
    original665k_metadata="PINNED_PROVENANCE_REFERENCE_NOT_OPENED_BY_ENGINEERING32",
    original665k_reference=DATASET_REFERENCE,
    full_pool_readiness=False,
    student_release_authority=False,
)


def validate_inputs(spec):
    from experiments.root_native_campaign_v3_20261023.config import ASSET_PINS
    from experiments.root_native_campaign_v3_20261023 import common as native

    c.require(spec["phase"] == "datatailor32" and spec["site"] == "babel"
              and spec["input_dependency_scope"] == INPUT_SCOPE,
              "V2 is only the explicitly scoped Babel DT32 engineering qualification")
    inputs = spec["inputs"]
    c.require(inputs["pool"]["sha256"] == ENGINEERING_POOL_SHA256,
              "Only the previously qualified exact Root engineering1024 pool is permitted")
    pool = c.checked(inputs["pool"])
    c.require(pool["records_json"]["sha256"] == ENGINEERING_RECORDS_SHA256,
              "Exact original engineering record bytes required")
    rows = c.checked(pool["records_json"])
    ids = pool["record_ids"]
    c.require(pool["engineering_only"] is True and pool["count"] == len(rows) == len(ids) == 1024
              and len(set(ids)) == len(pool["source_indices"]) == 1024,
              "Exact original1024 engineering construction required, not a scientific pool")
    c.require(ids == [f'{index:06d}:{row.get("id", "")}' for index, row in zip(pool["source_indices"], rows)],
              "Original engineering source-index identity differs")
    c.checked(pool["protected_exclusions"])
    c.checked(pool["image_manifest"])
    eligibility = c.checked(inputs["eligibility"])
    c.require(eligibility["pool"] == inputs["pool"] and eligibility["missing_media_count"] == 0
              and eligibility["protected_QA_image_RGB_overlap_count"] == 0,
              "Matching engineering media/protection evidence required")
    assets = c.checked(inputs["assets"])
    c.require(assets["status"] == "PASS_ASSETS" and assets["official_commit"] == native.PIN
              and set(assets["assets"]) == set(ASSET_PINS),
              "Exact original native asset receipt required")
    c.require(spec["cache_root"] == "/data/hf_cache", "Use the approved shared asset cache")
    for role, pin in ASSET_PINS.items():
        asset = assets["assets"][role]
        c.require((asset["repository"], asset["revision"]) == pin,
                  "Native model/projector/dataset reference revision differs: " + role)
        c.require(native.within(asset["snapshot"], spec["cache_root"]),
                  "Asset snapshot outside the approved shared cache")
    metadata = assets["assets"]["LLaVA665K"]["files"]
    c.require(len(metadata) == 1
              and Path(metadata[0]["path"]).name == DATASET_REFERENCE["filename"]
              and all(metadata[0][key] == DATASET_REFERENCE[key] for key in ("bytes", "sha256")),
              "The original665298 metadata provenance reference must remain exact")
    for role in MODEL_ROLES:
        asset = assets["assets"][role]
        c.require(asset["files"], "Empty required native model asset: " + role)
        for description in asset["files"]:
            c.require(Path(description["path"]).absolute().is_relative_to(Path(asset["snapshot"]).absolute())
                      and native.within(description["path"], spec["cache_root"]),
                      "Required model file outside its pinned snapshot or approved cache")
            c.require(c.desc(description["path"]) == {key: description[key] for key in ("path", "bytes", "sha256")},
                      "Required model/projector bytes differ; no fallback download")
    environment = c.checked(inputs["environment"])
    c.require(Path(sys.executable).absolute() == Path(environment["python_executable"]).absolute(),
              "Use the existing exact isolated native interpreter")
    c.require(environment["versions"]["torch"].startswith("2.1.2")
              and environment["versions"]["transformers"] == "4.37.2",
              "Actual native operator runtime receipt differs")
    return pool, rows, ids, environment


def collect_preparation(request, observed):
    binding = c.checked(request["binding"])
    c.require(binding["job_id"] == request["job_id"] == "10909981" and binding["owner"] == c.OWNER,
              "Only the exact completed owned native15 preparation is collected")
    configuration = c.checked(binding["configuration"])
    root = Path(configuration["output"])
    c.require(str(root) == request["output"] and root == PREPARATION_OUTPUT,
              "Unexpected prior preparation output")
    complete_path = root / "PREPARATION_COMPLETE.json"
    c.require(complete_path.stat().st_size <= 8 << 20, "Oversized prior preparation metadata")
    complete = c.read(complete_path)
    c.require(complete["schema"] == "s7-native15-preparation-result-v2"
              and complete["status"] == "COMPLETE_SCOPED_PREPARATION_WITH_EXPLICIT_DATASET_GAPS"
              and complete["actual"]["job_id"] == binding["job_id"]
              and complete["configuration"] == binding["configuration"]
              and complete["native_dataset_or_inference_or_scoring_qualified"] is False,
              "Prior preparation identity or honest scope differs")
    original_attempt = root / "attempts" / "10909981-r0"
    result = c.read(original_attempt / "RESULT.json")
    c.require(result == complete, "Global preparation completion differs from its successful attempt")
    destination = Path(request["export_root"]) / (observed["job_id"] + "-r" + str(observed["restart"]))
    approved = c.ROOT.parent / CONTROL_RELATIVE
    c.require(request["approved_control_root"] == str(approved)
              and destination.resolve().is_relative_to(approved.resolve()),
              "Preparation metadata export outside the registered control root")
    exported = {}
    for name, description in (
        ("PREPARATION_COMPLETE.json", c.desc(complete_path)),
        ("RESULT.json", c.desc(original_attempt / "RESULT.json")),
        ("CACHE_INVENTORY.json", complete["inventory"]),
        ("ENVIRONMENT_PROBE.json", complete["environment_probe"]),
    ):
        c.require(Path(description["path"]).resolve().is_relative_to(root.resolve())
                  and description["bytes"] <= 8 << 20, "Foreign or oversized prior preparation metadata")
        value = c.checked(description)
        exported[name] = dict(source=description, exported=c.seal(destination / name, value))
    return c.seal(destination / "COLLECTION.json", dict(
        status="ACTUAL_OWNED_GPU_STEP_PREPARATION_METADATA_EXPORT",
        actual=observed, original_binding=request["binding"], exported=exported,
        native15_dataset_or_inference_or_scoring_qualified=False,
    ))


def run(binding_path):
    def validated(spec):
        pool, rows, ids, environment = validate_inputs(spec)
        job, restart = prior.os.environ["SLURM_JOB_ID"], prior.os.environ["SLURM_RESTART_COUNT"]
        attempt = Path(spec["output"]) / ("s7-attempt-" + job + "-r" + restart)
        c.seal(attempt / "INPUT_DEPENDENCY_SCOPE.json", dict(
            scope=INPUT_SCOPE, actual_job=job, restart=int(restart), inputs=spec["inputs"],
            physical_model_bytes_verified=True, original665k_metadata_body_opened=False,
            no_production_validator_changed=True,
        ))
        return pool, rows, ids, environment

    binding = c.read(binding_path)
    spec = c.checked(binding["configuration"])
    c.verify_sources(spec["sources"])
    c.verify_sources(spec["dependencies"])
    review = c.checked(binding["review"])
    c.require(review["status"] == "PASS_EXACT_S7_PREREQUISITE_SOURCE_AND_CONFIG"
              and review["configuration"] == binding["configuration"] and review["sources"] == spec["sources"],
              "Independent exact V2 source/config review required before DATA access")
    observed = prior.actual(binding, spec)
    collection = collect_preparation(spec["preparation_readout"], observed)
    c.require(c.checked(collection)["native15_dataset_or_inference_or_scoring_qualified"] is False,
              "Metadata collection must not invent native15 readiness")
    runner = types.FunctionType(prior.run.__code__, dict(vars(prior), validate_inputs=validated))
    return runner(binding_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.binding))
