"""Read-only Tiny8 raw-evidence collection inside a separately owned GPU allocation."""
import argparse
import hashlib
import math
from pathlib import Path
import sqlite3
import struct
import subprocess

from . import common as c
from . import feature_v1 as prior


PARENT_JOB = "2249668"
CAMPAIGN = Path("/scratch/user/u.lz341613/dsmm_native_campaign_20261023")
PARENT_OUTPUT = CAMPAIGN / "session7-native-prerequisites/tiny8-v1/output"
PARENT_SPEC_SHA256 = "e84117c52b219c26c28128211d2a8d59d1c524b26b2ac8cd20da99f106c65545"


def summarize_database(description, expected_ids, identity):
    c.require(c.desc(description["path"]) == description, "Completed Tiny feature database changed")
    database = sqlite3.connect(Path(description["path"]).absolute().as_uri() + "?mode=ro", uri=True)
    try:
        metadata = dict(database.execute("SELECT key,value FROM metadata"))
        c.require(metadata == dict(identity=identity, method="coincide", roster_sha256=c.key(expected_ids)),
                  "Tiny database measurement/roster ownership differs")
        rows = list(database.execute("SELECT record_id,vector,info,tau,length,nan_count,identity FROM features"))
    finally:
        database.close()
    c.require(len(rows) == len(expected_ids) == 8 and {row[0] for row in rows} == set(expected_ids),
              "Exact complete eight-record Tiny roster required")
    summarized = {}
    for identifier, blob, info, tau, length, nans, actual_identity in rows:
        c.require(actual_identity == identity and len(blob) == 20480 * 2 and info is None and tau is None,
                  "Wrong Tiny vector function, dtype or dimension")
        values = [value[0] for value in struct.iter_unpack("<e", blob)]
        c.require(all(math.isfinite(value) for value in values) and 0 <= nans <= 20480,
                  "Nonfinite or corrupt author-NaN-repaired Tiny vector")
        summarized[identifier] = dict(
            dimension=20480, dtype="<f2", vector_sha256=hashlib.sha256(blob).hexdigest(),
            author_replaced_NaNs=nans, L2=math.sqrt(math.fsum(value * value for value in values)),
            nonzero_coordinates=sum(value != 0 for value in values), length=length,
        )
    return {identifier: summarized[identifier] for identifier in expected_ids}


def collect(spec, actual):
    binding = c.checked(spec["parent_binding"])
    c.require(binding["job_id"] == PARENT_JOB and binding["owner"] == c.OWNER
              and binding["configuration"]["sha256"] == PARENT_SPEC_SHA256,
              "Only the exact completed owned Tiny8 prerequisite is collected")
    configuration = c.checked(binding["configuration"])
    c.require(configuration["phase"] == "tiny8" and configuration["site"] == "aces"
              and Path(configuration["output"]) == PARENT_OUTPUT, "Parent Tiny construction/site differs")
    c.verify_sources(configuration["sources"], Path(configuration["code_root"]))
    c.verify_sources(configuration["dependencies"], Path(configuration["code_root"]))
    parent_review = c.checked(binding["review"])
    c.require(parent_review["status"] == "PASS_EXACT_S7_PREREQUISITE_SOURCE_AND_CONFIG"
              and parent_review["configuration"] == binding["configuration"], "Parent exact source review differs")
    scheduler = subprocess.check_output([
        "sacct", "-D", "-X", "-j", PARENT_JOB, "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,NodeList",
        "-n", "-P",
    ], text=True, timeout=30)
    rows = [line.split("|") for line in scheduler.splitlines() if line.startswith(PARENT_JOB + "|")]
    c.require(len(rows) == 1 and rows[0][1:3] == ["COMPLETED", "0:0"],
              "Collect only the terminal successful exact Tiny prerequisite")
    s7_attempt = PARENT_OUTPUT / ("s7-attempt-" + PARENT_JOB + "-r0")
    native_attempt = PARENT_OUTPUT / ("attempt-" + PARENT_JOB + "-r0")
    s7_result = c.read(s7_attempt / "RESULT.json")
    native_result = c.read(native_attempt / "RESULT.json")
    exact = c.read(s7_attempt / "EXACT_AUTHOR_BATCH.json")
    native_config = c.read(s7_attempt / "DERIVED_NATIVE_CONFIG.json")
    expected_ids = exact["exact_batch"]["record_ids"]
    c.require(s7_result["native_return_code"] == 0 and s7_result["phase"] == "tiny8"
              and s7_result["actual"]["job_id"] == PARENT_JOB
              and native_config["exact_batch"] == exact["exact_batch"]
              and native_result["status"] == "COMPLETE_ACTUAL_TINY_BATCH8_ENGINEERING_EVIDENCE"
              and native_result["record_ids"] == expected_ids and native_result["records"] == 8
              and native_result["newly_acquired_records"] == 8 and native_result["cached_records"] == 0
              and native_result["qualification_promoted"] is False,
              "Raw Tiny completion/function/count identity differs")
    padding = c.read(native_attempt / "ACTUAL_BATCH_PADDING.json")
    frozen = c.read(PARENT_OUTPUT / "engineering-evidence/FROZEN_PARAMETERS.json")
    c.require(padding["records"] == 8 and padding["native_tokens_and_labels_preserved"] is True
              and len(set(padding["original_token_lengths"])) > 1
              and any(value > 0 for value in padding["padded_positions"]),
              "Actual unequal-padding author peers were not demonstrated")
    c.require(frozen["before"] == frozen["after"] and frozen["actual_checked"] is True
              and frozen["optimizer"] is None and frozen["updates"] == 0 and frozen["full_pool_qualified"] is False,
              "Actual frozen Tiny parameter evidence differs")
    media_path = s7_attempt / "media/RUNTIME_IMAGES.json"
    media = c.read(media_path)
    claim_path = s7_attempt / "TRANSPORT_CLAIM.json"
    claim = c.read(claim_path)
    c.require(media["status"] == "PASS_BYTE_IDENTICAL_RUNTIME_IMAGES" and media["scope"] == "ENGINEERING_ONLY"
              and media["job_id"] == PARENT_JOB and media["node"] == s7_result["actual"]["node"]
              and media["actual"]["restart"] == 0 and media["logical_pool"] == configuration["inputs"]["pool"]
              and media["ownership"] == c.desc(claim_path) and claim["job_id"] == PARENT_JOB
              and 0 < len(media["assets"]) <= 8 and media["scientific_pool_readiness_claim"] is False,
              "Historical current-parent image/RGB transport evidence differs")
    c.require(Path(native_result["database"]["path"]) == PARENT_OUTPUT / "features.sqlite3",
              "Foreign feature database outside the exact parent output")
    vectors = summarize_database(native_result["database"], expected_ids, exact["measurement_identity"])
    control = CAMPAIGN / "session7-native-prerequisites/tiny8-collect-v1/control"
    output = Path(spec["output"]) / (actual["job_id"] + "-r" + str(actual["restart"]))
    c.require(output.resolve().is_relative_to(control.resolve()), "Collection export outside its owned control root")
    batch_files = list((PARENT_OUTPUT / "engineering-evidence").glob("batch-*.json"))
    c.require(len(batch_files) == 1, "Expected exactly one original author feature batch")
    batch = c.read(batch_files[0])
    c.require(batch["record_ids"] == expected_ids and batch["original_positions"] == exact["exact_batch"]["positions"]
              and batch["author_outer_feature_loop_AST_unchanged"] is True,
              "Captured author batch/source order differs")
    files = [
        s7_attempt / "START.json", s7_attempt / "EXACT_AUTHOR_BATCH.json",
        s7_attempt / "DERIVED_NATIVE_CONFIG.json", s7_attempt / "DERIVED_NATIVE_BINDING.json",
        s7_attempt / "RESULT.json", native_attempt / "START.json",
        native_attempt / "MODEL_AND_PROCESSOR_AUDIT.json", native_attempt / "ACTUAL_BATCH_PADDING.json",
        native_attempt / "RESULT.json", PARENT_OUTPUT / "engineering-evidence/FROZEN_PARAMETERS.json",
        batch_files[0], claim_path, s7_attempt / "TRANSPORT.json", media_path,
    ]
    exported = {}
    for path in files:
        c.require(path.resolve().is_relative_to(PARENT_OUTPUT.resolve()) and path.stat().st_size <= 1 << 20,
                  "Foreign or oversized Tiny raw evidence metadata")
        relative = path.relative_to(PARENT_OUTPUT)
        exported[str(relative)] = dict(source=c.desc(path), exported=c.seal(output / "raw" / relative, c.read(path)))
    stdout = spec["parent_stdout"]
    c.require(Path(stdout["path"]) == Path(spec["parent_binding"]["path"]).parent / ("job-" + PARENT_JOB + ".out")
              and stdout["bytes"] <= 1 << 20 and c.desc(stdout["path"]) == stdout,
              "Parent constructor stdout changed or belongs to another allocation")
    stdout_export = c.seal(output / "CONSTRUCTOR_STDOUT.json",
                           dict(source=stdout, stdout=Path(stdout["path"]).read_bytes().decode("utf-8")))
    result = dict(
        status="ACTUAL_TINY8_RAW_EVIDENCE_COLLECTED_PENDING_INDEPENDENT_RUNTIME_REVIEW",
        actual_collection=actual, parent_job=PARENT_JOB, parent_binding=spec["parent_binding"],
        parent_scheduler=scheduler, parent_H100_seconds=int(rows[0][3]),
        parent_CPU_core_seconds=int(rows[0][3]) * int(rows[0][4]),
        records=vectors, padding=padding, frozen_parameters=frozen,
        historical_image_RGB_transport=media,
        model_and_processor=c.read(native_attempt / "MODEL_AND_PROCESSOR_AUDIT.json"), exported=exported,
        constructor_stdout=stdout_export,
        author_NaN_replacements_total=sum(record["author_replaced_NaNs"] for record in vectors.values()),
        database_copied=False, model_or_image_or_dataset_copied=False, new_neural_forwards=0,
        full_pool_or_feature_runtime_or_student_release_PASS=False,
        preserved_counterevidence="Constructor unused-weight warnings remain in the complete parent scheduler stdout; independent model/vision fidelity review still required.",
    )
    return c.seal(output / "COLLECTION.json", result)


def run(binding_path):
    binding = c.read(binding_path)
    spec = c.checked(binding["configuration"])
    c.require(spec["schema"] == "s7-tiny8-raw-collection-spec-v1" and spec["owner"] == c.OWNER
              and spec["site"] == "aces", "Only the exact owned ACES result-only continuation is permitted")
    c.verify_sources(spec["sources"])
    c.verify_sources(spec["dependencies"])
    review = c.checked(binding["review"])
    c.require(review["status"] == "PASS_EXACT_S7_PREREQUISITE_SOURCE_AND_CONFIG"
              and review["configuration"] == binding["configuration"] and review["sources"] == spec["sources"]
              and review["dependencies"] == spec["dependencies"], "Exact independent collection review required")
    actual = prior.actual(binding, spec)
    c.seal(Path(spec["output"]) / ("INTENT-" + actual["job_id"] + "-r" + str(actual["restart"]) + ".json"),
           dict(actual=actual, configuration=binding["configuration"], parent=spec["parent_binding"], no_neural_work=True))
    print(c.canonical(collect(spec, actual)).decode(), end="")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.binding))
