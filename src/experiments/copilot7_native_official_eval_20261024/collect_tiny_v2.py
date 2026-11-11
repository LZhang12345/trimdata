"""Export the completed Tiny8 vectors and independently rechecked RGB; no forward."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import sqlite3
import struct
import subprocess
import tempfile

from . import common as c
from . import feature_v1 as features
from . import prerequisite_v3 as admission
from . import collect_tiny_v1 as original
from experiments.root_native_campaign_20261023 import common as producer


PARENT_JOB = original.PARENT_JOB
PARENT_OUTPUT = original.PARENT_OUTPUT
PARENT_SPEC_SHA256 = original.PARENT_SPEC_SHA256
OUTPUT = original.CAMPAIGN / "session7-native-prerequisites/tiny8-collect-v2/control/readout"


def seal_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        c.require(path.read_bytes() == payload, "Immutable raw export differs")
        return c.desc(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".raw-export-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            c.require(path.read_bytes() == payload, "Concurrent raw export differs")
        handle = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)
    finally:
        os.unlink(temporary)
    return c.desc(path)


def vectors(description, expected_ids, identity):
    c.require(c.desc(description["path"]) == description, "Completed Tiny database changed")
    connection = sqlite3.connect(Path(description["path"]).absolute().as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        c.require(metadata == dict(identity=identity, method="coincide",
                                   roster_sha256=producer.key(expected_ids)),
                  "Actual producer serialization, measurement or roster differs")
        rows = list(connection.execute("SELECT record_id,vector,info,tau,length,nan_count,identity FROM features"))
    finally:
        connection.close()
    c.require(c.desc(description["path"]) == description, "Database changed during read-only collection")
    c.require(len(rows) == len(expected_ids) == len(set(expected_ids)) == 8
              and {row[0] for row in rows} == set(expected_ids), "Exact eight actual feature records required")
    decoded, payloads = {}, {}
    for ident, blob, info, tau, length, nans, measured in rows:
        c.require(measured == identity and len(blob) == 40960 and info is None and tau is None,
                  "Wrong feature identity/dimension/type")
        values = [value[0] for value in struct.iter_unpack("<e", blob)]
        c.require(all(math.isfinite(value) for value in values)
                  and type(nans) is int and 0 <= nans <= 20480, "Corrupt vector or author NaN accounting")
        payloads[ident] = blob
        decoded[ident] = dict(
            dimension=20480, dtype="<f2", vector_sha256=hashlib.sha256(blob).hexdigest(),
            author_replaced_NaNs=nans, length=length,
            L2=math.sqrt(math.fsum(value * value for value in values)),
            nonzero_coordinates=sum(value != 0 for value in values),
        )
    return {ident: decoded[ident] for ident in expected_ids}, b"".join(payloads[ident] for ident in expected_ids)


def export_media(configuration, exact, historical, actual, output):
    from PIL import Image
    pool = c.checked(configuration["inputs"]["pool"])
    rows = c.checked(pool["records_json"])
    ids = pool["record_ids"]
    media, alias = features.tiny_runtime_media(configuration, pool, rows, ids, exact, actual, output / "current-media")
    c.require(set(media["assets"]) == set(historical["assets"]), "Collection and parent image rosters differ")
    payloads = {}
    for name, item in sorted(media["assets"].items()):
        expected = historical["assets"][name]
        c.require(all(item[key] == expected[key] for key in ("bytes", "sha256", "width", "height", "rgb_sha256")),
                  "Actual collection pixels differ from the historical parent transport")
        path = Path(item["path"])
        c.require(c.desc(path) == {key: item[key] for key in ("path", "bytes", "sha256")},
                  "Fresh collection encoded alias changed")
        with Image.open(path) as opened:
            rgb = opened.convert("RGB")
            try:
                raw = rgb.tobytes()
                c.require((rgb.width, rgb.height) == (item["width"], item["height"])
                          and hashlib.sha256(raw).hexdigest() == item["rgb_sha256"], "Fresh raw RGB changed")
            finally:
                rgb.close()
        tag = hashlib.sha256(name.encode()).hexdigest()
        payloads[name] = dict(
            encoded=seal_bytes(output / "payload" / (tag + ".encoded"), path.read_bytes()),
            RGB=seal_bytes(output / "payload" / (tag + ".rgb"), raw),
            width=item["width"], height=item["height"], RGB_format="RGB_uint8_row_major",
            historical_alias=expected["path"],
            collection_alias=item["path"], original_expected=expected,
        )
    return dict(current_runtime_images=alias, images=payloads,
                historical_parent_job=PARENT_JOB, current_alias_job=actual["job_id"],
                collection_aliases_are_not_relabelled_as_parent_aliases=True)


def collect(spec, actual):
    c.require(actual["job_id"] != PARENT_JOB, "Never repurpose the completed parent allocation")
    binding = c.checked(spec["parent_binding"])
    c.require(binding["job_id"] == PARENT_JOB and binding["owner"] == c.OWNER
              and binding["configuration"]["sha256"] == PARENT_SPEC_SHA256, "Wrong completed Tiny parent")
    configuration = c.checked(binding["configuration"])
    c.require(configuration["phase"] == "tiny8" and configuration["site"] == "aces"
              and Path(configuration["output"]) == PARENT_OUTPUT, "Wrong parent function or output")
    c.verify_sources(configuration["sources"], Path(configuration["code_root"]))
    c.verify_sources(configuration["dependencies"], Path(configuration["code_root"]))
    review = c.checked(binding["review"])
    c.require(review["status"] == "PASS_EXACT_S7_PREREQUISITE_SOURCE_AND_CONFIG"
              and review["configuration"] == binding["configuration"], "Parent source/config review differs")
    scheduler = subprocess.check_output([
        "sacct", "-D", "-X", "-j", PARENT_JOB,
        "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,NodeList", "-n", "-P",
    ], text=True, timeout=30)
    accounting = [row.split("|") for row in scheduler.splitlines() if row.startswith(PARENT_JOB + "|")]
    c.require(len(accounting) == 1 and accounting[0][1:3] == ["COMPLETED", "0:0"],
              "Actual parent successful terminal accounting required")
    outer = PARENT_OUTPUT / ("s7-attempt-" + PARENT_JOB + "-r0")
    native = PARENT_OUTPUT / ("attempt-" + PARENT_JOB + "-r0")
    result, measured = c.read(outer / "RESULT.json"), c.read(native / "RESULT.json")
    exact = c.read(outer / "EXACT_AUTHOR_BATCH.json")
    config = c.read(outer / "DERIVED_NATIVE_CONFIG.json")
    ids = exact["exact_batch"]["record_ids"]
    c.require(result["native_return_code"] == 0 and result["actual"]["job_id"] == PARENT_JOB
              and result["phase"] == "tiny8" and config["exact_batch"] == exact["exact_batch"]
              and measured["status"] == "COMPLETE_ACTUAL_TINY_BATCH8_ENGINEERING_EVIDENCE"
              and measured["record_ids"] == ids and measured["records"] == 8
              and measured["newly_acquired_records"] == 8 and measured["cached_records"] == 0
              and measured["qualification_promoted"] is False, "Parent batch/measurement evidence differs")
    padding = c.read(native / "ACTUAL_BATCH_PADDING.json")
    c.require(padding["records"] == 8 and padding["native_tokens_and_labels_preserved"] is True
              and len(set(padding["original_token_lengths"])) > 1
              and any(value > 0 for value in padding["padded_positions"]), "Actual author padding peers differ")
    frozen_path = PARENT_OUTPUT / "engineering-evidence/FROZEN_PARAMETERS.json"
    frozen = c.read(frozen_path)
    c.require(frozen["before"] == frozen["after"] and frozen["actual_checked"] is True
              and frozen["optimizer"] is None and frozen["updates"] == 0 and frozen["full_pool_qualified"] is False,
              "Actual full frozen-parameter evidence differs")
    historical_path = outer / "media/RUNTIME_IMAGES.json"
    historical = c.read(historical_path)
    claim_path = outer / "TRANSPORT_CLAIM.json"
    claim = c.read(claim_path)
    c.require(historical["status"] == "PASS_BYTE_IDENTICAL_RUNTIME_IMAGES"
              and historical["scope"] == "ENGINEERING_ONLY" and historical["job_id"] == PARENT_JOB
              and historical["node"] == result["actual"]["node"] and historical["actual"]["restart"] == 0
              and historical["logical_pool"] == configuration["inputs"]["pool"]
              and historical["ownership"] == c.desc(claim_path) and claim["job_id"] == PARENT_JOB
              and 0 < len(historical["assets"]) <= 8
              and historical["scientific_pool_readiness_claim"] is False, "Historical raw RGB/alias function differs")
    c.require(Path(measured["database"]["path"]) == PARENT_OUTPUT / "features.sqlite3", "Foreign feature database")
    summary, raw_vectors = vectors(measured["database"], ids, exact["measurement_identity"])
    c.require(Path(spec["output"]) == OUTPUT, "Wrong owned collection root")
    output = OUTPUT / (actual["job_id"] + "-r" + str(actual["restart"]))
    output.mkdir(parents=True, exist_ok=True)
    fs = os.statvfs(output)
    c.require(fs.f_bavail * fs.f_frsize >= 1 << 30 and fs.f_favail >= 1000, "Insufficient measured raw export storage")
    batches = list((PARENT_OUTPUT / "engineering-evidence").glob("batch-*.json"))
    c.require(len(batches) == 1, "Exactly one original author batch required")
    batch = c.read(batches[0])
    c.require(batch["record_ids"] == ids and batch["original_positions"] == exact["exact_batch"]["positions"]
              and batch["author_outer_feature_loop_AST_unchanged"] is True, "Actual source-order batch differs")
    paths = [outer / name for name in ("START.json", "EXACT_AUTHOR_BATCH.json", "DERIVED_NATIVE_CONFIG.json",
                                      "DERIVED_NATIVE_BINDING.json", "RESULT.json", "TRANSPORT.json")]
    paths += [native / name for name in ("START.json", "MODEL_AND_PROCESSOR_AUDIT.json",
                                        "ACTUAL_BATCH_PADDING.json", "RESULT.json")]
    paths += [frozen_path, batches[0], claim_path, historical_path]
    exported = {}
    for path in paths:
        c.require(path.resolve().is_relative_to(PARENT_OUTPUT.resolve()) and path.stat().st_size <= 1 << 20,
                  "Foreign or oversized raw evidence metadata")
        source = c.desc(path)
        target = seal_bytes(output / "raw" / path.relative_to(PARENT_OUTPUT), path.read_bytes())
        c.require(all(target[key] == source[key] for key in ("bytes", "sha256")), "Raw evidence copy changed")
        exported[str(path.relative_to(PARENT_OUTPUT))] = dict(source=source, exported=target)
    stdout = spec["parent_stdout"]
    c.require(Path(stdout["path"]) == Path(spec["parent_binding"]["path"]).parent / ("job-" + PARENT_JOB + ".out")
              and stdout["bytes"] <= 1 << 20 and c.desc(stdout["path"]) == stdout, "Parent warning log changed")
    vector_payload = seal_bytes(output / "payload/vectors.fp16", raw_vectors)
    index = c.seal(output / "VECTOR_INDEX.json", dict(
        record_ids=ids, dimension=20480, dtype="<f2", row_bytes=40960,
        offsets={ident: position * 40960 for position, ident in enumerate(ids)}, payload=vector_payload, records=summary))
    media = export_media(configuration, exact["exact_batch"], historical, actual, output)
    receipt = dict(
        status="ACTUAL_TINY8_VECTOR_RGB_ALIAS_PAYLOAD_EXPORT_PENDING_INDEPENDENT_REVIEW",
        parent_job=PARENT_JOB, parent_binding=spec["parent_binding"], actual_collection=actual,
        parent_scheduler=scheduler, parent_H100_seconds=int(accounting[0][3]),
        parent_CPU_core_seconds=int(accounting[0][3]) * int(accounting[0][4]),
        vector_index=index, records=summary, media=media, exported=exported,
        parent_stdout=seal_bytes(output / "parent-constructor.stdout", Path(stdout["path"]).read_bytes()),
        author_NaN_replacements_total=sum(value["author_replaced_NaNs"] for value in summary.values()),
        frozen_parameters=frozen, padding=padding, historical_image_RGB_transport=historical,
        model_and_processor=c.read(native / "MODEL_AND_PROCESSOR_AUDIT.json"),
        new_neural_forwards=0, database_or_model_copied=False,
        selected_existing_media_only=True, full_pool_or_runtime_or_student_PASS=False,
        preserved_counterevidence="Original unused-weight warnings remain byte-exact; current rechecked aliases are not original consumer aliases.",
    )
    return c.seal(output / "COLLECTION.json", receipt)


def run(binding_path):
    binding = c.read(binding_path)
    spec = c.checked(binding["configuration"])
    review = c.checked(binding["review"])
    admission.verify_spec(spec, binding["configuration"], review)
    c.require(spec["phase"] == "tiny8_collection" and spec["new_neural_forwards"] == 0,
              "Only existing raw evidence collection, never another forward")
    actual = admission.actual(binding, spec)
    c.seal(OUTPUT / ("INTENT-" + actual["job_id"] + "-r" + str(actual["restart"]) + ".json"),
           dict(actual=actual, configuration=binding["configuration"], parent=spec["parent_binding"], no_neural_work=True))
    print(c.canonical(collect(spec, actual)).decode())
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.binding))
