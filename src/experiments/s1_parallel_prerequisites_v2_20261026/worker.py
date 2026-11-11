"""Collect exact provider evidence inside owned Babel CPU+GPU allocations."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
HOME = Path("/home/luyangz/data_selection_multimodal")
CONTROL = HOME / "collaboration/s1-parallel-prerequisites-20261026/registry-v2"
LOCK = HOME / "collaboration/babel-two-track-20261019/ADMISSION.lock"
REVIEW = "PASS_INDEPENDENT_S1_PARALLEL_PREREQUISITE_SOURCE_CONFIG_V2"
UNITS = ("dt16-metadata", "gqa-native-member", "start-custody")
RESOURCES = dict(cpus=4, memory_mb=16384, gpus=1, minutes=15)
SOURCE_NAMES = ("__init__.py", "worker.py", "admit.py", "run.sbatch")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def literal(path):
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts, "Absolute literal path required")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlinked control path")
    return path


def snapshot(path, maximum):
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts, "Absolute input path required")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(0 < before.st_size <= maximum, "Input exceeds bounded metadata scope: " + str(path))
        data = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    require(identity(before) == identity(after) and len(data) == after.st_size,
            "Input changed during bounded read: " + str(path))
    return dict(path=str(path), bytes=len(data), sha256=digest(data)), data


def checked(ref, maximum=16 << 20):
    require(set(ref) == {"path", "bytes", "sha256"} and type(ref["bytes"]) is int
            and re.fullmatch("[0-9a-f]{64}", ref["sha256"]) is not None,
            "Exact input descriptor required")
    actual, data = snapshot(ref["path"], maximum)
    require(actual == ref, "Input descriptor mismatch: " + ref["path"])
    return json.loads(data)


def publish(path, value):
    path = literal(path)
    data = canonical(value) + b"\n"
    capacity(path.parent, len(data), 8 << 20)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return dict(path=str(path), bytes=len(data), sha256=digest(data))


def capacity(directory, additional, maximum):
    used = sum(path.stat().st_size for path in Path(directory).iterdir() if path.is_file())
    require(used + additional <= maximum, "Bounded metadata output capacity exceeded")


def sources():
    return {str((HERE / name).relative_to(ROOT)): digest((HERE / name).read_bytes())
            for name in SOURCE_NAMES}


def load_plan(plan_ref, review_ref):
    plan = checked(plan_ref)
    require(plan["schema"] == "s1-parallel-prerequisites-plan-v2"
            and plan["owner"] == "Session1" and plan["source_map"] == sources()
            and plan["resources"] == RESOURCES and tuple(plan["units"]) == UNITS
            and plan["output_limit_bytes_per_unit"] == 8 << 20
            and plan["models"] == plan["forwards"] == plan["downloads"] == 0
            and plan["student_release_authorized"] is False,
            "Exact finite source/configuration scope required")
    require(plan["function_inputs"] == {
        "dt16-metadata": plan["dt_inputs"],
        "gqa-native-member": plan["gqa_protocol"],
        "start-custody": plan["start_banks"],
    }, "Function identities must bind the actual consumed inputs")
    review = checked(review_ref)
    require(review["status"] == REVIEW and review["independent"] is True
            and isinstance(review["reviewer"], str)
            and review["reviewer"] not in ("", "Session1", "S1")
            and review["plan"] == plan_ref and review["source_map"] == plan["source_map"]
            and review["approved_units"] == list(UNITS),
            "Independent exact source/config review required")
    return plan


def fields(raw):
    rows = [row for row in raw.splitlines() if row.strip()]
    require(len(rows) == 1, "Exactly one scheduler record required")
    tokens = list(re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=", rows[0]))
    require(tokens and tokens[0].start() == 0, "Malformed scheduler record")
    result = {}
    for index, token in enumerate(tokens):
        require(token[1] not in result, "Duplicate scheduler field")
        end = tokens[index + 1].start() if index + 1 < len(tokens) else len(rows[0])
        result[token[1]] = rows[0][token.end():end].strip()
    return result


def submission_argv(unit, nonce):
    require(unit in UNITS and re.fullmatch("[0-9a-f]{32}", nonce) is not None,
            "Exact registered unit/nonce required")
    directory = CONTROL / unit
    return ["sbatch", "--parsable", "--hold", "--no-requeue", "--nodes=1", "--ntasks=1",
            "--account=cx", "--partition=general", "--qos=normal", "--gres=gpu:L40S:1",
            "--cpus-per-task=4", "--mem=16G", "--time=00:15:00",
            "--job-name=dsmm-s1-prereq-" + unit, "--comment=dsmm-s1-prerequisite:" + nonce,
            "--chdir=" + str(ROOT), "--output=" + str(directory / "slurm-%j.out"),
            str(HERE / "run.sbatch"), str(directory / "BINDING.json")]


def scheduler(binding, running, query=subprocess.check_output):
    raw = query(["scontrol", "show", "job", "-o", binding["job_id"]], text=True, timeout=30)
    actual = fields(raw)
    expected = dict(JobId=binding["job_id"], UserId="luyangz(2708899)", Account="cx",
                    Partition="general", QOS="normal", JobName=binding["job_name"],
                    NumCPUs="4", NumTasks="1", TimeLimit="00:15:00", Restarts="0",
                    Command=str(HERE / "run.sbatch"), WorkDir=str(ROOT),
                    Comment="dsmm-s1-prerequisite:" + binding["nonce"],
                    TresPerNode="gres/gpu:L40S:1",
                    JobState="RUNNING" if running else "PENDING")
    require(all(actual.get(key) == value for key, value in expected.items())
            and actual.get("NumNodes") in ("1", "1-1")
            and actual.get("ArrayJobId") in (None, "(null)")
            and actual.get("ArrayTaskId") in (None, "(null)"),
            "Exact owned scalar route/resources/source differ")
    if not running:
        require(actual.get("Reason") == "JobHeldUser" and actual.get("StartTime") == "Unknown"
                and actual.get("RunTime") == "00:00:00"
                and actual.get("AllocTRES") in (None, "(null)", "")
                and actual.get("Dependency") in (None, "(null)", "None"),
                "Exact never-started held request required")
    tres = dict(item.split("=", 1) for item in actual["ReqTRES"].split(",") if "=" in item)
    require(tres.get("cpu") == "4" and tres.get("mem") in ("16G", "16384M")
            and tres.get("gres/gpu") == "1", "Requested resource mismatch")
    if running:
        allocated = dict(item.split("=", 1) for item in actual["AllocTRES"].split(",") if "=" in item)
        require(allocated.get("cpu") == "4" and allocated.get("mem") in ("16G", "16384M")
                and allocated.get("gres/gpu") == "1", "Actual allocated resources differ")
    argv = shlex.split(actual.get("SubmitLine", ""))
    require(argv and Path(argv[0]).name == "sbatch"
            and ["sbatch", *argv[1:]] == submission_argv(binding["unit"], binding["nonce"]),
            "Actual submission arguments differ, including binding custody")
    spool = query(["scontrol", "write", "batch_script", binding["job_id"], "-"], timeout=30)
    require(spool == (HERE / "run.sbatch").read_bytes(), "Physical batch spool differs")
    return raw


def copy_metadata(path, directory, index, expected=None):
    ref, data = snapshot(path, 2 << 20)
    if expected is not None:
        require(ref == expected, "Pinned provider metadata changed: " + str(path))
    value = json.loads(data)
    require(isinstance(value, dict), "Provider metadata must be a JSON object")
    target = directory / ("input-%02d.json" % index)
    capacity(directory, len(data), 6 << 20)
    with target.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(original=ref, collected=dict(path=str(target), bytes=len(data), sha256=digest(data)))


def dt16(plan, directory):
    dependencies = checked(plan["dt_inputs"])
    entries = dependencies["feature_shards"]
    require(len(entries) == 16 and {item["shard_index"] for item in entries} == set(range(16)),
            "Exactly sixteen existing DT providers required")
    results = []
    for item in entries:
        original = item["kind"] == "original"
        path = item["required_existing_completion_path"] if original else item["manifest"]["path"]
        row = dict(shard_index=item["shard_index"], kind=item["kind"], requested_path=path)
        try:
            row["metadata"] = copy_metadata(path, directory, item["shard_index"],
                                            None if original else item["manifest"])
            row["status"] = "COLLECTED_JSON_METADATA"
        except (OSError, ValueError) as error:
            row.update(status="FAILED_METADATA_READ", error_class=type(error).__name__, error=str(error))
        results.append(row)
        publish(directory / ("shard-%02d.json" % item["shard_index"]), row)
    success = all(row["status"] == "COLLECTED_JSON_METADATA" for row in results)
    return success, dict(status="COMPLETE_DT16_METADATA_EXPORT" if success else "INCOMPLETE_DT16_METADATA_EXPORT",
                         providers=results, database_payloads_read=0, full_union_qualified=False,
                         geometry_or_student_authorized=False)


def gqa(plan, directory):
    protocol = checked(plan["gqa_protocol"])
    require(protocol["scope"] == "PUBLIC_NATIVE_EVALUATION_ONLY;NO_SELECTOR_OR_PROBING_REFERENCE"
            and protocol["closed_legacy_read_or_decode_permitted"] is False
            and protocol["selection_reference_access"] is False,
            "Public EVAL-only authority required; legacy QA remains closed")
    archive_ref = protocol["native_input_archive"]
    actual, archive = snapshot(archive_ref["path"], 64 << 20)
    require(actual == archive_ref, "Exact cached native archive changed")
    member = protocol["only_allowed_member"]
    require(member == "gqa/llava_gqa_testdev_balanced.jsonl", "Only authorized GQA member allowed")
    with zipfile.ZipFile(io.BytesIO(archive)) as store:
        matches = [info for info in store.infolist() if info.filename == member]
        require(len(matches) == 1 and 0 < matches[0].file_size <= 32 << 20,
                "Exactly one bounded authorized native GQA member required")
        data = store.read(matches[0])
        crc = matches[0].CRC
    return gqa_rows(data, directory, archive_ref, member, crc)


def gqa_rows(data, directory, archive_ref, member, crc):
    identifiers = set()
    ordered = hashlib.sha256()
    records = hashlib.sha256()
    target = directory / "PUBLIC_QA_HASHES.jsonl"
    written = 0
    with target.open("xb") as stream:
        for line in data.splitlines():
            require(line.strip(), "Blank native GQA row")
            row = json.loads(line)
            identifier = row["question_id"]
            require(type(identifier) in (str, int) and str(identifier)
                    and str(identifier) not in identifiers,
                    "Duplicate, empty or unsupported native GQA question identity")
            require(isinstance(row["image"], str) and row["image"]
                    and isinstance(row["text"], str) and row["text"],
                    "Native GQA image/question schema mismatch")
            identifiers.add(str(identifier))
            ordered.update(canonical(identifier) + b"\n")
            records.update(canonical(row) + b"\n")
            metadata = dict(question_id=identifier, image_sha256=digest(row["image"].encode()),
                            question_sha256=digest(row["text"].encode()),
                            source_row_sha256=digest(canonical(row)))
            encoded = canonical(metadata) + b"\n"
            written += len(encoded)
            require(written <= 7 << 20, "Bounded derived GQA hash output exceeded")
            stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    require(identifiers, "Empty native GQA member")
    require(target.stat().st_size <= 7 << 20, "Bounded derived GQA hash output exceeded")
    return True, dict(status="QUALIFIED_PUBLIC_NATIVE_GQA_MEMBER_ONLY",
                      archive=archive_ref, member=member, member_bytes=len(data),
                      member_sha256=digest(data), member_crc32=crc, questions=len(identifiers),
                      ordered_question_ids_sha256=ordered.hexdigest(),
                      ordered_native_rows_sha256=records.hexdigest(),
                      GT_qualified=False, RGB_qualified=False, official_scorer_qualified=False,
                      bank_READY=False, selector_reference_access=False, raw_QA_exported=False)


def answer_metadata(path, maximum):
    path = Path(path)
    count = 0
    total = 0
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(0 < before.st_size <= maximum, "Raw answer export outside existing output envelope")
        for line in stream:
            total += len(line)
            require(total <= maximum and line.strip(), "Invalid or oversized answer export")
            value = json.loads(line)
            require("question_id" in value and isinstance(value["text"], str),
                    "Native answer row schema differs")
            count += 1
            sha.update(line)
        after = os.fstat(stream.fileno())
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            and total == after.st_size, "Answer export changed during read")
    return dict(path=str(path), bytes=total, sha256=sha.hexdigest(), lines=count,
                raw_answers_exported=False, semantic_coverage_independently_validated=False)


def verify_answer_reference(ref, observed, expected_path):
    require(isinstance(ref, dict) and set(ref) == {"path", "bytes", "sha256"}
            and ref["path"] == str(expected_path)
            and type(ref["bytes"]) is int and ref["bytes"] > 0
            and isinstance(ref["sha256"], str)
            and re.fullmatch("[0-9a-f]{64}", ref["sha256"]) is not None,
            "Well-typed producer native_answers at the exact existing output required")
    require(ref == {key: observed[key] for key in ("path", "bytes", "sha256")},
            "Observed raw answers differ from the pinned producer artifact")


def start_custody(plan, directory):
    banks = plan["start_banks"]
    require(set(banks) == {"POPE", "VQAv2", "VizWiz"}, "Only three existing START producers")
    counts = {"POPE": 8910, "VQAv2": 107394, "VizWiz": 8000}
    expected_jobs = {"POPE": "10923671", "VQAv2": "10923672", "VizWiz": "10923673"}
    results = []
    for index, (bank, config_ref) in enumerate(banks.items()):
        row = dict(bank=bank, original_job_id=expected_jobs[bank])
        try:
            cfg = checked(config_ref)
            output = Path(cfg["output"])
            require(cfg["bank"] == bank and output == Path(
                "/data/user_data/luyangz/native_campaign_20261023/start-evaluation-v2") / bank,
                "Existing exact START output route required")
            complete_ref, data = snapshot(output / "COMPLETE.json", 2 << 20)
            complete = json.loads(data)
            require(complete["bank"] == bank and complete["questions"] == counts[bank]
                    and complete["actual"]["job_id"] == expected_jobs[bank]
                    and complete["configuration"]["sha256"] == config_ref["sha256"],
                    "Existing producer completion/configuration identity differs")
            result = checked(complete["result"])
            require(result["status"] == "COMPLETE_REAL_BANK" and result["accepted"] == counts[bank]
                    and result["remaining"] == 0, "Producer did not complete the full bank")
            native_answers = result["native_answers"]
            row["completion"] = complete_ref
            row["collected_completion"] = copy_metadata(output / "COMPLETE.json", directory, 20 + index)
            row["result"] = complete["result"]
            row["collected_result"] = copy_metadata(complete["result"]["path"], directory, 30 + index,
                                                    complete["result"])
            row["answers"] = answer_metadata(output / "RAW_NATIVE_ANSWERS.jsonl",
                                              cfg["output_limit_bytes"])
            verify_answer_reference(native_answers, row["answers"],
                                    output / "RAW_NATIVE_ANSWERS.jsonl")
            require(row["answers"]["lines"] == counts[bank], "Native answer count differs")
            row.update(status="COLLECTED_EXISTING_START_CUSTODY", official_score_reviewed=False,
                       existing_score_status=result.get("score", {}).get("status"))
        except (OSError, ValueError, KeyError) as error:
            row.update(status="FAILED_START_CUSTODY", error_class=type(error).__name__, error=str(error))
        results.append(row)
        publish(directory / (bank + ".json"), row)
    success = all(row["status"] == "COLLECTED_EXISTING_START_CUSTODY" for row in results)
    return success, dict(status="COMPLETE_EXISTING_START_CUSTODY" if success else "INCOMPLETE_START_CUSTODY",
                         banks=results, new_answers=0, official_score_reviewed=False,
                         student_results=False)


def run(binding_path):
    began = time.monotonic()
    require(os.getuid() == 2708899, "Actual Babel project user required")
    binding_path = literal(binding_path)
    binding = json.loads(binding_path.read_bytes())
    unit = binding["unit"]
    require(unit in UNITS and binding["owner"] == "Session1"
            and binding_path == CONTROL / unit / "BINDING.json"
            and os.environ.get("SLURM_JOB_ID") == binding["job_id"]
            and os.environ.get("SLURM_RESTART_COUNT", "0") == "0"
            and not os.environ.get("SLURM_ARRAY_JOB_ID"), "Exact fresh S1 scalar binding required")
    plan = load_plan(binding["plan"], binding["review"])
    intent = checked(binding["intent"])
    claim = checked(binding["claim"])
    require(intent["nonce"] == binding["nonce"] and intent["claim"] == binding["claim"]
            and intent["argv"] == submission_argv(unit, binding["nonce"])
            and claim["owner"] == "Session1" and claim["scope"] == unit
            and claim["plan"] == binding["plan"]
            and claim["function_identity"] == digest(canonical(dict(
                scope=unit, inputs=plan["function_inputs"][unit]))),
            "Registered intent/function custody differs")
    require(Path(plan["remote_source_root"]) == ROOT and Path.cwd() == ROOT,
            "Execute only the frozen private source root")
    actual = scheduler(binding, True)
    output = binding_path.parent / "output"
    output.mkdir(exist_ok=False)
    available = os.statvfs(output)
    require(available.f_bavail * available.f_frsize >= plan["output_limit_bytes_per_unit"]
            and available.f_favail >= 64, "Insufficient small HOME metadata output capacity")
    publish(output / "START.json", dict(unit=unit, binding_path=str(binding_path), actual=actual,
                                       models=0, forwards=0, downloads=0))
    action = {"dt16-metadata": dt16, "gqa-native-member": gqa, "start-custody": start_custody}[unit]
    try:
        success, result = action(plan, output)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        publish(output / "FAILURE.json", dict(error_class=type(error).__name__, error=str(error),
                                            unit=unit, elapsed_seconds=time.monotonic() - began))
        raise
    result.update(unit=unit, job_id=binding["job_id"], success=success, models=0, forwards=0,
                  downloads=0, GPU_computation=False, student_READY=False,
                  elapsed_seconds=time.monotonic() - began)
    ref = publish(output / "RESULT.json", result)
    require(sum(path.stat().st_size for path in output.iterdir() if path.is_file())
            <= plan["output_limit_bytes_per_unit"], "Small metadata output budget exceeded")
    print(json.dumps(ref), flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True)
    raise SystemExit(run(parser.parse_args().binding))
