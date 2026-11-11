"""S1-owned repairs; frozen provider readers and numerical operators are reused."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sqlite3
import subprocess
import time

from experiments.s1_parallel_prerequisites_v2_20261026 import worker as base

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
HOME = base.HOME
CONTROL = HOME / "collaboration/s1-missing-only-repair-20261026/registry-v3"
LOCK = base.LOCK
UNITS = ("dt16-union",)
REVIEW = "PASS_INDEPENDENT_S1_MISSING_ONLY_REPAIR_SOURCE_CONFIG_V1"
PYTHON = "/data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python"
RESOURCES = {
    "dt16-union": dict(cpus=8, memory_mb=65536, gpus=1, minutes=90, python=PYTHON),
}
OWN_FILES = ("__init__.py", "worker.py", "admit.py", "admit_union_only.py", "freeze.py", "run.sbatch")


def load_plan(plan_ref, review_ref):
    plan = base.checked(plan_ref)
    base.require(plan["schema"] == "s1-missing-only-repair-plan-v1"
                 and plan["owner"] == "Session1" and plan["units"] == list(UNITS)
                 and plan["resources"] == RESOURCES and plan["models"] == plan["forwards"] == 0
                 and plan["student_release_authorized"] is False,
                 "Exact finite missing-only repair plan required")
    for relative, sha in plan["sources"].items():
        path = ROOT / relative
        base.require(not Path(relative).is_absolute() and ".." not in Path(relative).parts
                     and path.is_file() and not path.is_symlink()
                     and base.digest(path.read_bytes()) == sha, "Frozen repair source mismatch: " + relative)
    base.require(all(str((HERE / name).relative_to(ROOT)) in plan["sources"] for name in OWN_FILES),
                 "Every repair execution source must be pinned")
    review = base.checked(review_ref)
    base.require(review["status"] == REVIEW and review["independent"] is True
                 and review["reviewer"] not in ("", "S1", "Session1")
                 and review["plan"] == plan_ref and review["sources"] == plan["sources"]
                 and review["approved_units"] == list(UNITS), "Independent exact repair review required")
    return plan


def submission_argv(unit, nonce, parents):
    base.require(unit in UNITS and re.fullmatch("[0-9a-f]{32}", nonce) is not None,
                 "Exact repair unit/nonce required")
    required = ["dt3-header"] if unit == "dt16-union" else []
    base.require(sorted(parents) == required
                 and all(str(job).isdecimal() for job in parents.values()),
                 "Only the full-union job depends on the one-header repair")
    resource = RESOURCES[unit]
    minutes = resource["minutes"]
    wall = "%02d:%02d:00" % divmod(minutes, 60)
    argv = ["sbatch", "--parsable", "--hold", "--no-requeue", "--nodes=1", "--ntasks=1",
            "--account=cx", "--partition=general", "--qos=normal", "--gres=gpu:L40S:1",
            "--cpus-per-task=" + str(resource["cpus"]), "--mem=" + str(resource["memory_mb"]) + "M",
            "--time=" + wall, "--job-name=dsmm-s1-repair-" + unit,
            "--comment=dsmm-s1-missing-repair:" + nonce, "--chdir=" + str(ROOT),
            "--output=" + str(CONTROL / unit / "slurm-%j.out")]
    return argv + [str(HERE / "run.sbatch"), str(CONTROL / unit / "BINDING.json"), resource["python"]]


def scheduler(binding, running, query=subprocess.check_output):
    unit = binding["unit"]
    resource = RESOURCES[unit]
    raw = query(["scontrol", "show", "job", "-o", binding["job_id"]], text=True, timeout=30)
    observed = base.fields(raw)
    expected = dict(JobId=binding["job_id"], UserId="luyangz(2708899)", Account="cx",
                    Partition="general", QOS="normal", JobName="dsmm-s1-repair-" + unit,
                    NumCPUs=str(resource["cpus"]), NumTasks="1", Restarts="0",
                    TimeLimit="%02d:%02d:00" % divmod(resource["minutes"], 60),
                    Command=str(HERE / "run.sbatch"), WorkDir=str(ROOT),
                    Comment="dsmm-s1-missing-repair:" + binding["nonce"],
                    TresPerNode="gres/gpu:L40S:1", JobState="RUNNING" if running else "PENDING")
    base.require(all(observed.get(key) == value for key, value in expected.items())
                 and observed.get("NumNodes") in ("1", "1-1")
                 and observed.get("ArrayJobId") in (None, "(null)"), "Exact repair scalar/resource custody")
    if not running:
        base.require(observed.get("Reason") == "JobHeldUser" and observed.get("StartTime") == "Unknown"
                     and observed.get("RunTime") == "00:00:00"
                     and observed.get("AllocTRES") in (None, "(null)", ""),
                     "Never-started held repair request required")
        actual_parents = set(re.findall(r"afterok:(\d+)", observed.get("Dependency", "")))
        base.require(not actual_parents, "Completed paid parent must not be resubmitted as a Slurm dependency")
    for name in (("ReqTRES", "AllocTRES") if running else ("ReqTRES",)):
        values = dict(item.split("=", 1) for item in observed[name].split(",") if "=" in item)
        base.require(values.get("cpu") == str(resource["cpus"])
                     and values.get("mem") in (str(resource["memory_mb"]) + "M",
                                               str(resource["memory_mb"] // 1024) + "G")
                     and values.get("gres/gpu") == "1", "Requested/allocated repair resource mismatch")
    actual_argv = shlex.split(observed.get("SubmitLine", ""))
    base.require(actual_argv and Path(actual_argv[0]).name == "sbatch"
                 and ["sbatch", *actual_argv[1:]] == submission_argv(unit, binding["nonce"], binding["parents"]),
                 "Physical repair argv/binding/dependency changed")
    base.require(query(["scontrol", "write", "batch_script", binding["job_id"], "-"], timeout=30)
                 == (HERE / "run.sbatch").read_bytes(), "Repair spool changed")
    return raw


def export_header(plan, directory, binding):
    packet = base.checked(plan["reconciliation"])
    pointer = packet["exact_current_pointer"]
    result = base.checked(pointer["HOME_terminal_result"])
    cfg = base.checked(pointer["configuration"])
    original = cfg["composite_shards"]["3"]
    base.require(result["job_id"] == "10922850" and result["restart"] == 1
                 and result["accepted"] == result["required"] == 41211
                 and result["progress"] == pointer["one_remaining_DATA_metadata_header"]
                 and original["legacy_databases"] == []
                 and original["original_measurement"] == pointer["original_measurement"],
                 "Actual existing paid structured-composite provider required")
    ref, data = base.snapshot(pointer["one_remaining_DATA_metadata_header"]["path"], 64 << 10)
    base.require(ref == pointer["one_remaining_DATA_metadata_header"], "Exact missing-one header changed")
    header = json.loads(data)
    base.require(header["schema"] == "s7-compatible-native-DT-composite-v2"
                 and header["status"] == "COMPLETE_COMPATIBLE_COMPOSITE_FEATURE_SHARD"
                 and header["shard_index"] == 3 and header["shard_count"] == 16
                 and header["original_measurement"] == original["original_measurement"]
                 and header["bound_configuration"] == pointer["configuration"]
                 and header["pool"] == plan["dt_config"]["pool"]
                 and header["legacy_databases"] == []
                 and header["after"] == header["required"] == 41211
                 and header["overlay_database"]["path"] == original["overlay_output"] + "/features.sqlite3",
                 "Actual structured-composite header identity/count/type mismatch")
    target = directory / "GLOBAL3_PROGRESS.json"
    with target.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    exported = dict(path=str(target), bytes=len(data), sha256=base.digest(data))
    entries = plan["dt_config"]["feature_shards"]
    base.require(len(entries) == 16 and entries[3]["kind"] == "composite_v3"
                 and entries[3]["provider_type"] == "STRUCTURED_COMPOSITE_V3_ROOT"
                 and entries[3]["manifest"] == exported, "Typed sixteen-provider binding differs")
    manifest = base.publish(directory / "DT16_INPUT_MANIFEST.json",
                            dict(status="TYPED_DT16_METADATA_REPAIRED_NOT_SEMANTIC_UNION",
                                 global3=exported, feature_shards=entries, reused_headers=15,
                                 new_header_reads=1, full16_semantic_READY=False))
    return dict(status="DT3_HEADER_AND_TYPED16_INPUTS_REPAIRED", exported_header=exported,
                input_manifest=manifest, successful_headers_reexported=0, database_payloads_read=0)


def qualify_union(plan, directory, binding):
    parent = base.checked(binding["parent_bindings"]["dt3-header"])
    parent_result_ref = plan["completed_parent_result"]
    parent_result = base.checked(parent_result_ref)
    base.require(parent["job_id"] == binding["parents"]["dt3-header"]
                 and parent["plan"] == plan["completed_parent_plan"]
                 and parent_result["job_id"] == parent["job_id"]
                 and parent_result["status"] == "DT3_HEADER_AND_TYPED16_INPUTS_REPAIRED",
                 "Actual missing-one repair parent required")
    from experiments.root_dt_union_qualification_v1_20261025 import worker as native
    cfg = plan["dt_config"]
    original = base.checked(cfg["original_configuration"])
    runtime = native.runtime(original)
    attempt = directory / "qualification"
    attempt.mkdir(exist_ok=False)
    measurement = dict(original, sources=original["measurement_sources"])
    result = native.execute(cfg, measurement, attempt)
    return dict(status="ACTUAL_DT16_UNION_AND_BOUNDED_PROFILE_REPAIRED",
                runtime=runtime, result=result, numerical_reader_and_operators_unchanged=True,
                new_neural_forwards=0, full_operator_or_student_READY=False)


def readonly_journal(path):
    path = Path(path)
    base.require(path.is_file() and not path.is_symlink()
                 and (not Path(str(path) + "-wal").exists()
                      or Path(str(path) + "-wal").stat().st_size == 0),
                 "Quiescent checkpointed paid journal required; never ignore paid WAL rows")
    # Immutable mode must not discard any paid rows that exist only in a WAL.
    return sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True, timeout=30)


def vqa_progress(plan, directory, binding):
    from experiments.root_babel_start_evaluation_v2_20261025 import banks, worker as original_worker
    from experiments.session6_native_start_execution_v6_20261024 import audit
    from experiments.session6_official_eval_v2_20261024 import common as native
    old_binding = base.checked(plan["vqav2_binding"])
    cfg = base.checked(old_binding["configuration"])
    base.require(cfg["bank"] == "VQAv2" and old_binding["job_id"] == "10923672",
                 "Exact terminal VQAv2 producer required")
    output = Path(cfg["output"])
    terminal = base.checked(plan["vqav2_terminal"])
    base.require(terminal["actual"]["job_id"] == "10923672"
                 and terminal["actual"]["restart"] == 2
                 and terminal["configuration"] == old_binding["configuration"],
                 "Exact paid final-attempt receipt required")
    with (output / "EXECUTION.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        function = native.read(output / "FUNCTION.json")
        inputs = native.read(output / "INPUTS.json")
        qualified = native.read(output / "QUALIFIED_GENERATION.json")
        model = native.read(output / "MODEL_IDENTITY.json")
        base.require(function["bank"] == "VQAv2" and function["job_id"] == old_binding["job_id"]
                     and function["nonce"] == old_binding["nonce"]
                     and function["configuration"] == old_binding["configuration"]
                     and function["sources_sha256"] == native.key(cfg["sources"])
                     and function["model_identity_sha256"] == native.key(model)
                     and function["media"] == cfg["media"] and function["decoding"] == cfg["decoding"]
                     and function["training_updates"] == 0
                     and qualified["status"] == "PASS_SCOPED_ACTUAL_BABEL_TWO_QA_FRESH_RELOAD_FOUR_OUTPUTS"
                     and qualified["function"] == function, "Frozen generation function or actual qualifier differs")
        first = native.checked(qualified["first"])
        repeat = native.checked(qualified["repeat"])
        original_worker.same_repeat(first["entries"], repeat["entries"])
        roster, _, _ = banks.load(cfg)
        rows = roster["records"]
        base.require(len(rows) == 107394 and function["QA_roster_sha256"] == native.key(roster)
                     and inputs["roster_sha256"] == native.key(roster)
                     and inputs["configuration"] == old_binding["configuration"],
                     "Exact full authorized VQAv2 roster differs")
        expected = {row["semantic_key"]: row for row in rows}
        db_path = output / "production/raw_answers.sqlite3"
        before = db_path.stat()
        present = set()
        accepted_digest = hashlib.sha256()
        connection = readonly_journal(db_path)
        try:
            metadata = dict(connection.execute("SELECT k,v FROM metadata"))
            base.require(metadata == dict(function=native.key(function), ordered_roster=native.key(rows)),
                         "Paid journal function/ordered-QA identity differs")
            for key, raw, record in connection.execute("SELECT k,raw,audit FROM answers ORDER BY k"):
                base.require(key in expected and key not in present, "Foreign or duplicate paid QA key")
                entry = dict(semantic_key=key, raw=json.loads(raw), audit=json.loads(record))
                audit.validate_entry(entry, expected[key], function)
                context = entry["audit"]["actual"]
                base.require(context["job_id"] == "10923672" and context["nonce"] == old_binding["nonce"]
                             and context["restart"] in (0, 1, 2), "Foreign physical paid-QA provenance")
                present.add(key)
                accepted_digest.update(native.canonical(entry))
        finally:
            connection.close()
        after = db_path.stat()
        base.require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                     == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                     and (not Path(str(db_path) + "-wal").exists()
                          or Path(str(db_path) + "-wal").stat().st_size == 0),
                     "Paid journal changed during read-only repair")
        missing = [row["semantic_key"] for row in rows if row["semantic_key"] not in present]
        base.require(len(present) >= terminal["accepted"]
                     and len(present) + len(missing) == 107394, "Accepted paid-prefix reconciliation failed")
        body = "".join(key + "\n" for key in missing).encode()
        base.require(len(body) <= 7 << 20, "Finite missing-QA roster output bound")
        target = directory / "MISSING_QA_KEYS.txt"
        with target.open("xb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        return dict(status="EXACT_VQAV2_PAID_PREFIX_AND_MISSING_ROSTER_FROZEN",
                    expected_questions=107394, accepted=len(present), remaining=len(missing),
                    original_terminal_accepted=terminal["accepted"],
                    ordered_roster_sha256=native.key(rows), function_sha256=native.key(function),
                    accepted_record_digest=accepted_digest.hexdigest(),
                    missing_keys=dict(path=str(target), bytes=len(body), sha256=base.digest(body)),
                    old_journal_modified=False, model_or_QA_replayed=False,
                    full_bank_complete=len(missing) == 0, native_export_or_official_score_qualified=False,
                    next="Reviewed missing-only native generation is required if remaining>0; no counter alone is a complete bank.")


def run(binding_path):
    binding_path = base.literal(binding_path)
    binding = json.loads(binding_path.read_bytes())
    unit = binding["unit"]
    base.require(os.getuid() == 2708899 and unit in UNITS
                 and binding_path == CONTROL / unit / "BINDING.json"
                 and binding["owner"] == "Session1"
                 and os.environ.get("SLURM_JOB_ID") == binding["job_id"]
                 and os.environ.get("SLURM_RESTART_COUNT", "0") == "0"
                 and not os.environ.get("SLURM_ARRAY_JOB_ID"), "Exact fresh S1 repair actor required")
    plan = load_plan(binding["plan"], binding["review"])
    base.require(ROOT == Path(plan["remote_source_root"]) and Path.cwd() == ROOT
                 and Path(os.sys.executable).absolute() == Path(RESOURCES[unit]["python"]).absolute(),
                 "Exact private repair source/interpreter required")
    intent = base.checked(binding["intent"])
    base.require(intent["argv"] == submission_argv(unit, binding["nonce"], binding["parents"])
                 and intent["claim"] == binding["claim"], "Repair intent custody differs")
    claim = base.checked(binding["claim"])
    base.require(claim["owner"] == "Session1" and claim["unit"] == unit
                 and claim["plan"] == binding["plan"], "Exact registered repair ownership required")
    actual = scheduler(binding, True)
    output = binding_path.parent / "output"
    output.mkdir(exist_ok=False)
    capacity = os.statvfs(output)
    base.require(capacity.f_bavail * capacity.f_frsize >= 8 << 20 and capacity.f_favail >= 64,
                 "Insufficient bounded HOME metadata space")
    began = time.monotonic()
    base.publish(output / "START.json", dict(unit=unit, binding_path=str(binding_path), actual=actual))
    action = {"dt3-header": export_header, "dt16-union": qualify_union,
              "vqav2-progress": vqa_progress}[unit]
    try:
        result = action(plan, output, binding)
        result.update(unit=unit, job_id=binding["job_id"], models=0, forwards=0,
                      elapsed_seconds=time.monotonic() - began, student_READY=False)
        ref = base.publish(output / "RESULT.json", result)
        base.require(sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) <= 8 << 20,
                     "Bounded metadata output exceeded")
        print(json.dumps(ref), flush=True)
    except (OSError, ValueError, KeyError, RuntimeError, sqlite3.Error) as error:
        base.publish(output / "FAILURE.json", dict(unit=unit, error_class=type(error).__name__,
                     error=str(error), elapsed_seconds=time.monotonic() - began))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True)
    run(parser.parse_args().binding)
