"""Actual final-pool feature shards; durable missing-only same-job continuations."""
import argparse
import fcntl
import importlib
import importlib.metadata
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import types

from experiments.root_native_campaign_20261023 import common as c
from experiments.root_native_input_authority_20261023 import pool_authority_v1 as authority
from experiments.root_native_baselines_20261023.store import Store
from experiments.root_native_baselines_20261023 import features as coin
from experiments.root_native_datatailor_20261023 import features as tailor


N = 660763
K = {"0.05": 33038, "0.10": 66076, "0.20": 132152}
ROOT = Path(__file__).resolve().parents[2]


class CapacityStore(Store):
    def commit(self, measurements):
        capacity = os.statvfs(self.path.parent)
        c.require(capacity.f_bavail * capacity.f_frsize >= 512 << 20,
                  "Feature filesystem below commit safety floor; retain accepted data and diagnose")
        return super().commit(measurements)


def checked_review(spec, descriptor, review):
    c.require(spec["schema"] == "s7-finalpool-published-feature-array-v1"
              and spec["owner"] == "Session7" and spec["method"] in ("coincide", "datatailor"),
              "Exact finite final-pool publication acquisition required")
    c.require(spec["expected_N"] == N and spec["expected_K"] == K
              and spec["selected_data_budget_unit"] == "whole_original_conversation",
              "Do not substitute nominal or availability-defined denominator")
    c.require(review["status"] == "PASS_EXACT_S7_FULLPOOL_FEATURE_SOURCE_CONFIG"
              and review["configuration"] == descriptor and review["sources"] == spec["sources"],
              "Actual independent exact worker/config review required")
    for relative, digest in spec["sources"].items():
        path = Path(relative)
        c.require(not path.is_absolute() and ".." not in path.parts and c.sha(ROOT / path) == digest,
                  "Staged import/source closure differs: " + relative)
    c.require(spec["shard_count"] > 0 and len(set(spec["global_shard_indices"])) == len(spec["global_shard_indices"])
              and spec["global_shard_indices"]
              and all(type(index) is int and 0 <= index < spec["shard_count"]
                      for index in spec["global_shard_indices"]), "Exact distinct shard routing required")
    c.require(1 <= spec["array_concurrency"] <= 8
              and spec["site"]["name"] == "babel"
              and spec["site"]["gpu_type"] == "L40S"
              and spec["allocation"]["gpus"] == 1
              and spec["allocation"]["cpus"] == 8
              and spec["allocation"]["memory_mb"] == 65536
              and 0 < spec["allocation"]["reserve_seconds"] < spec["allocation"]["minutes"] * 60,
              "Explicit finite CPU+GPU array resources required")
    c.require(spec["source_root"] == str(ROOT)
              and c.within(spec["output"], spec["campaign_output"])
              and c.within(spec["campaign_output"], "/data/user_data/luyangz/native_campaign_20261023"),
              "Exact isolated source and approved derived-data output required")
    c.require(c.desc(spec["author_policy"]["path"]) == spec["author_policy"],
              "Author policy bytes changed")
    policy = Path(spec["author_policy"]["path"]).read_text()
    c.require("all Babel General slots" in policy and "up to eight" in policy,
              "Latest author eight-slot policy required")
    if spec["method"] == "coincide":
        c.require(spec["shard_count"] % 4 == 0, "Never change the four native shuffle42 author chunks")


def actual(binding, spec):
    job = os.environ.get("SLURM_JOB_ID", "")
    array = os.environ.get("SLURM_ARRAY_JOB_ID", "")
    task = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    c.require(job.isdecimal() and array == binding["job_id"] and task.isdecimal(),
              "Exact owned expanded array task required")
    task = int(task)
    c.require(task < len(spec["global_shard_indices"]), "Unregistered expanded array task")
    request = array + "_" + str(task)
    raw = subprocess.check_output(["scontrol", "show", "job", "-o", request], text=True, timeout=30)
    fields = dict(re.findall(r"(\w+)=(\S+)", raw))
    profile, allocation = spec["site"], spec["allocation"]
    intent = c.checked(binding["intent"])
    expected = dict(ArrayJobId=array, ArrayTaskId=str(task), JobState="RUNNING",
                    JobName=spec["job_name"], Account=profile["account"], QOS=profile["qos"],
                    Partition=profile["partition"], WorkDir=str(ROOT),
                    Command=str(ROOT / spec["wrapper"]), NumTasks="1",
                    NumCPUs=str(allocation["cpus"]), TresPerNode="gres/gpu:" + profile["gpu_type"] + ":1",
                    Comment="s7-fullpool:" + intent["nonce"])
    c.require(all(fields.get(key) == value for key, value in expected.items())
              and fields["JobId"] == job
              and fields["UserId"].split("(")[0] == profile["user"]
              and fields["NodeList"] == socket.gethostname().split(".")[0]
              and fields["Restarts"].isdecimal(), "Actual array ownership/route/node differs")
    tres = dict(item.split("=", 1) for item in fields["AllocTRES"].split(",") if "=" in item)
    c.require(tres.get("gres/gpu") == "1" and tres.get("mem") in ("64G", "65536M"),
              "One actual allocated GPU and exact memory per scientific shard")
    c.require(intent["configuration"] == binding["configuration"]
              and intent["review"] == binding["review"], "Stale submission intent")
    if "SLURM_RESTART_COUNT" in os.environ:
        c.require(os.environ["SLURM_RESTART_COUNT"] == fields["Restarts"], "Stale inherited restart identity")
    os.environ["SLURM_RESTART_COUNT"] = fields["Restarts"]
    def duration(value):
        days, clock = value.split("-", 1) if "-" in value else ("0", value)
        hours, minutes, seconds = map(int, clock.split(":"))
        return int(days) * 86400 + hours * 3600 + minutes * 60 + seconds
    c.require(duration(fields["TimeLimit"]) == allocation["minutes"] * 60,
              "Actual finite attempt duration differs")
    remaining = duration(fields["TimeLimit"]) - duration(fields["RunTime"])
    c.require(remaining > allocation["reserve_seconds"], "No accepted-boundary reserve remains")
    return dict(job_id=job, array_job_id=array, array_task_id=task,
                shard=spec["global_shard_indices"][task], restart=int(fields["Restarts"]),
                node=fields["NodeList"], raw_slurm=raw,
                deadline=time.time() + remaining - allocation["reserve_seconds"])


def roster(method, config, rows, ids):
    if method == "datatailor":
        return tailor.roster(config, rows, ids)
    batches = coin.batches(ids, "coincide", config["shard_index"], config["shard_count"])
    return [ids[position] for batch in batches for position in batch]


def runtime(spec):
    c.require(Path(sys.executable).absolute() == Path(spec["python"]).absolute(),
              "Exact existing qualified method-specific interpreter required")
    environment = c.checked(spec["environment"])
    if spec["method"] == "datatailor":
        from experiments.copilot7_native_official_eval_20261024.feature_v4 import observe
        observed = observe(environment)
        accepted = c.checked(spec["runtime_component"])
        c.require(accepted["environment"] == spec["environment"] and accepted["runtime"] == observed,
                  "Imported modules/origins/NumPy binaries differ from actual closed DT runtime")
    else:
        observed = {}
        for name, expected in spec["runtime_versions"].items():
            module = importlib.import_module(name)
            distribution = importlib.metadata.distribution(name)
            c.require(module.__version__ == distribution.version == expected,
                      "Actual imported/distribution feature runtime differs: " + name)
            observed[name] = dict(version=module.__version__, distribution_version=distribution.version,
                                  source=c.desc(module.__file__))
        for name, expected in spec["distribution_only_versions"].items():
            c.require(importlib.metadata.version(name) == expected,
                      "Actual explicitly distribution-only runtime differs: " + name)
    import torch
    c.require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
              "Actual CPU+GPU feature allocation required")
    return observed


def register_acquisition(spec, identity):
    claim = c.checked(spec["ownership"])
    c.require(claim["owner"] == "Session7" and claim["method"] == spec["method"]
              and claim["pool"] == spec["pool"] and claim["shard_count"] == spec["shard_count"]
              and claim["measurement_sources"] == spec["measurement_sources"]
              and claim["environment"] == spec["environment"]
              and claim["feature_assets"] == spec["feature_assets"]
              and claim["output"] == spec["output"], "Exact exclusive feature plan claim differs")
    root = Path(spec["campaign_output"])
    root.mkdir(parents=True, exist_ok=True)
    registry = Path(spec["storage_registry"])
    registry.mkdir(parents=True, exist_ok=True)
    with Path(spec["admission_lock"]).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        c.seal(root / (spec["method"] + "-MEASUREMENT.json"), dict(
            owner="Session7", measurement_identity=identity, pool=spec["pool"],
            ownership=spec["ownership"], output=spec["output"], shard_count=spec["shard_count"]))
        reservation = registry / (spec["method"] + "-RESERVATION.json")
        if reservation.exists():
            prior = c.read(reservation)
            c.require(prior["pool"] == spec["pool"] and prior["output"] == spec["output"]
                      and prior["measurement_identity"] == identity
                      and prior["reserved_bytes"] == spec["required_storage_bytes"],
                      "Existing storage claim belongs to another measurement")
        else:
            others = [c.read(path) for path in registry.glob("*-RESERVATION.json")]
            c.require(all(item["owner"] == "Session7" and item["pool"] == spec["pool"] for item in others),
                      "Foreign or stale collective reservation requires explicit reconciliation")
            capacity = os.statvfs(root)
            available = capacity.f_bavail * capacity.f_frsize
            outstanding = sum(item["reserved_bytes"] for item in others)
            c.require(available >= outstanding + spec["required_storage_bytes"] + (2 << 30)
                      and capacity.f_favail >= 10000,
                      "Measured collective feature storage unavailable; preserve evidence, never overcommit")
            c.seal(reservation, dict(owner="Session7", pool=spec["pool"], output=spec["output"],
                measurement_identity=identity, reserved_bytes=spec["required_storage_bytes"],
                free_bytes=available, free_inodes=capacity.f_favail, other_reservations=others,
                scope="Collective S7 feature reservations; other owners retain their storage ownership",
                actual_allocation=os.environ["SLURM_JOB_ID"], measured_utc=c.now()))
    capacity = os.statvfs(root)
    c.require(capacity.f_bavail * capacity.f_frsize >= 2 << 30, "Feature filesystem lost safety headroom")


def verify_fidelity(spec):
    proof = c.checked(spec["scoped_fidelity"])
    contract = coin.CONTRACTS["coincide"] if spec["method"] == "coincide" else tailor.CONTRACT
    c.require(proof["method"] == spec["method"] and proof["feature_contract"] == contract
              and proof["pool"] == spec["pool"] and proof["fullpool_previously_qualified"] is False,
              "Scoped constructor reuse must not masquerade as completed full-pool acquisition")
    authorization = c.checked(proof["author_acquisition_instruction"])
    c.require(authorization["owner"] == "Session7" and authorization["fullpool_feature_acquisition"] is True
              and authorization["student_release"] is False, "Author inference-only authorization required")
    reviews = [c.checked(descriptor) for descriptor in proof["closed_reviews"]]
    c.require(spec["method"] != "datatailor" or reviews[-1]["status"] == "PASS_SCOPED_DT32_PAYLOAD_PREFIX",
              "DT B1 must already be closed by the genuine independent V2 review")
    c.require(proof["constructor_sources"] == {name: spec["sources"][name]
              for name in proof["constructor_sources"]}, "Closed numerical constructor source changed")
    return contract


def validate_array_pool(binding, spec):
    # Retain Root's complete logical validator, replacing only its scalar admission
    # callback with the stronger exact-array guard. No Slurm environment is hidden.
    consumer = types.FunctionType(authority.validate_pool.__code__,
        dict(vars(authority), require_allocation=lambda: actual(binding, spec)),
        authority.validate_pool.__name__, authority.validate_pool.__defaults__)
    consumer.__kwdefaults__ = authority.validate_pool.__kwdefaults__
    return consumer(spec["pool"], spec["eligibility"])


def run(binding_path):
    started = time.monotonic()
    binding = c.read(binding_path)
    c.require(binding["owner"] == "Session7", "Wrong publication acquisition owner")
    spec = c.checked(binding["configuration"])
    checked_review(spec, binding["configuration"], c.checked(binding["review"]))
    observed = actual(binding, spec)
    control_attempt = Path(binding_path).parent / "attempts" / (
        observed["job_id"] + "-r" + str(observed["restart"]))
    c.seal(control_attempt / "ACTUAL.json", dict(actual=observed, binding=c.desc(binding_path),
                                                started_utc=c.now(), no_DATA_read_yet=True))
    stopped = [False]
    def stop(*_):
        stopped[0] = True
    signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        return acquire(binding_path, binding, spec, observed, started, stopped, control_attempt)
    except BaseException as error:
        c.seal(control_attempt / "FAILURE.json", dict(error=repr(error), actual=observed,
               elapsed_seconds=time.monotonic() - started, all_allocated_cost_counts=True))
        raise


def acquire(binding_path, binding, spec, observed, started, stopped, control_attempt):
    versions = runtime(spec)
    pool, rows, ids = validate_array_pool(binding, spec)
    c.require(len(ids) == N and pool["count"] == N
              and c.checked(pool["input_authority"])["K"] == K,
              "Actual full final pool is required, not engineering or an archive subset")
    from experiments.root_native_baselines_20261023 import pins
    pins.verify(spec["published_sources"])
    assets = c.checked(spec["feature_assets"])
    if spec["method"] == "coincide":
        for role in ("tiny_proxy", "tiny_siglip"):
            pins.verify_tiny_snapshot(assets["assets"][role], role)
    else:
        from experiments.root_native_campaign_v3_20261023.config import ASSET_PINS
        for role in ("vicuna", "clip336", "stage1_projector"):
            asset = assets["assets"][role]
            c.require((asset["repository"], asset["revision"]) == ASSET_PINS[role],
                      "Exact existing Stage1 component required")
            for description in asset["files"]:
                c.require(c.desc(description["path"]) == {key: description[key] for key in ("path", "bytes", "sha256")},
                          "Stage1 component bytes changed")
    contract = verify_fidelity(spec)
    output = Path(spec["output"]) / ("shard-%04d" % observed["shard"])
    output.mkdir(parents=True, exist_ok=True)
    config = dict(method=spec["method"], role="features", pool=spec["pool"],
                  feature_assets=spec["feature_assets"], environment=spec["environment"],
                  sources=spec["measurement_sources"], native_sources=spec["native_sources"],
                  shard_index=observed["shard"], shard_count=spec["shard_count"],
                  measurement_contract=contract, output=str(output),
                  site_profile=spec["feature_profile"])
    identity = (tailor.identity if spec["method"] == "datatailor" else coin.measurement_identity)(config, pool)
    selected = roster(spec["method"], config, rows, ids)
    c.require(selected, "No empty padded or duplicate scientific shard")
    register_acquisition(spec, identity)
    with (output / "EXECUTION.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt = output / ("attempt-" + observed["job_id"] + "-r" + str(observed["restart"]))
        c.seal(attempt / "START.json", dict(actual=observed, binding=c.desc(binding_path),
            runtime=versions, method=spec["method"], measurement_identity=identity,
            required_records=len(selected), roster_sha256=c.key(selected)))
        if (output / "COMPLETE.json").exists():
            previous = c.read(output / "COMPLETE.json")
            c.require(previous["measurement_identity"] == identity
                      and c.desc(previous["database"]["path"]) == previous["database"],
                      "Existing completed shard differs")
            return 0
        store = CapacityStore(output / "features.sqlite3", identity, selected, spec["method"])
        before = len(store.existing)
        should_stop = lambda: stopped[0] or time.time() >= observed["deadline"]
        try:
            if len(store.existing) != len(selected) and not should_stop():
                if spec["method"] == "datatailor":
                    tailor.compute(config, pool, rows, ids, store, should_stop, False, attempt)
                else:
                    coin.coincide(config, pool, rows, ids, store, should_stop, False)
            count = len(store.existing)
        except BaseException as error:
            c.seal(attempt / "FAILURE.json", dict(error=repr(error), accepted_records=len(store.existing),
                required_records=len(selected), elapsed_seconds=time.monotonic() - started,
                source_config=binding["configuration"], failed_cost_counts=True))
            raise
        finally:
            store.close()
        result = dict(status="COMPLETE_PUBLISHED_FEATURE_SHARD" if count == len(selected)
                      else "DURABLE_MISSING_ONLY_FEATURE_CONTINUATION",
                      method=spec["method"], measurement_identity=identity, pool=spec["pool"],
                      environment=spec["environment"], feature_assets=spec["feature_assets"],
                      sources=spec["measurement_sources"], native_sources=spec["native_sources"],
                      shard_index=observed["shard"], shard_count=spec["shard_count"],
                      counts=dict(before=before, after=count, required=len(selected)),
                      roster_sha256=c.key(selected), database=c.desc(output / "features.sqlite3"),
                      actual=observed, elapsed_seconds=time.monotonic() - started,
                      no_student_or_scientific_PASS=True)
        c.seal(attempt / "RESULT.json", result)
        c.seal(control_attempt / "RESULT.json", dict(
            result=c.desc(attempt / "RESULT.json"), accepted_records=count, required_records=len(selected),
            elapsed_seconds=time.monotonic() - started, status=result["status"]))
        if count == len(selected):
            c.seal(output / "COMPLETE.json", result)
            return 0
        c.seal(attempt / "REQUEUE_INTENT.json", dict(job_id=observed["job_id"], accepted_records=count,
                                                   immutable_progress_identity=identity, UTC=c.now()))
        response = subprocess.run(["scontrol", "requeue", observed["array_job_id"] + "_" + str(observed["array_task_id"])],
                                  capture_output=True, text=True, timeout=30)
        c.seal(attempt / "REQUEUE_RESPONSE.json", dict(rc=response.returncode,
               stdout=response.stdout, stderr=response.stderr))
        c.require(response.returncode == 0, "Missing-only same-task requeue unacknowledged")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.binding))
