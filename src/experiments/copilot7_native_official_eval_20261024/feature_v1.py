"""Separate S7 engineering guard around unchanged Tiny8 and DataTailor32 kernels."""
import argparse
import copy
import importlib
import os
from pathlib import Path
import re
import resource
import shutil
import socket
import subprocess
import sys
import time
import types

from . import common as c


def actual(binding, spec):
    job = os.environ.get("SLURM_JOB_ID", "")
    c.require(job.isdecimal() and binding["job_id"] == job and not os.environ.get("SLURM_ARRAY_JOB_ID"),
              "Exact scalar S7 engineering allocation required")
    raw = subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True, timeout=30)
    row = dict(re.findall(r"(\w+)=(\S+)", raw))
    site = spec["site"]
    expected = dict(JobId=job, JobState="RUNNING", JobName=spec["job_name"],
                    Account="cx" if site == "babel" else "158416130568",
                    Partition="general" if site == "babel" else "gpu", QOS="normal",
                    WorkDir=str(c.ROOT), Command=str(c.ROOT / spec["wrapper"]),
                    TresPerNode="gres/gpu:" + ("L40S" if site == "babel" else "h100") + ":1",
                    NumTasks="1", NumCPUs="8", TimeLimit="00:55:00")
    c.require(all(row.get(k) == v for k, v in expected.items()) and row.get("NumNodes") in {"1", "1-1"},
              "Actual S7 feature resources/site/command differ")
    c.require(row["UserId"].split("(")[0] == ("luyangz" if site == "babel" else "u.lz341613")
              and row["NodeList"] == socket.gethostname().split(".")[0], "Actual feature user/node differs")
    tres = dict(x.split("=", 1) for x in row.get("ReqTRES", "").split(",") if "=" in x)
    c.require(tres.get("mem") in {"64G", "65536M"} and tres.get("gres/gpu") == "1",
              "Actual feature host memory/GPU count differs")
    c.require(row["Restarts"].isdecimal(), "Scheduler restart identity absent")
    if "SLURM_RESTART_COUNT" in os.environ:
        c.require(os.environ["SLURM_RESTART_COUNT"] == row["Restarts"], "Stale restart identity")
    else:
        os.environ["SLURM_RESTART_COUNT"] = row["Restarts"]
    intent = c.checked(binding["intent"])
    c.require(row.get("Comment") == "dsmm-s7-native:" + intent["nonce"]
              and intent["configuration"] == binding["configuration"], "Actual feature pre-RPC intent differs")
    def seconds(value):
        days, clock = value.split("-", 1) if "-" in value else ("0", value)
        hours, minutes, second = map(int, clock.split(":"))
        return int(days) * 86400 + hours * 3600 + minutes * 60 + second
    remaining = seconds(row["TimeLimit"]) - seconds(row["RunTime"])
    c.require(remaining > 300, "Actual allocation has no remaining accepted-boundary reserve")
    os.environ["SLURM_JOB_END_TIME"] = str(int(time.time()) + remaining)
    return dict(job_id=job, restart=int(row["Restarts"]), node=row["NodeList"], start=row["StartTime"],
                raw_slurm=raw)


def validate_inputs(spec):
    pool = c.checked(spec["inputs"]["pool"])
    rows = c.checked(pool["records_json"])
    ids = pool["record_ids"]
    c.require(pool["engineering_only"] is True and pool["count"] == len(rows) == len(ids) == 1024
              and len(set(ids)) == 1024 and len(pool["source_indices"]) == 1024,
              "Exact original1024 engineering construction required, not scientific pool")
    c.require(ids == [f'{index:06d}:{row.get("id", "")}' for index, row in zip(pool["source_indices"], rows)],
              "Original engineering source-index identity differs")
    c.checked(pool["protected_exclusions"])
    c.checked(pool["image_manifest"])
    eligibility = c.checked(spec["inputs"]["eligibility"])
    c.require(eligibility["missing_media_count"] == 0 and eligibility["protected_QA_image_RGB_overlap_count"] == 0,
              "Exact engineering media/protection evidence required")
    if eligibility["pool"] != spec["inputs"]["pool"]:
        c.require(spec["phase"] == "tiny8" and pool["source_Babel_pool"] == eligibility["pool"]
                  and all(spec["inputs"]["eligibility"][k] == pool["source_Babel_eligibility"][k]
                          for k in ("bytes", "sha256")), "Engineering eligibility belongs to another construction")
        proof = c.checked(spec["rebind_provenance"])
        c.require(proof["source_pool"] == pool["source_Babel_pool"]
                  and proof["target_pool"] == spec["inputs"]["pool"]
                  and proof["canonical_image_bytes_SHA_RGB_and_dimensions_identical"] is True,
                  "Exact engineering-only cross-site provenance missing")
    assets = c.checked(spec["inputs"]["assets"])
    environment = c.checked(spec["inputs"]["environment"])
    if spec["phase"] == "tiny8":
        from experiments.root_native_baselines_20261023 import pins
        c.require(assets["status"] == "PASS_PINNED_TINY_PROXY_ASSETS", "Actual exact Tiny asset provider required")
        for role in ("tiny_proxy", "tiny_siglip"):
            pins.verify_tiny_snapshot(assets["assets"][role], role)
    else:
        from experiments.root_native_campaign_v3_20261023.config import ASSET_PINS
        c.require(assets["status"] == "PASS_ASSETS", "Actual exact native Stage1 asset provider required")
        for role, pin in ASSET_PINS.items():
            c.require((assets["assets"][role]["repository"], assets["assets"][role]["revision"]) == pin,
                      "Native Stage1 model/projector/metadata revision differs")
    c.require(Path(sys.executable).absolute() == Path(environment["python_executable"]).absolute(),
              "Use the existing exact isolated interpreter, not a shared environment")
    c.require(environment["versions"]["torch"].startswith("2.2.0" if spec["phase"] == "tiny8" else "2.1.2")
              and environment["versions"]["transformers"] == "4.37.2", "Actual operator runtime receipt differs")
    for asset in assets["assets"].values():
        for description in asset["files"]:
            c.require(c.desc(description["path"]) == {k: description[k] for k in ("path", "bytes", "sha256")},
                      "Existing exact asset bytes differ; no fallback download")
    return pool, rows, ids, environment


def tiny_runtime_media(spec, pool, rows, ids, exact, observed, attempt):
    from experiments.root_native_media_transport_20261023 import transport
    by_id = dict(zip(ids, rows))
    selected = [by_id[key] for key in exact["record_ids"]]
    selected_desc = c.seal(attempt / "TRANSPORT_RECORDS_ONLY_NOT_FEATURE_DATASET.json", selected)
    index = c.checked(spec["transport"]["locator_index"])
    claim = c.seal(attempt / "TRANSPORT_CLAIM.json", dict(
        status="ACTIVE_EXCLUSIVE_MEDIA_TRANSPORT_CLAIM", owner=c.OWNER, job_id=observed["job_id"],
        logical_pool=spec["inputs"]["pool"], records_json=selected_desc,
        transport_index=spec["transport"]["locator_index"]))
    template = dict(spec["transport"])
    template.update(schema="native-media-transport-v1", scope="ENGINEERING_ONLY", ownership=claim,
                    canonical_image_manifest=pool["image_manifest"], reserve_bytes=1 << 30,
                    reserve_inodes=1000, deadline_reserve_seconds=300)
    c.require(index["logical_pool"] == spec["inputs"]["pool"], "Tiny transport index has another construction")
    transport_desc = c.seal(attempt / "TRANSPORT.json", template)
    alias = transport.prepare(spec["inputs"]["pool"], selected_desc, transport_desc, attempt / "media")
    value = transport.validate(alias, spec["inputs"]["pool"], selected_desc)
    return value, alias


def run(binding_path):
    started = time.monotonic()
    binding = c.read(binding_path)
    spec = c.checked(binding["configuration"])
    c.require(spec["schema"] == "s7-native-feature-qualification-spec-v1" and spec["owner"] == c.OWNER
              and spec["phase"] in {"tiny8", "datatailor32"}, "Only bounded S7 native feature qualification")
    c.verify_sources(spec["sources"])
    review = c.checked(binding["review"])
    c.require(review["status"] == "PASS_EXACT_S7_PREREQUISITE_SOURCE_AND_CONFIG"
              and review["configuration"] == binding["configuration"] and review["sources"] == spec["sources"],
              "Genuine S6 exact source/spec review required")
    observed = actual(binding, spec)
    c.verify_sources(spec["dependencies"])
    output = Path(spec["output"])
    approved = (Path("/data/user_data/luyangz/native_campaign_20261023") if spec["site"] == "babel"
                else Path("/scratch/user/u.lz341613/dsmm_native_campaign_20261023"))
    c.require(output.is_relative_to(approved), "New engineering output outside approved root")
    output.mkdir(parents=True, exist_ok=True)
    c.require(shutil.disk_usage(output).free >= (2 << 30), "Insufficient actual engineering output capacity")
    c.seal(output / "ACQUISITION_CLAIM.json", dict(
        status="ACTIVE_EXCLUSIVE_S7_NATIVE_FEATURE_ENGINEERING", owner=c.OWNER,
        job_id=observed["job_id"], function=spec["function"], reviewed_spec=binding["configuration"],
        logical_inputs=spec["inputs"], scope="ENGINEERING_ONLY_NOT_FULL_POOL"))
    attempt = output / ("s7-attempt-" + observed["job_id"] + "-r" + str(observed["restart"]))
    c.seal(attempt / "START.json", dict(actual=observed, spec=binding["configuration"], UTC=c.now(),
                                       no_training_or_common_mask_repair=True))
    fs = os.statvfs(output)
    c.require(fs.f_favail >= 1000, "Insufficient actual engineering output inodes")
    c.seal(attempt / "ENGINEERING_CAPACITY.json", dict(
        free_bytes=fs.f_bavail * fs.f_frsize, free_inodes=fs.f_favail, bounded_output_bytes=256 << 20,
        production_quota_or_collective_student_reservation=False, no_checkpoint_or_base_weight_writes=True))
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    c.require(hard == resource.RLIM_INFINITY or hard >= (256 << 20), "Inherited raw-evidence file ceiling too small")
    resource.setrlimit(resource.RLIMIT_FSIZE, (256 << 20, hard))
    pool, rows, ids, environment = validate_inputs(spec)
    profile = dict(python=environment["python_executable"], code_root=str(c.ROOT),
                   cache_root=spec["cache_root"], output_root=str(approved), site=spec["site"],
                   account="cx" if spec["site"] == "babel" else "158416130568",
                   partition="general" if spec["site"] == "babel" else "gpu",
                   qos="normal", gpu_type="L40S" if spec["site"] == "babel" else "h100")
    profile_desc = c.seal(attempt / "ACTUAL_PROFILE.json", profile)
    os.environ.update(HF_HOME=spec["cache_root"], HF_HUB_CACHE=spec["cache_root"] + "/hub",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    if spec["phase"] == "tiny8":
        from experiments.root_native_tiny_qualification_20261023 import qualifier, prepare
        from experiments.root_native_baselines_20261023 import features
        exact = prepare.one_batch(ids, rows)
        config = dict(schema="tiny-native-batch8-qualification-config-v1", owner="Root",
                      method="coincide", role="qualify_features", phase="tiny8",
                      measurement_contract=features.CONTRACTS["coincide"], feature_assets=spec["inputs"]["assets"],
                      environment=spec["inputs"]["environment"], sources={**spec["dependencies"], **spec["sources"]},
                      pool=spec["inputs"]["pool"], output=str(output), site_profile=profile_desc,
                      exact_batch=exact, shard_index=exact["shard_index"], shard_count=exact["shard_count"])
        identity = features.measurement_identity(config, pool)
        c.seal(attempt / "EXACT_AUTHOR_BATCH.json", dict(exact_batch=exact, logical_pool=spec["inputs"]["pool"],
                                                      full1024_data_path=pool["records_json"],
                                                      measurement_identity=identity))
        media, alias = tiny_runtime_media(spec, pool, rows, ids, exact, observed, attempt)
        class MediaFeatures:
            def __getattr__(self, name):
                return getattr(features, name)
            def coincide(self, configuration, logical_pool, original_rows, original_ids, *args, **kwargs):
                runtime_pool = dict(logical_pool, image_root=media["image_root"],
                                    _verified_image_assets=media["assets"])
                return features.coincide(configuration, runtime_pool, original_rows, original_ids, *args, **kwargs)
        def validated(_):
            return profile, environment, pool, rows, ids, identity
        namespace = dict(vars(qualifier), validate=validated,
                         actual=lambda *_: observed["raw_slurm"], features=MediaFeatures())
        native_schema = "tiny-native-batch8-qualification-binding-v1"
        runner = types.FunctionType(qualifier.main.__code__, namespace)
    else:
        from experiments.root_native_datatailor_20261023 import provider, features
        selected = features.engineering_ids(rows, ids)
        c.require(selected == spec["engineering_ids"], "Frozen DataTailor32 record identities changed")
        config = dict(schema="native-datatailor-feature-job-v1", id="s7-datatailor32-v1",
                      owner="Root", method="datatailor", role="qualify_features",
                      engineering_ids=selected, engineering_pause_after=16, shard_index=0, shard_count=1,
                      measurement_contract=features.CONTRACT, native_sources=spec["native_sources"],
                      feature_assets=spec["inputs"]["assets"], environment=spec["inputs"]["environment"],
                      pool=spec["inputs"]["pool"], output=str(output),
                      sources={**spec["dependencies"], **spec["sources"]},
                      site_profile=profile_desc, allocation=dict(gpus=1, cpus=8, memory_mb=65536, minutes=55, reserve_seconds=300))
        identity = features.identity(config, pool)
        c.seal(attempt / "EXACT_DATATAILOR32.json", dict(record_ids=selected, measurement_identity=identity,
                                                     full1024_construction=spec["inputs"]["pool"]))
        def validated(_):
            return profile, pool, rows, ids, selected, identity
        namespace = dict(vars(provider), validate=validated, actual=lambda *_: observed["raw_slurm"])
        native_schema = "native-datatailor-feature-binding-v1"
        runner = types.FunctionType(provider.main.__code__, namespace)
    native_config = c.seal(attempt / "DERIVED_NATIVE_CONFIG.json", config)
    native_binding = c.seal(attempt / "DERIVED_NATIVE_BINDING.json", dict(
        schema=native_schema, owner="Root", job_id=observed["job_id"], configuration=native_config,
        intent=binding["intent"], delegated_actor=c.OWNER, reviewed_spec=binding["configuration"]))
    old_argv = sys.argv[:]
    try:
        sys.argv = ["s7-source-reviewed-native-feature", "--binding", native_binding["path"]]
        code = runner()
    finally:
        sys.argv = old_argv
    c.seal(attempt / "RESULT.json", dict(status="RAW_ENGINEERING_EVIDENCE_ONLY_PENDING_S6_RUNTIME_REVIEW",
                                        phase=spec["phase"], native_return_code=code, elapsed_seconds=time.monotonic() - started,
                                        actual=observed, native_config=native_config,
                                        full_pool_or_feature_runtime_PASS_created=False))
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.binding))
