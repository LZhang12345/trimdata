"""[v1d: v1c plus diagnosed-failure reset: a failed link with a FAILURE_DIAGNOSIS-<link>.json (cause+repair) ends the consecutive-failure streak, so a chain stalled by an external, repaired cause (2026-11-05 hf_cache sweeps) can resume; undiagnosed failures still stall after 3] [v1c: v1b on Babel General (normal QoS, 12 h links; same trainer, checkpoints, control and parity)] [v1b: duplicate-link tie-break, single continuation, resume command] Tool2 resumable <=1 h Babel Preempt training links (port of Tool3's reviewed requeue chain).

Every link trains from the newest COMPLETE checkpoint until its deadline, commits a checkpoint whose
marker embeds the progress audit, keeps only that checkpoint and exits; a finished link requeues
itself (preemption requeues automatically), so one submission carries the whole cell. A success
RESULT.json ends the cell. Differences from Tool3: Babel only; the trainer and selections are pinned
by hash; the first link of every cell is admitted held under the shared ADMISSION.lock with
INTENT -> ACK -> scheduler/spool readback -> release.
"""

import argparse
import base64
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

ARMS = ("uniform_scored_rand", "cqmr_scored_rand", "m1l_scored_rand", "uniform_vndrop", "cqmr_vndrop", "m1l_vndrop",
        "m1_local_rand")
TAGS = {"uniform_scored_rand": "unisr", "cqmr_scored_rand": "cqsr", "m1l_scored_rand": "m1lsr",
        "uniform_vndrop": "univn", "cqmr_vndrop": "cqvn", "m1l_vndrop": "m1lvn", "m1_local_rand": "m1lr"}
HOME = "/home/luyangz/data_selection_multimodal"
SITE = {
    "home": HOME, "source": "experiments/helper_tool2_preempt_train_v1d_20261105", "run": "run.sbatch",
    "control": "collaboration/helper-tool2-method-20261101/preempt-train-v1",
    "lock": "collaboration/babel-two-track-20261019/ADMISSION.lock",
    "output_root": "/data/hf_cache/project_checkpoints/DS-MultiModal/helper-tool2-preempt-train-v1-20261104",
    "python": "/data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python",
    "root": HOME + "/coincide_native_exec_src_v2_20261028",
    "hf_home": "/data/hf_cache", "torch_extensions": "/data/hf_cache/project_envs/DS-MultiModal/torch_extensions",
    "sbatch": ["--partition=general", "--account=cx", "--qos=normal", "--gres=gpu:L40S:1",
               "--exclude=babel-p9-20", "--cpus-per-task=8", "--mem=64G"],
    "user": "luyangz", "user_home": "/home/luyangz", "save_steps": 20, "reserve_seconds": 600,
}
MODEL_SHA256 = "681ef89c8e40152d3f4f901ec77cfa7730ed6336c73ae409907cd48c228843b7"
SELECTIONS = {"path": HOME + "/collaboration/helper-tool2-method-20261101/preempt-train-v1/SELECTIONS_V1.json",
              "bytes": 4030, "sha256": "4f81630c1ecafe6e7bf479ca6289755b1e27ebcb5b41f7a06dd59933c5f238d0"}
TEMPLATE = ("collaboration/root-general-all-plan-20261028/submission-check-20261029/baseline-ready-v1/"
            "cells/coincide-native10-s17/CONFIG_FINAL.json")
POOL = "/data/user_data/luyangz/native_campaign_20261023/root-wholepool-authority-v1/authority/POOL.json"
POOL_SHA256 = "c76040a17ca0dc1e99fea14514618edade3171a9b05f6ccc3e4751a5939e4964"
EXTRA_SOURCES = {
    "experiments/root_native_deterministic_attention_v1_20261024/__init__.py":
        "2bc1ea7614940d05a7e407672097f1323dd7cb85ecaa156b267f7f5ef9bfadc3",
    "experiments/root_native_deterministic_attention_v1_20261024/policy.py":
        "872234b24e1779f4aa7dcb474a82ab1675ac144dc2e00356c7825b599608c1ce",
    "experiments/root_native_deterministic_attention_v2_20261025/__init__.py":
        "7815f5ffeb6384074f66e50aa46e278ff55c1c87ef86d3b475a01cb1e4e114b2",
    "experiments/root_native_deterministic_attention_v2_20261025/policy.py":
        "8e4d809df0a4aa19496c473dc0916b4c6c642694e889635ef5f20d515021f7b0",
    "experiments/root_native_operator_trace_v1_20261024/__init__.py":
        "e5f02e2c3fa4ff0c48b4b66f3024d3741ffb9c42f959a7c2b931a85284137dbd",
    "experiments/root_native_operator_trace_v1_20261024/runtime_trace.py":
        "09fc86b731f347ea0b8c8d1372ca87c2c555a604b898eaea7c6e6e6e016a1652",
}
TEMPLATE_SHA256 = "450160a8c561fbd33f4e98acca9805cc04ed727955dfb253c58e461cdd0fe9ec"
ASSET_RECEIPT_SHA256 = "b6fe9f87e7fc20809f6176ad75014469e16fdfbc2120fa83cd1c512850bed027"
MAX_LINKS, MAX_CONSECUTIVE_FAILURES, FRESH_JOB_AFTER_RESTARTS = 150, 3, 4
SUCCESS = ("ACTUAL_FULL_NATIVE_TOOL2_PREEMPT_EPOCH_NOT_EVALUATED", "QUALIFICATION_COMPLETE")
ACTIVE = ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED", "REQUEUE_HOLD",
          "REQUEUE_FED", "RESIZING", "SIGNALING", "STAGE_OUT", "STOPPED")
SUBMIT_LIMIT = 50


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp-" + str(os.getpid()))
    temporary.write_text(json.dumps(value, sort_keys=True, indent=1))
    os.replace(temporary, path)


def utc():
    return datetime.utcnow().isoformat() + "Z"


def cell_name(arm, selection_seed, training_seed):
    return "t2pr-" + arm + "-sel" + str(selection_seed) + "-train" + str(training_seed)


def job_name(arm, selection_seed, training_seed, variant=None):
    if variant:
        return "dsmm-t2-pr-resume-qual-" + variant
    return "dsmm-t2-pr-" + TAGS[arm] + "-s" + str(selection_seed) + "-t" + str(training_seed)


def scheduler(*args):
    env = {"PATH": "/usr/bin:/bin", "HOME": SITE["user_home"], "USER": SITE["user"], "LOGNAME": SITE["user"],
           "LC_ALL": "C"}
    return subprocess.run(list(args), capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL,
                          env=env)


def result_ok(output):
    path = Path(output) / "RESULT.json"
    return path.is_file() and json.loads(path.read_bytes()).get("status") in SUCCESS


def active_jobs(name):
    queue = scheduler("/usr/bin/squeue", "-h", "-u", SITE["user"], "-n", name, "-o", "%i|%T")
    require(queue.returncode == 0, "Queue query failed: " + queue.stderr.strip())
    rows = [line.split("|") for line in queue.stdout.splitlines() if line.strip()]
    return {job: state for job, state in rows if state in ACTIVE}


def argv(control, args, dependency, variant=None, time_limit="12:00:00", hold=False):
    command = ["/usr/bin/sbatch", "--parsable", "--export=NIL", "--time=" + time_limit, "--nodes=1",
               "--ntasks=1", *SITE["sbatch"], "--job-name=" + job_name(args.arm, args.selection_seed,
                                                                       args.training_seed, variant),
               "--chdir=" + SITE["home"], "--requeue", "--open-mode=append",
               "--output=" + str(control / "job-%j.log")]
    if hold:
        command.insert(2, "--hold")
    if dependency:
        command.append("--dependency=" + dependency)
    command += [str(Path(SITE["home"]) / SITE["source"] / SITE["run"]),
                "qualify-worker-" + variant if variant else "worker",
                args.arm, str(args.selection_seed), str(args.training_seed)]
    return command


def submit(control, args, dependency, variant=None, time_limit="12:00:00", hold=False):
    command = argv(control, args, dependency, variant, time_limit, hold)
    intent = control / ("SUBMIT_INTENT-" + str(time.time_ns()) + ".json")
    write_json(intent, {"argv": command, "dependency": dependency, "utc": utc()})
    result = scheduler(*command)
    job = result.stdout.strip().split(";")[0]
    ok = result.returncode == 0 and re.fullmatch(r"[1-9][0-9]*", job) is not None
    write_json(control / ("SUBMIT_ACK-" + (job if ok else "FAILED-" + str(time.time_ns())) + ".json"),
               {"intent": str(intent), "returncode": result.returncode, "stdout": result.stdout,
                "stderr": result.stderr, "job_id": job if ok else None, "utc": utc()})
    require(ok, "Submission failed: " + result.stderr.strip())
    return job, command


def fields(raw):
    """scontrol -o fields; SubmitLine contains spaces and is followed by WorkDir=."""
    line = re.search(r"\bSubmitLine=(.*?) WorkDir=", raw)
    values = dict(token.split("=", 1) for token in raw.split() if "=" in token)
    values["SubmitLine"] = line.group(1) if line else ""
    return values


def held_readback_and_release(control, job, command, name):
    """Verify the held request (identity, resources, requeue, argv, exact spool) before releasing it."""
    raw = scheduler("/usr/bin/scontrol", "show", "job", "-o", job).stdout
    row = fields(raw)
    expected = {"JobId": job, "JobName": name, "Partition": "general", "QOS": "normal", "Account": "cx",
                "TresPerNode": "gres/gpu:L40S:1", "ExcNodeList": "babel-p9-20", "Requeue": "1",
                "Command": str(Path(SITE["home"]) / SITE["source"] / SITE["run"]), "WorkDir": SITE["home"],
                "JobState": "PENDING", "Reason": "JobHeldUser"}
    require(all(row.get(k) == v for k, v in expected.items()) and shlex.split(row.get("SubmitLine", "")) == command,
            "Held request identity/resources/argv differ: " + json.dumps({k: row.get(k) for k in expected}))
    spool = subprocess.check_output(["/usr/bin/scontrol", "write", "batch_script", job, "-"], timeout=30,
                                    stdin=subprocess.DEVNULL)
    require(spool == (Path(SITE["home"]) / SITE["source"] / SITE["run"]).read_bytes(),
            "Held spool differs from the reviewed run.sbatch")
    write_json(control / ("HELD_READBACK-" + job + ".json"),
               {"job_id": job, "scheduler": raw, "spool_base64": base64.b64encode(spool).decode()})
    write_json(control / ("RELEASE_INTENT-" + job + ".json"), {"job_id": job, "utc": utc()})
    released = scheduler("/usr/bin/scontrol", "release", job)
    after = scheduler("/usr/bin/scontrol", "show", "job", "-o", job).stdout
    write_json(control / ("RELEASE_RESULT-" + job + ".json"),
               {"returncode": released.returncode, "stderr": released.stderr, "scheduler_after": after})
    require(released.returncode == 0 and fields(after).get("Reason") != "JobHeldUser", "Release failed")


def locked():
    lock = os.open(Path(SITE["home"]) / SITE["lock"], os.O_RDWR)
    fcntl.flock(lock, fcntl.LOCK_EX)
    return lock


def preempt_requests():
    queue = scheduler("/usr/bin/squeue", "-h", "-u", SITE["user"], "-q", "normal", "-o", "%i")
    require(queue.returncode == 0, "Queue query failed")
    return len([line for line in queue.stdout.splitlines() if line.strip()])


def link_history(control):
    """Ordered (link, exit-or-None) for every started link; a start without exit was killed/preempted."""
    starts = sorted((json.loads(p.read_bytes())["started_ns"], p.name[len("LINK_START-"):-5])
                    for p in control.glob("LINK_START-*.json"))
    history = []
    for _, link in starts:
        exit_path = control / ("LINK_EXIT-" + link + ".json")
        history.append((link, json.loads(exit_path.read_bytes()) if exit_path.is_file() else None))
    return history


def consecutive_failures(control, current):
    count = 0
    for link, value in reversed(link_history(control)):
        if link == current:
            continue
        if value is None:
            continue  # preemption; not evidence of a code failure
        elif value["returncode"] == 0 or value.get("progress"):
            break
        elif diagnosed(control, link):
            break  # v1d/v2d: failures up to a diagnosed, repaired link are not evidence against later links
        else:
            count += 1
    return count


def diagnosed(control, link):
    path = control / ("FAILURE_DIAGNOSIS-" + link + ".json")
    if not path.is_file():
        return False
    value = json.loads(path.read_bytes())
    return (value.get("link") == link and isinstance(value.get("cause"), str) and len(value["cause"]) >= 20
            and isinstance(value.get("repair"), str) and len(value["repair"]) >= 20)


def job_end_epoch(job):
    raw = scheduler("/usr/bin/scontrol", "show", "job", "-o", job).stdout
    end = re.search(r"\bEndTime=(\S+)", raw).group(1)
    return int(datetime.fromisoformat(end).timestamp())


def inputs(args):
    home = Path(SITE["home"])
    require(Path(SELECTIONS["path"]).stat().st_size == SELECTIONS["bytes"] and sha(SELECTIONS["path"]) == SELECTIONS["sha256"],
            "Pinned selections manifest changed")
    manifest = json.loads(Path(SELECTIONS["path"]).read_bytes())
    require(manifest["selection_seed"] == args.selection_seed, "Only the materialized selection seed is available")
    cell = manifest["cells"][args.arm]
    records = Path(cell["path"])
    require(records.stat().st_size == cell["records_bytes"] and sha(records) == cell["records_sha256"],
            "Materialized selected records differ from the pinned manifest")
    require(sha(POOL) == POOL_SHA256, "Pool authority changed")
    pool = json.loads(Path(POOL).read_bytes())
    require(sha(home / TEMPLATE) == TEMPLATE_SHA256, "Template packet changed")
    template = json.loads((home / TEMPLATE).read_bytes())
    assets_ref = template["providers"]["assets"]
    require(assets_ref["sha256"] == ASSET_RECEIPT_SHA256 and sha(assets_ref["path"]) == ASSET_RECEIPT_SHA256,
            "Asset receipt changed")
    assets = json.loads(Path(assets_ref["path"]).read_bytes())["assets"]
    return {"path": str(records), "bytes": cell["records_bytes"], "sha256": cell["records_sha256"]}, \
        pool["image_root"], assets, template["sources"]


def clean_residual_checkpoints(output):
    training = Path(output) / "training"
    removed = []
    for path in (training.iterdir() if training.is_dir() else ()):
        if path.is_dir() and (path.name.startswith("checkpoint-") or path.name.startswith("tmp-checkpoint-")):
            shutil.rmtree(path)
            removed.append(path.name)
    return removed


def continue_requeue(control, args, job, variant):
    pending = sorted(o for o, s in active_jobs(job_name(args.arm, args.selection_seed, args.training_seed, variant)).items()
                     if o != job and s == "PENDING")
    if pending:
        # v1b: a pending link already carries this cell; never add a second continuation.
        return {"mode": "existing_pending_link", "jobs": pending}
    restarts = int(os.environ.get("SLURM_RESTART_COUNT") or 0)
    if restarts >= FRESH_JOB_AFTER_RESTARTS:
        lock = locked()
        try:
            fresh, _ = submit(control, args, "afterany:" + job, variant)
            return {"mode": "fresh_job", "job": fresh}
        except Exception as error:
            fallback_error = repr(error)
        finally:
            os.close(lock)
    else:
        fallback_error = None
    result = scheduler("/usr/bin/scontrol", "requeue", job)
    report = {"mode": "self_requeue", "returncode": result.returncode, "stderr": result.stderr.strip(),
              "fresh_job_error": fallback_error}
    if result.returncode != 0 and fallback_error is None:
        lock = locked()
        try:
            report["fresh_job"], _ = submit(control, args, "afterany:" + job, variant)
        except Exception as error:
            report["fresh_job_error"] = repr(error)
        finally:
            os.close(lock)
    return report


def worker(args, variant=None):
    home = Path(SITE["home"])
    job = os.environ["SLURM_JOB_ID"]
    restarts = os.environ.get("SLURM_RESTART_COUNT") or "0"
    link = job + "-r" + restarts
    cell = cell_name(args.arm, args.selection_seed, args.training_seed)
    control = home / SITE["control"] / (("qualify-" + variant) if variant else cell)
    output = Path(SITE["output_root"]) / (cell + ("-qualify-" + variant if variant else ""))
    started = time.time()
    write_json(control / ("LINK_START-" + link + ".json"), {"started_ns": time.time_ns(), "job": job,
                                                           "restart": restarts, "utc": utc(),
                                                           "node": os.uname().nodename})

    def finish(code, **values):
        write_json(control / ("LINK_EXIT-" + link + ".json"),
                   dict(values, returncode=code, elapsed_seconds=round(time.time() - started, 1)))
        return code

    if result_ok(output):
        return finish(0, status="ALREADY_COMPLETE", residual_removed=clean_residual_checkpoints(output))
    others = {other: state for other, state in active_jobs(job_name(
        args.arm, args.selection_seed, args.training_seed, variant)).items() if other != job}
    # v1b: deterministic tie-break -- only a lower-id active link blocks this one, so two links started together
    # cannot both exit (the v1 race that ended a chain with no successor).
    blockers = sorted(int(o) for o, s in others.items() if s not in ("PENDING", "COMPLETING") and int(o) < int(job))
    if blockers:
        return finish(0, status="DUPLICATE_ACTIVE_LINK_NOT_TRAINING", others=others, blocker=str(blockers[0]))
    history = link_history(control)
    if consecutive_failures(control, link) >= MAX_CONSECUTIVE_FAILURES or len(history) > MAX_LINKS:
        return finish(3, status="CHAIN_STALLED", links=len(history))
    code, status = None, None
    try:
        job_end = job_end_epoch(job)
        deadline = job_end - SITE["reserve_seconds"]
        records, image_root, assets, sources = inputs(args)
        parent = Path(SITE["output_root"])
        parent.mkdir(mode=0o700, exist_ok=True)
        output.mkdir(mode=0o700, exist_ok=True)
        model = home / SITE["source"] / "model.py"
        require(sha(model) == MODEL_SHA256, "Pinned resumable trainer changed")
        config = {"role": "TOOL2_PREEMPT_RESUMABLE_FULL_EPOCH_NOT_EVALUATED", "job_id": job, "link_id": link,
                  "arm": args.arm, "selection_seed": args.selection_seed, "training_seed": args.training_seed,
                  "site": "babel", "input": records, "image_root": image_root, "assets": assets,
                  "sources": dict(sources, **EXTRA_SOURCES), "root": SITE["root"], "output": str(output),
                  "model_source": {"path": str(model), "sha256": MODEL_SHA256},
                  "deadline_epoch": deadline, "save_steps": SITE["save_steps"], "media_staging": None}
        if variant:
            config.update(role="TOOL2_RESUME_QUALIFICATION_NOT_SCIENCE", qualification_max_steps=4, save_steps=2,
                          stop_after_steps=2 if variant == "chunked" else None)
        config_path = control / ("MODEL_CONFIG-" + link + ".json")
        write_json(config_path, config)
        env = {"PATH": "/usr/bin:/bin", "HOME": SITE["user_home"], "USER": SITE["user"], "LC_ALL": "C",
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
               "HF_HOME": SITE["hf_home"], "TORCH_EXTENSIONS_DIR": SITE["torch_extensions"],
               "PYTHONDONTWRITEBYTECODE": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        env.update({key: value for key, value in os.environ.items()
                    if key.startswith("SLURM_") or key in ("CUDA_VISIBLE_DEVICES", "TMPDIR")})
        command = [SITE["python"], "-B", "-X", "pycache_prefix=/dev/null/no-cache", "-m", "torch.distributed.run",
                   "--standalone", "--nnodes=1", "--nproc-per-node=1", "--no-python", SITE["python"], "-B", "-X",
                   "pycache_prefix=/dev/null/no-cache", str(model), "--config", str(config_path)]
        budget = job_end - 60 - time.time()
        if deadline - time.time() < 300:
            code, status = 0, "NO_TRAINING_TIME_LEFT"
        else:
            try:
                code = subprocess.run(command, env=env, stdin=subprocess.DEVNULL, timeout=budget).returncode
            except subprocess.TimeoutExpired:
                code = 124
    except Exception as error:
        code, status = 2, "LINK_SETUP_FAILED: " + repr(error)[-1500:]
    done = result_ok(output)
    chunks = sorted((output / "chunks").glob("chunk-" + link + ".json"))
    chunk = json.loads(chunks[0].read_bytes()) if chunks else None
    if chunk:
        chunk = {key: value for key, value in chunk.items() if key not in ("consumed_indices", "step_events")}
    progress = bool(chunk and (chunk.get("end_step") or 0) > chunk.get("resume_step", 0)) or done
    failed = code != 0 and not progress
    stalled = consecutive_failures(control, link) + failed >= MAX_CONSECUTIVE_FAILURES
    finish(code, status=status or ("RESULT_SUCCESS" if done else "CHUNK_EXIT"), result_exists=done, progress=progress,
           chunk_record=chunk)
    if not done and not stalled:
        write_json(control / ("CONTINUATION-" + link + ".json"), continue_requeue(control, args, job, variant))
    return code


def admit_first(args, control, variant=None, time_limit="12:00:00"):
    """Held first link under the shared lock: census, INTENT/ACK, readback, release."""
    name = job_name(args.arm, args.selection_seed, args.training_seed, variant)
    lock = locked()
    try:
        active = active_jobs(name)
        require(not active, "Cell already has active jobs; refusing a duplicate start: " + json.dumps(active))
        count = preempt_requests()
        require(count < SUBMIT_LIMIT, "Actual General submission limit reached")
        job, command = submit(control, args, None, variant, time_limit, hold=True)
        held_readback_and_release(control, job, command, name)
        return job
    finally:
        os.close(lock)


def qualify_compare(args):
    """Exact parity readout: resumed steps 3-4 versus continuous steps 3-4, final adapters and exposure."""
    cell = cell_name(args.arm, args.selection_seed, args.training_seed)
    out = {variant: Path(SITE["output_root"]) / (cell + "-qualify-" + variant) for variant in ("chunked", "continuous")}
    results = {variant: json.loads((path / "RESULT.json").read_bytes()) for variant, path in out.items()}
    chunks = {variant: sorted((json.loads(p.read_bytes()) for p in (path / "chunks").glob("chunk-*.json")),
                              key=lambda value: value["resume_step"]) for variant, path in out.items()}
    resumed = [value for value in chunks["chunked"] if value["resume_step"] == 2]
    continuous = [value for value in chunks["continuous"] if value["resume_step"] == 0]
    require(len(resumed) == 1 and len(continuous) == 1, "Exact qualification chunk records required")
    resumed_losses, continuous_losses = resumed[0]["first_losses"], continuous[0]["last_losses"]
    differences = [abs(a - b) for a, b in zip(resumed_losses, continuous_losses)]
    files = {variant: {name: value["sha256"] for name, value in results[variant]["output_files"].items()}
             for variant in out}
    report = {
        "schema": "tool2-resume-parity-v1", "utc": utc(), "statuses": {v: results[v]["status"] for v in out},
        "audit_checks": {v: results[v]["audit_checks"] for v in out},
        "resumed_restores": resumed[0]["restores"], "resumed_microbatches": len(resumed_losses),
        "loss_max_abs_difference_steps_3_4": max(differences) if differences else None,
        "loss_bitwise_equal_steps_3_4": resumed_losses == continuous_losses and len(resumed_losses) == 64,
        "exposure_sha256": {v: results[v].get("exposure_sha256") for v in out},
        "exposure_equal": results["chunked"].get("exposure_sha256") == results["continuous"].get("exposure_sha256"),
        "adapter_files_equal": {name: files["chunked"].get(name) == files["continuous"].get(name)
                                for name in sorted(set(files["chunked"]) | set(files["continuous"]))
                                if name.startswith(("adapter_model", "non_lora"))},
        "chunked_links": sorted(p.name for p in (out["chunked"] / "chunks").glob("chunk-*.json")),
    }
    checks = {
        "both_success": all(results[v]["status"] == "QUALIFICATION_COMPLETE" for v in out),
        "resumed_exactly_once": resumed[0]["restores"] == 1,
        "losses_bitwise_equal": report["loss_bitwise_equal_steps_3_4"],
        "exposure_equal": report["exposure_equal"] and None not in report["exposure_sha256"].values(),
        "adapters_equal": len(report["adapter_files_equal"]) >= 2 and all(report["adapter_files_equal"].values()),
    }
    report.update(checks=checks, status="PASS_RESUME_PARITY" if all(checks.values()) else "FAIL_RESUME_PARITY")
    write_json(Path(SITE["home"]) / SITE["control"] / "qualify-continuous" / "PARITY.json", report)
    print(json.dumps(report, indent=1))
    return 0 if report["status"] == "PASS_RESUME_PARITY" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("start", "resume", "worker", "qualify-start", "qualify-worker-chunked",
                                            "qualify-worker-continuous", "qualify-compare"))
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("selection_seed", type=int, choices=(1234,))
    parser.add_argument("training_seed", type=int, choices=(17, 29, 43))
    args = parser.parse_args()
    if args.command == "worker":
        return worker(args)
    if args.command.startswith("qualify-worker-"):
        return worker(args, variant=args.command.rsplit("-", 1)[1])
    if args.command == "qualify-compare":
        return qualify_compare(args)
    if args.command == "qualify-start":
        jobs = {}
        for variant in ("chunked", "continuous"):
            control = Path(SITE["home"]) / SITE["control"] / ("qualify-" + variant)
            control.mkdir(mode=0o700, parents=True, exist_ok=False)
            jobs[variant] = admit_first(args, control, variant, "00:40:00")
        print(json.dumps(jobs))
        return 0
    parity = json.loads((Path(SITE["home"]) / SITE["control"] / "qualify-continuous" / "PARITY.json").read_bytes())
    require(parity["status"] == "PASS_RESUME_PARITY", "Science links require this trainer's actual resume parity PASS")
    control = Path(SITE["home"]) / SITE["control"] / cell_name(args.arm, args.selection_seed, args.training_seed)
    control.mkdir(mode=0o700, parents=True, exist_ok=True)
    require(not result_ok(Path(SITE["output_root"]) / control.name), "Cell already complete")
    if args.command == "resume":
        # v1b: restart a chain with no active link from its newest complete checkpoint (same control/output).
        require(list(control.glob("SUBMIT_ACK-*.json")), "Resume requires an admitted cell")
        write_json(control / ("RESUME_INTENT-" + str(time.time_ns()) + ".json"),
                   {"reason": "no active link; chain v1b resume", "utc": utc()})
        print(json.dumps({"cell": control.name, "resumed": [admit_first(args, control)]}))
        return 0
    require(not list(control.glob("SUBMIT_ACK-*.json")), "Cell already admitted; reconcile, never duplicate")
    print(json.dumps({"cell": control.name, "submitted": [admit_first(args, control)]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
