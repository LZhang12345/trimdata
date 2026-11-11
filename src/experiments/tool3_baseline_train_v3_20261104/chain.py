"""Resumable <=1 h training links for registered baseline cells (Tool3 baseline program v3, Babel).

Cells come from the registry CELLS.json (cell -> records file/bytes/sha256, K, training seed). Each cell
is one --requeue Preempt or General job; preemption requeues it and a finished link requeues itself.
Every link trains from the newest COMPLETE checkpoint; a success RESULT.json ends the cell.
"""

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

HOME_DIR = "/home/luyangz/data_selection_multimodal"
BASE_SBATCH = ["--account=cx", "--gres=gpu:L40S:1", "--exclude=babel-p9-20", "--cpus-per-task=8", "--mem=64G"]
SITES = {
    "babel": {
        "home": HOME_DIR, "source": "experiments/tool3_baseline_train_v3_20261104", "run": "run.sbatch",
        "control": "collaboration/helper-tool3-baselines-20261104/training",
        "lock": "collaboration/babel-two-track-20261019/ADMISSION.lock",
        "output_root": "/data/hf_cache/project_checkpoints/DS-MultiModal/tool3-baselines-20261104",
        "python": "/data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python",
        "root": HOME_DIR + "/coincide_native_exec_src_v2_20261028",
        "hf_home": "/data/hf_cache", "torch_extensions": "/data/hf_cache/project_envs/DS-MultiModal/torch_extensions",
        "routes": {"preempt": ["--partition=preempt", "--qos=preempt_qos", *BASE_SBATCH],
                   # v3.1 (2026-11-05): fresh General successors keep the agreed Nice 5 (below Tool1's Nice 0).
                   "general": ["--partition=general", "--qos=normal", "--nice=5", *BASE_SBATCH],
                   # v3.2 (2026-11-06): secondary budget-curve cells yield to main-table work (Tool2 V2 receipt).
                   "general_budget": ["--partition=general", "--qos=normal", "--nice=200", *BASE_SBATCH]},
        "user": "luyangz", "user_home": "/home/luyangz",
        "save_steps": 20, "reserve_seconds": 600, "depth": 1, "mode": "requeue",
    },
}
REGISTRY = HOME_DIR + "/collaboration/helper-tool3-baselines-20261104/CELLS.json"
TEMPLATE = ("collaboration/root-general-all-plan-20261028/submission-check-20261029/baseline-ready-v1/"
            "cells/coincide-native10-s17/CONFIG_FINAL.json")
POOL = "/data/user_data/luyangz/native_campaign_20261023/root-wholepool-authority-v1/authority/POOL.json"
POOL_SHA256 = "c76040a17ca0dc1e99fea14514618edade3171a9b05f6ccc3e4751a5939e4964"
# Trainer import closure beyond the 91-file template packet (static AST closure on Babel, CLOSURE.json).
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
MAX_LINKS, MAX_CONSECUTIVE_FAILURES, FRESH_JOB_AFTER_RESTARTS = 600, 3, 4
SUCCESS = ("ACTUAL_FULL_NATIVE_BASELINE_EPOCH_NOT_EVALUATED", "QUALIFICATION_COMPLETE")
ACTIVE = ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED", "REQUEUE_HOLD",
          "REQUEUE_FED", "RESIZING", "SIGNALING", "STAGE_OUT", "STOPPED")


SITE_NAME = "babel"
NAME = re.compile(r"[a-z0-9][a-z0-9-]{2,60}")


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


def scheduler(*args):
    site = SITES[SITE_NAME]
    env = {"PATH": "/usr/bin:/bin", "HOME": site["user_home"], "USER": site["user"], "LOGNAME": site["user"],
           "LC_ALL": "C"}
    return subprocess.run(list(args), capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL,
                          env=env)


def result_ok(output):
    path = Path(output) / "RESULT.json"
    return path.is_file() and json.loads(path.read_bytes()).get("status") in SUCCESS


def active_jobs(site, name):
    queue = scheduler("/usr/bin/squeue", "-h", "-u", site["user"], "-n", name, "-o", "%i|%T")
    require(queue.returncode == 0, "Queue query failed: " + queue.stderr.strip())
    rows = [line.split("|") for line in queue.stdout.splitlines() if line.strip()]
    return {job: state for job, state in rows if state in ACTIVE}


def locked(site):
    lock = os.open(Path(site["home"]) / site["lock"], os.O_RDWR)
    fcntl.flock(lock, fcntl.LOCK_EX)
    return lock


def link_history(control):
    """Ordered (link, exit-or-None) for every started link; a start without exit was killed/preempted."""
    starts = sorted((json.loads(p.read_bytes())["started_ns"], p.name[len("LINK_START-"):-5])
                    for p in control.glob("LINK_START-*.json"))
    history = []
    for _, link in starts:
        exit_path = control / ("LINK_EXIT-" + link + ".json")
        history.append((link, json.loads(exit_path.read_bytes()) if exit_path.is_file() else None))
    return history


def consecutive_failures(site, control, current):
    count = 0
    for link, value in reversed(link_history(control)):
        if link == current:
            continue
        if value is None:
            if site["mode"] == "requeue":
                continue  # preemption on Babel; not evidence of a code failure
            count += 1
        elif value["returncode"] == 0 or value.get("progress"):
            break
        else:
            count += 1
    return count


def job_end_epoch(job):
    raw = scheduler("/usr/bin/scontrol", "show", "job", "-o", job).stdout
    end = re.search(r"\bEndTime=(\S+)", raw).group(1)
    return int(datetime.fromisoformat(end).timestamp())


def clean_residual_checkpoints(output):
    training = Path(output) / "training"
    removed = []
    for path in (training.iterdir() if training.is_dir() else ()):
        if path.is_dir() and (path.name.startswith("checkpoint-") or path.name.startswith("tmp-checkpoint-")):
            shutil.rmtree(path)
            removed.append(path.name)
    return removed


def registry_cell(cell):
    cells = json.loads(Path(REGISTRY).read_bytes())["cells"]
    shorts = [entry["short"] for entry in cells.values()]
    require(len(shorts) == len(set(shorts)), "Registry short names must be unique")
    for name, entry in cells.items():
        require(NAME.fullmatch(name) and not name.endswith("-qualify-chunked") and NAME.fullmatch(entry["short"])
                and type(entry["K"]) is int and entry["K"] >= 512 and type(entry["training_seed"]) is int
                and entry["route"] in SITES["babel"]["routes"]
                and set(entry["records"]) == {"path", "bytes", "sha256"}, "Malformed registry entry: " + name)
    require(cell in cells, "Unregistered cell: " + cell)
    return cells[cell]


def cell_result_ok(output, cell, variant):
    """Success RESULT bound to this exact cell and role (not merely any success status)."""
    path = Path(output) / "RESULT.json"
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return False
    expected = "QUALIFICATION_COMPLETE" if variant else "ACTUAL_FULL_NATIVE_BASELINE_EPOCH_NOT_EVALUATED"
    return value.get("status") == expected and value.get("config", {}).get("cell") == cell


def job_name(entry, variant=None):
    return "dsmm-t3b-" + entry["short"] + ("-qual-" + variant if variant else "")


def submit(site, control, cell, entry, dependency, variant=None, time_limit="01:00:00"):
    command = ["/usr/bin/sbatch", "--parsable", "--export=NIL", "--time=" + time_limit, "--nodes=1",
               "--ntasks=1", *site["routes"][entry["route"]], "--job-name=" + job_name(entry, variant),
               "--chdir=" + site["home"], "--requeue", "--open-mode=append",
               "--output=" + str(control / "job-%j.log")]
    if dependency:
        command.append("--dependency=" + dependency)
    command += [str(Path(site["home"]) / site["source"] / site["run"]),
                "qualify-worker-" + variant if variant else "worker", cell]
    intent = control / ("SUBMIT_INTENT-" + str(time.time_ns()) + ".json")
    write_json(intent, {"argv": command, "dependency": dependency, "utc": utc()})
    result = scheduler(*command)
    job = result.stdout.strip().split(";")[0]
    ok = result.returncode == 0 and re.fullmatch(r"[1-9][0-9]*", job) is not None
    write_json(control / ("SUBMIT_ACK-" + (job if ok else "FAILED-" + str(time.time_ns())) + ".json"),
               {"intent": str(intent), "returncode": result.returncode, "stdout": result.stdout,
                "stderr": result.stderr, "job_id": job if ok else None, "utc": utc()})
    require(ok, "Submission failed: " + result.stderr.strip())
    return job


def continue_requeue(site, control, cell, entry, job, variant):
    """Hand the cell to the same job again (or a fresh job after many restarts / refused requeue).

    Exactly one continuation: a PENDING job of the same name is adopted instead of requeueing/submitting.
    """
    restarts = int(os.environ.get("SLURM_RESTART_COUNT") or 0)
    report = {"mode": "self_requeue", "fresh_job_error": None}
    waiting = sorted((o for o, state in active_jobs(site, job_name(entry, variant)).items()
                      if o != job and state == "PENDING"), key=int)
    if waiting:
        report.update(mode="adopted_pending_successor", job=waiting[-1])
        return report
    if restarts < FRESH_JOB_AFTER_RESTARTS:
        result = scheduler("/usr/bin/scontrol", "requeue", job)
        report.update(returncode=result.returncode, stderr=result.stderr.strip())
        if result.returncode == 0:
            return report
    for attempt in range(3):
        lock = locked(site)
        try:
            report.update(mode="fresh_job", job=submit(site, control, cell, entry, "afterany:" + job, variant))
            report.pop("fresh_job_error", None)
            return report
        except Exception as error:
            report["fresh_job_error"] = repr(error)
        finally:
            os.close(lock)
        time.sleep(30 * (attempt + 1))
    report["mode"] = "CHAIN_BROKEN_NO_CONTINUATION"
    write_json(control / ("CHAIN_BROKEN-" + job + ".json"), report)
    return report


def worker(cell, variant=None):
    site = SITES["babel"]
    home = Path(site["home"])
    entry = registry_cell(cell)
    job = os.environ["SLURM_JOB_ID"]
    link = job + "-r" + (os.environ.get("SLURM_RESTART_COUNT") or "0")
    control = home / site["control"] / (cell + ("-qualify-" + variant if variant else ""))
    output = Path(site["output_root"]) / (cell + ("-qualify-" + variant if variant else ""))
    started = time.time()
    write_json(control / ("LINK_START-" + link + ".json"), {"started_ns": time.time_ns(), "job": job,
                                                           "utc": utc(), "node": os.uname().nodename})

    def finish(code, **fields):
        write_json(control / ("LINK_EXIT-" + link + ".json"),
                   dict(fields, returncode=code, elapsed_seconds=round(time.time() - started, 1)))
        return code

    if cell_result_ok(output, cell, variant):
        return finish(0, status="ALREADY_COMPLETE", residual_removed=clean_residual_checkpoints(output))
    for path in control.glob("CONTINUATION-" + job + "-r*.json"):
        if json.loads(path.read_bytes()).get("mode") in ("fresh_job", "adopted_pending_successor"):
            return finish(0, status="HANDED_OFF_TO_SUCCESSOR_NOT_TRAINING", successor=json.loads(path.read_bytes()).get("job"))
    # Exclusive per-cell training lease, held until this process exits (released by the kernel on kill).
    lease = os.open(control / "TRAINING.lease", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lease)
        return finish(0, status="DUPLICATE_LEASE_HELD_NOT_TRAINING")
    history = link_history(control)
    if consecutive_failures(site, control, link) >= MAX_CONSECUTIVE_FAILURES or len(history) > MAX_LINKS:
        return finish(3, status="CHAIN_STALLED", links=len(history))
    code, status = None, None
    try:
        job_end = job_end_epoch(job)
        deadline = job_end - site["reserve_seconds"]
        records = entry["records"]
        require(Path(records["path"]).stat().st_size == records["bytes"] and sha(records["path"]) == records["sha256"],
                "Registered selected records differ")
        require(sha(POOL) == POOL_SHA256, "Pool authority changed")
        pool = json.loads(Path(POOL).read_bytes())
        require(sha(home / TEMPLATE) == TEMPLATE_SHA256, "Template packet changed")
        template = json.loads((home / TEMPLATE).read_bytes())
        receipt = template["providers"]["assets"]
        require(receipt["sha256"] == ASSET_RECEIPT_SHA256 and sha(receipt["path"]) == ASSET_RECEIPT_SHA256,
                "Asset receipt changed")
        output.mkdir(mode=0o700, parents=True, exist_ok=True)
        model = home / site["source"] / "model.py"
        config = {"role": "TOOL3_BASELINE_RESUMABLE_FULL_EPOCH_NOT_EVALUATED", "job_id": job, "link_id": link,
                  "cell": cell, "registry_entry": entry, "K": entry["K"], "training_seed": entry["training_seed"],
                  "input": records, "image_root": pool["image_root"],
                  "assets": json.loads(Path(receipt["path"]).read_bytes())["assets"],
                  "sources": dict(template["sources"], **EXTRA_SOURCES), "root": site["root"], "output": str(output),
                  "model_source": {"path": str(model), "sha256": sha(model)},
                  "deadline_epoch": deadline, "save_steps": site["save_steps"]}
        if variant:
            config.update(role="TOOL3_RESUME_QUALIFICATION_NOT_SCIENCE", qualification_max_steps=4, save_steps=2,
                          stop_after_steps=2 if variant == "chunked" else None)
        config_path = control / ("MODEL_CONFIG-" + link + ".json")
        write_json(config_path, config)
        env = {"PATH": "/usr/bin:/bin", "HOME": site["user_home"], "USER": site["user"], "LC_ALL": "C",
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
               "HF_HOME": site["hf_home"], "TORCH_EXTENSIONS_DIR": site["torch_extensions"],
               "PYTHONDONTWRITEBYTECODE": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        env.update({k: v for k, v in os.environ.items() if k.startswith("SLURM_") or k in ("CUDA_VISIBLE_DEVICES", "TMPDIR")})
        command = [site["python"], "-B", "-X", "pycache_prefix=/dev/null/no-cache", "-m", "torch.distributed.run",
                   "--standalone", "--nnodes=1", "--nproc-per-node=1", "--no-python", site["python"], "-B", "-X",
                   "pycache_prefix=/dev/null/no-cache", str(model), "--config", str(config_path)]
        if deadline - time.time() < 300:
            code, status = 0, "NO_TRAINING_TIME_LEFT"
        else:
            try:
                code = subprocess.run(command, env=env, stdin=subprocess.DEVNULL,
                                      timeout=job_end - 60 - time.time()).returncode
            except subprocess.TimeoutExpired:
                code = 124
    except Exception as error:
        code, status = 2, "LINK_SETUP_FAILED: " + repr(error)[-1500:]
    done = cell_result_ok(output, cell, variant)
    chunks = sorted((output / "chunks").glob("chunk-" + link + ".json"))
    chunk = json.loads(chunks[0].read_bytes()) if chunks else None
    if chunk:
        chunk = {key: value for key, value in chunk.items() if key not in ("consumed_indices", "step_events")}
    progress = bool(chunk and (chunk.get("end_step") or 0) > chunk.get("resume_step", 0)) or done
    failed = code != 0 and not progress
    stalled = consecutive_failures(site, control, link) + failed >= MAX_CONSECUTIVE_FAILURES
    finish(code, status=status or ("RESULT_SUCCESS" if done else "CHUNK_EXIT"), result_exists=done,
           progress=progress, chunk_record=chunk)
    if not done and not stalled:
        write_json(control / ("CONTINUATION-" + link + ".json"), continue_requeue(site, control, cell, entry, job, variant))
    return code


def start(cell, after=None, variant=None):
    site = SITES["babel"]
    entry = registry_cell(cell)
    control = Path(site["home"]) / site["control"] / (cell + ("-qualify-" + variant if variant else ""))
    control.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = locked(site)
    try:
        active = active_jobs(site, job_name(entry, variant))
        require(not active, "Cell already has active jobs; refusing a duplicate start: " + json.dumps(active))
        require(not cell_result_ok(Path(site["output_root"]) / control.name, cell, variant), "Cell already complete")
        job = submit(site, control, cell, entry, ("afterok:" + after) if after else None, variant,
                     "00:40:00" if variant else "01:00:00")
    finally:
        os.close(lock)
    print(json.dumps({"cell": cell, "variant": variant, "submitted": job}))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("start", "worker", "qualify-start", "qualify-worker-chunked"))
    parser.add_argument("cell")
    parser.add_argument("--after", default=None)
    args = parser.parse_args()
    require(args.after is None or args.after.isdecimal(), "Gate must be a job id")
    if args.command == "worker":
        return worker(args.cell)
    if args.command == "qualify-worker-chunked":
        return worker(args.cell, variant="chunked")
    if args.command == "qualify-start":
        return start(args.cell, args.after, variant="chunked")
    return start(args.cell, args.after)


if __name__ == "__main__":
    sys.exit(main())
