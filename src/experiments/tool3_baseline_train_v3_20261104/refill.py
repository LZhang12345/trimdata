"""Tool3 baseline Preempt refill: when the user's preempt_qos job count is below THRESHOLD, submit the
next BATCH ready items from the ordered queue (QUEUE.json), recording intent/ACK in STATE.json.

Item kinds:
  start / qualify : `chain.py start|qualify-start <cell> [--after <job of item 'after'>]` (v3 trainer)
  eval_quality    : Tool3 quality-cell FULL evaluation, ready once that cell's training wrote RESULT
                    (a LINK_EXIT with result_exists=true in its HOME control dir)
  after_done      : item becomes ready only when the named items' jobs have COMPLETED (sacct), e.g. a
                    cell gated on its qualification passing.
Run on the Babel login node; read-only except STATE.json/REFILL_LOG.jsonl and the submissions themselves.
"""

import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HOME = Path("/home/luyangz/data_selection_multimodal")
ROOT = HOME / "collaboration/helper-tool3-baselines-20261104"
QUEUE, STATE, LOG = ROOT / "QUEUE.json", ROOT / "STATE.json", ROOT / "REFILL_LOG.jsonl"
LOCK = HOME / "collaboration/babel-two-track-20261019/ADMISSION.lock"
TRAIN = HOME / "experiments/tool3_baseline_train_v3_20261104/chain.py"
EVAL = HOME / "experiments/tool3_quality_eval_v1_20261102/evalchain.py"
QUALITY_CONTROL = HOME / "collaboration/helper-tool3-quality-20261102/training-v2"
THRESHOLD, BATCH = 90, 5
LIMIT = re.compile(r"QOS|limit|Invalid qos|AssocMax", re.IGNORECASE)
REFUSALS = {}
ENV = {"PATH": "/usr/bin:/bin", "HOME": "/home/luyangz", "USER": "luyangz", "LOGNAME": "luyangz", "LC_ALL": "C"}


def run(*argv, timeout=120):
    return subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout, env=ENV,
                          stdin=subprocess.DEVNULL)


def write_json(path, value):
    tmp = path.with_name("." + path.name + ".tmp-" + str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True))
    os.replace(tmp, path)


def preempt_count():
    result = run("/usr/bin/squeue", "-h", "-u", "luyangz", "-q", "preempt_qos", "-o", "%i")
    if result.returncode != 0:
        raise RuntimeError("squeue failed: " + result.stderr.strip())
    return len([line for line in result.stdout.splitlines() if line.strip()])


TRAIN_CONTROL = HOME / "collaboration/helper-tool3-baselines-20261104/training"
EVAL_CONTROL = HOME / "collaboration/helper-tool3-quality-20261102/eval-v1"


def result_written(control):
    """A link of this chain exited with its exact-cell success RESULT (follows any continuation job)."""
    for path in Path(control).glob("LINK_EXIT-*.json"):
        if json.loads(path.read_bytes()).get("result_exists") is True:
            return True
    return False


def quality_cell_done(cell):
    return result_written(QUALITY_CONTROL / cell)


def control_dir(item):
    if item["kind"] == "eval_quality":
        return EVAL_CONTROL / item["cell"]
    return TRAIN_CONTROL / (item["cell"] + ("-qualify-chunked" if item["kind"] == "qualify" else ""))


REGISTRY = HOME / "collaboration/helper-tool3-baselines-20261104/CELLS.json"
SHORT_ARM = {"matched_random": "mr", "exact_quality": "eq", "loose_quality": "lq"}


def job_name(item):
    if item["kind"] == "eval_quality":
        arm, sel, train = re.fullmatch(r"quality-(\w+)-sel(\d+)-train(\d+)", item["cell"]).groups()
        return "dsmm-t3-e-" + SHORT_ARM[arm] + "-s" + sel + "-t" + train
    short = json.loads(REGISTRY.read_bytes())["cells"][item["cell"]]["short"]
    return "dsmm-t3b-" + short + ("-qual-chunked" if item["kind"] == "qualify" else "")


def active_job(item):
    result = run("/usr/bin/squeue", "-h", "-u", "luyangz", "-n", job_name(item), "-o", "%i")
    jobs = [line.strip() for line in result.stdout.splitlines() if line.strip().isdecimal()]
    return max(jobs, key=int) if jobs else None


def reconcile(item, record):
    """An intent without a job: adopt a SUBMIT_ACK written after the intent, else allow a retry."""
    acks = []
    for path in Path(control_dir(item)).glob("SUBMIT_ACK-[0-9]*.json"):
        value = json.loads(path.read_bytes())
        if value.get("job_id") and value.get("utc", "") >= record["intent_utc"].rstrip("Z"):
            acks.append(value["job_id"])
    return max(acks, key=int) if acks else active_job(item)


def ready(item, state, queue):
    if state.get(item["id"], {}).get("job") or item.get("hold"):
        return False
    if result_written(control_dir(item)):
        state[item["id"]] = {"job": "RESULT_ALREADY_WRITTEN", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        write_json(STATE, state)
        return False
    existing = active_job(item)
    if existing:
        state[item["id"]] = {"job": existing, "adopted_active": True}
        write_json(STATE, state)
        return False
    by_id = {other["id"]: other for other in queue}
    for name in item.get("after_result", []):
        if not result_written(control_dir(by_id[name])):
            return False
    if item["kind"] == "eval_quality" and not quality_cell_done(item["cell"]):
        return False
    return True


def command(item, state):
    if item["kind"] in ("start", "qualify"):
        argv = ["/usr/bin/python3", "-I", "-B", str(TRAIN), "qualify-start" if item["kind"] == "qualify" else "start",
                item["cell"]]
        return argv
    if item["kind"] == "eval_quality":
        arm, sel, train = re.fullmatch(r"quality-(\w+)-sel(\d+)-train(\d+)", item["cell"]).groups()
        return ["/usr/bin/python3", "-I", "-B", str(EVAL), "start", "--mode", "FULL", arm, sel, train]
    raise ValueError("Unknown item kind: " + item["kind"])


def submitted_job(stdout):
    for line in reversed(stdout.strip().splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        return str(value.get("submitted") or value.get("job") or "")
    return ""


def main():
    single = os.open(ROOT / "REFILL.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        return locked_main(single)
    finally:
        os.close(single)


def locked_main(single):
    try:
        fcntl.flock(single, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps({"skipped": "another refill run is active"}))
        return 0
    queue = json.loads(QUEUE.read_bytes())["items"]
    ids = [item["id"] for item in queue]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate queue item ids")
    lock = os.open(LOCK, os.O_RDWR)
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        state = json.loads(STATE.read_bytes()) if STATE.exists() else {}
        global REFUSALS
        REFUSALS = dict(state.get("_refusals", {}))
        event = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "submitted": [], "reconciled": []}
        for item in queue:
            record = state.get(item["id"])
            if item["id"].startswith("_"):
                raise ValueError("Reserved item id")
            if record and not record.get("job") and record.get("intent_utc"):
                job = reconcile(item, record)
                if job:
                    record.update(job=job, reconciled=True)
                    event["reconciled"].append({"id": item["id"], "job": job})
                else:
                    state.pop(item["id"])  # the submitter never acknowledged: safe to retry
                write_json(STATE, state)
        count = preempt_count()
        event["preempt_jobs"] = count
        if count < THRESHOLD:
            pending = sorted((item for item in queue if ready(item, state, queue)), key=lambda item: item["priority"])
            for item in pending[:BATCH]:
                argv = command(item, state)
                state[item["id"]] = {"intent_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), "argv": argv,
                                     "job": None}
                write_json(STATE, state)
                os.close(lock)  # the submitters take the same admission lock themselves
                lock = None
                result = run(*argv)
                lock = os.open(LOCK, os.O_RDWR)
                fcntl.flock(lock, fcntl.LOCK_EX)
                job = submitted_job(result.stdout)
                record = {"argv": argv, "returncode": result.returncode, "stdout": result.stdout[-600:],
                          "stderr": result.stderr[-600:], "job": job if job.isdecimal() else None, "utc": event["utc"]}
                refusals = REFUSALS.get(item["id"], 0)
                if record["job"]:
                    state[item["id"]] = dict(record, intent_utc=state[item["id"]]["intent_utc"])
                elif refusals + 1 >= 3 and not LIMIT.search(record["stderr"]):
                    state[item["id"]] = {"job": "BLOCKED_AFTER_3_REFUSALS", "last_error": record["stderr"][-400:]}
                else:
                    state.pop(item["id"])
                    state.setdefault("_refusals", {})[item["id"]] = refusals + 1
                write_json(STATE, state)
                event["submitted"].append({"id": item["id"], **{k: record[k] for k in ("job", "returncode")},
                                           "error": None if record["job"] else record["stderr"][-300:]})
                if not record["job"] and LIMIT.search(record["stderr"]):
                    break  # scheduler limit: stop this batch; other refusals only affect their own item
        with LOG.open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        print(json.dumps(event))
    finally:
        if lock is not None:
            os.close(lock)


if __name__ == "__main__":
    sys.exit(main())
