"""Independent metadata/job primitives outside the frozen native85 namespaces."""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
from datetime import datetime, timezone


OWNER = "Session7_native_official_execution_20261024"
ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def key(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def desc(path):
    path = Path(path).absolute()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text())


def checked(description):
    require(desc(description["path"]) == description, "Artifact descriptor differs: " + description["path"])
    return read(description["path"])


def seal(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = canonical(value)
    if path.exists():
        require(path.read_bytes() == body, "Immutable receipt differs: " + str(path))
        return desc(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".s7-transaction-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(path.read_bytes() == body, "Concurrent immutable receipt differs")
        handle = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)
    finally:
        os.unlink(temporary)
    return desc(path)


def verify_sources(sources, root=ROOT):
    require(isinstance(sources, dict) and sources, "Exact source closure required")
    for relative, digest in sources.items():
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts, "Unsafe source relative path")
        require(sha(root / path) == digest, "Changed source: " + relative)


def actual(binding, config):
    job = os.environ.get("SLURM_JOB_ID", "")
    require(job.isdecimal() and binding["job_id"] == job and binding["owner"] == OWNER,
            "Exact owned scalar prerequisite job required")
    require(not os.environ.get("SLURM_ARRAY_JOB_ID"), "Scalar prerequisite only")
    raw = subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True, timeout=30)
    fields = dict(re.findall(r"(\w+)=(\S+)", raw))
    expected = dict(
        JobId=job, JobState="RUNNING", Account="cx", Partition="general", QOS="normal",
        JobName=config["job_name"], WorkDir=str(ROOT),
        Command=str(ROOT / config["wrapper"]), TresPerNode="gres/gpu:L40S:1",
        NumTasks="1", NumCPUs="8", TimeLimit="00:55:00",
    )
    require(all(fields.get(name) == value for name, value in expected.items()), "Actual prerequisite route differs")
    require(fields.get("NumNodes") in {"1", "1-1"} and fields.get("UserId", "").split("(")[0] == "luyangz",
            "Actual prerequisite node/user differs")
    require(fields.get("NodeList") == socket.gethostname().split(".")[0], "Actual compute node differs")
    intent = checked(binding["intent"])
    require(fields.get("Comment") == "dsmm-s7-native:" + intent["nonce"]
            and intent["configuration"] == binding["configuration"], "Actual pre-RPC ownership intent differs")
    tres = dict(part.split("=", 1) for part in fields["ReqTRES"].split(",") if "=" in part)
    require(tres.get("mem") in {"64G", "65536M"} and tres.get("gres/gpu") == "1", "Actual RAM/GPU request differs")
    restart = fields.get("Restarts", "")
    require(restart.isdecimal(), "Authoritative restart identity absent")
    if "SLURM_RESTART_COUNT" in os.environ:
        require(os.environ["SLURM_RESTART_COUNT"] == restart, "Stale inherited restart identity")
    else:
        os.environ["SLURM_RESTART_COUNT"] = restart
    return dict(job_id=job, restart=int(restart), node=fields["NodeList"], start_time=fields["StartTime"],
                hostname=socket.gethostname(), raw_slurm=raw, requested_GPU_count=1, requested_CPUs=8)
