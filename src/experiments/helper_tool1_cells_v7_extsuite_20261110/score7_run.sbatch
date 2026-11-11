#!/bin/bash
# Route (partition/QoS) comes only from the reviewed plan's sbatch argv: general/normal or preempt/preempt_qos.
# Stdlib scoring in the allocation's own CPU share; no model is loaded (Babel: CPU work inside a GPU allocation).
#SBATCH --account=cx
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-p9-20
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
set -euo pipefail
export PATH=/usr/bin:/bin
unset BASH_ENV ENV LD_PRELOAD LD_LIBRARY_PATH PYTHONHOME PYTHONPATH PYTHONOPTIMIZE
script=/home/luyangz/data_selection_multimodal/experiments/helper_tool1_cells_v7_extsuite_20261110/score7.py
root=/home/luyangz/data_selection_multimodal/collaboration/helper-tool1-baseline-independent-20261101
exec /usr/bin/python3 -I -S -B -u -c '
import hashlib, os, re, stat, sys
from contextlib import ExitStack

def parent(path):
    if not path.startswith("/") or path.startswith("//") or path.endswith("/") or "\0" in path:
        raise ValueError("Literal absolute bootstrap path required")
    parts = path[1:].split("/")
    if not all(p not in ("", ".", "..") for p in parts):
        raise ValueError("Literal bootstrap path components required")
    with ExitStack() as stack:
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        stack.callback(os.close, fd)
        for component in parts[:-1]:
            fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            stack.callback(os.close, fd)
        return os.dup(fd), parts[-1]

path, root, scorer, batch, attempt, route = sys.argv[1:]
ROOTS = {"bench6": "/bench6-eval-v3/scoring-afterok-v7", "bench4": "/bench4-eval-v3/scoring-afterok-v7",
         "vizwiz": "/vizwiz-eval-v3/vizwiz-test-scoring-afterok-v7"}
if (scorer not in ROOTS or batch not in ("0", "1", "2", "3", "4", "5", "6", "7")
        or attempt not in ("0", "1", "2", "3") or route not in ("general", "preempt")):
    raise ValueError("Exact finite Tool1 CELLS_V7 afterok scoring request required")
SOURCE_SIZE = 41465
job = os.environ.get("SLURM_JOB_ID", "")
if os.getuid() != 2708899 or re.fullmatch(r"[1-9][0-9]*", job) is None:
    raise ValueError("Owned scalar Slurm job required for bootstrap")
directory, leaf = parent(root + ROOTS[scorer] + "/batch-" + batch + "/attempt-" + attempt + "/worker-" + job + ".log")
try:
    info = os.fstat(directory)
    if info.st_uid != 2708899 or info.st_mode & 0o022:
        raise ValueError("Owned private control directory required")
    log = os.open(leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_SYNC,
                  0o600, dir_fd=directory)
    os.fsync(directory)
finally:
    os.close(directory)
os.dup2(log, 1)
os.dup2(log, 2)
os.close(log)
directory, leaf = parent(path)
try:
    source = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=directory)
finally:
    os.close(directory)
try:
    before = os.fstat(source)
    if not stat.S_ISREG(before.st_mode) or before.st_uid != 2708899 or before.st_size != SOURCE_SIZE:
        raise ValueError("Exact bounded owned bootstrap source required")
    raw = bytearray()
    while len(raw) <= SOURCE_SIZE:
        block = os.read(source, min(65536, SOURCE_SIZE + 1 - len(raw)))
        if not block:
            break
        raw.extend(block)
    after = os.fstat(source)
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if (len(raw) != SOURCE_SIZE or hashlib.sha256(raw).hexdigest() != "65b4d2d44f602dd245bc236a410c218f812a4964ce148a063c464f5ddf6d10c7"
            or any(getattr(before, k) != getattr(after, k) for k in stable)):
        raise ValueError("Exact stable reviewed scoring source bytes required")
finally:
    os.close(source)
sys.argv = [path, "worker", "--scorer", scorer, "--batch", batch, "--attempt", attempt, "--route", route]
exec(compile(bytes(raw), path, "exec", dont_inherit=True), {"__name__": "__main__", "__file__": path})
' "$script" "$root" "$1" "$2" "$3" "$4"
