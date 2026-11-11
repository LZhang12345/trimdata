#!/bin/bash
# Route (partition/QoS) comes only from the reviewed plan's sbatch argv: general/normal or preempt/preempt_qos.
#SBATCH --account=cx
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-p9-20
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
set -euo pipefail
export PATH=/usr/bin:/bin
unset BASH_ENV ENV LD_PRELOAD LD_LIBRARY_PATH PYTHONHOME PYTHONPATH PYTHONOPTIMIZE
script=/home/luyangz/data_selection_multimodal/experiments/helper_tool1_bench6_eval_v3_20261104/prepare6.py
root=/home/luyangz/data_selection_multimodal/collaboration/helper-tool1-baseline-independent-20261101/bench6-eval-v3
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

path, root, cell, attempt, route = sys.argv[1:]
CELLS = ("b6-datatailor-b05-k33032-train17", "b6-datatailor-b05-k33032-train29",
         "b6-datatailor-b05-k33032-train43", "b6-datatailor-b15-k99108-train17",
         "b6-datatailor-b15-k99108-train29", "b6-datatailor-b15-k99108-train43",
         "b6-datatailor-b20-k132147-train17", "b6-datatailor-b20-k132147-train29",
         "b6-datatailor-b20-k132147-train43", "b6-uniform-mk1235-b05-k33032-train17",
         "b6-uniform-mk1235-b05-k33032-train29", "b6-uniform-mk1235-b05-k33032-train43",
         "b6-uniform-mk1235-b15-k99108-train17", "b6-uniform-mk1235-b15-k99108-train29",
         "b6-uniform-mk1235-b15-k99108-train43", "b6-uniform-mk1235-b20-k132147-train17",
         "b6-uniform-mk1235-b20-k132147-train29", "b6-uniform-mk1235-b20-k132147-train43",
         "m10-datatailor-k66071-train17", "m10-datatailor-k66071-train29", "m10-datatailor-k66071-train43",
         "m10-uniform-sel1235-train17", "m10-uniform-sel1235-train29", "m10-uniform-sel1235-train43",
         "m10-uniform-sel1236-train17", "m10-uniform-sel1236-train29", "m10-uniform-sel1236-train43")
if cell not in CELLS or attempt not in ("0", "1", "2", "3", "4", "5", "6", "7") or route not in ("general", "preempt"):
    raise ValueError("Exact finite Tool1 BENCH6 CELLS_V3 evaluation request required")
control = root + "/" + cell + "/attempt-" + attempt
SOURCE_SIZE = 37105
job = os.environ.get("SLURM_JOB_ID", "")
if os.getuid() != 2708899 or re.fullmatch(r"[1-9][0-9]*", job) is None:
    raise ValueError("Owned scalar Slurm job required for bootstrap")
directory, leaf = parent(control + "/worker-" + job + ".log")
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
    if (len(raw) != SOURCE_SIZE or hashlib.sha256(raw).hexdigest() != "6fdb5907d2081bd9c6ea95579193f1b3de3eeaf2478ca626755c768209949bcd"
            or any(getattr(before, k) != getattr(after, k) for k in stable)):
        raise ValueError("Exact stable reviewed evaluation source bytes required")
finally:
    os.close(source)
sys.argv = [path, "worker", "--cell", cell, "--attempt", attempt, "--route", route]
exec(compile(bytes(raw), path, "exec", dont_inherit=True), {"__name__": "__main__", "__file__": path})
' "$script" "$root" "$1" "$2" "$3"
