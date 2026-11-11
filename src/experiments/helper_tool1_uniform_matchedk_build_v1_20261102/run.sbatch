#!/bin/bash
#SBATCH --account=cx
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
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
script=/home/luyangz/data_selection_multimodal/experiments/helper_tool1_uniform_matchedk_build_v1_20261102/prepare.py
root=/home/luyangz/data_selection_multimodal/collaboration/helper-tool1-baseline-independent-20261101/uniform-matchedk-build-v1
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

path, control = sys.argv[1:]
SOURCE_SIZE = 43101
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
    if (len(raw) != SOURCE_SIZE or hashlib.sha256(raw).hexdigest() != "9c97a4953a3deb270432d6ea5c133846e860d5c9094a452bc409b4de78b64cfd"
            or any(getattr(before, k) != getattr(after, k) for k in stable)):
        raise ValueError("Exact stable reviewed Uniform matched-K builder bytes required")
finally:
    os.close(source)
sys.argv = [path, "worker"]
exec(compile(bytes(raw), path, "exec", dont_inherit=True), {"__name__": "__main__", "__file__": path})
' "$script" "$root"
