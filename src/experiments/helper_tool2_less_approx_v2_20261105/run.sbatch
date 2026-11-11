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
#SBATCH --requeue
set -euo pipefail
export PATH=/usr/bin:/bin
unset BASH_ENV ENV LD_PRELOAD LD_LIBRARY_PATH PYTHONHOME PYTHONPATH PYTHONOPTIMIZE
case "$1" in *[!a-z0-9_+-]*|"") echo "Exact probe identifier required" >&2; exit 2 ;; esac
case "$2" in [1-9]) ;; *) echo "Exact attempt required" >&2; exit 2 ;; esac
script=/home/luyangz/data_selection_multimodal/experiments/helper_tool2_less_approx_v2_20261105/prepare.py
control=/home/luyangz/data_selection_multimodal/collaboration/helper-tool2-method-20261101/less-approx-v2/${1}/attempt-${2}
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

path, control, probe, attempt = sys.argv[1:]
if re.fullmatch(r"[a-z0-9][a-z0-9_+-]{0,63}", probe) is None or re.fullmatch(r"[1-9]", attempt) is None:
    raise ValueError("Exact designed probe and attempt required")
SOURCE_SIZE = 33056
job = os.environ.get("SLURM_JOB_ID", "")
if os.getuid() != 2708899 or re.fullmatch(r"[1-9][0-9]*", job) is None:
    raise ValueError("Owned scalar Slurm job required for bootstrap")
restart = os.environ.get("SLURM_RESTART_COUNT", "0")
if re.fullmatch(r"[0-9]+", restart) is None:
    raise ValueError("Exact restart count required")
directory, leaf = parent(control + "/worker-" + job + "-r" + restart + ".log")
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
    if (len(raw) != SOURCE_SIZE or hashlib.sha256(raw).hexdigest() != "17683b84c4eb3dfe5300479201af5957ea1d16aeb11b3865430f5cf6477550b2"
            or any(getattr(before, k) != getattr(after, k) for k in stable)):
        raise ValueError("Exact stable reviewed preview source bytes required")
finally:
    os.close(source)
sys.argv = [path, "worker", "--probe", probe, "--attempt", attempt]
exec(compile(bytes(raw), path, "exec", dont_inherit=True), {"__name__": "__main__", "__file__": path})
' "$script" "$control" "$1" "$2"
