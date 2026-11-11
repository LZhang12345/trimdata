#!/bin/bash
#SBATCH --account=cx
#SBATCH --partition=general
#SBATCH --qos=normal
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
script=/home/luyangz/data_selection_multimodal/experiments/helper_tool1_baseline_eval_v4_20261105/prepare.py
root=/home/luyangz/data_selection_multimodal/collaboration/helper-tool1-baseline-independent-20261101/baseline-eval-v4
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

path, root, cell, roster, roster_sha, attempt = sys.argv[1:]
if attempt not in ("0", "1", "2", "3", "4", "5"):
    raise ValueError("Finite evaluation attempt index required")
trainer = "/home/luyangz/data_selection_multimodal/collaboration/helper-tool1-baseline-independent-20261101/baseline-train-v5"
if re.fullmatch(r"[a-z0-9][a-z0-9-]{2,60}", cell) is None:
    raise ValueError("Literal producer cell name required")
if (re.fullmatch(r"[A-Za-z0-9_./-]+", roster) is None or not roster.startswith(trainer + "/")
        or "/../" in roster or re.fullmatch(r"[0-9a-f]{64}", roster_sha) is None):
    raise ValueError("Pinned trainer roster required")
control = root + "/" + cell
SOURCE_SIZE = 49863
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
    if (len(raw) != SOURCE_SIZE or hashlib.sha256(raw).hexdigest() != "683066d95fce17b5251c6a733a3823fafae23148f63e13b903fe19e98d5cb16b"
            or any(getattr(before, k) != getattr(after, k) for k in stable)):
        raise ValueError("Exact stable reviewed evaluation v4 source bytes required")
finally:
    os.close(source)
sys.argv = [path, "worker", "--cell", cell, "--roster", roster, "--roster-sha256", roster_sha, "--attempt", attempt]
exec(compile(bytes(raw), path, "exec", dont_inherit=True), {"__name__": "__main__", "__file__": path})
' "$script" "$root" "$1" "$2" "$3" "$4"
