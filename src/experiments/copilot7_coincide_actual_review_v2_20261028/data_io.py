"""Exact-inventory paid DATA reads: no chmod, generic alias or HOME relaxation."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from experiments.root_current_four_bank_eval_v1_20261027.admission import parent_fd
from experiments.meta_probe_v1_20261028.contracts import require

UID = 2708899
TRUSTED_GID = 500
MAX_RECORD_JSON = 2 << 30


def descriptor(ref, maximum=MAX_RECORD_JSON):
    require(type(ref) is dict and set(ref) == {"path", "bytes", "sha256"}
            and type(ref["path"]) is str and Path(ref["path"]).is_absolute()
            and str(Path(ref["path"])) == ref["path"]
            and ".." not in Path(ref["path"]).parts and "\x00" not in ref["path"]
            and type(ref["bytes"]) is int and 0 < ref["bytes"] <= maximum
            and type(ref["sha256"]) is str
            and re.fullmatch("[0-9a-f]{64}", ref["sha256"]) is not None,
            "Exact bounded absolute path/bytes/SHA256 descriptor required")
    return ref


def allowed_stat(info, *, uid=UID, gid=TRUSTED_GID):
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_uid == uid
            and not info.st_mode & 0o002, "Paid DATA requires UID-owned singleton regular no-world-write")
    if info.st_mode & 0o020:
        require(info.st_gid == gid and stat.S_IMODE(info.st_mode) == 0o660,
                "Only exact reviewed0660/trustedGID DATA exception; no blanket mode relaxation")


def read_exact(ref, roots, *, collect=True, maximum=32 << 20, uid=UID, gid=TRUSTED_GID):
    descriptor(ref, maximum)
    path = Path(ref["path"])
    require(path.is_absolute() and ".." not in path.parts
            and any(path.is_relative_to(Path(r)) and path != Path(r) for r in roots),
            "DATA path outside exact reviewed roots/inventory")
    signature = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns, x.st_uid, x.st_gid, x.st_mode, x.st_nlink)
    with parent_fd(path) as (parent, name):
        ancestry = os.fstat(parent)
        before_path = os.stat(name, dir_fd=parent, follow_symlinks=False)
        allowed_stat(before_path, uid=uid, gid=gid)
        require(before_path.st_size == ref["bytes"], "Actual paid DATA length differs")
        fd = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=parent)
        try:
            before = os.fstat(fd)
            allowed_stat(before, uid=uid, gid=gid)
            require(signature(before) == signature(before_path), "DATA replaced before anchored open")
            digest = hashlib.sha256()
            chunks, count = [], 0
            while count < before.st_size:
                raw = os.read(fd, min(4 << 20, before.st_size - count))
                require(bool(raw), "DATA truncated during exact read")
                digest.update(raw)
                count += len(raw)
                if collect:
                    chunks.append(raw)
            after = os.fstat(fd)
            current = os.stat(name, dir_fd=parent, follow_symlinks=False)
            allowed_stat(after, uid=uid, gid=gid)
            allowed_stat(current, uid=uid, gid=gid)
            require(signature(before) == signature(after) == signature(current), "Paid DATA changed while hashing")
            with parent_fd(path) as (fresh, fresh_name):
                require((ancestry.st_dev, ancestry.st_ino)
                        == (os.fstat(fresh).st_dev, os.fstat(fresh).st_ino)
                        and signature(os.stat(fresh_name, dir_fd=fresh, follow_symlinks=False)) == signature(after),
                        "Paid DATA path/ancestor retargeted")
        finally:
            os.close(fd)
    require(count == ref["bytes"] and digest.hexdigest() == ref["sha256"],
            "Actual paid DATA byte SHA mismatch")
    return b"".join(chunks) if collect else None


def _unique(items):
    result = {}
    for key, value in items:
        require(type(key) is str and key not in result, "Duplicate/non-string DATA JSON key")
        result[key] = value
    return result


def decode_json(raw):
    def nonfinite(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(raw, object_pairs_hook=_unique, parse_constant=nonfinite)


def load_json(ref, roots, *, maximum=32 << 20, uid=UID, gid=TRUSTED_GID):
    raw = read_exact(ref, roots, maximum=maximum, uid=uid, gid=gid)
    return decode_json(raw)


@contextmanager
def heavy_read_lock(path):
    import fcntl
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Existing Root heavy metadata lock required")
    with parent_fd(path) as (parent, name):
        fd = os.open(name, os.O_RDWR | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=parent)
        try:
            info = os.fstat(fd)
            require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and info.st_nlink == 1,
                    "Root heavy lock inode/owner invalid")
            fcntl.flock(fd, fcntl.LOCK_EX)
            with parent_fd(path) as (fresh, fresh_name):
                current = os.stat(fresh_name, dir_fd=fresh, follow_symlinks=False)
                require((current.st_dev, current.st_ino) == (info.st_dev, info.st_ino),
                        "Root heavy lock changed")
            yield
        finally:
            os.close(fd)
