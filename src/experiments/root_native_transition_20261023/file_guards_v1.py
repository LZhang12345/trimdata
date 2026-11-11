"""Standalone additive full-ancestor checkpoint-file guard.

The Chain/file_row/checked algorithm is retained from reviewed stage_a_v4;
no historical modules or missing mount helpers are imported or modified.
"""
import hashlib
import json
import os
from pathlib import Path
import stat


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(4 << 20), b''):
            value.update(block)
    return value.hexdigest()


def seal(path, value):
    path = Path(path)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    fd = os.open(path.parent, os.O_DIRECTORY | os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def identity(st):
    return (st.st_dev, st.st_ino, st.st_mode, st.st_nlink,
            st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def dir_id(st):
    return (st.st_dev, st.st_ino, stat.S_IFMT(st.st_mode))


class Chain:
    def __init__(self, path, root):
        self.path, self.root, self.fds = Path(path), Path(root), []
        require(self.path.is_absolute() and '..' not in self.path.parts
                and self.path != self.root, 'Literal file path required')
        self.path.relative_to(self.root)
        try:
            self.fds.append(os.open('/', os.O_RDONLY | os.O_DIRECTORY))
            for name in self.path.parent.parts[1:]:
                self.fds.append(os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                        dir_fd=self.fds[-1]))
            self.ids = [dir_id(os.fstat(fd)) for fd in self.fds]
        except BaseException:
            self.close(); raise

    def close(self):
        for fd in reversed(self.fds):
            os.close(fd)
        self.fds = []

    def validate_live(self, expected):
        require(self.ids == [tuple(x) for x in expected], 'Ancestor changed since manifest')
        fresh = Chain(self.path, self.root)
        try:
            require(fresh.ids == self.ids, 'Live ancestor chain changed')
            for i, name in enumerate(self.path.parent.parts[1:]):
                live = os.stat(name, dir_fd=fresh.fds[i], follow_symlinks=False)
                old = os.stat(name, dir_fd=self.fds[i], follow_symlinks=False)
                require(dir_id(live) == self.ids[i+1] == dir_id(old),
                        'Ancestor renamed, replaced or aliased')
            require(dir_id(os.stat('/', follow_symlinks=False)) == self.ids[0], 'Root changed')
        finally:
            fresh.close()


def file_row(path, root):
    chain = Chain(path, root)
    try:
        st = os.stat(Path(path).name, dir_fd=chain.fds[-1], follow_symlinks=False)
        require(st.st_dev == Path(root).lstat().st_dev, 'Foreign file filesystem refused')
        return dict(path=str(path), identity=list(identity(st)), ancestors=chain.ids,
                    bytes=st.st_size, reported_allocated_bytes=st.st_blocks*512)
    finally:
        chain.close()


def checked(row, root, delete=False):
    chain = Chain(row['path'], root); fd = None
    try:
        chain.validate_live(row['ancestors'])
        fd = os.open(chain.path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=chain.fds[-1])
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                and identity(before) == tuple(row['identity']), 'Candidate identity changed')
        digest = hashlib.sha256()
        for block in iter(lambda: os.read(fd, 4 << 20), b''):
            digest.update(block)
        value = digest.hexdigest()
        require(identity(os.fstat(fd)) == identity(before), 'Payload changed during read')
        if row.get('sha256'):
            require(value == row['sha256'], 'Payload hash differs')
        chain.validate_live(row['ancestors'])
        require(identity(os.stat(chain.path.name, dir_fd=chain.fds[-1], follow_symlinks=False))
                == identity(before), 'Final basename no longer names authorized inode')
        if delete:
            require(row.get('sha256') == value, 'Deletion requires reviewed payload digest')
            os.unlink(chain.path.name, dir_fd=chain.fds[-1]); os.fsync(chain.fds[-1])
        return dict(path=row['path'], sha256=value, bytes=before.st_size,
                    reported_allocated_bytes=before.st_blocks*512, deleted=delete)
    finally:
        if fd is not None:
            os.close(fd)
        chain.close()


def verify_snapshot(row):
    require(identity(Path(row['path']).lstat()) == tuple(row['identity']),
            'Protected artifact changed: ' + row['path'])
