"""Small, standard-library identity/durability primitives."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PIN = 'c121f0432da27facab705978f83c4ada465e46fd'
NATIVE = 'collaboration/published-foundation-20261022/reference_sources/LLaVA/' + PIN

def require(condition, message):
    if not condition:
        raise ValueError(message)

def now():
    return datetime.now(timezone.utc).isoformat()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()

def key(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def desc(path):
    path = Path(path).absolute()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))

def read(path):
    return json.loads(Path(path).read_text())

def checked(descriptor):
    require(desc(descriptor['path']) == descriptor, 'Artifact identity differs: ' + descriptor['path'])
    return read(descriptor['path'])

def seal(path, value):
    """Immutable, fsynced JSON: repeated equal writes are allowed, conflicts fail."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical(value) + b'\n'
    if path.exists():
        require(path.read_bytes() == payload, 'Immutable path already differs: ' + str(path))
        return desc(path)
    fd, temporary = tempfile.mkstemp(prefix='.transaction-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(path.read_bytes() == payload, 'Concurrent immutable identity conflict')
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        os.unlink(temporary)
    return desc(path)

def sources():
    paths = list(HERE.glob('*.py')) + list(HERE.glob('*.sbatch')) + [HERE/'zero3.json']
    paths += list((ROOT/NATIVE).rglob('*.py')) + [ROOT/NATIVE/'pyproject.toml']
    paths += [ROOT/'experiments/root_native_selection_20261023'/name for name in ('selection.py','common.py')]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths) if not p.name.startswith('test_')}

def verify_sources(packet):
    require(packet == sources(), 'Execution source packet differs from reviewed packet')

def within(path, root):
    return Path(path).resolve().is_relative_to(Path(root).resolve())
