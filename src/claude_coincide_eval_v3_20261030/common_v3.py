"""COINCIDE four-bank evaluation v3: stdlib identity primitives, pins and review gates.

Imports nothing from the frozen evaluator so that login-node staging/submission
(system python3.9) and pure tests can use it. Heavy evaluation code is only
imported by worker.py inside the GPU allocation.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PACKAGE = HERE.name
HOME = Path('/home/luyangz/data_selection_multimodal')
SOURCE_ROOT = HOME / 'claude_coincide_eval_src_v3_20261030'
TRAINING_RECORDS = HOME / 'collaboration/claude-baseline-recovery-20261030/coincide-v3'
EVAL_RECORDS = HOME / 'collaboration/claude-baseline-recovery-20261030/coincide-v3-eval'
REVIEW = EVAL_RECORDS / 'INDEPENDENT_REVIEW_EVAL_V3.json'
CELLS = HOME / 'collaboration/root-general-all-plan-20261028/submission-check-20261029/baseline-ready-v1/cells'
V2_REPORT = HOME / 'collaboration/session7-coincide-actual-output-review-20261028/v2/native'
OUTPUT = Path('/data/hf_cache/project_features/DS-MultiModal/coincide_native_evaluation_20261028/v3')
ADMISSION = HOME / 'collaboration/babel-two-track-20261019/ADMISSION.lock'
PYTHON = '/data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python'
UID, USER = 2708899, 'luyangz'

# Qualified basis: the Root v7 official-join four-bank configuration that produced
# the published uniform-s17 scores (verified read-only on Babel 2026-10-29).
BASIS = dict(path=str(HOME / 'collaboration/root-current-four-bank-eval-v7-20261027/configs/'
                      'CONFIG_production-student-v1-10-uniform-s17.json'),
             bytes=74597, sha256='a5174ca4b2104e7e683e864507aa887ba82a32c05c9276a285f607cc7996a899')
# Frozen (never executed) COINCIDE evaluator v2 handoff and its 429-file archive.
HANDOFF = dict(path=str(HOME / 'collaboration/session7-coincide-actual-output-review-20261028/v2/evaluation/'
                        'EVALUATION_EXECUTABLE_HANDOFF_V2.json'),
               bytes=64177, sha256='41f61ee93dcf896c2b470918b098de3a1ba83499f7c39099403f3572baedf300')
ARCHIVE = dict(bytes=776602, sha256='ae40cdbe6be7027ff1568302d3723bb0de437369826b739593fd9a18fff4358e')

NAME_PREFIX = 'dsmm-coincide-fourbank-v3-s'
COMMENT_PREFIX = 'dsmm-coincide-fourbank-v3:'
TIME_LIMIT, MINUTES = '06:00:00', 360
REQUEST_SCHEMA = 'coincide-fourbank-eval-v3-request-v1'
REVIEW_STATUS = 'PASS_INDEPENDENT_COINCIDE_FOURBANK_EVAL_V3_EXACT_SOURCE_CONFIG'
TRAINING_REVIEW_STATUS = 'PASS_INDEPENDENT_COINCIDE_RECOVERY_V3_EXACT_SOURCE'
TRAINING_REQUEST_SCHEMA = 'coincide-recovery-v3-request-v1'
TRAINING_BINDING_SCHEMA = 'coincide-recovery-v3-binding-v1'
TRAINING_JOB_PREFIX = 'dsmm-coincide-v3-s'
SEEDS = (17, 29, 43)
# Accepted v2 prefix of the site-interrupted trajectories (training common.PREDECESSORS).
PREFIX = {
    17: dict(mode='resume', v2_job='10978109', v2_attempt='10978109-r0-2c905725', step=244),
    29: dict(mode='resume', v2_job='10978116', v2_attempt='10978116-r0-6b78b213', step=242),
    43: dict(mode='fresh', v2_job='10978117', v2_attempt=None, step=0),
}
TERMINAL = ('FAILED', 'COMPLETED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'BOOT_FAIL',
            'DEADLINE', 'PREEMPTED')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    """Same bytes as the native v7 common.canonical used by the v2 collectors."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


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
    require(type(descriptor) is dict and set(descriptor) == {'path', 'bytes', 'sha256'},
            'Exact path/bytes/sha256 descriptor required')
    require(desc(descriptor['path']) == descriptor, 'Artifact identity differs: ' + descriptor['path'])
    return read(descriptor['path'])


def same_bytes(ref, pin):
    return ref['bytes'] == pin['bytes'] and ref['sha256'] == pin['sha256']


def seal(path, value):
    """Immutable fsynced JSON; an equal rewrite is allowed, a conflict fails."""
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


def cell_id(seed):
    require(type(seed) is int and seed in SEEDS, 'Only COINCIDE fixed10 paired seeds 17/29/43')
    return 'coincide-native10-s' + str(seed)


def seed_of(cell):
    match = re.fullmatch(r'coincide-native10-s(17|29|43)', cell)
    require(match is not None, 'Unknown COINCIDE cell')
    return int(match[1])


def package_sources(root=None):
    """This package's executable bytes (tests excluded), keyed relative to the source root."""
    root = Path(root) if root is not None else ROOT
    package = root / PACKAGE
    paths = sorted(p for p in package.iterdir() if p.suffix in ('.py', '.sbatch') and not p.name.startswith('test_'))
    return {str(p.relative_to(root)): sha(p) for p in paths}


def frozen_tree(handoff_sources, basis_sources):
    """Archive files, then the qualified basis overlay (basis bytes win on conflict).

    Returns (tree, superseded) where superseded lists archive paths whose bytes the
    basis replaces. The package is added separately by package_sources().
    """
    for name in list(handoff_sources) + list(basis_sources):
        parts = Path(name).parts
        require(not Path(name).is_absolute() and '..' not in parts and parts and parts[0] in ('experiments', 'collaboration')
                and not name.startswith(PACKAGE + '/'), 'Unsafe/foreign frozen source path: ' + name)
    tree = dict(handoff_sources)
    superseded = sorted(k for k, v in basis_sources.items() if k in tree and tree[k] != v)
    tree.update(basis_sources)
    return dict(sorted(tree.items())), superseded


def fields(raw):
    """One scontrol -o record; values may contain spaces (e.g. Command)."""
    rows = [line for line in raw.splitlines() if line.strip()]
    require(len(rows) == 1, 'One exact scalar scheduler record')
    matches = list(re.finditer(r'(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=', rows[0]))
    require(matches and matches[0].start() == 0, 'Malformed scheduler record')
    result = {}
    for i, match in enumerate(matches):
        require(match[1] not in result, 'Duplicate scheduler key')
        end = matches[i + 1].start() if i + 1 < len(matches) else len(rows[0])
        result[match[1]] = rows[0][match.end():end].strip()
    return result


def _not_implementer(review):
    name = review.get('reviewer')
    require(type(name) is str and name.strip() and name.isascii(), 'Named independent reviewer required')
    normalized = re.sub('[^a-z0-9]', '', name.casefold())
    require('implementer' not in normalized and normalized not in ('claude', 'root', 'session7', 'author', 's7')
            and not normalized.startswith('copilot7'), 'Implementer cannot accept its own adapter')


def independent(review, current_sources, configuration_ref, cell):
    """Single v3 gate: separate (disclosed AI) reviewer of exact sources AND this cell's config."""
    _not_implementer(review)
    require(review.get('status') == REVIEW_STATUS and review.get('independent') is True
            and review.get('ai_review_disclosed') is True and review.get('blocking_findings') == []
            and review.get('sources') == current_sources
            and type(review.get('configurations')) is dict
            and review['configurations'].get(cell) == configuration_ref,
            'Exact current-source and per-cell configuration independent evaluation review required')
    return review


def training_independent(review, training_sources):
    """The v3 training adapter's own review (training common.independent semantics)."""
    _not_implementer(review)
    require(review.get('status') == TRAINING_REVIEW_STATUS and review.get('sources') == training_sources
            and review.get('ai_review_disclosed') is True and review.get('blocking_findings') == [],
            'Bound v3 training adapter lacks its exact independent review')
    return review


def job_name(seed):
    cell_id(seed)
    return NAME_PREFIX + str(seed)


def expected_actor(job, seed, request_nonce, node, root=None):
    root = Path(root) if root is not None else ROOT
    return dict(JobId=job, JobState='RUNNING', JobName=job_name(seed), UserId=USER + '(' + str(UID) + ')',
        Partition='general', Account='cx', QOS='normal', NumCPUs='8', NumTasks='1', NumNodes='1',
        Restarts='0', TimeLimit=TIME_LIMIT, TresPerNode='gres/gpu:L40S:1',
        Command=str(root / PACKAGE / 'run.sbatch'), WorkDir=str(root),
        Comment=COMMENT_PREFIX + request_nonce, NodeList=node)


def check_actor(f, expected):
    bad = {k: (f.get(k), v) for k, v in expected.items() if f.get(k) != v}
    require(not bad, 'Actual evaluator job identity/resources differ: ' + repr(bad))
    tres = dict(item.split('=', 1) for item in f.get('AllocTRES', '').split(',') if '=' in item)
    require(tres.get('cpu') == '8' and tres.get('mem') in ('64G', '65536M') and tres.get('gres/gpu') == '1',
            'Allocated CPU8/64GiB/GPU1 required')
    return tres


def records_identity(request_ref, request, intent, ack, job):
    """REQUEST/INTENT/ACK chain of this exact physical job (training driver.current semantics)."""
    folder = Path(request_ref['path']).parent
    require(request.get('schema') == REQUEST_SCHEMA and folder.name == request.get('nonce')
            and folder.parent == EVAL_RECORDS / cell_id(request.get('seed'))
            and intent.get('request') == request_ref and ack.get('job_id') == job and job.isdecimal(),
            'Actual intent/ACK for this exact evaluation request; never another job identity')
    return folder


def live_eval_jobs(squeue_output, seed, own_job=None):
    rows = []
    for line in squeue_output.splitlines():
        if not line.strip():
            continue
        job, name, state = line.split('|')[:3]
        if name == job_name(seed) and job != own_job:
            rows.append(dict(job=job, name=name, state=state))
    return rows
