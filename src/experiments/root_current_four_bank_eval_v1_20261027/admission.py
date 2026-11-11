"""Additive Root census/LOGIN repair. Importing never runs an operation.

Run on Babel against separately deployed, reviewed HOME metadata and source bytes.
This module never stages code, reads DATA payloads, requeues, or changes other jobs.
All scheduler and filesystem access is injectable for declared synthetic tests.
The frozen six V1 configs remain blocked on Root's failed shared qualifier.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from dataclasses import dataclass, field
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import time

from . import contract as c, checks

UID = 2708899
USER = "luyangz"
RELATIVE = "experiments/root_current_four_bank_eval_v1_20261027"
HELPER = "experiments/root_current_four_bank_eval_v1_20261027/admission.py"
SCRIPT = RELATIVE + "/run.sbatch"
ADMISSIONS = c.CONTROL / "admission-v1"
APPROVAL_STATUS = "ROOT_PERSONALLY_APPROVED_CURRENT12_OFFICIAL_JOIN_ADMISSION_V7"
STORAGE_STATUS = "ACTUAL_REGISTERED_GPU_SHARED_FS_PREFLIGHT_V1"
EXPORT_STATUS = "ACTUAL_ROOT_COMPARATOR_PREREQUISITE_ARTIFACT_CUSTODY_V1"
MAX_METADATA = 32 << 20
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "RESIZING", "STAGE_OUT"}
ALLOCATED = ACTIVE - {"PENDING"}
QOS_COLUMNS = (
    "Name", "MaxTRESPU", "MaxJobsPU", "MaxSubmitPU",
    "MaxTRESPA", "MaxJobsPA", "MaxSubmitPA", "GrpTRES", "GrpJobs", "GrpSubmitJobs",
)
QUEUE_ARGV = ["squeue", "--all", "--array", "--noheader", "--user=luyangz",
              "--format=%i|%u|%a|%q|%T|%j|%k"]


class AdmissionError(RuntimeError):
    """A closed gate; retained intent/ACK must be reconciled, never resubmitted."""


def require(condition, message):
    if not condition:
        raise AdmissionError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def literal(path):
    raw = str(path)
    parsed = PurePosixPath(raw)
    require(parsed.is_absolute() and str(parsed) == raw and ".." not in parsed.parts
            and not any(ch.isspace() for ch in raw) and "\x00" not in raw,
            "An absolute canonical literal metadata path is required")
    return Path(raw)


def reference(path, raw):
    return {"path": str(path), "bytes": len(raw), "sha256": digest(raw)}


def descriptor(ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "bytes", "sha256"}
            and type(ref["bytes"]) is int and 0 < ref["bytes"] <= MAX_METADATA
            and isinstance(ref["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]) is not None,
            "An actual bounded metadata descriptor, not a future hash, is required")
    literal(ref["path"])
    return ref


def unpack_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON metadata key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          AdmissionError("Non-finite JSON metadata")))


@contextmanager
def parent_fd(path):
    """Anchor every component; never follow a directory or leaf symlink."""
    path = literal(path)
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=fd)
            os.close(fd)
            fd = next_fd
        yield fd, path.name
    finally:
        os.close(fd)


def inode(info):
    return info.st_dev, info.st_ino


def regular_owned(info, uid=UID):
    require(stat.S_ISREG(info.st_mode) and info.st_uid == uid and info.st_nlink == 1
            and not info.st_mode & 0o022, "Regular, singly linked, actor-owned custody required")


@contextmanager
def canonical_lock(path, uid=UID):
    """Use the existing lock inode, including a canonical-path recheck after flock."""
    with parent_fd(path) as (parent, name):
        before = os.stat(name, dir_fd=parent, follow_symlinks=False)
        regular_owned(before, uid)
        fd = os.open(name, os.O_RDWR | os.O_NOFOLLOW, dir_fd=parent)
        try:
            regular_owned(os.fstat(fd), uid)
            require(inode(before) == inode(os.fstat(fd)), "Shared admission lock changed on open")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            def recheck():
                with parent_fd(path) as (current_parent, current_name):
                    current = os.stat(current_name, dir_fd=current_parent, follow_symlinks=False)
                    regular_owned(current, uid)
                    require(inode(os.fstat(parent)) == inode(os.fstat(current_parent))
                            and inode(current) == inode(os.fstat(fd)),
                            "Canonical ADMISSION.lock was replaced or re-anchored")
            recheck()
            yield recheck
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


class NativeFiles:
    """Small HOME/control records only. No DATA traversal or source writes."""

    @staticmethod
    def allowed(path):
        path = literal(path)
        require(path.is_relative_to(c.HOME), "LOGIN must use a real HOME custody export; no DATA reads")
        return path

    def read(self, path):
        path = self.allowed(path)
        with parent_fd(path) as (parent, name):
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            try:
                before = os.fstat(fd)
                regular_owned(before)
                require(0 <= before.st_size <= MAX_METADATA
                        and (before.st_size > 0 or path.is_relative_to(c.SOURCE_ROOT)),
                        "Bounded HOME metadata or reviewed empty source required")
                chunks, size = [], 0
                while True:
                    chunk = os.read(fd, min(1 << 20, MAX_METADATA + 1 - size))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    size += len(chunk)
                    require(size <= MAX_METADATA, "HOME metadata exceeded its size bound")
                after = os.fstat(fd)
                signature = lambda info: (inode(info), info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                require(signature(before) == signature(after) and size == after.st_size,
                        "HOME metadata changed while reading")
                require(inode(os.stat(name, dir_fd=parent, follow_symlinks=False)) == inode(after),
                        "HOME metadata leaf was replaced")
                return b"".join(chunks)
            finally:
                os.close(fd)

    def exists(self, path):
        path = self.allowed(path)
        try:
            with parent_fd(path) as (parent, name):
                info = os.stat(name, dir_fd=parent, follow_symlinks=False)
                require(not stat.S_ISLNK(info.st_mode), "Symlink custody is forbidden")
            return True
        except FileNotFoundError:
            return False

    def mkdir(self, path):
        path = self.allowed(path)
        require(path.is_relative_to(ADMISSIONS), "Only the new admission control namespace is writable")
        missing = []
        cursor = path
        while not self.exists(cursor):
            missing.append(cursor)
            cursor = cursor.parent
        for directory in reversed(missing):
            with parent_fd(directory) as (parent, name):
                os.mkdir(name, 0o700, dir_fd=parent)
                os.fsync(parent)

    def seal(self, path, value):
        path = self.allowed(path)
        require(path.is_relative_to(ADMISSIONS), "Sources and other control scopes are immutable")
        raw = c.canonical(value) + b"\n"
        require(len(raw) <= MAX_METADATA, "Bounded control receipt required")
        with parent_fd(path) as (parent, name):
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=parent)
            try:
                with os.fdopen(fd, "wb", closefd=False) as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(fd)
            finally:
                os.close(fd)
            os.fsync(parent)
        return reference(path, raw)

    def lock(self):
        return canonical_lock(c.LOCK)

    @staticmethod
    def home_capacity():
        value = os.statvfs(c.HOME)
        return value.f_bavail * value.f_frsize, value.f_favail


@dataclass(frozen=True)
class Result:
    returncode: int | None
    stdout: bytes = b""
    stderr: bytes = b""
    exception: str | None = None

    def receipt(self):
        return {"returncode": self.returncode,
                "stdout_base64": base64.b64encode(self.stdout).decode("ascii"),
                "stderr_base64": base64.b64encode(self.stderr).decode("ascii"),
                "exception_type": self.exception}


def run_command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, timeout=60, check=False)
        return Result(result.returncode, result.stdout, result.stderr)
    except subprocess.TimeoutExpired as error:
        return Result(None, error.stdout or b"", error.stderr or b"", "TimeoutExpired")
    except OSError as error:
        return Result(None, exception=type(error).__name__)


@dataclass
class Runtime:
    files: object = field(default_factory=NativeFiles)
    run: object = run_command
    uid: object = os.getuid
    now: object = time.time
    executing_helper: Path = field(default_factory=lambda: Path(__file__).absolute())

    def command(self, argv):
        result = self.run(list(argv))
        require(isinstance(result, Result), "Injected commands must return lossless Result objects")
        return result

    def query(self, argv):
        result = self.command(argv)
        require(result.returncode == 0, "Metadata query failed; Root must reconcile the bounded operation")
        return result.stdout.decode("utf-8", errors="strict")

    def checked(self, ref):
        descriptor(ref)
        raw = self.files.read(ref["path"])
        require(reference(ref["path"], raw) == ref, "Actual metadata descriptor changed")
        return unpack_json(raw)

    def document(self, path):
        path = literal(path)
        raw = self.files.read(path)
        return reference(path, raw), unpack_json(raw)


def fields(raw):
    rows = [row for row in raw.splitlines() if row.strip()]
    require(len(rows) == 1, "Exactly one scheduler record is required")
    matches = list(re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=", rows[0]))
    require(matches and matches[0].start() == 0, "Malformed scheduler record")
    result = {}
    for index, match in enumerate(matches):
        require(match[1] not in result, "Duplicate scheduler field")
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(rows[0])
        result[match[1]] = rows[0][match.end():stop].strip()
    return result


def table(raw, width):
    result = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = line.split("|")
        if len(row) == width + 1 and row[-1] == "":
            row.pop()
        require(len(row) == width, "Unexpected metadata columns; preserve empty trailing fields")
        result.append(row)
    return result


def tres(raw):
    result = {}
    if raw in (None, "", "(null)", "N/A"):
        return result
    for token in raw.split(","):
        pair = token.split("=", 1)
        require(len(pair) == 2 and pair[0] not in result, "Malformed or duplicate requested TRES")
        result[pair[0]] = pair[1]
    return result


def gpu_count(record):
    values = tres(record.get("ReqTRES"))
    gpu_values = {name: value for name, value in values.items() if name.startswith("gres/gpu")}
    require(all(re.fullmatch(r"\d+", value) for value in gpu_values.values()),
            "Unparseable GPU multiplicity")
    typed = sum(int(value) for name, value in gpu_values.items() if name.startswith("gres/gpu:"))
    total = int(gpu_values["gres/gpu"]) if "gres/gpu" in gpu_values else typed
    require(not typed or "gres/gpu" not in gpu_values or typed == total,
            "Aggregate and typed GPU counts disagree")
    per_node = record.get("TresPerNode", "")
    require(not ("gpu" in per_node and not gpu_values),
            "GPU request omitted from authoritative ReqTRES")
    return total


def limit(value):
    if value in ("", "None", "UNLIMITED", "N/A"):
        return None
    require(re.fullmatch(r"[1-9]\d*", value) is not None, "Unrecognized finite QoS limit")
    return int(value)


def qos_gpu_limit(value):
    values = tres(value)
    if not values:
        return None
    require("gres/gpu" in values, "QoS TRES lacks the required aggregate GPU limit")
    return limit(values["gres/gpu"])


def allocation_identity(selector, record):
    """Link an expanded array selector to its actual physical allocation."""
    match = re.fullmatch(r"([1-9]\d*)(?:_(0|[1-9]\d*))?", selector)
    require(match is not None, "Expanded scalar/array selector required; heterogeneous requests unsupported")
    physical = record.get("JobId", "")
    require(re.fullmatch(r"[1-9]\d*", physical) is not None, "Actual physical allocation ID required")
    master, task = match.groups()
    if task is None:
        require(physical == selector
                and all(record.get(name) in (None, "", "(null)") for name in
                        ("ArrayJobId", "ArrayTaskId", "ArrayTaskStr", "HetJobId", "HetJobOffset")),
                "Scalar census record must not impersonate an array or heterogeneous component")
    else:
        require(record.get("ArrayJobId") == master and record.get("ArrayTaskId") == task
                and record.get("ArrayTaskStr") in (None, "", task)
                and all(record.get(name) in (None, "", "(null)") for name in ("HetJobId", "HetJobOffset")),
                "Expanded selector must match exact ArrayJobId/ArrayTaskId")
    return physical


def comment_identity(value):
    # Only the observed absence spellings are equivalent. Text such as "None",
    # "N/A" or a different nonempty nonce remains a real, distinct comment.
    return "" if value is None or value in ("", "(null)") else value


def pending_master_alias(selector, record):
    """Only an unallocated, unmaterialized expanded Slurm array task."""
    match = re.fullmatch(r"([1-9]\d*)_(0|[1-9]\d*)", selector)
    return bool(match and record.get("JobId") == record.get("ArrayJobId") == match[1]
        and record.get("ArrayTaskId") == match[2]
        and record.get("JobState") == "PENDING"
        and record.get("AllocTRES") in (None, "", "(null)")
        and record.get("NodeList") in (None, "", "(null)")
        and record.get("RunTime") == "00:00:00"
        and record.get("Restarts") == "0")


def shared_pending_master_alias(previous, selector, record):
    previous_selector = previous["_requested_selector"]
    require(previous_selector != selector
        and pending_master_alias(previous_selector, previous)
        and pending_master_alias(selector, record),
        "Only distinct unallocated pending tasks may share their own array master")
    fields = ("JobId", "ArrayJobId", "UserId", "GroupId", "Account", "QOS", "Partition",
              "JobName", "ReqTRES", "NumCPUs", "NumTasks", "NumNodes", "TresPerNode",
              "TresPerTask", "Command", "WorkDir", "TimeLimit", "Requeue", "Restarts")
    require(all(previous.get(name) == record.get(name) for name in fields)
        and comment_identity(previous.get("Comment")) == comment_identity(record.get("Comment")),
        "Shared pending array master changed owner, account, QoS, resources or source")


def census(runtime, additional=0):
    qos_raw = runtime.query(["sacctmgr", "-nP", "show", "qos", "normal",
                             "format=" + ",".join(QOS_COLUMNS)])
    qos_rows = table(qos_raw, len(QOS_COLUMNS))
    require(len(qos_rows) == 1 and qos_rows[0][0] == "normal", "One fresh normal QoS policy required")
    policy = dict(zip(QOS_COLUMNS, qos_rows[0]))
    require(all(policy[name] for name in ("MaxTRESPU", "MaxJobsPU", "MaxSubmitPU")),
            "Root must resolve missing actual normal user limits")
    nonuser_columns = ("MaxTRESPA", "MaxJobsPA", "MaxSubmitPA", "GrpTRES", "GrpJobs", "GrpSubmitJobs")
    require(all(policy[name] == "" for name in nonuser_columns),
            "Own-user census supports only genuinely unconfigured account/group QoS limits; use a separately reviewed global route")
    raw = runtime.query(QUEUE_ARGV)
    queue = table(raw, 7)
    ids, physical_ids, records, outside_scope = set(), {}, [], []
    for job, user, account, qos, state, name, comment in queue:
        require(user == USER, "Own-user scheduler filter returned a foreign user; no admission")
        comparator = name.startswith(("dsmm-root-comparator-eval-", "dsmm-current-four-bank-", "dsmm-native-trained-eval-v")) or comment.startswith(c.COMMENT)
        if qos != "normal" and not comparator:
            # Normal's user/account/group limits apply to all normal rows, not
            # another QoS. Comparator controllers still need duplicate checks.
            outside_scope.append([job, user, account, qos, state, name, comment])
            continue
        require(re.fullmatch(r"[1-9]\d*(?:_(?:0|[1-9]\d*))?", job) is not None and job not in ids,
                "Expanded, unique scalar/array selectors required; heterogeneous requests unsupported")
        ids.add(job)
        record_raw = runtime.query(["scontrol", "show", "job", "-o", job])
        record = fields(record_raw)
        physical = allocation_identity(job, record)
        if physical in physical_ids:
            shared_pending_master_alias(physical_ids[physical], job, record)
        else:
            physical_ids[physical] = record
        require(record.get("JobState") in ACTIVE
                and record.get("JobState") == state and record.get("Account") == account
                and record.get("QOS") == qos and record.get("JobName") == name
                and comment_identity(record.get("Comment")) == comment_identity(comment)
                and record.get("UserId", "").split("(", 1)[0] == user,
                "Census changed or an all-user allocation is unauthenticated; query again")
        if user == USER:
            require(record["UserId"] == f"{USER}({UID})", "Root metadata user's actual UID differs")
        record["_requested_selector"] = job
        record["_physical_job_id"] = physical
        record["_queue_row"] = [job, user, account, qos, state, name, comment]
        record["_queue_comment_raw"] = comment
        record["_scontrol_comment_raw"] = record.get("Comment")
        record["_raw"] = record_raw
        record["_gpus"] = gpu_count(record)
        records.append(record)
    require(table(runtime.query(QUEUE_ARGV), 7) == queue, "Expanded census changed during its query")
    summaries = {}
    scopes = (
        ("user", lambda row: row["UserId"] == f"{USER}({UID})",
         ("MaxTRESPU", "MaxJobsPU", "MaxSubmitPU")),
        ("account", lambda row: row["Account"] == "cx",
         ("MaxTRESPA", "MaxJobsPA", "MaxSubmitPA")),
        ("qos", lambda row: True, ("GrpTRES", "GrpJobs", "GrpSubmitJobs")),
    )
    for label, belongs, columns in scopes:
        rows = [row for row in records if row["QOS"] == "normal" and belongs(row)]
        running = [row for row in rows if row["JobState"] in ALLOCATED]
        summary = {
            "expanded_submitted": len(rows),
            "running_jobs": len(running),
            "running_gpus": sum(row["_gpus"] for row in running),
            "all_requested_gpus": sum(row["_gpus"] for row in rows),
            "held_pending_gpus": sum(row["_gpus"] for row in rows
                                     if row["JobState"] == "PENDING" and row.get("Reason") == "JobHeldUser"),
        }
        gpu_max, jobs_max, submitted_max = (
            qos_gpu_limit(policy[columns[0]]), limit(policy[columns[1]]), limit(policy[columns[2]]))
        require(gpu_max is None or gpu_max >= c.RESOURCES["gpus"], "This request exceeds the actual GPU limit")
        require(gpu_max is None or summary["running_gpus"] <= gpu_max, "Actual running GPU census exceeds current QoS")
        require(jobs_max is None or jobs_max >= 1 and summary["running_jobs"] <= jobs_max,
                "Actual running job census exceeds current QoS")
        require(submitted_max is None or len(rows) + additional <= submitted_max,
                "Actual expanded submission capacity is exhausted")
        summaries[label] = summary
    # Pending requests do not consume running GPU/job limits. Slurm, not a private
    # project fence, schedules released jobs under the freshly queried limits.
    return {"qos": policy, "qos_raw": qos_raw, "queue_raw": raw, "queue_rows": queue,
            "census_scope": "ALL_OWN_USER_JOBS_ACROSS_PROJECTS_AND_ACCOUNTS_UNCONFIGURED_NONUSER_QOS_LIMITS",
            "nonuser_limits_verified_unconfigured": {name: policy[name] for name in nonuser_columns},
            "nonuser_count_summaries_are_own_user_observations_not_global": True,
            "other_qos_rows_not_counted": outside_scope,
            "records": records, "counts": summaries, "epoch": runtime.now()}


def function_identity(cfg):
    return c.key({
        "schema": "root264-final516-whole-four-bank-admission-function-v1",
        "training_configuration": cfg["training_configuration"],
        "student_completion_path": cfg["student_completion_path"],
        "final_accepted_global_step": 516,
        "official_bank_contract_configuration": cfg["official_bank_contract_configuration"],
        "banks": cfg["banks"], "counts": cfg["counts"],
    })


def reviewed(runtime, configuration_path, packet_path, review_path, approval_path):
    packet_ref, packet = runtime.document(packet_path)
    review_ref, review = runtime.document(review_path)
    approval_ref, approval = runtime.document(approval_path)
    require(packet["source_root"] == str(c.SOURCE_ROOT) and isinstance(packet["sources"], dict)
            and len(packet["configurations"]) == 12,
            "Exact separately staged twelve-config source packet required")
    require(review["status"] == c.REVIEW_STATUS and review["independent"] is True
            and isinstance(review["reviewer"], str) and review["reviewer"].strip()
            and review["reviewer"].replace(" ", "").casefold() not in ("session5", "s5-comparator-admission")
            and review["configurations"] == packet["configurations"]
            and review["sources"] == packet["sources"], "Independent exact-source/twelve-config review missing")
    require(approval["status"] == APPROVAL_STATUS and approval["owner"] == "Root"
            and approval["source_packet"] == packet_ref
            and approval["independent_source_config_review"] == review_ref
            and approval["helper_sha256"] == packet["sources"].get(HELPER),
            "Root personal approval for the exact new helper/source/config hashes required")
    require(runtime.executing_helper == c.SOURCE_ROOT / HELPER, "Exact separately deployed metadata helper")
    configs = {}
    for student_id, ref in packet["configurations"].items():
        cfg = runtime.checked(ref)
        c.validate_configuration(cfg, load=runtime.checked)
        require(cfg["student_id"] == student_id and cfg["sources"] == packet["sources"]
                and re.fullmatch(r"[0-9a-f]{32}", cfg["nonce"]) is not None, "Exact reviewed config/nonce/source map")
        configs[student_id] = cfg
    require(set(configs) == set(c.STUDENTS) and len({cfg["nonce"] for cfg in configs.values()}) == 12,
            "Exactly twelve genuine distinct current-model functions")
    selected = [cfg for sid, cfg in configs.items()
                if packet["configurations"][sid]["path"] == str(literal(configuration_path))]
    require(len(selected) == 1, "Configuration outside Root's exact twelve-source/config review")
    verify_sources(runtime, packet["sources"])
    return selected[0], configs, dict(source_packet=packet_ref, independent_source_config_review=review_ref,
        root_personal_approval=approval_ref, configuration=packet["configurations"][selected[0]["student_id"]])


def verify_sources(runtime, sources):
    require(HELPER in sources and SCRIPT in sources, "The helper and actual wrapper must be in the source review")
    for relative, expected in sources.items():
        path = PurePosixPath(relative)
        require(not path.is_absolute() and str(path) == relative and ".." not in path.parts
                and relative not in ("", ".") and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
                "Canonical relative source/hash required")
        require(digest(runtime.files.read(c.SOURCE_ROOT / relative)) == expected,
                "Separately deployed source bytes changed; no overwrite or staging is permitted")


def duplicate_check(snapshot, cfg, configs, existing=False):
    known = {c.COMMENT + value["nonce"]: value for value in configs.values()}
    matches = []
    for row in snapshot["records"]:
        scoped = row["JobName"].startswith(("dsmm-root-comparator-eval-", "dsmm-current-four-bank-", "dsmm-native-trained-eval-v")) or row.get("Comment", "").startswith(c.COMMENT)
        if not scoped:
            continue
        require(row.get("Comment") in known and row["JobName"] == c.NAME
                and row["UserId"] == f"{USER}({UID})" and row["Account"] == "cx"
                and row["Partition"] == "general" and row["QOS"] == "normal",
                "Unresolved or foreign comparator controller: Root must reconcile before admission")
        if function_identity(known[row["Comment"]]) == function_identity(cfg):
            matches.append(row["JobId"])
    require(len(matches) <= 1 and (existing or not matches), "A live or ambiguous scientific function already exists")
    return matches


def dependency_ids(raw):
    if raw in (None, "", "(null)", "None"):
        return set()
    result = []
    for group in raw.split(","):
        require(group.startswith("afterok:"), "Only genuine afterok prerequisites are authorized")
        for token in group[len("afterok:"):].split(":"):
            match = re.fullmatch(r"([1-9]\d*)(?:\((?:unfulfilled|fulfilled)\))?", token)
            require(match is not None, "Malformed afterok prerequisite")
            result.append(match[1])
    require(len(result) == len(set(result)), "Duplicate scheduler prerequisite")
    return set(result)


def exported_documents(runtime, export_path):
    if export_path is None:
        return {}
    _, value = runtime.document(export_path)
    require(value["status"] == EXPORT_STATUS and value["owner"] == "Root" and value["read_only"] is True,
            "Actual Root HOME artifact custody export required")
    documents = value["documents"]
    require(isinstance(documents, dict) and len(documents) <= 64, "Bounded actual artifact export required")
    for path, entry in documents.items():
        ref = descriptor(entry["reference"])
        raw = entry["text"].encode("utf-8")
        require(ref["path"] == path and reference(path, raw) == ref,
                "Exported actual artifact bytes/descriptor differ")
    return documents


def artifact(runtime, path, exports):
    path = literal(path)
    if path.is_relative_to(c.HOME) and runtime.files.exists(path):
        return runtime.document(path)
    entry = exports.get(str(path))
    require(entry is not None,
            "Root bounded operation required: export the exact completed prerequisite JSON artifact "
            + str(path) + " to HOME with its actual descriptor; LOGIN cannot inspect DATA")
    ref = descriptor(entry["reference"])
    raw = entry["text"].encode("utf-8")
    require(reference(path, raw) == ref, "Actual custody artifact descriptor changed")
    return ref, unpack_json(raw)


def validate_artifact(runtime, cfg, role, path, exports):
    ref, value = artifact(runtime, path, exports)
    if role == "training":
        training = runtime.checked(cfg["training_configuration"])
        checks.training_result(cfg, value, training)
        accepted_ref = descriptor(value["accepted"])
        actual_ref, accepted = artifact(runtime, accepted_ref["path"], exports)
        require(actual_ref == accepted_ref and accepted["status"] == "ACCEPTED_FULL_NATIVE_STATE"
                and accepted["global_step"] == 516
                and accepted["identity"] == value["config_identity"],
                "Actual completed parent lacks its exact accepted final516 metadata")
    elif role == "qualifier":
        checks.qualification_result(cfg, value, runtime.checked)
    else:
        qref = validate_artifact(runtime, cfg, "qualifier",
                                cfg["required_actual_qualification_result_path"], exports)
        checks.independent_result(cfg, value, qref, runtime.checked)
    return ref


def completed_accounting(runtime, job, job_name):
    columns = "JobIDRaw,State,ExitCode,User,Account,Partition,QOS,JobName%256,Start,ElapsedRaw"
    raw = runtime.query(["sacct", "-X", "-nP", "-j", job, "--format=" + columns])
    rows = [row for row in table(raw, 10) if row[0] == job]
    require(len(rows) == 1 and rows[0][1:8] == [
        "COMPLETED", "0:0", USER, "cx", "general", "normal", job_name]
        and rows[0][8] not in ("", "Unknown", "None", "N/A")
        and rows[0][9].isdecimal() and int(rows[0][9]) > 0,
        "Aged-out parent requires actual successful owned allocation accounting, not stale afterok")
    return raw


def prerequisites(runtime, cfg, exports):
    expected = {
        "training": (cfg["required_training_job_id"], cfg["student_completion_path"]),
        "qualifier": (cfg["required_loader_qualifier_job_id"], cfg["required_actual_qualification_result_path"]),
    }
    entries = cfg["prerequisites"]
    require(len(entries) == 3 and {entry["role"] for entry in entries} == set(expected)|{"reviewer"},
            "Exact real training/qualifier/reviewer prerequisites required")
    dependencies, proofs = [], []
    for entry in entries:
        role = entry["role"]
        if role=="reviewer":
            contract,value,qref,verified=c.materialized_reviewer(cfg,runtime.checked)
            binding=runtime.checked(contract['binding'])
            require(entry['kind']==c.REGISTERED_SCHEMA and 'job_id'not in entry
                and entry['contract']==cfg['registered_independent_review_contract']
                and entry['binding']==contract['binding']and entry['registered_step']==binding['registered_step']
                and entry['artifact_path']==contract['review']['path'],
                'Exact already completed registered step reviewer; never scalar afterok')
            # Still authenticate real DATA qualifier output via bounded exact export.
            actual_qref=validate_artifact(runtime,cfg,'qualifier',cfg['required_actual_qualification_result_path'],exports)
            checks.independent_result(cfg,value,actual_qref,runtime.checked)
            proofs.append(dict(role='reviewer',registered_step=binding['registered_step'],binding=contract['binding'],
                materialized_contract=cfg['registered_independent_review_contract'],artifact=contract['review'],
                completed_registered_step=verified,no_parent_afterok=True,no_fictitious_batch_ACK=True))
            continue
        job, path = expected[role]
        require(entry["job_id"] == job and entry["artifact_path"] == str(literal(path)),
                "Exact actual predecessor ID/path required")
        binding = runtime.checked(entry["binding"])
        require(binding["job_id"] == job and binding["owner"] == "Root", "Actual predecessor binding differs")
        proof = dict(role=role, job_id=job, binding=entry["binding"], artifact_path=path)
        if role == "training":
            require(entry["binding"] == cfg["training_cell"]["training_binding"]
                    and binding["job_name"].endswith("-" + cfg["student_id"]),
                    "Original exact training physical binding and authenticated job name")
            name = binding["job_name"]
        else:
            if role == "qualifier":
                require(binding["status"] == "ROOT_BOUND_NATIVE_TRAINED_EVAL_QUALIFICATION_V4"
                        and binding["configuration"] == cfg["qualified_loader_configuration"],
                        "Actual V4 qualifier configuration differs")
                name = "dsmm-native-trained-eval-qual-v4"
            else:
                require(binding["status"] == "ROOT_BOUND_INDEPENDENT_TRAINED_QUAL_REVIEW_V4",
                        "Actual V4 independent reviewer binding differs")
                review_cfg = runtime.checked(binding["configuration"])
                require(review_cfg["subject_configuration"] == cfg["qualified_loader_configuration"]
                        and review_cfg["subject_job_id"] == cfg["required_loader_qualifier_job_id"]
                        and review_cfg["review_output_path"] == path, "Reviewer must consume this exact V4 subject")
                name = "dsmm-trained-qual-review-v4"
            proof["genuine_original_intent_ACK"] = checks.acked_prerequisite(
                runtime, entry["binding"], binding, cfg, role)
        result = runtime.command(["scontrol", "show", "job", "-o", job])
        if result.returncode == 0:
            record = fields(result.stdout.decode("utf-8"))
            require(record.get("JobId") == job and record.get("JobName") == name
                    and record.get("UserId") == f"{USER}({UID})" and record.get("Account") == "cx"
                    and record.get("Partition") == "general" and record.get("QOS") == "normal"
                    and record.get("JobState") in ACTIVE | {"COMPLETED"},
                    "Actual predecessor failed, cancelled, foreign or mismatched")
            if role != "training":
                prefix = "dsmm-trained-eval-qual:" if role == "qualifier" else "dsmm-trained-qual-review:"
                require(record.get("Comment") == prefix + binding["nonce"],
                        "Current prerequisite nonce differs from its genuine ACKed binding")
            proof["scheduler"] = result.stdout.decode("utf-8")
            if record["JobState"] == "COMPLETED":
                require(record.get("ExitCode") == "0:0", "Predecessor completed with a failure")
                proof["artifact"] = validate_artifact(runtime, cfg, role, path, exports)
            else:
                require(record.get("Reason") != "DependencyNeverSatisfied",
                        "Terminal dependency failure is not genuine pending readiness")
                dependencies.append(job)
                proof["actual_PASS_not_inferred_from_pending_afterok"] = True
        else:
            require(result.returncode is not None and b"Invalid job id specified" in result.stderr,
                    "Ambiguous parent lookup is not aged-out completion")
            proof["completed_accounting"] = completed_accounting(runtime, job, name)
            proof["artifact"] = validate_artifact(runtime, cfg, role, path, exports)
            proof["aged_out_no_stale_afterok"] = True
        proofs.append(proof)
    return dependencies, proofs


def storage(runtime, cfg, path, configs, own_reservation=None):
    ref, value = runtime.document(path)
    require(value["status"] == STORAGE_STATUS and value["owner"] == "Root"
            and value["read_only"] is True and value["neural_forwards"] == 0
            and 0 <= runtime.now() - value["epoch"] <= 300
            and value["output_prefix"] == str(c.OUTPUT),
            "Fresh actual registered GPU observation of the exact result filesystem is required")
    registration = runtime.checked(value["registration"])
    job, step = str(value["actual_job_id"]), str(value["actual_step_id"])
    require(job.isdecimal() and step.isdecimal() and registration["owner"] == "Root"
            and str(registration["job_id"]) == job and str(registration["step_id"]) == step
            and registration["output_prefix"] == str(c.OUTPUT) and registration["read_only"] is True,
            "Storage observation lacks its actual owned registered GPU step")
    recorded = fields(value["actual_parent_scheduler"])
    current = fields(runtime.query(["scontrol", "show", "job", "-o", job]))
    for record in (recorded, current):
        require(record.get("JobId") == job and record.get("UserId") == f"{USER}({UID})"
                and record.get("Account") == "cx" and record.get("Partition") == "general"
                and record.get("QOS") == "normal" and record.get("JobState") in {"RUNNING", "COMPLETING"}
                and record.get("JobName", "").startswith("dsmm")
                and record.get("TresPerNode") == "gres/gpu:L40S:1"
                and record.get("NodeList") == value["node"].split(".")[0],
                "Storage measurement must retain a real current owned General GPU actor")
    claims = runtime.checked(value["collective_claims"])
    require(claims["owner"] == "Root" and claims["covers_all_project_obligations"] is True
            and 0 <= runtime.now() - claims["epoch"] <= 300,
            "Root must reconcile actual collective unwritten storage claims")
    outstanding, inodes, included = 0, 0, []
    for entry in claims["entries"]:
        grant = runtime.checked(entry["reservation"])
        require(type(entry["unwritten_bytes"]) is int and 0 <= entry["unwritten_bytes"] <= grant["reserved_bytes"]
                and type(entry["unwritten_inodes"]) is int and entry["unwritten_inodes"] >= 0
                and entry["reservation"] not in included, "Invalid or duplicate collective storage obligation")
        if entry["reservation"] != own_reservation:
            outstanding += entry["unwritten_bytes"]
            inodes += entry["unwritten_inodes"]
        included.append(entry["reservation"])
    for student_id in configs:
        prior = ADMISSIONS / student_id / "STORAGE.json"
        if runtime.files.exists(prior):
            prior_ref, _ = runtime.document(prior)
            require(prior_ref == own_reservation or prior_ref in included,
                    "Collective storage export omitted an existing comparator obligation")
    require(type(value["HF_free_bytes"]) is int and value["HF_free_bytes"] >= outstanding + cfg["output_limit_bytes"]
            and type(value["HF_free_inodes"]) is int and value["HF_free_inodes"] >= inodes + 128,
            "Measured shared storage cannot cover this result and collective obligations")
    free_bytes, free_inodes = runtime.files.home_capacity()
    require(free_bytes >= 32 << 20 and free_inodes >= 128, "HOME control receipt capacity is insufficient")
    return ref, outstanding, inodes


def argv_for(cfg, binding_path, dependencies):
    argv = [
        "sbatch", "--parsable", "--hold", "--requeue", "--partition=general", "--account=cx",
        "--qos=normal", "--gres=gpu:L40S:1", "--nodes=1", "--ntasks=1", "--cpus-per-task=8",
        "--mem=64G", "--time=06:00:00", "--exclude="+c.NODE_EXCLUSION, "--job-name=" + c.NAME,
        "--comment=" + c.COMMENT + cfg["nonce"], "--chdir=" + str(c.SOURCE_ROOT),
        "--output=" + str(binding_path.parent / "slurm-%j.out"),
    ]
    if dependencies:
        argv.append("--dependency=afterok:" + ":".join(dependencies))
    return argv + [str(c.SOURCE_ROOT / SCRIPT), str(binding_path)]


def inspect_request(runtime, job, cfg, dependencies, *, held):
    require(re.fullmatch(r"[1-9]\d*", job) is not None, "Own evaluation request must be a scalar, not an array/heterogeneous task")
    raw = runtime.query(["scontrol", "show", "job", "-o", job])
    record = fields(raw)
    wanted = {
        "JobId": job, "JobName": c.NAME, "UserId": f"{USER}({UID})", "Account": "cx",
        "Partition": "general", "QOS": "normal", "TimeLimit": "06:00:00",
        "Requeue": "1", "NumCPUs": "8", "NumTasks": "1",
        "Command": str(c.SOURCE_ROOT / SCRIPT), "WorkDir": str(c.SOURCE_ROOT),
        "TresPerNode": "gres/gpu:L40S:1", "Comment": c.COMMENT + cfg["nonce"],
    }
    require(all(record.get(key) == value for key, value in wanted.items())
            and record.get("NumNodes") in ("1", "1-1")
            and record.get("ArrayJobId") in (None, "", "(null)", "0")
            and record.get("ArrayTaskId") in (None, "", "(null)", "N/A")
            and record.get("HetJobId") in (None, "", "(null)", "0"),
            "Exact owned scalar resource/source request differs; no release")
    require(record.get("ExcNodeList")==c.NODE_EXCLUSION,"Exact temporary new-request node exclusion differs")
    requested = tres(record.get("ReqTRES"))
    require(requested.get("cpu") == "8" and requested.get("node") == "1"
            and requested.get("mem") in ("64G", "65536M") and gpu_count(record) == 1,
            "Requested CPU8/64GiB/GPU1 route differs")
    observed_dependencies = dependency_ids(record.get("Dependency"))
    require(observed_dependencies <= set(dependencies), "Foreign dependency in the actual request")
    if held:
        require(record.get("JobState") == "PENDING" and record.get("Reason") == "JobHeldUser"
                and record.get("Restarts") == "0" and record.get("RunTime") == "00:00:00"
                and record.get("StartTime") in ("Unknown", "None", "N/A", "(null)")
                and record.get("NodeList", "") in ("", "(null)", "None")
                and record.get("AllocTRES") in (None, "", "(null)"),
                "Only this exact never-started UserHeld request can be released")
    else:
        require(record.get("JobState") in ACTIVE
                and record.get("Reason") not in ("JobHeldUser", "DependencyNeverSatisfied"),
                "Release acknowledgement is not proof of an eligible request")
    spool = runtime.command(["scontrol", "write", "batch_script", job, "-"])
    require(spool.returncode == 0 and spool.stdout == runtime.files.read(c.SOURCE_ROOT / SCRIPT),
            "Actual immutable batch spool differs from the reviewed deployed bytes")
    return {"scheduler": raw, "spool_sha256": digest(spool.stdout),
            "dependencies": sorted(observed_dependencies)}


def seal_same(runtime, path, value):
    if runtime.files.exists(path):
        ref, previous = runtime.document(path)
        require(previous == value, "Immutable admission receipt differs; Root must reconcile")
        return ref
    return runtime.files.seal(path, value)


def ack_job(value):
    stdout = base64.b64decode(value["stdout_base64"], validate=True)
    match = re.fullmatch(rb"([1-9]\d*)(?:;[A-Za-z0-9_.-]+)?\s*", stdout)
    return match[1].decode("ascii") if value["returncode"] == 0 and match else None


def verify_review_refs(runtime, refs):
    for ref in refs.values():
        runtime.checked(ref)


def verify_owned_binding(runtime, binding_ref, cfg, refs, job):
    binding = runtime.checked(binding_ref)
    require(binding["status"] == c.BINDING_STATUS and binding["owner"] == "Root"
            and binding["job_id"] == job and binding["nonce"] == cfg["nonce"]
            and binding["function_key"] == function_identity(cfg)
            and all(binding[key] == value for key, value in refs.items()),
            "Immutable owned binding differs from the exact reviewed function")
    claim = runtime.checked(binding["ownership"])
    grant = runtime.checked(binding["storage_reservation"])
    require(claim["owner"] == "Root" and claim["job_id"] == job
            and claim["configuration"] == refs["configuration"] and claim["nonce"] == cfg["nonce"]
            and claim["function_key"] == function_identity(cfg)
            and claim["training_configuration"] == cfg["training_configuration"],
            "Immutable exact-function ownership differs")
    require(grant["configuration"] == refs["configuration"] and grant["output"] == cfg["output"]
            and grant["reserved_bytes"] == cfg["output_limit_bytes"]
            and grant["reserved_inodes"] == 128
            and type(grant["other_outstanding_reserved_bytes"]) is int
            and grant["other_outstanding_reserved_bytes"] >= 0
            and type(grant["other_outstanding_reserved_inodes"]) is int
            and grant["other_outstanding_reserved_inodes"] >= 0
            and grant["final_checkpoint_retained_for_evaluation"] is True
            and grant["checkpoint_retention"] == cfg["checkpoint_retention"],
            "Immutable output/retention/current-other-claims grant differs")
    return binding, grant


def release_receipt(cfg, job, binding, release_ack, observed):
    return {
        "status": "ACTUAL_ROOT_COMPARATOR_SCALAR_RELEASED_WITH_REAL_PREREQUISITES_V1",
        "owner": "Root", "student_id": cfg["student_id"], "job_id": job, "binding": binding,
        "release_ack": release_ack, "verified_released": observed,
        "actual_training_or_qualification_PASS_not_inferred_from_afterok": True,
        "no_scientific_evaluation_result_claimed": True,
    }


def admit(configuration_path, packet_path, review_path, root_approval_path, storage_path,
          historical_proof_path=None, *, execute_root_reviewed=False,
          reconcile_existing=False, runtime=None):
    require(execute_root_reviewed is True, "No operation: --execute-root-reviewed is required")
    runtime = runtime or Runtime()
    require(runtime.uid() == UID, "Only the actual Babel Root-owned project actor may admit")
    with runtime.files.lock() as lock_check:
        require(callable(lock_check), "An inode-rechecking canonical shared lock is required")
        cfg, configs, refs = reviewed(runtime, configuration_path, packet_path, review_path, root_approval_path)
        c.source7_prefix_gate(cfg,load=runtime.checked)
        folder = ADMISSIONS / cfg["student_id"]
        binding_path = folder / "BINDING.json"
        intent_path, ack_path = folder / "INTENT.json", folder / "ACK.json"
        function_key = function_identity(cfg)
        function_path = ADMISSIONS / ("FUNCTION-" + function_key + ".json")
        existing = runtime.files.exists(intent_path)
        if runtime.files.exists(folder / "RELEASED.json"):
            _, receipt = runtime.document(folder / "RELEASED.json")
            verify_owned_binding(runtime, receipt["binding"], cfg, refs, receipt["job_id"])
            return receipt
        require(not runtime.files.exists(folder) or existing,
                "Partial custody directory exists: reconcile explicitly, never submit into it")
        require(not existing or reconcile_existing,
                "Existing intent/ACK: use --reconcile-existing; never repeat sbatch")
        snapshot = census(runtime, additional=0 if existing else 1)
        matches = duplicate_check(snapshot, cfg, configs, existing=existing)
        exports = exported_documents(runtime, historical_proof_path)
        dependencies, parent_proofs = prerequisites(runtime, cfg, exports)
        if not existing:
            require(not runtime.files.exists(function_path), "Scientific function already has immutable custody")
            measured, other_reserved_bytes, other_reserved_inodes = storage(runtime, cfg, storage_path, configs)
            runtime.files.mkdir(folder)
            function_ref = runtime.files.seal(function_path, {
                "owner": "Root", "function_key": function_key, "configuration": refs["configuration"],
                "training_configuration": cfg["training_configuration"], "student_id": cfg["student_id"],
                "banks": cfg["banks"], "counts": cfg["counts"], "transaction": str(folder),
            })
            grant = runtime.files.seal(folder / "STORAGE.json", {
                "configuration": refs["configuration"], "output": cfg["output"],
                "reserved_bytes": cfg["output_limit_bytes"], "actual_collective_measurement": measured,
                "reserved_inodes": 128, "other_outstanding_reserved_bytes": other_reserved_bytes,
                "other_outstanding_reserved_inodes": other_reserved_inodes,
                "training_configuration": cfg["training_configuration"],
                "final_checkpoint_retained_for_evaluation": True,
                "checkpoint_retention": cfg["checkpoint_retention"],
                "retention_release_condition": "Root_acknowledges_exact_configured_consumer_closure",
                "late_bound_final_accepted_step": 516, "no_checkpoint_or_base_copies": True,
            })
            argv = argv_for(cfg, binding_path, dependencies)
            intent = runtime.files.seal(intent_path, {
                "owner": "Root", **refs, "function_key": function_key, "function_claim": function_ref,
                "storage_reservation": grant, "nonce": cfg["nonce"], "argv": argv,
                "dependencies": dependencies, "prerequisites": parent_proofs,
                "census": snapshot, "epoch": runtime.now(),
                "actual_qualification_PASS_not_assumed": True,
                "worker_must_validate_historical_physical272_and_full_saved_state": True,
            })
            # The raw response is durably sealed before any job-ID parsing.
            lock_check()
            ack_ref = runtime.files.seal(ack_path, runtime.command(argv).receipt())
            ack = runtime.checked(ack_ref)
        else:
            intent, original = runtime.document(intent_path)
            require(original["owner"] == "Root" and original["configuration"] == refs["configuration"]
                    and original["function_key"] == function_key and original["nonce"] == cfg["nonce"]
                    and original["source_packet"] == refs["source_packet"]
                    and original["root_personal_approval"] == refs["root_personal_approval"],
                    "Original immutable intent differs from the reviewed transaction")
            require(len(original["dependencies"]) == len(set(original["dependencies"]))
                    and set(original["dependencies"]) <= {entry["job_id"] for entry in cfg["prerequisites"]if entry["role"]in("training","qualifier")}
                    and original["argv"] == argv_for(cfg, binding_path, original["dependencies"]),
                    "Original scalar argv/dependency custody differs")
            dependencies = original["dependencies"]
            require(runtime.checked(original["function_claim"])["function_key"] == function_key,
                    "Scientific custody differs")
            grant = original["storage_reservation"]
            runtime.checked(grant)
            if runtime.files.exists(ack_path):
                ack_ref, ack = runtime.document(ack_path)
            else:
                ack_ref, ack = None, None
        job = ack_job(ack) if ack is not None else None
        if job is None:
            snapshot = census(runtime)
            matches = duplicate_check(snapshot, cfg, configs, existing=True)
            require(len(matches) == 1,
                    "Ambiguous submission: preserve intent/raw ACK; Root must identify the exact held request, never retry sbatch")
            job = matches[0]
            reconciliation_path = folder / "ACK_RECONCILIATION.json"
            if runtime.files.exists(reconciliation_path):
                _, previous = runtime.document(reconciliation_path)
                require(previous["job_id"] == job and previous["intent"] == intent
                        and previous["raw_ack"] == ack_ref,
                        "Original submission reconciliation identifies a different transaction")
            else:
                runtime.files.seal(reconciliation_path, {
                    "job_id": job, "intent": intent, "raw_ack": ack_ref,
                    "raw_ack_absent_after_interruption": ack is None,
                    "exact_nonce_census": snapshot,
                })
        require(not matches or matches == [job], "ACK and actual live controller disagree")
        previous_release = 0
        while runtime.files.exists(folder / f"RELEASE_INTENT_{previous_release + 1}.json"):
            previous_release += 1
        require(not runtime.files.exists(folder / f"RELEASE_ACK_{previous_release + 1}.json"),
                "Release ACK without its durable intent: Root custody reconciliation required")
        if existing and previous_release:
            bound_ref, _ = runtime.document(binding_path)
            bound, _ = verify_owned_binding(runtime, bound_ref, cfg, refs, job)
            _, last_release = runtime.document(folder / f"RELEASE_INTENT_{previous_release}.json")
            require(last_release["job_id"] == job and last_release["binding"] == bound_ref
                    and bound["admission_intent"] == intent,
                    "Previous release intent does not belong to this immutable transaction")
            current_record = fields(runtime.query(["scontrol", "show", "job", "-o", job]))
            if current_record.get("JobState") != "PENDING" or current_record.get("Reason") != "JobHeldUser":
                observed = inspect_request(runtime, job, cfg, dependencies, held=False)
                verify_review_refs(runtime, refs)
                release_ack_path = folder / f"RELEASE_ACK_{previous_release}.json"
                release_ack_ref = runtime.document(release_ack_path)[0] if runtime.files.exists(release_ack_path) else None
                reconciled = seal_same(runtime, folder / "RELEASE_RECONCILIATION.json", {
                    "job_id": job, "binding": bound_ref, "release_intent": last_release,
                    "raw_release_ack": release_ack_ref, "actual_eligible_request": observed,
                    "no_scheduler_mutation_during_reconciliation": True,
                })
                receipt = release_receipt(cfg, job, bound_ref, release_ack_ref, observed)
                receipt["reconciliation"] = reconciled
                runtime.files.seal(folder / "RELEASED.json", receipt)
                return receipt
        observed = inspect_request(runtime, job, cfg, dependencies, held=True)
        ownership = seal_same(runtime, folder / "OWNERSHIP.json", {
            "owner": "Root", "scope": cfg["scope"], "job_id": job, "nonce": cfg["nonce"],
            "configuration": refs["configuration"], "training_configuration": cfg["training_configuration"],
            "function_key": function_key, "banks": cfg["banks"], "counts": cfg["counts"],
            "no_live_duplicate_checkpoint_evaluation": True, "function_claim": original["function_claim"] if existing else function_ref,
        })
        binding = seal_same(runtime, binding_path, {
            "status": c.BINDING_STATUS, "owner": "Root", "job_id": job, "nonce": cfg["nonce"],
            **refs, "ownership": ownership, "storage_reservation": grant,
            "function_key": function_key, "admission_intent": intent,
            "same_job_continuation": {"missing_only": True, "max_physical_attempts": cfg["max_physical_attempts"],
                                      "progress_unit": "accepted_exact_function_QA"},
        })
        verify_sources(runtime, cfg["sources"])
        verify_review_refs(runtime, refs)
        _, grant_value = verify_owned_binding(runtime, binding, cfg, refs, job)
        current_measurement, other_reserved_bytes, other_reserved_inodes = storage(
            runtime, cfg, storage_path, configs, own_reservation=grant)
        require(grant_value["other_outstanding_reserved_bytes"] >= other_reserved_bytes
                and grant_value["other_outstanding_reserved_inodes"] >= other_reserved_inodes,
                "Current collective obligations exceed the immutable grant; Root storage reconciliation required")
        current = census(runtime)
        require(duplicate_check(current, cfg, configs, existing=True) == [job],
                "Exact held request is missing from the fresh all-user census")
        observed = inspect_request(runtime, job, cfg, dependencies, held=True)
        _, current_parents = prerequisites(runtime, cfg, exports)
        nonterminal = {proof["job_id"] for proof in current_parents if "artifact" not in proof}
        require(nonterminal <= set(observed["dependencies"]), "A nonterminal afterok prerequisite disappeared")
        verify_review_refs(runtime, refs)
        require(0 <= runtime.now() - runtime.checked(current_measurement)["epoch"] <= 300,
                "Fresh actual storage observation expired during release checks")
        lock_check()
        release_number = previous_release + 1
        release_intent = runtime.files.seal(folder / f"RELEASE_INTENT_{release_number}.json", {
            "job_id": job, "binding": binding, "verified_held": observed, "census": current,
            "current_prerequisites": current_parents,
            "actual_runtime_completion_and_qualification_still_required": True, "epoch": runtime.now(),
        })
        release_ack = runtime.command(["scontrol", "release", job])
        release_ref = runtime.files.seal(folder / f"RELEASE_ACK_{release_number}.json", {
            "intent": release_intent, **release_ack.receipt(),
        })
        require(release_ack.returncode == 0,
                "Release failed or ambiguous; retained exact request/ACK requires reconciliation, never resubmission")
        released = inspect_request(runtime, job, cfg, dependencies, held=False)
        receipt = release_receipt(cfg, job, binding, release_ref, released)
        runtime.files.seal(folder / "RELEASED.json", receipt)
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("configuration", "packet", "review", "root-approval", "storage"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--historical-proof", type=Path)
    parser.add_argument("--execute-root-reviewed", action="store_true")
    parser.add_argument("--reconcile-existing", action="store_true")
    args = parser.parse_args(argv)
    receipt = admit(args.configuration, args.packet, args.review, args.root_approval,
                    args.storage, args.historical_proof,
                    execute_root_reviewed=args.execute_root_reviewed,
                    reconcile_existing=args.reconcile_existing)
    print(json.dumps({"status": receipt["status"], "student_id": receipt["student_id"],
                      "job_id": receipt["job_id"]}, sort_keys=True))


if __name__ == "__main__":
    main()
