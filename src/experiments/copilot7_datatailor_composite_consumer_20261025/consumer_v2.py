"""Locked, audited missing-only overlays; original accepted files stay immutable."""
from contextlib import contextmanager, ExitStack
import fcntl
import json
import os
from pathlib import Path
import re
import socket
import sqlite3
import subprocess
import sys
import time

import numpy as np

from . import consumer as prior


c = prior.c
native = prior.native
hook_v1 = prior.hook_v1


def reviewed_parity(result_ref, review_ref):
    review = c.checked(review_ref)
    reviewer = review["reviewer"]
    c.require(isinstance(reviewer, str)
              and not reviewer.strip().casefold().startswith(("session7", "producer", "implementer")),
              "Independent actual-result reviewer required")
    result = prior.reviewed_parity(result_ref, review_ref)
    audit = c.checked(review["preprocess_audit"])
    c.require(len(audit["calls"]) == 33 and audit["contract"] == hook_v1.CONTRACT
              and result["repaired_record"]["record_id"] == "642651:djWVD94_0",
              "Exact actual33 preprocessing custody and repaired record required")
    return result


def rows(descriptor, identity, roster, *, overlay=False, permitted_repairs=()):
    path = Path(descriptor["path"])
    c.require(c.desc(path) == descriptor, "Immutable feature database changed")
    wal = Path(str(path) + "-wal")
    c.require(not wal.exists() or wal.stat().st_size == 0, "Uncheckpointed/live legacy database")
    database = sqlite3.connect(path.absolute().as_uri() + "?mode=ro&immutable=1", uri=True)
    allowed = set(roster)
    try:
        c.require(dict(database.execute("SELECT key,value FROM metadata")) == dict(
            identity=identity, method="datatailor", roster_sha256=c.key(roster)),
            "Feature measurement/method/roster identity differs")
        values = {}
        for ident, blob, info, tau, length, nans, actual in database.execute("SELECT * FROM features"):
            vector = np.frombuffer(blob, dtype="<f4")
            c.require(ident in allowed and ident not in values and actual == identity
                      and vector.shape == (4096,) and np.isfinite(vector).all()
                      and info is not None and tau is not None
                      and np.isfinite(info) and np.isfinite(tau) and tau > 0 and length > 0 and nans == 0,
                      "Invalid accepted feature row")
            values[ident] = dict(vector=vector.copy(), info=info, tau=tau, length=length, nan_count=nans)
        if overlay:
            audited = {}
            for ident, body in database.execute("SELECT record_id,event FROM preprocess_audit"):
                event = json.loads(body)
                valid_event(event, permitted_repairs)
                c.require(ident in values and ident not in audited, "Foreign/duplicate preprocessing custody")
                audited[ident] = event
            c.require(set(audited) == set(values), "A new overlay row lacks actual hook custody")
    finally:
        database.close()
    c.require(c.desc(path) == descriptor and (not wal.exists() or wal.stat().st_size == 0),
              "Legacy database changed while reading; Root must reconcile its accepted boundary")
    return values


def valid_event(event, permitted_repairs):
    c.require(event["contract"] == hook_v1.CONTRACT and event["input_ids_identical"] is True
              and event["ordinary_masks_identical"] is True
              and event["positive_expanded_target_guard_unchanged"] is True,
              "Actual structured hook evidence differs")
    if any(event["labels_changed"]):
        c.require(event["typed_sources_sha256"] in permitted_repairs,
                  "Unexpected changed mask requires a versioned diagnosis")


class CompatibleStore(prior.CompatibleStore):
    def __init__(self, path, identity, roster, legacy, permitted_repairs):
        super().__init__(path, identity, roster, legacy)
        self.permitted_repairs = set(permitted_repairs)
        self.pending_event = None
        self.new.db.execute("CREATE TABLE IF NOT EXISTS preprocess_audit(record_id TEXT PRIMARY KEY,event TEXT NOT NULL)")
        self.new.db.commit()
        audited = {ident for ident, in self.new.db.execute("SELECT record_id FROM preprocess_audit")}
        if audited != self.new.existing:
            self.new.close()
            raise ValueError("Existing overlay has no matching durable hook custody")
        self._existing = set(self.legacy) | self.new.existing

    @property
    def existing(self):
        return self._existing

    def observe(self, event):
        valid_event(event, self.permitted_repairs)
        c.require(self.pending_event is None, "Multiple preprocessing calls before one native B1 commit")
        self.pending_event = event

    def commit(self, measurements):
        c.require(len(measurements) == 1 and self.pending_event is not None,
                  "No new native feature may commit without its actual B1 mask hook")
        ident = measurements[0]["record_id"]
        c.require(self.missing_batch([ident]), "Never repeat accepted legacy/overlay forwards")
        try:
            self.new.db.execute("INSERT INTO preprocess_audit VALUES(?,?)",
                                (ident, c.canonical(self.pending_event).decode()))
            super().commit(measurements)
        except BaseException:
            self.new.db.rollback()
            raise
        self._existing.add(ident)
        self.pending_event = None


def identity_components(original):
    return dict(original_measurement=original, structured_contract=hook_v1.CONTRACT,
        hook_source=c.sha(hook_v1.__file__), structured_source=c.sha(hook_v1.qualified.__file__),
        continuation_consumer_source=c.sha(__file__))


def overlay_identity(original):
    return c.key(identity_components(original))


@contextmanager
def exclusive(output):
    with (Path(output) / "COMPOSITE_EXECUTION.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


@contextmanager
def legacy_locks(references):
    with ExitStack() as stack:
        for path in sorted({Path(ref["path"]).parent / "EXECUTION.lock" for ref in references}):
            c.require(path.is_file() and not path.is_symlink(), "Original accepted writer lock required")
            lock = stack.enter_context(path.open("rb"))
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        yield


def acquire(config, pool, records, ids, legacy_refs, parity_ref, review_ref,
            native_module, output, attempt, should_stop, *, bound_configuration):
    started = time.monotonic()
    c.require(len(ids) == len(records) == pool["count"] == 660763 and ids == pool["record_ids"]
              and config["method"] == "datatailor" and config["role"] == "features"
              and config["shard_count"] == 16, "Exact final native16-shard feature function required")
    raw = subprocess.check_output(["scontrol", "show", "job", "-o", os.environ["SLURM_JOB_ID"]],
                                  text=True, timeout=30)
    fields = dict(re.findall(r"(\w+)=(\S+)", raw))
    c.require(fields["JobId"] == os.environ["SLURM_JOB_ID"] and fields["JobState"] == "RUNNING"
              and fields["Partition"] == "general" and fields["QOS"] == "normal"
              and fields["Account"] == "cx" and fields["UserId"].split("(")[0] == "luyangz"
              and fields["NodeList"] == socket.gethostname().split(".")[0]
              and fields["TresPerNode"] == "gres/gpu:L40S:1",
              "Actually owned General CPU+GPU allocation required")
    proof = reviewed_parity(parity_ref, review_ref)
    bound = c.checked(bound_configuration)
    c.require(all(bound[field] == config[field] for field in
                  ("pool", "environment", "feature_assets", "native_sources", "shard_count"))
              and config["shard_index"] in bound["global_shard_indices"],
              "Root's bound route/scientific CONFIG differs from the supplied native function")
    qualified = c.checked(proof["configuration"])
    c.require(qualified["inputs"]["environment"] == config["environment"]
              and qualified["inputs"]["assets"] == config["feature_assets"]
              and qualified["native_sources"] == config["native_sources"],
              "Qualified model/runtime/native function differs")
    c.require(all(qualified["sources"].get(name) == digest for name, digest in config["sources"].items()),
              "The actual qualification source closure does not cover the supplied measurement")
    c.require(native_module is sys.modules.get("llava.train.train"),
              "Hook target is not the native module actually reimported by the constructor")
    for relative, path in (
        ("experiments/root_native_datatailor_20261023/features.py", native.__file__),
        ("experiments/copilot7_datatailor_mask_repair_20261024/hook_v1.py", hook_v1.__file__),
        ("experiments/root_native_campaign_v5_20261024/structured_preprocess.py", hook_v1.qualified.__file__),
    ):
        c.require(qualified["sources"][relative] == c.sha(path), "Qualified source changed")
    output = Path(output)
    original_directories = [Path(config["output"])] + [Path(ref["path"]).parent for ref in legacy_refs]
    c.require(all(output.resolve() != path.resolve()
                  and output.resolve() not in path.resolve().parents
                  and path.resolve() not in output.resolve().parents for path in original_directories),
              "Overlay output must be distinct and non-nested with original stores")
    attempt = Path(attempt)
    c.require(output.resolve() in attempt.resolve().parents, "Consumer attempt must be within owned overlay output")
    output.mkdir(parents=True, exist_ok=True)
    with exclusive(output), legacy_locks(legacy_refs):
        attempt.mkdir(parents=True, exist_ok=False)
        roster = native.roster(config, records, ids)
        original = native.identity(config, pool)
        legacy = {}
        for ref in legacy_refs:
            part = rows(ref, original, roster)
            c.require(not legacy.keys() & part.keys(), "Duplicate legacy feature")
            legacy.update(part)
        identity = overlay_identity(original)
        store = CompatibleStore(output / "features.sqlite3", identity, roster, legacy,
                                proof["permitted_changed_typed_sources"])
        before = len(store.existing)
        def observe(event):
            try:
                store.observe(event)
            except ValueError as error:
                try:
                    c.seal(attempt / "REFUSED_PREPROCESS.json", dict(event=event, error=str(error)))
                except Exception as receipt_error:
                    raise RuntimeError(
                        f"Refused preprocess {event!r}: {error}; receipt failed: {receipt_error!r}") from error
                raise
        try:
            with hook_v1.installed(native_module, observe):
                if before < len(roster) and not should_stop():
                    native.compute(dict(config, output=str(output)), pool, records, ids,
                                   store, should_stop, False, attempt)
            after = len(store.existing)
        except BaseException as error:
            try:
                store.close()
            except Exception as close_error:
                raise RuntimeError(f"Acquisition failed: {error!r}; close failed: {close_error!r}") from error
            raise
        else:
            store.close()
        for ref in legacy_refs:
            c.require(c.desc(ref["path"]) == ref
                      and (not Path(ref["path"] + "-wal").exists()
                           or Path(ref["path"] + "-wal").stat().st_size == 0),
                      "Original writer/state changed; do not seal a compatible completion")
        return c.seal(attempt / "COMPOSITE_PROGRESS.json", dict(
            schema="s7-compatible-native-DT-composite-v2",
            status="COMPLETE_COMPATIBLE_COMPOSITE_FEATURE_SHARD" if after == len(roster)
                   else "MISSING_ONLY_COMPATIBLE_FEATURE_PROGRESS",
            pool=config["pool"], shard_index=config["shard_index"], shard_count=16,
            original_measurement=original, overlay_measurement=identity,
            overlay_identity_components=identity_components(original), bound_configuration=bound_configuration,
            legacy_databases=legacy_refs, overlay_database=c.desc(output / "features.sqlite3"),
            legacy_records=len(legacy), before=before, after=after, required=len(roster),
            roster_sha256=c.key(roster), actual_parity=parity_ref, independent_review=review_ref,
            hook_custody_atomic_with_each_overlay_row=True, exclusive_overlay_execution=True,
            no_legacy_copy_relabel_overwrite_or_reacquisition=True, actual_slurm=raw,
            Root_must_own_any_continuation=True, automatic_submission_or_requeue=False,
            elapsed_seconds=time.monotonic() - started, new_forwards=after - before,
            fullpool_student_or_scientific_PASS=False))


def load_complete(manifest_ref, roster, expected_original):
    manifest = c.checked(manifest_ref)
    c.require(manifest["schema"] == "s7-compatible-native-DT-composite-v2"
              and manifest["status"] == "COMPLETE_COMPATIBLE_COMPOSITE_FEATURE_SHARD"
              and manifest["original_measurement"] == expected_original
              and manifest["overlay_measurement"] == overlay_identity(expected_original)
              and manifest["overlay_identity_components"] == identity_components(expected_original)
              and manifest["roster_sha256"] == c.key(roster) and manifest["required"] == len(roster),
              "Complete shard, same frozen function and actual roster required")
    proof = reviewed_parity(manifest["actual_parity"], manifest["independent_review"])
    values = {}
    for ref in manifest["legacy_databases"]:
        part = rows(ref, expected_original, roster)
        c.require(not values.keys() & part.keys(), "Duplicate old row")
        values.update(part)
    part = rows(manifest["overlay_database"], manifest["overlay_measurement"], roster,
                overlay=True, permitted_repairs=proof["permitted_changed_typed_sources"])
    c.require(not values.keys() & part.keys(), "Repeated old/new feature row")
    values.update(part)
    c.require(set(values) == set(roster), "Actual composite shard coverage incomplete")
    return values


def load_full_pool(manifest_refs, ids, expected_pool, expected_original):
    c.require(len(manifest_refs) == 16 and len(ids) == len(set(ids)) == 660763,
              "All16 exact fullpool shards required")
    expected = native.roster
    values = {}
    seen = set()
    for ref in manifest_refs:
        manifest = c.checked(ref)
        shard = manifest["shard_index"]
        c.require(manifest["pool"] == expected_pool and manifest["shard_count"] == 16
                  and type(shard) is int and 0 <= shard < 16 and shard not in seen,
                  "Different/duplicate shard or pool")
        selected = expected(dict(role="features", shard_index=shard, shard_count=16), [], ids)
        part = load_complete(ref, selected, expected_original)
        c.require(not values.keys() & part.keys(), "Repeated fullpool feature")
        values.update(part)
        seen.add(shard)
    c.require(seen == set(range(16)) and set(values) == set(ids), "Incomplete fullpool union")
    return values
