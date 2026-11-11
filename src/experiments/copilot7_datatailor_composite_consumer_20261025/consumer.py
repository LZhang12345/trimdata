from pathlib import Path
import sqlite3

import numpy as np

from experiments.root_native_campaign_20261023 import common as c
from experiments.root_native_baselines_20261023.store import Store
from experiments.root_native_datatailor_20261023 import features as native
from experiments.copilot7_datatailor_mask_repair_20261024 import hook_v1


def reviewed_parity(result_ref, review_ref):
    result, review = c.checked(result_ref), c.checked(review_ref)
    c.require(result["status"] == "PASS_ACTUAL_NATIVE33_FEATURE_PARITY_PENDING_INDEPENDENT_REVIEW"
              and result["accepted_records"] == 33 and result["reference_controls"] == 32
              and result["repaired_records"] == 1 and result["repaired_record"]["targets"] == 499
              and result["exact_tolerance"] == {"absolute": 0, "relative": 0},
              "Actual exact native33 result is required, not source tests or a pending job")
    c.require(review["status"] == "PASS_INDEPENDENT_ACTUAL_S7_NATIVE33_PARITY"
              and review["result"] == result_ref
              and review["configuration"] == result["configuration"]
              and review["reviewer"] not in ("", "Session7", "producer", None),
              "Independent exact-result consumption required before old-row compatibility")
    controls = result["exact_controls"]
    c.require(len(controls) == 32 and all(all(item["exact_fields"].values())
              and item["FP32_vector_bytes_identical"] and item["scalars_identical"] for item in controls),
              "A reversed/failed native parity comparison cannot authorize compatibility")
    audit = c.checked(review["preprocess_audit"])
    repaired = [event for event in audit["calls"] if any(event["labels_changed"])]
    c.require(len(repaired) == 1 and repaired[0]["native_shifted_target_counts"] == [0]
              and repaired[0]["repaired_shifted_target_counts"] == [499]
              and repaired[0]["contract"] == hook_v1.CONTRACT,
              "The exact actually qualified changed-mask input must be bound by independent review")
    return dict(result, permitted_changed_typed_sources={repaired[0]["typed_sources_sha256"]})


def rows(descriptor, identity, roster):
    path = Path(descriptor["path"])
    c.require(c.desc(path) == descriptor, "Immutable feature database changed")
    wal = Path(str(path) + "-wal")
    c.require(not wal.exists() or wal.stat().st_size == 0,
              "Live/uncheckpointed legacy WAL requires Root's accepted-boundary reconciliation")
    allowed = set(roster)
    database = sqlite3.connect(path.absolute().as_uri() + "?mode=ro", uri=True)
    try:
        metadata = dict(database.execute("SELECT key,value FROM metadata"))
        c.require(metadata == dict(identity=identity, method="datatailor", roster_sha256=c.key(roster)),
                  "Legacy measurement/method/original roster identity differs")
        values = {}
        for ident, blob, info, tau, length, nans, actual in database.execute("SELECT * FROM features"):
            vector = np.frombuffer(blob, dtype="<f4")
            c.require(ident in allowed and ident not in values and actual == identity
                      and vector.shape == (4096,) and np.isfinite(vector).all()
                      and np.isfinite(info) and np.isfinite(tau) and tau > 0
                      and type(length) is int and length > 0 and nans == 0,
                      "Foreign, duplicate or invalid accepted DataTailor value")
            values[ident] = dict(vector=vector.copy(), info=info, tau=tau, length=length, nan_count=nans)
    finally:
        database.close()
    c.require(c.desc(path) == descriptor, "Legacy database changed during read-only validation")
    return values


class CompatibleStore:
    """Never copy, reidentify, append to, or overwrite the legacy database."""

    def __init__(self, path, identity, roster, legacy):
        self.allowed = set(roster)
        self.legacy = frozenset(legacy)
        c.require(self.legacy <= self.allowed, "Legacy rows outside the exact global shard")
        self.new = Store(path, identity, roster, "datatailor")
        if self.legacy.intersection(self.new.existing):
            self.new.close()
            raise ValueError("Legacy/new duplicate acquisition")

    @property
    def existing(self):
        return self.legacy | self.new.existing

    def missing_batch(self, ids):
        c.require(set(ids) <= self.allowed and len(ids) == len(set(ids)), "Foreign feature batch")
        present = [ident in self.existing for ident in ids]
        c.require(all(present) or not any(present), "Partial original batch requires diagnosis")
        return not all(present)

    def commit(self, measurements):
        c.require(self.missing_batch([item["record_id"] for item in measurements]),
                  "Never reacquire or relabel an accepted legacy row")
        return self.new.commit(measurements)

    def close(self):
        self.new.close()


def acquire(config, pool, records, ids, legacy_refs, parity_ref, review_ref,
            native_module, output, attempt, should_stop):
    """Call only from Root's separately reviewed, actual General GPU wrapper."""
    import os
    import re
    import socket
    import subprocess
    c.require(len(ids) == len(records) == pool["count"] == 660763
              and ids == pool["record_ids"] and config["shard_count"] == 16
              and config["method"] == "datatailor" and config["role"] == "features",
              "Exact final whole-conversation population and feature role required")
    raw = subprocess.check_output(["scontrol", "show", "job", "-o", os.environ["SLURM_JOB_ID"]],
                                  text=True, timeout=30)
    fields = dict(re.findall(r"(\w+)=(\S+)", raw))
    c.require(fields["JobId"] == os.environ["SLURM_JOB_ID"] and fields["JobState"] == "RUNNING"
              and fields["Partition"] == "general" and fields["QOS"] == "normal"
              and fields["Account"] == "cx" and fields["UserId"].split("(")[0] == "luyangz"
              and fields["NodeList"] == socket.gethostname().split(".")[0]
              and fields["TresPerNode"] == "gres/gpu:L40S:1",
              "New consumer work is General-only inside an actually owned CPU+GPU allocation")
    proof = reviewed_parity(parity_ref, review_ref)
    qualification = c.checked(proof["configuration"])
    c.require(qualification["inputs"]["environment"] == config["environment"]
              and qualification["inputs"]["assets"] == config["feature_assets"]
              and qualification["native_sources"] == config["native_sources"],
              "Compatibility proof uses another model/runtime/native source")
    for relative, path in (
        ("experiments/root_native_datatailor_20261023/features.py", native.__file__),
        ("experiments/copilot7_datatailor_mask_repair_20261024/hook_v1.py", hook_v1.__file__),
        ("experiments/root_native_campaign_v5_20261024/structured_preprocess.py", hook_v1.qualified.__file__),
    ):
        c.require(qualification["sources"][relative] == c.sha(path), "Qualified numerical/hook source changed")
    roster = native.roster(config, records, ids)
    base_identity = native.identity(config, pool)
    legacy = {}
    for descriptor in legacy_refs:
        value = rows(descriptor, base_identity, roster)
        c.require(not legacy.keys() & value.keys(), "Repeated legacy record across stores")
        legacy.update(value)
    overlay_identity = c.key(dict(original_measurement=base_identity, structured_contract=hook_v1.CONTRACT,
        hook_source=c.sha(hook_v1.__file__), structured_source=c.sha(hook_v1.qualified.__file__),
        continuation_consumer_source=c.sha(__file__)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    c.require(output.resolve() != Path(config["output"]).resolve(),
              "New overlay output must be distinct from the bound original shard output")
    c.require(all((output / "features.sqlite3").resolve() != Path(item["path"]).resolve()
                  for item in legacy_refs), "Never alias a legacy database as the new output")
    adapted = dict(config, output=str(output))
    store = CompatibleStore(output / "features.sqlite3", overlay_identity, roster, legacy)
    before = len(store.existing)
    audit = []
    def observed(event):
        if any(event["labels_changed"]):
            c.require(event["typed_sources_sha256"] in proof["permitted_changed_typed_sources"],
                      "Unknown mask-repair domain requires a versioned diagnosis")
        audit.append(event)
    try:
        with hook_v1.installed(native_module, observed):
            if len(store.existing) < len(roster) and not should_stop():
                native.compute(adapted, pool, records, ids, store, should_stop, False, attempt)
        after = len(store.existing)
    finally:
        store.close()
        c.seal(Path(attempt) / "MASK_AUDIT.json", dict(events=audit, original_measurement=base_identity,
            overlay_measurement=overlay_identity, no_legacy_rows_reidentified=True))
    for descriptor in legacy_refs:
        c.require(c.desc(descriptor["path"]) == descriptor, "Legacy store changed during continuation")
    return c.seal(Path(attempt) / "COMPOSITE_PROGRESS.json", dict(
        status="COMPLETE_COMPATIBLE_COMPOSITE_FEATURE_SHARD" if after == len(roster)
               else "MISSING_ONLY_COMPATIBLE_FEATURE_PROGRESS",
        pool=config["pool"], shard_index=config["shard_index"], shard_count=config["shard_count"],
        original_measurement=base_identity, overlay_measurement=overlay_identity,
        legacy_databases=legacy_refs, overlay_database=c.desc(output / "features.sqlite3"),
        legacy_records=len(legacy), before=before, after=after, required=len(roster),
        roster_sha256=c.key(roster), actual_parity=parity_ref, independent_review=review_ref,
        no_legacy_copy_relabel_overwrite_or_reacquisition=True, actual_slurm=raw,
        Root_must_own_any_continuation=True, automatic_submission_or_requeue=False,
        fullpool_student_or_scientific_PASS=False))


def load_complete(manifest, roster):
    c.require(manifest["status"] == "COMPLETE_COMPATIBLE_COMPOSITE_FEATURE_SHARD"
              and manifest["roster_sha256"] == c.key(roster) and manifest["required"] == len(roster),
              "Only a complete exact composite shard may feed native operators")
    reviewed_parity(manifest["actual_parity"], manifest["independent_review"])
    values = {}
    for descriptor, identity in [
        *[(ref, manifest["original_measurement"]) for ref in manifest["legacy_databases"]],
        (manifest["overlay_database"], manifest["overlay_measurement"]),
    ]:
        part = rows(descriptor, identity, roster)
        c.require(not values.keys() & part.keys(), "Duplicate legacy/overlay feature record")
        values.update(part)
    c.require(set(values) == set(roster), "Composite full-shard feature coverage incomplete")
    return values
