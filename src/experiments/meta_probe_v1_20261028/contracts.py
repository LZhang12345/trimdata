"""Strict shared identity, reference-role and forecast contracts."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

from .errors import NotReadyError, ProbeError, require

ABI_VERSION = "shared-native-probe-abi-v3"
TASKS = ("POPE", "GQA", "TextVQA", "ScienceQA_image")
K = 66076
GLOBAL_BATCH = 128
FORMAL_UPDATES = 516
MAX_METADATA = 32 << 20


def _json_value(value: object) -> None:
    if isinstance(value, dict):
        require(all(type(key) is str for key in value), "Canonical JSON requires string keys")
        for item in value.values():
            _json_value(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _json_value(item)
    else:
        require(value is None or type(value) in (str, bool, int, float), "Unsupported identity value")
        if type(value) is float:
            require(math.isfinite(value), "Nonfinite identity value")


def canonical(value: object) -> bytes:
    _json_value(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def identity(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def descriptor(path: str | Path, raw: bytes) -> dict[str, object]:
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON metadata key: " + key)
        result[key] = value
    return result


class ScopedReader:
    """Actual execution pins finite roots/inventory; no generic DATA permission."""

    def __init__(self, allowed_roots: Sequence[str | Path], inventory: Sequence[Mapping[str, Any]],
                 *, owners: Sequence[int] | None = None):
        require(bool(allowed_roots) and bool(inventory), "Explicit finite metadata roots/inventory required")
        self.roots = tuple(Path(root) for root in allowed_roots)
        require(all(root.is_absolute() and ".." not in root.parts for root in self.roots),
                "Canonical absolute allowed metadata roots")
        self.inventory = {ref["path"]: dict(ref) for ref in inventory}
        require(len(self.inventory) == len(inventory), "Duplicate declared inventory path")
        self.owners = frozenset((os.getuid(),) if owners is None else owners)
        require(bool(self.owners) and all(type(uid) is int and uid >= 0 for uid in self.owners),
                "Explicit actual metadata owner identities")

    def bytes(self, ref: Mapping[str, Any]) -> bytes:
        return _read_bytes(ref, self)

    def __call__(self, ref: Mapping[str, Any]) -> dict[str, Any]:
        return _decode(self.bytes(ref))


def _read_bytes(ref: Mapping[str, Any], policy: ScopedReader) -> bytes:
    from experiments.root_current_four_bank_eval_v1_20261027.admission import parent_fd
    require(set(ref) == {"path", "bytes", "sha256"}, "Exact real metadata descriptor required")
    require(type(ref["path"]) is str and type(ref["bytes"]) is int
            and 0 < ref["bytes"] <= MAX_METADATA
            and isinstance(ref["sha256"], str)
            and re.fullmatch(r"[a-f0-9]{64}", ref["sha256"]) is not None,
            "Bounded actual bytes/SHA256 metadata descriptor required")
    path = Path(ref["path"])
    require(path.is_absolute() and str(path) == ref["path"] and ".." not in path.parts
            and "\x00" not in ref["path"] and policy.inventory.get(str(path)) == dict(ref)
            and any(path.is_relative_to(root) for root in policy.roots),
            "Foreign or undeclared metadata path/inventory")
    signature = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def regular(info):
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_uid in policy.owners
                and not info.st_mode & 0o022 and info.st_size == ref["bytes"],
                "Regular single-link owned safe-mode metadata required")

    with parent_fd(path) as (parent, name):
        parent_before = os.fstat(parent)
        before_path = os.stat(name, dir_fd=parent, follow_symlinks=False)
        regular(before_path)
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            before = os.fstat(fd)
            regular(before)
            require(signature(before_path) == signature(before), "Metadata replaced before open")
            chunks, size = [], 0
            while size < before.st_size:
                chunk = os.read(fd, min(65536, before.st_size - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
            after = os.fstat(fd)
            regular(after)
            current = os.stat(name, dir_fd=parent, follow_symlinks=False)
            regular(current)
            require(signature(before) == signature(after) == signature(current) and size == before.st_size,
                    "Metadata changed during bounded single-FD read")
            with parent_fd(path) as (fresh_parent, fresh_name):
                current_parent = os.fstat(fresh_parent)
                require((parent_before.st_dev, parent_before.st_ino) ==
                        (current_parent.st_dev, current_parent.st_ino)
                        and signature(os.stat(fresh_name, dir_fd=fresh_parent, follow_symlinks=False)) == signature(after),
                        "Canonical metadata parent/path was retargeted")
        finally:
            os.close(fd)
    raw = b"".join(chunks)
    require(descriptor(path, raw) == dict(ref), "Metadata descriptor hash changed")
    return raw


def _decode(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw, object_pairs_hook=_unique,
                       parse_constant=lambda text: (_ for _ in ()).throw(ProbeError("Nonfinite JSON: " + text)))
    require(type(value) is dict, "Metadata artifact must be a mapping")
    _json_value(value)
    return value


def read_checked(ref: Mapping[str, Any]) -> dict[str, Any]:
    raise NotReadyError("Actual metadata reads require a ScopedReader with reviewed allowed roots/inventory")


def validate_measurement_target(
    target: Mapping[str, Any], candidate_ids: Sequence[str], candidate_image_groups: Sequence[str],
    *, intrinsic_context: Mapping[str, Any] | None = None,
    targeted_mode: bool = False, targeted_protocol=None, load=read_checked,
) -> dict[str, Any]:
    kind = target.get("kind")
    if kind == "NONE_INTRINSIC_TRAIN_TARGET":
        require(target.get("external_reference") is None,
                "Intrinsic TRAIN measurements do not invent an external reference")
        if intrinsic_context is None:
            raise NotReadyError("Actual original native TRAIN pool/target provenance is missing")
        require(intrinsic_context.get("role") == "QUALIFIED_NATIVE_TRAIN_RECORDS"
                and intrinsic_context.get("evaluation_label_targeting") is False
                and target.get("pool") == intrinsic_context.get("pool")
                and target.get("native_target_protocol") == intrinsic_context.get("native_target_protocol")
                and target.get("source_provenance") == intrinsic_context.get("source_provenance")
                and all(target.get(key) is not None for key in ("pool", "native_target_protocol", "source_provenance"))
                and set(candidate_ids) <= set(intrinsic_context["record_ids"]),
                "Genuine original TRAIN target/pool/source identity required")
        evaluation_ids = target.get("evaluation_record_ids")
        if evaluation_ids is not None:
            require(type(evaluation_ids) is list and bool(evaluation_ids)
                    and len(evaluation_ids) == len(set(evaluation_ids))
                    and set(evaluation_ids) <= set(intrinsic_context["record_ids"]),
                    "Intrinsic evaluation rows must be exact original TRAIN membership")
        return dict(kind=kind, units="NATS", task_vector_status="NOT_APPLICABLE",
                    native_target_protocol=target["native_target_protocol"], pool=target["pool"],
                    evaluation_record_ids=evaluation_ids)
    if kind == "QUALIFIED_TRAIN_REFERENCE":
        manifest = validate_references(
            target["reference_bundle"], candidate_ids, candidate_image_groups,
            targeted_mode=targeted_mode, targeted_protocol=targeted_protocol, load=load)
        return dict(kind=kind, manifest=manifest, units="REFERENCE_LOSS")
    raise ProbeError("Unsupported measurement target kind")


def validate_references(
    bundle: Mapping[str, object],
    candidate_ids: Sequence[str],
    candidate_image_groups: Sequence[str],
    *,
    targeted_mode: bool = False,
    targeted_protocol: Mapping[str, object] | None = None,
    load=read_checked,
) -> dict[str, Any]:
    require(type(targeted_mode) is bool, "Targeted mode must be an explicit Boolean")
    if bundle.get("manifest") is None or bundle.get("qualification") is None:
        raise NotReadyError("Qualified native TRAIN reference manifest/provider is not available")
    manifest = load(bundle["manifest"])
    peer = load(bundle["qualification"])
    require(peer.get("independent") is True
            and peer.get("reviewer") not in (None, "", "Session1", "S1")
            and peer.get("manifest") == bundle["manifest"], "Independent exact reference qualification required")
    role = manifest.get("role")
    if role == "TRAIN_PRIMARY":
        require(peer.get("status") == "PASS_INDEPENDENT_QUALIFIED_TRAIN_REFERENCE_V1"
                and manifest.get("source_split") == "train"
                and manifest.get("dataset_revision") and manifest.get("provenance_receipt")
                and manifest.get("protected_evaluation_overlap") is False,
                "Qualified TRAIN provenance/exclusion required, not a split-name assertion")
    elif role == "EXPOSED_DEV_TARGETED_DIAGNOSTIC":
        require(targeted_mode and targeted_protocol is not None,
                "Public evaluation labels are disabled for the primary method")
        protocol = load(targeted_protocol)
        require(protocol.get("status") == "ROOT_APPROVED_SEPARATE_TARGETED_DIAGNOSTIC_PROTOCOL_V1"
                and protocol.get("reference_manifest") == bundle["manifest"]
                and protocol.get("nonoverlap_matched_access_full_reporting") is True
                and peer.get("status") == "PASS_INDEPENDENT_QUALIFIED_TARGETED_REFERENCE_V1",
                "Separate actual Root targeted protocol and qualification required")
    else:
        raise ProbeError("Unsupported reference role")
    records = manifest.get("records")
    require(isinstance(records, list) and bool(records), "Actual qualified reference records required")
    ids = [row["semantic_key"] for row in records]
    require(len(ids) == len(set(ids)) and all(row["task"] in TASKS for row in records),
            "Unique typed reference membership/task identity required")
    record_overlap = set(candidate_ids).intersection(row["record_id"] for row in records)
    image_overlap = set(candidate_image_groups).intersection(
        row["image_group_id"] for row in records if row.get("image_group_id") is not None)
    if record_overlap or image_overlap:
        crossfit = manifest.get("crossfit")
        require(isinstance(crossfit, dict) and crossfit.get("independently_qualified") is True
                and crossfit.get("candidate_reference_disjoint_for_this_fold") is True
                and set(crossfit.get("excluded_candidate_record_ids", [])) >= record_overlap
                and set(crossfit.get("excluded_candidate_image_groups", [])) >= image_overlap,
                "Actual record/image overlap requires explicit qualified fold exclusion")
    return manifest


def require_record_ranking(provider: Mapping[str, object], candidate_ids: Sequence[str]) -> None:
    require(provider.get("scope") == "RECORD_COMPLETE"
            and provider.get("candidate_dependent") is True
            and provider.get("within_group_tied") is False,
            "Constant/group-only provider cannot authorize within-group record ranking")
    covered = provider.get("record_ids")
    require(isinstance(covered, list) and len(covered) == len(set(covered))
            and set(covered) == set(candidate_ids), "Record-complete feature coverage required")
    missing = provider.get("missing_record_ids")
    require(isinstance(missing, list) and not missing, "Missing scores require abstention, never zero fill")
    rows = provider.get("feature_rows")
    require(isinstance(rows, list) and len(rows) == len(candidate_ids)
            and {row["record_id"] for row in rows} == set(candidate_ids),
            "Actual record feature rows, not a rank-authority flag alone")
    groups: dict[str, set[str]] = {}
    sizes: dict[str, int] = {}
    for row in rows:
        group = row["group_id"]
        groups.setdefault(group, set()).add(identity(row["features"]))
        sizes[group] = sizes.get(group, 0) + 1
    require(any(sizes[group] > 1 and len(values) > 1 for group, values in groups.items()),
            "Actual features are group-constant; only allocation/tied selection is supported")


def validate_feature_role(value: Mapping[str, Any]) -> None:
    role = value.get("role")
    times = (value.get("feature_epoch"), value.get("terminal_start_epoch"), value.get("forecast_epoch"))
    require(all(type(item) in (float, int) and math.isfinite(item) for item in times),
            "Truthful actual feature/terminal/forecast timestamps required")
    require(value.get("assessed_terminal_state_or_outcome_used_as_feature") is False,
            "Assessed terminal states/outcomes cannot be selector features")
    if role == "RETROSPECTIVE_CAL":
        require(value.get("all_aliases_role") == "CAL"
                and value.get("feature_state_tier") in ("TRAIN_STATIC", "PRETERMINAL_SHARED_PROBE")
                and value.get("equal_access_across_fit_families") is True,
                "Retrospective TRAIN calibration must preserve CAL aliases and equal access")
    else:
        require(role in ("CAL", "DEV", "CONFIRM", "PREDICT")
                and times[0] < times[1] and times[2] < times[1],
                "Assessment features/forecasts must be prospectively sealed before terminal training")


def validate_forecast(forecast: Mapping[str, Any], *, start_provider: object = None, load=read_checked) -> None:
    kind = forecast.get("kind")
    require(kind in ("TERMINAL_RAW_SCORE_FORECAST", "TERMINAL_START_DELTA_FORECAST"),
            "A loss proxy is not a terminal score forecast")
    if kind == "TERMINAL_START_DELTA_FORECAST" and start_provider is None:
        raise NotReadyError("Qualified compatible START is required for delta/retention")
    vector = forecast.get("task_vector")
    require(isinstance(vector, dict) and set(vector) == set(TASKS), "Exact four-task vector required")
    require(all(type(value) in (int, float) and math.isfinite(value) for value in vector.values()),
            "Finite nonmissing task forecasts required")
    if kind == "TERMINAL_RAW_SCORE_FORECAST":
        require(all(0 <= value <= 100 for value in vector.values()), "Raw metric units are percentages")
    if forecast.get("calibration_geometry") is None or forecast.get("geometry_qualification") is None:
        raise NotReadyError("Actual S6 calibration geometry and independent qualification are missing")
    geometry = load(forecast["calibration_geometry"])
    peer = load(forecast["geometry_qualification"])
    require(forecast.get("frozen_before_assessed_outcome") is True
            and peer.get("independent") is True
            and peer.get("status") == "PASS_INDEPENDENT_QUALIFIED_CALIBRATION_GEOMETRY_V1"
            and peer.get("geometry") == forecast["calibration_geometry"]
            and geometry.get("supports_declared_model") is True,
            "S6-qualified geometry and prospective forecast boundary required")
