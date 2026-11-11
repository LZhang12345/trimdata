"""Preserve upstream authority bytes; defer physical images to owned node TMPDIR."""
from collections import Counter
import fcntl
from pathlib import Path
import time
import zipfile

from experiments.root_native_input_authority_20261023 import pool_authority_v1 as upstream
from experiments.root_native_media_transport_20261023 import transport

c = upstream.c


def source_identity():
    root = c.ROOT
    files = ("experiments/session5_native_archive_authority_20261024/__init__.py",
             "experiments/session5_native_archive_authority_20261024/archive_authority_v1.py",
             "experiments/root_native_campaign_20261023/common.py",
             "experiments/root_native_input_authority_20261023/pool_authority_v1.py",
             "experiments/root_native_media_transport_20261023/transport.py")
    return {name: {"bytes": (root / name).stat().st_size, "sha256": c.sha(root / name)} for name in files}


def _descriptor(value):
    c.require(isinstance(value, dict) and set(value) == {"path", "bytes", "sha256"} and
              Path(value["path"]).is_absolute() and type(value["bytes"]) is int and value["bytes"] > 0 and
              isinstance(value["sha256"], str) and len(value["sha256"]) == 64 and
              all(char in "0123456789abcdef" for char in value["sha256"]), "Exact file descriptor required")
    return value


class MetadataBindings:
    """Only explicit byte-identical metadata aliases, never implicit path fallback."""

    def __init__(self, document):
        c.require(document["schema"] == "native-exact-metadata-path-bindings-v1", "Wrong metadata rebind schema")
        self.bindings, self.local, self.values = {}, {}, {}
        for value in document["bindings"]:
            original, local = _descriptor(value["original"]), _descriptor(value["local"])
            c.require(original["path"] not in self.bindings and
                      (original["bytes"], original["sha256"]) == (local["bytes"], local["sha256"]),
                      "Duplicate source path or changed metadata bytes")
            self.bindings[original["path"]] = original
            self.local[original["path"]] = local

    def local_descriptor(self, original):
        _descriptor(original)
        c.require(self.bindings.get(original["path"]) == original, "Missing or foreign original metadata binding")
        return self.local[original["path"]]

    def checked(self, original):
        local = self.local_descriptor(original)
        if original["path"] not in self.values:
            self.values[original["path"]] = c.checked(local)
        return self.values[original["path"]]


def _upstream_identity(pool, authority, eligibility, expected_pool, expected_authority, expected_eligibility):
    c.require(pool["schema"] == "published-llava-pool-v1" and pool["status"] == "PASS_POOL" and
              pool["pool_kind"] == "all_source_protected_bridge" and
              authority["schema"] == "native-full-source-pool-authority-v1" and
              authority["status"] == "PASS_EXACT_ORIGINAL_PARTITION_AND_ELIGIBLE_MEDIA" and
              authority["source"] == upstream.source_identity() and
              authority["physical_media_readiness_scope"] == "producer_snapshot_only" and
              authority["media_closure"]["actual_encoded_and_decoded_RGB_checks"] is True,
              "Genuinely completed full-source upstream producer receipt required; engineering cannot substitute")
    c.require(eligibility["status"] == "PASS_FROZEN_POOL_EVALUATION_ELIGIBILITY" and
              eligibility["pool"] == expected_pool and eligibility["input_authority"] == expected_authority and
              eligibility["protected_contract"] == authority["protected_contract"] and
              eligibility["protected_QA_image_RGB_overlap_count"] == 0 and
              eligibility["missing_media_count"] == 0 and eligibility["outcome_tuning"] is False and
              pool["input_authority"] == expected_authority,
              "Independent upstream pool/authority/protection eligibility differs")
    c.require(expected_eligibility["path"] and authority["original_metadata"] == pool["source_json"] and
              authority["records_json"] == pool["records_json"] and
              authority["image_manifest"] == pool["image_manifest"] and
              authority["exclusions"] == pool["protected_exclusions"],
              "Upstream original/records/partition/media identity differs")


def export_upstream(pool_description, eligibility_description, execution_proof_description, output):
    """Inside a registered upstream GPU worker; export references, not image trees."""
    actual = upstream.require_allocation()
    pool, _, _ = upstream.validate_pool(pool_description, eligibility_description)
    eligibility = c.checked(eligibility_description)
    authority_description = pool["input_authority"]
    authority = c.checked(authority_description)
    _upstream_identity(pool, authority, eligibility, pool_description, authority_description, eligibility_description)
    proof = c.checked(execution_proof_description)
    c.require(proof["status"] == "PASS_ACTUAL_FULL_POOL_AUTHORITY_EXECUTION" and
              proof["pool"] == pool_description and proof["authority"] == authority_description and
              proof["eligibility"] == eligibility_description and proof["producer_source"] == authority["source"] and
              proof["original_source_count"] == upstream.ORIGINAL_COUNT and
              proof["full_eligible_count"] == authority["eligible_N"] and proof["engineering_only"] is False,
              "Independent genuine completed upstream function proof required")
    protected = c.checked(eligibility["protected_contract"])
    descriptors = [
        pool_description, eligibility_description, authority_description, execution_proof_description,
        pool["source_json"], pool["records_json"], pool["image_manifest"], pool["protected_exclusions"],
        authority["media_inventory"], authority["protected_contract"], protected["role_manifest"],
        protected["evaluation_protocol"],
    ]
    unique = {}
    for description in descriptors:
        _descriptor(description)
        c.checked(description)
        if description["path"] in unique:
            c.require(unique[description["path"]] == description, "Conflicting upstream metadata path")
        unique[description["path"]] = description
    return c.seal(output, {
        "schema": "native-full-authority-export-v1", "status": "COMPLETE_UPSTREAM_AUTHORITY_METADATA_EXPORT",
        "upstream_pool": pool_description, "upstream_eligibility": eligibility_description,
        "upstream_authority": authority_description, "upstream_execution_proof": execution_proof_description,
        "raw_upstream_authority": authority, "metadata": list(unique.values()),
        "producer_source": authority["source"], "adapter_sources": source_identity(), "actual": actual,
        "physical_images_exported": 0, "downstream_site_physical_PASS": False})


def _logical(export, expected_eligibility, bindings):
    c.require(export["schema"] == "native-full-authority-export-v1" and
              export["status"] == "COMPLETE_UPSTREAM_AUTHORITY_METADATA_EXPORT" and
              export["upstream_eligibility"] == expected_eligibility and export["physical_images_exported"] == 0 and
              export["adapter_sources"] == source_identity(),
              "Exact external upstream eligibility/export required")
    c.require({value["path"] for value in export["metadata"]} == set(bindings.bindings),
              "Metadata rebind is missing entries or silently adds foreign inputs")
    for value in export["metadata"]:
        bindings.checked(value)
    pool = bindings.checked(export["upstream_pool"])
    eligibility = bindings.checked(expected_eligibility)
    authority = bindings.checked(export["upstream_authority"])
    c.require(authority == export["raw_upstream_authority"] and
              authority["source"] == export["producer_source"] == upstream.source_identity(),
              "Raw original upstream receipt or producer code changed")
    _upstream_identity(pool, authority, eligibility, export["upstream_pool"], export["upstream_authority"],
                       expected_eligibility)
    proof = bindings.checked(export["upstream_execution_proof"])
    c.require(proof["status"] == "PASS_ACTUAL_FULL_POOL_AUTHORITY_EXECUTION" and
              proof["pool"] == export["upstream_pool"] and proof["authority"] == export["upstream_authority"] and
              proof["eligibility"] == expected_eligibility and proof["producer_source"] == authority["source"] and
              proof["engineering_only"] is False and proof["original_source_count"] == upstream.ORIGINAL_COUNT and
              proof["full_eligible_count"] == authority["eligible_N"], "Upstream execution proof differs")
    original_description = pool["source_json"]
    c.require(original_description["bytes"] == upstream.ORIGINAL_BYTES and
              original_description["sha256"] == upstream.ORIGINAL_SHA, "Pinned original665298 metadata differs")
    original = bindings.checked(original_description)
    c.require(isinstance(original, list) and len(original) == upstream.ORIGINAL_COUNT, "Original source roster differs")
    media = bindings.checked(authority["media_inventory"])
    c.require(media["schema"] == "native-authority-media-inventory-v1" and
              media["original_metadata"] == original_description, "Upstream source-bound media manifest differs")
    protected = bindings.checked(eligibility["protected_contract"])
    c.require(protected["evaluation_protocol"] == eligibility["evaluation_protocol"],
              "Independent protection/evaluation protocol differs")
    roles = bindings.checked(protected["role_manifest"])
    protocol = bindings.checked(protected["evaluation_protocol"])
    upstream.verify_protection_authority(protected, roles, protocol)
    index = upstream.ProtectedIndex(protected)
    partition = upstream.partition(original, media["assets"], index)
    rows = bindings.checked(pool["records_json"])
    exclusions = bindings.checked(pool["protected_exclusions"])
    upstream.verify_partition(original, rows, pool["source_indices"], pool["record_ids"], exclusions["excluded"], partition)
    c.require(pool["count"] == len(rows) == authority["eligible_N"] and
              authority["K"] == upstream.budgets(len(rows)) and
              authority["validity_policy"] == exclusions["validity_policy"] == upstream.VALIDITY_POLICY and
              authority["source_counts"] == exclusions["source_counts"] == partition["source_counts"] and
              exclusions["original_count"] == upstream.ORIGINAL_COUNT and exclusions["retained_count"] == len(rows) and
              exclusions["retained_indices_sha256"] == upstream.row_hash(pool["source_indices"]) and
              exclusions["protected_contract"] == eligibility["protected_contract"] and
              index.coverage == authority["role_coverage"], "Complete partition/roles/denominator/validity drift")
    images = bindings.checked(pool["image_manifest"])["assets"]
    c.require(set(images) == set(partition["required_eligible_images"]) and
              all(images[name] == media["assets"][name] for name in images), "Canonical eligible media closure differs")
    closure = upstream.logical_media_closure(media, partition["checked_images"])
    c.require(authority["media_closure"] == {**closure, "actual_encoded_and_decoded_RGB_checks": True} and
              authority["eligible_unique_images"] == len(images) and pool["image_root"] == media["image_root"],
              "Original dimensioned encoded/RGB producer proof differs")
    c.require(authority["protected_audit_counts"] == dict(Counter(
        reason["reason"] for value in partition["excluded"] for reason in value["reasons"])) and
        authority["outcome_tuning"] is False and authority["top_up_or_replacement"] is False and
        authority["trainer_or_signal_qualified"] is False,
        "Original protection/policy/source scope was changed")
    logical = {
        "original_metadata_sha256": original_description["sha256"],
        "records_sha256": pool["records_json"]["sha256"],
        "record_ids_sha256": upstream.row_hash(pool["record_ids"]),
        "source_indices_sha256": upstream.row_hash(pool["source_indices"]),
        "partition_sha256": pool["protected_exclusions"]["sha256"],
        "dimensioned_media_digest": closure["dimensioned_encoded_RGB_identity_sha256"],
        "protected_contract_sha256": eligibility["protected_contract"]["sha256"],
        "evaluation_protocol_sha256": eligibility["evaluation_protocol"]["sha256"],
        "eligible_N": len(rows), "K": authority["K"],
    }
    return pool, rows, images, logical


def _catalog(document, images):
    c.require(document["schema"] == "native-archive-locator-catalog-v1" and
              set(document["images"]) == set(images), "Missing/foreign eligible archive image locator")
    roots = [Path(value) for value in document["allowed_read_roots"]]
    c.require(roots and all(any(root == allowed or allowed in root.parents for allowed in transport.READ_ROOTS)
                          for root in roots), "Archive/shared-file root not approved")
    used = set()
    for name, locator in document["images"].items():
        transport.relative_name(name)
        c.require(locator["canonical_identity"] ==
                  {key: images[name][key] for key in ("bytes", "sha256", "width", "height", "rgb_sha256")},
                  "Archive locator changes canonical image bytes/RGB")
        if locator["kind"] == "zip":
            transport.relative_name(locator["member"])
            c.require(locator["archive"] in document["archives"], "Missing archive descriptor")
            used.add(locator["archive"])
        else:
            c.require(locator["kind"] == "shared_file", "Unknown media locator kind")
            path = Path(locator["path"])
            c.require(path.is_absolute() and ".." not in path.parts and
                      any(path == root or root in path.parents for root in roots),
                      "Shared image locator escaped approved roots")
    c.require(used == set(document["archives"]), "Unreferenced or incomplete archive set")
    for value in document["archives"].values():
        _descriptor({key: value[key] for key in ("path", "bytes", "sha256")})
        transport.literal(value["path"], roots)
        provider = c.checked(value["provider"])
        c.require(provider["status"] in ("PASS_MEDIA_ARCHIVE", "STAGED_EXACT_SELECTED_MEDIA_ARCHIVE") and
                  {key: provider["archive"][key] for key in ("path", "bytes", "sha256")} ==
                  {key: value[key] for key in ("path", "bytes", "sha256")},
                  "Exact durable archive provider proof differs")
    return roots


def _review(config, description):
    review = c.checked(description)
    c.require(review["status"] == "PASS_INDEPENDENT_EXACT_NATIVE_ARCHIVE_REBIND" and
              review["config_identity"] == c.key(config) and review["sources"] == source_identity(),
              "Exact independent archive rebind config/source review required")
    return review


def _archive_stat(path):
    value = Path(path).stat()
    return {"device": value.st_dev, "inode": value.st_ino, "bytes": value.st_size,
            "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns}


def _accepted_archive(path, expected, key, config, catalog, images, output, actual, deadline):
    before = _archive_stat(path)
    identity = {"archive": expected, "stat": before, "config_identity": c.key(config),
                "sources": source_identity()}
    receipt_path = output / "accepted-archives" / (c.key(identity) + ".json")
    if receipt_path.exists():
        previous = c.read(receipt_path)
        c.require(previous["identity"] == identity and previous["status"] == "PASS_EXACT_ARCHIVE_AND_HEADER_MAPPING",
                  "Accepted archive boundary belongs to another source/file/config")
        return c.desc(receipt_path), 0
    c.require({"path": str(path), "bytes": before["bytes"],
               "sha256": transport.sha(path, deadline)} == expected, "Durable archive encoded bytes changed")
    wanted = {locator["member"] for locator in catalog["images"].values()
              if locator["kind"] == "zip" and locator["archive"] == key}
    with zipfile.ZipFile(path) as archive:
        members = transport._zip_members(archive, wanted)
        for name, locator in catalog["images"].items():
            if time.monotonic() >= deadline:
                raise InterruptedError("Accepted archive/header boundary; retry missing-only without new acquisition")
            if locator["kind"] == "zip" and locator["archive"] == key:
                c.require(members[locator["member"]].file_size == images[name]["bytes"],
                          "Archive member size differs from canonical image")
    c.require(_archive_stat(path) == before, "Archive changed during hash/header verification")
    receipt = c.seal(receipt_path, {"status": "PASS_EXACT_ARCHIVE_AND_HEADER_MAPPING", "identity": identity,
                                  "actual": actual, "encoded_bytes_hashed": expected["bytes"],
                                  "member_payloads_rehashed": False,
                                  "scope": "Archive file and header mapping only; selected images still hash/decode"})
    return receipt, expected["bytes"]


def _produce_rebind(config_description, review_description, output):
    """Verify durable archive files once; publish logical-only local transport."""
    actual = transport._actual()
    started = time.monotonic()
    config = c.checked(config_description)
    _review(config, review_description)
    c.require(config["schema"] == "native-archive-authority-rebind-config-v1", "Wrong rebind config")
    export = c.checked(config["upstream_export"])
    bindings = MetadataBindings(c.checked(config["metadata_bindings"]))
    pool, rows, images, logical = _logical(export, config["expected_upstream_eligibility"], bindings)
    catalog = c.checked(config["locator_catalog"])
    _catalog(catalog, images)
    output = Path(output)
    archives, archive_receipts, hashed_bytes = {}, {}, 0
    reserve = config["deadline_reserve_seconds"]
    c.require(type(reserve) is int and 180 <= reserve <= 600, "Explicit accepted-boundary reserve required")
    deadline = started + actual["remaining_seconds"] - reserve
    c.require(time.monotonic() < deadline, "Insufficient remaining allocation for archive authority")
    for key, value in catalog["archives"].items():
        path = Path(value["path"])
        expected = {name: value[name] for name in ("path", "bytes", "sha256")}
        proof, count = _accepted_archive(path, expected, key, config, catalog, images, output, actual, deadline)
        archive_receipts[key] = proof
        hashed_bytes += count
        archives[key] = dict(value)
    local_pool = {
        "schema": "native-archive-site-pool-view-v1", "status": "LOGICAL_ONLY_RUNTIME_TRANSPORT_REQUIRED",
        "upstream_pool": export["upstream_pool"], "upstream_authority": export["upstream_authority"],
        "records_json": bindings.local_descriptor(pool["records_json"]),
        "source_json": bindings.local_descriptor(pool["source_json"]),
        "image_manifest": bindings.local_descriptor(pool["image_manifest"]),
        "protected_exclusions": bindings.local_descriptor(pool["protected_exclusions"]),
        "record_ids": pool["record_ids"], "source_indices": pool["source_indices"], "count": pool["count"],
        "current_physical_media_readiness": False, "runtime_transport_required": True,
        "logical_identity": logical}
    site_pool = c.seal(output / "SITE_POOL.json", local_pool)
    locator_index = c.seal(output / "LOCATOR_INDEX.json", {
        "schema": "native-media-locator-index-v1", "scope": "ALL_SOURCE", "logical_pool": site_pool,
        "canonical_image_manifest": local_pool["image_manifest"],
        "source_coverage": dict(Counter(name.split("/", 1)[0] for name in images)),
        "images": {name: {key: value for key, value in locator.items() if key != "canonical_identity"}
                   for name, locator in catalog["images"].items()}, "archives": archives})
    receipt = {
        "schema": "native-full-authority-archive-rebind-v1",
        "status": "COMPLETE_LOGICAL_ARCHIVE_REBIND_SELECTED_PHYSICAL_CHECKS_REQUIRED",
        "configuration": config_description, "review": review_description,
        "upstream_raw_authority": export["upstream_authority"],
        "upstream_eligibility": config["expected_upstream_eligibility"], "producer_source": export["producer_source"],
        "site_pool": site_pool, "locator_index": locator_index, "logical_identity": logical,
        "accepted_archive_boundaries": archive_receipts,
        "current_physical_media_readiness": False, "runtime_transport_required": True,
        "actual": actual, "sources": source_identity(), "full_pool_images_extracted": 0,
        "new_site_producer_physical_PASS": False,
        "cost": {"elapsed_seconds": time.monotonic() - started,
                 "durable_archive_bytes_hashed_this_attempt": hashed_bytes,
                 "accepted_archive_cost_provenance": archive_receipts}}
    receipt_path = output / "REBIND.json"
    if receipt_path.exists():
        previous = c.read(receipt_path)
        transient = {"actual", "cost"}
        c.require({key: value for key, value in previous.items() if key not in transient} ==
                  {key: value for key, value in receipt.items() if key not in transient},
                  "Accepted rebind/provenance differs")
        return c.desc(receipt_path)
    return c.seal(receipt_path, receipt)


def produce_rebind(config_description, review_description, output):
    upstream.require_allocation()
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "REBIND.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _produce_rebind(config_description, review_description, directory)


def validate_archive_pool(rebind_description, expected_upstream_eligibility, expected_review):
    """No full archive hashing or image decoding in the logical consumer."""
    upstream.require_allocation()
    value = c.checked(rebind_description)
    c.require(value["schema"] == "native-full-authority-archive-rebind-v1" and
              value["status"] == "COMPLETE_LOGICAL_ARCHIVE_REBIND_SELECTED_PHYSICAL_CHECKS_REQUIRED" and
              value["upstream_eligibility"] == expected_upstream_eligibility and value["review"] == expected_review and
              value["sources"] == source_identity() and value["current_physical_media_readiness"] is False and
              value["runtime_transport_required"] is True and value["new_site_producer_physical_PASS"] is False and
              value["full_pool_images_extracted"] == 0, "Foreign rebind or false current physical readiness")
    config = c.checked(value["configuration"])
    _review(config, expected_review)
    c.require(config["expected_upstream_eligibility"] == expected_upstream_eligibility,
              "Independent upstream eligibility changed")
    export = c.checked(config["upstream_export"])
    bindings = MetadataBindings(c.checked(config["metadata_bindings"]))
    pool, rows, images, logical = _logical(export, expected_upstream_eligibility, bindings)
    catalog = c.checked(config["locator_catalog"])
    _catalog(catalog, images)
    c.require(set(value["accepted_archive_boundaries"]) == set(catalog["archives"]),
              "Missing accepted durable archive boundary")
    for key, description in value["accepted_archive_boundaries"].items():
        proof = c.checked(description)
        expected = {name: catalog["archives"][key][name] for name in ("path", "bytes", "sha256")}
        c.require(proof["status"] == "PASS_EXACT_ARCHIVE_AND_HEADER_MAPPING" and
                  proof["identity"]["archive"] == expected and proof["identity"]["config_identity"] == c.key(config) and
                  proof["identity"]["sources"] == source_identity() and
                  Path(expected["path"]).stat().st_size == expected["bytes"],
                  "Original accepted archive proof or current file availability differs")
    site_pool = c.checked(value["site_pool"])
    c.require(site_pool["logical_identity"] == logical == value["logical_identity"] and
              site_pool["record_ids"] == pool["record_ids"] and site_pool["source_indices"] == pool["source_indices"] and
              site_pool["count"] == len(rows) and
              site_pool["records_json"] == bindings.local_descriptor(pool["records_json"]) and
              site_pool["image_manifest"] == bindings.local_descriptor(pool["image_manifest"]) and
              site_pool["source_json"] == bindings.local_descriptor(pool["source_json"]) and
              site_pool["protected_exclusions"] == bindings.local_descriptor(pool["protected_exclusions"]) and
              site_pool["upstream_pool"] == export["upstream_pool"] and
              site_pool["upstream_authority"] == export["upstream_authority"] and
              site_pool["current_physical_media_readiness"] is False and site_pool["runtime_transport_required"] is True,
              "Rebased pool changes logical content, metadata paths or current transport requirement")
    index = c.checked(value["locator_index"])
    c.require(index["schema"] == "native-media-locator-index-v1" and index["scope"] == "ALL_SOURCE" and
              index["logical_pool"] == value["site_pool"] and
              index["canonical_image_manifest"] == site_pool["image_manifest"] and
              index["source_coverage"] == dict(Counter(name.split("/", 1)[0] for name in images)) and
              index["archives"] == catalog["archives"] and
              index["images"] == {name: {key: data for key, data in item.items() if key != "canonical_identity"}
                                  for name, item in catalog["images"].items()},
              "Exact archive locator/provenance changed")
    c.require(value["upstream_raw_authority"] == export["upstream_authority"] and
              value["producer_source"] == export["producer_source"], "Original upstream raw receipt was replaced")
    return site_pool, rows, pool["record_ids"]


def prepare_selected(rebind_description, expected_upstream_eligibility, expected_review,
                     selection_description, selected_records_local, transport_description, report):
    """Reuse the frozen, reviewed per-attempt hash/decode materializer."""
    site_pool, rows, ids = validate_archive_pool(rebind_description, expected_upstream_eligibility, expected_review)
    value = c.checked(rebind_description)
    selection = c.checked(selection_description)
    c.require(selection["schema"] == "native-record-selection-v1" and selection["status"] == "PASS_SELECTION" and
              selection["pool"] == site_pool["upstream_pool"], "Unchanged upstream selection identity required")
    chosen = selection["record_ids"]
    c.require(0 < len(chosen) == len(set(chosen)) == selection["count"] and set(chosen) <= set(ids),
              "Foreign, duplicate or incomplete selected original IDs")
    fraction = selection["fraction"]
    c.require(fraction in upstream.budgets(len(ids)) and
              len(chosen) <= upstream.budgets(len(ids))[fraction] and
              selection["budget"] == {"unit": "whole_conversation_count", "value": len(chosen),
                                      "fraction": fraction, "eligible_pool_count": len(ids)},
              "Actual selected-data fraction/denominator/K differs; no silent topup")
    c.require((selection["records_json"]["bytes"], selection["records_json"]["sha256"]) ==
              (selected_records_local["bytes"], selected_records_local["sha256"]),
              "Cross-site selected whole-conversation bytes changed")
    selected_rows = c.checked(selected_records_local)
    originals = dict(zip(ids, rows))
    c.require(len(selected_rows) == len(chosen) and
              all(upstream.row_hash(row) == upstream.row_hash(originals[key])
                  for row, key in zip(selected_rows, chosen)), "Changed selected row/turn/order")
    cfg = c.checked(transport_description)
    c.require(cfg["scope"] == "ALL_SOURCE" and cfg["locator_index"] == value["locator_index"] and
              cfg["canonical_image_manifest"] == site_pool["image_manifest"],
              "Current consumer transport must use the exact complete rebound index")
    alias = transport.prepare(value["site_pool"], selected_records_local, transport_description, report)
    transport.validate(alias, value["site_pool"], selected_records_local)
    return alias
