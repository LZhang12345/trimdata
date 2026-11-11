"""Unqualified transport proposal only; never creates a site authority PASS."""
from . import pool_authority_v1 as authority

c = authority.c


def propose(upstream_authority, upstream_eligibility, transport_plan, output):
    actual = authority.require_allocation()
    raw = c.checked(upstream_authority)
    eligibility = c.checked(upstream_eligibility)
    c.require(raw["schema"] == "native-full-source-pool-authority-v1" and
              raw["status"] == "PASS_EXACT_ORIGINAL_PARTITION_AND_ELIGIBLE_MEDIA" and
              raw["source"] == authority.source_identity() and
              raw["media_closure"]["actual_encoded_and_decoded_RGB_checks"] is True and
              eligibility["status"] == "PASS_FROZEN_POOL_EVALUATION_ELIGIBILITY" and
              eligibility["input_authority"] == upstream_authority and
              eligibility["protected_contract"] == raw["protected_contract"],
              "Original completed upstream raw authority and independent eligibility are required")
    pool = c.checked(eligibility["pool"])
    c.require(pool["input_authority"] == upstream_authority and
              pool["records_json"] == raw["records_json"] and pool["image_manifest"] == raw["image_manifest"] and
              pool["protected_exclusions"] == raw["exclusions"], "Upstream pool/partition/media receipt differs")
    plan = c.checked(transport_plan)
    c.require(plan["schema"] == "native-archive-selected-tmpdir-transport-plan-v1" and
              plan["persist_extracted_full_pool"] is False and plan["rehash_redecode_every_consumed_image"] is True,
              "Explicit archive plus selected-only verified materialization plan required")
    logical = {"records_sha256": raw["records_json"]["sha256"],
               "record_ids_sha256": authority.row_hash(pool["record_ids"]),
               "source_indices_sha256": authority.row_hash(pool["source_indices"]),
               "partition_sha256": raw["exclusions"]["sha256"],
               "dimensioned_media_digest": raw["media_closure"]["dimensioned_encoded_RGB_identity_sha256"],
               "eligible_N": raw["eligible_N"], "K": raw["K"]}
    return c.seal(output, {
        "schema": "native-site-rebind-unqualified-proposal-v1",
        "status": "UNQUALIFIED_SITE_REBIND_PROPOSAL",
        "upstream_raw_authority": upstream_authority, "upstream_eligibility": upstream_eligibility,
        "upstream_producer_source": raw["source"], "logical_identity_must_be_preserved": logical,
        "transport_plan": transport_plan, "current_physical_media_readiness": False,
        "runtime_transport_required": True, "runtime_consumer_enabled": False, "actual": actual,
        "remaining": [
            "Independent exact-source and exact rebind/path-only proof review",
            "Verify unchanged logical rows/IDs/source indices and original partition/dimensioned RGB content after transfer",
            "Bind exact durable archives and selected-only TMPDIR aliases, checking every consumed image",
            "Actual site runtime/restore qualification; no new producer physical PASS is inferred",
        ],
        "no_runtime_rebinding_performed": True, "no_new_physical_media_PASS": True})
