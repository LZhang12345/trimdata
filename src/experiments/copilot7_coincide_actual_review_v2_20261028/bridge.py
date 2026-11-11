"""Original intact records only; detached byte-checked review gates publication."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import re
from typing import Callable, Mapping, Sequence

from experiments.meta_probe_v1_20261028.contracts import canonical, require
from experiments.copilot7_coincide_actual_review_20261028.bridge import ADAPTATION, whole_record
from .data_io import decode_json, descriptor

N, K = 660763, 66076
POOL = dict(path="/data/user_data/luyangz/native_campaign_20261023/root-wholepool-authority-v1/authority/POOL.json",
            bytes=18754787, sha256="c76040a17ca0dc1e99fea14514618edade3171a9b05f6ccc3e4751a5939e4964")
CANDIDATE = dict(path="/data/hf_cache/project_features/DS-MultiModal/coincide_native_operator_20261027/v3/budget-0.10/CANDIDATE.json",
                 bytes=1856449, sha256="e704324dc5e6841db1d23f3b990829b54b2b8d564b946c58da8ccec67a6adb24")
SCHEMA = "s7-native-coincide-materialized-selection-candidate-v2"
PENDING = "CANDIDATE_MATERIALIZED_PENDING_INDEPENDENT_IMPLEMENTATION_AND_OUTPUT_REVIEW"


def key(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def independent(review):
    name = review.get("reviewer")
    require(review.get("independent") is True and type(name) is str and name.strip()
            and name.isascii(), "Named genuine independent review required")
    principal = re.sub("[^a-z0-9]", "", name.casefold())
    require(principal not in ("s7", "copilot7", "session7", "sourceauthor")
            and "session7" not in principal and not principal.startswith("copilot7"),
            "Implementer may not accept its own selection")


def materialize_candidate(pool: Mapping, pool_ref: Mapping, original_rows: Sequence[dict],
                          loaded_records_ref: Mapping, candidate: Mapping,
                          candidate_ref: Mapping, write_records: Callable[[list[dict]], dict]) -> dict:
    require(pool_ref == POOL and candidate_ref == CANDIDATE
            and pool["status"] == "PASS_POOL" and pool["count"] == N,
            "Only exact current paid full-source pool and fixed10 candidate")
    ids = pool["record_ids"]
    require(len(original_rows) == len(ids) == N and len(set(ids)) == N
            and loaded_records_ref == pool["records_json"] and len(pool["source_indices"]) == N
            and ids == [f'{i:06d}:{row.get("id", "")}' for i, row in zip(pool["source_indices"], original_rows)],
            "Qualified original records/IDs/source positions differ")
    positions = candidate["positions"]
    require(candidate["method"] == "coincide" and candidate["fraction"] == "0.10"
            and candidate["status"] == "PENDING_INDEPENDENT_AUTHOR_OPERATOR_OUTPUT_REVIEW_NOT_STUDENT_READY"
            and candidate["selection_seed"] == 1234 and candidate["pool"] == POOL
            and candidate["records_json"] is None and candidate["student_READY"] is False
            and type(candidate["requested_K"]) is int and type(candidate["actual_K"]) is int
            and candidate["requested_K"] == candidate["actual_K"] == K
            and type(positions) is list and all(type(i) is int and 0 <= i < N for i in positions)
            and len(positions) == len(set(positions)) == K and positions == sorted(positions)
            and candidate["record_ids"] == [ids[i] for i in positions]
            and candidate["canonical_record_ids_sha256"] == key(ids)
            and candidate["selected_record_ids_sha256"] == key(candidate["record_ids"]),
            "Exact ordered native candidate, not another method/budget/seed/topup")
    selected = [original_rows[i] for i in positions]
    for row in selected:
        whole_record(row)
    before = key(selected)
    records_ref = descriptor(write_records(deepcopy(selected)))
    expected = canonical(selected) + b"\n"
    require(key(selected) == before and records_ref["bytes"] == len(expected)
            and records_ref["sha256"] == hashlib.sha256(expected).hexdigest(),
            "Sink codec/bytes changed intact original conversations")
    return dict(schema=SCHEMA, status=PENDING, pool=dict(POOL), method="coincide",
        fraction="0.10", selection_seed=1234, count=K, requested_K=K, actual_K=K,
        record_ids=list(candidate["record_ids"]), selected_record_ids_sha256=key(candidate["record_ids"]),
        selected_positions=list(positions), selected_positions_sha256=key(positions),
        original_records_provider=dict(loaded_records_ref), operator_candidate=dict(CANDIDATE),
        records_json=records_ref, adaptation_label=ADAPTATION, method_review=None,
        student_READY=False, original_whole_conversations_unchanged=True, no_target_specific_tuning=True)


def finalize_accepted_selection(materialized: Mapping, review_ref: dict,
                                read_review_bytes: Callable[[dict], bytes],
                                publish: Callable[[dict], dict]) -> dict:
    descriptor(review_ref, maximum=32 << 20)
    raw = read_review_bytes(deepcopy(review_ref))
    require(type(raw) is bytes and len(raw) == review_ref["bytes"]
            and hashlib.sha256(raw).hexdigest() == review_ref["sha256"],
            "Review mapping must be the actual descriptor's checked bytes")
    review = decode_json(raw)
    require(type(review) is dict, "Actual independent review object")
    independent(review)
    ids, positions = materialized["record_ids"], materialized["selected_positions"]
    require(materialized["schema"] == SCHEMA and materialized["status"] == PENDING
            and materialized["pool"] == POOL and materialized["operator_candidate"] == CANDIDATE
            and materialized["method"] == "coincide" and materialized["fraction"] == "0.10"
            and type(materialized["selection_seed"]) is int and materialized["selection_seed"] == 1234
            and all(type(materialized[k]) is int and materialized[k] == K
                    for k in ("count", "requested_K", "actual_K"))
            and type(ids) is list and len(ids) == len(set(ids)) == K
            and all(type(i) is str and i for i in ids)
            and type(positions) is list and len(positions) == len(set(positions)) == K
            and all(type(i) is int and 0 <= i < N for i in positions)
            and positions == sorted(positions)
            and materialized["selected_record_ids_sha256"] == key(ids)
            and materialized["selected_positions_sha256"] == key(positions)
            and materialized["original_whole_conversations_unchanged"] is True
            and materialized["no_target_specific_tuning"] is True
            and materialized["student_READY"] is False and materialized["method_review"] is None,
            "Materialized fixed10 protocol, ID count/order/identity or candidate state differs")
    descriptor(materialized["records_json"])
    require(review["status"] == "PASS_EXACT_SELECTOR_IMPLEMENTATION_AND_OUTPUT"
            and all(review[name] == materialized[name] for name in
                    ("pool", "records_json", "method", "selection_seed", "fraction",
                     "requested_K", "actual_K", "selected_record_ids_sha256", "selected_positions_sha256",
                     "operator_candidate", "original_records_provider"))
            and review["materialized_candidate_identity"] == key(materialized)
            and type(review["eligible_pool_count"]) is int and review["eligible_pool_count"] == N,
            "Detached independent actual review must bind this entire materialization")
    value = dict(schema="native-record-selection-v1", status="PASS_SELECTION",
        pool=dict(POOL), method="coincide", fraction="0.10", selection_seed=1234, count=K,
        record_ids=list(ids), records_json=materialized["records_json"], method_review=dict(review_ref),
        budget=dict(unit="whole_conversation_count", value=K, fraction="0.10", eligible_pool_count=N),
        adaptation_label=ADAPTATION)
    return publish(value)
