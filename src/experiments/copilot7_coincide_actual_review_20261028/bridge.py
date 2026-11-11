"""Pure intact-record bridge; actual reads/writes belong to Root's owned actor.

Materialization is a candidate, not an independent review. Final publication
requires a detached genuine independent implementation/output acceptance.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Callable, Mapping, Sequence

from experiments.meta_probe_v1_20261028.contracts import canonical, require

ADAPTATION = (
    "COINCIDE author TinyLLaVA/SigLIP features + native FAISS10000/50/spherical "
    "and FP64 MMD; full-source eligible LLaVA665K selection transferred to "
    "Stage1 LLaVA1.5-7B native LoRA/projector student, not paper-identical student"
)


def whole_record(row: Mapping) -> None:
    turns = row.get("conversations")
    require(isinstance(turns, list) and len(turns) >= 2 and len(turns) % 2 == 0,
            "Original complete paired conversation required")
    require(all(t.get("from") == ("human" if i % 2 == 0 else "gpt")
                and type(t.get("value")) is str for i, t in enumerate(turns)),
            "No changed/malformed native conversation targets")


def materialize_candidate(pool: Mapping, pool_ref: Mapping, original_rows: Sequence[dict],
                          loaded_records_ref: Mapping, candidate: Mapping,
                          write_records: Callable[[list[dict]], dict]) -> dict:
    ids = pool["record_ids"]
    require(pool["status"] == "PASS_POOL" and len(original_rows) == len(ids) == pool["count"]
            and len(set(ids)) == len(ids) and loaded_records_ref == pool["records_json"],
            "Exact qualified full-source original-record provider required")
    require(len(pool["source_indices"]) == len(ids)
            and ids == [f'{i:06d}:{row.get("id", "")}' for i, row in zip(pool["source_indices"], original_rows)],
            "Actual records do not match canonical original IDs/source positions")
    positions = candidate["positions"]
    require(candidate["method"] == "coincide" and candidate["fraction"] == "0.10"
            and candidate["status"] == "PENDING_INDEPENDENT_AUTHOR_OPERATOR_OUTPUT_REVIEW_NOT_STUDENT_READY"
            and candidate["selection_seed"] == 1234 and candidate["pool"] == pool_ref
            and candidate["records_json"] is None and candidate["student_READY"] is False
            and candidate["requested_K"] == candidate["actual_K"] == len(ids) // 10
            and len(positions) == len(set(positions)) == candidate["actual_K"]
            and positions == sorted(positions)
            and all(type(i) is int and 0 <= i < len(ids) for i in positions)
            and candidate["record_ids"] == [ids[i] for i in positions],
            "Exact sorted existing10percent native candidate, not a rescore/topup")
    selected = [original_rows[i] for i in positions]
    for row in selected:
        whole_record(row)
    before = hashlib.sha256(canonical(selected)).hexdigest()
    # Give the owned sink detached records, not permission to mutate the pool.
    records_ref = write_records(deepcopy(selected))
    require(hashlib.sha256(canonical(selected)).hexdigest() == before,
            "Original supervised data mutated during bridge")
    expected = canonical(selected) + b"\n"
    require(set(records_ref) == {"path", "bytes", "sha256"}
            and records_ref["bytes"] == len(expected)
            and records_ref["sha256"] == hashlib.sha256(expected).hexdigest(),
            "Owned sink must use declared native canonical-plus-newline bytes; actual custody still independently reviewed")
    return dict(schema="s7-native-coincide-materialized-selection-candidate-v1",
        status="CANDIDATE_MATERIALIZED_PENDING_INDEPENDENT_IMPLEMENTATION_AND_OUTPUT_REVIEW",
        pool=dict(pool_ref), method="coincide", fraction="0.10", selection_seed=1234,
        count=len(selected), requested_K=candidate["requested_K"], actual_K=len(selected),
        record_ids=list(candidate["record_ids"]), records_json=records_ref,
        adaptation_label=ADAPTATION, method_review=None, student_READY=False,
        original_whole_conversations_unchanged=True, no_target_specific_tuning=True)


def finalize_accepted_selection(materialized: Mapping, review: Mapping, review_ref: Mapping,
                                publish: Callable[[dict], dict]) -> dict:
    require(review["status"] == "PASS_EXACT_SELECTOR_IMPLEMENTATION_AND_OUTPUT"
            and review["independent"] is True
            and type(review["reviewer"]) is str and bool(review["reviewer"])
            and review["reviewer"].casefold() not in ("session7", "s7", "copilot7")
            and all(review[name] == materialized[name] for name in
                    ("pool", "records_json", "method", "selection_seed", "fraction"))
            and review["requested_K"] == materialized["requested_K"]
            and review["actual_K"] == materialized["actual_K"],
            "Genuine detached exact independent actual-output acceptance required")
    require(set(review_ref) == {"path", "bytes", "sha256"} and type(review_ref["bytes"]) is int
            and review_ref["bytes"] > 0 and type(review_ref["sha256"]) is str
            and len(review_ref["sha256"]) == 64, "Actual detached review descriptor required")
    count = materialized["count"]
    require(count == materialized["requested_K"] == materialized["actual_K"]
            == review["eligible_pool_count"] // 10, "Exact fixed10percent budget required")
    value = dict(schema="native-record-selection-v1", status="PASS_SELECTION",
        pool=materialized["pool"], method="coincide", fraction="0.10",
        selection_seed=1234, count=count, record_ids=materialized["record_ids"],
        records_json=materialized["records_json"], method_review=dict(review_ref),
        budget=dict(unit="whole_conversation_count", value=count, fraction="0.10",
                    eligible_pool_count=review["eligible_pool_count"]),
        adaptation_label=ADAPTATION)
    return publish(value)
