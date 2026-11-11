"""Consume genuine parent publications without selecting, materializing or training."""
from __future__ import annotations

from copy import deepcopy
import re

from . import contract as c


def _whole_records(ids: list, rows: list) -> None:
    c.require(type(ids) is list and type(rows) is list and len(ids) == len(rows) == c.K,
              "Actual66076 intact selected records required")
    c.require(all(type(ident) is str and re.fullmatch(r"\d{6}:.*", ident) is not None
                  for ident in ids) and len(set(ids)) == c.K, "Unique canonical original IDs required")
    positions = [int(ident.split(":", 1)[0]) for ident in ids]
    c.require(positions == sorted(set(positions)), "Original source order must be preserved")
    for ident, row in zip(ids, rows):
        c.require(type(row) is dict and ident.split(":", 1)[1] == str(row.get("id", "")),
                  "Selected original record ID/content join differs")
        turns = row.get("conversations")
        c.require(type(turns) is list and len(turns) >= 2 and len(turns) % 2 == 0,
                  "Whole paired conversations required, not flattened QA")
        c.require(all(type(turn) is dict and turn.get("from") == ("human" if i % 2 == 0 else "gpt")
                      and type(turn.get("value")) is str for i, turn in enumerate(turns)),
                  "Original native turn/target format differs")


def bind_accepted_selection(selection_ref: dict, materialized_ref: dict,
                            actual_output_review_ref: dict, documents: c.Documents) -> dict:
    """Bind the parent's accepted records and detached review, never publish PASS.

    Original-vs-pool custody is the independent parent's responsibility. This
    consumes that explicit acceptance and rechecks selected bytes/IDs/whole turns;
    it does not repeat acquisition or claim a second original-pool verification.
    """
    selected = documents.object(selection_ref)
    c.require(selected.get("schema") == "native-record-selection-v1"
              and selected.get("status") == "PASS_SELECTION"
              and selected.get("pool") == c.POOL and selected.get("method") == "coincide"
              and selected.get("fraction") == "0.10"
              and type(selected.get("selection_seed")) is int
              and selected["selection_seed"] == c.SELECTION_SEED
              and type(selected.get("count")) is int and selected["count"] == c.K
              and selected.get("adaptation_label") == c.ADAPTATION,
              "Genuine fixed10 COINCIDE PASS_SELECTION with explicit adaptation label required")
    c.require(selected.get("budget") == dict(unit="whole_conversation_count", value=c.K,
                  fraction="0.10", eligible_pool_count=c.N), "Frozen whole-conversation budget differs")
    rows = documents.load(selected.get("records_json"))
    ids = selected.get("record_ids")
    _whole_records(ids, rows)

    materialized = documents.object(materialized_ref)
    c.require(materialized.get("schema") == "s7-native-coincide-materialized-selection-candidate-v1"
              and materialized.get("status") ==
                  "CANDIDATE_MATERIALIZED_PENDING_INDEPENDENT_IMPLEMENTATION_AND_OUTPUT_REVIEW"
              and materialized.get("original_whole_conversations_unchanged") is True
              and materialized.get("no_target_specific_tuning") is True
              and materialized.get("student_READY") is False
              and materialized.get("method_review") is None
              and materialized.get("requested_K") == materialized.get("actual_K") == c.K
              and all(materialized.get(name) == selected[name] for name in (
                  "pool", "method", "fraction", "selection_seed", "count", "record_ids",
                  "records_json", "adaptation_label")),
              "Exact parent materialized original-record candidate required")
    actual = documents.object(actual_output_review_ref)
    c.require(actual.get("status") == "ACTUAL_READONLY_OUTPUT_REVIEW_CANDIDATE_NOT_INDEPENDENT_ACCEPTANCE"
              and actual.get("operator_job") == c.OPERATOR_JOB
              and actual.get("function_identity") == c.FUNCTION
              and actual.get("actual_DATA_result") == c.OPERATOR_RESULT,
              "Current10961650/bbde actual-output review required, not old exported geometry")
    outputs = actual.get("outputs")
    c.require(type(outputs) is list and all(type(row) is dict for row in outputs),
              "Actual per-budget readout required")
    matching = [row for row in outputs if row.get("fraction") == "0.10"]
    c.require(len(matching) == 1, "Exactly one actual fixed10 candidate")
    output = matching[0]
    c.require(output.get("candidate") == c.CANDIDATE
              and output.get("requested_K") == output.get("actual_K")
                  == output.get("unique_record_ids") == c.K
              and output.get("underfill") == 0 and output.get("no_topup") is True
              and output.get("selected_record_ids_sha256") == c.key(ids),
              "Accepted membership does not bind the actual fixed10 candidate")

    review_ref = c.descriptor(selected.get("method_review"))
    review = documents.object(review_ref)
    c.independent(review)
    c.require(review.get("status") == "PASS_EXACT_SELECTOR_IMPLEMENTATION_AND_OUTPUT"
              and all(review.get(name) == selected[name] for name in
                  ("pool", "records_json", "method", "selection_seed", "fraction"))
              and review.get("eligible_pool_count") == c.N
              and review.get("requested_K") == review.get("actual_K") == c.K
              and review.get("actual_output_review") == actual_output_review_ref
              and review.get("materialized_candidate") == materialized_ref
              and review.get("original_whole_conversations_unchanged") is True
              and review.get("no_target_specific_tuning") is True,
              "Independent exact implementation/output/original-record acceptance graph required")
    return dict(status="VALIDATED_ACCEPTED_SELECTION_BINDING_NOT_RUNTIME_READY",
        runtime_qualified=False, selection=c.descriptor(selection_ref),
        materialized_candidate=c.descriptor(materialized_ref),
        actual_output_review=c.descriptor(actual_output_review_ref),
        independent_method_review=review_ref, records_json=c.descriptor(selected["records_json"]),
        pool=deepcopy(c.POOL), candidate=deepcopy(c.CANDIDATE), method="coincide",
        count=c.K, selected_record_ids_sha256=c.key(ids), adaptation_label=c.ADAPTATION,
        original_pool_custody="CONSUMED_PARENT_INDEPENDENT_ACCEPTANCE_NOT_REREAD_FULL_POOL")


def bind_training_template(template: dict, selection_ref: dict, materialized_ref: dict,
                           actual_output_review_ref: dict, documents: c.Documents) -> dict:
    c.validate_training_template(template)
    accepted = bind_accepted_selection(selection_ref, materialized_ref, actual_output_review_ref, documents)
    result = deepcopy(template)
    result["bindings"].update({name: accepted[name] for name in (
        "materialized_candidate", "actual_output_review", "independent_method_review")})
    result["bindings"]["accepted_selection"] = accepted["selection"]
    # Match the native scientific duplicate key, not a relabelled control method.
    pool = documents.object(c.POOL)
    c.require(pool.get("status") == "PASS_POOL" and type(pool.get("count")) is int
              and pool["count"] == c.N and pool.get("pool_kind") == "all_source_protected_bridge"
              and pool.get("input_authority") is not None, "Actual full-source authority pool required")
    pool_records = c.descriptor(pool.get("records_json"))
    result["function_identity"] = c.native.key(dict(
        recipe=result["native"]["recipe"], pool_records=pool_records["sha256"],
        selected_records=accepted["records_json"]["sha256"], training_seed=result["training_seed"],
        role="student", qualification_max_steps=None, replicate_site="babel"))
    c.validate_training_template(result)
    return result

