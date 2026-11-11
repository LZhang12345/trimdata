"""Complete authorized native JSONL banks, preserving raw question IDs/prompts."""
import hashlib
import json

from experiments.session6_official_eval_v2_20261024 import common as c
from experiments.session6_official_eval_v2_20261024.journal import semantic_key

JSONL_BANKS = {
    "VQAv2", "VizWiz", "TextVQA", "POPE", "SEED_image",
    "MME_perception", "MME_cognition", "LLaVA_Bench_in_the_Wild", "MM_Vet",
}


def _input_location(reference, read_aliases):
    if read_aliases is None:
        return reference
    c.require(isinstance(read_aliases, dict) and reference["path"] in read_aliases,
              "Explicit runtime alias required for every original input")
    binding = read_aliases[reference["path"]]
    c.require(isinstance(binding, dict) and binding.get("original") == reference
              and isinstance(binding.get("local"), dict),
              "Runtime alias differs from the exact original descriptor")
    local = binding["local"]
    c.require(set(local) == {"path", "bytes", "sha256"}
              and local["bytes"] == reference["bytes"]
              and local["sha256"] == reference["sha256"],
              "Runtime alias byte identity differs from the original")
    return dict(local)


def build_jsonl(metric, specification, source_reference, support_reference, *, read_aliases=None):
    from experiments.root_official_input_consumer_v2_20261024 import parsers
    from experiments.root_official_protected_inputs_v1_20261024.producer import checked, checked_bytes, source_support
    c.require(metric in JSONL_BANKS and specification["metric"] == metric
              and specification["qa_access"] == "authorized",
              "This adapter cannot upgrade image-only or unsupported QA access")
    c.require(len(specification["components"]) == 1, "One exact original native source required")
    component = specification["components"][0]
    c.require(component["source"] == source_reference
              and component["support"] == support_reference,
              "Foreign original question/support descriptor")
    source_location = _input_location(source_reference, read_aliases)
    support_location = _input_location(support_reference, read_aliases)
    raw = checked_bytes(source_location)
    support = checked(support_location)
    c.require(support["metric"] == metric and support["source"] == source_reference
              and support["edition"] == specification["edition"]
              and support["split"] == specification["split"]
              and support["component"] == component["name"], "Complete original bank scope differs")
    source_rows = [json.loads(line) for line in raw.splitlines()]
    source_row_policy = component["parser"] == "c121-author-source-row-jsonl-v2"
    parsed = parsers.author_jsonl(raw, dict(
        qa_access="authorized", row_identity="source-row" if source_row_policy else "native"))
    expected_ids = [row["id"] for row in support["rows"]]
    c.require(len(expected_ids) == len(set(expected_ids)), "Ambiguous frozen semantic membership")
    allowed = set(expected_ids)
    selected = [row for row in parsed if row["id"] in allowed]
    c.require(source_support(selected, component["name"]) == support["rows"]
              and [row["id"] for row in selected] == expected_ids,
              "Incomplete or reordered full official bank; no smaller proxy allowed")
    hashes = [dict(id=row["id"], question_sha256=hashlib.sha256(row["question"].encode()).hexdigest())
              for row in selected]
    c.require(hashes == support["ordered_question_input_hashes"], "Frozen prompt bytes differ")
    roster = []
    for row in selected:
        index = row["original_row_index"]
        original = source_rows[index]
        c.require(original["text"] == row["question"] and original["image"] == row["image"],
                  "Native prompt/image changed by projection")
        native_id = original["question_id"]
        key = semantic_key(metric, source_reference["sha256"], index, native_id)
        roster.append(dict(
            semantic_key=key, bank=metric, source_sha256=source_reference["sha256"],
            original_row_index=index, native_id=native_id,
            native_input={k: original[k] for k in ("question_id", "image", "text")},
            frozen_input_id=row["id"], image_reference=component["name"] + ":" + row["image"],
        ))
    return dict(
        schema="session6-complete-native-question-roster-v1",
        status="COMPLETE_FROZEN_QA_ROSTER_NOT_MODEL_OR_SCORER_QUALIFICATION",
        bank=metric, edition=specification["edition"], split=specification["split"],
        source=source_reference, support=support_reference, records=roster,
        runtime_inputs=dict(source=source_location, support=support_location),
        expected_questions=len(roster), inference_answers_excluded=True,
        full_declared_scope=True,
    )
