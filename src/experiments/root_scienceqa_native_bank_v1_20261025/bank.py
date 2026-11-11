"""Frozen S6 bank algorithm with exact Root-producer support-row contract.

Only the ordered support projection differs from frozen f627367e. Typed IDs,
full original test membership, original questions, GT and scorer stay unchanged.
"""
import hashlib
import io

from experiments.root_official_input_consumer_v2_20261024 import parsers
from experiments.root_official_protected_inputs_v1_20261024.producer import checked, checked_bytes, source_support
from experiments.root_scienceqa_input_qualification_v3_20261024 import formats as native
from experiments.session6_official_eval_v2_20261024 import common as c
from experiments.session6_official_eval_v2_20261024.journal import semantic_key
from experiments.session6_start_evaluation_v1_20261024.bank import _input_location


def original_typed_ids(raw):
    reader = native.Reader(io.BytesIO(raw), len(raw))
    result = []

    def row(index):
        values = []

        def field(key):
            if key == "id":
                value = native.scalar(reader)
                c.require(type(value) in (str, int), "Original ScienceQA ID type differs")
                values.append(value)
            else:
                reader.skip_value()
        reader.object(field)
        c.require(len(values) == 1, "Exactly one original typed ScienceQA ID required")
        result.append(values[0])
    native.array(reader, row)
    native.end(reader)
    return result


def official_answers(raw, wanted):
    reader = native.Reader(io.BytesIO(raw), len(raw))
    result = {}

    def problem(pid):
        if pid not in wanted:
            reader.skip_value()
            return
        values = []

        def field(key):
            if key == "answer":
                values.append(native.scalar(reader))
            else:
                reader.skip_value()
        reader.object(field)
        c.require(pid not in result and len(values) == 1 and type(values[0]) is int,
                  "Original unique integer ScienceQA answer required")
        result[pid] = values[0]
    reader.object(problem)
    native.end(reader)
    c.require(set(result) == wanted, "Incomplete original official ScienceQA image-test GT")
    return result


def build(specification, *, read_aliases):
    c.require(specification["metric"] == "ScienceQA_image"
              and specification["qa_access"] == "authorized"
              and len(specification["components"]) == 1,
              "Explicit complete ScienceQA image-QA authorization required")
    component = specification["components"][0]
    c.require(component["parser"] == "c121-scienceqa-image-v2"
              and len(component["official_scienceqa_membership_sources"]) == 2,
              "Exact native image-test parser and original official membership/GT required")
    source, support_ref = component["source"], component["support"]
    locations = {name: _input_location(reference, read_aliases) for name, reference in (
        ("source", source), ("support", support_ref), ("media", specification["media"]),
        ("problems", component["official_scienceqa_membership_sources"][0]),
        ("splits", component["official_scienceqa_membership_sources"][1]))}
    raw = checked_bytes(locations["source"])
    support = checked(locations["support"])
    media = checked(locations["media"])
    c.require(support["metric"] == media["metric"] == "ScienceQA_image"
              and support["source"] == source and support["component"] == component["name"]
              and support["edition"] == media["edition"] == specification["edition"]
              and support["split"] == media["split"] == specification["split"],
              "Exact original ScienceQA source/support/media scope differs")
    parsed = parsers.scienceqa_image(raw, {"qa_access": "authorized"})
    c.require([{key: row[key] for key in ("id", "image", "original_row_index")}
               for row in parsed] == support["rows"],
              "Missing, duplicated or reordered original ScienceQA image QA")
    hashes = [dict(id=row["id"], question_sha256=hashlib.sha256(row["question"].encode()).hexdigest())
              for row in parsed]
    c.require(hashes == support["ordered_question_input_hashes"],
              "Original frozen native ScienceQA prompt changed")
    inputs = native.native(raw)
    typed = original_typed_ids(raw)
    c.require(len(inputs) == len(typed), "Native original-row ID projection differs")
    test_ids = native.splits(checked_bytes(locations["splits"]))
    problems_raw = checked_bytes(locations["problems"])
    problems = native.problems(problems_raw, test_ids)
    c.require({row["id"] for row in inputs} == set(test_ids),
              "Full original native test membership differs from official test split")
    image_ids = {pid for pid, problem in problems.items() if problem["image"] is not None}
    c.require({row["id"] for row in parsed} == image_ids and parsed,
              "Full official image-test denominator differs; no smaller proxy")
    answers = official_answers(problems_raw, image_ids)
    roster, ground_truth, image_refs = [], [], set()
    for row in parsed:
        index, pid = row["original_row_index"], row["id"]
        original = inputs[index]
        c.require(original["id"] == pid and str(typed[index]) == pid
                  and row["image"] == pid + "/" + problems[pid]["image"],
                  "Original typed PID/row/native-image/official-image join differs")
        image_ref = component["name"] + ":" + row["image"]
        c.require(image_ref in media["images"], "Exact declared ScienceQA encoded/RGB identity absent")
        image_refs.add(image_ref)
        choices = problems[pid]["choices"]
        c.require(all(isinstance(choice, str) for choice in choices)
                  and 0 <= answers[pid] < len(choices) <= 5, "Original ScienceQA choices/GT index differs")
        key = semantic_key("ScienceQA_image", source["sha256"], index, typed[index])
        roster.append(dict(semantic_key=key, bank="ScienceQA_image", source_sha256=source["sha256"],
            original_row_index=index, native_id=typed[index], image_reference=image_ref,
            native_input=dict(id=typed[index], image=row["image"],
                conversations=[{"from": "human", "value": original["human"]}])))
        ground_truth.append(dict(semantic_key=key, official_pid=pid, native_id=typed[index],
                                 choices=choices, answer=answers[pid]))
    c.require(set(media["images"]) == image_refs, "Extra or missing full ScienceQA media identity")
    return dict(roster=dict(schema="session6-authorized-full-scienceqa-image-roster-v1",
        bank="ScienceQA_image", edition=specification["edition"], split=specification["split"],
        source=source, support=support_ref, expected_questions=len(roster), records=roster,
        inference_answers_excluded=True, full_declared_scope=True, runtime_inputs=locations),
        GT=dict(status="COMPLETE_ORIGINAL_SCIENCEQA_IMAGE_GT_JOIN_NOT_PREDICTIONS",
                questions=len(ground_truth), records=ground_truth,
                original_problems=component["official_scienceqa_membership_sources"][0]),
        policy=dict(native_runner="llava.eval.model_vqa_science", conv_mode="vicuna_v1",
                    max_new_tokens=1024, single_pred_prompt=True,
                    native_num_beams_keyword=False, native_top_p_keyword=False,
                    full_actual_inherited_GenerationConfig_required=True),
        physical_media_or_model_generation_qualified=False)
