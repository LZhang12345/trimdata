import hashlib
import itertools
import json
from pathlib import Path

from experiments.root_native_campaign_v7_20261024 import common as c

JOBS = ("10954463", "10954865", "10954867", "10954869", "10954871", "10954873", "10954875")
QUESTIONS = 12578
REFERENCE_URL = (
    "https://raw.githubusercontent.com/haotian-liu/LLaVA/"
    "c121f0432da27facab705978f83c4ada465e46fd/scripts/convert_gqa_for_eval.py")
REFERENCE_SHA256 = "016cea68c91cbc141c276e9e7568420e2f5db5aba90c4f594701663c72929932"
CONTROL = Path(
    "/home/luyangz/data_selection_multimodal/collaboration/"
    "root-session1-priority-continuation-20261027")


def native_prediction(text):
    c.require(type(text) is str, "GQA prediction must be the preserved native string")
    return text.rstrip(".").lower()


def compare_records(records, targets, answers):
    c.require(bool(records) and len({row["semantic_key"] for row in records}) == len(records)
              and set(targets) == {row["semantic_key"] for row in records},
              "Exact unique ordered semantic QA membership")
    counts = dict(questions=0, raw_exact_hits=0, native_converted_exact_hits=0,
                  case_changed=0, terminal_period_removed=0, empty_raw=0, empty_converted=0)
    digest = hashlib.sha256()
    for row, answer in itertools.zip_longest(records, answers):
        c.require(row is not None and answer is not None, "Missing or extra paid GQA prediction")
        expected = row["native_id"]
        c.require(type(expected) in (str, int) and type(answer["question_id"]) is type(expected)
                  and answer["question_id"] == expected, "Exact ordered typed original GQA ID")
        target = targets[row["semantic_key"]]
        c.require(type(target) is str and bool(target), "Exact qualified nonempty GQA target")
        text = answer["text"]
        converted = native_prediction(text)
        raw_hit, native_hit = text == target, converted == target
        counts["questions"] += 1
        counts["raw_exact_hits"] += raw_hit
        counts["native_converted_exact_hits"] += native_hit
        counts["case_changed"] += text.lower() != text
        counts["terminal_period_removed"] += text.rstrip(".") != text
        counts["empty_raw"] += not text
        counts["empty_converted"] += not converted
        digest.update(c.canonical(dict(semantic_key=row["semantic_key"],
                                      raw_exact=raw_hit, native_converted_exact=native_hit)))
    return dict(counts, raw_exact_score=100.0 * counts["raw_exact_hits"] / len(records),
                candidate_author_native_GQA_score=100.0 * counts["native_converted_exact_hits"] / len(records),
                paired_credit_digest=digest.hexdigest(),
                paired_credit_arrays_or_raw_predictions_or_GT_exported=False)


def check_path(path):
    path = Path(path)
    c.require(path.is_absolute() and ".." not in path.parts
              and not any(p.is_symlink() for p in (path, *path.parents)),
              "Exact nonsymlinked existing paid metadata/JSONL path")
    return path


def model_diagnosis(model, records, targets):
    bank = model["banks"]["GQA"]
    prediction = bank["raw_export"]
    source = check_path(prediction["path"])
    c.require(c.desc(source) == prediction and bank["questions"] == QUESTIONS,
              "Exact previously audited full paid GQA export")
    original = c.checked(bank["score_record"])
    c.require(original["predictions"] == prediction and original["complete_questions"] == QUESTIONS
              and c.key(original["function"]) == bank["function_sha256"]
              and original["score"]["normalization"] == "none"
              and original["score"]["questions"] == QUESTIONS,
              "Preserve the actual original raw-comparison score and model function")
    with source.open() as stream:
        counts = compare_records(records, targets, (json.loads(line) for line in stream))
    c.require(counts["questions"] == QUESTIONS
              and counts["raw_exact_score"] == original["score"]["score"]
              == model["bank_metrics"]["GQA"] and c.desc(source) == prediction,
              "Original score must reproduce and paid predictions must remain unchanged")
    return dict(student_id=model["student_id"], evaluation_job_id=model["evaluation_job_id"],
                training_seed=model["training_seed"], training_configuration=model["training_configuration"],
                actual_original_score=bank["score_record"], paid_predictions=prediction,
                original_function_sha256=bank["function_sha256"], **counts,
                format_deviation_has_observed_score_effect=
                counts["raw_exact_hits"] != counts["native_converted_exact_hits"])


def run(configuration, actual, registration):
    from experiments.root_native_trained_evaluation_v3_20261026 import banks
    cfg = c.checked(configuration)
    c.require(cfg["schema"] == "s1-paid-gqa-native-conversion-diagnosis-v1"
              and cfg["owner"] == "Session1-author-delegated"
              and cfg["new_NN"] == 0 and cfg["checkpoint_writes"] is False
              and cfg["raw_QA_GT_or_credit_arrays_exported"] is False
              and cfg["native_converter_URL"] == REFERENCE_URL
              and cfg["native_converter_sha256"] == REFERENCE_SHA256,
              "Author-native conversion only; no normalization search or model replay")
    table = c.checked(cfg["audited_score_table"])
    c.require(table["status"] ==
              "ACTUAL_SEVEN_VALIDATED_SCORE_AGGREGATES_EXPORTED_REQUIRE_INDEPENDENT_ACCEPTANCE"
              and table["paid_QAs_revalidated"] == 199535
              and tuple(row["evaluation_job_id"] for row in table["models"]) == JOBS,
              "Exact completed and already-audited seven-model readout")
    results = []
    provider = None
    for model in table["models"]:
        evaluation = c.checked(model["configuration"])
        spec = evaluation["bank_inputs"]["GQA"]
        qualified = c.checked(spec["qualified_result"])
        c.require(qualified["status"] == "QUALIFIED_PUBLIC_GQA12578_QA_GT_RGB_AND_EXACT_MATCH_SCORE_INPUTS"
                  and qualified["questions"] == QUESTIONS and qualified["selector_reference_access"] is False,
                  "Exact existing evaluation-only full GQA provider")
        c.require(provider is None or qualified["roster"] == provider,
                  "All paired models must use the same qualified GQA target roster")
        provider = qualified["roster"]
        roster, targets = banks.gqa_roster(c.checked(provider)["records"])
        c.require(len(roster["records"]) == QUESTIONS, "Full protected GQA scope")
        results.append(model_diagnosis(model, roster["records"], targets))
    target = check_path(cfg["output"])
    c.require(CONTROL in target.parents, "Only the declared small HOME diagnosis summary")
    return c.seal(target, dict(
        status="ACTUAL_GQA_AUTHOR_CONVERSION_DIAGNOSIS_REQUIRES_INDEPENDENT_ACCEPTANCE",
        configuration=configuration, source_audited_score_table=cfg["audited_score_table"],
        actual=actual, registration=registration, models=results, questions_per_model=QUESTIONS,
        existing_qualified_target_roster=provider,
        native_converter_URL=REFERENCE_URL, native_converter_sha256=REFERENCE_SHA256,
        exact_native_prediction_conversion='text.rstrip(".").lower()',
        new_NN=0, paid_answers_regenerated=0, image_or_tensor_reads=0,
        checkpoint_writes=False, original_scores_or_paid_answers_modified=False,
        raw_QA_GT_or_credit_arrays_exported=False, no_normalization_variant_search=True,
        no_new_scientific_gain_or_model_ceiling_claim=True,
        corrected_full_fourbank_primary_readout_requires_independent_protocol_acceptance=True))
