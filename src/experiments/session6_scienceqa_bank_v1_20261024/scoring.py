from experiments.session6_official_eval_v2_20261024 import common as c
from experiments.session6_official_eval_v2_20261024.scoring import scienceqa


def score_complete(roster, ground_truth, predictions):
    rows, labels = roster["records"], ground_truth["records"]
    c.require(roster["bank"] == "ScienceQA_image" and roster["full_declared_scope"] is True
              and roster["inference_answers_excluded"] is True and rows
              and len(rows) == len(labels) == roster["expected_questions"] == ground_truth["questions"],
              "Complete original ScienceQA image-QA/GT denominator required")
    expected = [row["semantic_key"] for row in rows]
    c.require(len(expected) == len(set(expected)), "Duplicate original ScienceQA semantic QA")
    c.require(len(predictions) == len(rows)
              and len({value["semantic_key"] for value in predictions}) == len(rows)
              and {value["semantic_key"] for value in predictions} == set(expected),
              "Incomplete, duplicate or foreign ScienceQA predictions; never zero-fill missing QA")
    by_key = {value["semantic_key"]: value["raw"] for value in predictions}
    scoring_rows = []
    for row, label in zip(rows, labels):
        raw = by_key[row["semantic_key"]]
        c.require(label["semantic_key"] == row["semantic_key"]
                  and type(label["native_id"]) is type(row["native_id"])
                  and label["native_id"] == row["native_id"]
                  and str(row["native_id"]) == label["official_pid"]
                  and type(raw["question_id"]) is type(row["native_id"])
                  and raw["question_id"] == row["native_id"] and isinstance(raw["text"], str),
                  "Individual native-ID type/value or original official GT PID join differs")
        scoring_rows.append(dict(prediction=raw["text"], choices=label["choices"], answer=label["answer"]))
    result = scienceqa(scoring_rows)
    result.update(semantic_order=expected,
        original_raw_predictions=[by_key[key]["text"] for key in expected],
        empty_predictions=sum(not value["prediction"].strip() for value in scoring_rows),
        bank=roster["bank"], edition=roster["edition"], split=roster["split"],
        source=roster["source"], complete_original_QA_GT_prediction_join=True,
        actual_model_runtime_and_generation_qualification_required_by_caller=True,
        scientific_round_complete=False)
    return result
