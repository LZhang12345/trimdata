"""Pinned official algorithms with explicit coverage and malformed-answer accounting."""
import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import re
import tempfile
import types

from . import common as c


def official(name):
    path = c.scorer_source(name)
    spec = importlib.util.spec_from_file_location(
        "s6_pinned_" + path.stem, path)
    c.require(spec is not None and spec.loader is not None, "Official scorer cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def textvqa(records):
    c.require(records, "A complete declared question bank is required")
    evaluator = official("m4c_evaluator.py").TextVQAAccuracyEvaluator()
    scores = []
    for row in records:
        c.require(isinstance(row["prediction"], str)
                  and len(row["answers"]) == 10
                  and all(isinstance(x, str) for x in row["answers"]),
                  "Exact raw prediction and ten original human answers required")
        normalized = evaluator.answer_processor(row["prediction"])
        credit = evaluator._compute_answer_scores(row["answers"]).get(normalized, 0.0)
        scores.append(credit)
    return dict(metric="TextVQA_native_v051_accuracy", score=100 * sum(scores) / len(scores),
                per_question_credit=scores, questions=len(scores),
                empty_predictions=sum(not row["prediction"].strip() for row in records),
                scorer=c.desc(c.scorer_source("m4c_evaluator.py")),
                official_algorithm=True)


def scienceqa_answer(text):
    options = ["A", "B", "C", "D", "E"]
    if text in options:
        return text
    if len(text) >= 3 and text[0] in options and text[1:3] == ". ":
        return text[0]
    matches = re.compile(r"The answer is ([A-Z]).").findall(text)
    return matches[0] if len(matches) == 1 else "FAILED"


def scienceqa(records):
    c.require(records, "A complete declared question bank is required")
    native = official("eval_science_qa.py")
    credits, malformed = [], 0
    for row in records:
        c.require(isinstance(row["prediction"], str) and type(row["answer"]) is int
                  and 0 <= row["answer"] < len(row["choices"]) <= 5,
                  "Original ScienceQA choice/answer identity differs")
        parsed = scienceqa_answer(row["prediction"])
        index = native.get_pred_idx(parsed, row["choices"], ["A", "B", "C", "D", "E"])
        malformed += int(index == -1)
        credits.append(int(index == row["answer"]))
    return dict(metric="ScienceQA_native_image_accuracy",
                score=100 * sum(credits) / len(credits), per_question_credit=credits,
                questions=len(credits), malformed_predictions=malformed,
                scorer=c.desc(c.scorer_source("eval_science_qa.py")), official_algorithm=True)


def pope(records):
    c.require(records and all(isinstance(r["prediction"], str) and r["label"] in ("yes", "no")
                             for r in records), "Exact complete POPE prediction/label roster required")
    native = official("eval_pope.py")
    answers = [dict(text=r["prediction"]) for r in records]
    with tempfile.TemporaryDirectory(prefix="s6-official-pope-") as directory:
        labels = Path(directory) / "labels.jsonl"
        labels.write_text("".join(json.dumps(dict(label=r["label"])) + "\n" for r in records))
        output = io.StringIO()
        with labels.open() as label_stream, redirect_stdout(output):
            def scoped_open(path, mode="r"):
                c.require(str(path) == str(labels) and mode == "r",
                          "Official POPE requested an undeclared input")
                return label_stream
            runner = types.FunctionType(native.eval_pope.__code__,
                                        dict(vars(native), open=scoped_open))
            runner(answers, str(labels))
    values = {}
    for line in output.getvalue().splitlines():
        for label in ("Accuracy", "Precision", "Recall", "F1 score", "Yes ratio"):
            if line.startswith(label + ": "):
                values[label] = float(line.split(": ", 1)[1])
    c.require(set(values) == {"Accuracy", "Precision", "Recall", "F1 score", "Yes ratio"},
              "Pinned POPE output shape changed")
    return dict(metric="POPE_native_accuracy", score=100 * values["Accuracy"],
                official_values=values, questions=len(records),
                per_question_credit=[int(a["text"] == r["label"]) for a, r in zip(answers, records)],
                empty_predictions=sum(not r["prediction"].strip() for r in records),
                official_stdout=output.getvalue(),
                scorer=c.desc(c.scorer_source("eval_pope.py")), official_algorithm=True)


def qualify():
    evaluator = official("m4c_evaluator.py").TextVQAAccuracyEvaluator()
    evidence = []
    for count in range(11):
        row = dict(prediction="The TWO.", answers=["2"] * count + ["other"] * (10 - count))
        expected = (count * min((count - 1) / 3, 1) +
                    (10 - count) * min(count / 3, 1)) / 10 if count else 0.0
        actual = textvqa([row])["per_question_credit"][0]
        direct = evaluator.eval_pred_list([dict(pred_answer=row["prediction"], gt_answers=row["answers"])])
        c.require(abs(actual - expected) < 1e-12 and actual == direct,
                  "Official VQA leave-one-out credit differs")
        evidence.append(dict(count=count, actual=actual, expected=expected))
    originals = [dict(prediction="yes", label="yes"), dict(prediction="No.", label="no"),
                 dict(prediction="yes", label="no"), dict(prediction="not visible", label="yes")]
    before = copy.deepcopy(originals)
    result = pope(originals)
    c.require(originals == before and result["official_values"]["Accuracy"] == .5,
              "POPE normalization, metrics or raw preservation changed")
    return dict(status="PASS_PINNED_SCORER_FIXTURES_ONLY_NOT_FULL_BANK_EVALUATION",
                TextVQA_leave_one_out=evidence, POPE=result,
                ScienceQA_invalid_choice=official("eval_science_qa.py").get_pred_idx(
                    "FAILED", ["a", "b"], ["A", "B", "C", "D", "E"]),
                model_or_dataset_payloads_loaded=False)
