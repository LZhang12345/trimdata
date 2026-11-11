"""Frozen official scoring with the already declared native GQA conversion."""
from experiments.root_native_trained_evaluation_v3_20261026 import banks
from experiments.copilot7_coincide_native_consumer_20261028.evaluation import gqa_native_prediction


def score(bank, roster, journal, targets):
    if bank != "GQA":
        return banks.score(bank, roster, journal, targets)
    rows = roster["records"]
    entries = banks.original.entries(journal, rows)
    banks.u.require(len(entries) == len(rows) == 12578 and not journal.missing(),
                    "Only full12578 original GQA bank may score")
    correct = sum(gqa_native_prediction(entry["raw"]["text"]) == targets[row["semantic_key"]]
                  for row, entry in zip(rows, entries))
    return dict(metric="Official balanced test-dev exact answer match", questions=len(rows),
        prediction_conversion='text.rstrip(".").lower()', target_normalization="none",
        correct=correct, score=100.0 * correct / len(rows), official_algorithm=True,
        native_conversion_source=banks.original.c.desc(
            banks.original.c.ROOT / "experiments/copilot7_coincide_native_consumer_20261028/evaluation.py"),
        no_extra_target_or_whitespace_punctuation_normalization=True)
