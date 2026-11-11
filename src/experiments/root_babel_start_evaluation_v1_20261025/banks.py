"""Exact frozen native rows and official target joins; targets never enter generation."""
import json
from collections import Counter
from pathlib import PurePosixPath

from experiments.session6_official_eval_v2_20261024 import common as c
from experiments.session6_start_evaluation_v1_20261024.bank import build_jsonl
from experiments.root_official_protected_inputs_v1_20261024.producer import checked_bytes

COUNTS = {"POPE": (8910, 500), "VQAv2": (107394, 36807), "VizWiz": (8000, 8000)}


def typed(value):
    c.require(type(value) in (str, int), "Original scalar native ID required")
    return type(value).__name__, value


def pope_join(native, originals):
    """The official c121 driver filters by category, then scores in file order."""
    c.require(set(originals) == {"adversarial", "popular", "random"}, "Exact three POPE categories required")
    groups = {k: [] for k in originals}
    for i, row in enumerate(native):
        c.require(set(row) >= {"question_id", "image", "text", "category"}
                  and row["category"] in groups, "Original native category missing")
        groups[row["category"]].append((i, row))
    result = {}
    for category, official in originals.items():
        rows = groups[category]
        expected = 2910 if category == "random" else 3000
        c.require(len(rows) == len(official) == expected, "Actual POPE source/category denominator differs")
        c.require(len({typed(x["question_id"]) for x in official}) == expected,
                  "Original POPE IDs duplicate")
        for (index, row), target in zip(rows, official):
            # Native conversion appends this literal instruction; no case/fuzzy normalization.
            c.require(row["image"] == target["image"] and
                      row["text"] == target["text"] + "\nAnswer the question using a single word or phrase.",
                      "POPE ordered image/question transformation differs; no guessed label join")
            c.require(target["label"] in ("yes", "no"), "Original binary POPE label required")
            result[index] = dict(category=category, label=target["label"],
                                 original_id=list(typed(target["question_id"])))
    c.require(len(result) == len(native), "Incomplete POPE target join")
    return result


def load(cfg):
    scope = c.checked(cfg["frozen_scope"])["banks"][cfg["bank"]]
    component = scope["components"][0]
    c.require(scope["qa_access"] == "authorized" and scope["metric"] == cfg["bank"],
              "No image-only permission expansion")
    c.require(cfg["media"] == scope["media"] and cfg["native_source"] == component["source"],
              "Exact configured frozen source/media descriptors required")
    roster = build_jsonl(cfg["bank"], scope, component["source"], component["support"])
    media = c.checked(scope["media"])
    provider = c.checked(component["provider_complete"])
    c.require(provider["status"] == "COMPLETE_ACTUAL_ORDERED_SOURCE_SUPPORT_AND_RGB" and
              provider["frozen_support"] == component["support"] and provider["media"] == scope["media"],
              "Genuine actual input provider differs")
    c.require((len(roster["records"]), len(media["images"])) == COUNTS[cfg["bank"]] and
              {r["image_reference"] for r in roster["records"]} == set(media["images"]),
              "Full native bank/media membership differs")
    targets = None
    if cfg["bank"] == "POPE":
        native = [json.loads(line) for line in checked_bytes(cfg["native_source"]).splitlines()]
        originals = {k: [json.loads(line) for line in checked_bytes(v).splitlines()]
                     for k, v in cfg["official_targets"].items()}
        targets = pope_join(native, originals)
    return roster, media, targets
