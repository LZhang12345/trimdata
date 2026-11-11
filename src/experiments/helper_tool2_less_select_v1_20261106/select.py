"""LESS-approx selection (Tool2 baseline): score = max over target tasks of mean cosine(candidate grad, task grads)
(LESS, Xia et al. 2024, Eq. "max over validation tasks of mean influence"); global top-K, ties by pool order;
materialized as records.json (selected original records in pool order, v3i canonical rendering).
Inputs: 24 candidate shards + 1 target unit of the LESS-approx v2/v2g gradient features (count-sketch 8192, L2-normalized).
Disclosed deviations (see design receipt): 10% random warm-up adapter (one checkpoint), SGD gradients, count-sketch,
candidate pool = 3x Uniform per source + 3x text-only (not the full pool), 2 dev target tasks (GQA-val, ScienceQA-val)."""
import hashlib, importlib.util, json, os, sys
from collections import Counter
from pathlib import Path

HOME = "/home/luyangz/data_selection_multimodal"
FEATURE_ROOT = Path("/data/hf_cache/project_features/DS-MultiModal/helper-tool2-less-approx-v2-20261105")
OUT = Path("/data/hf_cache/project_features/DS-MultiModal/helper-tool2-selections-less-v1-20261106/less_approx-sel1234")
V3I = HOME + "/experiments/helper_method_allocation_train_v3i_20261104/prepare.py"
K = 66076
TARGET_TASKS = ("dev-gqa-val", "dev-scienceqa-val")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def less_scores(candidates, targets):
    """candidates: (n,d) L2-normalized; targets: {task: (m,d) L2-normalized}. Returns (scores, argmax-task index)."""
    import numpy as np
    per_task = np.stack([candidates @ targets[t].mean(axis=0) for t in sorted(targets)], axis=1)
    return per_task.max(axis=1), per_task.argmax(axis=1)


def top_k(pool_positions, scores, k):
    """Indices of the k highest scores; ties broken by smaller pool position (deterministic)."""
    import numpy as np
    order = np.lexsort((np.asarray(pool_positions), -np.asarray(scores, dtype=np.float64)))
    return order[:k]


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as s:
        for block in iter(lambda: s.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_unit(job, dataset):
    import numpy as np
    folder = FEATURE_ROOT / ("job-" + job)
    result = json.loads((folder / "RESULT.json").read_bytes())
    require(result["status"] == "ACTUAL_LESS_APPROX_GRADIENT_FEATURES" and str(result["job_id"]) == job, "Unit RESULT invalid: " + job)
    meta = json.loads((folder / ("FEATURES-" + dataset + ".json")).read_bytes())
    path = folder / ("FEATURES-" + dataset + ".npy")
    require(sha_file(path) == meta["features_sha256"] == result["datasets"][dataset]["features_sha256"], "Feature sha mismatch: " + job)
    feats = np.load(path).astype(np.float32)
    require(feats.shape == (meta["count"], 8192) and len(meta["record_ids"]) == meta["count"], "Feature shape mismatch: " + job)
    return feats, meta["record_ids"], result


def main(jobs_path, v3i_sha, manifest_path):
    import numpy as np
    jobs = json.loads(Path(jobs_path).read_text())
    require(sorted(jobs) == sorted(["less-targets"] + ["less-s%02d" % i for i in range(24)]), "All 25 LESS units required")
    raw = Path(V3I).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == v3i_sha, "Exact v3i supervisor required")
    spec = importlib.util.spec_from_file_location("tool2_v3i_ro", V3I); v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
    targets, sketches = {}, set()
    for task in TARGET_TASKS:
        f, ids, res = load_unit(jobs["less-targets"], task); targets[task] = f
        sketches.add(json.dumps([res["sketch"], res["model"].get("adapter")], sort_keys=True))
    cand, cand_ids = [], []
    for i in range(24):
        f, ids, res = load_unit(jobs["less-s%02d" % i], "records-candidates-s%d" % i)
        cand.append(f); cand_ids += ids
        sketches.add(json.dumps([res["sketch"], res["model"].get("adapter")], sort_keys=True))
    require(len(sketches) == 1, "All units must share one sketch and warm-up adapter")
    require(len(cand_ids) == len(set(cand_ids)), "Candidate shards must be disjoint")
    C = np.concatenate(cand)
    pool, pool_fd = v.read_data(v.POOL, "pool"); groups, groups_fd = v.read_data(v.GROUPS, "group"); records, records_fd = v.read_data(v.RECORDS, "records")
    pool_value = json.loads(pool); ids = pool_value["record_ids"]; position = {ident: n for n, ident in enumerate(ids)}
    require(all(c in position for c in cand_ids), "Candidates must be pool records")
    scores, arg = less_scores(C, targets)
    chosen_idx = top_k([position[c] for c in cand_ids], scores, K)
    require(len(chosen_idx) == K and len(cand_ids) >= K, "Fixed K required")
    threshold = float(scores[chosen_idx[-1]]); ties_in = int(np.sum(scores[chosen_idx] == threshold)); ties_all = int(np.sum(scores == threshold))
    selected = sorted((cand_ids[i] for i in chosen_idx), key=position.get)
    labels = json.loads(groups)["groups"]
    candidate = {"__name__": "_exact_owned_candidate_v3"}
    exec(compile(v.checked(v.CANDIDATE), v.CANDIDATE["path"], "exec", dont_inherit=True), candidate)
    originals = candidate["checked_records"](records, pool_value["records_json"], len(ids))
    keep = set(selected); chosen = []
    for ident, record in zip(ids, originals):
        if ident in keep:
            require(isinstance(record, dict) and str(record["id"]) == ident.split(":", 1)[1]
                    and candidate["valid_native_turns"](record.get("conversations")), "Original record identity required")
            chosen.append(record)
    require(len(chosen) == K, "Selected payload must retain exact K")
    payload = candidate["canonical"](chosen); del originals
    OUT.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = OUT / "records.json"
    require(not path.exists(), "Selection output is create-once")
    tmp = OUT / (".tmp-" + str(os.getpid()))
    with open(tmp, "wb") as s:
        s.write(payload); s.flush(); os.fsync(s.fileno())
    os.chmod(tmp, 0o440); os.replace(tmp, path)
    ids_sha = hashlib.sha256(v.canonical(selected)).hexdigest()
    manifest = {"schema": "tool2-less-approx-selection-v1", "method": "less_approx", "selection_seed": 1234, "K": K,
                "rule": "score=max_task mean_j cos(c, r_j); global top-K; ties by pool position; records in pool order",
                "target_tasks": list(TARGET_TASKS), "feature_jobs": jobs, "v3i_supervisor_sha256": v3i_sha,
                "pool_fd": pool_fd, "groups_fd": groups_fd, "records_fd": records_fd,
                "candidates": len(cand_ids), "threshold": threshold, "boundary_ties_selected": ties_in, "boundary_ties_total": ties_all,
                "score_quantiles": [float(x) for x in np.quantile(scores, [0, .25, .5, .75, .9, .99, 1])],
                "argmax_task_counts_selected": {t: int(np.sum(arg[chosen_idx] == n)) for n, t in enumerate(sorted(targets))},
                "cells": {"less_approx": {"path": str(path), "records_bytes": len(payload), "records_sha256": hashlib.sha256(payload).hexdigest(),
                                          "ordered_ID_sha256": ids_sha, "counts": dict(sorted(Counter(labels[i] for i in selected).items()))}},
                "candidate_counts": dict(sorted(Counter(labels[i] for i in cand_ids).items()))}
    Path(manifest_path).write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(json.dumps({k: manifest[k] for k in ("candidates", "threshold", "boundary_ties_selected", "argmax_task_counts_selected")}), flush=True)
    print(json.dumps(manifest["cells"]["less_approx"]["counts"]), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
