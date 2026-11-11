"""(1) Materialize COINCIDE native-operator v3 selections at 5% and 20% into training records (read-only inputs).
(2) Read out four-bank scores of finished Tool3 evaluations. Writes only under its own output paths."""
import glob, hashlib, json, os, sys
from pathlib import Path

OP = Path("/data/hf_cache/project_features/DS-MultiModal/coincide_native_operator_20261027/v3")
RECORDS = ("/data/user_data/luyangz/native_campaign_20261023/root-wholepool-authority-v1/authority/records.json",
           760064389, "1c53e08b940af27483201efc4c2fa27e3f5c4f7862484e078736c718673c1d36")
OUT = Path("/data/hf_cache/project_features/DS-MultiModal/tool3-baselines-20261104")
EVAL = Path("/data/hf_cache/project_artifacts/DS-MultiModal/tool3-quality-eval-v1-20261102")


def shape(value, depth=0):
    if isinstance(value, dict):
        return {k: shape(v, depth + 1) for k, v in list(value.items())[:25]} if depth < 2 else "dict(%d)" % len(value)
    if isinstance(value, list):
        return ["list(%d)" % len(value), shape(value[0], depth + 1) if value else None]
    return type(value).__name__


def selected_rows(candidate, n_pool, ids):
    for key in ("rows", "selected_rows", "pool_rows", "indices", "selected_indices"):
        if isinstance(candidate, dict) and key in candidate and all(type(x) is int for x in candidate[key]):
            return candidate[key], key
    for key in ("record_ids", "selected_record_ids", "ids", "selected_ids", "selected"):
        if isinstance(candidate, dict) and key in candidate and all(isinstance(x, str) for x in candidate[key]):
            # POOL.json record_ids are aligned with records.json positions (native "<source row>:<id>" form).
            index = {rid: i for i, rid in enumerate(ids)}
            missing = [x for x in candidate[key] if x not in index]
            if missing:
                raise ValueError("Candidate ids absent from POOL record_ids: " + ", ".join(missing[:3]))
            return [index[x] for x in candidate[key]], key
    if isinstance(candidate, list) and all(type(x) is int for x in candidate):
        return candidate, "list"
    return None, None


report = {"job": os.environ.get("SLURM_JOB_ID"), "coincide": {}, "scores": {}}
raw = Path(RECORDS[0]).read_bytes()
assert len(raw) == RECORDS[1] and hashlib.sha256(raw).hexdigest() == RECORDS[2]
records = json.loads(raw)
POOL = ("/data/user_data/luyangz/native_campaign_20261023/root-wholepool-authority-v1/authority/POOL.json",
        18754787, "c76040a17ca0dc1e99fea14514618edade3171a9b05f6ccc3e4751a5939e4964")
pool_raw = Path(POOL[0]).read_bytes()
assert len(pool_raw) == POOL[1] and hashlib.sha256(pool_raw).hexdigest() == POOL[2]
ids = json.loads(pool_raw)["record_ids"]
assert len(ids) == len(records) == len(set(ids))
assert all(rid.split(":", 1)[1] == str(r.get("id")) for rid, r in zip(ids[:1000], records[:1000]))
for budget in ("0.05", "0.20"):
    entry = report["coincide"][budget] = {}
    path = OP / ("budget-" + budget) / "CANDIDATE.json"
    candidate = json.loads(path.read_bytes())
    entry["candidate_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    entry["shape"] = shape(candidate)
    complete = OP / ("budget-" + budget) / "COMPLETE.json"
    entry["complete"] = json.loads(complete.read_bytes()) if complete.exists() else None
    rows, key = selected_rows(candidate, len(records), ids)
    if rows is None:
        entry["status"] = "UNRECOGNIZED_STRUCTURE"
        continue
    if len(set(rows)) != len(rows):
        entry["status"] = "DUPLICATE_ROWS"
        continue
    body = json.dumps([records[r] for r in rows], ensure_ascii=False).encode() + b"\n"
    target = OUT / ("coincide-b" + budget[2:]) / "records.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp"); tmp.write_bytes(body); os.replace(tmp, target)
    entry.update(status="MATERIALIZED", key=key, K=len(rows),
                 records={"path": str(target), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})
for result in sorted(EVAL.glob("*/RESULT.json")):
    value = json.loads(result.read_bytes())
    report["scores"][result.parent.name] = {"status": value.get("status"),
        "banks": {bank: v.get("score") for bank, v in (value.get("banks") or {}).items()}}
Path(sys.argv[1]).write_text(json.dumps(report, indent=1))
print("ok")
