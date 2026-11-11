"""Materialize the LESS/ICONS shared warmup draw W (Claude PROTOCOL_DECISIONS_V1 D2b).

W = int(0.05 * 660,763) = 33,038 records drawn uniformly from the candidate set C with seed 42, using
LESS's load_raw_dataset rule (np.random.seed(seed); np.random.permutation(len(C))[:size]; select in that
order). C excludes reference images and the 2,000 reference records. Reads only frozen inputs.
"""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np

CANDIDATES = ("/data/hf_cache/project_artifacts/DS-MultiModal/claude_less_icons_v1_20261030/inputs-v2/CANDIDATES.json",
              18553342, "50e0239b736e005686c3fe0cd940cd25f85c3748c1225099dbf5c7005f6fabfc")
RECORDS = ("/data/user_data/luyangz/native_campaign_20261023/root-wholepool-authority-v1/authority/records.json",
           760064389, "1c53e08b940af27483201efc4c2fa27e3f5c4f7862484e078736c718673c1d36")
OUT = Path("/data/hf_cache/project_features/DS-MultiModal/tool3-baselines-20261104/less-icons-warmup-W")
N_POOL, SEED = 660763, 42
SIZE = int(0.05 * N_POOL)


def checked(path, size, digest):
    raw = Path(path).read_bytes()
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Frozen input changed: " + path)
    return json.loads(raw)


def rows_of(candidates):
    value = candidates["candidates"] if isinstance(candidates, dict) and "candidates" in candidates else candidates
    if isinstance(value, dict) and "rows" in value:
        value = value["rows"]
    if all(type(item) is int for item in value):
        return value
    if all(isinstance(item, dict) and type(item.get("pool_row")) is int for item in value):
        return [item["pool_row"] for item in value]
    raise ValueError("Unrecognized CANDIDATES structure: " + json.dumps(value[:2])[:300])


def main(receipt_path):
    candidates = checked(*CANDIDATES)
    rows = rows_of(candidates)
    if len(rows) != 653770 or len(set(rows)) != len(rows) or not all(0 <= r < N_POOL for r in rows):
        raise ValueError("Candidate set differs from the frozen N'=653,770")
    records = checked(*RECORDS)
    if len(records) != N_POOL:
        raise ValueError("Pool size differs")
    np.random.seed(SEED)
    index = np.random.permutation(len(rows))[:SIZE]
    selected_rows = [rows[int(i)] for i in index]
    selected = [records[r] for r in selected_rows]
    OUT.mkdir(parents=True, exist_ok=True)
    body = json.dumps(selected, ensure_ascii=False).encode() + b"\n"
    tmp = OUT / ".records.json.tmp"
    tmp.write_bytes(body)
    os.replace(tmp, OUT / "records.json")
    rows_body = json.dumps(selected_rows).encode() + b"\n"
    (OUT / "pool_rows.json").write_bytes(rows_body)
    receipt = {"schema": "tool3-less-icons-warmup-W-v1", "size": SIZE, "seed": SEED, "rule": "np.random.seed(42); permutation(len(C))[:33038]",
               "candidates": dict(zip(("path", "bytes", "sha256"), CANDIDATES)), "pool_records": dict(zip(("path", "bytes", "sha256"), RECORDS)),
               "candidate_structure_first": json.dumps(candidates if isinstance(candidates, list) else list(candidates))[:200],
               "records": {"path": str(OUT / "records.json"), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()},
               "pool_rows": {"path": str(OUT / "pool_rows.json"), "sha256": hashlib.sha256(rows_body).hexdigest()},
               "numpy": np.__version__, "first_rows": selected_rows[:5], "job": os.environ.get("SLURM_JOB_ID")}
    Path(receipt_path).write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt["records"]))


if __name__ == "__main__":
    main(sys.argv[1])
