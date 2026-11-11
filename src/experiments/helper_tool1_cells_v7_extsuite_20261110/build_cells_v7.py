"""One-shot builder for CELLS_V7.json (run on the Babel login node; HOME records only, producer RESULT pins from the
on-node hashing receipt PRODUCER_RESULT_HASHES_V7_20261110.json). Prints the canonical JSON; never writes.
Derived from build_cells_v6.py: two cells, per-cell roster (ROSTER_V8), schema/status/request/supersedes of V7."""
import hashlib, json, os, sys

H = "/home/luyangz/data_selection_multimodal"
C = H + "/collaboration/helper-tool1-baseline-independent-20261101"
T = C + "/baseline-train-v5"
T3 = H + "/collaboration/helper-tool3-baselines-20261104"
T3C = C + "/t3-coincide-budget-eval-v2"


def ref(path):
    raw = open(path, "rb").read()
    return {"path": path, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def load(path):
    return json.load(open(path))


hashes = load(sys.argv[1])["cells"]
TOOL1 = [  # cell, kind, roster, short, fraction, human method
    ("t2-m1ur-b05", "tool1_v10_original", "ROSTER_V8.json", "t2ur05s17", "0.05",
     "Tool2 external selection m1ur (Tool2-owned selection; trained by Tool1 trainer v10)"),
    ("t2-univn-b05", "tool1_v10_original", "ROSTER_V8.json", "t2uvn05s17", "0.05",
     "Tool2 external selection univn (Tool2-owned selection; trained by Tool1 trainer v10)"),
]
cells, rosters = {}, {}
for cell, kind, roster_name, short, fraction, method in TOOL1:
    rosters[roster_name] = ref(T + "/" + roster_name)
    entry = load(T + "/" + roster_name)["cells"][cell]
    folder = T + "/" + cell
    results = sorted(n for n in os.listdir(folder) if n.startswith("TRAINING_RESULT-"))
    assert len(results) == 1, (cell, results)
    tr = load(folder + "/" + results[0])
    job = tr["job_id"]
    spec = dict(K=entry["K"], fraction=fraction, kind="student", method=method, method_key=entry["method"],
                producer_kind=kind, producer_job=job, producer_job_name=entry["job_name"],
                producer_result={k: hashes[cell]["result"][k] for k in ("path", "bytes", "sha256")},
                roster=roster_name, roster_cell=cell, selection_seed=entry["selection_seed"], short=short,
                steps=(entry["K"] + 3) // 4 // 32, training_result=ref(folder + "/" + results[0]),
                training_seed=entry["training_seed"])
    assert hashes[cell]["job"] == job and hashes[cell]["status"] == "ACTUAL_FULL_NATIVE_BASELINE_EPOCH_NOT_EVALUATED"
    assert hashes[cell]["full"] is True and hashes[cell]["steps"] == spec["steps"] and hashes[cell]["K"] == spec["K"]
    if kind == "tool1_v8_successor":
        n = tr["successor"]
        plan = folder + "/PLAN-v8.json"
        records = {"PLAN": ref(plan)}
        for name in ("ACK", "FUNCTION_CLAIM", "HELD_READBACK"):
            records[name] = ref("%s/successor-%d/%s.json" % (folder, n, name))
        spec.update(successor=n, trajectory=dict(owner_job=tr["trajectory_owner_job"], chain=tr["predecessor_chain"],
                                                 original_records={x: ref(folder + "/" + x + ".json")
                                                                   for x in ("PLAN", "ACK")}))
    else:
        assert tr.get("successor") is None
        records = {name: ref(folder + "/" + name + ".json") for name in ("PLAN", "ACK", "FUNCTION_CLAIM", "HELD_READBACK")}
    assert tr["plan"] == records["PLAN"], cell
    spec["trainer_records"] = records
    spec["trainer_prepare_sha256"] = load(records["PLAN"]["path"])["sources"]["prepare.py"]["sha256"]
    cells[cell] = spec

doc = dict(
    schema="tool1-baseline-cells-v7-extsuite-custody-v1",
    status="REGISTERED_2_BUDGET_CURVE_PRODUCERS_TOOL1_V10_ORIGINAL_CUSTODY_ONLY_ELIGIBILITY_LATE_BOUND",
    request="Tool1 dispatch 2026-11-10: add Tool2's trained 5% cells t2-m1ur-b05 (trainer v10 job 11097270 COMPLETED, "
            "sealed TRAINING_RESULT-11097270-r0) and t2-univn-b05 (job 11097271 COMPLETED, TRAINING_RESULT-11097271-r0), "
            "both ROSTER_V8, to the extended suite exactly as CELLS_V6 did for t2-m1ur-b20. t2-p34vn-b05, coincide-b20-s17, "
            "t2-m1uvn-b20 and t2-univn-b20 are excluded here (add-later; HANDOFF_CELLS_V7_20261110.json).",
    custody_policy="Custody comes only from each producer's own sealed records, by kind: Tool1 trainer v6/v10 "
                   "original PLAN/ACK/FUNCTION_CLAIM/HELD_READBACK (v10: ROSTER_V7 or ROSTER_V8), trainer v8 PLAN-v8 + successor-<n> records and "
                   "trajectory, or the Tool3 chain-v3 receipts under the t3coin v2 accept rule plus its executed, "
                   "independently reviewed plan. The completion record and producer RESULT.json are pinned by size and "
                   "sha256. A cell is plannable only when a completed cells-v7 inventory RESULT says PASS for it.",
    horizon_policy="steps = (K + 3) // 4 // 32; final adapter in <checkpoint_root>/<cell>/training (no checkpoint-<step>).",
    rosters=rosters, cells=cells,
    supersedes=dict(scope="custody of the listed cells only; CELLS_V3/CELLS_V4/CELLS_V5/CELLS_V6 and their bound "
                          "evaluations are unchanged and remain authoritative for every other cell",
                    CELLS_V3=ref(H + "/experiments/helper_tool1_cells_v3_20261104/CELLS_V3.json"),
                    CELLS_V4=ref(H + "/experiments/helper_tool1_cells_v4_custody_20261107/CELLS_V4.json"),
                    CELLS_V5=ref(H + "/experiments/helper_tool1_cells_v5_extsuite_20261108/CELLS_V5.json"),
                    CELLS_V6=ref(H + "/experiments/helper_tool1_cells_v6_extsuite_20261109/CELLS_V6.json")),
    producer_result_hashing=ref(sys.argv[1]))
print(json.dumps(doc, indent=1, sort_keys=True))
