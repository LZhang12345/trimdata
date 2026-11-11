"""Seal PLAN/OWNER_REVIEW and admit (held submit, readback, release) LESS-approx batches through prepare.py."""
import importlib.util, json, os, sys
HOME = "/home/luyangz/data_selection_multimodal"
spec = importlib.util.spec_from_file_location("less_prepare", HOME + "/experiments/helper_tool2_less_approx_v2_20261105/prepare.py")
q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
p, d = q.modules()
out = {}
for unit in sys.argv[1:]:
    folder = q.control(unit, 1)
    os.makedirs(folder, 0o700, exist_ok=True)
    try:
        if not (folder / "PLAN.json").exists():
            plan = q.make_plan(p, d, unit, 1)
            plan_ref = p.seal(folder / "PLAN.json", plan)
            p.seal(folder / "OWNER_REVIEW.json", dict(
                status=q.REVIEW_STATUS, independent=False, reviewer=q.OWNER, plan=plan_ref, sources=plan["sources"],
                blocking_findings=[], admitted_by="less-approx-admit-v2",
                self_review=["record-signal v5 supervisor with LESS child; same candidate pool + text_only 3x; deviations listed in plan"]))
        job = q.admit(p, d, unit, 1)
        released = q.admit(p, d, unit, 1, release=True)
        out[unit] = job if released == job else "RELEASE_MISMATCH"
    except Exception as error:
        out[unit] = "ERROR " + repr(error)[:300]
    print(unit, out[unit], flush=True)
