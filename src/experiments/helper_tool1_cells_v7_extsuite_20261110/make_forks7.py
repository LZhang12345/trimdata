"""Generate the CELLS_V7 custody-only forks from the exact executed sources (exact-count string replacements).
make_forks6.py through v7_tokens.py plus: the two-cell list and the score adapter score6.py -> score7.py.

Run locally: python3 make_forks.py <src_root> <out_dir>. Prints sha256 of every generated file. Every replacement
asserts its exact occurrence count, so a drifted source fails loudly instead of silently mis-forking."""
import hashlib
import sys
from pathlib import Path

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
H = "/home/luyangz/data_selection_multimodal"
V5 = "helper_tool1_cells_v7_extsuite_20261110"


def pin(name):
    raw = (OUT / name).read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def fork(src, dst, reps):
    text = (SRC / src).read_text()
    for old, new, count in reps:
        found = text.count(old)
        assert found == count, (dst, old[:90], found, count)
        text = text.replace(old, new)
    (OUT / dst).write_text(text)
    return dst


CELLS_BYTES, CELLS_SHA = pin("CELLS_V7.json")
CUST_BYTES, CUST_SHA = pin("custody7.py")
CELLS_PIN = '{"path": str(SOURCE / "CELLS_V7.json"), "bytes": %d, "sha256": "%s"}' % (CELLS_BYTES, CELLS_SHA)
CUST_PIN = '{"path": str(SOURCE / "custody7.py"), "bytes": %d, "sha256": "%s"}' % (CUST_BYTES, CUST_SHA)
CELLS = ["t2-m1ur-b05", "t2-univn-b05"]
CELLS_TUPLE = "CELLS = (" + ",\n         ".join(
    ", ".join('"%s"' % c for c in CELLS[i:i + 2]) for i in range(0, len(CELLS), 2)) + ("," if len(CELLS) == 1 else "") + ")"

PLAN_CUSTODY_DOC = ('    """Plan-time custody through custody7.plan_custody (the pinned records of the cell\'s producer kind, read with\n'
                    '    this controller\'s exact readers) plus the cells-v7 inventory ACK."""\n')
PLAN_CUSTODY_BODY = (
    '    custody3 = bound_custody(p)\n'
    '    custody = custody3.plan_custody(cells, cell, p.checked, p.descriptor, os.listdir)\n'
    '    binding = custody3.inventory_binding(inventory_ref, json.loads(p.checked(inventory_ref)))\n'
    '    require(cell in binding["cells"], "The bound cells-v7 inventory run does not cover this cell")\n'
    '    return dict(custody, inventory=binding)\n')
OLD_PLAN_CUSTODY_BODY = (
    '    custody3 = bound_custody(p)\n'
    '    spec = custody3.cell_spec(cells, cell)\n'
    '    records = {name: json.loads(p.checked(spec["trainer_records"][name])) for name in custody3.RECORD_NAMES}\n'
    '    roster = json.loads(p.checked(cells["rosters"][spec["roster"]]))\n'
    '    folder = Path(custody3.TRAIN_CONTROL) / cell\n'
    '    names = sorted(name for name in os.listdir(folder) if name.startswith("TRAINING_RESULT-"))\n'
    '    require(len(names) == 1, "Exactly one sealed TRAINING_RESULT required; unfinished cells stay registered only")\n'
    '    result_ref = p.descriptor(folder / names[0])\n'
    '    custody = custody3.trainer_custody(cells, cell, records, result_ref, json.loads(p.checked(result_ref)), roster)\n'
    '    binding = custody3.inventory_binding(inventory_ref, json.loads(p.checked(inventory_ref)))\n'
    '    require(cell in binding["cells"], "The bound cells-v3 inventory run does not cover this cell")\n'
    '    return dict(custody, inventory=binding)\n')
OLD_PLAN_CUSTODY_DOC = ('    """Plan-time custody: the producer\'s own HOME records (complete epoch only) plus the cells-v3 inventory ACK."""\n')

# ----------------------------------------------------------------------------------------------- inventory admit7
ADMIT_HEADER_OLD = '''"""Plan, hold-submit, release and run Tool1's CELLS_V4 custody and parameter-identity inventory (one GPU allocation).

CELLS_V4 custody-only fork of helper_tool1_cells_v3_20261104/admit3.py (2026-11-07): identical inventory logic, bound
to CELLS_V4.json/custody4.py (General successor-trainer custody) with its own control/output roots, owner, statuses,
job prefix and comment. The native child is the byte-identical identity3.py (it requires the v3 ROLE string and the
"custody3.py" source key, which here names custody4.py). The original v3 docstring follows.
'''
ADMIT_HEADER_NEW = '''"""Plan, hold-submit, release and run Tool1's CELLS_V7 custody and parameter-identity inventory (one GPU allocation).

CELLS_V7 custody-only fork of helper_tool1_cells_v4_custody_20261107/admit4.py (2026-11-10): identical inventory logic,
bound to CELLS_V7.json/custody7.py (four producer kinds: Tool1 trainer v6 original, v8 successor, v10 original and the
Tool3 chain-v3 RESULT under the t3coin v2 accept rule) with its own control/output roots, owner, statuses, job prefix
and comment. Plan-time custody goes through custody7.plan_custody (pinned records of the cell's kind, this
controller's exact readers). The native child is the byte-identical identity3.py (it requires the v3 ROLE string and
the "custody3.py" source key, which here names custody7.py). The CELLS_V4 docstring follows.

CELLS_V4 custody-only fork of helper_tool1_cells_v3_20261104/admit3.py (2026-11-07): identical inventory logic, bound
to CELLS_V4.json/custody4.py (General successor-trainer custody) with its own control/output roots, owner, statuses,
job prefix and comment. The original v3 docstring follows.
'''
ADMIT_TRAINER_OLD = '''def trainer_custody(p, custody3, cells, cell):
    """Same plan-time custody the v3 evaluators bind (custody3.trainer_custody over pinned HOME records)."""
    spec = custody3.cell_spec(cells, cell)
    records = {name: json.loads(p.checked(spec["trainer_records"][name])) for name in custody3.RECORD_NAMES}
    roster = json.loads(p.checked(cells["rosters"][spec["roster"]]))
    folder = Path(custody3.TRAIN_CONTROL) / cell
    names = sorted(name for name in os.listdir(folder) if name.startswith("TRAINING_RESULT-"))
    require(len(names) == 1, "Exactly one sealed TRAINING_RESULT required; unfinished cells stay registered only: "
            + cell)
    result_ref = p.descriptor(folder / names[0])
    return custody3.trainer_custody(cells, cell, records, result_ref, json.loads(p.checked(result_ref)), roster)
'''
ADMIT_TRAINER_NEW = '''def trainer_custody(p, custody3, cells, cell):
    """Same plan-time custody the CELLS_V7 evaluators bind (custody7.plan_custody over the pinned HOME records)."""
    return custody3.plan_custody(cells, cell, p.checked, p.descriptor, os.listdir)
'''
fork("helper_tool1_cells_v4_custody_20261107/admit4.py", "admit7.py", [
    ("admit4.py", "admit7.py", 5),
    ('SOURCE = HOME / "experiments/helper_tool1_cells_v4_custody_20261107"', 'SOURCE = HOME / "experiments/%s"' % V5, 1),
    ("cells-v4-inventory-v1", "cells-v7-inventory-v1", 4),
    ("helper-tool1-cells-v7-inventory-v1-20261107", "helper-tool1-cells-v7-inventory-v1-20261110", 1),
    ('"A live cells-v4 inventory already exists on some route"', '"A live cells-v7 inventory already exists on some route"', 1),
    ('OWNER = "Tool1-owned-CELLS-V4-custody', 'OWNER = "Tool1-owned-CELLS-V7-custody', 1),
    ("TOOL1_CELLS_V4_INVENTORY", "TOOL1_CELLS_V7_INVENTORY", 2),
    ('"dsmm-t1v4-inv-r"', '"dsmm-t1v7-inv-r"', 1),
    ('    # Source key kept as "custody3.py" because identity3.py requires that exact key; it names custody4.py.\n'
     '    "custody3.py": {"path": str(SOURCE / "custody4.py"), "bytes": 23676, "sha256": "b30f8c8906210b473feb8429d6dfb527cfc569b42a6361d264fd039a0f3cc744"},\n'
     '    "CELLS_V4.json": {"path": str(SOURCE / "CELLS_V4.json"), "bytes": 5729, "sha256": "eba430cd79521c73a8b9f8f953c2530f84f9b1eda3f95d7f5579f7f2ffe9388c"},\n',
     '    # Source key kept as "custody3.py" because identity3.py requires that exact key; it names custody7.py.\n'
     '    "custody3.py": %s,\n    "CELLS_V7.json": %s,\n' % (CUST_PIN, CELLS_PIN), 1),
    ('SOURCES["CELLS_V4.json"]', 'SOURCES["CELLS_V7.json"]', 1),
    ("tool1_cells_v4_", "tool1_cells_v7_", 4),
    ('"Inventory identity differs from custody4.py"', '"Inventory identity differs from custody7.py"', 1),
    (ADMIT_TRAINER_OLD, ADMIT_TRAINER_NEW, 1),
    ('purpose="on-node-CELLS-V4-successor-producer-custody-and-loaded-parameter-digest"',
     'purpose="on-node-CELLS-V7-four-kind-producer-custody-and-loaded-parameter-digest"', 1),
    ('"inv_run.sbatch"', '"inv_run7.sbatch"', 5),
    (ADMIT_HEADER_OLD, ADMIT_HEADER_NEW, 1),
])

# ----------------------------------------------------------------------------------------------- VizWiz prepare7_vz
VZ_HEADER_OLD = '''"""Plan, hold-submit, release and run Tool1 VizWiz-test evaluations of CELLS_V4 (successor-custody) students.
'''
VZ_HEADER_NEW = '''"""Plan, hold-submit, release and run Tool1 VizWiz-test evaluations of CELLS_V7 (four-kind custody) students.

CELLS_V7 custody-only fork of helper_tool1_cells_v4_custody_20261107/prepare4.py (2026-11-10): identical v3 owner,
role, control/output roots, generation/export/scoring files and afterok-inventory admission; the "CELLS_V3.json"/
"custody3.py" source keys (read by the byte-identical v3 model.py) name CELLS_V7.json/custody7.py; plan-time custody
goes through custody7.plan_custody (pinned records of the cell's producer kind); own job names dsmm-t1v7-vz-*,
comment prefix, plan schema and review statuses; every request carries --dependency=afterok:<cells-v7 inventory job>.
The CELLS_V4 docstring follows.
'''
fork("helper_tool1_cells_v4_custody_20261107/prepare4.py", "prepare7_vz.py", [
    ("prepare4.py", "prepare7_vz.py", 5),
    ('SOURCE = HOME / "experiments/helper_tool1_cells_v4_custody_20261107"', 'SOURCE = HOME / "experiments/%s"' % V5, 1),
    ('"OWNER_REVIEWED_TOOL1_VIZWIZ_EVAL_V3_CUSTODY_V4_NOT_INDEPENDENT_ACCEPTANCE"',
     '"OWNER_REVIEWED_TOOL1_VIZWIZ_EVAL_V3_CUSTODY_V7_NOT_INDEPENDENT_ACCEPTANCE"', 1),
    ('"PASS_INDEPENDENT_EXACT_SOURCE_CONFIG_TOOL1_VIZWIZ_EVAL_V3_CUSTODY_V4"',
     '"PASS_INDEPENDENT_EXACT_SOURCE_CONFIG_TOOL1_VIZWIZ_EVAL_V3_CUSTODY_V7"', 1),
    ('COMMENT_PREFIX = "dsmm-t1-vz-eval-v3c4:"', 'COMMENT_PREFIX = "dsmm-t1-vz-eval-v3c7:"', 1),
    ('    "CELLS_V3.json": {"path": str(SOURCE / "CELLS_V4.json"), "bytes": 5729, "sha256": "eba430cd79521c73a8b9f8f953c2530f84f9b1eda3f95d7f5579f7f2ffe9388c"},\n'
     '    "custody3.py": {"path": str(SOURCE / "custody4.py"), "bytes": 23676, "sha256": "b30f8c8906210b473feb8429d6dfb527cfc569b42a6361d264fd039a0f3cc744"},\n',
     '    "CELLS_V3.json": %s,\n    "custody3.py": %s,\n' % (CELLS_PIN, CUST_PIN), 1),
    ('"Registered CELLS_V4 model required;', '"Registered CELLS_V7 model required;', 1),
    ('"tool1_cells_v4_custody"', '"tool1_cells_v7_custody"', 1),
    (OLD_PLAN_CUSTODY_DOC + OLD_PLAN_CUSTODY_BODY, PLAN_CUSTODY_DOC + PLAN_CUSTODY_BODY, 1),
    ('"dsmm-t1v4-%s-%s-a%d"', '"dsmm-t1v7-%s-%s-a%d"', 1),
    ('schema="tool1-vizwiz-eval-v3-custody-v4-plan-v1"', 'schema="tool1-vizwiz-eval-v3-custody-v7-plan-v1"', 1),
    ('"vz_run.sbatch"', '"vz_run7.sbatch"', 5),
    ('<cells-v4-inventory run-N/ACK.json>', '<cells-v7-inventory run-N/ACK.json>', 1),
    ('"The cells-v3 inventory run ACK.json path is required"', '"The cells-v7 inventory run ACK.json path is required"', 1),
    (VZ_HEADER_OLD, VZ_HEADER_NEW, 1),
])


# ----------------------------------------------------------------------------------------------- BENCH6/BENCH4 prepares
def bench_fork(bank, src, dst, sbatch):
    short = {"bench6": "b6", "bench4": "b4"}[bank]
    upper = bank.upper()
    v3dir = {"bench6": "helper_tool1_bench6_eval_v3_20261104", "bench4": "helper_tool1_bench4_eval_v3_20261104"}[bank]
    model = {"bench6": "model6.py", "bench4": "model4.py"}[bank]
    header_new = '''"""Plan, hold-submit, release and run Tool1 %s evaluations of CELLS_V7 (four-kind custody) students.

CELLS_V7 custody-only fork of %s/%s (2026-11-10), in the reviewed CELLS_V4 VizWiz pattern
(helper_tool1_cells_v4_custody_20261107/prepare4.py). Generation, journals, export and scoring code are the
byte-identical v3 files (%s, banks, routes; the v3 OWNER, ROLE, CONTROL_ROOT and OUTPUT_ROOT are unchanged, so the
v3 registered scoring steps and the BENCH4 judge see the result exactly like the other v3 cells). Differences, all
custody/admission:
  * custody: the "CELLS_V3.json"/"custody3.py" source keys (read by the byte-identical %s) name CELLS_V7.json and
    custody7.py; plan-time custody goes through custody7.plan_custody (pinned records of the cell's producer kind:
    Tool1 trainer v6 original, v8 successor, v10 original, or the Tool3 chain-v3 RESULT under the t3coin v2 accept
    rule) and binds the cells-v7 inventory run by its ACK;
  * admission: own job names (dsmm-t1v7-%s-*), comment prefix, plan schema and review statuses; every request
    carries --dependency=afterok:<cells-v7 inventory job> (the job the plan's custody binds). Release requires that
    inventory's sealed PASS RESULT or a still pending/running inventory job (RELEASE_INVENTORY_GATE.json); the worker
    late-binds the PASS RESULT before any data exactly as in v3 (custody_ready in work()).
The v3 docstring follows.

''' % (upper, v3dir, src.split("/")[-1], model, model, short)
    old_src = src.split("/")[-1]
    reps = [
        ('SOURCE = HOME / "experiments/%s"\n' % v3dir,
         'SOURCE = HOME / "experiments/%s"\nV3SRC = HOME / "experiments/%s"\n' % (V5, v3dir), 1),
        ('str(SOURCE / "%s")' % model, 'str(V3SRC / "%s")' % model, 1),
        ('    "CELLS_V3.json": {"path": str(CELLS3 / "CELLS_V3.json"), "bytes": 52268, "sha256": "e1bcb9c713f3250234d0b10c790006d3b101807d728b59718f6a75963baf5500"},\n'
         '    "custody3.py": {"path": str(CELLS3 / "custody3.py"), "bytes": 18002, "sha256": "11fbdcac5034b85eb41a7ffc32bde09875337bf32a67f30cbca12c27b8105164"},\n',
         '    # Keys kept because the byte-identical %s reads sources["CELLS_V3.json"] and sources["custody3.py"].\n'
         '    "CELLS_V3.json": %s,\n    "custody3.py": %s,\n' % (model, CELLS_PIN, CUST_PIN), 1),
        ('REVIEW_STATUS = "OWNER_REVIEWED_TOOL1_%s_EVAL_V3_NOT_INDEPENDENT_ACCEPTANCE"' % upper,
         'REVIEW_STATUS = "OWNER_REVIEWED_TOOL1_%s_EVAL_V3_CUSTODY_V7_NOT_INDEPENDENT_ACCEPTANCE"' % upper, 1),
        ('INDEPENDENT_STATUS = "PASS_INDEPENDENT_EXACT_SOURCE_CONFIG_TOOL1_%s_EVAL_V3"' % upper,
         'INDEPENDENT_STATUS = "PASS_INDEPENDENT_EXACT_SOURCE_CONFIG_TOOL1_%s_EVAL_V3_CUSTODY_V7"' % upper, 1),
        ('COMMENT_PREFIX = "dsmm-t1-%s-v3:"' % bank, 'COMMENT_PREFIX = "dsmm-t1-%s-v3c7:"' % bank, 1),
        ('"Registered CELLS_V3 model required;', '"Registered CELLS_V7 model required;', 1),
        ('    return bound_module(p, "custody3.py", "tool1_cells_v3_custody")',
         '    return bound_module(p, "custody3.py", "tool1_cells_v7_custody")', 1),
        (OLD_PLAN_CUSTODY_DOC + OLD_PLAN_CUSTODY_BODY, PLAN_CUSTODY_DOC + PLAN_CUSTODY_BODY, 1),
        ('"dsmm-t1v3-%s-%%s-a%%d"' % short, '"dsmm-t1v7-%s-%%s-a%%d"' % short, 1),
        ('schema="tool1-%s-eval-v3-plan-v1"' % bank, 'schema="tool1-%s-eval-v3-custody-v7-plan-v1"' % bank, 1),
        ('        sources=dict({"%s": p.descriptor(SOURCE / "%s"),\n'
         '                      "run.sbatch": p.descriptor(SOURCE / "run.sbatch")}, **SOURCES),\n' % (old_src, old_src),
         '        sources=dict({"%s": p.descriptor(SOURCE / "%s"),\n'
         '                      "%s": p.descriptor(SOURCE / "%s")}, **SOURCES),\n'
         '        dependency="afterok:" + custody["inventory"]["job_id"],\n' % (dst, dst, sbatch, sbatch), 1),
        ('            str(SOURCE / "run.sbatch"), cell, str(attempt), check_route(plan["route"])]',
         '            "--dependency=" + plan["dependency"],\n'
         '            str(SOURCE / "%s"), cell, str(attempt), check_route(plan["route"])]' % sbatch, 1),
        ('        Command=str(SOURCE / "run.sbatch"), WorkDir=str(HOME), Dependency="(null)",\n',
         '        Command=str(SOURCE / "%s"), WorkDir=str(HOME),\n' % sbatch, 1),
        ('            and all(row.get(key) == value for key, value in expected.items()) and nodes\n',
         '            and all(row.get(key) == value for key, value in expected.items()) and nodes\n'
         '            and row.get("Dependency") in dependencies\n', 1),
        ('    # Slurm shows a pending single-node request as NumNodes=1-1 and a running one as 1.\n',
         '    # Slurm shows afterok:<job>(unfulfilled) while unsatisfied and afterok:<job>[(fulfilled)] or (null) afterwards.\n'
         '    dependencies = (("(null)", plan["dependency"], plan["dependency"] + "(fulfilled)")\n'
         '                    + (() if running else (plan["dependency"] + "(unfulfilled)",)))\n'
         '    # Slurm shows a pending single-node request as NumNodes=1-1 and a running one as 1.\n', 1),
        ('    require(spool == p.checked(plan["sources"]["run.sbatch"]), "Exact immutable bound spool required")',
         '    require(spool == p.checked(plan["sources"]["%s"]), "Exact immutable bound spool required")' % sbatch, 1),
        ('p.descriptor(SOURCE / "%s")' % old_src, 'p.descriptor(SOURCE / "%s")' % dst, 2),
        ('        custody_ready(p, plan)\n        intent = p.seal(afolder / "RELEASE_INTENT.json",',
         '        gate = inventory_gate(p, plan)\n'
         '        p.seal(afolder / "RELEASE_INVENTORY_GATE.json", dict(ACK=ack_ref, job_id=job, gate=gate))\n'
         '        intent = p.seal(afolder / "RELEASE_INTENT.json",', 1),
        ('\n\ndef job_name(cells, cell, attempt):',
         '\n\ndef inventory_gate(p, plan):\n'
         '    """Release gate: the bound inventory\'s PASS RESULT (v3 custody_ready), or its job still pending/running so that the\n'
         '    afterok request can only start after it COMPLETED; the worker re-checks custody_ready before any data."""\n'
         '    custody = plan["function"]["custody"]\n'
         '    job = custody["inventory"]["job_id"]\n'
         '    require(plan["dependency"] == "afterok:" + job, "Dependency must name exactly the bound inventory job")\n'
         '    if Path(custody["inventory"]["result_path"]).exists():\n'
         '        return dict(mode="inventory_result_pass", custody=custody_ready(p, plan))\n'
         '    state, code = sacct_state(p, job)\n'
         '    require(state in ("PENDING", "RUNNING"), "Bound inventory job is neither live nor sealed a RESULT: " + state)\n'
         '    return dict(mode="inventory_dependency_live", inventory_job=job, state=state, exit_code=code)\n'
         '\n\ndef job_name(cells, cell, attempt):', 1),
        ("<cells-v3-inventory run-N/ACK.json>", "<cells-v7-inventory run-N/ACK.json>", 1),
        ("  %s plan " % old_src, "  %s plan " % dst, 1),
        ("  %s self-review " % old_src, "  %s self-review " % dst, 1),
        ("  %s submit|release " % old_src, "  %s submit|release " % dst, 1),
    ]
    if bank == "bench6":
        reps.append(('require(args.inventory, "The cells-v3 inventory run ACK.json path is required")',
                     'require(args.inventory, "The cells-v7 inventory run ACK.json path is required")', 1))
    else:
        reps.append(('"The cells-v3 inventory run ACK and the BENCH4 input inventory ACK paths are required"',
                     '"The cells-v7 inventory run ACK and the BENCH4 input inventory ACK paths are required"', 1))
    reps.append(('"""Plan, hold-submit, release and run Tool1 ', header_new + 'Plan, hold-submit, release and run Tool1 ', 1))
    fork(src, dst, reps)


bench_fork("bench6", "helper_tool1_bench6_eval_v3_20261104/prepare6.py", "prepare7_b6.py", "b6_run7.sbatch")
bench_fork("bench4", "helper_tool1_bench4_eval_v3_20261104/prepare4.py", "prepare7_b4.py", "b4_run7.sbatch")


# ----------------------------------------------------------------------------------------------- sbatch bootstraps
def sbatch(src, dst, script, old_script, root_old, root_new, size_old, sha_old, extra=()):
    size, sha = pin(script)
    reps = [(old_script, "%s/experiments/%s/%s" % (H, V5, script), 1),
            ("SOURCE_SIZE = %d" % size_old, "SOURCE_SIZE = %d" % size, 1),
            ('"%s"' % sha_old, '"%s"' % sha, 1)] + list(extra)
    if root_old != root_new:
        reps.append((root_old, root_new, 1))
    fork(src, dst, reps)


OLD_V3_CELLS = None
sbatch("helper_tool1_cells_v4_custody_20261107/inv_run.sbatch", "inv_run7.sbatch", "admit7.py",
       H + "/experiments/helper_tool1_cells_v4_custody_20261107/admit4.py",
       "cells-v4-inventory-v1", "cells-v7-inventory-v1", 34637,
       "468d2644a2bf1a152adea77de7c488d94398742cc9b68405fac408daa22db4e0")
sbatch("helper_tool1_cells_v4_custody_20261107/vz_run.sbatch", "vz_run7.sbatch", "prepare7_vz.py",
       H + "/experiments/helper_tool1_cells_v4_custody_20261107/prepare4.py", "", "", 39631,
       "63e7753a0779269f26f08b7057f7a9f60dc3e060e0138b0f5463eaf6902fb8f0",
       [('CELLS = ("m10-datatailor-k66071-train17",)', CELLS_TUPLE, 1),
        ("CELLS_V4 evaluation request required", "CELLS_V7 evaluation request required", 1)])
B6_CELLS_OLD = (SRC / "helper_tool1_bench6_eval_v3_20261104/run.sbatch").read_text().split("CELLS = (")[1].split(")")[0]
B4_CELLS_OLD = (SRC / "helper_tool1_bench4_eval_v3_20261104/run.sbatch").read_text().split("CELLS = (")[1].split(")")[0]
sbatch("helper_tool1_bench6_eval_v3_20261104/run.sbatch", "b6_run7.sbatch", "prepare7_b6.py",
       H + "/experiments/helper_tool1_bench6_eval_v3_20261104/prepare6.py", "", "", 37105,
       "6fdb5907d2081bd9c6ea95579193f1b3de3eeaf2478ca626755c768209949bcd",
       [("CELLS = (" + B6_CELLS_OLD + ")", CELLS_TUPLE, 1),
        ("BENCH6 CELLS_V3 evaluation request required", "BENCH6 CELLS_V7 evaluation request required", 1)])
sbatch("helper_tool1_bench4_eval_v3_20261104/run.sbatch", "b4_run7.sbatch", "prepare7_b4.py",
       H + "/experiments/helper_tool1_bench4_eval_v3_20261104/prepare4.py", "", "", 39588,
       "f03c756601f32c005d9c86e6799fc0e47857e8fd232e390696e5c5184dc2ee32",
       [("CELLS = (" + B4_CELLS_OLD + ")", CELLS_TUPLE, 1),
        ("BENCH4 CELLS_V3 evaluation request required", "BENCH4 CELLS_V7 evaluation request required", 1)])

for name in ("custody7.py", "CELLS_V7.json", "admit7.py", "inv_run7.sbatch", "prepare7_vz.py", "vz_run7.sbatch",
             "prepare7_b6.py", "b6_run7.sbatch", "prepare7_b4.py", "b4_run7.sbatch"):
    size, sha = pin(name)
    print("%-18s %7d %s" % (name, size, sha))


# ----------------------------------------------------------------------------------------------- score6 adapter
def sha_of(path):
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


template = (Path(__file__).parent / "score7.template.py").read_text()
fills = {"@CELLS_BYTES@": str(CELLS_BYTES), "@CELLS_SHA@": CELLS_SHA, "@CUST_BYTES@": str(CUST_BYTES),
         "@CUST_SHA@": CUST_SHA}
for key, name in (("B6", "prepare7_b6.py"), ("B4", "prepare7_b4.py"), ("VZ", "prepare7_vz.py")):
    size, sha = pin(name)
    fills["@%s_BYTES@" % key], fills["@%s_SHA@" % key] = str(size), sha
for key, rel, size_expected in (("B6STEP", "helper_tool1_bench6_eval_v3_20261104/score_step6.py", 33657),
                                ("B4STEP", "helper_tool1_bench4_score_v3_20261105/score_step4_v3.py", 58039)):
    size, sha = sha_of(SRC / rel)
    assert size == size_expected, rel
    fills["@%s_SHA@" % key] = sha
vz_size, vz_sha = sha_of(SRC / "helper_tool1_vizwiz_eval_v3_20261104/score_step.py")
assert (vz_size, vz_sha) == (38369, "33ad9775ff6215319e32c5f4bee347e385cf869b85792d875c363287ff91b05b")
for key, value in fills.items():
    assert template.count(key) >= 1, key
    template = template.replace(key, value)
assert "@" not in "".join(line for line in template.splitlines() if "@" in line and "SHA@" in line)
(OUT / "score7.py").write_text(template)
size, sha = pin("score7.py")
C5 = H + "/collaboration/helper-tool1-baseline-independent-20261101"
fork("helper_tool1_cells_v4_custody_20261107/score4_run.sbatch", "score7_run.sbatch", [
    (H + "/experiments/helper_tool1_cells_v4_custody_20261107/score4.py", "%s/experiments/%s/score7.py" % (H, V5), 1),
    ("root=" + C5 + "/vizwiz-eval-v3/vizwiz-test-scoring-afterok-v4\n", "root=" + C5 + "\n", 1),
    ('path, root, batch, attempt, route = sys.argv[1:]\n'
     'if (batch not in ("0", "1", "2", "3", "4", "5", "6", "7") or attempt not in ("0", "1", "2", "3")\n'
     '        or route not in ("general", "preempt")):\n'
     '    raise ValueError("Exact finite Tool1 VizWiz afterok scoring request required")\n',
     'path, root, scorer, batch, attempt, route = sys.argv[1:]\n'
     'ROOTS = {"bench6": "/bench6-eval-v3/scoring-afterok-v7", "bench4": "/bench4-eval-v3/scoring-afterok-v7",\n'
     '         "vizwiz": "/vizwiz-eval-v3/vizwiz-test-scoring-afterok-v7"}\n'
     'if (scorer not in ROOTS or batch not in ("0", "1", "2", "3", "4", "5", "6", "7")\n'
     '        or attempt not in ("0", "1", "2", "3") or route not in ("general", "preempt")):\n'
     '    raise ValueError("Exact finite Tool1 CELLS_V7 afterok scoring request required")\n', 1),
    ('parent(root + "/batch-" + batch', 'parent(root + ROOTS[scorer] + "/batch-" + batch', 1),
    ("SOURCE_SIZE = 29168", "SOURCE_SIZE = %d" % size, 1),
    ('"a0ad6b768f0bf5a9be70418387af4882469178032a6985c607b7818dd6760474"', '"%s"' % sha, 1),
    ('sys.argv = [path, "worker", "--batch", batch, "--attempt", attempt, "--route", route]',
     'sys.argv = [path, "worker", "--scorer", scorer, "--batch", batch, "--attempt", attempt, "--route", route]', 1),
    ("' \"$script\" \"$root\" \"$1\" \"$2\" \"$3\"\n", "' \"$script\" \"$root\" \"$1\" \"$2\" \"$3\" \"$4\"\n", 1),
])
for name in ("score7.py", "score7_run.sbatch"):
    size, sha = pin(name)
    print("%-18s %7d %s" % (name, size, sha))
