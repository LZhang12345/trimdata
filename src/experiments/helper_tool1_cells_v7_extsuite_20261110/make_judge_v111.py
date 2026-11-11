"""Generate judge v1.11 (helper_tool1_bench4_judge_v111_20261110) from the executed v1.10 step: custody bump only
(family bench4-v3c7 for CELLS_V7 cells outside CELLS_V3/CELLS_V5/CELLS_V6, afterok mode extended to CELLS_V7, v1.10
retirement). Pattern of make_judge_v110.py. Run: python3 make_judge_v111.py <v1.10 dir> <out dir> <CELLS_V7 dir>."""
import hashlib
import sys
from pathlib import Path

SRC, OUT, V7OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
H = "/home/luyangz/data_selection_multimodal"


def pin(path):
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


B4_BYTES, B4_SHA = pin(V7OUT / "prepare7_b4.py")
C7_BYTES, C7_SHA = pin(V7OUT / "CELLS_V7.json")
text = (SRC / "score_step4_v110.py").read_text()
doc_end = text.index('"""', 3) + 3
doc, code = text[:doc_end], text[doc_end:]


def rep(old, new, count=1):
    global code
    found = code.count(old)
    assert found == count, (old[:100], found, count)
    code = code.replace(old, new)


rep("v110", "v111", code.count("v110"))
rep("V110", "V111", code.count("V110"))
rep('                    "v1.9": (EVAL_CONTROL / "scoring-judge-v19", "dsmm-t1-b4j-v19-")}',
    '                    "v1.9": (EVAL_CONTROL / "scoring-judge-v19", "dsmm-t1-b4j-v19-"),\n'
    '                    "v1.10": (EVAL_CONTROL / "scoring-judge-v110", "dsmm-t1-b4j-v110-")}')
rep('                          EVAL_CONTROL / "scoring-judge-v18", EVAL_CONTROL / "scoring-judge-v19")',
    '                          EVAL_CONTROL / "scoring-judge-v18", EVAL_CONTROL / "scoring-judge-v19",\n'
    '                          EVAL_CONTROL / "scoring-judge-v110")')
rep('                        (RETIRED_SCHEMA, RETIRED_STATUS, "Tool1-owned-BENCH4-registered-judge-v111"))',
    '                        ("tool1-bench4-judge-version-retirement-v110",\n'
    '                         "OWNER_RETIRED_BENCH4_JUDGE_VERSION_NO_LIVE_JOB_SUPERSEDED_BY_V110",\n'
    '                         "Tool1-owned-BENCH4-registered-judge-v110"),\n'
    '                        (RETIRED_SCHEMA, RETIRED_STATUS, "Tool1-owned-BENCH4-registered-judge-v111"))')
rep("# valid; v1.8 sealed v1.7's; v1.9 sealed v1.8's; v1.10 seals its own (v1.9).",
    "# valid; v1.8 sealed v1.7's; v1.9 sealed v1.8's; v1.10 sealed v1.9's; v1.11 seals its own (v1.10).")
rep('RETIRE_COMMAND_VERSIONS = ("v1.9",)', 'RETIRE_COMMAND_VERSIONS = ("v1.10",)')
rep('"Earlier judge version v1.9 required (v1.1/v1.2 retired by v1.3, v1.3 by v1.4, v1.4 by v1.5, v1.5 by v1.6, v1.6 by v1.7, v1.7 by v1.8, v1.8 by v1.9)"',
    '"Earlier judge version v1.10 required (v1.1/v1.2 retired by v1.3, v1.3 by v1.4, v1.4 by v1.5, v1.5 by v1.6, v1.6 by v1.7, v1.7 by v1.8, v1.8 by v1.9, v1.9 by v1.10)"')
rep('SOURCE = HOME / "experiments/helper_tool1_bench4_judge_v111_20261109"\n',
    'SOURCE = HOME / "experiments/helper_tool1_bench4_judge_v111_20261110"\n')
rep('CELLS6 = HOME / "experiments/helper_tool1_cells_v6_extsuite_20261109"\n',
    'CELLS6 = HOME / "experiments/helper_tool1_cells_v6_extsuite_20261109"\n'
    'CELLS7 = HOME / "experiments/helper_tool1_cells_v7_extsuite_20261110"\n')
rep('    "CELLS_V6.json": {"path": str(CELLS6 / "CELLS_V6.json"), "bytes": 4929,\n',
    '    "bench4_v7_prepare7_b4.py": {"path": str(CELLS7 / "prepare7_b4.py"), "bytes": %d,\n'
    '                                 "sha256": "%s"},\n'
    '    "CELLS_V7.json": {"path": str(CELLS7 / "CELLS_V7.json"), "bytes": %d,\n'
    '                      "sha256": "%s"},\n'
    '    "CELLS_V6.json": {"path": str(CELLS6 / "CELLS_V6.json"), "bytes": 4929,\n' % (B4_BYTES, B4_SHA, C7_BYTES, C7_SHA))
rep('V6_GENERATION_SCHEMA = "tool1-bench4-eval-v3-custody-v6-plan-v1"\n'
    'V5_FAMILIES = ("bench4-v3", "bench4-v3c5", "bench4-v3c6")\n',
    'V6_GENERATION_SCHEMA = "tool1-bench4-eval-v3-custody-v6-plan-v1"\n'
    'V7_GENERATION_SCHEMA = "tool1-bench4-eval-v3-custody-v7-plan-v1"\n'
    'V5_FAMILIES = ("bench4-v3", "bench4-v3c5", "bench4-v3c6", "bench4-v3c7")\n')
rep('''def families(p):
    value = ((v1_family(p), v3_family(p)) + tuple(t2_family(p, name) for name in T2_FAMILY_SOURCES)
             + (v5_family(p), v6_family(p)))
''', '''def v7_family(p):
    """CELLS_V7 students outside CELLS_V3, CELLS_V5 and CELLS_V6 (Tool1 trainer v10 t2-m1ur-b05, t2-univn-b05), read only
    through the exact bound CELLS_V7 BENCH4 controller, whose roots, owner, attempts and banks are the v3 controller's own."""
    v7 = bound_module(p, "bench4_v7_prepare7_b4.py", "tool1_bench4_judge_v111_v7_controller")
    v3 = bound_module(p, "bench4_v3_prepare4.py", "tool1_bench4_judge_v111_v3_controller_for_v7")
    require(v7.COMPLETE_STATUS == GENERATION_COMPLETE and tuple(v7.BANKS) == BANKS
            and v7.SOURCES["banks4.py"] == SOURCES["banks4.py"]
            and v7.SOURCES["CELLS_V3.json"] == SOURCES["CELLS_V7.json"]
            and (str(v7.CONTROL_ROOT), v7.OWNER, str(v7.OUTPUT_ROOT), v7.MAX_ATTEMPTS)
            == (str(v3.CONTROL_ROOT), v3.OWNER, str(v3.OUTPUT_ROOT), v3.MAX_ATTEMPTS),
            "CELLS_V7 BENCH4 generation semantics differ")
    cells = (set(v7.bound_cells(p)["cells"]) - set(v3.bound_cells(p)["cells"]) - set(cells_v5(p))
             - set(cells_v6(p)))
    return dict(name="bench4-v3c7", control=Path(v7.CONTROL_ROOT), owner=v7.OWNER, output_root=Path(v7.OUTPUT_ROOT),
                attempts=v7.MAX_ATTEMPTS, cells=sorted(cells))


def cells_v7(p):
    cells = json.loads(p.checked(SOURCES["CELLS_V7.json"]))
    require(cells.get("schema") == "tool1-baseline-cells-v7-extsuite-custody-v1" and isinstance(cells.get("cells"), dict),
            "Exact CELLS_V7 roster required")
    return sorted(cells["cells"])


def families(p):
    value = ((v1_family(p), v3_family(p)) + tuple(t2_family(p, name) for name in T2_FAMILY_SOURCES)
             + (v5_family(p), v6_family(p), v7_family(p)))
''')
rep('''    """The newest acknowledged attempt of a CELLS_V5/CELLS_V6 cell's BENCH4 generation, made by that version's pinned
    BENCH4 controller (a cell is registered in exactly one of the two)."""
    family = family_of(families_, cell)
    in5, in6 = cell in cells_v5(p), cell in cells_v6(p)
    require(family["name"] in V5_FAMILIES and in5 != in6, "Afterok judging needs a CELLS_V5 or CELLS_V6 generation: " + cell)
    schema, name, key, roster = ((V6_GENERATION_SCHEMA, "prepare6_b4.py", "bench4_v6_prepare6_b4.py", "CELLS_V6.json")
                                 if in6 else (V5_GENERATION_SCHEMA, "prepare5_b4.py", "bench4_v5_prepare5_b4.py",
                                              "CELLS_V5.json"))
''', '''    """The newest acknowledged attempt of a CELLS_V5/CELLS_V6/CELLS_V7 cell's BENCH4 generation, made by that version's
    pinned BENCH4 controller (a cell is registered in exactly one of the three)."""
    family = family_of(families_, cell)
    in5, in6, in7 = cell in cells_v5(p), cell in cells_v6(p), cell in cells_v7(p)
    require(family["name"] in V5_FAMILIES and [in5, in6, in7].count(True) == 1,
            "Afterok judging needs a CELLS_V5, CELLS_V6 or CELLS_V7 generation: " + cell)
    schema, name, key, roster = ((V7_GENERATION_SCHEMA, "prepare7_b4.py", "bench4_v7_prepare7_b4.py", "CELLS_V7.json")
                                 if in7 else
                                 (V6_GENERATION_SCHEMA, "prepare6_b4.py", "bench4_v6_prepare6_b4.py", "CELLS_V6.json")
                                 if in6 else (V5_GENERATION_SCHEMA, "prepare5_b4.py", "bench4_v5_prepare5_b4.py",
                                              "CELLS_V5.json"))
''')
rep('''            "Generation must be the pinned CELLS_V5/CELLS_V6 BENCH4 controller's own attempt")''',
    '''            "Generation must be the pinned CELLS_V5/CELLS_V6/CELLS_V7 BENCH4 controller's own attempt")''')
rep('''and set(cells) <= set(campaign) and set(cells) <= set(cells_v5(p)) | set(cells_v6(p))''',
    '''and set(cells) <= set(campaign) and set(cells) <= set(cells_v5(p)) | set(cells_v6(p)) | set(cells_v7(p))''')
rep('''                        generation_controller={cell: SOURCES["bench4_v6_prepare6_b4.py" if cell in cells_v6(p)
''', '''                        generation_controller={cell: SOURCES["bench4_v7_prepare7_b4.py" if cell in cells_v7(p)
                                                             else "bench4_v6_prepare6_b4.py" if cell in cells_v6(p)
''')
rep('''                        cells_v5=SOURCES["CELLS_V5.json"], cells_v6=SOURCES["CELLS_V6.json"],''',
    '''                        cells_v5=SOURCES["CELLS_V5.json"], cells_v6=SOURCES["CELLS_V6.json"],
                        cells_v7=SOURCES["CELLS_V7.json"],''')
new_doc = '''"""Registered BENCH4 GPT-judge step v1.11 for Tool1 (MM-Vet + LLaVA-Bench-in-the-Wild), judge kind only.

v1.11 (2026-11-10) is v1.10 (helper_tool1_bench4_judge_v110_20261109/score_step4_v110.py, unchanged) with one
custody-only change for the CELLS_V7 extended-suite producers t2-m1ur-b05 and t2-univn-b05 (Tool1 dispatch, 2026-11-10):
  * a ninth generation family, bench4-v3c7 = the CELLS_V7 cells outside CELLS_V3, CELLS_V5 and CELLS_V6, read through
    the pinned CELLS_V7 BENCH4 controller (helper_tool1_cells_v7_extsuite_20261110/prepare7_b4.py), whose roots, owner,
    attempts and banks are the v3 controller's own. Campaign roster order v1 + v3 + Tool2 v1..v4 + v3c5 + v3c6 + v3c7,
    so every cell's PRIMARY (rank 0) precedes any SUPPLEMENTARY.
  * the afterok plan mode accepts CELLS_V5, CELLS_V6 or CELLS_V7 generations, each bound to its own version's
    controller, plan schema and roster pin (generation_controller is recorded per cell).
  * version identity: SOURCE, CONTROL scoring-judge-v111, owner, statuses, comment, job prefix dsmm-t1-b4j-v111-; v1.10
    is retired by RETIRED.json (sealed by v1.11 `retire --version v1.10` after v1.10's own `withdraw` of its PARTIAL
    budget-stopped batch-0, whose PRIMARY scores stay valid evidence) and joins the foreign judge controls.
Judge policy file judge4_v13.py, prompts, models (PRIMARY gpt-4o-2024-08-06, SUPPLEMENTARY gpt-5.6-terra, no
CALIBRATION), request parameters, prices, the SAME shared append-only ledger, STOP_USD 80.00 / HARD_BUDGET_USD 81.29,
single live judge writer and every receipt schema are unchanged. Text below this paragraph is the v1.10 description.

'''
assert doc.startswith('"""Registered BENCH4 GPT-judge step v1.10 for Tool1')
(OUT / "score_step4_v111.py").write_text(new_doc + doc[3:] + code)
sb = (SRC / "score_run_v110.sbatch").read_text()
size, sha = pin(OUT / "score_step4_v111.py")
old_size, old_sha = pin(SRC / "score_step4_v110.py")
for old, new in ((H + "/experiments/helper_tool1_bench4_judge_v110_20261109/score_step4_v110.py",
                  H + "/experiments/helper_tool1_bench4_judge_v111_20261110/score_step4_v111.py"),
                 ("scoring-judge-v110", "scoring-judge-v111"), ("SOURCE_SIZE = %d" % old_size, "SOURCE_SIZE = %d" % size),
                 ('"%s"' % old_sha, '"%s"' % sha),
                 ("# BENCH4 GPT-judge v1.10 (judge4_v13.py via score_step4_v110.py; adds the CELLS_V6 family).",
                  "# BENCH4 GPT-judge v1.11 (judge4_v13.py via score_step4_v111.py; adds the CELLS_V7 family).")):
    assert sb.count(old) == 1, old
    sb = sb.replace(old, new)
assert "v110" not in sb and "V110" not in sb
(OUT / "score_run_v111.sbatch").write_text(sb)
for name in ("score_step4_v111.py", "score_run_v111.sbatch"):
    print("%-22s %7d %s" % ((name,) + pin(OUT / name)))
