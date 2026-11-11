"""Derive make_forks7.py, score7.template.py and test_cells_v7.py (test is further hand-edited) from the executed,
independently reviewed CELLS_V6 generators: v7_tokens.apply plus exact-count edits for the cell list and the
non-colliding score adapter name score7.py (the bench6 v3 step key "score6.py" is preserved)."""
import sys
from v7_tokens import apply

V6 = sys.argv[1]


def derive(src, dst, reps):
    text = apply(open(V6 + "/" + src).read())
    for old, new, count in reps:
        assert text.count(old) == count, (dst, old[:90], text.count(old), count)
        text = text.replace(old, new)
    open(dst, "w").write(text)


derive("make_forks6.py", "out/make_forks7.py", [
    ('"""Generate the CELLS_V7 custody-only forks from the exact executed sources (exact-count string replacements).\n',
     '"""Generate the CELLS_V7 custody-only forks from the exact executed sources (exact-count string replacements).\n'
     'make_forks6.py through v7_tokens.py plus: the two-cell list and the score adapter score6.py -> score7.py.\n', 1),
    ('CELLS = ["t2-m1ur-b20"]', 'CELLS = ["t2-m1ur-b05", "t2-univn-b05"]', 1),
    ('(OUT / "score6.py").write_text(template)', '(OUT / "score7.py").write_text(template)', 1),
    ('size, sha = pin("score6.py")', 'size, sha = pin("score7.py")', 1),
    ('"%s/experiments/%s/score6.py" % (H, V5)', '"%s/experiments/%s/score7.py" % (H, V5)', 1),
    ('for name in ("score6.py", "score7_run.sbatch"):', 'for name in ("score7.py", "score7_run.sbatch"):', 1),
])
derive("score6.template.py", "out/score7.template.py", [
    ("  score6.py plan ", "  score7.py plan ", 1),
    ("  score6.py self-review ", "  score7.py self-review ", 1),
    ("  score6.py submit|release ", "  score7.py submit|release ", 1),
    ('sources=dict({"score6.py": p.descriptor(SOURCE / "score6.py"),',
     'sources=dict({"score7.py": p.descriptor(SOURCE / "score7.py"),', 1),
    ('score = s.bound_module(p, "score6.py", "tool1_bench6_score_cells_v7_worker")',
     'score = s.bound_module(p, "score6.py", "tool1_bench6_score_cells_v7_worker")', 1),
])
derive("test_cells_v6.py", "out/test_cells_v7.py", [])
