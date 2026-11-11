"""Derive custody7.py from the executed, independently reviewed custody6.py (exact-count replacements; run locally:
python3 make_custody7.py <custody6.py> <out custody7.py>)."""
import sys
from v7_tokens import apply

src = open(sys.argv[1]).read()
OLD_HEAD = ('"""CELLS_V6 custody: custody5.py (CELLS_V5, independently reviewed 2026-11-08) with only the version identity changed\n'
            '(roster schema, inventory control/output roots, owner, schemas); every kind rule is unchanged. CELLS_V6 registers the\n'
            'single trainer-v10 original producer t2-m1ur-b20 (Tool2 request, 2026-11-09). CELLS_V5 text follows.\n')
assert src.startswith(OLD_HEAD)
NEW_HEAD = (
    '"""CELLS_V7 custody: custody6.py (CELLS_V6, independently reviewed 2026-11-09) with the version identity changed\n'
    '(roster schema, inventory control/output roots, owner, schemas) and one allow-list extension: kind tool1_v10_original\n'
    'may name ROSTER_V7.json or ROSTER_V8.json (both Tool1 trainer v10 rosters; ROSTER_V8 registers Tool2\'s 5% cells).\n'
    'Each cell still binds exactly one pinned roster whose entry must equal its producer PLAN; every other kind rule is\n'
    'unchanged. CELLS_V7 registers the two trainer-v10 original producers t2-m1ur-b05 and t2-univn-b05 (ROSTER_V8; Tool2\n'
    'request, 2026-11-10). CELLS_V6 text follows.\n\n')
body = apply(src[len(OLD_HEAD):])
REPS = [
    ('KIND_ROSTERS = {"tool1_v6_original": "ROSTER_V6.json", "tool1_v8_successor": "ROSTER_V6.json",\n'
     '                "tool1_v10_original": "ROSTER_V7.json"}\n',
     'KIND_ROSTERS = {"tool1_v6_original": ("ROSTER_V6.json",), "tool1_v8_successor": ("ROSTER_V6.json",),\n'
     '                "tool1_v10_original": ("ROSTER_V7.json", "ROSTER_V8.json")}\n'
     'ROSTER_NAMES = ("ROSTER_V6.json", "ROSTER_V7.json", "ROSTER_V8.json")\n', 1),
    ('        require(name in ("ROSTER_V6.json", "ROSTER_V7.json") and is_ref(ref)\n',
     '        require(name in ROSTER_NAMES and is_ref(ref)\n', 1),
    ('spec.get("roster") == KIND_ROSTERS[kind]\n', 'spec.get("roster") in KIND_ROSTERS[kind]\n', 1),
    ('  * tool1_v10_original  Tool1 trainer v10 original job (ROSTER_V7, Tool2 selection',
     '  * tool1_v10_original  Tool1 trainer v10 original job (ROSTER_V7/ROSTER_V8, Tool2 selection', 1),
]
for old, new, count in REPS:
    assert body.count(old) == count, (old[:80], body.count(old))
    body = body.replace(old, new)
assert "KIND_ROSTERS" not in body.replace("KIND_ROSTERS = {", "").replace("KIND_ROSTERS[kind]", "")
open(sys.argv[2], "w").write(NEW_HEAD + OLD_HEAD[3:] + body)
