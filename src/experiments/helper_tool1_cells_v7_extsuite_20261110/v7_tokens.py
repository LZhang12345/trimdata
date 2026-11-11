"""Exact V6 -> V7 token map used to derive the CELLS_V7 generators/custody from the reviewed CELLS_V6 ones.

The score adapter is NOT renamed by a bare "score6" token: "score6.py" is also the BENCH6 v3 step's own source key
(ADDENDUM_SCORE6_BENCH6_SOURCE_KEY_20261109.json), so the adapter becomes the non-colliding score7.py only through
the explicit, counted replacements in make_forks7.py."""
TOKENS = [
    ("helper_tool1_cells_v6_extsuite_20261109", "helper_tool1_cells_v7_extsuite_20261110"),
    ("helper-tool1-cells-v6-inventory-v1-20261109", "helper-tool1-cells-v7-inventory-v1-20261110"),
    ("CELLS_V6", "CELLS_V7"), ("CELLS-V6", "CELLS-V7"), ("cells-v6", "cells-v7"), ("cells_v6", "cells_v7"),
    ("custody6", "custody7"), ("admit6", "admit7"), ("inv_run6", "inv_run7"), ("prepare6_", "prepare7_"),
    ("vz_run6", "vz_run7"), ("b6_run6", "b6_run7"), ("b4_run6", "b4_run7"), ("score6_run", "score7_run"),
    ("score6.template", "score7.template"),
    ("dsmm-t1v6-", "dsmm-t1v7-"), ("custody-v6", "custody-v7"), ("CUSTODY_V6", "CUSTODY_V7"),
    ("scoring-afterok-v6", "scoring-afterok-v7"), ("v6_custody", "v7_custody"), ("v3c6", "v3c7"),
    ("dsmm-t1-score-v6a:", "dsmm-t1-score-v7a:"), ("(2026-11-09)", "(2026-11-10)")]


def apply(text):
    for old, new in TOKENS:
        text = text.replace(old, new)
    return text
