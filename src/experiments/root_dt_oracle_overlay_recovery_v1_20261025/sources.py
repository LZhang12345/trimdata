"""Frozen additive runtime closure, separate from unchanged measurement198/native85."""
from pathlib import Path
from experiments.root_native_campaign_20261023 import common as c
from experiments.root_native_campaign_v7_20261024 import common as native91
ROOT=Path(__file__).resolve().parents[2]

def sources(original):
    result=dict(original['sources']);result.update(native91.sources())
    packages=('root_dt_oracle_overlay_recovery_v1_20261025','root_s7_dt_composite_general_v1_20261025',
        'copilot7_datatailor_composite_consumer_20261025','root_native_qualification_v6_20261024',
        'root_dt_mask_row_diagnostic_v1_20261025','root_native_campaign_v5_20261024')
    for package in packages:
        for path in (ROOT/'experiments'/package).iterdir():
            if path.suffix in ('.py','.sbatch') and not path.name.startswith('test_'):
                result[str(path.relative_to(ROOT))]=c.sha(path)
    for relative,digest in result.items():c.require(c.sha(ROOT/relative)==digest,'Existing frozen dependency changed: '+relative)
    return dict(sorted(result.items()))
