"""Pre-release gate: are the S1 V2 repairs on Root's BOUND execution path?

The V2 fixes live in S1's namespace; they only protect training once Root's
bootstrap/selection modules call them. This gate exercises Root's live modules
on synthetic engineering fixtures (CPU only, no NN) and reports each check:

G1 bootstrap builds S1 controls with controls_v2 (not controls_v1).
G2 selection.validate_selection accepts a valid control-v2 receipt AND rejects
   resealed forged v2 receipts (budget without nominal_requested_K; forged
   covariates), i.e. it actually calls controls_v2.validate (no shallow pass).
G3 selection.validate_selection rejects a legacy control-v1 receipt that does
   NOT depend on source labels (v1 uniform_actualK, independently accepted by
   controls_v1.validate), so a groups_for repair alone cannot satisfy G3.
G4 selection.groups_for rejects V1 labels even in engineering mode (V1 permits
   an unpinned-catalog SHA override while emitting the production status).
G5 selection.groups_for rejects V2 ENGINEERING labels (distinct status).

Preconditions verify the fixtures themselves (forgeries rejected by
controls_v2.validate, legacy receipt accepted by controls_v1.validate); if any
precondition fails the status is GATE_FIXTURE_INVALID, never PASS.
Status PASS_V2_ON_BOUND_PATH only if all pass; otherwise NOT_INTEGRATED. This
script edits nothing and submits nothing; Root owns the wiring.
"""
import json
from pathlib import Path
import tempfile

from experiments.root_native_campaign_v3_20261023 import common as c
from experiments.root_native_campaign_v3_20261023 import selection as sel
from experiments.s1_native_selection_controls_20261023 import controls_v1, controls_v2
from experiments.s1_native_selection_controls_20261023 import source_labels_v1, source_labels_v2
from experiments.s1_native_selection_controls_20261023.test_s1_controls_v1 import (
    catalog_row, make_features, make_pool, make_published, sample_indices)


def _raises(fn):
    try:
        fn()
    except (ValueError, KeyError, TypeError) as error:
        return True, f'{type(error).__name__}: {error}'
    return False, 'accepted'


def check():
    from experiments.root_native_campaign_v3_20261023 import bootstrap
    out = dict(schema='s1-v2-integration-gate-v2', root_sources_sha256={
        p: c.sha(Path(c.HERE) / p) for p in ('bootstrap.py', 'selection.py', 'config.py', 'plan.py')}, checks={})
    out['checks']['G1_bootstrap_uses_controls_v2'] = dict(passed=getattr(bootstrap, 'controls', None) is controls_v2,
                                                          actual=getattr(getattr(bootstrap, 'controls', None), '__name__', None))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = [catalog_row(i) for i in range(source_labels_v2.NOMINAL_N)]
        catalog = root / 'catalog.json'; catalog.write_text(json.dumps(rows)); sha = c.sha(catalog)
        indices = sample_indices()
        pool, ids, prow = make_pool(root, indices, rows)
        feats, assets = make_features(root, pool, ids, prow)
        v2_labels = source_labels_v2.produce(pool, catalog, root / 'l2', allow_engineering=True, engineering_catalog_sha=sha)
        v1_labels = source_labels_v1.produce(pool, catalog, root / 'l1', allow_engineering=True, expected_sha=sha)
        v2_sel = controls_v2.build(pool, 'source_loss_low', '0.10', 1234, root / 's2', v2_labels, feats, assets, allow_engineering=True)
        good = c.checked(v2_sel)
        budget = {k: v for k, v in good['budget'].items() if k != 'nominal_requested_K'}
        cov = json.loads(json.dumps(good['covariates']))
        first = sorted(cov)[0]
        cov[first] = (cov[first] + 1) if isinstance(cov[first], (int, float)) and not isinstance(cov[first], bool) else 'forged'
        forged = {'budget_no_nominal_K': c.seal(root / 'f_budget' / 'SELECTION.json', dict(good, budget=budget)),
                  'covariates_forged': c.seal(root / 'f_cov' / 'SELECTION.json', dict(good, covariates=cov))}
        pub = make_published(root, pool, ids, prow, '0.05')
        v1_uk = controls_v1.build(pool, 'uniform_actualK', '0.05', 1234, root / 'uk1', published_desc=pub, allow_engineering=True)
        pre = {}
        for name, desc in forged.items():
            pre[f'P_controls_v2_rejects_{name}'] = dict(zip(('passed', 'detail'), _raises(
                lambda d=desc: controls_v2.validate(d, pool, 'source_loss_low', '0.10', allow_engineering=True))))
        ok, why = _raises(lambda: controls_v1.validate(v1_uk, pool, 'uniform_actualK', '0.05', allow_engineering=True))
        pre['P_controls_v1_accepts_label_free_legacy_receipt'] = dict(passed=not ok, detail=why)
        out['preconditions'] = pre
        bad, why = _raises(lambda: sel.validate_selection(v2_sel, pool, 'source_loss_low', '0.10', allow_engineering=True))
        out['checks']['G2a_validate_selection_accepts_control_v2'] = dict(passed=not bad, detail=why)
        for name, desc in forged.items():
            bad, why = _raises(lambda d=desc: sel.validate_selection(d, pool, 'source_loss_low', '0.10', allow_engineering=True))
            out['checks'][f'G2_validate_selection_rejects_forged_v2_{name}'] = dict(passed=bad, detail=why)
        bad, why = _raises(lambda: sel.validate_selection(v1_uk, pool, 'uniform_actualK', '0.05', allow_engineering=True))
        out['checks']['G3_validate_selection_rejects_label_free_control_v1'] = dict(passed=bad, detail=why)
        # Engineering mode, so rejection must come from the label schema, not the production pool gate.
        bad, why = _raises(lambda: sel.groups_for(v1_labels, pool, ids, allow_engineering=True))
        out['checks']['G4_groups_for_rejects_v1_unpinned_override_labels'] = dict(passed=bad, detail=why)
        bad, why = _raises(lambda: sel.groups_for(v2_labels, pool, ids))
        out['checks']['G5_groups_for_rejects_v2_engineering_labels'] = dict(passed=bad, detail=why)
    if not all(v['passed'] for v in out['preconditions'].values()):
        out['status'] = 'GATE_FIXTURE_INVALID'
    else:
        out['status'] = 'PASS_V2_ON_BOUND_PATH' if all(v['passed'] for v in out['checks'].values()) else 'NOT_INTEGRATED'
    return out


if __name__ == '__main__':
    print(json.dumps(check(), indent=1))
