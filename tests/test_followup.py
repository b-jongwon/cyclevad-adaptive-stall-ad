import json
from pathlib import Path
import numpy as np
import pytest
from cycle_vad.followup import audit_folds, compose, refit_mean
from cycle_vad.model import descriptor
from test_cycle_vad import fitted, synthetic


def test_confidence_factorization_and_trace_causality(fitted):
    d = synthetic(101)
    a = fitted.tracker.predict(descriptor(d), d['indices'])
    d['global'][32:] += 100
    b = fitted.tracker.predict(descriptor(d), d['indices'])
    np.testing.assert_allclose(a['confidence'], a['location_confidence']*a['template_agreement'])
    for name in a:
        np.testing.assert_allclose(a[name][:32], b[name][:32], atol=1e-6)
    assert np.all((a['location_confidence'] >= 0) & (a['location_confidence'] <= 1+1e-12))


def test_fit_intervention_shares_tracker_appearance_and_hyperparameters(fitted):
    data = [synthetic(i) for i in range(4)]
    cycles = [fitted.tracker.predict(descriptor(d), d['indices']) for d in data]
    for kind in ('legacy', 'location_only', 'uniform'):
        arm = refit_mean(fitted, data, cycles, kind)
        assert arm.tracker is fitted.tracker and arm.pooled is fitted.pooled and arm.local is fitted.local
        assert (arm.harmonics, arm.ridge) == (fitted.harmonics, fitted.ridge)
        assert arm.conditional is not fitted.conditional
        if kind == 'legacy':
            np.testing.assert_allclose(arm.coef, fitted.coef, rtol=1e-6, atol=1e-7)


def test_gate_removal_rescues_only_conditional_channel():
    c = {k: np.array([2.]) for k in ('pooled','pooled_inside','local','alignment','innovation','progress')}
    c.update(conditional=np.array([10.]), conditional_inside=np.array([5.]))
    trace = {'confidence': np.array([.1]), 'location_confidence': np.array([.5])}
    scores = [compose(c, trace, g)[0] for g in range(3)]
    assert [s['conditional_only'][0] for s in scores] == [1,5,10]
    assert [s['appearance_conditional'][0] for s in scores] == [2,5,10]


def test_all_locked_folds_cover_normal_files_without_leakage():
    manifest = json.loads((Path(__file__).parents[1]/'configs/experiments/followup/normal_folds.json').read_text())
    rows = [{'id': i, 'scene': s, 'partition': 'training'} for s, folds in manifest['scenes'].items()
            for ids in folds[0]['splits'].values() for i in ids]
    assert len(audit_folds(manifest, rows)) == 20
    manifest['scenes']['R01'][0]['splits']['fit'].append(manifest['scenes']['R01'][0]['splits']['normal_evaluation'][0])
    with pytest.raises(ValueError, match='Overlap'):
        audit_folds(manifest, rows)
