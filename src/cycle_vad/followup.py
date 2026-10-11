"""Locked five-way follow-up splits and confidence fit/gate factorial.

No test label is read until all training, reference scales and thresholds are
fixed. Historical tests are diagnostics; normal evaluation is out of fold.
"""
from copy import copy
from pathlib import Path
import time
import joblib
import numpy as np

from .data import balanced_sample, cycle_segments, hold, labels_for, write_json, fingerprint
from .model import CycleVAD, ResidualSpace, TailCalibrator, descriptor
from .tracker import fourier
from .pipeline import load_rows, code_fingerprint
from .metrics import frame_metrics, event_metrics

READOUTS = ('conditional_only', 'appearance_conditional', 'full')


def audit_folds(folds, rows):
    result = []
    for scene, entries in folds['scenes'].items():
        expected = {r['id'] for r in rows if r['scene'] == scene and r['partition'] == 'training'}
        evaluated = []
        for entry in entries:
            parts = entry['splits']
            flat = [i for ids in parts.values() for i in ids]
            if len(set(flat)) != len(flat) or set(flat) != expected:
                raise ValueError(f'Overlap, missing, or non-normal IDs: {scene}/{entry["fold"]}')
            evaluated.extend(parts['normal_evaluation'])
            result.append({'scene': scene, 'fold': entry['fold'], 'disjoint': True,
                           'sizes': {k: len(v) for k, v in parts.items()}})
        if len(evaluated) != len(expected) or set(evaluated) != expected:
            raise ValueError('Every normal source must be evaluated exactly once')
    return result


def refit_mean(base, data, cycles, weight_kind):
    """Reuse theta, selected Fourier/ridge and appearance model across rows."""
    model = copy(base)
    cfg = model.config
    z = [descriptor(d) for d in data]
    weights = [np.ones(len(c['angle'])) if weight_kind == 'uniform' else
               np.maximum(c['location_confidence' if weight_kind == 'location_only' else 'confidence'], .1)
               for c in cycles]
    rows = balanced_sample([np.concatenate([fourier(c['angle'], model.harmonics), x, w[:, None]], 1).astype(np.float32)
                            for x, c, w in zip(z, cycles, weights)], cfg['fit_samples'], cfg['seed'])
    cols = 1+2*model.harmonics
    b, x, w = rows[:, :cols], rows[:, cols:-1], rows[:, -1]
    penalty = np.eye(cols)*model.ridge*len(b)
    penalty[0, 0] = 1e-8
    model.coef = np.linalg.solve(b.T@(w[:, None]*b)+penalty, b.T@(w[:, None]*x)).astype(np.float32)
    residuals = [x-fourier(c['angle'], model.harmonics)@model.coef for x, c in zip(z, cycles)]
    model.conditional = ResidualSpace().fit(balanced_sample(residuals, cfg['fit_samples'], cfg['seed']),
                                            cfg['subspace_rank'], seed=cfg['seed'])
    return model


def compose(calibrated, cycle, gate):
    appearance = np.maximum.reduce([calibrated[k] for k in ('pooled', 'pooled_inside', 'local')])
    conditional = np.maximum(calibrated['conditional'], calibrated['conditional_inside'])
    g = (cycle['confidence'], cycle['location_confidence'], np.ones(len(conditional)))[gate]
    gated = conditional*g
    ac = np.maximum(appearance, gated)
    process = np.maximum.reduce([calibrated[k] for k in ('alignment', 'innovation', 'progress')])
    return {'conditional_only': gated, 'appearance_conditional': ac, 'full': np.maximum(ac, process)}, appearance, conditional


def quantiles(x):
    return dict(zip(('p50', 'p90', 'p99'), map(float, np.quantile(x, [.5, .9, .99])))) if len(x) else None


def run_fold(scene, entry, rows, config, cache, output, public):
    start = time.perf_counter()
    folder = Path(output)/scene/f'fold{entry["fold"]}'
    folder.mkdir(parents=True, exist_ok=True)
    splits = entry['splits']
    lookup = {r['id']: r for r in rows}
    groups = {k: [lookup[i] for i in ids] for k, ids in splits.items()}
    if entry['fold'] == 0:
        groups['historical_test'] = [r for r in rows if r['scene'] == scene and r['partition'] == 'testing']
    ordered = [r for rs in groups.values() for r in rs]
    data = dict(zip([r['id'] for r in ordered], load_rows(cache, ordered)))
    if any(not np.all(np.diff(d['indices']) == config['stride']) for d in data.values()):
        raise ValueError('Cache stride mismatch')
    fit = [data[i] for i in splits['fit']]
    validation = [data[i] for i in splits['validation']]
    segments = [cycle_segments(data[i], lookup[i])[0] for i in splits['fit']]
    base = CycleVAD(config['model']).fit(fit, validation, segments)
    cycles, raws = {}, {}
    for row in ordered:
        i = row['id']
        raw, cycle, _ = base.raw(data[i])
        raws[i], cycles[i] = raw, cycle
    # Check legacy retraining matches the fitted control; catches sampling/order drift.
    legacy = refit_mean(base, fit, [cycles[i] for i in splits['fit']], 'legacy')
    np.testing.assert_allclose(base.coef, legacy.coef, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(base.conditional.basis, legacy.conditional.basis, rtol=1e-5, atol=1e-6)
    models = [base] + [refit_mean(base, fit, [cycles[i] for i in splits['fit']], kind)
                       for kind in ('location_only', 'uniform')]
    per_video, aggregates, distributions, mechanism = [], [], [], []
    score_bank = {r['id']: {} for r in ordered}
    for train, model in enumerate(models):
        row_raw = {}
        for row in ordered:
            i = row['id']
            raw = dict(raws[i])
            if train:
                res = descriptor(data[i])-fourier(cycles[i]['angle'], model.harmonics)@model.coef
                raw['conditional'], raw['conditional_inside'], _ = model.conditional.score(res)
            row_raw[i] = raw
        calibrators = {k: TailCalibrator().fit(np.concatenate([row_raw[i][k] for i in splits['reference']]))
                       for k in row_raw[splits['fit'][0]]}
        model.calibrators = calibrators
        calibrated = {i: {k: calibrators[k].score(v) for k, v in raw.items()} for i, raw in row_raw.items()}
        for gate in range(3):
            cell = f'C-{train}{gate}'
            streams = {r['id']: compose(calibrated[r['id']], cycles[r['id']], gate) for r in ordered}
            thresholds = {name: float(np.quantile(np.concatenate([
                hold(data[i]['indices'], streams[i][0][name], lookup[i]['frames']) for i in splits['threshold']]),
                config['model']['threshold_quantile'], method='higher')) for name in READOUTS}
            if train == gate == 0:
                model.thresholds = {'combined': thresholds['full'], 'appearance': 0., 'cycle_conditioned': thresholds['appearance_conditional']}
                for i in splits['threshold']:
                    np.testing.assert_allclose(model.score(data[i])['combined'], streams[i][0]['full'], rtol=1e-6, atol=1e-7)
            for partition, selected in groups.items():
                if partition in ('fit', 'validation'):
                    continue
                # All thresholds above were computed without labels.
                ys, collected = [], {name: [] for name in READOUTS}
                local_records = []
                for row in selected:
                    i, d = row['id'], data[row['id']]
                    y = labels_for(row)
                    ys.append(y)
                    for name in READOUTS:
                        values = hold(d['indices'], streams[i][0][name], row['frames'])
                        score_bank[i][f'{cell}__{name}'] = values.astype(np.float64)
                        collected[name].append(values)
                        rec = {'scene': scene, 'fold': entry['fold'], 'partition': partition, 'id': i,
                               'cell': cell, 'readout': name, **frame_metrics(y, values, thresholds[name]),
                               **event_metrics(y, values, thresholds[name])}
                        per_video.append(rec)
                        local_records.append(rec)
                    if train == gate == 0:
                        for name in READOUTS:
                            values = collected[name][-1]
                            distributions.append({'scene': scene, 'fold': entry['fold'], 'partition': partition,
                                'id': i, 'readout': name, 'normal_quantiles': quantiles(values[y == 0])})
                    for target, mask in [('normal', y == 0), ('anomaly', y == 1)]:
                        if not mask.any():
                            continue
                        ungated = hold(d['indices'], streams[i][2], row['frames'])[mask]
                        app = hold(d['indices'], streams[i][1], row['frames'])[mask]
                        gated = collected['conditional_only'][-1][mask]
                        mechanism.append({'scene': scene, 'fold': entry['fold'], 'partition': partition, 'id': i,
                            'cell': cell, 'target': target, 'frames': int(mask.sum()),
                            'raw_outside': quantiles(hold(d['indices'], row_raw[i]['conditional'], row['frames'])[mask]),
                            'raw_inside': quantiles(hold(d['indices'], row_raw[i]['conditional_inside'], row['frames'])[mask]),
                            'calibrated_ungated': quantiles(ungated), 'gated': quantiles(gated),
                            'activation_fraction': float(np.mean(gated > app)),
                            'mean_r': float(hold(d['indices'], cycles[i]['location_confidence'], row['frames'])[mask].mean()),
                            'mean_legacy_confidence': float(hold(d['indices'], cycles[i]['confidence'], row['frames'])[mask].mean())})
                for name in READOUTS:
                    records = [r for r in local_records if r['readout'] == name]
                    fprs = [r['fpr'] for r in records if r['fpr'] is not None]
                    ev = sum(r['events'] for r in records)
                    de = sum(r['detected_events'] for r in records)
                    aggregates.append({'scene': scene, 'fold': entry['fold'], 'partition': partition, 'cell': cell,
                        'readout': name, **frame_metrics(np.concatenate(ys), np.concatenate(collected[name]), thresholds[name]),
                        'video_mean_fpr': float(np.mean(fprs)) if fprs else None,
                        'events': ev, 'detected_events': de, 'event_coverage': de/ev if ev else None})
        joblib.dump(model, folder/f'train{train}.joblib', compress=3)
    for row in ordered:
        i = row['id']
        if score_bank[i]:
            target = folder/'scores'/f'{row["partition"]}_{row["sequence"]}.npz'
            target.parent.mkdir(exist_ok=True)
            np.savez_compressed(target, labels=labels_for(row), **score_bank[i])
        # Save T0/T1/T2 predictions, explicitly without accuracy claims.
        if entry['fold'] == 0:
            c, d = cycles[i], data[i]
            path = folder/'tracker'/f'{row["partition"]}_{row["sequence"]}.npz'
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, indices=d['indices'],
                T0=(c['observation_angle'][0]+(d['indices']-d['indices'][0])/base.tracker.period)%1,
                T1=c['observation_angle'], T2=c['angle'], r=c['location_confidence'], confidence=c['confidence'])
    dest = Path(public)/scene/f'fold{entry["fold"]}'
    for name, value in [('metrics', aggregates), ('per_video', per_video), ('distributions', distributions), ('mechanism', mechanism)]:
        write_json(dest/f'{name}.json', value)
    report = {'scene': scene, 'fold': entry['fold'], 'splits': splits, 'status': 'complete',
              'config': config, 'code_fingerprint': code_fingerprint(), 'diagnostics': base.diagnostics,
              'encoder_identity': str(next(iter(data.values()))['encoder_identity']),
              'cache_signatures': {i: str(d['signature']) for i, d in data.items()},
              'shared_tracker_and_appearance': True, 'legacy_refit_equivalence_passed': True,
              'legacy_score_equivalence_passed': True, 'fourier_selection_frozen_across_rows': True,
              'model_rows': 3, 'cells': 9, 'readouts_per_cell': 3,
              'tracking_accuracy': None, 'historical_tests_diagnostic_only': True,
              'wall_seconds': time.perf_counter()-start}
    write_json(dest/'run.json', report)
    print(f'[complete] {scene} fold={entry["fold"]}: 3 fits, 9 cells, 27 readouts, {report["wall_seconds"]:.1f}s', flush=True)
    return report
