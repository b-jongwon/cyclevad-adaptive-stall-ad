"""Check follow-up provenance, out-of-fold coverage and saved-score metrics."""
import json
from pathlib import Path
import numpy as np
from cycle_vad.metrics import frame_metrics
ROOT=Path(__file__).resolve().parents[1]
count=0;metric_count=0
for result_root in [ROOT/'results/E8']+sorted((ROOT/'results/E8').glob('seed*')):
    local=ROOT/'runs/followup'/('' if result_root.name=='E8' else result_root.name)
    for scene in ['R01','R02','R03','R04']:
        evaluated=[]
        for fold in range(5):
            folder=result_root/scene/f'fold{fold}'
            if not (folder/'run.json').exists():continue
            report=json.loads((folder/'run.json').read_text())
            assert report['status']=='complete' and report['legacy_refit_equivalence_passed'] and report['legacy_score_equivalence_passed']
            ids=[i for part in report['splits'].values() for i in part]
            assert len(ids)==len(set(ids))
            evaluated.extend(report['splits']['normal_evaluation'])
            ident=json.loads(report['encoder_identity'])
            assert ident['device']=='cuda' and ident['resolved_revision']=='f9e44c814b77203eaa57a6bdbbd535f21ede1415'
            assert ident['seed']==42 and ident['patch_grid']==6 and ident['image_size']==336
            per_video=json.loads((folder/'per_video.json').read_text())
            metric=json.loads((folder/'metrics.json').read_text())
            for m in metric:
                rows=[r for r in per_video if all(r[k]==m[k] for k in ('partition','cell','readout'))]
                ys=[];scores=[];fprs=[]
                for r in rows:
                    _,part,seq=r['id'].split('/')
                    with np.load(local/scene/f'fold{fold}'/'scores'/f'{part}_{seq}.npz') as d:
                        y=d['labels'];score=d[f'{m["cell"]}__{m["readout"]}']
                        ys.append(y);scores.append(score)
                    actual=frame_metrics(y,score,m['threshold'])
                    for k,v in actual.items():
                        if v is None:assert r[k] is None
                        else:np.testing.assert_allclose(r[k],v,rtol=1e-10,atol=1e-10)
                    metric_count+=1
                actual=frame_metrics(np.concatenate(ys),np.concatenate(scores),m['threshold'])
                for k,v in actual.items():
                    if v is None:assert m[k] is None
                    else:np.testing.assert_allclose(m[k],v,rtol=1e-10,atol=1e-10)
            if report['config']['model']['seed'] != 42:
                primary=json.loads((ROOT/f'results/E8/{scene}/fold{fold}/run.json').read_text())
                for k in ('selected_harmonics','selected_ridge'):
                    assert primary['diagnostics'][k]==report['diagnostics'][k]
                assert primary['splits']==report['splits'] and primary['cache_signatures']==report['cache_signatures']
            count+=1
        assert len(evaluated)==len(set(evaluated))
print(f'PASS: {count} completed scene-folds; {metric_count} per-video/readout metrics recomputed from saved scores.')
print('PASS: disjoint five-way splits, frozen CUDA feature identity, seed robustness hyperparameters and cache signatures.')
print('Independent cycle annotations: absent; no tracking accuracy claimed. Historical tests remain exploratory.')
