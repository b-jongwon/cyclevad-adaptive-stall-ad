"""Validate completed experiment artifacts before publication."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    files=list((ROOT/'results').glob('E*/**/metrics.json'))
    assert files
    for p in files:
        m=json.loads(p.read_text()); protocols=p.parent/'protocol.json'
        protocol=json.loads(protocols.read_text())
        splits=protocol['splits']
        fixed=json.loads((ROOT/'results/E0/splits.json').read_text())[p.parent.name]
        assert splits==fixed,(p,'Normal split changed across variants/seeds')
        ids=[set(x) for x in splits.values()]
        assert all(not (a&b) for i,a in enumerate(ids) for b in ids[i+1:])
        assert all('/training/' in x for a in ids for x in a)
        counts={(v['frames'],v['valid_frames'],v['unknown_frames'],v['anomaly_frames'],v['events']) for v in m.values()}
        assert len(counts)==1,(p,counts)
        for name,r in m.items():
            assert r['valid_frames']+r['unknown_frames']==r['frames']
            for k in ['auroc','ap','fpr','recall','event_coverage']:
                assert r[k] is None or 0<=r[k]<=1,(p,name,k)
            assert np.isfinite(r['threshold'])
        if 'thresholds' in protocol:
            assert all(protocol['thresholds'][k]==v['threshold'] for k,v in m.items())
    for p in (ROOT/'results').glob('E2/R*/metrics.json'):
        m=json.loads(p.read_text()); orig=json.loads((ROOT/'runs/stride2_seed42'/p.parent.name/'metrics.json').read_text())
        for new,old in [('full','combined'),('appearance','appearance'),('cycle_conditioned','cycle_conditioned')]:
            for key in ['auroc','ap','fpr','recall','threshold']:
                np.testing.assert_allclose(m[new][key],orig[old][key],rtol=1e-6,atol=1e-7)
    for p in (ROOT/'results').glob('E4/R*/metrics.json'):
        m=json.loads(p.read_text()); original=json.loads((ROOT/'results/E2'/p.parent.name/'metrics.json').read_text())
        for v in original:
            assert m[v]==original[v],(p,v,'Shared core outputs changed')
    print(f'Validated {len(files)} scene-result files: split disjointness, common label masks, thresholds, metric ranges, original-branch agreement.')

if __name__=='__main__': main()
