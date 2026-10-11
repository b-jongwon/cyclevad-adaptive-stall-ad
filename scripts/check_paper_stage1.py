"""Artifact-level validation of unified and subspace experiments."""
import argparse,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from cycle_vad.metrics import frame_metrics
from cycle_vad.paper_experiments import calibrate,maxwindows,event_rows,ARMS
from cycle_vad.paper_subspaces import nested_subset
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def compare(record,actual):
    for k,v in actual.items():
        if v is None:assert record[k] is None,(k,record[k])
        else:np.testing.assert_allclose(record[k],v,atol=1e-10,rtol=1e-10)
def e11():
    ps=sorted((ROOT/'results/E11').glob('seed*/R*/fold*/run.json'));assert len(ps)==60;checks=0;cases=0;prefix=0
    for p in ps:
        r=read(p);assert r['status']=='complete' and r['identity_max_error']<=1e-10 and r['score_prefix_max_error']<=1e-9
        prefix=max(prefix,r['score_prefix_max_error']);ids=sum(r['splits'].values(),[]);assert len(ids)==len(set(ids))
        if r['seed']!=42:
            primary=read(ROOT/'results/E11/seed42'/r['scene']/f'fold{r["fold"]}'/'run.json');assert (r['harmonics'],r['ridge'])==(primary['harmonics'],primary['ridge'])
        base=ROOT/'runs/E11'/f'seed{r["seed"]}'/r['scene']/f'fold{r["fold"]}';vs=read(p.with_name('per_video.json'));bank={}
        for i in set(v['id'] for v in vs):
            _,part,seq=i.split('/')
            with np.load(base/f'{part}_{seq}.npz') as d:bank[i]=dict(d)
        score={i:v['scores'] for i,v in bank.items()};th,ft,counts=calibrate(score,r['splits'],r['T'],r['D'])
        np.testing.assert_array_equal(th,r['window_thresholds']);np.testing.assert_array_equal(ft,r['frame_thresholds']);assert counts==r['threshold_window_counts']
        for v in vs:
            a=ARMS.index(v['arm']);d=bank[v['id']];compare(v,frame_metrics(d['labels'],d['scores'][:,a],th[3,a]));compare(v['frame_q99'],frame_metrics(d['labels'],d['scores'][:,a],ft[a]));checks+=1
        for m in read(p.with_name('metrics.json')):
            selected=[v for v in vs if v['partition']==m['partition'] and v['arm']==m['arm']];a=ARMS.index(m['arm']);y=np.concatenate([bank[v['id']]['labels'] for v in selected]);s=np.concatenate([bank[v['id']]['scores'][:,a] for v in selected]);compare(m,frame_metrics(y,s,th[3,a]))
        for d in bank.values():
            s=d['scores'];np.testing.assert_array_equal(s[:,3],np.maximum.reduce([s[:,0],s[:,4],s[:,5]]))
        for c in read(p.with_name('cases.json')):
            hit=np.array(c['edited_max_D'])>th[3];np.testing.assert_array_equal(hit,[v is not None for v in c['delay_primary']]);cases+=1
    assert cases==11730
    print(f'PASS E11: 60 units, {checks} per-video/readout metrics, {cases} edited cases; thresholds recomputed from normal saved scores.')
    print(f'PASS E11: shared splits/hyperparameters, identity and score prefix invariance; max prefix score difference {prefix:.3g}.')

def e12():
    ps=sorted((ROOT/'results/E12').glob('seed*/R*/fold*/run.json'));assert len(ps)==60;checks=0;heads=0
    for p in ps:
        r=read(p);base=ROOT/'runs/E12'/f'seed{r["seed"]}'/r['scene']/f'fold{r["fold"]}';vs=read(p.with_name('per_video.json'));bank={}
        for i in set(v['id'] for v in vs):
            _,part,seq=i.split('/')
            with np.load(base/f'{part}_{seq}.npz') as d:bank[i]=dict(d)
        for sub in r['subsets']:assert sub['fit_ids']==nested_subset(r['splits']['fit'],sub['fraction'],r['seed'])
        for fraction in [.25,.5,1.]:
            models=[m for m in r['models'] if m['fraction']==fraction and m['status']=='complete'];ranks=[m['actual_total_rank'] for m in models if m['arm']!='S5']
            assert len(set(ranks))<=1
        for m in r['models']:
            if m['status']!='complete':continue
            heads+=1;key=f'{m["arm"]}_f{int(m["fraction"]*100)}';selected=[v for v in vs if v['arm']==m['arm'] and v['fraction']==m['fraction']]
            for v in selected:
                j=0 if v['readout']=='head_only' else 1;d=bank[v['id']];compare(v,frame_metrics(d['labels'],d[key][:,j],m['frame_thresholds'][j]));checks+=1
            threshold=np.quantile(np.concatenate([bank[i][key] for i in r['splits']['threshold']]),.99,axis=0,method='higher');np.testing.assert_array_equal(threshold,m['frame_thresholds'])
        for m in read(p.with_name('metrics.json')):
            selected=[v for v in vs if all(v[k]==m[k] for k in ['arm','fraction','readout','partition'])];j=0 if m['readout']=='head_only' else 1;key=f'{m["arm"]}_f{int(m["fraction"]*100)}';y=np.concatenate([bank[v['id']]['labels'] for v in selected]);s=np.concatenate([bank[v['id']][key][:,j] for v in selected]);compare(m,frame_metrics(y,s,m['threshold']))
    print(f'PASS E12: 60 units, {heads} completed heads, {checks} per-video/readout metrics; frame thresholds recomputed.')
    print('PASS E12: nested FIT-only subsets, identical total ranks across PCA conditions, explicit unavailable status; frozen full-data tracker scope.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['E11','E12'],required=True);a=p.parse_args()
    with threadpool_limits(limits=4):
        if a.stage=='E11':e11()
        else:e12()
