"""Recheck public E8B metrics and replay representative E9S edits from artifacts."""
import argparse,json
from pathlib import Path
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
from cycle_vad.metrics import frame_metrics
from cycle_vad.data import hold,fingerprint
from cycle_vad.model import descriptor
from cycle_vad.process_experiment import edit_map,dynamics,combine,window_max,starts
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def check_e8():
    models=0;checks=0
    for p in sorted((ROOT/'results/E8B').glob('seed*/R*/fold*/run.json')):
        r=read(p);seed,scene,fold=r['seed'],r['scene'],r['fold'];vs=read(p.with_name('per_video.json'));ms=read(p.with_name('metrics.json'))
        assert r['status']=='complete' and r['M0_fusion_equivalence'] and r['M3_mean_equivalence']
        ids=sum(r['splits'].values(),[]);assert len(ids)==len(set(ids))
        primary=read(ROOT/'results/E8'/scene/f'fold{fold}'/'run.json')
        for m in r['models']:
            if m['arm'].split('_')[0]!='M0' and not m['arm'].endswith('retuned'):
                assert m['harmonics']==primary['diagnostics']['selected_harmonics'] and m['ridge']==primary['diagnostics']['selected_ridge']
        if seed==42:
            rank={m['arm']:m['rank'] for m in r['models']};assert rank['M0_fixedrank']==rank['M3_fixedrank']
        bank={}
        for i in set(v['id'] for v in vs):
            _,part,seq=i.split('/')
            with np.load(ROOT/'runs/E8B'/f'seed{seed}'/scene/f'fold{fold}'/'scores'/f'{part}_{seq}.npz') as d:bank[i]=dict(d)
        for m in ms:
            rows=[v for v in vs if all(v[k]==m[k] for k in ['arm','readout','partition'])];ys=[];ss=[]
            for v in rows:
                d=bank[v['id']];y=d['labels'];s=d[f'{m["arm"]}__{m["readout"]}'];actual=frame_metrics(y,s,m['threshold']);ys.append(y);ss.append(s)
                for k,val in actual.items():
                    if val is None:assert v[k] is None
                    else:np.testing.assert_allclose(v[k],val,rtol=1e-10,atol=1e-10)
                checks+=1
            actual=frame_metrics(np.concatenate(ys),np.concatenate(ss),m['threshold'])
            for k,val in actual.items():
                if val is None:assert m[k] is None
                else:np.testing.assert_allclose(m[k],val,rtol=1e-10,atol=1e-10)
        models+=len(r['models'])
    assert models==640,models
    print(f'PASS E8B: 60 scene-fold-seed units, {models} arm models, {checks} per-video/readout metrics replayed.')
    print('PASS E8B: disjoint splits, frozen hyperparameters across seeds/arms, exact common fixed rank, M0/M3 equivalence checks.')

def check_e9():
    manifest={c['id']:c for c in read(ROOT/'configs/experiments/followup/e9_cases_v2.json')};units=0;replayed=0;normalchecks=0;maxerror=0.
    for p in sorted((ROOT/'results/E9S').glob('stride*/seed*/R*/fold*/run.json')):
        r=read(p);stride,seed,scene,fold=r['stride'],r['seed'],r['scene'],r['fold'];units+=1
        assert r['status']=='complete' and r['identity_max_score_error']<=1e-10
        assert r['protocol_sha256']==fingerprint(read(ROOT/'configs/experiments/followup/phase_process_v2.json'))
        ids=sum(r['splits'].values(),[]);assert len(ids)==len(set(ids))
        folder=ROOT/'runs/E9S'/f'stride{stride}'/f'seed{seed}'/scene/f'fold{fold}';model=joblib.load(folder/'model.joblib');thresholds=np.array(r['window_thresholds']);primary=thresholds[3]
        for v in read(p.with_name('normal.json')):
            d=np.load(folder/f'{v["id"].split("/")[-1]}_normal.npz');score=d['scores'];loc=starts(len(score),r['T'],r['D']);vmax=window_max(score,loc,r['D'])
            np.testing.assert_array_equal(loc,d['grid_starts']);np.testing.assert_array_equal(vmax,d['grid_maxima'])
            np.testing.assert_allclose((vmax[:,None,:]>thresholds[None,:,:]).mean(0),v['grid_alarm_rates'],atol=1e-12)
            np.testing.assert_allclose((score>primary).mean(0),v['frame_fpr_primary'],atol=1e-12);normalchecks+=1
        cs=read(p.with_name('cases.json'));selected=[next(c for c in cs if c['edit']==edit) for edit in ['freeze','reverse','skip','swap_adjacent_blocks']]
        cached={}
        for c in selected:
            spec=manifest[c['id']];i=c['source_id'];assert i in r['splits']['normal_evaluation']
            if i not in cached:
                with np.load(ROOT/'runs/stride1_seed42/cache'/f'{i}.npz') as d:data=dict(d)
                identity=json.loads(str(data['encoder_identity']));assert identity['device']=='cuda' and identity['resolved_revision']=='f9e44c814b77203eaa57a6bdbbd535f21ede1415'
                z=descriptor(data);x=model['tracker'].reducer.transform(z)/model['tracker'].scale
                out,inside,_=model['pooled'].score(z);loc,_=model['local'].score(data['patches'])
                app=np.maximum.reduce([model['appcal'][k].score(v) for k,v in {'pooled':out,'inside':inside,'local':loc}.items()]);cached[i]=(x,app)
            x,app=cached[i];o,L=spec['onset'],spec['length'];mapping=edit_map(len(x),o,L,c['edit']);take=np.arange(0,o+r['W'],stride);source=mapping[take]
            raw,valid=dynamics(model['tracker'],x[source],take,model['ar']);s=hold(take,combine(app[source],raw,valid,model['cal']),o+r['W'])
            for k,actual in [('edited_max_W',s[o:o+r['W']].max(0)),('edited_max_D',s[o:o+r['D']].max(0))]:
                np.testing.assert_allclose(actual,c[k],rtol=1e-5,atol=1e-6,err_msg=c['id']);maxerror=max(maxerror,float(np.max(np.abs(actual-c[k]))))
            for a,delay in enumerate(c['delay_primary']):
                hits=np.flatnonzero(s[o:o+r['D'],a]>primary[a]);assert delay==(int(hits[0]) if len(hits) else None)
            replayed+=1
    assert units==80,units
    print(f'PASS E9S: {units} units, {normalchecks} original normal streams, {replayed} edit replays; maximum score difference {maxerror:.3g}.')
    print('PASS E9S: fixed dense CUDA cache, disjoint normal splits, frozen protocol, identity controls, output-time tracking, saved-window and delay consistency.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['E8B','E9S'],required=True);a=p.parse_args()
    with threadpool_limits(limits=4):
        if a.stage=='E8B':check_e8()
        else:check_e9()
