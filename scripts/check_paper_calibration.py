"""Check fixed-arm reproduction and all saved event/window decisions."""
import argparse
from pathlib import Path
import numpy as np
from cycle_vad.paper_calibration import read,load_bank,deserialize_thresholds,alarms,SCENES
ROOT=Path(__file__).resolve().parents[1]
def b0():
    paths=sorted((ROOT/'results/E15B0').glob('seed*/R*/fold*/run.json'));assert len(paths)==60
    events_count=0
    for p in paths:
        r=read(p);rel=p.parent.relative_to(ROOT/'results/E15B0');data=load_bank(ROOT,rel);th=deserialize_thresholds(r['thresholds'])
        old=read(ROOT/'results/E11B'/rel/'run.json');np.testing.assert_array_equal(th[:,1,0],np.array(old['window_thresholds'])[[3,1,0],6])
        saved={e['id']:e for e in read(ROOT/'results/E11B'/rel/'historical_events.json')}
        for e in read(p.with_name('events.json')):
            h=np.array(e['hit'],bool);d=np.array(e['delay']);o=saved[e['id']]
            assert (d[0,0] if d[0,0]>=0 else None)==o['delay_primary'][4]
            assert (d[0,1] if d[0,1]>=0 else None)==o['delay_primary'][6]
            assert np.all(~h[:,0]|h[:,4]);events_count+=1
        for n in read(p.with_name('normal.json')):
            o=n['overlap_q99'];f=np.array(n['grid_far'])[0]
            np.testing.assert_allclose(f[4],f[0]+o['P_only'],atol=1e-14)
            baseline=next(v for v in read(ROOT/'results/E11B'/rel/'normal.json') if v['id']==n['id'])
            np.testing.assert_array_equal(np.array(n['grid_far'])[:,0],np.array(baseline['grid_alarm_rates'])[[3,1,0],4])
            np.testing.assert_array_equal(np.array(n['grid_far'])[:,1],np.array(baseline['grid_alarm_rates'])[[3,1,0],6])
    assert events_count==198
    text=f'PASS E15B0: 60 units, {events_count} seed-specific historical events; original C/CP delays and normal grid alarms reproduced; OR preservation and normal cost identity verified.\n'
    (ROOT/'results/E15B0/validation.txt').write_text(text);print(text)
def bc():
    import joblib
    from cycle_vad.paper_calibration import pool_windows,policy_thresholds,VideoEqualTail,branches
    from cycle_vad.paper_diagnostics import sha
    from cycle_vad.data import fingerprint
    from cycle_vad.process_experiment import starts
    paths=sorted((ROOT/'results/E15BC').glob('seed*/R*/fold*/run.json'));assert len(paths)==60
    case_count=0;frame_records=0;event_count=0;max_error=0.
    cfg=read(ROOT/'configs/experiments/followup/e15bc_v1.json')
    for p in paths:
        r=read(p);rel=p.parent.relative_to(ROOT/'results/E15BC');out=ROOT/'runs/E15BC'/rel
        checkpoint=ROOT/'runs/E11'/rel/'model.joblib';assert r['frozen_checkpoint_sha256']==sha(checkpoint)
        assert r['config_sha256']==fingerprint(cfg) and r['baseline_sha256']==sha(ROOT/'results/E11B'/rel/'run.json')
        assert r['replay_error']<=1e-9 and r['prefix_error']<=1e-9;max_error=max(max_error,r['replay_error'],r['prefix_error'])
        m=joblib.load(checkpoint);original=load_bank(ROOT,rel,True)
        frozen=m['ccal']+[m['pcal'][k] for k in ['alignment','innovation','progress']]
        weighted=[VideoEqualTail().fit([original[i]['raw'][::2,3+j] for i in r['splits']['reference']],frozen[j]) for j in range(5)]
        ids=sum(r['splits'].values(),[]);assert len(ids)==len(set(ids))
        bank={}
        for f in out.glob('*.npz'):
            if f.stem=='edited_raw':continue
            i=f'{r["scene"]}/{f.stem.replace("_","/",1)}'
            with np.load(f) as d:bank[i]=dict(d)
            np.testing.assert_array_equal(bank[i]['labels'],original[i]['labels'])
            np.testing.assert_array_equal(bank[i]['scores'][0],original[i]['scores'][:,[4,5]])
            np.testing.assert_allclose(bank[i]['scores'][1],branches(original[i]['raw'][:,3:],weighted),atol=1e-12)
        th=np.array([deserialize_thresholds(t) for t in r['thresholds']])
        for g in range(2):
            v,w,c=pool_windows({i:d['scores'][g] for i,d in bank.items()},r['splits']['threshold'],r['T'],r['D'])
            np.testing.assert_array_equal(policy_thresholds(v,w),th[g]);assert c==r['threshold_window_counts']
        for n in read(p.with_name('normal.json')):
            d=bank[n['id']];assert np.all(d['labels']==0)
            a=(d['scores'][n['g'],:,None,None,:]>th[n['g']]).any(-1)
            grid=np.array([a[o:o+r['D']].any(0) for o in starts(len(a),r['T'],r['D'])])
            np.testing.assert_array_equal(grid.mean(0),n['grid_far']);np.testing.assert_array_equal(a.mean(0),n['frame_fpr'])
        for e in read(p.with_name('events.json')):
            d=bank[e['source_id']];g=e['g'];assert np.all(d['labels'][e['onset']:e['end']]==1)
            a=(d['scores'][g,e['onset']:min(e['end'],e['onset']+r['D']),None,None,:]>th[g]).any(-1)
            np.testing.assert_array_equal(a.any(0),e['hit']);np.testing.assert_array_equal(np.where(a.any(0),a.argmax(0),-1),e['delay']);event_count+=1
        for f in read(p.with_name('frames.json')):
            d=bank[f['id']];y=d['labels'];prior=np.cumsum(y==1)>0
            masks={'normal':y==0,'anomaly':y==1,'normal_before':(y==0)&~prior,'normal_after':(y==0)&prior}
            mask=masks[f['stratum']];assert int(mask.sum())==f['frames']
            a=(d['scores'][f['g'],mask,None,None,:]>th[f['g']]).any(-1);np.testing.assert_array_equal(a.sum(0),f['alarm_count']);frame_records+=1
        lookup={(c['id'],c['g']):c for c in read(p.with_name('cases.json'))}
        specs={c['id']:c for c in read(ROOT/'configs/experiments/followup/e9_cases_v2.json') if c['eligible']}
        with np.load(out/'edited_raw.npz') as d:
            for j,identity in enumerate(d['ids']):
                for g,cal in enumerate([frozen,weighted]):
                    np.testing.assert_allclose(branches(d['raw'][j],cal),d['scores'][j,g],atol=1e-9,rtol=1e-9)
                    spec=specs[str(identity)];c=lookup[(str(identity),g)];score=d['scores'][j,g]
                    a=(score[:,None,None,:]>th[g]).any(-1);hit=a[:r['D']].any(0)
                    np.testing.assert_array_equal(hit,c['hit']);np.testing.assert_array_equal(np.where(hit,a[:r['D']].argmax(0),-1),c['delay']);np.testing.assert_array_equal(a.mean(0),c['alarm_fraction_W'])
                    i=spec['source_id'];o=spec['onset'];orig=bank[i]['scores'][g,o:o+r['D']]
                    np.testing.assert_array_equal((orig[:,None,None,:]>th[g]).any(-1).any(0),c['matched_hit'])
                case_count+=1
    assert case_count==11730 and event_count==396
    text=f'PASS E15BC: 60 units; 11730 raw edited windows recalibrated; all 30 calibration/alpha/policy decisions and delays recomputed.\nPASS E15BC: {event_count} calibration-specific historical events; {frame_records} frame strata; normal grid/frame alarms; exact G0 arrays; normal-only threshold recomputation.\nPASS E15BC: checkpoint/config/baseline hashes and split separation verified; max causal prefix/replay error {max_error:.3g}.\n'
    (ROOT/'results/E15BC/validation.txt').write_text(text);print(text)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bc',action='store_true');a=p.parse_args();b0()
    if a.bc:bc()
