"""Recompute published metrics, case delays and component alarm identities."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from cycle_vad.paper_diagnostics import ARMS,strata,diagnostic_record,sha
from cycle_vad.paper_experiments import calibrate,event_rows
from cycle_vad.metrics import frame_metrics
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def compare(a,b):
    if b is None:assert a is None
    elif isinstance(b,dict):
        for k,v in b.items():compare(a[k],v)
    elif isinstance(b,str):assert a==b
    else:np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-10)
def main():
    paths=sorted((ROOT/'results/E11B').glob('seed*/R*/fold*/run.json'));assert len(paths)==60;cases=videos=diag=0;max_error=0
    for p in paths:
        r=read(p);suffix=p.parent.relative_to(ROOT/'results/E11B');base=ROOT/'runs/E11B'/suffix;dbase=ROOT/'runs/E15A'/suffix;old=ROOT/'runs/E11'/suffix
        assert r['status']=='complete' and r['frozen_checkpoint_sha256']==sha(old/'model.joblib')
        max_error=max(max_error,r['identity_max_error'],r['score_prefix_max_error']);assert max_error<=1e-9
        ps=read(p.with_name('per_video.json'));bank={}
        for i in sorted(set(v['id'] for v in ps)):
            _,part,seq=i.split('/')
            with np.load(base/f'{part}_{seq}.npz') as d:bank[i]=dict(d)
            with np.load(old/f'{part}_{seq}.npz') as d:np.testing.assert_array_equal(bank[i]['scores'][:,:6],d['scores'])
            s=bank[i]['scores'];np.testing.assert_array_equal(s[:,6],np.maximum(s[:,4],s[:,5]))
        th,ft,counts=calibrate({i:v['scores'] for i,v in bank.items()},r['splits'],r['T'],r['D']);np.testing.assert_array_equal(th,r['window_thresholds']);np.testing.assert_array_equal(ft,r['frame_thresholds']);assert counts==r['threshold_window_counts']
        for v in ps:
            a=ARMS.index(v['arm']);d=bank[v['id']];compare(v,frame_metrics(d['labels'],d['scores'][:,a],th[3,a]));compare(v['frame_q99'],frame_metrics(d['labels'],d['scores'][:,a],ft[a]));videos+=1
        for m in read(p.with_name('metrics.json')):
            selected=[v for v in ps if v['partition']==m['partition'] and v['arm']==m['arm']];a=ARMS.index(m['arm']);y=np.concatenate([bank[v['id']]['labels'] for v in selected]);s=np.concatenate([bank[v['id']]['scores'][:,a] for v in selected]);compare(m,frame_metrics(y,s,th[3,a]))
        events=[]
        for i in sorted(set(v['id'] for v in ps if v['partition']=='historical_test')):events+=event_rows(i,bank[i]['labels'],bank[i]['scores'],th[3],r['D'])
        expected=read(p.with_name('historical_events.json'));assert {v['id']:v for v in events}=={v['id']:v for v in expected}
        cr=read(p.with_name('cases.json'))
        with np.load(base/'edited_windows.npz') as d:
            assert d['ids'].tolist()==[c['id'] for c in cr]
            for c,w in zip(cr,d['scores']):
                compare(c['edited_max_W'],w.max(0));compare(c['edited_max_D'],w[:r['D']].max(0));compare(c['alarm_frames_W'],(w>th[3]).sum(0))
                for a in range(7):
                    hit=np.flatnonzero(w[:r['D'],a]>th[3,a]);assert c['delay_primary'][a]==(int(hit[0]) if len(hit) else None)
                cases+=1
        dp=ROOT/'results/E15A'/suffix;dr=read(dp/'run.json');cals={};drecords=read(dp/'per_video_strata.json')
        for i in bank:
            _,part,seq=i.split('/')
            with np.load(dbase/f'{part}_{seq}.npz') as d:
                cals[i]=d['calibrated'].copy();s=bank[i]['scores'];np.testing.assert_allclose(d['calibrated'].max(1),s[:,3],atol=1e-9,rtol=1e-9)
                np.testing.assert_array_equal((d['calibrated']>th[3,3]).any(1),s[:,3]>th[3,3])
                for name,mask in strata(d['labels'],d['theta']).items():
                    actual=diagnostic_record(d['raw'],d['calibrated'],s,mask,np.array(dr['component_window_thresholds'])[3],th[3,3]);found=[v for v in drecords if v['id']==i and v['stratum']==name]
                    if actual is None:assert not found;continue
                    assert len(found)==1
                    if name not in ['normal','anomaly']:actual.pop('component_raw_quantiles');actual.pop('component_calibrated_quantiles')
                    compare(found[0],actual);np.testing.assert_allclose(sum(found[0]['component_winner_share']),1,atol=1e-12);np.testing.assert_allclose(sum(found[0]['branch_winner_share']),1,atol=1e-12);diag+=1
        component_th,_,_=calibrate(cals,r['splits'],r['T'],r['D']);np.testing.assert_array_equal(component_th,dr['component_window_thresholds'])
    assert cases==11730
    a=f'PASS E11B: 60 units, {videos} per-video/arm metrics; original6 score arrays unchanged; all11730 synthetic windows/delays/durations independently recomputed.\nPASS E11B: normal thresholds recomputed, frozen checkpoint hashes verified, max replay/prefix error {max_error:.3g}.\n'
    b=f'PASS E15A: 60 units, {diag} per-video strata recomputed from saved raw/calibrated arrays; component thresholds and Full alarm union verified; tied winner shares sum to1.\n'
    (ROOT/'results/E11B/validation.txt').write_text(a);(ROOT/'results/E15A/validation.txt').write_text(b);print(a+b)
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
