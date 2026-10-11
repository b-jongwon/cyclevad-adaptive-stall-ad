import numpy as np
from cycle_vad.ablation import compose
from cycle_vad.tracker import CycleTracker
from test_cycle_vad import synthetic, CONFIG
from cycle_vad.model import CycleVAD


def test_registered_removals_isolate_the_target_signal():
    q={k:np.zeros(3) for k in ['pooled','pooled_inside','local','conditional','conditional_inside','alignment','innovation','progress','single_progress','discrete','discrete_inside']}
    q['progress'][0]=10; q['conditional'][1]=10; q['local'][2]=10
    s=compose(q,np.array([1,.2,1]))
    np.testing.assert_array_equal(s['full'],[10,2,10])
    np.testing.assert_array_equal(s['no_progress'],[0,2,10])
    np.testing.assert_array_equal(s['single_lag'],[0,2,10])
    np.testing.assert_array_equal(s['no_confidence'],[10,10,10])
    np.testing.assert_array_equal(s['no_local'],[10,2,0])


def test_ablation_full_matches_original_model():
    ds=[synthetic(i) for i in range(7)]
    m=CycleVAD(CONFIG).fit(ds[:4],ds[4:5],[[np.arange(64)] for _ in range(4)])
    m.calibrate(ds[5:6],ds[6:7]); d=synthetic(88)
    raw,cycle,_=m.raw(d)
    s=compose({k:m.calibrators[k].score(v) for k,v in raw.items()},cycle['confidence'])
    np.testing.assert_allclose(s['full'],m.score(d)['combined'])
    np.testing.assert_allclose(s['appearance'],m.score(d)['appearance'])
    np.testing.assert_allclose(s['cycle_conditioned'],m.score(d)['cycle_conditioned'])
    multi=m.tracker.predict(__import__('cycle_vad.model',fromlist=['descriptor']).descriptor(d),d['indices'])
    single=m.tracker.predict(__import__('cycle_vad.model',fromlist=['descriptor']).descriptor(d),d['indices'],progress_lags=(1,))
    np.testing.assert_array_equal(multi['angle'],single['angle'])
    assert np.all(single['progress']<=multi['progress'])


def test_variant_thresholds_are_normal_only_and_full_is_reproduced(tmp_path):
    import json
    from cycle_vad.ablation import evaluate_variants, CORE, REMOVALS
    from cycle_vad.pipeline import fit_scene, evaluate_scene
    rows=[]
    for part,count in [('training',10),('testing',2)]:
        for i in range(count):
            rid=f'R01/{part}/{i:02d}'
            label=tmp_path/f'label_{i}.npy'
            row={'id':rid,'scene':'R01','partition':part,'sequence':f'{i:02d}','frames':128,'labels':str(label)}
            rows.append(row)
            d=synthetic(200+i)
            if part=='testing':
                d['global'][20:35]+=1
                np.save(label,np.r_[np.zeros(40),np.ones(30),np.zeros(58)].astype(np.int8))
            p=tmp_path/'cache'/(rid+'.npz'); p.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(p,**d)
    cfg={'seed':42,'stride':2,'model':CONFIG}
    m=fit_scene(rows,tmp_path,cfg)
    original=evaluate_scene(m,rows,tmp_path)
    result=evaluate_variants(m,rows,tmp_path,tmp_path/'extended',extended=True)
    assert set(result)==set(CORE+REMOVALS)
    for key in ['threshold','auroc','ap','fpr','recall']:
        np.testing.assert_allclose(result['full'][key],original['combined'][key])
    before=json.loads((tmp_path/'extended/protocol.json').read_text())['thresholds']
    for r in rows:
        if r['partition']=='testing': np.save(r['labels'],np.zeros(128,dtype=np.int8))
    evaluate_variants(m,rows,tmp_path,tmp_path/'changed',extended=True)
    after=json.loads((tmp_path/'changed/protocol.json').read_text())['thresholds']
    assert before==after
