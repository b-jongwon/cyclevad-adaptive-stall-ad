import numpy as np
from cycle_vad.paper_experiments import fuse,event_rows,calibrate,maxwindows
from cycle_vad.paper_subspaces import nested_subset,bins,equal_rank,AppearanceHead

def test_nested_subsets_keep_source_identity_and_seed():
    ids=[f'R01/training/{i:02d}' for i in range(12)]
    a=nested_subset(ids,.25,42);b=nested_subset(ids,.5,42);c=nested_subset(ids,1,42)
    assert len(a)==3 and len(b)==6 and a==b[:3] and b==c[:6]
    assert set(c)==set(ids) and c==nested_subset(ids[::-1],1,42)

def test_exact_total_rank_and_coverage_rule():
    t=np.tile((np.arange(128)+.25)/128,3);owner=np.repeat(np.arange(3),128)
    r,c=equal_rank(t,owner,100)
    assert r==64 and min(c['8']['source_counts'])==3
    assert equal_rank(t,np.zeros(len(t)),100)[0]==0
    np.testing.assert_array_equal(bins(np.array([0,.249,.25,.999,1,-.01]),4),[0,0,1,3,0,3])

def test_same_continuous_mean_with_shared_and_local_spaces():
    rng=np.random.default_rng(4);theta=np.tile((np.arange(128)+.25)/128,3);z=rng.normal(size=(len(theta),24)).astype('float32')
    a=AppearanceHead('continuous',0,2,.1).fit(z,theta,16,42)
    b=AppearanceHead('continuous',4,2,.1).fit(z,theta,16,42)
    np.testing.assert_array_equal(a.coef,b.coef)
    assert sum(len(s.basis) for s in a.spaces)==sum(len(s.basis) for s in b.spaces)==16
    assert len(b.spaces)==4

def test_zero_residual_without_pca_is_zero():
    z=np.full((40,10),3,dtype='float32');theta=np.arange(40)/40
    h=AppearanceHead('global',None,2,.1).fit(z,theta,0,42)
    raw,mse=h.raw(z,theta);np.testing.assert_array_equal(raw,0);np.testing.assert_array_equal(mse,0)

def test_fusion_score_dominance_does_not_imply_alarm_union():
    s=fuse(np.array([1,4]),np.array([5,2]),np.array([3,6]))
    np.testing.assert_array_equal(s[:,3],np.max(s[:,[0,4,5]],axis=1))
    assert s[0,1]>4 and not s[0,3]>6

def test_events_never_cross_unknowns_and_deadline_is_causal():
    y=np.array([0,1,1,-1,1,1,1,0]);s=np.zeros((8,6));s[2,:]=2;s[6,:]=2
    ev=event_rows('x',y,s,np.ones(6),2)
    assert len(ev)==2 and ev[0]['delay_primary']==[1]*6
    assert ev[1]['delay_primary']==[None]*6 and all(ev[1]['hit_full_interval'])

def test_video_weighted_window_calibration_ignores_eval():
    scores={'a':np.ones((100,2)),'b':np.ones((400,2))*3,'eval':np.ones((200,2))*1000}
    splits={'threshold':['a','b'],'normal_evaluation':['eval']}
    t,ft,counts=calibrate(scores,splits,100,20)
    assert np.all(t==3) and counts['a']<counts['b'] and np.all(ft==3)
    scores['eval']*=1000
    np.testing.assert_array_equal(calibrate(scores,splits,100,20)[0],t)


def test_conditional_float64_score_is_prefix_stable():
    from cycle_vad.paper_experiments import conditional,conditional_raw
    from cycle_vad.model import ResidualSpace,TailCalibrator
    from cycle_vad.tracker import fourier
    rng=np.random.default_rng(9);z=rng.normal(size=(151,40)).astype('float32');theta=np.arange(151)/151;coef=rng.normal(size=(5,40)).astype('float32')
    space=ResidualSpace().fit(z-fourier(theta,2)@coef,16,seed=42)
    ref=conditional_raw(z[:70],theta[:70],coef,2,space);cal=[TailCalibrator().fit(ref[k]) for k in [0,1]]
    full=conditional(z[70:],theta[70:],coef,2,space,cal);prefix=conditional(z[70:111],theta[70:111],coef,2,space,cal)
    np.testing.assert_allclose(full[:41],prefix,atol=1e-9,rtol=1e-9)
