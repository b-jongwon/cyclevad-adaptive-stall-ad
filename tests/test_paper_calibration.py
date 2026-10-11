import numpy as np
from cycle_vad.model import TailCalibrator
from cycle_vad.paper_calibration import VideoEqualTail,policy_thresholds,alarms,event_records,serial_thresholds,deserialize_thresholds

def test_video_equal_tail_matches_original_for_equal_lengths_and_extrapolates():
    seq=[np.array([0.,1,1,4]),np.array([1.,2,3,7])];old=TailCalibrator().fit(np.concatenate(seq));new=VideoEqualTail().fit(seq,old)
    x=np.array([-1,0,1,2,4,7,8,10.]);np.testing.assert_allclose(old.score(x),new.score(x),atol=1e-12)
    assert new.score([10])[0]>new.score([8])[0]>new.score([7])[0]

def test_video_equal_weights_differ_from_frame_weights():
    seq=[np.array([0.]),np.array([10.,10.,10.])];old=TailCalibrator().fit(np.concatenate(seq));new=VideoEqualTail().fit(seq,old)
    np.testing.assert_allclose(new.score([5]),[-np.log(3/5)])
    assert new.score([5])[0]>old.score([5])[0]

def test_union_preserves_c_but_adds_normal_cost_budgeted_or_can_lose_c():
    th=np.array([[[2,np.inf],[4,4],[3,4],[4,4],[2,4]]])
    scores=np.array([[2.5,0],[0,5],[0,0]])
    a=alarms(scores,th)[:,0]
    assert a[:,0].sum()==1 and a[:,4].sum()==2 and not a[0,2]
    assert np.all(~a[:,0]|a[:,4])
    assert a[:,4].mean()==a[:,0].mean()+np.mean(a[:,4]&~a[:,0])

def test_event_deadlines_unknown_labels_and_strict_threshold():
    scores=np.array([[10,0],[3,0],[4,0],[10,0],[10,0],[10,0]])
    th=np.array([[[3,np.inf]]]);y=np.array([-1,1,1,0,1,1])
    r=event_records('x',y,scores,th,1)
    assert len(r)==2 and r[0]['delay']==[[-1]] and r[1]['delay']==[[0]]

def test_threshold_roundtrip_and_sparse_quantile_collapse():
    x=np.column_stack([np.arange(10.),np.arange(10.)+1]);w=np.ones(10)/10
    t=policy_thresholds(x,w);np.testing.assert_array_equal(t,deserialize_thresholds(serial_thresholds(t)))
    np.testing.assert_array_equal(t[0,2],t[0,3]);assert np.isinf(t[0,0,1])
