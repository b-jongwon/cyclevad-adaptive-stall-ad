import numpy as np
from cycle_vad.paper_diagnostics import extend,ties,strata,diagnostic_record

def test_cp_window_max_commutes_but_new_threshold_is_needed():
    x=np.zeros((3,6));x[:,4]=[3,0,1];x[:,5]=[0,4,1]
    assert extend(x)[:,6].max()==max(x[:,4].max(),x[:,5].max())
    assert not np.any(extend(x)[:,6]>4)
    assert np.any(x[:,4]>2)

def test_winners_allocate_ties_without_first_column_bias():
    np.testing.assert_array_equal(ties(np.array([[3,3,1],[0,0,0]])),[[.5,.5,0],[1/3,1/3,1/3]])

def test_normal_history_excludes_unknown_and_anomalous_frames():
    masks=strata(np.array([0,-1,1,0,1,0]),np.zeros(6))
    np.testing.assert_array_equal(masks['normal_no_prior_anomaly'],[1,0,0,0,0,0])
    np.testing.assert_array_equal(masks['normal_after_first_anomaly'],[0,0,0,1,0,1])
    assert sum(masks[f'normal_phase{k}'].sum() for k in range(4))==3

def test_diagnostic_exceedance_is_not_winner_attribution():
    c=np.array([[5.,6,0,0,0,0,0,0],[5.,0,0,0,0,0,0,0]])
    s=np.zeros((2,7));s[:,[0,3]]=c[:,:2].max(1)[:,None]
    r=diagnostic_record(c,c,s,np.ones(2,bool),np.ones(8)*4,4.)
    assert r['full_fpr']==1 and r['full_threshold_exceed'][:2]==[1,.5]
    assert r['exclusive_full_alarm'][:2]==[.5,0]
    assert r['component_winner_share'][:2]==[.5,.5]
    assert diagnostic_record(c,c,s,np.zeros(2,bool),np.ones(8),4.) is None
