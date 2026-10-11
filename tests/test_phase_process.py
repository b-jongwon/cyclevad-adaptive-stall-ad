import numpy as np
import pytest
from scipy.ndimage import convolve1d
from scipy.special import softmax
from cycle_vad.phase_experiment import offset, phase, fit_mean
from cycle_vad.process_experiment import PhysicalTracker,TwoTail,weighted_quantile,edit_map,speed_map,dynamics
from cycle_vad.model import descriptor
from cycle_vad.tracker import CycleTracker
from test_cycle_vad import synthetic


def test_edit_maps_preserve_declared_content_and_prefix():
    np.testing.assert_array_equal(edit_map(10,3,2,'freeze'),[0,1,2,2,2,3,4,5,6,7,8,9])
    np.testing.assert_array_equal(edit_map(10,3,2,'reverse'),[0,1,2,4,3,5,6,7,8,9])
    np.testing.assert_array_equal(edit_map(10,3,2,'skip'),[0,1,2,5,6,7,8,9])
    np.testing.assert_array_equal(edit_map(10,3,2,'swap_adjacent_blocks'),[0,1,2,5,6,3,4,7,8,9])
    for k in ['reverse','swap_adjacent_blocks']:np.testing.assert_array_equal(np.sort(edit_map(10,3,2,k)),np.arange(10))
    with pytest.raises(ValueError):edit_map(10,9,2,'skip')
    np.testing.assert_array_equal(speed_map(10,1),np.arange(10))
    assert np.all(speed_map(10,.8)<10)


def test_two_tails_detects_motion_loss_and_large_motion():
    c=TwoTail().fit(np.arange(1.,101.))
    assert c.score([0])[0]>c.score([50])[0]
    assert c.score([150])[0]>c.score([50])[0]
    assert TwoTail().fit(np.zeros(100)).score([0])[0]==0
    assert weighted_quantile([1,2,100],[.49,.50,.01],.99)==2


def test_circular_convolution_direction_matches_legacy_transition():
    rng=np.random.default_rng(3);p=rng.random(128);p/=p.sum();offsets=np.arange(-32,33)
    w=softmax(-.5*((offsets-2.1)/1.4)**2)
    expected=sum(a*np.roll(p,int(k)) for a,k in zip(w,offsets))
    np.testing.assert_allclose(convolve1d(p,w,mode='wrap'),expected,rtol=1e-12,atol=1e-14)


@pytest.fixture(scope='module')
def tracker():
    ds=[synthetic(i) for i in range(3)];zs=[descriptor(d) for d in ds]
    t=PhysicalTracker(2,42);t.fit(zs,[d['indices'] for d in ds],[[np.arange(64)] for _ in ds]);return t


def test_physical_tracker_matches_legacy_at_stride2(tracker):
    d=synthetic(14);z=descriptor(d);x=tracker.reducer.transform(z)/tracker.scale
    a=tracker.predict_latent(x,d['indices']);b=CycleTracker.predict(tracker,z,d['indices'])
    for k in ['angle','alignment','innovation','progress','observation_angle']:
        np.testing.assert_allclose(a[k],b[k],atol=2e-5,rtol=2e-5,err_msg=k)


def test_future_or_prefix_length_cannot_change_temporal_scores(tracker):
    d=synthetic(23);x=tracker.reducer.transform(descriptor(d))/tracker.scale
    a=tracker.predict_latent(x,d['indices']);changed=x.copy();changed[30:]+=50
    b=tracker.predict_latent(changed,d['indices']);c=tracker.predict_latent(x[:30],d['indices'][:30])
    for k in a:
        np.testing.assert_allclose(a[k][:30],b[k][:30],atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(a[k][:30],c[k],atol=1e-12,rtol=1e-12)


def test_output_clock_hides_original_source_mapping(tracker):
    d=synthetic(25);x=tracker.reducer.transform(descriptor(d))/tracker.scale
    mapping=edit_map(len(x),20,8,'reverse');out=tracker.predict_latent(x[mapping],np.arange(len(mapping))*2)
    with pytest.raises(ValueError):tracker.predict_latent(x[mapping],mapping*2)
    assert np.isfinite(out['progress']).all()


def test_location_scramble_is_source_specific_and_constant():
    trace={'angle':np.array([.1,.2,.3]),'observation_angle':np.array([.11,.21,.31])}
    ids=np.array([0,2,4]);a=phase(trace,ids,100,'M4_1101','R01/training/01')
    assert offset('R01/training/01',1101)!=offset('R01/training/02',1101)
    np.testing.assert_allclose((a-trace['angle'])%1,offset('R01/training/01',1101))
    np.testing.assert_allclose(phase(trace,ids,100,'M1','x'),[.11,.13,.15])


def test_missing_history_differences_are_not_zero_motion_anomalies(tracker):
    d=synthetic(27);x=tracker.reducer.transform(descriptor(d))/tracker.scale
    raw,valid=dynamics(tracker,x,d['indices'],np.zeros((x.shape[1]+1,x.shape[1])))
    assert not valid['diff32'][:16].any() and valid['diff32'][16:].all()
    assert not valid['ar'][0]


def test_physical_lags_match_source_time_at_both_strides(tracker):
    import copy
    t=copy.deepcopy(tracker);t.stride=1
    # A linear feature trajectory makes squared increments exactly comparable.
    x=np.arange(128)[:,None]*np.ones((1,len(t.scale)))*.01
    ar=np.zeros((x.shape[1]+1,x.shape[1]))
    dense,valid=dynamics(t,x,np.arange(128),ar)
    sparse,valid2=dynamics(tracker,x[::2],np.arange(0,128,2),ar)
    for h in [2,8,32]:
        np.testing.assert_allclose(dense[f'diff{h}'][::2],sparse[f'diff{h}'],atol=1e-12)
        np.testing.assert_array_equal(valid[f'diff{h}'][::2],valid2[f'diff{h}'])
    changed=x.copy();changed[35:]+=99
    causal=t.predict_latent(x,np.arange(128));future=t.predict_latent(changed,np.arange(128))
    for k in causal:np.testing.assert_allclose(causal[k][:35],future[k][:35],atol=1e-12)
