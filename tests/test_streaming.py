import copy
from types import SimpleNamespace
import numpy as np
import pytest
from cycle_vad.streaming import StreamingPhysicalTracker
from cycle_vad.process_experiment import PhysicalTracker

@pytest.mark.parametrize('stride',[1,2])
def test_incremental_tracker_exact_reference_and_bounded_memory(stride):
    rng=np.random.default_rng(18);g=128
    t=PhysicalTracker(stride);t.grid_size=g;t.period=200.;t.temperature=.3;t.template=rng.normal(size=(g,12));t.unit=np.exp(2j*np.pi*np.arange(g)/g)
    x=rng.normal(size=(250,6));ids=np.arange(len(x))*stride
    expected=t.predict_latent(x,ids);stream=StreamingPhysicalTracker(t,stride)
    records=[stream.step(v,int(i)) for v,i in zip(x,ids)]
    for k in expected:np.testing.assert_allclose([r[k] for r in records],expected[k],atol=1e-12,rtol=1e-12)
    assert len(stream.x_history)<=4//stride+1 and len(stream.angle_history)<=32//stride+1
    stream.reset();short=[stream.step(v,int(i)) for v,i in zip(x[:50],ids[:50])]
    for k in expected:np.testing.assert_array_equal([r[k] for r in short],[r[k] for r in records[:50]])
    stream.reset();changed=x.copy();changed[50:]+=100
    future=[stream.step(v,int(i)) for v,i in zip(changed,ids)]
    for k in expected:np.testing.assert_array_equal([r[k] for r in future[:50]],[r[k] for r in records[:50]])
    with pytest.raises(ValueError):stream.step(x[0],999)
