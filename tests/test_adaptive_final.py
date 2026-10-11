import numpy as np

from cycle_vad.adaptive_final import ConditionalEmpiricalTail, _k_of_n, _warmup_adjust


def test_phase_conditioning_removes_phase_scale():
    phase = np.r_[np.full(100, .1), np.full(100, .6)]
    values = np.r_[np.linspace(0, 1, 100), np.linspace(10, 11, 100)]
    model = ConditionalEmpiricalTail(bins=2, minimum=10).fit(values, phase)
    score = model.score(np.array([.9, 10.9]), np.array([.1, .6]))
    np.testing.assert_allclose(score[0], score[1])


def test_warmup_shift_is_bounded_and_robust_to_large_values():
    adjusted, shift = _warmup_adjust(np.array([2., 20., 20., 20.]), 4, 1., .5, .2)
    assert shift == .5
    np.testing.assert_allclose(adjusted, np.array([1.5, 19.5, 19.5, 19.5]))


def test_k_of_n_requires_persistence():
    alarm = _k_of_n(np.array([0., 2., 0., 2., 2.]), threshold=1., k=2, window=3)
    assert alarm.tolist() == [False, False, False, True, True]
