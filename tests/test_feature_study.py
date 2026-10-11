import numpy as np
import torch

from cycle_vad.feature_study import FeatureHeads, contexts, differences


def test_prediction_context_never_contains_current_or_future_features():
    x = np.arange(20, dtype=np.float32)[:, None]
    c = contexts(x, 3)
    np.testing.assert_array_equal(c[5, :, 0], [2, 3, 4])
    np.testing.assert_array_equal(c[:3, :, 0], [[0, 0, 0], [0, 0, 0], [0, 0, 1]])


def test_position_specific_memory_matches_known_euclidean_distances():
    h = FeatureHeads()
    h.device = "cpu"
    bank = torch.tensor([[[0., 0.], [1., 1.]], [[2., 2.], [3., 3.]]])
    values = np.array([[[0., 0.], [2., 2.]], [[2., 1.], [4., 3.]]], dtype=np.float32)
    np.testing.assert_allclose(h.patch_distance(values, bank), [[0., 0.], [1., 1.]], atol=1e-6)


def test_motion_is_causal_and_preserves_sequence_length():
    x = np.arange(6, dtype=np.float32)[:, None]
    np.testing.assert_array_equal(differences(x, 2)[:, 0], [0, 1, 2, 2, 2, 2])
