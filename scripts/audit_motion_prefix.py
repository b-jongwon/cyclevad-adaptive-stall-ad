"""Reproduce known startup future-dependence; do not change saved models."""
import json
import numpy as np
from cycle_vad.adaptive_final import _motion

original = np.tile([1., 0.], (8, 1))
changed = original.copy()
changed[4] = [0., 1.]
print(json.dumps({
    'unchanged_prefix_observations': 4,
    'modified_observation_zero_based': 4,
    'lag': 4,
    'original_initial_motion': _motion(original, 4)[:4].tolist(),
    'changed_initial_motion': _motion(changed, 4)[:4].tolist(),
    'prefix_invariant': bool(np.array_equal(_motion(original, 4)[:4], _motion(changed, 4)[:4])),
    'status': 'known limitation; no correction or new training performed in this audit',
}, indent=2))
