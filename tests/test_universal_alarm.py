import numpy as np

from cycle_vad.universal_alarm import alarm_rule, consecutive_alarm


def test_causal_prefix_outputs_do_not_change_when_future_changes():
    stream = {"score": np.r_[np.ones(30), np.full(10, 8.)],
              "stall": np.zeros(40), "normal_baseline": .5,
              "base_threshold": 2., "stall_threshold": 3.}
    params = {"gain": .3, "persistence": 2, "stall": False}
    full, thresholds = alarm_rule(stream, params)
    for count in (8, 23, 24, 35):
        short = {**stream, "score": stream["score"][:count], "stall": stream["stall"][:count]}
        actual, _ = alarm_rule(short, params)
        np.testing.assert_array_equal(actual, full[:count])
    np.testing.assert_array_equal(thresholds[:23], np.full(23, 2.))


def test_stall_changes_alarm_without_mutating_ranking_score():
    score = np.zeros(30)
    stream = {"score": score, "stall": np.r_[np.zeros(25), np.ones(5) * 4],
              "normal_baseline": 0., "base_threshold": 2., "stall_threshold": 3.}
    alarm, _ = alarm_rule(stream, {"gain": .3, "persistence": 2, "stall": True})
    assert alarm[-1]
    np.testing.assert_array_equal(score, np.zeros(30))


def test_persistence_resets_after_gap():
    np.testing.assert_array_equal(consecutive_alarm([1, 1, 0, 1, 1, 1], 3),
                                  [0, 0, 0, 0, 0, 1])
