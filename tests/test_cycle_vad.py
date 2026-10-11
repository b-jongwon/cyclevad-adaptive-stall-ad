"""Causality, leakage, cycle wrap, failure modes, and real local model smoke tests."""
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from cycle_vad.data import (cycle_segments, hold, inventory, labels_for, load_cache,
                            split_normal, validate_cache)
from cycle_vad.metrics import event_metrics, frame_metrics
from cycle_vad.model import CycleVAD, TailCalibrator
from cycle_vad.tracker import circular_difference, dtw_map, fourier


def synthetic(seed=0, n=64):
    rng = np.random.default_rng(seed)
    angle = np.arange(n)/n
    base = fourier(angle, 3)[:, 1:]
    global_z = base + rng.normal(0, .01, base.shape)
    patches = np.stack([base[:, :4]+.05*p for p in range(4)], axis=1)
    patches += rng.normal(0, .01, patches.shape)
    return {"global": global_z.astype(np.float32), "patches": patches.astype(np.float32),
            "indices": np.arange(n)*2, "frame_count": np.array(n*2),
            "signature": np.array(f"synthetic-{seed}"), "encoder_identity": np.array('{}')}


CONFIG = {"cycle_grid": 32, "cycle_latent": 6, "harmonics_candidates": [2, 3],
          "ridge_candidates": [.001, .01], "fit_samples": 256,
          "subspace_rank": 5, "local_memory": 32, "seed": 42, "threshold_quantile": .99}


@pytest.fixture(scope="module")
def fitted():
    fit = [synthetic(i) for i in range(4)]
    model = CycleVAD(CONFIG).fit(fit, [synthetic(4)], [[np.arange(64)] for _ in fit])
    model.calibrate([synthetic(5)], [synthetic(6)])
    return model


def test_split_recording_groups_never_overlap():
    rows = [{"id": str(i), "partition": "training"} for i in range(24)]
    groups = {str(i): str(i//2) for i in range(24)}
    split = split_normal(rows, groups=groups)
    owners = {}
    for name, ids in split.items():
        for key in ids:
            assert groups[key] not in owners or owners[groups[key]] == name
            owners[groups[key]] = name
    assert sorted(s for ids in split.values() for s in ids) == sorted(str(i) for i in range(24))
    assert split == split_normal(rows[::-1], groups=groups)


def test_test_data_cannot_enter_normal_splits():
    with pytest.raises(ValueError, match="Only training"):
        split_normal([{"id": "x", "partition": "testing"}])


def test_too_few_recordings_fail_explicitly():
    with pytest.raises(ValueError, match="At least 8"):
        split_normal([{"id": str(i), "partition": "training"} for i in range(7)])


def test_wrap_is_continuous():
    assert circular_difference(.01, .99) == pytest.approx(.02)
    assert circular_difference(.99, .01) == pytest.approx(-.02)
    np.testing.assert_allclose(fourier([0], 4), fourier([1], 4), atol=1e-12)


def test_dtw_aligns_speed_change():
    template = fourier(np.arange(32)/32, 2)
    sequence = fourier((np.arange(48)/48)**1.1, 2)
    mapping = dtw_map(sequence, template)
    assert np.all(np.diff(mapping) >= 0)
    assert mapping[0] < 2 and mapping[-1] > 28


def test_boundaries_use_source_frames_and_reject_bad_values():
    d = synthetic()
    row = {"id": "R01/training/01", "frames": 128}
    parts, source = cycle_segments(d, row, {row["id"]: [0, 64, 128]})
    assert len(parts) == 2 and len(parts[0]) == 32
    assert source == "provided_cycle_boundaries"
    assert cycle_segments(d, row)[1] == "weak_recording_alignment"
    for bounds in ([0, 130], [64, 32], [0, .5, 128]):
        with pytest.raises(ValueError):
            cycle_segments(d, row, {row["id"]: bounds})


def test_future_frames_cannot_change_past_scores(fitted):
    normal = synthetic(17)
    changed = {k: v.copy() for k, v in normal.items()}
    changed["global"][32:] += 20
    changed["patches"][32:] -= 20
    a, b = fitted.score(normal), fitted.score(changed)
    for key in a:
        np.testing.assert_allclose(a[key][:32], b[key][:32], rtol=1e-5, atol=1e-6, err_msg=key)


def test_prefix_length_is_not_cycle_position(fitted):
    data = synthetic(18)
    prefix = {k: v[:32] if k in ("global", "patches", "indices") else v.copy() for k, v in data.items()}
    prefix["frame_count"] = np.array(64)
    a, b = fitted.score(data), fitted.score(prefix)
    for key in a:
        np.testing.assert_allclose(a[key][:32], b[key], rtol=1e-5, atol=1e-6, err_msg=key)


def test_unknown_frame_count_does_not_change_scores(fitted):
    data = synthetic(18)
    altered = dict(data, frame_count=np.array(100000))
    np.testing.assert_array_equal(fitted.score(data)["combined"], fitted.score(altered)["combined"])


def test_detects_injected_visual_corruption(fitted):
    data = synthetic(19)
    normal = fitted.score(data)["appearance"]
    data["patches"][25:40, 1] += 2
    score = fitted.score(data)["appearance"]
    assert np.median(score[25:40]) > np.quantile(normal, .99)


def test_stall_and_reverse_leave_temporal_evidence(fitted):
    data = synthetic(20)
    normal = fitted.score(data)
    stalled = {k: v.copy() for k, v in data.items()}
    reversed_ = {k: v.copy() for k, v in data.items()}
    for key in ("global", "patches"):
        stalled[key][24:48] = stalled[key][24]
        reversed_[key][24:48] = reversed_[key][24:48][::-1]
    stall = fitted.score(stalled)
    reverse = fitted.score(reversed_)
    assert np.mean(stall["raw_progress"][32:48]) > np.mean(normal["raw_progress"][32:48])
    assert np.max(reverse["raw_innovation"][24:48]) > np.max(normal["raw_innovation"][24:48])


def test_calibration_does_not_saturate_at_reference_max():
    cal = TailCalibrator().fit(np.arange(100))
    score = cal.score([98, 99, 100, 200])
    assert np.all(np.diff(score) > 0)
    constant = TailCalibrator().fit(np.zeros(100))
    assert constant.score([1])[0] > constant.score([0])[0]


def test_model_persistence(fitted, tmp_path):
    path = tmp_path/"model.joblib"
    joblib.dump(fitted, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(fitted.score(synthetic(10))["combined"], restored.score(synthetic(10))["combined"], rtol=1e-6, atol=1e-7)


def test_strict_label_mismatch_and_hold(tmp_path):
    path = tmp_path/"001.npy"
    np.save(path, [0, 1, 0])
    assert np.all(labels_for({"partition": "testing", "labels": str(path), "frames": 4}) == -1)
    np.testing.assert_array_equal(hold([0, 3], [.2, .8], 5), [.2, .2, .2, .8, .8])
    with pytest.raises(ValueError):
        hold([1, 3], [.2, .8], 5)


def test_event_metrics_do_not_bridge_unknown_frames():
    y = [0, 1, 1, -1, 1, 0]
    s = [0, 0, 1, 1, 0, 0]
    result = event_metrics(y, s, .5)
    assert result["events"] == 2 and result["detected_events"] == 1
    assert result["detected_event_delays_source_frames"] == [1]
    assert frame_metrics(y, s, .5)["unknown_frames"] == 1


def test_invalid_feature_cache_rejected():
    d = synthetic()
    d["global"][0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_cache(d)


@pytest.mark.parametrize("family", ["dinov2", "clip"])
def test_actual_tiny_visual_tower_and_cache_resume(tmp_path, family):
    torch = pytest.importorskip("torch")
    from PIL import Image
    from transformers import (BitImageProcessor, CLIPImageProcessor, CLIPVisionConfig,
                              CLIPVisionModel, Dinov2Config, Dinov2Model)
    from cycle_vad.features import VisualEncoder, extract_sequence
    torch.manual_seed(0)
    if family == "dinov2":
        model = Dinov2Model(Dinov2Config(image_size=28, patch_size=14, hidden_size=32,
                                       num_hidden_layers=3, num_attention_heads=4))
        processor = BitImageProcessor(do_resize=False, do_center_crop=False)
    else:
        model = CLIPVisionModel(CLIPVisionConfig(image_size=28, patch_size=14, hidden_size=32,
                                                intermediate_size=64, num_hidden_layers=3, num_attention_heads=4))
        processor = CLIPImageProcessor(do_resize=False, do_center_crop=False)
    # Non-native input size also exercises positional-embedding interpolation.
    config = {"family": family, "model_id": "tiny-local-test", "image_size": 42,
              "patch_grid": 2, "patch_dim": 8, "device": "cpu"}
    encoder = VisualEncoder(config, model, processor)
    directory = tmp_path/"frames"
    directory.mkdir()
    for i in range(4):
        Image.new("RGB", (60, 20), (i*50, 40, 80)).save(directory/f"{i}.jpg")
    row = {"id": "R01/training/01", "directory": str(directory)}
    path = extract_sequence(encoder, row, tmp_path/"cache", stride=1, batch_size=2)
    data = load_cache(path)
    assert data["global"].shape == (4, 32) and data["patches"].shape == (4, 4, 8)
    timestamp = path.stat().st_mtime_ns
    extract_sequence(encoder, row, tmp_path/"cache", stride=1, batch_size=2)
    assert path.stat().st_mtime_ns == timestamp
    Image.new("RGB", (60, 20), (255, 0, 0)).save(directory/"3.jpg")
    extract_sequence(encoder, row, tmp_path/"cache", stride=1, batch_size=2)
    assert str(load_cache(path)["signature"]) != str(data["signature"])


def test_complete_fit_evaluate_protocol_without_pretrained_download(tmp_path):
    from PIL import Image
    from cycle_vad.pipeline import run
    root, output = tmp_path/"data", tmp_path/"out"
    for part, count in (("training", 10), ("testing", 2)):
        for i in range(count):
            directory = root/"R01"/part/"frames"/f"{i:02d}"
            directory.mkdir(parents=True)
            for frame in range(64):
                Image.new("RGB", (4, 4)).save(directory/f"{frame}.jpg")
            d = synthetic(i+100, n=32)
            d["frame_count"] = np.array(64)
            if part == "testing" and i == 1:
                d["patches"][10:20] += 2
            cache = output/"cache"/"R01"/part/f"{i:02d}.npz"
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **d)
            if part == "testing":
                label = root/"R01"/"test_label"/f"{i:03d}.npy"
                label.parent.mkdir(parents=True, exist_ok=True)
                y = np.zeros(64, dtype=np.int8)
                if i == 1:
                    y[20:40] = 1
                np.save(label, y)
    config = {"seed": 42, "scenes": ["R01"], "stride": 2, "model": CONFIG}
    run(root, output, config, stage="fit")
    protocol_before = (output/"R01"/"protocol.json").read_bytes()
    result = run(root, output, config, stage="evaluate")
    assert result["R01"]["combined"]["valid_frames"] == 128
    assert result["R01"]["appearance"]["auroc"] > .9
    assert (output/"R01"/"protocol.json").read_bytes() == protocol_before
    assert (output/"R01"/"predictions"/"testing_01.npz").exists()
