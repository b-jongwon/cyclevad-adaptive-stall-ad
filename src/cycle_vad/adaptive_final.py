"""Adaptive score normalization and a dedicated freeze detector.

The module fits only on normal recordings. Test annotations are loaded after
all scores and alarms for a sequence have been produced.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import hold, inventory, labels_for, load_cache, split_normal, write_json
from .pipeline import cache_path, load_rows, validate_model_cache


def _ema(values, alpha: float):
    values = np.asarray(values, dtype=np.float64)
    out = values.copy()
    for index in range(1, len(out)):
        out[index] = alpha * values[index] + (1.0 - alpha) * out[index - 1]
    return out


def _k_of_n(values, threshold: float, k: int, window: int):
    raw = np.asarray(values) > threshold
    counts = np.convolve(raw.astype(np.int16), np.ones(window, dtype=np.int16), mode="full")[: len(raw)]
    return counts >= k


def _warmup_adjust(values, warmup: int, expected_median: float, max_shift: float,
                   baseline_fraction: float = 0.2):
    values = np.asarray(values, dtype=np.float64)
    count = min(len(values), max(4, warmup))
    early = np.sort(values[:count])[: max(3, int(np.ceil(count * baseline_fraction)))]
    shift = float(np.clip(np.median(early) - expected_median, -max_shift, max_shift))
    return values - shift, shift


class ConditionalEmpiricalTail:
    """Phase-conditional empirical upper/lower-tail surprisal."""

    def __init__(self, bins=16, minimum=64):
        self.bins = int(bins)
        self.minimum = int(minimum)

    def fit(self, values, phases):
        values = np.asarray(values, dtype=np.float64)
        phases = np.asarray(phases, dtype=np.float64) % 1.0
        self.global_reference = np.sort(values)
        bin_ids = np.floor(phases * self.bins).astype(int).clip(0, self.bins - 1)
        self.references = []
        for bin_id in range(self.bins):
            selected = values[bin_ids == bin_id]
            self.references.append(np.sort(selected) if len(selected) >= self.minimum else self.global_reference)
        return self

    @staticmethod
    def _surprisal(reference, values, upper):
        side = "left" if upper else "right"
        ranks = np.searchsorted(reference, values, side=side)
        if upper:
            probability = (len(reference) - ranks + 1) / (len(reference) + 1)
        else:
            probability = (ranks + 1) / (len(reference) + 1)
        return -np.log(np.maximum(probability, 1.0 / (len(reference) + 1)))

    def score(self, values, phases, upper=True):
        values = np.asarray(values, dtype=np.float64)
        phases = np.asarray(phases, dtype=np.float64) % 1.0
        bin_ids = np.floor(phases * self.bins).astype(int).clip(0, self.bins - 1)
        result = np.empty(len(values), dtype=np.float64)
        for bin_id, reference in enumerate(self.references):
            selected = bin_ids == bin_id
            result[selected] = self._surprisal(reference, values[selected], upper)
        return result


def _motion(global_features, lag):
    features = np.asarray(global_features, dtype=np.float32)
    result = np.full(len(features), np.nan, dtype=np.float64)
    if len(features) > lag:
        result[lag:] = 1.0 - np.sum(features[lag:] * features[:-lag], axis=1)
        result[:lag] = result[lag]
    else:
        result[:] = 0.0
    return result


@dataclass
class AdaptiveModel:
    phase_combined: ConditionalEmpiricalTail
    phase_appearance: ConditionalEmpiricalTail
    phase_process: ConditionalEmpiricalTail
    motion: dict
    expected_warmup_median: float
    expected_intensity_baseline: float
    thresholds: dict
    config: dict

    @classmethod
    def fit(cls, base_model, reference, threshold, config):
        bins = int(config.get("phase_bins", 16))
        minimum = int(config.get("minimum_bin_samples", 64))
        reference_scores = [base_model.score(item) for item in reference]

        def fit_tail(key):
            return ConditionalEmpiricalTail(bins, minimum).fit(
                np.concatenate([score[key] for score in reference_scores]),
                np.concatenate([score["angle"] for score in reference_scores]),
            )

        combined = fit_tail("combined")
        appearance = fit_tail("appearance")
        process = fit_tail("process")
        motion_models = {}
        for lag in config.get("stall_lags", [1, 4]):
            motion_models[int(lag)] = ConditionalEmpiricalTail(bins, minimum).fit(
                np.concatenate([_motion(item["global"], int(lag)) for item in reference]),
                np.concatenate([score["angle"] for score in reference_scores]),
            )

        provisional = cls(combined, appearance, process, motion_models, 0.0, 0.0, {}, dict(config))
        reference_variants = [provisional.score(base_model, item, allow_warmup=False)[0]
                              for item in reference]
        provisional.expected_warmup_median = float(np.median(np.concatenate(
            [item["phase_branch"] for item in reference_variants]
        )))
        warmup = int(config.get("warmup_samples", 24))
        fraction = float(config.get("warmup_baseline_fraction", 0.2))
        intensity_baselines = []
        for score in reference_scores:
            count = min(len(score["combined"]), max(4, warmup))
            early = np.sort(score["combined"][:count])[:max(3, int(np.ceil(count * fraction)))]
            intensity_baselines.append(float(np.median(early)))
        provisional.expected_intensity_baseline = float(np.median(intensity_baselines))
        threshold_variants = [provisional.score(base_model, item, allow_warmup=True)[0]
                              for item in threshold]
        quantile = float(config.get("threshold_quantile", 0.99))
        names = threshold_variants[0].keys()
        provisional.thresholds = {
            name: float(np.quantile(np.concatenate([
                hold(item["indices"], variants[name], int(item["frame_count"]))
                for item, variants in zip(threshold, threshold_variants)
            ]), quantile, method="higher"))
            for name in names
        }
        return provisional

    def score(self, base_model, data, allow_warmup=True):
        base = base_model.score(data)
        phase = base["angle"]
        phase_combined = self.phase_combined.score(base["combined"], phase)
        appearance = self.phase_appearance.score(base["appearance"], phase)
        process = self.phase_process.score(base["process"], phase)
        phase_branch = np.maximum(appearance, process)
        warmup, shift = _warmup_adjust(
            phase_branch,
            int(self.config.get("warmup_samples", 24)),
            self.expected_warmup_median,
            float(self.config.get("max_warmup_shift", 2.0)),
            float(self.config.get("warmup_baseline_fraction", 0.2)),
        ) if allow_warmup else (phase_branch.copy(), 0.0)
        smooth = _ema(warmup, float(self.config.get("ema_alpha", 0.4)))

        lower_motion = [model.score(_motion(data["global"], lag), phase, upper=False)
                        for lag, model in sorted(self.motion.items())]
        stall = np.minimum.reduce(lower_motion)
        stall = _ema(stall, float(self.config.get("stall_ema_alpha", 0.5)))
        adaptive_with_stall = np.maximum(smooth, stall)
        count = min(len(base["combined"]), max(4, int(self.config.get("warmup_samples", 24))))
        fraction = float(self.config.get("warmup_baseline_fraction", 0.2))
        early = np.sort(base["combined"][:count])[:max(3, int(np.ceil(count * fraction)))]
        intensity_offset = max(0.0, float(np.median(early)) - self.expected_intensity_baseline)
        intensity_threshold = (
            float(base_model.thresholds["combined"])
            + float(self.config.get("intensity_threshold_gain", 0.3)) * intensity_offset
        )
        return {
            "phase_combined": phase_combined,
            "phase_branch": phase_branch,
            "warmup_phase": warmup,
            "adaptive_ema": smooth,
            "stall": stall,
            "adaptive_with_stall": adaptive_with_stall,
        }, {"warmup_shift": shift, "base": base,
            "intensity_offset": intensity_offset,
            "intensity_threshold": intensity_threshold}


def _metrics(labels, scores, threshold, alarm=None):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    valid = labels >= 0
    y, s = labels[valid], scores[valid]
    alarm = (scores > threshold) if alarm is None else np.asarray(alarm, dtype=bool)
    a = alarm[valid]
    return {
        "frames": int(len(labels)), "valid_frames": int(valid.sum()),
        "unknown_frames": int((~valid).sum()), "anomaly_frames": int((y == 1).sum()),
        "auroc": float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None,
        "ap": float(average_precision_score(y, s)) if np.any(y == 1) else None,
        "fpr": float(a[y == 0].mean()) if np.any(y == 0) else None,
        "recall": float(a[y == 1].mean()) if np.any(y == 1) else None,
        "threshold": float(threshold),
    }


def _event_metrics(labels, alarm):
    positive = np.asarray(labels) == 1
    starts = np.flatnonzero(np.diff(np.r_[False, positive, False].astype(int)) == 1)
    ends = np.flatnonzero(np.diff(np.r_[False, positive, False].astype(int)) == -1)
    delays = []
    for start, end in zip(starts, ends):
        hits = np.flatnonzero(np.asarray(alarm)[start:end])
        if len(hits):
            delays.append(int(hits[0]))
    return {"events": int(len(starts)), "detected_events": int(len(delays)),
            "event_coverage": len(delays) / len(starts) if len(starts) else None,
            "median_delay_frames": float(np.median(delays)) if delays else None}


def _freeze_copy(data, onset, length):
    result = {key: np.array(value, copy=True) for key, value in data.items()}
    end = min(len(result["global"]), onset + length)
    result["global"][onset:end] = result["global"][onset]
    result["patches"][onset:end] = result["patches"][onset]
    return result, end


def run_scene(scene, rows, run_root, config):
    normal = [row for row in rows if row["partition"] == "training"]
    splits = split_normal(normal, int(config.get("seed", 42)))
    groups = {name: [row for row in normal if row["id"] in ids] for name, ids in splits.items()}
    data = {name: load_rows(run_root, selected) for name, selected in groups.items()}
    base_model = joblib.load(Path(run_root) / scene / "model.joblib")
    adaptive = AdaptiveModel.fit(base_model, data["reference"], data["threshold"], config["adaptive"])

    joined = {}
    per_video = []
    k = int(config["adaptive"].get("alarm_k", 3))
    window = int(config["adaptive"].get("alarm_window", 8))
    test_rows = [row for row in rows if row["partition"] == "testing"]
    for row in test_rows:
        item = load_cache(cache_path(run_root, row))
        validate_model_cache(base_model, item)
        variants, detail = adaptive.score(base_model, item)
        base_combined = detail["base"]["combined"]
        variants = {"fixed_cyclevad": base_combined,
                    "intensity_threshold": base_combined,
                    "intensity_threshold_with_stall": base_combined,
                    **variants}
        thresholds = {"fixed_cyclevad": float(base_model.thresholds["combined"]), **adaptive.thresholds}
        # Labels are intentionally loaded only after all model outputs exist.
        labels = labels_for(row)
        for name, sampled in variants.items():
            frame_scores = hold(item["indices"], sampled, row["frames"])
            if name in {"intensity_threshold", "intensity_threshold_with_stall"}:
                threshold = detail["intensity_threshold"]
                alarm = frame_scores > threshold
                if name == "intensity_threshold_with_stall":
                    stall_frames = hold(item["indices"], variants["stall"], row["frames"])
                    alarm |= _k_of_n(stall_frames, thresholds["stall"], k, window)
            else:
                threshold = thresholds[name]
                alarm = _k_of_n(frame_scores, threshold, k, window) if name in {
                    "adaptive_ema", "stall", "adaptive_with_stall"
                } else frame_scores > threshold
            report = {"id": row["id"], "method": name,
                      **_metrics(labels, frame_scores, threshold, alarm),
                      **_event_metrics(labels, alarm),
                      "warmup_shift": detail["warmup_shift"],
                      "intensity_offset": detail["intensity_offset"],
                      "dynamic_threshold": threshold}
            per_video.append(report)
            joined.setdefault(name, [[], [], []])
            joined[name][0].append(labels)
            joined[name][1].append(frame_scores)
            joined[name][2].append(alarm)

    metrics = {}
    for name, (labels, scores, alarms) in joined.items():
        selected = [row for row in per_video if row["method"] == name]
        events = sum(row["events"] for row in selected)
        detected = sum(row["detected_events"] for row in selected)
        metrics[name] = {
            **_metrics(np.concatenate(labels), np.concatenate(scores),
                       float(base_model.thresholds["combined"]) if name in {
                           "fixed_cyclevad", "intensity_threshold", "intensity_threshold_with_stall"
                       } else adaptive.thresholds[name], np.concatenate(alarms)),
            "events": events, "detected_events": detected,
            "event_coverage": detected / events if events else None,
        }
        if name in {"intensity_threshold", "intensity_threshold_with_stall"}:
            metrics[name]["threshold_mode"] = "per-video intensity-normalized"
            metrics[name]["mean_dynamic_threshold"] = float(np.mean([
                row["dynamic_threshold"] for row in selected
            ]))

    freeze_rows = []
    for item, row in zip(data["validation"], groups["validation"]):
        for fraction in config["adaptive"].get("freeze_fractions", [0.05, 0.1, 0.2]):
            for onset_fraction in config["adaptive"].get("freeze_onsets", [0.2, 0.4, 0.6]):
                onset = max(2, int(len(item["global"]) * float(onset_fraction)))
                length = max(4, int(len(item["global"]) * float(fraction)))
                edited, end = _freeze_copy(item, onset, length)
                variants, detail = adaptive.score(base_model, edited, allow_warmup=False)
                base = detail["base"]["combined"]
                variants = {"fixed_cyclevad": base,
                            "intensity_threshold": base,
                            "intensity_threshold_with_stall": base,
                            **variants}
                thresholds = {"fixed_cyclevad": float(base_model.thresholds["combined"]), **adaptive.thresholds}
                for name, values in variants.items():
                    if name in {"intensity_threshold", "intensity_threshold_with_stall"}:
                        threshold = detail["intensity_threshold"]
                        alarm = values > threshold
                        if name == "intensity_threshold_with_stall":
                            alarm |= _k_of_n(variants["stall"], thresholds["stall"], k, window)
                    else:
                        threshold = thresholds[name]
                        alarm = _k_of_n(values, threshold, k, window) if name in {
                            "adaptive_ema", "stall", "adaptive_with_stall"
                        } else values > threshold
                    hit = np.flatnonzero(alarm[onset:end])
                    freeze_rows.append({"id": row["id"], "fraction": fraction,
                                        "onset_fraction": onset_fraction, "method": name,
                                        "hit": bool(len(hit)),
                                        "delay_sampled_frames": int(hit[0]) if len(hit) else None})
    for name in metrics:
        selected = [row for row in freeze_rows if row["method"] == name]
        metrics[name]["synthetic_freeze_cases"] = len(selected)
        metrics[name]["synthetic_freeze_event_recall"] = (
            sum(row["hit"] for row in selected) / len(selected) if selected else None
        )

    return metrics, per_video, freeze_rows, {
        "splits": splits, "adaptive_thresholds": adaptive.thresholds,
        "base_threshold": float(base_model.thresholds["combined"]),
        "expected_warmup_median": adaptive.expected_warmup_median,
        "expected_intensity_baseline": adaptive.expected_intensity_baseline,
        "test_labels_used_for_fit_calibration_or_threshold": False,
    }


def run(data_root, run_root, output, config):
    scenes = config.get("scenes", ["R01", "R02", "R03", "R04"])
    rows = inventory(data_root, scenes)
    all_metrics = {}
    for scene in scenes:
        metrics, per_video, freezes, protocol = run_scene(
            scene, [row for row in rows if row["scene"] == scene], run_root, config
        )
        folder = Path(output) / scene
        write_json(folder / "metrics.json", metrics)
        write_json(folder / "per_video.json", per_video)
        write_json(folder / "synthetic_freeze.json", freezes)
        write_json(folder / "protocol.json", protocol)
        all_metrics[scene] = metrics
        print(json.dumps({"scene": scene,
                          "fixed_fpr": metrics["fixed_cyclevad"]["fpr"],
                          "adaptive_fpr": metrics["adaptive_with_stall"]["fpr"],
                          "fixed_freeze": metrics["fixed_cyclevad"]["synthetic_freeze_event_recall"],
                          "stall_freeze": metrics["stall"]["synthetic_freeze_event_recall"]}), flush=True)
    methods = next(iter(all_metrics.values()))
    macro = {method: {key: float(np.mean([all_metrics[scene][method][key] for scene in scenes
                                          if all_metrics[scene][method][key] is not None]))
                      for key in ("auroc", "ap", "fpr", "recall", "event_coverage",
                                  "synthetic_freeze_event_recall")}
             for method in methods}
    write_json(Path(output) / "metrics_by_scene.json", all_metrics)
    write_json(Path(output) / "macro_metrics.json", macro)
    write_json(Path(output) / "config.json", config)
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run(args.data_root, args.run_root, args.output,
        json.loads(args.config.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
