"""Explore lower-motion ranking fusion; all-scene shared ratios, no head retraining."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from .adaptive_final import _ema, _event_metrics, _metrics
from .data import hold, labels_for, load_cache, write_json
from .model import TailCalibrator
from .pipeline import cache_path
from .universal_alarm import SCENES, alarm_rule, low_baseline, read_stream


def run_lengths(mask):
    result = np.zeros(len(mask), dtype=float)
    length = 0
    for index, value in enumerate(mask):
        length = length + 1 if value else 0
        result[index] = length
    return result


def compose(stream, duration_limit):
    base = np.asarray(stream["score"])
    # Both scales are learned from disjoint normal reference/threshold splits.
    b = base / max(float(stream["base_threshold"]), 1e-6)
    stall = np.asarray(stream["stall"]) / max(float(stream["stall_threshold"]), 1e-6)
    values = {"baseline": base, "base_ema": _ema(base, .4), "stall_ranking": stall}
    duration = run_lengths(stall > .8) / max(float(duration_limit), 3.)
    excess_duration = np.maximum(duration - 1., 0.)
    for weight in (.1, .2, .5, 1.):
        values[f"duration_add_{weight}"] = b + weight * excess_duration
        values[f"duration_max_{weight}"] = np.maximum(b, weight * duration)
        values[f"duration_gated_{weight}"] = np.maximum(b, stall * np.minimum(duration, 1.) * weight)
    for w in (.05, .1, .2, .3, .5, .75):
        values[f"mix_{w}"] = (1-w) * b + w * stall
        values[f"mix_ema_{w}"] = _ema(values[f"mix_{w}"], .4)
        values[f"log_mix_{w}"] = (1-w) * np.log1p(b) + w * np.log1p(stall)
        values[f"max_mix_{w}"] = np.maximum(b, stall * (1+w))
    return values


def predict_fusion(data, bundle):
    """Executable fitted pipeline; annotations are not accepted or accessed."""
    from .adaptive_final import _motion
    s = bundle["base_model"].score(data)
    normal = bundle["motion_calibration"]
    st = _ema(np.minimum.reduce([
        tail.score(_motion(data["global"], lag), s["angle"], upper=False)
        for lag, tail in normal["motion"].items()
    ]), .5)
    stream = {"score": s["combined"], "stall": st,
              "base_threshold": bundle["base_model"].thresholds["combined"],
              "stall_threshold": normal["stall_threshold"]}
    method = bundle["selected_method"]
    score = compose(stream, bundle["duration_limit"])[method]
    stream.update(score=score, base_threshold=bundle["thresholds"][method],
                  normal_baseline=bundle["baselines"][method])
    alarms, threshold = alarm_rule(stream, bundle["alarm_parameters"])
    return {"score": score, "alarm": alarms, "indices": data["indices"],
            "dynamic_threshold": threshold, "method": method}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--alarm-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    parameters = json.loads((args.alarm_root / "selected_rule.json").read_text(encoding="utf-8"))
    aggregate = {}
    for scene in SCENES:
        folder = args.output / scene
        folder.mkdir(parents=True, exist_ok=True)
        model = joblib.load(args.run_root / scene / "model.joblib")
        calibration = joblib.load(args.alarm_root / f"{scene}_alarm_calibration.joblib")
        metadata = json.loads((args.alarm_root / "streams" / scene / "manifest.json").read_text(encoding="utf-8"))
        lookup = {r["id"]: r for r in metadata if r["kind"] == "normal_validation"}
        # Reuse the normal-only stall calibrator in every branch.
        def make_normal(identity):
            data = load_cache(args.run_root / "cache" / (identity + ".npz"))
            s = model.score(data)
            from .adaptive_final import _motion
            st = _ema(np.minimum.reduce([tail.score(_motion(data["global"], lag), s["angle"], upper=False)
                         for lag, tail in calibration["motion"].items()]), .5)
            return {"score": s["combined"], "stall": st, "indices": data["indices"],
                    "frames": int(data["frame_count"]), "base_threshold": model.thresholds["combined"],
                    "stall_threshold": calibration["stall_threshold"]}
        reference = [make_normal(i) for i in calibration["splits"]["reference"]]
        threshold = [make_normal(i) for i in calibration["splits"]["threshold"]]
        durations = [run_lengths(s["stall"] > s["stall_threshold"] * .8) for s in reference]
        occupied = np.concatenate(durations)
        duration_limit = max(3., float(np.quantile(occupied, .995)))
        refs = [compose(s, duration_limit) for s in reference]
        thresholds = [compose(s, duration_limit) for s in threshold]
        names = list(refs[0])
        normal_baselines = {k: float(np.median([low_baseline(r[k]) for r in refs])) for k in names}
        normal_thresholds = {k: float(np.quantile(np.concatenate([
            hold(s["indices"], t[k], int(s["frames"])) for s, t in zip(threshold, thresholds)
        ]), .99, method="higher")) for k in names}
        normal_thresholds["baseline"] = float(model.thresholds["combined"])
        bundle = {"base_model": model, "motion_calibration": calibration,
                  "duration_limit": duration_limit, "thresholds": normal_thresholds,
                  "baselines": normal_baselines, "alarm_parameters": parameters,
                  "selected_method": "max_mix_0.1",
                  "selection_status": "exploratory choice after test comparison; independent confirmation required"}
        joblib.dump(bundle, folder / "balanced_pipeline.joblib", compress=3)
        collected = {k: [[], [], []] for k in names}
        reports = []
        for row in [r for r in metadata if r["kind"] == "testing"]:
            stream = read_stream(row)
            variants = compose(stream, duration_limit)
            predictions, alarms = {}, {}
            for name, values in variants.items():
                modified = {**stream, "score": values, "normal_baseline": normal_baselines[name],
                            "base_threshold": normal_thresholds[name]}
                alarms[name] = hold(stream["indices"], alarm_rule(modified, parameters)[0], row["frames"])
                predictions[name] = hold(stream["indices"], values, row["frames"])
            y = labels_for({**row, "partition": "testing"})
            for name, values in predictions.items():
                a = alarms[name]
                reports.append({"id": row["id"], "method": name,
                                **_event_metrics(y, a), **_metrics(y, values, normal_thresholds[name], a)})
                collected[name][0].append(y); collected[name][1].append(values); collected[name][2].append(a)
        metrics = {}
        for name, (ys, ss, aa) in collected.items():
            y, s, a = np.concatenate(ys), np.concatenate(ss), np.concatenate(aa)
            tp, fp, fn = np.sum(a & (y == 1)), np.sum(a & (y == 0)), np.sum(~a & (y == 1))
            records = [r for r in reports if r["method"] == name]
            ev, de = sum(r["events"] for r in records), sum(r["detected_events"] for r in records)
            metrics[name] = {**_metrics(y, s, normal_thresholds[name], a),
                             "precision": float(tp/(tp+fp)) if tp+fp else 0.,
                             "f1": float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,
                             "event_coverage": de/ev if ev else None}
            metrics[name]["balanced_accuracy"] = (1-metrics[name]["fpr"]+metrics[name]["recall"])/2
        write_json(folder / "metrics.json", metrics)
        write_json(folder / "per_video.json", reports)
        write_json(folder / "protocol.json", {"normal_thresholds": normal_thresholds,
                   "normal_baselines": normal_baselines, "splits": calibration["splits"],
                   "normal_duration_limit": duration_limit,
                   "test_labels_used_for_fit": False, "comparisons_are_exploratory": True})
        aggregate[scene] = metrics
        print(scene, flush=True)
    keys = ("auroc", "ap", "fpr", "recall", "precision", "f1", "event_coverage", "balanced_accuracy")
    macro = {m: {k: float(np.mean([r[m][k] for r in aggregate.values() if r[m][k] is not None])) for k in keys}
             for m in next(iter(aggregate.values()))}
    write_json(args.output / "metrics_by_scene.json", aggregate)
    write_json(args.output / "macro_metrics.json", macro)
    print(json.dumps(macro), flush=True)


if __name__ == "__main__":
    main()
