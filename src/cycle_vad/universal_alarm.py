"""One validation-selected alarm rule for all 16 IPAD scenes; unchanged ranking."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .adaptive_final import ConditionalEmpiricalTail, _ema, _event_metrics, _freeze_copy, _metrics, _motion
from .data import hold, inventory, labels_for, load_cache, split_normal, write_json
from .pipeline import cache_path, load_rows, validate_model_cache


SCENES = [f"R{i:02d}" for i in range(1, 5)] + [f"S{i:02d}" for i in range(1, 13)]


def low_baseline(values, count=24):
    early = np.asarray(values)[:count]
    return float(np.median(np.sort(early)[:max(1, int(np.ceil(len(early) * .2)))]))


def consecutive_alarm(raw, length):
    raw = np.asarray(raw, dtype=bool)
    count = 0
    output = np.zeros(len(raw), dtype=bool)
    for i, value in enumerate(raw):
        count = count + 1 if value else 0
        output[i] = count >= length
    return output


def alarm_rule(stream, parameters):
    """Causal: first 24 observations use the original threshold, then freeze offset."""
    values = stream["score"]
    threshold = np.full(len(values), stream["base_threshold"], dtype=float)
    if len(values) >= 24:
        offset = max(0., low_baseline(values) - stream["normal_baseline"])
        threshold[23:] += parameters["gain"] * offset
    appearance = consecutive_alarm(values > threshold, parameters["persistence"])
    if parameters["stall"]:
        stall = consecutive_alarm(stream["stall"] > stream["stall_threshold"], 3)
        appearance |= stall
    return appearance, threshold


def prepare_scene(scene, rows, root, output):
    model = joblib.load(root / scene / "model.joblib")
    normal = [r for r in rows if r["partition"] == "training"]
    splits = split_normal(normal, 42)
    groups = {name: [r for r in normal if r["id"] in ids] for name, ids in splits.items()}
    reference = load_rows(root, groups["reference"])
    scored_reference = [model.score(d) for d in reference]
    baselines = [low_baseline(s["combined"]) for s in scored_reference]
    motion = {
        lag: ConditionalEmpiricalTail(16, 64).fit(
            np.concatenate([_motion(d["global"], lag) for d in reference]),
            np.concatenate([s["angle"] for s in scored_reference]),
        ) for lag in (1, 4)
    }

    def score(d):
        validate_model_cache(model, d)
        s = model.score(d)
        stall = _ema(np.minimum.reduce([
            tail.score(_motion(d["global"], lag), s["angle"], upper=False)
            for lag, tail in motion.items()
        ]), .5)
        return {"score": s["combined"], "stall": stall,
                "indices": d["indices"], "frames": int(d["frame_count"]),
                "normal_baseline": float(np.median(baselines)),
                "base_threshold": float(model.thresholds["combined"])}

    threshold_streams = [score(load_cache(cache_path(root, r))) for r in groups["threshold"]]
    stall_threshold = float(np.quantile(np.concatenate([
        hold(s["indices"], s["stall"], s["frames"]) for s in threshold_streams
    ]), .995, method="higher"))
    folder = output / "streams" / scene
    folder.mkdir(parents=True, exist_ok=True)
    manifest = []

    def save(stream, row, kind, event=None):
        stream["stall_threshold"] = stall_threshold
        key = row["sequence"] + ("_" + kind if kind != "testing" else "_test")
        path = folder / (key + ".npz")
        np.savez_compressed(path, **stream)
        manifest.append({"path": str(path), "id": row["id"], "scene": scene,
                         "kind": kind, "event": event,
                         "labels": row["labels"], "frames": row["frames"]})

    for row in groups["validation"]:
        d = load_cache(cache_path(root, row))
        save(score(d), row, "normal_validation")
        n = len(d["global"])
        onset, length = max(24, int(n * .4)), max(8, int(n * .1))
        if onset >= n - 3:
            continue
        frozen, end = _freeze_copy(d, onset, length)
        save(score(frozen), row, "freeze_validation", [onset, end])
        reversed_data = {k: np.array(v, copy=True) for k, v in d.items()}
        for k in ("global", "patches"):
            reversed_data[k][onset:end] = d[k][onset:end][::-1]
        save(score(reversed_data), row, "reverse_validation", [onset, end])
    for row in [r for r in rows if r["partition"] == "testing"]:
        save(score(load_cache(cache_path(root, row))), row, "testing")
    joblib.dump({"normal_baseline": float(np.median(baselines)), "motion": motion,
                 "stall_threshold": stall_threshold, "splits": splits},
                output / f"{scene}_alarm_calibration.joblib", compress=3)
    write_json(output / f"{scene}_protocol.json", {
        "splits": splits, "base_threshold": float(model.thresholds["combined"]),
        "stall_threshold": stall_threshold, "normal_baseline": float(np.median(baselines)),
        "ranking_score": "original CycleVAD combined, unchanged",
        "test_labels_used_for_selection": False,
    })
    write_json(folder / "manifest.json", manifest)
    return manifest


def read_stream(row):
    with np.load(row["path"], allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def select_rule(manifest, output):
    # Freeze the global rule before opening any real test annotation.
    validation = [(row, read_stream(row)) for row in manifest if row["kind"] != "testing"]
    candidates = []
    for gain, persistence, stall in itertools.product([0., .1, .2, .3], [1, 2, 3], [False, True]):
        parameters = {"gain": gain, "persistence": persistence, "stall": stall}
        normal_fpr = {s: [] for s in SCENES}
        synthetic_recall = {s: [] for s in SCENES}
        synthetic_events = {s: [] for s in SCENES}
        for row, stream in validation:
            alarm, _ = alarm_rule(stream, parameters)
            if row["kind"] == "normal_validation":
                normal_fpr[row["scene"]].append(float(hold(stream["indices"], alarm, int(stream["frames"])).mean()))
            else:
                start, end = row["event"]
                synthetic_recall[row["scene"]].append(float(alarm[start:end].mean()))
                synthetic_events[row["scene"]].append(float(alarm[start:end].any()))
        per_scene_fpr = {s: float(np.mean(v)) for s, v in normal_fpr.items()}
        candidates.append({"parameters": parameters,
                           "normal_fpr": float(np.mean(list(per_scene_fpr.values()))),
                           "per_scene_normal_fpr": per_scene_fpr,
                           "synthetic_frame_recall": float(np.mean([np.mean(v) for v in synthetic_recall.values()])),
                           "synthetic_event_coverage": float(np.mean([np.mean(v) for v in synthetic_events.values()]))})
    baseline = next(c for c in candidates if c["parameters"] == {"gain": 0., "persistence": 1, "stall": False})
    # Global objective, never a per-scene hand-picked rule. Normal-FPR tolerance
    # is 0.5 percentage point macro and 2 percentage points per scene.
    eligible = [c for c in candidates if c["normal_fpr"] <= baseline["normal_fpr"] + .005
                and all(c["per_scene_normal_fpr"][s] <= baseline["per_scene_normal_fpr"][s] + .02 for s in SCENES)
                and c["synthetic_frame_recall"] >= baseline["synthetic_frame_recall"] - .01]
    chosen = max(eligible, key=lambda c: (
        c["synthetic_event_coverage"] + c["synthetic_frame_recall"] - 5 * c["normal_fpr"],
        -c["parameters"]["gain"], -c["parameters"]["persistence"],
    ))
    write_json(output / "validation_selection.json", {
        "baseline": baseline, "chosen": chosen, "candidates": candidates,
        "candidate_count": len(candidates), "eligible_count": len(eligible),
        "selection_uses_real_anomaly_test_labels": False,
        "synthetic_anomalies": ["feature freeze", "feature subsequence reversal"],
    })
    write_json(output / "selected_rule.json", chosen["parameters"])
    return chosen["parameters"]


def evaluate(manifest, parameters, output):
    aggregate = {}
    per_video = []
    for scene in SCENES:
        joined = {name: [[], [], []] for name in ("baseline", "universal")}
        for row in [r for r in manifest if r["scene"] == scene and r["kind"] == "testing"]:
            stream = read_stream(row)
            validate = {"gain": 0., "persistence": 1, "stall": False}
            alarms = {"baseline": alarm_rule(stream, validate)[0],
                      "universal": alarm_rule(stream, parameters)[0]}
            scores = hold(stream["indices"], stream["score"], row["frames"])
            # This is the first test-label IO in this experiment.
            labels = labels_for({**row, "partition": "testing"})
            for name, sampled_alarm in alarms.items():
                alarm = hold(stream["indices"], sampled_alarm, row["frames"])
                report = {"id": row["id"], "method": name,
                          **_metrics(labels, scores, float(stream["base_threshold"]), alarm),
                          **_event_metrics(labels, alarm)}
                valid_normal = (labels == 0)
                transitions = np.diff(np.r_[False, alarm].astype(int)) == 1
                report["false_alarm_onsets"] = int(np.sum(transitions & valid_normal))
                per_video.append(report)
                joined[name][0].append(labels)
                joined[name][1].append(scores)
                joined[name][2].append(alarm)
        aggregate[scene] = {}
        for name, (ys, ss, aa) in joined.items():
            selected = [r for r in per_video if r["id"].startswith(scene + "/") and r["method"] == name]
            events = sum(r["events"] for r in selected)
            detected = sum(r["detected_events"] for r in selected)
            metrics = _metrics(np.concatenate(ys), np.concatenate(ss), 0., np.concatenate(aa))
            metrics.pop("threshold")
            y, a = np.concatenate(ys), np.concatenate(aa)
            valid = y >= 0
            tp, fp = int(np.sum(a & (y == 1))), int(np.sum(a & (y == 0)))
            fn = int(np.sum(~a & (y == 1)))
            metrics.update({"precision": tp / (tp + fp) if tp + fp else 0.,
                            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                            "specificity": 1. - metrics["fpr"],
                            "balanced_accuracy": (1. - metrics["fpr"] + metrics["recall"]) / 2.,
                            "events": events, "detected_events": detected,
                            "event_coverage": detected / events if events else None,
                            "false_alarm_onsets": sum(r["false_alarm_onsets"] for r in selected),
                            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
                            "true_negatives": int(np.sum(~a & (y == 0)))})
            aggregate[scene][name] = metrics
        assert aggregate[scene]["baseline"]["auroc"] == aggregate[scene]["universal"]["auroc"]
        assert aggregate[scene]["baseline"]["ap"] == aggregate[scene]["universal"]["ap"]
    keys = ("auroc", "ap", "fpr", "recall", "precision", "f1", "specificity", "balanced_accuracy", "event_coverage")
    macro = {name: {k: float(np.mean([aggregate[s][name][k] for s in SCENES
                                     if aggregate[s][name][k] is not None])) for k in keys}
             for name in ("baseline", "universal")}
    write_json(output / "metrics_by_scene.json", aggregate)
    write_json(output / "macro_metrics.json", macro)
    write_json(output / "per_video.json", per_video)
    write_json(output / "score_invariance.json", {
        "all_16_scenes_auroc_identical": True, "all_16_scenes_ap_identical": True,
        "score_arrays_identical_by_construction": True,
        "note": "Ranking score and alarm decisions are separate outputs; stall affects alarm only.",
    })
    return macro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    rows = inventory(args.data_root, SCENES)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    with threadpool_limits(limits=2):
        for scene in SCENES:
            path = args.output / "streams" / scene / "manifest.json"
            if path.exists():
                manifest.extend(json.loads(path.read_text(encoding="utf-8")))
            else:
                manifest.extend(prepare_scene(scene, [r for r in rows if r["scene"] == scene],
                                              args.run_root, args.output))
            print(f"prepared {scene}", flush=True)
        parameters = select_rule(manifest, args.output)
        print("selected " + json.dumps(parameters), flush=True)
        metrics = evaluate(manifest, parameters, args.output)
    write_json(args.output / "execution.json", {
        "scenes": SCENES, "scene_count": 16, "seconds": time.perf_counter() - start,
        "real_test_labels_used_for_rule_selection": False,
        "preexisting_test_experiments": "R01-R04 previously explored; S01-S12 not used to tune this rule",
        "normal_scene_calibration": "separate per-scene normal distribution, identical fitting algorithm",
    })
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
