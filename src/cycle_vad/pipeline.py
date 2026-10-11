"""Colab/CLI orchestration. Fitting cannot access anomalous test annotations."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .data import (cycle_segments, fingerprint, hold, inventory, labels_for,
                   load_cache, split_normal, write_json)
from .metrics import event_metrics, frame_metrics
from .model import CycleVAD


def cache_path(output, row):
    return Path(output)/"cache"/(row["id"]+".npz")


def load_rows(output, rows):
    data = [load_cache(cache_path(output, r)) for r in rows]
    identities = {str(d["encoder_identity"]) for d in data}
    if len(identities) != 1:
        raise ValueError("Mixed encoder caches: re-extract all selected scenes with one configuration")
    for d, row in zip(data, rows):
        if int(d["frame_count"]) != row["frames"]:
            raise ValueError(f"Stale frame count in feature cache: {row['id']}")
    return data


def fit_scene(rows, output, config, boundaries=None, groups=None):
    normal = [r for r in rows if r["partition"] == "training"]
    scene = normal[0]["scene"]
    splits = split_normal(normal, config.get("seed", 42), groups)
    row_groups = {name: [r for r in normal if r["id"] in ids] for name, ids in splits.items()}
    data = {name: load_rows(output, group) for name, group in row_groups.items()}
    identities = {str(d["encoder_identity"]) for group in data.values() for d in group}
    if len(identities) != 1:
        raise ValueError("Normal splits use different feature extractors")
    for group in data.values():
        for d in group:
            if not np.all(np.diff(d["indices"]) == config["stride"]):
                raise ValueError("Training cache stride differs from configuration; re-extract features")
    segments, sources = [], {}
    for d, row in zip(data["fit"], row_groups["fit"]):
        parts, source = cycle_segments(d, row, boundaries)
        segments.append(parts)
        sources[row["id"]] = source
    start = time.perf_counter()
    model = CycleVAD(config["model"])
    model.fit(data["fit"], data["validation"], segments)
    model.calibrate(data["reference"], data["threshold"])
    model.encoder_identity = next(iter(identities))
    model.stride = config["stride"]
    model.scene = scene
    model.run_config = config
    model.diagnostics.update({"scene": scene, "splits": splits, "cycle_coordinate_sources": sources,
                              "recording_groups_provided": bool(groups),
                              "group_independence_verified": False,
                              "fit_calibrate_seconds": time.perf_counter()-start,
                              "normal_only_model_selection": True,
                              "cycle_accuracy": None, "real_anomaly_performance": "not_yet_evaluated",
                              "frame_time_unit": "source_frame_index"})
    folder = Path(output)/scene
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder/"model.joblib"
    tmp = checkpoint.with_suffix(".tmp")
    joblib.dump(model, tmp, compress=3)
    os.replace(tmp, checkpoint)
    write_json(folder/"fit_report.json", model.diagnostics)
    write_json(folder/"protocol.json", {"config": config, "splits": splits,
                                       "encoder_identity": json.loads(model.encoder_identity),
                                       "normal_cache_signatures": {row["id"]: str(d["signature"])
                                            for name, group in row_groups.items() for row, d in zip(group, data[name])},
                                       "code_fingerprint": code_fingerprint()})
    print(f"[fitted] {scene}: thresholds={model.thresholds}", flush=True)
    return model


def code_fingerprint():
    return fingerprint({p.name: p.read_text(encoding="utf-8") for p in Path(__file__).parent.glob("*.py")})


def validate_model_cache(model, data):
    if str(data["encoder_identity"]) != model.encoder_identity:
        raise ValueError("Prediction cache was extracted with a different encoder configuration")
    if not np.all(np.diff(data["indices"]) == model.stride):
        raise ValueError("Sampling stride differs from normal calibration")


def save_prediction(model, row, data, output):
    validate_model_cache(model, data)
    prediction = model.score(data)
    path = Path(output)/row["scene"] / "predictions" / f"{row['partition']}_{row['sequence']}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".tmp").open("wb") as stream:
        np.savez_compressed(stream, indices=data["indices"], frame_count=data["frame_count"], **prediction)
    os.replace(path.with_suffix(".tmp"), path)
    return prediction, path


def evaluate_scene(model, rows, output):
    selected = [r for r in rows if r["partition"] == "testing"]
    labels, scores = [], {name: [] for name in model.thresholds}
    reports = []
    for row in selected:
        data = load_cache(cache_path(output, row))
        prediction, _ = save_prediction(model, row, data, output)
        # Label IO occurs only AFTER inference. Unknown lengths never alter
        # feature extraction, cycle tracking, scores, or the saved prediction.
        y = labels_for(row)
        labels.append(y)
        for name, threshold in model.thresholds.items():
            values = hold(data["indices"], prediction[name], row["frames"])
            scores[name].append(values)
            reports.append({"id": row["id"], "branch": name,
                            **frame_metrics(y, values, threshold), **event_metrics(y, values, threshold),
                            "mean_cycle_confidence": float(prediction["confidence"].mean())})
    if not labels:
        raise ValueError("No test sequences to evaluate")
    aggregate = {}
    for name, threshold in model.thresholds.items():
        events = [r for r in reports if r["branch"] == name]
        count = sum(r["events"] for r in events)
        detected = sum(r["detected_events"] for r in events)
        delays = [d for r in events for d in r["detected_event_delays_source_frames"]]
        aggregate[name] = {**frame_metrics(np.concatenate(labels), np.concatenate(scores[name]), threshold),
                           "events": count, "detected_events": detected,
                           "event_coverage": detected/count if count else None,
                           "median_detected_event_delay_frames": float(np.median(delays)) if delays else None}
    folder = Path(output)/model.scene
    write_json(folder/"metrics.json", aggregate)
    write_json(folder/"per_sequence.json", reports)
    with (folder/"per_sequence.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
    print(json.dumps({model.scene: aggregate}, indent=2), flush=True)
    return aggregate


def environment():
    packages = {}
    for name in ("torch", "torchvision", "transformers", "numpy", "scipy", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def run(data_root, output, config, stage="all", boundaries=None, groups=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = inventory(data_root, config["scenes"])
    write_json(output/"manifest.json", rows)
    write_json(output/"environment.json", environment())
    write_json(output/"config.json", config)
    encoder = None
    if stage in ("all", "extract"):
        from .features import VisualEncoder
        encoder = VisualEncoder(config["encoder"])
        write_json(output/"encoder.json", encoder.identity)
    results = {}
    # Bound peak host RAM by processing only one industrial scene at a time.
    for scene in config["scenes"]:
        scene_rows = [r for r in rows if r["scene"] == scene]
        if encoder is not None:
            from .features import extract_sequence
            for row in scene_rows:
                extract_sequence(encoder, row, output/"cache", config["stride"],
                                 config.get("batch_size", 8), config.get("workers", 4))
        if stage in ("all", "fit"):
            model = fit_scene(scene_rows, output, config, boundaries, groups)
        elif stage == "evaluate":
            # joblib is intentionally local/trusted model persistence only.
            model = joblib.load(output/scene/"model.joblib")
            if model.stride != config["stride"] or model.config != config["model"]:
                raise ValueError("Evaluation configuration differs from the fitted checkpoint")
        else:
            continue
        if stage in ("all", "evaluate"):
            results[scene] = evaluate_scene(model, scene_rows, output)
    if results:
        write_json(output/"metrics_by_scene.json", results)
        macro = {}
        for branch in next(iter(results.values())):
            macro[branch] = {}
            for key in ("auroc", "ap", "fpr", "recall", "event_coverage"):
                values = [r[branch][key] for r in results.values() if r[branch][key] is not None]
                macro[branch][key] = float(np.mean(values)) if values else None
        write_json(output/"macro_metrics.json", macro)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", default="configs/cycle_colab.json", type=Path)
    parser.add_argument("--stage", choices=["all", "extract", "fit", "evaluate"], default="all")
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--boundaries", type=Path, help="Normal FIT cycle boundaries JSON, source-frame units")
    parser.add_argument("--groups", type=Path, help="Normal recording group JSON; keys are scene/training/sequence")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.scenes:
        config["scenes"] = args.scenes
    boundaries = json.loads(args.boundaries.read_text(encoding="utf-8")) if args.boundaries else None
    groups = json.loads(args.groups.read_text(encoding="utf-8")) if args.groups else None
    with threadpool_limits(limits=args.cpu_threads):
        run(args.data_root, args.output, config, args.stage, boundaries, groups)


if __name__ == "__main__":
    main()
