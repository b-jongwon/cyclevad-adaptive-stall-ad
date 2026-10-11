"""Normal-only spatial, temporal, neural-head and fusion ablations on all scenes."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.decomposition import PCA
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from .adaptive_final import _ema, _event_metrics, _metrics
from .data import balanced_sample, hold, inventory, labels_for, load_cache, split_normal, write_json
from .model import TailCalibrator
from .pipeline import cache_path, load_rows
from .universal_alarm import SCENES, alarm_rule, low_baseline, read_stream


class Predictor(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.gru = nn.GRU(dimension, 64, batch_first=True)
        self.head = nn.Linear(64, dimension)

    def forward(self, x):
        return self.head(self.gru(x)[0][:, -1])


def contexts(x, length=8):
    ids = np.maximum(np.arange(len(x))[:, None] - np.arange(length, 0, -1)[None, :], 0)
    return x[ids]


def train_head(model, x, y, vx, vy, device):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    tx, ty = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
    tvx, tvy = torch.as_tensor(vx, device=device), torch.as_tensor(vy, device=device)
    best_loss, stale, best, history = float("inf"), 0, None, []
    for epoch in range(1, 31):
        model.train()
        total = 0.
        for indices in torch.randperm(len(tx), device=device).split(256):
            inputs = tx[indices]
            if inputs.ndim == 2:
                inputs = inputs + torch.randn_like(inputs) * .03
            loss = (model(inputs) - ty[indices]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        model.eval()
        with torch.inference_mode():
            validation = sum(float((model(a) - b).square().sum()) for a, b in
                             zip(tvx.split(512), tvy.split(512))) / tvy.numel()
        history.append({"epoch": epoch, "training_mse": total / len(tx), "normal_validation_mse": validation})
        if validation < best_loss * .999:
            best_loss, stale = validation, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 4:
                break
    model.load_state_dict(best)
    return model.eval(), history


def differences(array, lag=4):
    ids = np.maximum(np.arange(len(array)) - lag, 0)
    return array - array[ids]


class FeatureHeads:
    def fit(self, fit, validation, device):
        self.device = device
        torch.manual_seed(42)
        self.global_bank = torch.as_tensor(balanced_sample([d["global"] for d in fit], 1024), device=device)
        self.patch_bank = torch.as_tensor(balanced_sample([d["patches"] for d in fit], 512), device=device).transpose(0, 1).contiguous()
        self.motion_bank = torch.as_tensor(balanced_sample([differences(d["patches"]) for d in fit], 256), device=device).transpose(0, 1).contiguous()
        global_fit = balanced_sample([d["global"] for d in fit], 8192)
        self.pca = PCA(min(64, global_fit.shape[1], len(global_fit)-1), svd_solver="randomized", random_state=42).fit(global_fit)
        coords = self.pca.transform(global_fit)
        self.scale = np.maximum(coords.std(0), .01).astype(np.float32)
        project = lambda d: (self.pca.transform(d["global"]) / self.scale).astype(np.float32)
        train = [project(d) for d in fit]
        valid = [project(d) for d in validation]
        train_x = balanced_sample(train, 8192)
        valid_x = balanced_sample(valid, 4096)
        size = train_x.shape[1]
        ae = nn.Sequential(nn.Linear(size, 64), nn.GELU(), nn.Linear(64, 16), nn.GELU(),
                           nn.Linear(16, 64), nn.GELU(), nn.Linear(64, size))
        self.ae, ae_history = train_head(ae, train_x, train_x, valid_x, valid_x, device)
        paired_train = balanced_sample([np.concatenate([contexts(x).reshape(len(x), -1), x], 1) for x in train], 8192)
        paired_valid = balanced_sample([np.concatenate([contexts(x).reshape(len(x), -1), x], 1) for x in valid], 4096)
        self.gru, gru_history = train_head(Predictor(size), paired_train[:, :-size].reshape(-1, 8, size), paired_train[:, -size:],
                                         paired_valid[:, :-size].reshape(-1, 8, size), paired_valid[:, -size:], device)
        return {"autoencoder": ae_history, "gru": gru_history,
                "pca_dimension": size, "global_memory": len(self.global_bank),
                "patch_memory": self.patch_bank.shape[1], "motion_memory": self.motion_bank.shape[1]}

    def patch_distance(self, values, bank):
        output = []
        with torch.inference_mode():
            for chunk in torch.as_tensor(values, device=self.device).split(128):
                x = chunk.transpose(0, 1).contiguous()
                distances = (x.square().sum(-1, keepdim=True) + bank.square().sum(-1)[:, None, :]
                             - 2 * torch.bmm(x, bank.transpose(1, 2))).clamp_min(0).min(-1).values
                output.append(distances.T.cpu().numpy())
        return np.concatenate(output)

    def raw(self, data):
        global_values = torch.as_tensor(data["global"], device=self.device)
        global_score = []
        with torch.inference_mode():
            for x in global_values.split(256):
                distance = (x.square().sum(-1, keepdim=True) + self.global_bank.square().sum(-1)[None, :]
                            - 2 * x @ self.global_bank.T).clamp_min(0).min(-1).values
                global_score.append(distance.cpu().numpy())
        patches = self.patch_distance(data["patches"], self.patch_bank)
        motion = self.patch_distance(differences(data["patches"]), self.motion_bank)
        x = (self.pca.transform(data["global"]) / self.scale).astype(np.float32)
        tx, cx = torch.as_tensor(x, device=self.device), torch.as_tensor(contexts(x), device=self.device)
        with torch.inference_mode():
            ae = torch.cat([(self.ae(c) - c).square().mean(-1) for c in tx.split(512)]).cpu().numpy()
            gru = torch.cat([(self.gru(c) - t).square().mean(-1) for c, t in zip(cx.split(512), tx.split(512))]).cpu().numpy()
        return {"global_nn": np.concatenate(global_score), "patch_mean": patches.mean(1),
                "patch_top": np.sort(patches, axis=1)[:, -4:].mean(1),
                "motion": np.sort(motion, axis=1)[:, -4:].mean(1), "ae": ae, "gru": gru}

    def save(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pca": self.pca, "scale": self.scale,
                     "calibrators": self.calibrators}, folder / "normal_features.joblib", compress=3)
        torch.save({"ae": self.ae.state_dict(), "gru": self.gru.state_dict(),
                    "global_bank": self.global_bank.cpu(), "patch_bank": self.patch_bank.cpu(),
                    "motion_bank": self.motion_bank.cpu()}, folder / "heads.pt")


def compose(raw, base, calibrators):
    # Compress reference extrapolation so one unstable branch cannot dominate averages.
    q = {k: np.log1p(calibrators[k].score(v)) for k, v in raw.items()}
    b = np.log1p(calibrators["base"].score(base))
    proto = .5 * q["global_nn"] + .5 * q["patch_mean"]
    spatial_temporal = .7 * proto + .3 * q["motion"]
    full = .5 * proto + .2 * q["motion"] + .15 * q["ae"] + .15 * q["gru"]
    return {"baseline": base, **q, "prototype": proto, "spatial_temporal": spatial_temporal,
            "learned": .5 * q["ae"] + .5 * q["gru"], "full_features": full,
            "cycle_prototype": .5 * b + .5 * proto, "cycle_full": .25 * b + .75 * full,
            "prototype_ema": _ema(proto, .4), "full_ema": _ema(full, .4),
            "cycle_full_ema": _ema(.25 * b + .75 * full, .4)}


def run_scene(scene, rows, root, alarm_root, output, device):
    start = time.perf_counter()
    folder = output / scene
    folder.mkdir(parents=True, exist_ok=True)
    normal = [r for r in rows if r["partition"] == "training"]
    splits = split_normal(normal, 42)
    groups = {k: [r for r in normal if r["id"] in ids] for k, ids in splits.items()}
    data = {k: load_rows(root, v) for k, v in groups.items()}
    base = joblib.load(root / scene / "model.joblib")
    heads = FeatureHeads()
    history = heads.fit(data["fit"], data["validation"], device)
    reference_raw = [heads.raw(d) for d in data["reference"]]
    reference_base = [base.score(d)["combined"] for d in data["reference"]]
    calibrators = {k: TailCalibrator().fit(np.concatenate([r[k] for r in reference_raw])) for k in reference_raw[0]}
    calibrators["base"] = TailCalibrator().fit(np.concatenate(reference_base))
    heads.calibrators = calibrators
    refs = [compose(r, b, calibrators) for r, b in zip(reference_raw, reference_base)]
    baselines = {k: float(np.median([low_baseline(s[k]) for s in refs])) for k in refs[0]}
    thresholds_raw = [compose(heads.raw(d), base.score(d)["combined"], calibrators) for d in data["threshold"]]
    thresholds = {k: float(np.quantile(np.concatenate([
        hold(d["indices"], s[k], int(d["frame_count"])) for d, s in zip(data["threshold"], thresholds_raw)
    ]), .99, method="higher")) for k in refs[0]}
    thresholds["baseline"] = float(base.thresholds["combined"])
    # Universal alarm hyperparameters were frozen in the earlier normal-only study.
    alarm_parameters = json.loads((alarm_root / "selected_rule.json").read_text(encoding="utf-8"))
    manifest = json.loads((alarm_root / "streams" / scene / "manifest.json").read_text(encoding="utf-8"))
    alarm_lookup = {r["id"]: r for r in manifest if r["kind"] == "testing"}
    joined, reports = {}, []
    inference = 0.
    for row in [r for r in rows if r["partition"] == "testing"]:
        d = load_cache(cache_path(root, row))
        tick = time.perf_counter()
        original = read_stream(alarm_lookup[row["id"]])
        scores = compose(heads.raw(d), original["score"], calibrators)
        inference += time.perf_counter() - tick
        # Store predicted scores before touching annotations.
        target = folder / "predictions" / (row["sequence"] + ".npz")
        target.parent.mkdir(exist_ok=True)
        np.savez_compressed(target, indices=d["indices"], **scores)
        decisions = {}
        for k, s in scores.items():
            stream = {**original, "score": s, "base_threshold": thresholds[k], "normal_baseline": baselines[k]}
            decisions[k] = alarm_rule(stream, alarm_parameters)[0]
        labels = labels_for(row)
        for k, s in scores.items():
            fs = hold(d["indices"], s, row["frames"])
            fa = hold(d["indices"], decisions[k], row["frames"])
            record = {"id": row["id"], "method": k, **_metrics(labels, fs, thresholds[k], fa),
                      **_event_metrics(labels, fa)}
            reports.append(record)
            joined.setdefault(k, [[], [], []])
            joined[k][0].append(labels); joined[k][1].append(fs); joined[k][2].append(fa)
    metrics = {}
    for k, (ys, ss, aa) in joined.items():
        y, s, a = np.concatenate(ys), np.concatenate(ss), np.concatenate(aa)
        tp, fp, fn = np.sum(a & (y == 1)), np.sum(a & (y == 0)), np.sum(~a & (y == 1))
        recs = [r for r in reports if r["method"] == k]
        ev, de = sum(r["events"] for r in recs), sum(r["detected_events"] for r in recs)
        metrics[k] = {**_metrics(y, s, thresholds[k], a),
                      "precision": float(tp/(tp+fp)) if tp+fp else 0.,
                      "f1": float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,
                      "event_coverage": de/ev if ev else None, "events": ev, "detected_events": de}
        metrics[k]["balanced_accuracy"] = (1 - metrics[k]["fpr"] + metrics[k]["recall"]) / 2
    heads.save(folder / "models")
    write_json(folder / "metrics.json", metrics)
    write_json(folder / "training.json", history)
    write_json(folder / "per_video.json", reports)
    write_json(folder / "protocol.json", {"splits": splits, "thresholds": thresholds,
                "alarm_parameters": alarm_parameters, "test_labels_used_for_training": False,
                "pretrained_encoder_updated": False, "seconds": time.perf_counter()-start,
                "head_scoring_seconds": inference, "comparisons_are_exploratory_on_previously_inspected_test_set": True})
    print(json.dumps({"scene": scene, "epochs_ae": len(history["autoencoder"]), "epochs_gru": len(history["gru"]),
                      "seconds": round(time.perf_counter()-start, 1)}), flush=True)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("data-root", "run-root", "alarm-root", "output"):
        p.add_argument("--"+k, type=Path, required=True)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)
    rows = inventory(args.data_root, SCENES)
    results = {}
    with threadpool_limits(limits=2):
        for scene in SCENES:
            path = args.output / scene / "metrics.json"
            if path.exists():
                results[scene] = json.loads(path.read_text(encoding="utf-8"))
            else:
                results[scene] = run_scene(scene, [r for r in rows if r["scene"] == scene],
                                           args.run_root, args.alarm_root, args.output, device)
    keys = ("auroc", "ap", "fpr", "recall", "precision", "f1", "event_coverage", "balanced_accuracy")
    methods = next(iter(results.values())).keys()
    macro = {m: {k: float(np.mean([r[m][k] for r in results.values() if r[m][k] is not None])) for k in keys} for m in methods}
    write_json(args.output / "metrics_by_scene.json", results)
    write_json(args.output / "macro_metrics.json", macro)
    write_json(args.output / "execution.json", {"device": device, "scenes": SCENES,
               "method_count": len(macro), "seed": 42, "max_epochs": 30, "patience": 4,
               "normal_only_training": True, "test_comparison_status": "exploratory; no fresh untouched test set"})
    print(json.dumps(macro), flush=True)


if __name__ == "__main__":
    main()
