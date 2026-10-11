"""Continuous conditional mean + ONE shared residual subspace per scene.

Independent pooled/local appearance channels keep low-confidence localization
from suppressing anomalies. All component scales use separate normal reference
videos; the alarm threshold uses a further disjoint normal holdout.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from .data import balanced_sample
from .tracker import CycleTracker, fourier


def descriptor(data):
    global_z = np.asarray(data["global"], dtype=np.float32)
    patches = np.asarray(data["patches"], dtype=np.float32)
    return np.concatenate([global_z/np.float32(np.sqrt(2)),
                           patches.reshape(len(patches), -1)/np.float32(np.sqrt(2*patches.shape[1]))], axis=1)


class ResidualSpace:
    def fit(self, x, rank=64, variance=.95, seed=42):
        rank = min(rank, x.shape[1]-1, len(x)-2)
        if rank < 1:
            raise ValueError("Insufficient normal features for shared subspace")
        pca = PCA(rank, svd_solver="randomized", random_state=seed).fit(x)
        cumulative = np.cumsum(np.nan_to_num(pca.explained_variance_ratio_))
        kept = min(rank, int(np.searchsorted(cumulative, variance))+1)
        self.mean = pca.mean_
        self.basis = pca.components_[:kept]
        self.eigen = np.maximum(pca.explained_variance_[:kept], 1e-7)
        self.diagnostics = {"rank": kept, "samples": len(x),
                            "retained_variance": float(cumulative[kept-1]),
                            "variance_target_met": bool(cumulative[kept-1] >= variance)}
        return self

    def score(self, x):
        centered = x-self.mean
        coordinates = centered @ self.basis.T
        residual = centered-coordinates @ self.basis
        outside = np.mean(residual**2, axis=1)
        inside = np.mean(coordinates**2/self.eigen, axis=1)
        return outside, inside, residual


class TailCalibrator:
    """Monotone log-tail with extrapolation above reference max; no q99=1 trap."""
    def fit(self, values):
        self.reference = np.sort(np.asarray(values, dtype=float))
        if not len(self.reference) or not np.isfinite(self.reference).all():
            raise ValueError("Empty/nonfinite normal calibration reference")
        q50, q90, q99 = np.quantile(self.reference, [.5, .9, .99])
        self.scale = max(q99-q90, (q90-q50)*.1, abs(q99)*1e-3, 1e-6)
        return self

    def score(self, values):
        values = np.asarray(values)
        n = len(self.reference)
        lo = np.searchsorted(self.reference, values, side="left")
        hi = np.searchsorted(self.reference, values, side="right")
        survival = (n-(lo+hi)/2+1)/(n+1)
        # Linear excess on log-tail scale keeps distinct novel observations
        # distinguishable instead of clipping all of them to percentile 1.
        return -np.log(survival) + np.maximum(values-self.reference[-1], 0)/self.scale


class LocalMemory:
    """Position-aware normal patch support, shared across all cycle positions."""
    def fit(self, sequences, size=96, seed=42):
        self.bank = balanced_sample(sequences, size, seed).astype(np.float32)
        return self

    def score(self, patches):
        patches = np.asarray(patches, dtype=np.float32)
        n, positions, _ = patches.shape
        distances = np.empty((n, positions), dtype=np.float32)
        for position in range(positions):
            bank = self.bank[:, position]
            bank_norm = np.sum(bank**2, axis=1)
            for start in range(0, n, 128):
                x = patches[start:start+128, position]
                d = np.maximum(np.sum(x**2, axis=1)[:, None]+bank_norm-2*x@bank.T, 0)
                distances[start:start+len(x), position] = d.min(1)
        # A few local defects must not disappear in global average pooling.
        top = max(1, int(np.ceil(positions*.1)))
        score = np.sort(distances, axis=1)[:, -top:].mean(1)
        return score, distances


class CycleVAD:
    def __init__(self, config):
        self.config = dict(config)

    def fit(self, fit, validation, segments):
        cfg = self.config
        seed = cfg.get("seed", 42)
        z = [descriptor(d) for d in fit]
        vz = [descriptor(d) for d in validation]
        self.tracker = CycleTracker(cfg.get("cycle_grid", 128), cfg.get("cycle_latent", 24), seed)
        self.tracker.fit(z, [d["indices"] for d in fit], segments)
        cycles = [self.tracker.predict(x, d["indices"]) for x, d in zip(z, fit)]
        vcycles = [self.tracker.predict(x, d["indices"]) for x, d in zip(vz, validation)]
        self.selection = []
        best = None
        # Only normal validation, never test labels or final threshold videos.
        for harmonics in cfg.get("harmonics_candidates", [2, 4, 8]):
            for ridge in cfg.get("ridge_candidates", [.01, .1, 1.]):
                design = [fourier(c["angle"], harmonics) for c in cycles]
                rows = balanced_sample([np.concatenate([b, x, np.maximum(c["confidence"], .1)[:, None]], 1).astype(np.float32)
                                        for b, x, c in zip(design, z, cycles)], cfg.get("fit_samples", 4096), seed)
                cols = 1+2*harmonics
                b, x, weight = rows[:, :cols], rows[:, cols:-1], rows[:, -1]
                penalty = np.eye(cols)*ridge*len(b)
                penalty[0, 0] = 1e-8
                coef = np.linalg.solve(b.T@(weight[:, None]*b)+penalty, b.T@(weight[:, None]*x)).astype(np.float32)
                loss = float(np.mean([np.mean((x-fourier(c["angle"], harmonics)@coef)**2)
                                      for x, c in zip(vz, vcycles)]))
                self.selection.append({"harmonics": harmonics, "ridge": ridge, "normal_validation_mse": loss})
                if best is None or loss < best[0]:
                    best = (loss, harmonics, ridge, coef)
        _, self.harmonics, self.ridge, self.coef = best
        residuals = [x-fourier(c["angle"], self.harmonics)@self.coef for x, c in zip(z, cycles)]
        limit, rank = cfg.get("fit_samples", 4096), cfg.get("subspace_rank", 64)
        self.conditional = ResidualSpace().fit(balanced_sample(residuals, limit, seed), rank, seed=seed)
        self.pooled = ResidualSpace().fit(balanced_sample(z, limit, seed), rank, seed=seed)
        self.local = LocalMemory().fit([d["patches"] for d in fit], cfg.get("local_memory", 96), seed)
        self.feature_shape = (fit[0]["global"].shape[1], *fit[0]["patches"].shape[1:])
        self.diagnostics = {"conditional_subspace": self.conditional.diagnostics,
                            "pooled_subspace": self.pooled.diagnostics,
                            "cycle": self.tracker.fit_diagnostics,
                            "mean_fit_cycle_confidence": float(np.mean([c["confidence"].mean() for c in cycles])),
                            "selected_harmonics": self.harmonics, "selected_ridge": self.ridge,
                            "selection": self.selection}
        return self

    def raw(self, data):
        if (data["global"].shape[1], *data["patches"].shape[1:]) != self.feature_shape:
            raise ValueError("Feature dimensions differ from the fitted model")
        z = descriptor(data)
        cycle = self.tracker.predict(z, data["indices"])
        residual = z-fourier(cycle["angle"], self.harmonics)@self.coef
        outside, inside, error = self.conditional.score(residual)
        pooled, pooled_inside, _ = self.pooled.score(z)
        local, patch_map = self.local.score(data["patches"])
        raw = {"conditional": outside, "conditional_inside": inside,
               "pooled": pooled, "pooled_inside": pooled_inside, "local": local,
               "alignment": cycle["alignment"], "innovation": cycle["innovation"],
               "progress": cycle["progress"]}
        return raw, cycle, patch_map

    def calibrate(self, reference, threshold):
        reference_raw = [self.raw(d)[0] for d in reference]
        self.calibrators = {k: TailCalibrator().fit(np.concatenate([r[k] for r in reference_raw]))
                            for k in reference_raw[0]}
        # Same zero-order hold as final frame-level evaluation, not sampled-only.
        from .data import hold
        streams = {name: [] for name in ("combined", "appearance", "cycle_conditioned")}
        for d in threshold:
            scores = self.score(d)
            for name in streams:
                streams[name].append(hold(d["indices"], scores[name], int(d["frame_count"])))
        self.thresholds = {name: float(np.quantile(np.concatenate(values),
                                                   self.config.get("threshold_quantile", .99), method="higher"))
                           for name, values in streams.items()}
        self.diagnostics["thresholds"] = self.thresholds
        return self

    def score(self, data):
        raw, cycle, patch_map = self.raw(data)
        calibrated = {k: self.calibrators[k].score(v) for k, v in raw.items()}
        appearance = np.maximum.reduce([calibrated[k] for k in ("pooled", "pooled_inside", "local")])
        conditioned = np.maximum(calibrated["conditional"], calibrated["conditional_inside"])
        conditioned *= cycle["confidence"]
        cycle_conditioned = np.maximum(appearance, conditioned)
        process = np.maximum.reduce([calibrated[k] for k in ("alignment", "innovation", "progress")])
        combined = np.maximum(cycle_conditioned, process*self.config.get("process_weight", 1.))
        return {"combined": combined, "appearance": appearance, "cycle_conditioned": cycle_conditioned,
                "process": process, "angle": cycle["angle"], "confidence": cycle["confidence"],
                "patch_distance": patch_map, **{f"raw_{k}": v for k, v in raw.items()}}
