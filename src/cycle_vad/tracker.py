"""Normal-template fitting and causal circular Bayesian cycle localization.

The numerical quadrature grid is NOT a bank of separately fitted phase models.
Only one continuous angle and its uncertainty condition the appearance model.
"""
from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sklearn.decomposition import PCA

from .data import balanced_sample


def circular_difference(a, b):
    return (np.asarray(a) - np.asarray(b) + .5) % 1.0 - .5


def fourier(angle, harmonics):
    angle = np.asarray(angle).reshape(-1)
    theta = 2 * np.pi * angle[:, None] * np.arange(1, harmonics + 1)[None]
    return np.concatenate([np.ones((len(angle), 1)), np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)


def resample(sequence, count):
    # An exclusive cycle end: the last observed sample is not identified with 0.
    old = np.arange(len(sequence)) / len(sequence)
    new = np.arange(count) / count
    return np.stack([np.interp(new, old, col) for col in sequence.T], axis=1)


def dtw_map(sequence, template, band=.3):
    """Monotone alignment only during normal-template FIT; never on test futures."""
    n, m = len(sequence), len(template)
    cost = np.full((n+1, m+1), np.inf)
    parent = np.zeros((n, m), dtype=np.int8)
    cost[0, 0] = 0
    dist = np.mean((sequence[:, None] - template[None]) ** 2, axis=-1)
    for i in range(n):
        center = i * (m-1) / max(n-1, 1)
        lo, hi = max(0, int(center-band*m)), min(m, int(center+band*m)+2)
        for j in range(lo, hi):
            previous = [cost[i, j], cost[i, j+1], cost[i+1, j]]
            k = int(np.argmin(previous))
            parent[i, j] = k
            cost[i+1, j+1] = dist[i, j] + previous[k]
    if not np.isfinite(cost[n, m]):
        raise ValueError("Cycle alignment failed; check normal cycle boundaries")
    assignments = [[] for _ in range(n)]
    i, j = n-1, m-1
    while i >= 0 and j >= 0:
        assignments[i].append(j)
        k = parent[i, j]
        if k == 0:
            i, j = i-1, j-1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    return np.array([np.mean(a) if a else 0 for a in assignments])


class CycleTracker:
    def __init__(self, grid_size=128, latent_dim=24, seed=42):
        self.grid_size, self.latent_dim, self.seed = grid_size, latent_dim, seed

    def encode(self, z):
        x = self.reducer.transform(z) / self.scale
        # Entirely backward differences; append/change future frames without
        # changing these descriptors at any earlier time.
        lag = np.maximum(np.arange(len(x))-2, 0)
        return np.concatenate([x, .5*(x-x[lag])], axis=1)

    def fit(self, sequences, indices, segments):
        samples = balanced_sample(sequences, 4096, self.seed)
        dim = min(self.latent_dim, samples.shape[1], len(samples)-1)
        self.reducer = PCA(dim, svd_solver="randomized", random_state=self.seed).fit(samples)
        # Avoid whitening near-zero PCA axes, which amplifies image noise.
        self.scale = np.sqrt(np.maximum(self.reducer.explained_variance_,
                                        self.reducer.explained_variance_[0]*.05 + 1e-8))
        encoded = [self.encode(z) for z in sequences]
        cycles, periods = [], []
        for x, ids, parts in zip(encoded, indices, segments):
            for part in parts:
                if len(part) < 8:
                    raise ValueError("Need >=8 observations per normal cycle candidate")
                cycles.append(x[part])
                periods.append(float(ids[part[-1]] - ids[part[0]] + np.median(np.diff(ids))))
        self.period = float(np.median(periods))
        curves = np.stack([resample(c, self.grid_size) for c in cycles])
        median = np.median(curves, axis=0)
        template = curves[np.argmin(np.mean((curves-median)**2, axis=(1, 2)))].copy()
        # Two robust DTW barycenter updates. A frame cap bounds quadratic work.
        for _ in range(2):
            sums = np.zeros_like(template)
            counts = np.zeros(self.grid_size)
            for c in cycles:
                c = resample(c, min(len(c), 256))
                mapping = np.rint(dtw_map(c, template)).astype(int)
                np.add.at(sums, mapping, c)
                np.add.at(counts, mapping, 1)
            valid = counts > 0
            template[valid] = sums[valid] / counts[valid, None]
        self.template = template
        self.angles = np.arange(self.grid_size)/self.grid_size
        self.unit = np.exp(2j*np.pi*self.angles)
        distances = []
        for c in cycles:
            c = c[np.linspace(0, len(c)-1, min(128, len(c)), dtype=int)]
            distances.extend(np.mean((c[:, None]-template[None])**2, axis=-1).min(1))
        self.temperature = max(float(np.quantile(distances, .7)), .03)
        self.fit_diagnostics = {"normal_cycle_candidates": len(cycles),
                                "median_period_source_frames": self.period,
                                "template_temperature": self.temperature,
                                "period_q10_q90": np.quantile(periods, [.1, .9]).tolist()}
        return self

    def predict(self, z, indices, progress_lags=(1, 4, 16)):
        x = self.encode(z)
        ids = np.asarray(indices)
        if len(ids) != len(x) or np.any(np.diff(ids) <= 0):
            raise ValueError("Cycle tracker needs increasing source-frame indices")
        n, g = len(x), self.grid_size
        out = {key: np.zeros(n) for key in ("angle", "confidence", "alignment", "innovation", "progress",
                                           "location_confidence", "template_agreement", "observation_angle")}
        posterior = np.full(g, 1/g)
        offsets = np.arange(-g//4, g//4+1)
        for t, observation in enumerate(x):
            dt = float(ids[t]-ids[t-1]) if t else 0
            expected = dt/self.period*g
            if t:
                # Small backward/stationary moves remain possible. A small
                # restart mass permits recovery after missing/corrupted frames.
                weights = softmax(-.5*((offsets-expected)/max(1., abs(expected)*.5))**2)
                prior = sum(w*np.roll(posterior, int(k)) for k, w in zip(offsets, weights))
                prior = .999*prior + .001/g
            else:
                prior = posterior
            distance = np.mean((self.template-observation)**2, axis=1)
            likelihood = softmax(-distance/self.temperature)
            observed = np.dot(likelihood, self.unit)
            posterior = prior*likelihood
            posterior /= max(posterior.sum(), 1e-300)
            unit = np.dot(posterior, self.unit)
            angle = (np.angle(unit)/(2*np.pi)) % 1
            out["angle"][t] = angle
            out["confidence"][t] = abs(observed)*np.exp(-distance.min()/(4*self.temperature))
            out["location_confidence"][t] = abs(observed)
            out["template_agreement"][t] = np.exp(-distance.min()/(4*self.temperature))
            out["observation_angle"][t] = (np.angle(observed)/(2*np.pi)) % 1
            out["alignment"][t] = distance.min()
            if t:
                # Surprise before updating posterior: temporal smoothing cannot
                # silently explain away a skipped/reversed/late observation.
                out["innovation"][t] = -np.log(max(g*np.dot(prior, likelihood), 1e-12))
            # Multi-horizon progress catches a held frame even if its appearance
            # is an excellent match to one normal location. Wrap is circular.
            deviations = []
            for lag in progress_lags:
                if t >= lag:
                    elapsed = (ids[t]-ids[t-lag])/self.period
                    if elapsed < .45:
                        observed_delta = circular_difference(angle, out["angle"][t-lag])
                        deviations.append(abs(observed_delta-elapsed)/max(elapsed, 1/g))
            out["progress"][t] = max(deviations, default=0.)
        return out
