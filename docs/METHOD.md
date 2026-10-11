# Implemented method

## Appearance representation

DINOv2-base stays frozen. Each full frame is letterboxed to 336×336. The descriptor concatenates a normalized global token and 36 regional vectors formed from the final and third-last hidden layers, a fixed 64-dimensional random projection, and normalization. Global and regional groups are scaled before concatenation.

The pooled baseline scores PCA reconstruction error and the variance-normalized magnitude of in-subspace coordinates. A position-specific nearest-neighbor local memory contributes the mean of the largest 10% of regional distances. The local bank is shared across the cycle.

## Continuous cycle coordinate

Normal training recordings are weak cycle candidates unless verified cycle boundaries are supplied. DTW aligns only normal FIT sequences to build a template. Inference uses causal Bayesian filtering on a 128-point circular numerical grid; angle is a circular posterior mean. This grid is not 128 independently trained phase models.

Confidence combines the concentration of the observation likelihood and absolute template agreement. It is not a calibrated probability of correct cycle localization.

## Conditional subspace

The normal descriptor is modeled as

$$z_t = \mu(\theta_t) + Ua_t + \epsilon_t.$$

A Fourier regression estimates the continuous mean. One shared PCA is fit on residuals after subtracting that mean. Normal validation chooses Fourier order and ridge. Reconstruction error outside the residual subspace and variance-normalized coordinates inside it are separately scored.

## Process evidence

- Alignment: minimum distance from the observation to the normal cycle template.
- Innovation: mismatch between the prior predicted location and current observation likelihood, before posterior update.
- Progress: maximum normalized difference between observed and expected circular movement at lags 1, 4 and 16. Expected movement uses the median normal cycle-candidate period. Long horizons spanning at least 0.45 cycles are excluded to avoid ambiguous circular differences.

## Calibration and fusion

Every component uses its own empirical log-tail calibration on disjoint normal reference videos, with extrapolation above the largest reference observation. Scores are not anomaly probabilities.

Pooled and local channels are max-fused into appearance. Confidence weights the max of conditional inside/outside scores. The Full score is the maximum of appearance, weighted conditional score, and the three process scores (process weight 1 in these experiments). Every ablation uses its own q99 threshold estimated on the same normal threshold videos.

The comparisons isolate score contributions, not necessarily computational savings: removing the local score retains the regional descriptor and removing a conditional score retains the tracker when process evidence still needs it. Reported shared evaluation time must not be interpreted as per-variant runtime.

The discrete-phase control fits four phase-specific means and PCA spaces under a total rank cap of 64. It retains the same tracker, confidence, appearance fallback and process branch, and replaces the Fourier/shared-residual conditional branch. Actual retained ranks and storage are published; equal rank caps do not imply identical effective capacity.
