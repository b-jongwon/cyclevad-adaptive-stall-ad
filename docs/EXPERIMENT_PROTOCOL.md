# R01–R04 experiment protocol (2026-10-09 KST)

This protocol is fixed before new test evaluation. No test-driven model selection.

- Frozen DINOv2-base, 336px letterbox, layers -1/-3, 6x6 patches, fixed random projection seed 42.
- Primary stride 2, extraction batch 8 (same as supplied notebook), FP16 GPU extraction, CPU statistical fitting.
- Four disjoint normal-video splits, split seed 42. Recording group identity unavailable; group independence is not established.
- Fourier harmonics/ridge selected by normal validation MSE. Component calibration uses normal reference videos. Every variant gets its own q99 threshold on the same separate normal threshold videos.
- Test frame labels are loaded only after inference. R02 videos 12/13/14 have mismatched image/label lengths: preserve source and exclude those whole videos from metrics under strict-v1. No guessed trimming or padding. Predictions remain available. R01/R03/R04 have no length mismatch.
- No verified cycle boundaries. Every normal recording is a weak cycle candidate; cycle-location accuracy is unmeasured.
- Frame scores use causal zero-order hold. Event metrics are evaluated per video; delay is conditional on detection and must be read with missed-event count.
- Scene metrics: AUROC, AP, FPR, recall, event coverage, detected-event delay. Macro = unweighted mean over four scenes; partial runs never labeled four-scene macro.
- Seed robustness: model seeds 42, 43, 44 on fixed normal splits and fixed encoder projection. Report sample SD, not confidence interval.
- Stride sensitivity: stride 1, seed 42, same normal split, retrain/recalibrate all affected models, compare pooled subspace and Full across all scenes.

## Registered variants

1. pooled: pooled PCA outside + inside Mahalanobis, without local memory.
2. appearance (A0): pooled + local memory.
3. cycle_conditioned (A1): appearance + confidence-weighted conditional PCA.
4. appearance_process (A2): appearance + all process scores.
5. full (A3): A1 + all process scores.
6. no_alignment: Full excluding template minimum distance.
7. no_innovation: Full excluding predictive surprise.
8. no_progress: Full excluding progress deviation.
9. single_lag: Full with progress lag 1 only (new reference calibration).
10. no_confidence: Full with conditional confidence fixed to 1.
11. no_local: Full without local-memory scoring (regional descriptor remains).
12. no_pooled: Full without pooled PCA scoring (local fallback remains).
13. outside_only: Full without both PCA inside-Mahalanobis scores.
14. discrete_phase: four hard cycle bins, separate residual PCA per bin (rank cap 16 each), same causal tracker and appearance/process branches; actual rank and storage reported. This is a controlled implementation, not a reproduction of an external published baseline.

Removal experiments hold fitted shared models fixed unless the removed mechanism requires retraining. Score-only ablations reuse raw component predictions. Single-lag recalibrates its changed progress component. Discrete-phase fits only normal FIT descriptors and evaluates conditional distance with hard phase selection; confidence weighting remains.

## Stage gates and publication

E0: data audit, environment, tests and source recovery. E1: original R02 outputs compared to local rerun (different environment, no assumed bitwise equality). E2: all scenes, pooled/A0/A1/Full. E3: all four core factorial combinations. E4: all registered removal/phase variants. E5: three model seeds for core comparisons, stride sensitivity and qualitative score timelines.

Each completed stage publishes scripts/configs, numeric JSON/CSV results and generated README figures to GitHub. Raw frames, weights, feature caches and large prediction maps stay local. Claims of improvement require measured results; negative results stay visible. Test-example timelines use the first lexically ordered valid video containing an anomaly in each scene, not the best-looking example.

## Stride interpretation

The registered stride-1 run preserves observation-count lags (tracker descriptor lag 2; progress lags 1/4/16). Its effective source-frame horizons are therefore shorter than stride 2. This is an end-to-end configuration sensitivity test, not an isolated test of sampling density with matched temporal horizons. No settings were tuned after seeing the test result.
