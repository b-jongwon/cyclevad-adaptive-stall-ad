"""Strict frame and event metrics; thresholds always come from normal holdout."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def frame_metrics(labels, scores, threshold):
    y, score = np.asarray(labels), np.asarray(scores)
    if y.shape != score.shape or not np.isfinite(score).all():
        raise ValueError("Finite aligned scores required")
    mask = y >= 0
    y, score = y[mask], score[mask]
    alarm = score > threshold
    return {"frames": int(len(labels)), "valid_frames": int(mask.sum()),
            "unknown_frames": int((~mask).sum()), "anomaly_frames": int((y == 1).sum()),
            "auroc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
            "ap": float(average_precision_score(y, score)) if np.any(y == 1) else None,
            "fpr": float(alarm[y == 0].mean()) if np.any(y == 0) else None,
            "recall": float(alarm[y == 1].mean()) if np.any(y == 1) else None,
            "threshold": float(threshold)}


def event_metrics(labels, scores, threshold):
    # Called per video: never join events across sequence boundaries.
    positive = np.asarray(labels) == 1
    starts = np.flatnonzero(np.diff(np.r_[False, positive, False].astype(int)) == 1)
    ends = np.flatnonzero(np.diff(np.r_[False, positive, False].astype(int)) == -1)
    delays = []
    for lo, hi in zip(starts, ends):
        hits = np.flatnonzero(np.asarray(scores)[lo:hi] > threshold)
        if len(hits):
            delays.append(int(hits[0]))
    return {"events": len(starts), "detected_events": len(delays),
            "event_coverage": len(delays)/len(starts) if len(starts) else None,
            "detected_event_delays_source_frames": delays}
