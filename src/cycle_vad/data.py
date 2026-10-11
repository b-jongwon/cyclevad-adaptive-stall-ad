"""IPAD inventory, recording-group splits, and atomic, reusable feature caches."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def frame_paths(directory):
    files = [p for p in Path(directory).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    try:
        files.sort(key=lambda p: int(p.stem))
    except ValueError as exc:
        raise ValueError(f"Numeric frame filenames required: {directory}") from exc
    if not files or [int(p.stem) for p in files] != list(range(len(files))):
        raise ValueError(f"Expected contiguous frame IDs starting at 0: {directory}")
    return files


def inventory(root, scenes):
    root = Path(root)
    rows = []
    for scene in scenes:
        for part in ("training", "testing"):
            directory = root / scene / part / "frames"
            if not directory.is_dir():
                raise FileNotFoundError(f"Expected IPAD directory: {directory}")
            for seq in sorted(p for p in directory.iterdir() if p.is_dir()):
                files = frame_paths(seq)
                label = root / scene / "test_label" / f"{int(seq.name):03d}.npy" if part == "testing" else None
                rows.append({"id": f"{scene}/{part}/{seq.name}", "scene": scene,
                             "partition": part, "sequence": seq.name, "directory": str(seq.resolve()),
                             "frames": len(files), "labels": str(label.resolve()) if label else None})
    return rows


def split_normal(rows, seed=42, groups=None):
    """All four splits are recording-disjoint if the supplied grouping is correct.

    Validation selects regularization. Reference calibrates component scales.
    Threshold is an independent normal holdout for the final alarm threshold.
    """
    groups = groups or {}
    by_group = {}
    for row in rows:
        if row["partition"] != "training":
            raise ValueError("Only training videos may be split for model fitting")
        group = str(groups.get(row["id"], row["id"]))
        by_group.setdefault(group, []).append(row["id"])
    keys = sorted(by_group, key=lambda g: fingerprint([seed, g]))
    if len(keys) < 8:
        raise ValueError("At least 8 independent normal videos/groups per scene are required")
    n = max(1, int(round(len(keys) * .15)))
    chunks = {"validation": keys[:n], "reference": keys[n:2*n],
              "threshold": keys[2*n:3*n], "fit": keys[3*n:]}
    return {name: sorted(s for g in chunk for s in by_group[g]) for name, chunk in chunks.items()}


def labels_for(row):
    if row["partition"] == "training":
        return np.zeros(row["frames"], dtype=np.int8)
    values = np.load(row["labels"], allow_pickle=False)
    if values.ndim != 1 or not np.isin(values, [0, 1]).all():
        raise ValueError(f"Invalid binary annotation: {row['labels']}")
    if len(values) != row["frames"]:
        return np.full(row["frames"], -1, dtype=np.int8)
    return values.astype(np.int8)


def hold(indices, values, count):
    indices, values = np.asarray(indices), np.asarray(values)
    if len(indices) == 0 or indices[0] != 0 or len(indices) != len(values):
        raise ValueError("Sampled outputs must begin at frame zero and have matching lengths")
    if np.any(np.diff(indices) <= 0) or indices[-1] >= count:
        raise ValueError("Invalid sampled frame indices")
    return values[np.searchsorted(indices, np.arange(count), side="right") - 1]


def validate_cache(data):
    z, patches, ids = data["global"], data["patches"], data["indices"]
    if z.ndim != 2 or patches.ndim != 3 or len(z) != len(patches) or len(z) != len(ids):
        raise ValueError("Malformed feature cache")
    if len(ids) < 2 or ids[0] != 0 or np.any(np.diff(ids) <= 0):
        raise ValueError("Need at least two ordered observations starting at zero")
    if not np.isfinite(z).all() or not np.isfinite(patches).all():
        raise ValueError("Nonfinite feature cache")


def load_cache(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    validate_cache(data)
    return data


def balanced_sample(arrays, limit, seed=42):
    """Bound RAM and prevent long recordings from dominating FIT statistics."""
    rng = np.random.default_rng(seed)
    each = max(1, limit // len(arrays))
    return np.concatenate([a[np.sort(rng.choice(len(a), min(len(a), each), replace=False))] for a in arrays])


def cycle_segments(data, row, boundaries=None):
    """Boundary values are source-frame starts plus the exclusive final end.

    No annotation means weak full-recording alignment, NOT verified cycle GT.
    Test video length/boundaries are never passed to the online tracker.
    """
    bounds = (boundaries or {}).get(row["id"])
    if bounds is None:
        return [np.arange(len(data["indices"]))], "weak_recording_alignment"
    b = np.asarray(bounds)
    if b.ndim != 1 or len(b) < 2 or not np.isfinite(b).all() or np.any(b != b.astype(int)):
        raise ValueError(f"Invalid cycle boundaries: {row['id']}")
    if b[0] < 0 or b[-1] > row["frames"] or np.any(np.diff(b) <= 0):
        raise ValueError(f"Out-of-range/unordered cycle boundaries: {row['id']}")
    segments = [np.flatnonzero((data["indices"] >= lo) & (data["indices"] < hi)) for lo, hi in zip(b[:-1], b[1:])]
    if any(len(s) < 8 for s in segments):
        raise ValueError(f"Each annotated cycle needs >=8 sampled frames: {row['id']}")
    return segments, "provided_cycle_boundaries"
