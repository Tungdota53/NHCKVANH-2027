"""Offline ridge predictors with dataset-group split-conformal residuals.

No pretrained coefficients are shipped: fitting requires measured full runs.
Predictors reject unknown categorical values and out-of-range numeric inputs.
"""

import csv
import json
import math
from pathlib import Path
import random

CATEGORICAL = ("model", "precision", "optimizer", "gpu", "runtime_protocol")
NUMERIC = ("imgsz", "batch", "freeze", "nbs", "lr0", "train_images", "val_images",
           "classes", "objects_per_image", "small_object_ratio", "full_epochs")
TARGETS = ("final_map50_95", "peak_reserved_mb", "full_training_seconds")
SCHEMA_VERSION = 2


def quantile_upper(scores, alpha):
    if not 0 < alpha < 1 or not scores or not all(math.isfinite(s) and s >= 0 for s in scores):
        raise ValueError("Conformal scores must be nonnegative and finite; alpha in (0,1)")
    k = math.ceil((len(scores) + 1) * (1 - alpha))
    if k > len(scores):
        raise ValueError("Too few independent calibration groups for requested coverage")
    return sorted(scores)[k - 1]


def vector(row, spec):
    numeric = [(float(row[k]) - spec["means"][k]) / spec["scales"][k] for k in NUMERIC]
    onehot = [float(str(row[k]) == value) for k in CATEGORICAL for value in spec["categories"][k]]
    return [1.0] + numeric + onehot


def validate_row(row):
    for key in ("dataset_id",) + CATEGORICAL:
        if not str(row.get(key, "")).strip():
            raise ValueError(f"Missing offline CSV field: {key}")
    for key in NUMERIC + TARGETS:
        if key not in row or not math.isfinite(float(row[key])):
            raise ValueError(f"Missing or non-finite offline CSV field: {key}")
    if not 0 <= float(row["final_map50_95"]) <= 1:
        raise ValueError("final_map50_95 must be in [0,1]")
    for key in ("imgsz", "batch", "nbs", "lr0", "train_images", "val_images", "classes", "full_epochs", "full_training_seconds"):
        if float(row[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if float(row["peak_reserved_mb"]) < 0:
        raise ValueError("peak_reserved_mb must be nonnegative (CPU has no CUDA VRAM)")


def fit(csv_path, output, alpha=0.1, seed=17):
    import numpy as np

    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        validate_row(r)
    groups = sorted({r["dataset_id"] for r in rows})
    random.Random(seed).shuffle(groups)
    ncal = max(math.ceil(1 / alpha) - 1, math.ceil(len(groups) * 0.3))
    ntest = max(1, math.ceil(len(groups) * 0.1))
    if len(groups) - ncal - ntest < 3:
        raise ValueError(f"Need more distinct dataset IDs: {ncal} calibration + {ntest} test + >=3 training")
    calibration_ids, test_ids = set(groups[:ncal]), set(groups[ncal:ncal+ntest])
    training_ids = set(groups) - calibration_ids - test_ids
    splits = {"train": [r for r in rows if r["dataset_id"] in training_ids],
              "calibration": [r for r in rows if r["dataset_id"] in calibration_ids],
              "test": [r for r in rows if r["dataset_id"] in test_ids]}
    train = splits["train"]
    spec = {"categories": {k: sorted({str(r[k]) for r in train}) for k in CATEGORICAL},
            "means": {k: float(np.mean([float(r[k]) for r in train])) for k in NUMERIC},
            "scales": {k: max(1e-9, float(np.std([float(r[k]) for r in train]))) for k in NUMERIC},
            "ranges": {k: [min(float(r[k]) for r in train), max(float(r[k]) for r in train)] for k in NUMERIC}}
    for name in ("calibration", "test"):
        if any(str(r[k]) not in spec["categories"][k] for r in splits[name] for k in CATEGORICAL):
            raise ValueError(f"{name} contains unseen categorical values. Collect balanced configuration coverage across datasets.")
    x = np.asarray([vector(r, spec) for r in train])
    penalty = np.eye(x.shape[1])
    penalty[0, 0] = 0
    artifact = {"schema_version": SCHEMA_VERSION, "model": "ridge_group_split_conformal", "spec": spec,
                "alpha": alpha, "seed": seed, "predictors": {},
                "split_groups": {"train": sorted(training_ids), "calibration": sorted(calibration_ids), "test": sorted(test_ids)},
                "row_counts": {k: len(v) for k, v in splits.items()},
                "assumption": "Exchangeable dataset groups and matching full-training protocol; OOD coverage is not guaranteed."}
    for target in TARGETS:
        y = np.asarray([float(r[target]) for r in train])
        coefficients = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        residuals = {}
        for r in splits["calibration"]:
            residual = abs(float(r[target]) - float(np.dot(vector(r, spec), coefficients)))
            group = r["dataset_id"]
            residuals[group] = max(residuals.get(group, 0), residual)
        q = quantile_upper(list(residuals.values()), alpha)
        test_errors = [abs(float(r[target]) - float(np.dot(vector(r, spec), coefficients))) for r in splits["test"]]
        group_covered = {}
        for r, error in zip(splits["test"], test_errors):
            group = r["dataset_id"]
            group_covered[group] = group_covered.get(group, True) and error <= q
        artifact["predictors"][target] = {
            "coefficients": coefficients.tolist(), "calibration_radius": q,
            "test_mae": sum(test_errors) / len(test_errors),
            "test_row_coverage": sum(e <= q for e in test_errors) / len(test_errors),
            "test_group_coverage": sum(group_covered.values()) / len(group_covered),
        }
    Path(output).write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    return artifact


def predict(artifact, row):
    if artifact.get("schema_version") != SCHEMA_VERSION:
        return {"status": "out_of_domain", "reasons": ["Unsupported predictor schema; refit with current code"]}
    spec = artifact["spec"]
    reasons = []
    for k in CATEGORICAL:
        if str(row[k]) not in spec["categories"][k]:
            reasons.append(f"unknown {k}={row[k]}")
    for k in NUMERIC:
        value = float(row[k])
        lo, hi = spec["ranges"][k]
        if not math.isfinite(value) or not lo <= value <= hi:
            reasons.append(f"{k} outside offline training range [{lo}, {hi}]")
    if reasons:
        return {"status": "out_of_domain", "reasons": reasons}
    x = vector(row, spec)
    predictions = {}
    for target, params in artifact["predictors"].items():
        estimate = sum(a * b for a, b in zip(x, params["coefficients"]))
        radius = params["calibration_radius"]
        upper_limit = 1.0 if target == "final_map50_95" else float("inf")
        predictions[target] = {"estimate": min(upper_limit, max(0, estimate)),
                               "lower": min(upper_limit, max(0, estimate - radius)),
                               "upper": min(upper_limit, max(0, estimate + radius))}
    return {"status": "in_domain", "nominal_coverage": 1 - artifact["alpha"],
            "calibration_unit": "max residual per dataset group", "predictions": predictions}


def input_row(config, stats, hw, full_epochs):
    protocol = "|".join(["tuner-v1-adamw-warmup0.5-mosaic0",
                          hw.get("torch_version", "unknown"), hw.get("ultralytics_version", "unknown"),
                          str(hw.get("driver_version") or "none")])
    return {**config, "gpu": hw["gpu"] or "cpu", "runtime_protocol": protocol, "full_epochs": full_epochs,
            "train_images": stats["train_images"], "val_images": stats["val_images"],
            "classes": stats["classes"], "objects_per_image": stats["objects"] / stats["train_images"],
            "small_object_ratio": stats["small_object_ratio"]}
