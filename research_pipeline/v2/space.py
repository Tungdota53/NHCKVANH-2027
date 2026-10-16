"""Task-aware finite candidates; content identities ignore relocation paths."""

import json
import math
import itertools
from pathlib import Path

from research_pipeline.study import model_snapshot
from .file_integrity import content_hash
from yolo_tuner.research import digest
from .schema import TASKS


def canonical(config):
    c = dict(config)
    for key in ("model", "encoder_weights", "feature_cache"):
        if key in c and Path(str(c[key])).is_file():
            c[key] = content_hash(c[key])
    return c


def validate_config(task, c, hw):
    if task not in TASKS:
        raise ValueError("Unsupported task; no synthetic labels or silent adapter fallback")
    for key in ("imgsz", "batch", "accumulation"):
        if type(c.get(key)) is not int or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if c["imgsz"] < (64 if task == "detection" else 16) or task == "detection" and c["imgsz"] % 32:
        raise ValueError("Invalid image size for task")
    if c["precision"] not in {"fp32", "fp16", "bf16"} or c["precision"] != "fp32" and not hw.get(c["precision"], False):
        raise ValueError("Precision unsupported on measured backend")
    if hw["device"] in {"cpu", "mps"} and c["precision"] != "fp32":
        raise ValueError("CPU/MPS requires FP32 in this runner")
    if c["optimizer"] not in ({"AdamW", "SGD"} if task == "detection" else {"AdamW"}):
        raise ValueError("Optimizer unsupported by task; no optimizer auto")
    for key in ("lr0", "weight_decay"):
        if isinstance(c[key], bool) or not isinstance(c[key], (float, int)) or not math.isfinite(c[key]) or c[key] < 0 or key == "lr0" and c[key] == 0:
            raise ValueError(f"Invalid {key}")
    if task != "detection":
        if task == "classification" and c["model"] != "cnn_classifier":
            raise ValueError("Classification adapter currently supports cnn_classifier only")
        for key in (("width", "hidden") if task == "captioning" else ("width",)):
            if type(c[key]) is not int or c[key] <= 0 or c[key] % 4:
                raise ValueError(f"{key} must be a positive multiple of 4")
        if task == "classification" and c["weight_decay"] != 0.01:
            raise ValueError("Classification adapter currently fixes weight_decay=0.01")
        if task == "captioning":
            if c["model"] != "attn_gru_captioner" or c["encoder"] != "tiny":
                raise ValueError("V2 caption adapter currently supports self-contained tiny spatial-attention encoder; pretrained remains legacy")
            for key in ("dropout", "label_smoothing"):
                if not 0 <= c[key] < 1:
                    raise ValueError(f"{key} must be in [0,1)")
            if type(c["feature_grid"]) is not int or not 1 <= c["feature_grid"] <= 16:
                raise ValueError("feature_grid must be an integer in 1..16")
    elif type(c.get("freeze", 0)) is not int or c.get("freeze", 0) < 0:
        raise ValueError("freeze must be a nonnegative number of backbone blocks")
    elif c.get("freeze", 0) and not str(c["model"]).endswith(".pt"):
        raise ValueError("Freeze requires a local learned .pt checkpoint, not an untrained YAML architecture")


def build_space(task, hw, folder, grid=None, preset="compact", models=None, precision=None):
    precision = precision or ("fp16" if hw.get("fp16") else "fp32")
    base = {"model": "yolo11n.yaml" if task == "detection" else "cnn_classifier" if task == "classification" else "attn_gru_captioner",
            "imgsz": 320 if task == "detection" else 32 if task == "classification" else 160,
            "batch": 4 if task == "detection" else 16, "accumulation": 4 if task == "detection" else 2,
            "precision": precision, "optimizer": "AdamW", "lr0": 0.001,
            "weight_decay": 0.0005 if task == "detection" else 0.01}
    if task != "detection":
        base.update(width=16)
    if task == "captioning":
        base.update(hidden=256, encoder="tiny", pretrained=False, feature_grid=4, dropout=0.1, label_smoothing=0.05)
    if models:
        if task != "detection":
            raise ValueError("--models only supported for detection")
        base["model"] = models[0]
    if task == "detection":
        base["freeze"] = 0
    if grid:
        overrides = json.loads(Path(grid).read_text(encoding="utf-8"))
    elif preset == "research":
        choices = models or (["yolo11n.yaml", "yolo11s.yaml"] if task == "detection" else [base["model"]])
        sizes = (320, 640) if task == "detection" else (32, 64) if task == "classification" else (160, 224)
        batches = (4, 8) if task == "detection" else (8, 16)
        overrides = []
        for model, size, batch, lr in itertools.product(choices, sizes, batches, (0.001, 0.0003)):
            freezes = (0, 5) if task == "detection" and str(model).endswith(".pt") else (0,)
            for freeze in freezes:
                c = {"model": model, "imgsz": size, "batch": batch, "lr0": lr,
                     "accumulation": base["batch"] * base["accumulation"] // batch}
                if task == "detection":
                    c["freeze"] = freeze
                overrides.append(c)
    elif preset == "core":
        # 12 configs: baseline + controlled contrasts, not all cross-products.
        overrides = [{}]
        overrides += [{"lr0": lr} for lr in (0.0003, 0.003)]
        overrides += [{"batch": b} for b in ((1, 2, 8) if task == "detection" else (2, 8, 32))]
        sizes = (224, 480, 640) if task == "detection" else (48, 64, 96) if task == "classification" else (96, 224, 256)
        overrides += [{"imgsz": s} for s in sizes]
        if task == "detection":
            overrides += [{"model": (models or ["yolo11n.yaml", "yolo11s.yaml"])[-1]}, {"optimizer": "SGD"}, {"weight_decay": 0.001}]
        else:
            overrides += [{"width": 32}, {"width": 64}, {"hidden": 512} if task == "captioning" else {"lr0": 0.0001}]
        # Effective-batch contrasts use the baseline's nominal effective batch.
        for c in overrides:
            if "batch" in c:
                c["accumulation"] = max(1, base["batch"] * base["accumulation"] // c["batch"])
    else:
        overrides = [{}, {"lr0": 0.0003}, {"lr0": 0.003}]
    if not isinstance(overrides, list) or not overrides or len(overrides) > 512 or any(not isinstance(c, dict) or set(c) - set(base) for c in overrides):
        raise ValueError("Grid must contain 1..512 supported task-specific overrides")
    configs, identities, snapshots = [], set(), {}
    for override in overrides:
        c = {**base, **override}
        if precision is not None:
            c["precision"] = override.get("precision", precision)
        validate_config(task, c, hw)
        if task == "detection":
            if c["model"] not in snapshots:
                snapshots[c["model"]] = model_snapshot(c["model"], folder / "models")
            c["model"] = snapshots[c["model"]]
        identity = digest(canonical(c))
        if identity not in identities:
            configs.append(c)
            identities.add(identity)
    return configs
