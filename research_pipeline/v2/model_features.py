"""Architecture identities/descriptors, read locally without training/downloads."""

import math
import os
from pathlib import Path
import re

from yolo_tuner.research import digest
from .space import canonical


def architecture_key(config):
    c = canonical(config)
    return digest({k: c[k] for k in ("model", "width", "hidden", "encoder", "feature_grid") if k in c})


def model_identity(config):
    return digest({"model": canonical(config)["model"]})


def family_scale(config):
    match = re.search(r"(yolo11|yolo26|yolov8)([nslmx])", Path(config["model"]).name.lower())
    if match:
        return {"yolov8": 1, "yolo11": 2, "yolo26": 3}[match[1]], "nslmx".index(match[2]) + 1
    return (4 if config["model"] == "cnn_classifier" else 5 if config["model"] == "attn_gru_captioner" else 0), 0


def describe_models(task, configs, classes):
    import torch
    prior = torch.get_num_threads()
    torch.set_num_threads(min(4, prior))
    descriptors = {}
    try:
        for config in configs:
            key = architecture_key(config)
            if key in descriptors:
                continue
            descriptor = {"model_identity": model_identity(config), "family_scale": family_scale(config),
                          "parameters": None, "layers": None, "gflops_reference_32": None,
                          "parameter_scope": "local initialization before detection class-head adaptation",
                          "pretrained_checkpoint": Path(config["model"]).suffix.lower() == ".pt"}
            if task == "detection":
                os.environ.setdefault("YOLO_AUTOINSTALL", "false")
                model_path = Path(str(config["model"]))
                if model_path.suffix.lower() == ".pt" and not model_path.is_file():
                    raise ValueError(f"Local detection checkpoint not found (network downloads disabled): {config['model']}")
                from ultralytics import YOLO
                model = YOLO(config["model"]).model.cpu()
                block_params = [int(sum(p.numel() for p in block.parameters())) for block in model.model]
                total_p = int(sum(p.numel() for p in model.parameters()))
                trainable_p = int(sum(p.numel() for n, p in model.named_parameters() if ".dfl" not in n and p.dtype.is_floating_point))
                if trainable_p <= 0 and total_p > 0:
                    trainable_p = total_p
                descriptor.update(
                    parameters=total_p,
                    trainable_parameters=trainable_p,
                    layers=len(model.model),
                    block_parameters=block_params,
                )
                from ultralytics.utils.torch_utils import get_flops
                value = get_flops(model, imgsz=32)
                descriptor["gflops_reference_32"] = value if value and math.isfinite(value) else None
            elif task == "classification":
                from yolo_tuner.vision_models import build
                model = build(task, config, classes=classes)
                descriptor.update(
                    parameters=int(sum(p.numel() for p in model.parameters())),
                    trainable_parameters=int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                    layers=sum(1 for _ in model.modules()),
                )
            descriptors[key] = descriptor
    finally:
        torch.set_num_threads(prior)
    return descriptors


def descriptor_for(protocol, config):
    return protocol.get("model_descriptors", {}).get(architecture_key(config), {})


def trainable_parameters_for(protocol, config, observed_runs=None):
    """Return (total_parameters, trainable_parameters, param_basis) deterministically."""
    descriptor = descriptor_for(protocol, config)
    target_arch = architecture_key(config)
    target_freeze = int(config.get("freeze", 0) or 0)

    if observed_runs:
        for run in observed_runs:
            if not isinstance(run, dict):
                continue
            run_cfg = run.get("config")
            if isinstance(run_cfg, dict):
                if architecture_key(run_cfg) != target_arch or int(run_cfg.get("freeze", 0) or 0) != target_freeze:
                    continue
            obs = run.get("observed_forward")
            if isinstance(obs, dict):
                if "freeze" in obs and int(obs.get("freeze", 0) or 0) != target_freeze:
                    continue
                train_p = obs.get("trainable_parameters")
                if isinstance(train_p, (int, float)) and not isinstance(train_p, bool) and math.isfinite(train_p) and train_p > 0:
                    frozen_p = obs.get("frozen_parameters") or 0
                    total_p = descriptor.get("parameters") or (int(train_p) + int(frozen_p))
                    return int(total_p), int(train_p), "observed_forward"

    total_p = descriptor.get("parameters")
    if not (isinstance(total_p, (int, float)) and not isinstance(total_p, bool) and math.isfinite(total_p) and total_p > 0):
        model_name = str(config.get("model", ""))
        task = protocol.get("task", "detection")
        if model_name == "cnn_classifier" or task == "classification":
            w = int(config.get("width", 16) or 16)
            total_p = max(1000, 74 * w * w)
        elif model_name == "attn_gru_captioner" or task == "captioning":
            w = int(config.get("width", 16) or 16)
            h = int(config.get("hidden", 256) or 256)
            total_p = max(10000, 74 * w * w + 12 * h * h + 2000 * h)
        else:
            _, scale = family_scale(config)
            total_p = int(2_600_000 * (3.6 ** max(0, (scale or 1) - 1)))
    total_p = int(total_p)

    if target_freeze <= 0:
        unfrozen = descriptor.get("trainable_parameters")
        train_p = int(unfrozen) if isinstance(unfrozen, (int, float)) and unfrozen > 0 else total_p
        return total_p, train_p, "descriptor_unfrozen"

    blocks = descriptor.get("block_parameters")
    if isinstance(blocks, list) and blocks:
        frozen_p = sum(int(x) for x in blocks[:min(target_freeze, len(blocks))])
        return total_p, max(1, total_p - frozen_p), "descriptor_block_parameters"

    layers = max(1, int(descriptor.get("layers") or 24))
    frac = min(0.95, max(0.0, target_freeze / layers))
    frozen_ratio = frac ** 2
    return total_p, max(1, int(round(total_p * (1.0 - frozen_ratio)))), "freeze_layer_ratio_fallback"
