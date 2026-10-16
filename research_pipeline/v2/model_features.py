"""Architecture identities/descriptors, read locally without training/downloads."""

import math
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
                from ultralytics import YOLO
                model = YOLO(config["model"]).model.cpu()
                descriptor.update(parameters=sum(p.numel() for p in model.parameters()), layers=len(model.model))
                from ultralytics.utils.torch_utils import get_flops
                value = get_flops(model, imgsz=32)
                descriptor["gflops_reference_32"] = value if value and math.isfinite(value) else None
            elif task == "classification":
                from yolo_tuner.vision_models import build
                model = build(task, config, classes=classes)
                descriptor.update(parameters=sum(p.numel() for p in model.parameters()), layers=sum(1 for _ in model.modules()))
            descriptors[key] = descriptor
    finally:
        torch.set_num_threads(prior)
    return descriptors


def descriptor_for(protocol, config):
    return protocol.get("model_descriptors", {}).get(architecture_key(config), {})
