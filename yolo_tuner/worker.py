"""Isolated Ultralytics worker. Parent owns timeout and GPU process lifetime."""

from dataclasses import asdict
import json
import math
import os
import random
from pathlib import Path
import sys
import time
import traceback


class ProbeComplete(Exception):
    pass


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def execute(request, result_path):
    # Keep library configuration local to this trial and avoid implicit installs.
    os.environ["YOLO_CONFIG_DIR"] = str(Path(request["project"]) / "library_config")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import torch
    import psutil
    import ultralytics
    from ultralytics import YOLO

    ultralytics.settings.update({"sync": False})
    import numpy as np
    random.seed(request["seed"])
    np.random.seed(request["seed"])
    torch.manual_seed(request["seed"])

    config = request["config"]
    mode = request["mode"]
    device = request["device"]
    cuda = device not in {"cpu","mps"}
    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Install CUDA-enabled PyTorch or explicitly use --device cpu.")
        torch.cuda.set_device(int(device))
        if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("GPU does not support BF16")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    state = {
        "status": "running", "config": config, "mode": mode,
        "epochs_completed": 0, "peak_reserved_mb": 0.0, "peak_allocated_mb": 0.0,
        "actual_batch": config["batch"], "actual_precision": config["precision"], "peak_rss_mb":0.0,
        "torch_version": torch.__version__, "ultralytics_version": ultralytics.__version__,
        "accuracy_final_prediction": None, "accuracy_prediction_interval": None,
        "time_prediction_interval": None, "memory_prediction_interval": None,
    }
    timings, curves = [], []
    started = time.monotonic()
    epoch_started = started
    train_finished = started
    batch_count = 0
    epoch_batch_count = 0
    profile_image_count = 0
    profile_started = None
    awaiting_fit = False

    def capture(trainer):
        state["peak_rss_mb"]=max(state["peak_rss_mb"],psutil.Process().memory_info().rss/2**20)
        state["actual_batch"] = int(trainer.batch_size)
        if not trainer.amp:
            state["actual_precision"] = "fp32"
        if cuda:
            state["peak_reserved_mb"] = max(state["peak_reserved_mb"], torch.cuda.max_memory_reserved() / 2**20)
            state["peak_allocated_mb"] = max(state["peak_allocated_mb"], torch.cuda.max_memory_allocated() / 2**20)
        state["memory_mb"]=state["peak_reserved_mb"] if cuda else state["peak_rss_mb"]
        state["wall_seconds"] = time.monotonic() - started
        state["last_checkpoint"] = str(trainer.last) if trainer.last.is_file() else None

    def epoch_start(trainer):
        nonlocal epoch_started, profile_started, awaiting_fit, epoch_batch_count
        if cuda:
            torch.cuda.synchronize()
        epoch_started = time.monotonic()
        awaiting_fit = True
        epoch_batch_count = 0
        if profile_started is None:
            profile_started = epoch_started

    def batch_end(trainer):
        nonlocal batch_count, epoch_batch_count, profile_image_count
        batch_count += 1
        capture(trainer)
        remaining_images = len(trainer.train_loader.dataset) - epoch_batch_count * state["actual_batch"]
        profile_image_count += min(state["actual_batch"], max(0, remaining_images))
        epoch_batch_count += 1
        if mode == "profile" and batch_count >= request.get("profile_steps", 8):
            if cuda:
                torch.cuda.synchronize()
            elapsed = time.monotonic() - profile_started
            state["images_per_second_probe"] = profile_image_count / max(elapsed, 1e-6)
            state["optimizer_state_created"] = bool(trainer.optimizer.state)
            state["status"] = "ok" if state["optimizer_state_created"] else "error"
            state["error"] = None if state["optimizer_state_created"] else "Memory probe did not create optimizer state"
            write_json(result_path, state)
            raise ProbeComplete()

    def train_end(trainer):
        nonlocal train_finished
        if cuda:
            torch.cuda.synchronize()
        train_finished = time.monotonic()
        capture(trainer)
        if mode == "profile":
            # Avoid unnecessary validation at epoch end; another epoch continues.
            return

    def fit_end(trainer):
        nonlocal awaiting_fit
        capture(trainer)
        if not awaiting_fit:
            # Ultralytics also emits this event during final best.pt validation.
            # It is not another completed training epoch or an epoch timing sample.
            if mode == "train":
                metrics = trainer.metrics or {}
                value = float(metrics.get("metrics/mAP50-95(B)", float("nan")))
                if math.isfinite(value):
                    state["final_best_map50_95"] = value
            return
        awaiting_fit = False
        now = time.monotonic()
        timings.append((train_finished - epoch_started, now - train_finished))
        metrics = trainer.metrics or {}
        metric = float(metrics.get("metrics/mAP50(B)", float("nan")))
        metric95 = float(metrics.get("metrics/mAP50-95(B)", float("nan")))
        if mode != "profile" and not all(math.isfinite(m) for m in (metric, metric95)):
            raise RuntimeError("Validation did not provide finite detection mAP metrics")
        state["epochs_completed"] = trainer.epoch + 1
        if mode != "profile":
            state["map50_proxy"] = metric
            state["map50_95_proxy"] = metric95
            curves.append({"epoch": trainer.epoch + 1, "map50": metric, "map50_95": metric95})
            state["learning_curve"] = curves
            # Drop epoch 1 setup-heavy timing if multiple complete epochs exist.
            measured = timings[1:] or timings
            train_s = sum(t[0] for t in measured) / len(measured)
            val_s = sum(t[1] for t in measured) / len(measured)
            stats = request["dataset_stats"]
            state["train_epoch_seconds"] = train_s
            state["val_epoch_seconds"] = val_s
            state["measured_loop_seconds"] = sum(train + val for train, val in timings)
            state["estimated_full_seconds"] = request["full_epochs"] * (
                train_s * stats["train_images"] / stats["train_subset_images"] +
                val_s * stats["val_images"] / stats["val_subset_images"])
            state["time_estimate_basis"] = "measured train/val times scaled separately by split counts; no confidence guarantee"
        state["status"] = "ok"
        write_json(result_path, state)
        if mode == "search" and state["epochs_completed"] >= request["trial_epochs"]:
            raise ProbeComplete()

    model = YOLO(request.get("resume") or config["model"])
    if model.task != "detect":
        raise ValueError("This tuner currently supports YOLO detection models only")
    model.add_callback("on_train_epoch_start", epoch_start)
    model.add_callback("on_train_batch_end", batch_end)
    model.add_callback("on_train_epoch_end", train_end)
    model.add_callback("on_fit_epoch_end", fit_end)
    kwargs = {
        "data": request["data"], "epochs": request["full_epochs"],
        "imgsz": config["imgsz"], "batch": config["batch"], "device": device,
        "amp": False if config["precision"] == "fp32" else ("bf16" if config["precision"] == "bf16" else True),
        "freeze": config["freeze"], "nbs": config["nbs"], "optimizer": config["optimizer"],
        "lr0": config["lr0"], "warmup_epochs": 0.5, "warmup_bias_lr": 0.0,
        "close_mosaic": 0, "seed": request["seed"], "workers": 0,
        "cache": False, "plots": False, "save": mode == "train", "val": mode != "profile",
        "patience": request["full_epochs"] + 1,
        "project": request["project"], "name": "training", "exist_ok": True,
        "verbose": False,
    }
    if mode == "profile":
        # Create optimizer states immediately, so even batch=1 probes include them.
        kwargs.update(nbs=config["batch"], warmup_epochs=0.0,
                      epochs=max(request["full_epochs"], request.get("profile_steps", 8)))
    if request.get("resume"):
        kwargs["resume"] = True
    try:
        model.train(**kwargs)
        state["status"] = "ok"
        state["wall_seconds"] = time.monotonic() - started
        write_json(result_path, state)
    except ProbeComplete:
        pass
    except Exception as exc:
        if getattr(model, "trainer", None) is not None:
            capture(model.trainer)
        state["status"] = "oom" if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower() else "error"
        state["error"] = str(exc)
        write_json(result_path, state)
        traceback.print_exc()


def main():
    request_path, result_path = sys.argv[1:3]
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    try:
        execute(request, result_path)
    except Exception as exc:
        write_json(result_path, {"status": "error", "config": request["config"], "error": str(exc)})
        traceback.print_exc()


if __name__ == "__main__":
    main()
