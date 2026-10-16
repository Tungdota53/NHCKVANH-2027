"""Fixed-fidelity detection worker. No downloads, implicit AMP/batch recovery or test use."""

import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


class ProxyComplete(Exception):
    pass


def finite_losses(losses):
    import torch
    values = list(losses.values()) if isinstance(losses, dict) else list(losses) if isinstance(losses, (list, tuple)) else [] if losses is None else [losses]
    return bool(values) and all(torch.isfinite(torch.as_tensor(v)).all().item() for v in values)


def execute(req, output):
    os.environ["YOLO_CONFIG_DIR"] = str(Path(req["project"]) / "library_config")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import psutil
    import torch
    from ultralytics import YOLO, settings
    import ultralytics.engine.trainer as trainer_module
    from ultralytics.utils.torch_utils import init_seeds
    from yolo_tuner.pipeline import write_json
    from .integrity import sha
    from .runner import actual_matches, training_evidence_complete

    settings.update({"sync": False})
    config = req["config"]
    cuda = req["device"] not in {"cpu", "mps"}
    if cuda:
        torch.cuda.set_device(int(req["device"]))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif config["precision"] != "fp32":
        raise ValueError("CPU/MPS protocol requires FP32")
    # Capability was checked by parent; freeze requested precision instead of
    # allowing library AMP-check to download a different model or silently disable it.
    # This is not a numerical-equivalence certificate for FP16/BF16.
    trainer_module.check_amp = lambda model: True
    init_seeds(req["seed"], deterministic=True)
    state = {"config": config, "status": "running", "epochs_completed": 0,
             "memory_mb": 0.0, "peak_reserved_mb": 0.0, "peak_allocated_mb": 0.0,
             "peak_rss_mb": 0.0, "learning_curve": [], "mode": req["mode"],
             "precision_policy": "explicit frozen precision; no implicit AMP-download or fallback; finite metrics required",
             "full_data": req["mode"] == "train"}
    started = time.monotonic()
    timing, waiting = [], [False]
    tick = [started, started]

    def capture(trainer):
        state["peak_rss_mb"] = max(state["peak_rss_mb"], psutil.Process().memory_info().rss / 2**20)
        if cuda:
            state["peak_reserved_mb"] = max(state["peak_reserved_mb"], torch.cuda.max_memory_reserved() / 2**20)
            state["peak_allocated_mb"] = max(state["peak_allocated_mb"], torch.cuda.max_memory_allocated() / 2**20)
        state["memory_mb"] = state["peak_reserved_mb"] if cuda else state["peak_rss_mb"]
        opt = trainer.optimizer
        state["optimizer_state_created"] = bool(opt.state)
        state["training_loss_finite"] = finite_losses(trainer.tloss)
        if not state["training_loss_finite"]:
            raise ValueError("Training loss is missing or non-finite; no successful verification")
        actual_precision = "fp32" if not trainer.amp or not cuda else "bf16" if trainer.args.amp == "bf16" else "fp16"
        state["actual"] = {"batch": int(trainer.batch_size), "imgsz": int(trainer.args.imgsz),
                           "precision": actual_precision, "accumulation": int(trainer.accumulate),
                           "optimizer": type(opt).__name__,
                           "lr0": float(opt.param_groups[0].get("initial_lr", trainer.args.lr0)),
                           "weight_decay": max(float(group["weight_decay"]) for group in opt.param_groups)}
        state["actual_optimizer_groups"] = [{"lr": float(g["lr"]), "weight_decay": float(g["weight_decay"])} for g in opt.param_groups]
        if not actual_matches(config, state["actual"]):
            raise ValueError("Actual training parameters differ from the frozen configuration")
        state["wall_seconds"] = time.monotonic() - started
        state["last_checkpoint"] = str(trainer.last) if trainer.last.is_file() else None
        state["best_checkpoint"] = str(trainer.best) if trainer.best.is_file() else None

    def epoch_start(trainer):
        if cuda:
            torch.cuda.synchronize()
        waiting[0] = True
        tick[0] = time.monotonic()

    def train_end(trainer):
        if cuda:
            torch.cuda.synchronize()
        tick[1] = time.monotonic()
        capture(trainer)

    def batch_end(trainer):
        capture(trainer)

    def fit_end(trainer):
        capture(trainer)
        if not waiting[0]:
            return  # library final best.pt validation, not another training epoch
        waiting[0] = False
        metrics = trainer.metrics or {}
        q = float(metrics.get("metrics/mAP50-95(B)", float("nan")))
        m50 = float(metrics.get("metrics/mAP50(B)", float("nan")))
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (q, m50)):
            raise ValueError("Detection validation lacks finite mAP50-95/mAP50")
        now = time.monotonic()
        timing.append((tick[1] - tick[0], now - tick[1]))
        state["epochs_completed"] = trainer.epoch + 1
        state["learning_curve"].append({"epoch": state["epochs_completed"], "map50_95": q, "map50": m50})
        best = max(state["learning_curve"], key=lambda r: (r["map50_95"], r["epoch"]))
        state.update(quality_score=q, best_quality=best["map50_95"], best_epoch=best["epoch"],
                     map50_95_proxy=q, map50_proxy=m50,
                     best_map50_95=best["map50_95"], best_map50=best["map50"],
                     measured_loop_seconds=sum(t + v for t, v in timing))
        train_s, val_s = timing[-1]
        state["estimated_full_seconds"] = req["full_epochs"] * (
            train_s * req["full_counts"]["train"] / req["counts"]["train"] +
            val_s * req["full_counts"]["val"] / req["counts"]["val"])
        write_json(output, state)
        if req["mode"] == "search" and state["epochs_completed"] >= req["trial_epochs"]:
            raise ProxyComplete()

    model = YOLO(config["model"])
    if model.task != "detect":
        raise ValueError("Only YOLO detection is supported by this adapter")
    for event, callback in (("on_train_epoch_start", epoch_start), ("on_train_epoch_end", train_end),
                            ("on_train_batch_end", batch_end), ("on_fit_epoch_end", fit_end)):
        model.add_callback(event, callback)
    kwargs = {"data": req["data"], "epochs": req["full_epochs"], "imgsz": config["imgsz"],
              "batch": config["batch"], "device": req["device"], "optimizer": config["optimizer"],
              "lr0": config["lr0"], "weight_decay": config["weight_decay"],
              "amp": False if config["precision"] == "fp32" else "bf16" if config["precision"] == "bf16" else True,
              "nbs": config["batch"] * config["accumulation"], "warmup_epochs": 0.0,
              "seed": req["seed"], "deterministic": True, "workers": 0, "cache": False,
              "patience": req["full_epochs"] + 1, "save": req["mode"] == "train", "val": True,
              "close_mosaic": 0, "plots": False, "verbose": False, "project": req["project"],
              "name": "training", "exist_ok": True, "pretrained": config["model"].endswith(".pt")}
    try:
        model.train(**kwargs)
        capture(model.trainer)
        if state["epochs_completed"] != req["full_epochs"]:
            raise ValueError("Training did not complete declared horizon")
        best = Path(state["best_checkpoint"])
        state["checkpoint_sha256"] = sha(best)
        state["status"] = "ok"
    except ProxyComplete:
        state["status"] = "ok"
    except Exception as exc:
        state["status"] = "oom" if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower() else "error"
        state["error"] = str(exc)
        traceback.print_exc()
    if state["status"] == "ok" and not training_evidence_complete(state):
        state.update(status="error", error="No optimizer update observed; accumulation may exceed total training microsteps")
    state["wall_seconds"] = time.monotonic() - started
    write_json(output, state)


def main():
    from yolo_tuner.pipeline import write_json
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        execute(request, Path(sys.argv[2]))
    except Exception as exc:
        write_json(Path(sys.argv[2]), {"config": request["config"], "status": "error", "error": str(exc)})
        traceback.print_exc()


if __name__ == "__main__":
    main()
