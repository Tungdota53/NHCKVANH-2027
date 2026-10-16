"""Instrument frozen real task workers without changing legacy source files."""

from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

from yolo_tuner.pipeline import write_json
from research_pipeline.integrity import sha


def execute(req, output):
    import torch
    from .telemetry import PhaseMonitor
    config, task = req["config"], req["task"]
    cuda = req["device"] not in {"cpu", "mps"}
    if cuda:
        torch.cuda.set_device(int(req["device"]))
    monitor = PhaseMonitor(req["device"])
    optimizers, steps, original_save = [], [0], torch.save
    observed = {"training_precision": None, "max_microbatch": 0, "architecture": None, "image_shapes": []}
    original_adamw_init = torch.optim.AdamW.__init__

    def saving(*a, **kw):
        with monitor.during("checkpoint"):
            return original_save(*a, **kw)

    raw = {}
    try:
        # Optimizer classes stay classes: framework selection uses class identity.
        def attach_step(optimizer, *a, **kw):
            optimizers.append(optimizer)
            optimizer._research_initial_lr = float(optimizer.param_groups[0].get("initial_lr", optimizer.param_groups[0]["lr"]))
            optimizer.register_step_post_hook(lambda *_: steps.__setitem__(0, steps[0] + 1))
        def adamw_init(optimizer, *a, **kw):
            original_adamw_init(optimizer, *a, **kw)
            attach_step(optimizer)
        with patch("torch.save", side_effect=saving):
            if task == "detection":
                import ultralytics
                import ultralytics.data.utils
                import ultralytics.models.yolo.detect.train as detect_training
                from .loaders import resolve_policy, detection_loader_builder
                from research_pipeline.worker import execute as legacy_execute
                original_yolo = ultralytics.YOLO
                policy = resolve_policy(req.get("data_loader"), req["device"], req["counts"]["train"], req.get("logical_cpus", req.get("threads", 1)))
                detection_loader_observed = {}
                def training_started(trainer):
                    attach_step(trainer.optimizer)
                    frozen = int(trainer.args.freeze or 0)
                    requested_names = [name for name, _ in trainer.model.named_parameters() if any(name.startswith(f"model.{i}.") for i in range(frozen))]
                    parameters = dict(trainer.model.named_parameters())
                    observed.update(freeze=frozen, freeze_blocks_verified=all(not parameters[n].requires_grad for n in requested_names),
                                    frozen_parameters=sum(p.numel() for p in parameters.values() if not p.requires_grad),
                                    trainable_parameters=sum(p.numel() for p in parameters.values() if p.requires_grad))
                    loaders = (trainer.train_loader, trainer.test_loader)
                    actual_workers = [loader.num_workers for loader in loaders]
                    actual_pin = [loader.pin_memory for loader in loaders]
                    detection_loader_observed.update(**policy, observed_workers=actual_workers,
                        observed_pin_memory=actual_pin,
                        verified=all(0 <= n <= policy["workers"] for n in actual_workers) and all(v == policy["pin_memory"] for v in actual_pin),
                        cache_enabled=False, cache_note="Detection keeps Ultralytics image cache disabled; requested decoded-cache budget applies to classification/captioning only")
                class InstrumentedYOLO(original_yolo):
                    def __init__(self, *a, **kw):
                        super().__init__(*a, **kw)
                        self.add_callback("on_train_start", training_started)
                        self.add_callback("on_train_epoch_start", lambda _: monitor.set_phase("training"))
                        self.add_callback("on_train_epoch_end", lambda _: monitor.set_phase("validation"))
                        self.add_callback("on_val_start", lambda _: monitor.set_phase("validation"))
                        self.add_callback("on_fit_epoch_end", lambda _: monitor.set_phase("epoch_overhead"))
                    def train(self, *a, **kw):
                        kw["freeze"] = config.get("freeze", 0)
                        kw["workers"] = policy["workers"]
                        kw["cache"] = False
                        return super().train(*a, **kw)
                with patch("ultralytics.YOLO", InstrumentedYOLO), patch("ultralytics.data.utils.check_font", return_value=None), patch.object(detect_training, "build_dataloader", detection_loader_builder(detect_training.build_dataloader, policy)):
                    legacy_execute(req, output)
            else:
                from yolo_tuner import vision_models
                from .vision_worker import execute as task_execute
                original_build = vision_models.build
                def build(*a, **kw):
                    model = original_build(*a, **kw)
                    observed["architecture"] = type(model).__name__
                    def before_forward(module, inputs):
                        if not module.training:
                            return
                        enabled = cuda and torch.is_autocast_enabled("cuda")
                        observed["training_precision"] = "bf16" if enabled and torch.get_autocast_dtype("cuda") == torch.bfloat16 else "fp16" if enabled else "fp32"
                        if inputs and hasattr(inputs[0], "shape"):
                            observed["max_microbatch"] = max(observed["max_microbatch"], int(inputs[0].shape[0]))
                            shape = list(inputs[0].shape[-2:])
                            if shape not in observed["image_shapes"]:
                                observed["image_shapes"].append(shape)
                    model.register_forward_pre_hook(before_forward)
                    old_train = model.train
                    def train(mode=True):
                        monitor.set_phase("training" if mode else "validation")
                        return old_train(mode)
                    model.train = train
                    return model
                task_config = {**config, "effective_batch": config["batch"] * config["accumulation"],
                                 "epochs": req["full_epochs"], "patience": req["full_epochs"] + 1,
                                 "min_delta": 0.0, "deterministic": True}
                task_req = {**req, "config": task_config,
                              "device": f"cuda:{req['device']}" if cuda else req["device"]}
                with patch("yolo_tuner.vision_models.build", side_effect=build), patch.object(torch.optim.AdamW, "__init__", adamw_init):
                    if task == "captioning":
                        from .pooling import deterministic_adaptive_pool2d
                        with patch("torch.nn.functional.adaptive_avg_pool2d", deterministic_adaptive_pool2d):
                            task_execute(task_req, output)
                    else:
                        task_execute(task_req, output)
            raw = json.loads(output.read_text(encoding="utf-8"))
            if task == "detection":
                raw["data_loader"] = detection_loader_observed
    except Exception as exc:
        raw = {"status": "oom" if "out of memory" in str(exc).lower() else "error", "error": str(exc)}
        traceback.print_exc()
    finally:
        telemetry = monitor.close()
    checkpoint = raw.get("best_checkpoint") if task in {"captioning", "detection"} else raw.get("last_checkpoint")
    quality = raw.get("best_quality", raw.get("quality_score"))
    actual = dict(raw.get("actual", {}))
    if optimizers:
        optimizer = optimizers[-1]
        actual.update(optimizer=type(optimizer).__name__,
                      lr0=optimizer._research_initial_lr,
                      weight_decay=max(float(g["weight_decay"]) for g in optimizer.param_groups))
    if task != "detection":
        # Check observed forward shape instead of only echoing the request.
        actual.update(batch=config["batch"], imgsz=config["imgsz"], precision=observed["training_precision"],
                      accumulation=config["accumulation"])
    from research_pipeline.runner import actual_matches
    verified = bool(optimizers) and actual_matches(config, actual)
    if task != "detection":
        expected_max = min(config["batch"], raw.get("train_examples", 0))
        accumulation = raw.get("accumulation_observation", {})
        expected_effective = min(config["batch"] * config["accumulation"], raw.get("train_examples", 0))
        verified = (verified and observed["max_microbatch"] == expected_max
                    and observed["image_shapes"] == [[config["imgsz"], config["imgsz"]]]
                    and accumulation.get("max_images_per_update") == expected_effective
                    and raw.get("data_loader", {}).get("verified") is True)
    if task == "detection" and "freeze" in config:
        actual["freeze"] = observed.get("freeze")
        verified = verified and actual["freeze"] == config["freeze"] and observed.get("freeze_blocks_verified") is True and observed.get("trainable_parameters", 0) > 0
        verified = verified and raw.get("data_loader", {}).get("verified") is True
    finite = raw.get("training_loss_finite", math.isfinite(raw.get("train_loss", float("nan"))))
    result = {**raw, "config": config, "task": task, "seed": req["seed"], "mode": req["mode"],
              "quality": quality, "actual": actual, "actual_config_verified": verified,
              "optimizer_steps": steps[0], "training_loss_finite": finite,
              "observed_forward": observed,
              "memory_mb": max(raw.get("memory_mb", 0), telemetry["peak_reserved_mb"] if cuda else telemetry["peak_rss_mb"]),
              "telemetry": telemetry, "best_checkpoint": checkpoint,
              "checkpoint_sha256": sha(checkpoint) if checkpoint and Path(checkpoint).is_file() else None}
    if result["status"] == "ok" and (not verified or not finite or steps[0] < 1 or quality is None or not math.isfinite(quality)):
        result.update(status="training_evidence_missing", error="Missing actual parameters, finite loss, optimizer update or finite validation quality")
    write_json(output, result)


def main():
    request, output = Path(sys.argv[1]), Path(sys.argv[2])
    req = json.loads(request.read_text(encoding="utf-8"))
    try:
        execute(req, output)
    except Exception as exc:
        write_json(output, {"status": "error", "config": req["config"], "error": str(exc),
                            "task": req["task"], "seed": req["seed"], "mode": req["mode"]})
        traceback.print_exc()


if __name__ == "__main__":
    main()
