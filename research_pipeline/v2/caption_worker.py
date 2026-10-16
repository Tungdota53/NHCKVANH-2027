"""V2-owned caption training with generated BLEU/chrF selection and bounded IO."""

import json
import math
from pathlib import Path
import random
import shutil
import time


def execute(request, result_path):
    import numpy as np
    import psutil
    import torch
    from torch.nn import functional as F
    from torch.utils.data import Dataset, DataLoader
    from yolo_tuner.vision_models import build
    from .vision_worker import atomic, tokens
    from yolo_tuner.caption_features import FeatureStore, image_tensor, dataset_signature
    from yolo_tuner.captioning import decode_ids, score_captions, quality_assessment

    seed = request["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(request.get("threads", 4))
    config, vocab = request["config"], request["vocabulary"]
    if config.get("deterministic"):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
    device = torch.device(request["device"])
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
    manifest = json.loads(Path(request["manifest"]).read_text(encoding="utf-8"))
    data_signature = dataset_signature(manifest)
    maximum = request["max_tokens"]
    store = FeatureStore(config["feature_cache"]) if config.get("feature_cache") else None
    if store:
        recipe = store.metadata["recipe"]
        if (config.get("encoder") != "resnet18" or recipe["imgsz"] != config["imgsz"]
                or recipe["grid"] != config["feature_grid"] or recipe["weight_sha256"] != config["encoder_sha256"]):
            raise ValueError("Feature cache recipe differs from model/preprocessing")

    from .loaders import ImageRecords, resolve_policy, make_loader, observed_loaders
    if store:
        raise ValueError("V2 tiny encoder needs decoded images, not frozen encoder features")
    policy = resolve_policy(request.get("data_loader"), request["device"],
                            len(manifest["splits"]["train"]), request.get("logical_cpus", request.get("threads", 1)))
    train_set, val_set = [ImageRecords(manifest["splits"][split], config, "captioning", vocab, maximum,
                                      policy["cache_bytes_per_instance"]) for split in ("train", "val")]
    if not len(train_set) or not len(val_set):
        raise ValueError("Caption training needs independent train and validation data")
    train_generator = torch.Generator()
    loader = make_loader(train_set, config["batch"], policy, True, train_generator)
    val_loader = make_loader(val_set, config["batch"], policy)
    model = build("captioning", config, len(vocab), load_pretrained=not request.get("resume")).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config["lr0"], weight_decay=config.get("weight_decay", 0.01))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5)
    use_amp = cuda and config["precision"] in {"fp16", "bf16"}
    dtype = torch.bfloat16 if config["precision"] == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)
    project = Path(request["project"])
    last_path, best_path = project / "last.pt", project / "best.pt"
    start_epoch, best_quality, bad_epochs, best_epoch = 0, -math.inf, 0, 0
    prior_curve = []
    if request.get("resume"):
        payload = torch.load(request["resume"], map_location=device, weights_only=True)
        if payload["task"] != "captioning" or payload["vocabulary"] != vocab or payload["max_tokens"] != maximum:
            raise ValueError("Resume task/vocabulary/token policy differs")
        recorded_signature = payload.get("training_dataset_signature")
        if recorded_signature is None:
            # First v2 checkpoints can be verified against their saved request.
            previous_request = json.loads((Path(request["resume"]).parent / "request.json").read_text(encoding="utf-8"))
            previous_manifest = json.loads(Path(previous_request["manifest"]).read_text(encoding="utf-8"))
            recorded_signature = dataset_signature(previous_manifest)
        if recorded_signature != data_signature:
            raise ValueError("Resume dataset/targets differ; subset search epochs cannot count as full-training epochs")
        for key in ("model", "encoder", "hidden", "width", "imgsz", "feature_grid", "encoder_sha256"):
            if payload["config"].get(key) != config.get(key):
                raise ValueError(f"Resume architecture/preprocessing differs: {key}")
        model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"]); scheduler.load_state_dict(payload["scheduler"])
        start_epoch = payload["epoch"]
        best_quality = payload["best_quality"]; best_epoch = payload["best_epoch"]
        bad_epochs = payload["bad_epochs"]; prior_curve = payload["learning_curve"]
        torch.set_rng_state(payload["rng_state"].cpu())
        if cuda and payload.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all([x.cpu() for x in payload["cuda_rng_states"]])
        # Keep the earlier best checkpoint when recovering into a new trial folder.
        prior_best = Path(request["resume"]).parent / "best.pt"
        if prior_best.is_file():
            if prior_best.resolve() != best_path.resolve():
                shutil.copy2(prior_best, best_path)
        else:
            raise ValueError("Resume needs best.pt next to last.pt to preserve checkpoint selection")
        if Path(request["resume"]).resolve() != last_path.resolve():
            shutil.copy2(request["resume"], last_path)
    target_epochs = request["full_epochs"] if request["mode"] == "train" else request["trial_epochs"]
    state = {"data_loader": observed_loaders(policy, (loader, val_loader), (train_set, val_set)), "status": "running", "task": "captioning", "config": config, "epochs_completed": start_epoch,
             "training_completed": False, "early_stopped": False, "learning_curve": prior_curve,
             "metric_name": "validation_generated_bleu4_chrf", "pretrained": config.get("pretrained", False),
             "model_origin": "image-conditioned GRU + spatial cross-attention; frozen pretrained encoder when available",
             "peak_reserved_mb": 0, "peak_allocated_mb": 0, "peak_rss_mb": 0,
             "torch_version": str(torch.__version__), "device": str(device), "memory_mb": 0}
    already_stopped = bool(request.get("resume") and request["mode"]=="train"
                           and payload["metrics"].get("early_stopped")
                           and payload["metrics"].get("training_completed")
                           and target_epochs==payload["config"].get("epochs"))
    if start_epoch >= target_epochs or already_stopped:
        state.update(payload["metrics"], status="ok", already_completed=True, training_completed=True,
                     last_checkpoint=str(last_path), best_checkpoint=str(best_path), estimated_full_seconds=0.0,
                     best_epoch=best_epoch, best_quality=best_quality,
                     quality_eligible=payload["metrics"]["quality_assessment"]["status"]!="quality_not_established")
        state["peak_rss_mb"]=psutil.Process().memory_info().rss/2**20
        if cuda:
            state["peak_reserved_mb"]=torch.cuda.max_memory_reserved(device)/2**20
        state["memory_mb"]=state["peak_reserved_mb"] if cuda else state["peak_rss_mb"]
        atomic(result_path, state)
        return
    if request.get("resume") and payload["metrics"].get("early_stopped"):
        bad_epochs=0  # An explicit larger --epochs grants a fresh patience window.
    process = psutil.Process()
    started = time.monotonic()
    epoch_times = []
    accumulation_observation = {"update_attempts": 0, "max_images_per_update": 0,
                                "max_microbatches_per_update": 0, "nominal_effective_batch": config["effective_batch"]}
    baseline = request.get("caption_baseline", "")
    val_records = manifest["splits"]["val"]
    refs = [r["captions"] for r in val_records]
    baseline_metrics = score_captions([baseline] * len(val_records), refs)

    def sync():
        if cuda:
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    def resources():
        state["peak_rss_mb"] = max(state["peak_rss_mb"], process.memory_info().rss / 2**20)
        if cuda:
            state["peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / 2**20
            state["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
        state["memory_mb"] = state["peak_reserved_mb"] if cuda else state["peak_rss_mb"]

    def loss_for(inputs, targets):
        logits = model.forward_features(inputs, targets[:, :-1]) if store else model(inputs, targets[:, :-1])
        truth = targets[:, 1:]
        count = max(1, int((truth != 0).sum().item()))
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), truth.reshape(-1),
                               ignore_index=0, reduction="sum", label_smoothing=config.get("label_smoothing", 0.05))
        return loss, count

    for epoch in range(start_epoch, target_epochs):
        sync(); epoch_start = time.monotonic()
        train_generator.manual_seed(seed + epoch)
        model.train(); optimizer.zero_grad(set_to_none=True)
        train_sum, train_count, pending_tokens, pending_images, pending_microbatches = 0.0, 0, 0, 0, 0
        for index, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device, non_blocking=policy["non_blocking"]), targets.to(device, non_blocking=policy["non_blocking"])
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                loss, count = loss_for(inputs, targets)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Non-finite caption loss; reduce LR or use FP32")
            scaler.scale(loss).backward()
            train_sum += float(loss.detach()); train_count += count
            pending_tokens += count; pending_images += len(inputs)
            pending_microbatches += 1
            if pending_images >= config["effective_batch"] or index + 1 == len(loader):
                accumulation_observation["update_attempts"] += 1
                accumulation_observation["max_images_per_update"] = max(accumulation_observation["max_images_per_update"], pending_images)
                accumulation_observation["max_microbatches_per_update"] = max(accumulation_observation["max_microbatches_per_update"], pending_microbatches)
                scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(pending_tokens)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                pending_tokens = pending_images = pending_microbatches = 0
            resources()
        sync(); train_end = time.monotonic()
        model.eval(); val_sum, val_count = 0.0, 0
        with torch.inference_mode():
            for inputs, targets in val_loader:
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                    loss, count = loss_for(inputs.to(device, non_blocking=policy["non_blocking"]), targets.to(device, non_blocking=policy["non_blocking"]))
                val_sum += float(loss.float()); val_count += count
                resources()
            hypotheses = []
            for offset in range(0, len(val_records), config["batch"]):
                group = val_records[offset:offset + config["batch"]]
                inputs = torch.stack([val_set.image(r) for r in group])
                if policy["pin_memory"]:
                    inputs = inputs.pin_memory()
                inputs = inputs.to(device, non_blocking=policy["non_blocking"])
                # FP32 generation matches independent evaluate-caption, not teacher forcing.
                ids = model.generate_features(inputs, maximum) if store else model.generate(inputs, maximum)
                hypotheses.extend(decode_ids(row, vocab) for row in ids.cpu().tolist())
                resources()
        sync(); ended = time.monotonic()
        val_loss = val_sum / val_count
        if not math.isfinite(val_loss):
            raise ValueError("Non-finite validation loss")
        metrics = score_captions(hypotheses, refs)
        assessment = quality_assessment(metrics, baseline_metrics, hypotheses)
        quality = (metrics["bleu4"] + metrics["chrf"]) / 200
        scheduler.step(quality)
        improved = quality > best_quality + config.get("min_delta", 0.001)
        if improved:
            best_quality, bad_epochs, best_epoch = quality, 0, epoch + 1
        else:
            bad_epochs += 1
        epoch_times.append((train_end - epoch_start, ended - train_end))
        measured = epoch_times[1:] or epoch_times
        train_s = sum(x[0] for x in measured) / len(measured)
        val_s = sum(x[1] for x in measured) / len(measured)
        # Cached feature I/O is identical in search/full. Validation includes generation.
        full_time = request["full_epochs"] * (train_s * request["full_counts"]["train"] / len(train_set)
                    + val_s * request["full_counts"]["val"] / len(val_set))
        stopped = request["mode"] == "train" and bad_epochs >= config.get("patience", 5)
        state["data_loader"] = observed_loaders(policy, (loader, val_loader), (train_set, val_set))
        state["accumulation_observation"] = dict(accumulation_observation)
        state.update(status="ok", epochs_completed=epoch + 1, validation_loss=val_loss,
                     train_loss=train_sum / train_count, quality_score=quality, caption_metrics=metrics,
                     caption_baseline_metrics=baseline_metrics, quality_eligible=assessment["status"] != "quality_not_established",
                     quality_assessment=assessment, best_epoch=best_epoch, best_quality=best_quality,
                     early_stopped=stopped, training_completed=stopped or epoch + 1 == target_epochs,
                     train_examples=len(train_set), validation_examples=len(val_set), validation_images=len(val_records),
                     train_epoch_seconds=train_s, val_epoch_seconds=val_s, estimated_full_seconds=full_time,
                     throughput=len(train_set) / max(train_end - epoch_start, 1e-9),
                     measured_loop_seconds=sum(sum(x) for x in epoch_times), wall_seconds=ended - started,
                     last_checkpoint=str(last_path), best_checkpoint=str(best_path),
                     lr=optimizer.param_groups[0]["lr"], time_estimate_basis="cached features + teacher-forced validation + free generation; point estimate")
        state["learning_curve"].append({"epoch": epoch + 1, "train_loss": state["train_loss"], "validation_loss": val_loss,
                                        "bleu4": metrics["bleu4"], "chrf": metrics["chrf"], "quality_score": quality, "lr": state["lr"]})
        payload = {"task": "captioning", "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scaler": scaler.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch + 1,
                   "best_quality": best_quality, "best_epoch": best_epoch, "bad_epochs": bad_epochs,
                   "config": config, "vocabulary": vocab, "max_tokens": maximum, "names": manifest["names"],
                   "training_dataset_signature": data_signature,
                   "learning_curve": state["learning_curve"], "rng_state": torch.get_rng_state(),
                   "cuda_rng_states": torch.cuda.get_rng_state_all() if cuda else [],
                   "metrics": {k: state[k] for k in ("train_loss", "validation_loss", "quality_score", "caption_metrics",
                       "quality_assessment", "quality_eligible", "training_completed", "early_stopped", "train_examples", "validation_examples")}}
        temporary = project / "last.tmp"
        torch.save(payload, temporary); temporary.replace(last_path)
        if improved:
            temporary_best = project / "best.tmp"
            shutil.copy2(last_path, temporary_best); temporary_best.replace(best_path)
        atomic(project / "caption_validation.json", {"split": "val", "epoch": epoch + 1, "metrics": metrics,
               "predictions": [{"image": r["image"], "caption": cap} for r, cap in zip(val_records, hypotheses)]})
        atomic(result_path, state)
        if stopped:
            break


