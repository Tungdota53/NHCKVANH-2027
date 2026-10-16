"""Detection finite-domain protocol: budgeted decisions locked before oracle."""

from contextlib import contextmanager
from dataclasses import asdict
import json
import math
import os
import platform
from pathlib import Path
import random
import re
import time

import psutil
import yaml

from yolo_tuner.adapters import discover, subset, export_yolo
from yolo_tuner.core import Deadline
from yolo_tuner.pipeline import new_session, write_json, runtime_info
from yolo_tuner.research import Objective, digest, training_finished
from .integrity import sources, freeze_sources, runtime, sha, verify_live, hardware_changes
from .methods import METHODS, search
from .report import SCENARIOS, publish
from .runner import DetectionRunner


def competing_jobs():
    """Conservative Python ML-process preflight; not complete system isolation."""
    own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    jobs = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if process.pid in own or "python" not in (process.info["name"] or "").lower():
            continue
        try:
            if process.info["cmdline"] is None:
                jobs.append({"pid": process.pid, "command": "unreadable Python process; cannot rule out training"})
                continue
            cmd = " ".join(process.info["cmdline"])
        except psutil.AccessDenied:
            jobs.append({"pid": process.pid, "command": "unreadable Python process; cannot rule out training"})
            continue
        if any(s in cmd for s in ("yolo_tuner.vision_worker", "yolo_tuner.worker", "research_pipeline.worker", "research_pipeline.evaluation")) or re.search(r"train_config\.py\s+(verify-config|research|train|auto|run|tune|evaluate-detection)\b", cmd):
            jobs.append({"pid": process.pid, "command": cmd})
    return jobs


def require_idle(allow=False):
    jobs = competing_jobs()
    if jobs and not allow:
        raise ValueError(f"Other ML jobs active: {[j['pid'] for j in jobs]}. Refusing resource benchmark; use --prepare-only or wait. --allow-contention is smoke only, never certified.")
    return jobs


@contextmanager
def exclusive_run(folder):
    for file in folder.rglob("process.json"):
        record = json.loads(file.read_text(encoding="utf-8"))
        if record.get("status") != "running":
            continue
        try:
            active = psutil.Process(record["pid"])
            if abs(active.create_time() - record["create_time"]) < 0.01:
                raise ValueError("A study worker is still running; refusing duplicate execution")
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as exc:
            raise ValueError("Cannot establish study worker has stopped") from exc
    lock = folder / "run.lock"
    if lock.exists():
        record = json.loads(lock.read_text(encoding="utf-8"))
        try:
            active = psutil.Process(record["pid"])
            if abs(active.create_time() - record["create_time"]) < 0.01:
                raise ValueError("Study parent still running; refusing duplicate execution")
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as exc:
            raise ValueError("Cannot establish study process has stopped") from exc
        # Explicit single stale coordination file, not study data/results.
        lock.unlink()
    with lock.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "create_time": psutil.Process().create_time()}, handle)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def model_snapshot(name, folder):
    """Resolve installed architecture or local checkpoint; no network during trials."""
    path = Path(name)
    folder.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        path = path.resolve()
        destination = folder / (sha(path)[:12] + "_" + path.name)
        destination.write_bytes(path.read_bytes())
        return str(destination)
    match = re.fullmatch(r"(yolo11|yolo26|yolov8)([nslmx])\.yaml", name)
    if not match:
        raise ValueError(f"Model must be a local .pt/.yaml or an installed YOLO architecture: {name}")
    from importlib.metadata import distribution
    base, scale = match.groups()
    installed = Path(distribution("ultralytics").locate_file("ultralytics/cfg/models"))
    matches = list(installed.rglob(base + ".yaml"))
    if len(matches) != 1:
        raise ValueError(f"Installed runtime does not contain architecture {name}")
    config = yaml.safe_load(matches[0].read_text(encoding="utf-8"))
    config["scale"] = scale
    destination = folder / name
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return str(destination)


def grid(args, hw, folder):
    precision = args.precision or ("fp16" if hw.get("fp16") else "fp32")
    models = args.models or ["yolo11n.yaml", "yolo11s.yaml"]
    base = {"model": models[0], "imgsz": 320, "batch": 2, "precision": precision,
            "accumulation": 8, "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.0005}
    if args.grid:
        overrides = json.loads(Path(args.grid).read_text(encoding="utf-8"))
        if not isinstance(overrides, list) or not overrides or any(not isinstance(c, dict) or set(c) - set(base) for c in overrides):
            raise ValueError("Detection grid must be a nonempty list of supported config overrides")
    elif args.space == "expanded":
        overrides = [{"model": model, "imgsz": size, "batch": batch, "accumulation": 16 // batch, "lr0": lr}
                     for model in models for size in (320, 640) for batch in (2, 8) for lr in (0.001, 0.0003)]
    else:
        overrides = [{}, {"imgsz": 640}, {"model": models[-1]}, {"lr0": 0.0003}]
    output, model_paths = [], {}
    for override in overrides:
        c = {**base, **override}
        if args.precision:
            c["precision"] = args.precision
        if any(type(c[k]) is not int or c[k] < 1 for k in ("imgsz", "batch", "accumulation")) or c["imgsz"] < 64 or c["imgsz"] % 32:
            raise ValueError("Detection image size >=64 divisible by32, batch/accumulation positive integers")
        if c["optimizer"] not in {"AdamW", "SGD"}:
            raise ValueError("Use explicit AdamW or SGD; optimizer auto would change LR")
        if not math.isfinite(c["lr0"]) or c["lr0"] <= 0 or not math.isfinite(c["weight_decay"]) or c["weight_decay"] < 0:
            raise ValueError("Invalid learning rate/weight decay")
        if c["precision"] not in {"fp32", "fp16", "bf16"} or c["precision"] != "fp32" and not hw.get(c["precision"], False):
            raise ValueError("Requested precision is not supported on selected runtime")
        if hw["device"] in {"cpu", "mps"} and c["precision"] != "fp32":
            raise ValueError("CPU/MPS protocol requires FP32")
        name = c["model"]
        if name not in model_paths:
            model_paths[name] = model_snapshot(name, folder / "models")
        c["model"] = model_paths[name]
        if c not in output:
            output.append(c)
    return output


def study(args, manifest=None):
    if args.full_epochs < max(args.rungs) or args.rungs != sorted(set(args.rungs)) or min(args.rungs) < 1:
        raise ValueError("Rungs must be increasing and within full-epochs")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Training seeds must be nonempty and unique")
    if any(not math.isfinite(v) or v <= 0 for v in (args.search_budget, args.timeout_hours, args.trial_timeout, args.train_subset, args.val_subset)):
        raise ValueError("Budgets/timeouts/subset sizes must be positive")
    if len(set(args.methods)) != len(args.methods) or any(m not in METHODS for m in args.methods):
        raise ValueError("Invalid/duplicate search methods")
    if args.command == "verify-config" and args.methods != ["adaptive_verified"]:
        raise ValueError("Production verification uses adaptive_verified only")
    jobs = competing_jobs() if args.prepare_only else require_idle(args.allow_contention)
    manifest = manifest or discover(args.data, args.split_seed, "detection")
    if manifest["task"] != "detection" or not manifest["validation_ready"]:
        raise ValueError("Detection research needs genuine detection labels and independent validation")
    hw = runtime_info(args.device)
    hw["host_label"] = platform.node()
    if not hw["ml_runtime_ready"]:
        raise ValueError("Actual ML runtime required")
    folder = new_session(args.output)
    clock = Deadline(args.timeout_hours * 3600)
    write_json(folder / "dataset.json", manifest)
    sampled = subset(manifest, args.train_subset, args.val_subset)
    write_json(folder / "subset.json", sampled)
    export_yolo(manifest, folder / "full_data")
    export_yolo(sampled, folder / "subset_data")
    configs = grid(args, hw, folder)
    data_files = {p.relative_to(folder).as_posix(): sha(p) for sub in ("full_data", "subset_data")
                  for p in sorted((folder / sub).rglob("*")) if p.is_file()}
    memory = args.max_memory if args.max_memory is not None else (hw.get("vram_free_mb") or hw["ram_available_mb"]) * 0.8
    objective = Objective(args.objective, args.min_quality, args.max_time, memory, args.time_scale, args.memory_scale, tuple(args.weights))
    protocol = {"schema_version": 2, "adapter": "yolo_research", "task": "detection",
                "implementation_version": "0.6.1", "require_optimizer_evidence": True,
                "mode": "online_verification" if args.command == "verify-config" else "independent_research",
                "dataset_source": manifest["source"], "dataset_sha256": digest(manifest), "subset_sha256": digest(sampled),
                "hardware": hw, "configs": configs, "training_seeds": args.seeds, "split_seed": args.split_seed,
                "full_epochs": args.full_epochs, "rungs": args.rungs, "methods": args.methods,
                "search_budget_seconds": args.search_budget * 60, "trial_timeout": args.trial_timeout,
                "timeout_seconds": args.timeout_hours * 3600, "objective": asdict(objective),
                "quality_definition": "best validation mAP50-95 checkpoint within fixed full horizon",
                "memory_definition": "CUDA peak reserved allocator MiB; CPU/MPS process RSS MiB, not VRAM",
                "source_files": sources(), "model_files": {c["model"]: sha(c["model"]) for c in configs}, "data_files": data_files,
                "runtime_versions": runtime(), "common_setup_seconds": clock.elapsed,
                "weight_scenarios": SCENARIOS,
                "measurement_environment": "contended_smoke" if args.allow_contention else "uncontended_preflight",
                "jobs_seen_at_preparation": jobs, "test_policy": "No test metrics for search, checkpoint or weight selection",
                "training_policy": "fixed horizon, no early stopping, no OOM recovery, warmup0 for exact accumulation",
                "development_history": "Heuristic developed using historical ViECAP/CIFAR/Fashion; detection is a new test domain, no universal superiority established"}
    freeze_sources(folder, protocol["source_files"])
    protocol["protocol_sha256"] = digest(protocol)
    write_json(folder / "protocol.json", protocol)
    write_json(folder / "status.json", {"phase": "prepared"})
    print(f"Locked protocol: {folder / 'protocol.json'} | {len(configs)} configs x {len(args.seeds)} seeds x {args.full_epochs} epochs", flush=True)
    if args.prepare_only:
        print("Prepared only: no training or optimality claim. Start later with research-start --run PATH.", flush=True)
        return 0
    with exclusive_run(folder):
        return execute(folder, protocol, manifest, clock)


def evaluate(folder, protocol, manifest, index, epochs, seconds, seed, location, full):
    require_idle(protocol["measurement_environment"] == "contended_smoke")
    verify_live(protocol, manifest, folder=folder)
    sampled = json.loads((folder / "subset.json").read_text(encoding="utf-8"))
    if digest(sampled) != protocol["subset_sha256"]:
        raise ValueError("Proxy subset changed")
    data = manifest if full else sampled
    runner = DetectionRunner(location, folder / ("dataset.json" if full else "subset.json"),
                             folder / ("full_data" if full else "subset_data") / "data.yaml", protocol["hardware"],
                             protocol["full_epochs"], seed, data["counts"], manifest["counts"])
    print(f"{location.name} c{index:02d} seed{seed} epochs{epochs} full={full}", flush=True)
    result = runner.run(protocol["configs"][index], epochs, min(seconds, protocol["trial_timeout"]), "train" if full else "search")
    print(f"  {result['status']} | Q={result.get('best_quality')} | time={result['worker_wall_seconds']:.2f}s | memory={result.get('memory_mb')}", flush=True)
    return result


def execute(folder, protocol, manifest, deadline):
    write_json(folder / "status.json", {"phase": "search"})
    decisions = []
    methods = [m for m in protocol["methods"] if m != "default"]
    for repeat, seed in enumerate(protocol["training_seeds"]):
        if "default" in protocol["methods"]:
            decisions.append(search("default", protocol["configs"], None, Objective(**protocol["objective"]), protocol["search_budget_seconds"], protocol["rungs"], protocol["full_epochs"], seed))
        offset = repeat % len(methods) if methods else 0
        for method in methods[offset:] + methods[:offset]:
            if deadline.remaining < 1:
                raise ValueError("Search interrupted before decisions lock; preserved partials cannot be replayed with original budget")
            def evaluator(index, epochs, remaining, full):
                return evaluate(folder, protocol, manifest, index, epochs, min(remaining, deadline.remaining), seed,
                                folder / "search" / method / f"seed_{seed}", full)
            order_seed = protocol["split_seed"] if protocol["mode"] == "online_verification" else seed
            decision = search(method, protocol["configs"], evaluator, Objective(**protocol["objective"]),
                              min(protocol["search_budget_seconds"], deadline.remaining), protocol["rungs"], protocol["full_epochs"], order_seed)
            decision.update(seed=seed, ordering_seed=order_seed)
            decisions.append(decision)
            write_json(folder / "decisions.partial.json", decisions)
    write_json(folder / "decisions.locked.json", decisions)
    results = {f"c{i:02d}": [] for i in range(len(protocol["configs"]))}
    if protocol["mode"] == "online_verification":
        for d in decisions:
            for row in d["history"]:
                if row["full_data"]:
                    r = dict(row["result"])
                    r.update(training_seed=d["seed"], full_completed=row["complete"] and row["epochs"] == protocol["full_epochs"])
                    results[row["config_id"]].append(r)
    else:
        results = full_reference(folder, protocol, manifest, deadline, results)
    write_json(folder / "reference.partial.json", results)
    verify_live(protocol, manifest, folder=folder)
    return publish(folder, protocol, decisions, results, deadline.elapsed)


def full_reference(folder, protocol, manifest, deadline, results):
    if not (folder / "decisions.locked.json").exists():
        raise ValueError("Reference cannot run before online decisions are locked")
    write_json(folder / "status.json", {"phase": "full_reference"})
    jobs = [(i, s) for i in range(len(protocol["configs"])) for s in protocol["training_seeds"]]
    random.Random(protocol["split_seed"] + 1000).shuffle(jobs)
    for index, seed in jobs:
        key = f"c{index:02d}"
        if any(r["training_seed"] == seed and r.get("full_completed") for r in results[key]):
            continue
        if deadline.remaining < 1:
            break
        result = evaluate(folder, protocol, manifest, index, protocol["full_epochs"], deadline.remaining, seed,
                          folder / "reference" / key / f"seed_{seed}", True)
        result.update(training_seed=seed, full_completed=training_finished(result, protocol["full_epochs"]) and result.get("epochs_completed") == protocol["full_epochs"])
        old = next((r for r in results[key] if r["training_seed"] == seed), None)
        if old:
            result["previous_attempts"] = old.get("previous_attempts", []) + [{k: v for k, v in old.items() if k != "previous_attempts"}]
        results[key] = [r for r in results[key] if r["training_seed"] != seed] + [result]
        write_json(folder / "reference.partial.json", results)
    return results


def resume(args, start=False):
    folder = Path(args.run).resolve()
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    verify_live(protocol, manifest, folder=folder)
    require_idle(protocol["measurement_environment"] == "contended_smoke")
    hw = runtime_info(protocol["hardware"]["device"])
    hw["host_label"] = platform.node()
    changes = hardware_changes(protocol["hardware"], hw)
    if changes:
        raise ValueError(f"Resume hardware/runtime identity differs/unverified: {changes}")
    if not math.isfinite(args.timeout_hours) or args.timeout_hours <= 0:
        raise ValueError("Additional deadline must be positive")
    with exclusive_run(folder):
        deadline = Deadline(args.timeout_hours * 3600)
        if start:
            if json.loads((folder / "status.json").read_text(encoding="utf-8"))["phase"] != "prepared":
                raise ValueError("research-start only accepts a never-started prepared protocol")
            return execute(folder, protocol, manifest, deadline)
        if not (folder / "decisions.locked.json").exists():
            raise ValueError("Search decisions not locked; cannot resume by replaying search")
        decisions = json.loads((folder / "decisions.locked.json").read_text(encoding="utf-8"))
        results = json.loads((folder / "reference.partial.json").read_text(encoding="utf-8"))
        prior = json.loads((folder / "study.json").read_text(encoding="utf-8")).get("elapsed_seconds", 0) if (folder / "study.json").exists() else None
        results = full_reference(folder, protocol, manifest, deadline, results)
        write_json(folder / "resume.json", {"additional_active_seconds": deadline.elapsed, "prior_active_seconds": prior,
                                           "search_replayed": False, "downtime_included": False, "additional_budget_seconds": args.timeout_hours * 3600})
        return publish(folder, protocol, decisions, results, prior + deadline.elapsed if prior is not None else None, prior)
