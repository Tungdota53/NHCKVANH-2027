"""Subprocess isolation with wall budget, actual config checks and preserved logs."""

import json
from pathlib import Path
import subprocess
import sys
import time

import psutil
from yolo_tuner.runner import stop_process_tree
from yolo_tuner.pipeline import write_json


def actual_matches(config, actual):
    exact = ("batch", "imgsz", "precision", "accumulation", "optimizer")
    if any(actual.get(k) != config[k] for k in exact):
        return False
    return all(abs(actual.get(k, float("inf")) - config[k]) <= max(1e-12, abs(config[k]) * 1e-6)
               for k in ("lr0", "weight_decay"))


def training_evidence_complete(result):
    return result.get("optimizer_state_created") is True and result.get("training_loss_finite") is True


class DetectionRunner:
    def __init__(self, folder, manifest_path, data_yaml, hw, epochs, seed, counts, full_counts):
        self.folder = Path(folder).resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.manifest_path, self.data_yaml = str(manifest_path), str(data_yaml)
        self.hw, self.epochs, self.seed = hw, epochs, seed
        self.counts, self.full_counts = counts, full_counts

    def run(self, config, epochs, seconds, mode="train"):
        index = max((int(p.name.split("_")[-1]) for p in self.folder.glob("trial_*")), default=0) + 1
        folder = self.folder / f"trial_{index:04d}"
        folder.mkdir()
        req = {"config": config, "mode": mode, "manifest": self.manifest_path, "data": self.data_yaml,
               "device": self.hw["device"], "full_epochs": self.epochs, "trial_epochs": epochs,
               "seed": self.seed, "project": str(folder), "resume": None,
               "counts": self.counts, "full_counts": self.full_counts}
        write_json(folder / "request.json", req)
        started = time.monotonic()
        with (folder / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "research_pipeline.worker", str(folder / "request.json"), str(folder / "result.json")], stdout=log, stderr=subprocess.STDOUT)
            created = psutil.Process(process.pid).create_time()
            write_json(folder / "process.json", {"pid": process.pid, "create_time": created, "status": "running"})
            timeout = False
            try:
                process.wait(timeout=max(0.01, seconds))
            except subprocess.TimeoutExpired:
                timeout = True
                stop_process_tree(process)
            except BaseException:
                stop_process_tree(process)
                raise
        path = folder / "result.json"
        raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "error", "config": config, "error": "Worker exited without result"}
        if not path.exists():
            write_json(path, raw)
        result = {**raw, "worker_wall_seconds": time.monotonic() - started, "log": str(folder / "worker.log")}
        if timeout:
            result.update(status="timeout", error="Process wall-clock deadline exceeded")
        elif process.returncode != 0:
            result.update(status="error", error=f"Worker exit {process.returncode}")
        if result["status"] == "ok" and not actual_matches(config, result.get("actual", {})):
            result.update(status="config_mismatch", error="Actual batch/precision/accumulation/optimizer/LR/decay differs from requested configuration")
        if result["status"] == "ok" and not training_evidence_complete(result):
            result.update(status="training_evidence_missing", error="Optimizer state/finite training-loss evidence is missing")
        if result["status"] == "ok":
            setup = max(0, result["worker_wall_seconds"] - result.get("measured_loop_seconds", 0))
            if mode == "train":
                result["estimated_full_seconds"] = result["worker_wall_seconds"]
            else:
                result["estimated_full_seconds"] += setup
            result["time_estimate_basis"] = "actual full worker wall time" if mode == "train" else "split-scaled measured loops plus startup; heuristic, uncalibrated"
        write_json(folder / "parent_result.json", result)
        write_json(folder / "process.json", {"pid": process.pid, "create_time": created, "status": "finished", "exit_code": process.returncode})
        return result
