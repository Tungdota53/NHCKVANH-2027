"""Explicit held-out test evaluation, isolated from configuration search."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys

from yolo_tuner.pipeline import write_json, new_session
from yolo_tuner.runner import stop_process_tree
from .integrity import sha, verify_live


def evaluate(args):
    from .cli import read_run
    from .study import require_idle, exclusive_run
    require_idle()
    folder, report, _ = read_run(args.run)
    if not report["empirical_best"]:
        raise ValueError("No clean full-confirmed selection; test cannot rescue an unresolved selection")
    manifest = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    if not manifest["splits"]["test"]:
        raise ValueError("No independent test split")
    verify_live(report["protocol"], manifest, folder=folder)
    seed = args.seed if args.seed is not None else report["protocol"]["training_seeds"][0]
    if seed not in report["protocol"]["training_seeds"]:
        raise ValueError("Test seed must be one of the predeclared training seeds; no best-seed cherry picking")
    chosen = next(r for r in report["reference"] if r["config_id"] == report["empirical_best"])
    result = next(r for r in chosen["runs"] if r["training_seed"] == seed and r.get("full_completed"))
    checkpoint = Path(result["best_checkpoint"])
    if sha(checkpoint) != result["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed since full-training")
    if not math.isfinite(args.timeout_minutes) or args.timeout_minutes <= 0:
        raise ValueError("Evaluation timeout must be positive")
    with exclusive_run(folder):
        history_file = folder / "test_evaluation_history.json"
        history = json.loads(history_file.read_text(encoding="utf-8")) if history_file.exists() else []
        if any(r["seed"] == seed for r in history) and not args.allow_retest:
            raise ValueError("This seed's test has already been exposed; use --allow-retest only for explicitly disclosed engineering reruns")
        output = new_session(folder / "evaluation")
        request = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
                   "data": str(folder / "full_data" / "data.yaml"), "device": report["protocol"]["hardware"]["device"],
                   "imgsz": result["config"]["imgsz"], "batch": result["config"]["batch"], "seed": seed,
                   "project": str(output), "split": "test", "selection": report["empirical_best"],
                   "protocol_sha256": report["protocol"]["protocol_sha256"]}
        write_json(output / "request.json", request)
        history.append({"seed": seed, "selection": report["empirical_best"], "output": str(output), "status": "started"})
        write_json(history_file, history)  # record exposure before work, including failed attempts
        with (output / "worker.log").open("w", encoding="utf-8") as log:
            worker = subprocess.Popen([sys.executable, "-m", "research_pipeline.evaluation", str(output / "request.json")], stdout=log, stderr=subprocess.STDOUT)
            try:
                worker.wait(timeout=args.timeout_minutes * 60)
            except subprocess.TimeoutExpired:
                stop_process_tree(worker)
                raise ValueError("Test evaluation timeout; exposure remains recorded")
            except BaseException:
                stop_process_tree(worker)
                raise
        if worker.returncode != 0:
            raise ValueError(f"Evaluation worker failed; inspect {output / 'worker.log'}")
        measured = json.loads((output / "result.json").read_text(encoding="utf-8"))
        history[-1]["status"] = "evaluated"
        write_json(history_file, history)
        (output / "EVALUATION.md").write_text(f"# Held-out test detection\n\nConfig {request['selection']}, seed{seed}.\nTest mAP50-95={measured['map50_95']:.6f}; mAP50={measured['map50']:.6f}.\nNo selection/weights changed. This test is now exposed.\n", encoding="utf-8")
        print(f"Test evaluated: {output / 'EVALUATION.md'}")
    return 0


def worker(path):
    request = json.loads(Path(path).read_text(encoding="utf-8"))
    os.environ["YOLO_CONFIG_DIR"] = str(Path(request["project"]) / "library_config")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    if sha(request["checkpoint"]) != request["checkpoint_sha256"] or request["split"] != "test":
        raise ValueError("Evaluation checkpoint/split mismatch")
    from ultralytics import YOLO, settings
    settings.update({"sync": False})
    measured = YOLO(request["checkpoint"]).val(data=request["data"], split="test", imgsz=request["imgsz"],
                  batch=request["batch"], device=request["device"], workers=0, plots=False,
                  project=request["project"], name="validation", verbose=False)
    result = {"status": "ok", "map50_95": float(measured.box.map), "map50": float(measured.box.map50), "request": request}
    if any(not math.isfinite(result[k]) or not 0 <= result[k] <= 1 for k in ("map50_95", "map50")):
        raise ValueError("Test evaluation did not produce finite metrics")
    write_json(Path(request["project"]) / "result.json", result)


if __name__ == "__main__":
    worker(sys.argv[1])
