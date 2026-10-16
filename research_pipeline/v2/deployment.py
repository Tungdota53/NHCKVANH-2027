"""Explicit real deployment and post-lock test; scope changes are not hidden."""

import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from yolo_tuner.pipeline import new_session, write_json
from yolo_tuner.runner import stop_process_tree
from research_pipeline.study import exclusive_run
from research_pipeline.integrity import sha, hardware_changes
from .schema import decision, positive, complete
from .workflow import audit_run, verify_live, idle, prepare
from .ledger import Ledger
from .runner import run_worker
from .selection import select, presets


def train(args):
    positive(args.budget_seconds, "budget")
    if args.epochs is not None:
        positive(args.epochs, "epochs")
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    old, p, rows, _ = audit_run(selection["run"])
    selected = select(rows, p)
    if selected["selected_index"] is None or selection["index"] != selected["selected_index"] or selection["config"] != rows[selection["index"]]["config"] or selection["protocol_sha256"] != p["protocol_sha256"]:
        raise ValueError("Selection differs from audited full outcomes")
    preset = getattr(args, "preset", "Balanced")
    alternative = presets(rows, p)[preset]
    if alternative is None:
        raise ValueError("Preset has no full-confirmed feasible configuration")
    epochs = p["full_epochs"] if args.epochs is None else args.epochs
    positive(epochs, "epochs")
    if type(epochs) is not int:
        raise ValueError("epochs must be an integer")
    hw_device = args.device
    from yolo_tuner.pipeline import runtime_info
    hw = runtime_info(hw_device)
    config = dict(alternative["config"])
    if hw["device"] in {"cpu", "mps"}:
        config["precision"] = "fp32"
    folder = new_session(args.output)
    grid = folder / "deployment_grid.json"
    write_json(grid, [config])
    source_loader = p.get("execution_policy", {}).get("data_loader", {})
    prep = SimpleNamespace(command="train-v2", data=args.data or p["dataset_source"], task=p["task"],
                           device=hw_device, output=str(folder / "execution"), grid=str(grid), space="compact", models=None,
                           precision=config["precision"], full_epochs=epochs, probe_epochs=1, seeds=[selection["seed"]],
                           split_seed=p["split_seed"], search_seed=p["search_seed"], budget_seconds=args.budget_seconds,
                           trial_timeout=args.budget_seconds, probe_fraction=0.0, cost_margin=1.25, cleanup_seconds=5,
                           train_subset=64, val_subset=32, weights=p["objective"]["weights"], time_scale=p["objective"]["time_scale"],
                           memory_scale=p["objective"]["memory_scale"], max_memory=p["objective"]["max_memory"],
                           min_quality=p["objective"]["min_quality"], max_time=p["objective"]["max_time"], predictor=None,
                           method="default", mode="deployment", reference_boundary=None,
                           workers=source_loader.get("workers", "auto"),
                           image_cache_mb=source_loader.get("cache_mb", 64.0),
                           pin_memory=source_loader.get("pin_memory", "auto"))
    run, new, ledger = prepare(prep)
    changes = hardware_changes(p["hardware"], new["hardware"])
    if preset != "Balanced":
        changes.append(f"explicit deployment preset:{preset}; locked research choice unchanged")
    if new["dataset_content_id"] != p["dataset_content_id"]:
        changes.append("dataset")
    if epochs != p["full_epochs"]:
        changes.append("horizon")
    if config["precision"] != alternative["config"]["precision"]:
        changes.append("precision")
    if new["source_files"] != p["source_files"]:
        changes.append("source")
    if new.get("execution_policy") != p.get("execution_policy"):
        changes.append("execution_policy")
    with exclusive_run(run):
        idle()
        manifest, _ = verify_live(run, new)
        result = run_worker(run / "trials" / "attempt_00001", new, manifest, True, 0, selection["seed"], epochs,
                            max(0.01, args.budget_seconds - ledger.elapsed - 5), ledger)
        write_json(run / "state.json", {"results": {"0": [result]}, "probes": [], "actions": [], "ordering": [0]})
        verify_live(run, new)
        audit_run(run, check_published=False)
        write_json(run / "status.json", {"phase": "complete", "execution_verified": complete(result, epochs),
                                        "optimality_not_reestablished": True})
        report = {"result": result, "execution_verified": complete(result, epochs), "scope_changes": changes,
                  "requested_preset": preset, "deployed_config": config,
                  "source_selection": selection, "status": "execution_verified_optimality_not_reestablished" if complete(result, epochs) else "execution_failed",
                  "new_optimization_performed": False, "run": str(run)}
        write_json(folder / "TRAINING.json", report)
        print(f"Full training: {report['status']} | scope changes={changes} | {folder / 'TRAINING.json'}", flush=True)
    return report


def evaluate(args):
    positive(args.timeout_seconds, "timeout")
    folder, p, rows, _ = audit_run(args.run)
    if p["mode"] == "deployment":
        raise ValueError("Deployment is execution evidence, not a new locked optimization selection")
    if json.loads((folder / "status.json").read_text(encoding="utf-8"))["phase"] != "complete":
        raise ValueError("Test requires a completed locked recommendation, not interrupted search")
    selected = select(rows, p)
    if selected["selected_index"] is None:
        raise ValueError("No all-seed full-confirmed selection; test cannot rescue a proxy")
    manifest, _ = verify_live(folder, p)
    if not manifest["splits"]["test"]:
        raise ValueError("No independent test split")
    seed = p["training_seeds"][0]  # Frozen seed rule, not best-seed cherry picking.
    row = rows[selected["selected_index"]]
    result = next(r for r in reversed(row["runs"]) if r["seed"] == seed and complete(r, p["full_epochs"]))
    with exclusive_run(folder):
        idle()
        history_path = folder / "test_history.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
        if history and not args.allow_retest:
            raise ValueError("Test already exposed for this study; --allow-retest must be explicit")
        output = new_session(folder / "evaluation")
        request = {"task": p["task"], "checkpoint": result["best_checkpoint"], "checkpoint_sha256": result["checkpoint_sha256"],
                   "manifest": str(folder / "dataset.json"), "data": str(folder / "full_data" / "data.yaml"),
                   "device": p["hardware"]["device"], "imgsz": row["config"]["imgsz"], "batch": row["config"]["batch"],
                   "seed": seed, "split": "test", "selection": selected["selected_config_id"], "project": str(output),
                   "protocol_sha256": p["protocol_sha256"]}
        request["inference_precision"] = "fp32"
        write_json(output / "request.json", request)
        history.append({"seed": seed, "selection": request["selection"], "output": str(output), "status": "started",
                        "global_test_blindness": "not_certified; previous engineering exposure may exist"})
        write_json(history_path, history)
        with (output / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "research_pipeline.v2.deployment", str(output / "request.json")], stdout=log, stderr=subprocess.STDOUT)
            try:
                process.wait(timeout=args.timeout_seconds)
            except BaseException:
                stop_process_tree(process)
                raise
        if process.returncode or not (output / "result.json").exists():
            raise ValueError(f"Evaluation failed; exposure preserved; inspect {output / 'worker.log'}")
        measured = json.loads((output / "result.json").read_text(encoding="utf-8"))
        history[-1]["status"] = "evaluated"
        write_json(history_path, history)
        measured.update(test_used_for_selection=False, first_declared_seed=seed, global_test_blindness=history[-1]["global_test_blindness"])
        write_json(output / "evaluation.json", measured)
        (output / "EVALUATION.md").write_text("# Đánh giá test sau lựa chọn\n\n" + json.dumps(measured, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Test: {output / 'EVALUATION.md'}", flush=True)
        return measured


def evaluation_worker(path):
    import math
    import torch
    request = json.loads(Path(path).read_text(encoding="utf-8"))
    if request["split"] != "test" or sha(request["checkpoint"]) != request["checkpoint_sha256"]:
        raise ValueError("Test/checkpoint identity mismatch")
    if request["task"] == "detection":
        from research_pipeline.evaluation import worker
        from unittest.mock import patch
        import ultralytics.data.utils
        with patch("ultralytics.data.utils.check_font", return_value=None):
            return worker(path)
    manifest = json.loads(Path(request["manifest"]).read_text(encoding="utf-8"))
    records = manifest["splits"]["test"]
    if request["task"] == "captioning":
        from yolo_tuner.captioning import CaptionPredictor, score_captions
        from unittest.mock import patch
        from .pooling import deterministic_adaptive_pool2d
        predictor = CaptionPredictor(request["checkpoint"], request["device"])
        captions = []
        with torch.inference_mode(), patch("torch.nn.functional.adaptive_avg_pool2d", deterministic_adaptive_pool2d):
            for offset in range(0, len(records), request["batch"]):
                captions.extend(predictor.predict([r["image"] for r in records[offset:offset + request["batch"]]]))
        metrics = score_captions(captions, [r["captions"] for r in records])
        measured = {"bleu4": metrics["bleu4"], "chrf": metrics["chrf"], "quality": (metrics["bleu4"] + metrics["chrf"]) / 200}
    else:
        import numpy as np
        from PIL import Image
        from yolo_tuner.vision_models import build
        payload = torch.load(request["checkpoint"], map_location="cpu", weights_only=True)
        device = request["device"] if request["device"] in {"cpu", "mps"} else f"cuda:{request['device']}"
        model = build("classification", payload["config"], classes=len(payload["names"]))
        model.load_state_dict(payload["model"])
        model.to(device).eval()
        torch.set_num_threads(4)
        correct = 0
        with torch.inference_mode():
            for offset in range(0, len(records), request["batch"]):
                group = records[offset:offset + request["batch"]]
                images = []
                for r in group:
                    with Image.open(r["image"]) as im:
                        images.append(torch.from_numpy(np.asarray(im.convert("RGB").resize((request["imgsz"], request["imgsz"]))).copy()).permute(2, 0, 1).float() / 255)
                predictions = model(torch.stack(images).to(device)).argmax(1).cpu().tolist()
                correct += sum(pred == r["class_id"] for pred, r in zip(predictions, group))
        measured = {"accuracy": correct / len(records), "images": len(records)}
    if any(not math.isfinite(v) for v in measured.values()):
        raise ValueError("Non-finite test metric")
    write_json(Path(request["project"]) / "result.json", {"status": "ok", "task": request["task"], "inference_precision": "fp32", **measured})


if __name__ == "__main__":
    evaluation_worker(sys.argv[1])
