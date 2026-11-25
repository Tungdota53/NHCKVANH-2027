"""Explicit real deployment and post-lock test; scope changes are not hidden."""

import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from yolo_tuner.pipeline import new_session, write_json
from yolo_tuner.runner import stop_process_tree
from research_pipeline.study import exclusive_run
from research_pipeline.integrity import sha, hardware_changes
from .schema import decision, detection_metric_for, positive, complete
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
    det_metric = detection_metric_for(p)
    prep = SimpleNamespace(command="train-v2", data=args.data or p["dataset_source"], task=p["task"],
                           device=hw_device, output=str(folder / "execution"), grid=str(grid), space="compact", models=None,
                           precision=config["precision"], full_epochs=epochs, probe_epochs=1, seeds=[selection["seed"]],
                           split_seed=p["split_seed"], search_seed=p["search_seed"], budget_seconds=args.budget_seconds,
                           trial_timeout=args.budget_seconds, probe_fraction=0.0, cost_margin=1.25, cleanup_seconds=5,
                           train_subset=64, val_subset=32, weights=p["objective"]["weights"], time_scale=p["objective"]["time_scale"],
                           memory_scale=p["objective"]["memory_scale"], max_memory=p["objective"]["max_memory"],
                           min_quality=p["objective"]["min_quality"], max_time=p["objective"]["max_time"], predictor=None,
                           method="default", mode="deployment", reference_boundary=None,
                           detection_metric=det_metric or "map50",
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


def _summarize_multi_seed_test(per_seed, task, val_mean, val_sd, quality_def):
    keys = ("map50", "map75", "map50_95", "precision", "recall", "f1", "quality", "accuracy", "bleu4", "chrf")
    ordered_seeds = sorted(per_seed, key=int)
    summary = {}
    for key in keys:
        vals = [float(per_seed[s][key]) for s in ordered_seeds if isinstance(per_seed[s].get(key), (int, float)) and math.isfinite(per_seed[s][key])]
        if vals:
            summary[key] = {"n": len(vals), "mean": statistics.mean(vals),
                            "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                            "min": min(vals), "max": max(vals)}
    per_class_summary = []
    if task == "detection":
        by_cls = {}
        for s in ordered_seeds:
            for item in per_seed[s].get("per_class") or []:
                cid = int(item["class_id"])
                slot = by_cls.setdefault(cid, {"class_id": cid, "class_name": item["class_name"], "ap50": [], "ap50_95": []})
                slot["ap50"].append(float(item["ap50"]))
                slot["ap50_95"].append(float(item["ap50_95"]))
        for cid in sorted(by_cls):
            slot = by_cls[cid]
            per_class_summary.append({
                "class_id": cid,
                "class_name": slot["class_name"],
                "ap50_mean": statistics.mean(slot["ap50"]),
                "ap50_sd": statistics.stdev(slot["ap50"]) if len(slot["ap50"]) > 1 else 0.0,
                "ap50_95_mean": statistics.mean(slot["ap50_95"]),
                "ap50_95_sd": statistics.stdev(slot["ap50_95"]) if len(slot["ap50_95"]) > 1 else 0.0,
            })
    test_q = summary.get("quality", {}).get("mean")
    test_q_sd = summary.get("quality", {}).get("sd")
    val_vs_test = {
        "quality_definition": quality_def,
        "validation_mean": val_mean,
        "validation_sd": val_sd,
        "test_mean": test_q,
        "test_sd": test_q_sd,
        "generalization_gap_test_minus_val": (test_q - val_mean) if test_q is not None and val_mean is not None else None,
    }
    return summary, per_class_summary, val_vs_test


def _summarize_deep_detection_test(per_seed):
    ordered_seeds = sorted(per_seed, key=int)
    lat_seeds = [
        per_seed[s]["latency"] for s in ordered_seeds
        if isinstance(per_seed[s].get("latency"), dict) and per_seed[s]["latency"].get("available")
    ]
    if lat_seeds:
        lat_summary = {
            "available": True,
            "n_seeds": len(lat_seeds),
            "batch_size": lat_seeds[0].get("batch_size"),
            "precision": lat_seeds[0].get("precision"),
            "warmup_batches": lat_seeds[0].get("warmup_batches"),
            "device": lat_seeds[0].get("device"),
            "rect": False,
        }
        for key in ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms_per_image", "fps"):
            vals = [float(item[key]) for item in lat_seeds if isinstance(item.get(key), (int, float)) and math.isfinite(item[key])]
            if vals:
                lat_summary[key] = {
                    "n": len(vals),
                    "mean": statistics.mean(vals),
                    "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    "min": min(vals),
                    "max": max(vals),
                }
        if "total_ms_per_image" in lat_summary:
            lat_summary["mean_total_ms_per_image"] = lat_summary["total_ms_per_image"]["mean"]
        if "fps" in lat_summary:
            lat_summary["mean_fps"] = lat_summary["fps"]["mean"]
    else:
        lat_summary = {"available": False, "rect": False}

    pr_seeds = [
        per_seed[s]["pr_curve"] for s in ordered_seeds
        if isinstance(per_seed[s].get("pr_curve"), dict) and per_seed[s]["pr_curve"].get("available")
    ]
    if pr_seeds:
        grid = pr_seeds[0].get("recall_grid") or []
        matching = [p for p in pr_seeds if len(p.get("recall_grid") or []) == len(grid) and len(p.get("mean_precision") or []) == len(grid)]
        if matching and grid:
            mean_prec = np.mean(np.asarray([p["mean_precision"] for p in matching], dtype=float), axis=0).tolist()
            by_cls = {}
            for p in matching:
                for item in p.get("per_class") or []:
                    if len(item.get("precision") or []) == len(grid):
                        slot = by_cls.setdefault((int(item["class_id"]), str(item["class_name"])), [])
                        slot.append(item["precision"])
            per_cls_pr = [
                {"class_id": cid, "class_name": cname, "precision": np.mean(np.asarray(arrs, dtype=float), axis=0).tolist()}
                for (cid, cname), arrs in sorted(by_cls.items())
            ]
            pr_summary = {
                "available": True,
                "n_seeds": len(matching),
                "recall_grid": grid,
                "mean_precision": mean_prec,
                "per_class": per_cls_pr,
            }
        else:
            pr_summary = {"available": False, "recall_grid": [], "mean_precision": [], "per_class": []}
    else:
        pr_summary = {"available": False, "recall_grid": [], "mean_precision": [], "per_class": []}

    f1_seeds = [
        per_seed[s]["f1_curve"] for s in ordered_seeds
        if isinstance(per_seed[s].get("f1_curve"), dict) and per_seed[s]["f1_curve"].get("available")
    ]
    if f1_seeds:
        grid = f1_seeds[0].get("confidence_grid") or []
        matching = [f for f in f1_seeds if len(f.get("confidence_grid") or []) == len(grid) and len(f.get("mean_f1") or []) == len(grid)]
        if matching and grid:
            mean_f1 = np.mean(np.asarray([f["mean_f1"] for f in matching], dtype=float), axis=0)
            best_idx = int(np.argmax(mean_f1)) if mean_f1.size else 0
            by_cls = {}
            for f in matching:
                for item in f.get("per_class") or []:
                    if len(item.get("f1") or []) == len(grid):
                        slot = by_cls.setdefault((int(item["class_id"]), str(item["class_name"])), [])
                        slot.append(item["f1"])
            per_cls_f1 = [
                {"class_id": cid, "class_name": cname, "f1": np.mean(np.asarray(arrs, dtype=float), axis=0).tolist()}
                for (cid, cname), arrs in sorted(by_cls.items())
            ]
            f1_summary = {
                "available": True,
                "n_seeds": len(matching),
                "confidence_grid": grid,
                "mean_f1": mean_f1.tolist(),
                "best_confidence": float(grid[best_idx]),
                "max_mean_f1": float(mean_f1[best_idx]),
                "per_class": per_cls_f1,
            }
        else:
            f1_summary = {"available": False, "confidence_grid": [], "mean_f1": [], "best_confidence": None, "max_mean_f1": None, "per_class": []}
    else:
        f1_summary = {"available": False, "confidence_grid": [], "mean_f1": [], "best_confidence": None, "max_mean_f1": None, "per_class": []}

    cm_seeds = [
        per_seed[s]["confusion_matrix"] for s in ordered_seeds
        if isinstance(per_seed[s].get("confusion_matrix"), dict) and per_seed[s]["confusion_matrix"].get("available")
    ]
    if cm_seeds:
        base_shape = np.asarray(cm_seeds[0]["raw"], dtype=float).shape
        matching_cm = [np.asarray(c["raw"], dtype=float) for c in cm_seeds if np.asarray(c["raw"], dtype=float).shape == base_shape]
        sum_raw = np.sum(matching_cm, axis=0)
        mean_raw = sum_raw / len(matching_cm)
        col_sums = sum_raw.sum(axis=0, keepdims=True)
        norm_cm = np.divide(sum_raw, col_sums, out=np.zeros_like(sum_raw, dtype=float), where=col_sums > 0)
        cm_summary = {
            "available": True,
            "n_seeds": len(matching_cm),
            "labels": cm_seeds[0].get("labels", []),
            "nc": cm_seeds[0].get("nc", max(0, base_shape[0] - 1)),
            "axes": "rows=predicted, cols=true",
            "normalization": "column_true_sum",
            "sum_raw": sum_raw.tolist(),
            "mean_raw": mean_raw.tolist(),
            "normalized": norm_cm.tolist(),
        }
    else:
        cm_summary = {"available": False, "labels": [], "nc": 0, "axes": "rows=predicted, cols=true", "normalization": "column_true_sum", "sum_raw": [], "mean_raw": [], "normalized": []}

    return lat_summary, pr_summary, f1_summary, cm_summary


def _evaluation_markdown(measured, p):
    def fmt(v):
        return "—" if v is None else f"{v:.6f}"
    lines = ["# Đánh giá tập Test độc lập sau khi khóa lựa chọn", "",
             f"- Cấu hình đã khóa: `{measured['selection']}` (không thay đổi lựa chọn hay trọng số từ tập test)",
             f"- Tác vụ: `{p['task']}`; Metric chính: `{p['quality_definition']}`",
             f"- Seeds đã đánh giá trên tập test: `{measured['evaluated_seeds']}` (`seed_scope={measured['seed_scope']}`)", ""]
    summary = measured.get("multi_seed_summary") or {}
    if summary:
        lines += ["## Tổng hợp Test trên toàn bộ Seeds đã đánh giá", "",
                  "| Metric | Mean | SD | Min | Max | n |", "|---|---:|---:|---:|---:|---:|"]
        for k, s in summary.items():
            lines.append(f"| `{k}` | {fmt(s['mean'])} | {fmt(s['sd'])} | {fmt(s['min'])} | {fmt(s['max'])} | {s['n']} |")
    gap = measured.get("validation_vs_test") or {}
    if gap.get("test_mean") is not None:
        lines += ["", "## Đối chiếu Validation vs Held-out Test (Generalization Gap)", "",
                  f"- Validation mean ± SD: **{fmt(gap.get('validation_mean'))} ± {fmt(gap.get('validation_sd'))}**",
                  f"- Held-out Test mean ± SD: **{fmt(gap.get('test_mean'))} ± {fmt(gap.get('test_sd'))}**",
                  f"- Độ lệch tổng quát hóa (`Test - Val`): **{fmt(gap.get('generalization_gap_test_minus_val'))}**"]
    if measured.get("per_class_summary"):
        lines += ["", "## Chi tiết theo từng lớp (Per-Class AP trên tập Test)", "",
                  "| Class ID | Tên lớp | Test AP@50 mean±SD | Test AP@50:95 mean±SD |", "|---:|---|---:|---:|"]
        for c in measured["per_class_summary"]:
            lines.append(f"| {c['class_id']} | {c['class_name']} | {fmt(c['ap50_mean'])}±{fmt(c['ap50_sd'])} | {fmt(c['ap50_95_mean'])}±{fmt(c['ap50_95_sd'])} |")
    lat = measured.get("latency") or {}
    if lat.get("available"):
        lines += ["", "## Tốc độ suy luận (Inference Latency, `rect=False`)", "",
                  f"- Batch size: `{lat.get('batch_size')}` | Precision: `{lat.get('precision')}` | Warm-up batches: `{lat.get('warmup_batches')}`",
                  "| Thành phần | Mean | SD | Min | Max |", "|---|---:|---:|---:|---:|"]
        for key in ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms_per_image", "fps"):
            if isinstance(lat.get(key), dict):
                s = lat[key]
                lines.append(f"| `{key}` | {fmt(s['mean'])} | {fmt(s['sd'])} | {fmt(s['min'])} | {fmt(s['max'])} |")
    lines += ["", "## Kết quả từng Seed", ""]
    per_seed = measured.get("per_seed_results") or {}
    for s in sorted(per_seed, key=int):
        r = per_seed[s]
        metrics_str = ", ".join(f"{k}={r[k]:.6f}" for k in ("map50", "map75", "map50_95", "precision", "recall", "f1", "accuracy", "bleu4", "chrf", "quality") if isinstance(r.get(k), (int, float)))
        lines.append(f"- Seed `{s}`: {metrics_str}")
    lines += ["", "## Dữ liệu JSON gốc", "", "```json", json.dumps(measured, ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)


def evaluate(args):
    positive(args.timeout_seconds, "timeout")
    seed_scope = getattr(args, "seed_scope", "all")
    if seed_scope not in {"all", "first"}:
        raise ValueError("seed_scope must be 'all' or 'first'")
    inference_precision = str(getattr(args, "inference_precision", "fp32")).lower()
    if inference_precision not in {"fp32", "fp16"}:
        raise ValueError("inference_precision must be 'fp32' or 'fp16'")
    warmup_batches = getattr(args, "warmup_batches", 2)
    if type(warmup_batches) is not int or warmup_batches < 0:
        raise ValueError("warmup_batches must be a non-negative integer")
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
    first_seed = p["training_seeds"][0]
    seeds_to_run = list(p["training_seeds"]) if seed_scope == "all" else [first_seed]
    row = rows[selected["selected_index"]]
    det_metric = detection_metric_for(p)
    seed_runs = {}
    for s in seeds_to_run:
        r = next((item for item in reversed(row["runs"]) if item["seed"] == s and complete(item, p["full_epochs"])), None)
        if r is None:
            raise ValueError(f"Missing completed full run for required seed {s}")
        seed_runs[s] = r
    with exclusive_run(folder):
        idle()
        history_path = folder / "test_history.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
        if history and not args.allow_retest:
            raise ValueError("Test already exposed for this study; --allow-retest must be explicit")
        output = new_session(folder / "evaluation")
        history.append({"seed": first_seed, "evaluated_seeds": seeds_to_run, "seed_scope": seed_scope,
                        "selection": selected["selected_config_id"], "output": str(output), "status": "started",
                        "global_test_blindness": "not_certified; previous engineering exposure may exist"})
        write_json(history_path, history)
        started = time.monotonic()
        per_seed = {}
        for s in seeds_to_run:
            r = seed_runs[s]
            seed_dir = output if len(seeds_to_run) == 1 else output / f"seed_{s}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            request = {"task": p["task"], "checkpoint": r["best_checkpoint"], "checkpoint_sha256": r["checkpoint_sha256"],
                       "manifest": str(folder / "dataset.json"), "data": str(folder / "full_data" / "data.yaml"),
                       "device": p["hardware"]["device"], "imgsz": row["config"]["imgsz"], "batch": row["config"]["batch"],
                       "seed": s, "split": "test", "selection": selected["selected_config_id"], "project": str(seed_dir),
                       "protocol_sha256": p["protocol_sha256"], "inference_precision": inference_precision,
                       "warmup_batches": warmup_batches}
            if det_metric is not None:
                request["detection_metric"] = det_metric
            write_json(seed_dir / "request.json", request)
            remaining = args.timeout_seconds - (time.monotonic() - started)
            if remaining <= 0.05:
                raise ValueError("Evaluation deadline exhausted before all declared seeds finished; exposure preserved")
            with (seed_dir / "worker.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen([sys.executable, "-m", "research_pipeline.v2.deployment", str(seed_dir / "request.json")], stdout=log, stderr=subprocess.STDOUT)
                try:
                    process.wait(timeout=remaining)
                except BaseException:
                    stop_process_tree(process)
                    raise
            if process.returncode or not (seed_dir / "result.json").exists():
                raise ValueError(f"Evaluation failed on seed {s}; exposure preserved; inspect {seed_dir / 'worker.log'}")
            per_seed[str(s)] = json.loads((seed_dir / "result.json").read_text(encoding="utf-8"))
        summary, per_class_summary, val_vs_test = _summarize_multi_seed_test(
            per_seed, p["task"], row["quality_mean"], row["quality_sd"], p["quality_definition"]
        )
        lat_summary, pr_summary, f1_summary, cm_summary = _summarize_deep_detection_test(per_seed)
        measured = dict(per_seed[str(first_seed)])
        measured.update(
            selection=selected["selected_config_id"],
            seed_scope=seed_scope,
            evaluated_seeds=seeds_to_run,
            per_seed_results=per_seed,
            multi_seed_summary=summary,
            per_class_summary=per_class_summary,
            validation_vs_test=val_vs_test,
            test_used_for_selection=False,
            first_declared_seed=first_seed,
            global_test_blindness=history[-1]["global_test_blindness"],
        )
        if p["task"] == "detection":
            measured.update(
                latency=lat_summary,
                pr_curve=pr_summary,
                f1_curve=f1_summary,
                confusion_matrix=cm_summary,
            )
        history[-1]["status"] = "evaluated"
        write_json(history_path, history)
        write_json(output / "result.json", measured)
        write_json(output / "evaluation.json", measured)
        test_summary = {
            "schema_version": 1,
            "selection": selected["selected_config_id"],
            "task": p["task"],
            "quality_definition": p["quality_definition"],
            "seed_scope": seed_scope,
            "evaluated_seeds": seeds_to_run,
            "output_directory": str(output),
            "test_used_for_selection": False,
            "multi_seed_summary": summary,
            "per_class_summary": per_class_summary,
            "validation_vs_test": val_vs_test,
            "latency": lat_summary,
            "pr_curve": pr_summary,
            "f1_curve": f1_summary,
            "confusion_matrix": cm_summary,
        }
        write_json(output / "test_summary.json", test_summary)
        write_json(folder / "evaluation" / "test_summary.json", test_summary)
        (output / "EVALUATION.md").write_text(_evaluation_markdown(measured, p), encoding="utf-8")
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
        measured = {"accuracy": correct / len(records), "quality": correct / len(records), "images": len(records)}
    if any(not math.isfinite(v) for v in measured.values()):
        raise ValueError("Non-finite test metric")
    write_json(Path(request["project"]) / "result.json", {"status": "ok", "task": request["task"], "inference_precision": "fp32", **measured})


if __name__ == "__main__":
    evaluation_worker(sys.argv[1])
