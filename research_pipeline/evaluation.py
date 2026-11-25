"""Explicit held-out test evaluation, isolated from configuration search."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from yolo_tuner.pipeline import write_json, new_session
from yolo_tuner.runner import stop_process_tree
from .integrity import sha, verify_live


def _finite_float(value, default=0.0):
    try:
        val = float(value)
    except (TypeError, ValueError):
        return float(default)
    return val if math.isfinite(val) else float(default)


def _finite_list(values):
    return [_finite_float(v) for v in values]


def _extract_latency(measured, batch_size, precision, warmup_batches, device):
    speed = getattr(measured, "speed", None)
    if not isinstance(speed, dict) or not any(k in speed for k in ("preprocess", "inference", "postprocess")):
        return {
            "available": False,
            "preprocess_ms": None,
            "inference_ms": None,
            "postprocess_ms": None,
            "total_ms_per_image": None,
            "fps": None,
            "batch_size": int(batch_size),
            "precision": str(precision),
            "warmup_batches": int(warmup_batches),
            "device": str(device),
            "rect": False,
        }
    pre = max(0.0, _finite_float(speed.get("preprocess", 0.0)))
    inf = max(0.0, _finite_float(speed.get("inference", 0.0)))
    post = max(0.0, _finite_float(speed.get("postprocess", 0.0)))
    total = pre + inf + post
    fps = (1000.0 / total) if total > 0.0 else 0.0
    return {
        "available": True,
        "preprocess_ms": pre,
        "inference_ms": inf,
        "postprocess_ms": post,
        "total_ms_per_image": total,
        "fps": fps,
        "batch_size": int(batch_size),
        "precision": str(precision),
        "warmup_batches": int(warmup_batches),
        "device": str(device),
        "rect": False,
    }


def _extract_curves(measured, ap_indices, names):
    curves = getattr(measured, "curves_results", None)
    pr_out = {"available": False, "recall_grid": [], "mean_precision": [], "per_class": []}
    f1_out = {"available": False, "confidence_grid": [], "mean_f1": [], "per_class": [], "best_confidence": None, "max_mean_f1": None}
    if not isinstance(curves, (list, tuple)) or len(curves) < 2:
        return pr_out, f1_out

    def _to_2d(raw):
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1) if arr.size > 0 else np.zeros((0, 0), dtype=float)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    pr_entry = curves[0]
    if isinstance(pr_entry, (list, tuple)) and len(pr_entry) >= 2:
        px = _finite_list(np.asarray(pr_entry[0], dtype=float).ravel())
        py = _to_2d(pr_entry[1])
        if len(ap_indices) == 0 or py.size == 0 or py.shape[0] == 0:
            mean_p = [0.0] * len(px)
            per_cls_p = []
        else:
            if py.shape[0] == len(ap_indices):
                per_cls_p = [
                    {"class_id": int(cid), "class_name": str(names.get(int(cid), int(cid))), "precision": _finite_list(py[i])}
                    for i, cid in enumerate(ap_indices)
                ]
            else:
                per_cls_p = []
            mean_p = _finite_list(py.mean(axis=0))
        pr_out = {
            "available": True,
            "recall_grid": px,
            "mean_precision": mean_p,
            "per_class": per_cls_p,
        }

    f1_entry = curves[1]
    if isinstance(f1_entry, (list, tuple)) and len(f1_entry) >= 2:
        fx = _finite_list(np.asarray(f1_entry[0], dtype=float).ravel())
        fy = _to_2d(f1_entry[1])
        if len(ap_indices) == 0 or fy.size == 0 or fy.shape[0] == 0:
            mean_f1 = [0.0] * len(fx)
            per_cls_f1 = []
            best_conf = 0.0
            max_f1 = 0.0
        else:
            if fy.shape[0] == len(ap_indices):
                per_cls_f1 = [
                    {"class_id": int(cid), "class_name": str(names.get(int(cid), int(cid))), "f1": _finite_list(fy[i])}
                    for i, cid in enumerate(ap_indices)
                ]
            else:
                per_cls_f1 = []
            mean_arr = fy.mean(axis=0)
            mean_f1 = _finite_list(mean_arr)
            best_idx = int(np.argmax(mean_arr)) if mean_arr.size else 0
            best_conf = float(fx[best_idx]) if fx else 0.0
            max_f1 = float(mean_f1[best_idx]) if mean_f1 else 0.0
        f1_out = {
            "available": True,
            "confidence_grid": fx,
            "mean_f1": mean_f1,
            "per_class": per_cls_f1,
            "best_confidence": best_conf,
            "max_mean_f1": max_f1,
        }
    return pr_out, f1_out


def _extract_confusion_matrix(measured, names):
    cm = getattr(measured, "confusion_matrix", None)
    raw_attr = getattr(cm, "matrix", None) if cm is not None else None
    if cm is None or raw_attr is None:
        return {
            "available": False,
            "labels": [],
            "nc": len(names),
            "axes": "rows=predicted, cols=true",
            "normalization": "column_true_sum",
            "raw": [],
            "normalized": [],
        }
    raw_mat = np.nan_to_num(np.asarray(raw_attr, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    nc = int(getattr(cm, "nc", max(0, raw_mat.shape[0] - 1)))
    cm_names = getattr(cm, "names", None) or names or {}
    labels = [str(cm_names.get(i, i)) for i in range(nc)] + ["background"]
    col_sums = raw_mat.sum(axis=0, keepdims=True)
    norm_mat = np.divide(raw_mat, col_sums, out=np.zeros_like(raw_mat, dtype=float), where=col_sums > 0)
    return {
        "available": True,
        "labels": labels,
        "nc": nc,
        "axes": "rows=predicted, cols=true",
        "normalization": "column_true_sum",
        "raw": [_finite_list(row) for row in raw_mat],
        "normalized": [_finite_list(row) for row in norm_mat],
    }


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
                   "detection_metric": report["protocol"].get("primary_metric", "map50"),
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
        (output / "EVALUATION.md").write_text(
            f"# Held-out test detection\n\nConfig {request['selection']}, seed {seed}.\n"
            f"Primary Test mAP50={measured['map50']:.6f}; mAP75={measured.get('map75', 0.0):.6f}; mAP50-95={measured['map50_95']:.6f}; "
            f"Precision={measured.get('precision', 0.0):.6f}; Recall={measured.get('recall', 0.0):.6f}; F1={measured.get('f1', 0.0):.6f}.\n"
            "No selection/weights changed. This test is now exposed.\n",
            encoding="utf-8",
        )
        print(f"Test: {output / 'EVALUATION.md'}")
    return 0


def worker(path):
    request = json.loads(Path(path).read_text(encoding="utf-8"))
    os.environ["YOLO_CONFIG_DIR"] = str(Path(request["project"]) / "library_config")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    if sha(request["checkpoint"]) != request["checkpoint_sha256"] or request["split"] != "test":
        raise ValueError("Evaluation checkpoint/split mismatch")
    inference_precision = str(request.get("inference_precision", "fp32")).lower()
    if inference_precision not in {"fp32", "fp16"}:
        raise ValueError(f"Unsupported inference_precision: {inference_precision}; expected 'fp32' or 'fp16'")
    warmup_batches = request.get("warmup_batches", 2)
    if type(warmup_batches) is not int or warmup_batches < 0:
        raise ValueError("warmup_batches must be a non-negative integer")
    device_str = str(request.get("device", "cpu")).lower()
    use_half = (inference_precision == "fp16") and (device_str not in {"cpu", "mps"})
    actual_precision = "fp16" if use_half else "fp32"
    batch_size = max(1, int(request["batch"]))

    from unittest.mock import patch
    import torch
    from ultralytics import YOLO, settings
    import ultralytics.data.utils
    import ultralytics.utils.checks
    import ultralytics.utils.plotting
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.nn.autobackend import AutoBackend
    from ultralytics.utils.nms import non_max_suppression

    settings.update({"sync": False})

    def batch_warmup(backend_self, imgsz=(1, 3, 640, 640), im=None):
        if warmup_batches <= 0:
            return
        if not getattr(backend_self, "end2end", False):
            import torchvision  # noqa: F401
        target_shape = (batch_size, int(imgsz[1]), int(imgsz[2]), int(imgsz[3]))
        dtype = torch.half if getattr(backend_self, "fp16", False) else torch.float
        dummy = im if (im is not None and tuple(im.shape) == target_shape) else torch.empty(*target_shape, dtype=dtype, device=backend_self.device)
        for _ in range(warmup_batches):
            backend_self.forward(dummy)
            if not getattr(backend_self, "end2end", False):
                warmup_boxes = torch.rand(1, 84, 16, device=backend_self.device)
                warmup_boxes[:, :4] *= dummy.shape[-1]
                non_max_suppression(warmup_boxes)

    orig_update_metrics = DetectionValidator.update_metrics

    def update_metrics_with_confusion_matrix(validator_self, *args, **kwargs):
        val_args = getattr(validator_self, "args", None)
        if val_args is None:
            return orig_update_metrics(validator_self, *args, **kwargs)
        orig_plots = getattr(val_args, "plots", False)
        orig_visualize = getattr(val_args, "visualize", False)
        val_args.plots = True
        val_args.visualize = False
        try:
            return orig_update_metrics(validator_self, *args, **kwargs)
        finally:
            val_args.plots = orig_plots
            val_args.visualize = orig_visualize

    with patch("ultralytics.data.utils.check_font", return_value=None), \
         patch("ultralytics.utils.checks.check_font", return_value=None), \
         patch("ultralytics.utils.plotting.check_font", return_value=None), \
         patch.object(AutoBackend, "warmup", batch_warmup), \
         patch.object(DetectionValidator, "update_metrics", update_metrics_with_confusion_matrix):
        measured = YOLO(request["checkpoint"]).val(
            data=request["data"], split="test", imgsz=request["imgsz"],
            batch=request["batch"], rect=False, quantize=(16 if use_half else None), device=request["device"],
            workers=0, plots=False, project=request["project"], name="validation", exist_ok=True, verbose=False,
        )

    primary = request.get("detection_metric", "map50")
    map50 = float(measured.box.map50)
    map75 = float(measured.box.map75)
    map50_95 = float(measured.box.map)
    precision = float(measured.box.mp)
    recall = float(measured.box.mr)
    precision = precision if math.isfinite(precision) and 0 <= precision <= 1 else 0.0
    recall = recall if math.isfinite(recall) and 0 <= recall <= 1 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    names = getattr(measured, "names", {}) or {}
    ap_indices = list(getattr(measured.box, "ap_class_index", []))
    ap50_list = list(getattr(measured.box, "ap50", []))
    ap_list = list(getattr(measured.box, "ap", []))
    p_list = list(getattr(measured.box, "p", []))
    r_list = list(getattr(measured.box, "r", []))
    per_class = []
    for idx, cls_id in enumerate(ap_indices):
        cid = int(cls_id)
        c_ap50 = float(ap50_list[idx]) if idx < len(ap50_list) and math.isfinite(float(ap50_list[idx])) else 0.0
        c_ap = float(ap_list[idx]) if idx < len(ap_list) and math.isfinite(float(ap_list[idx])) else 0.0
        c_p = float(p_list[idx]) if idx < len(p_list) and math.isfinite(float(p_list[idx])) else 0.0
        c_r = float(r_list[idx]) if idx < len(r_list) and math.isfinite(float(r_list[idx])) else 0.0
        per_class.append({"class_id": cid, "class_name": str(names.get(cid, cid)),
                          "precision": c_p, "recall": c_r, "ap50": c_ap50, "ap50_95": c_ap})

    latency = _extract_latency(measured, batch_size, actual_precision, warmup_batches, request["device"])
    pr_curve, f1_curve = _extract_curves(measured, ap_indices, names)
    confusion_matrix = _extract_confusion_matrix(measured, names)

    result = {
        "status": "ok",
        "primary_metric": primary,
        "rect": False,
        "inference_precision": actual_precision,
        "warmup_batches": warmup_batches,
        "quality": map50 if primary == "map50" else map50_95,
        "map50": map50,
        "map75": map75 if math.isfinite(map75) and 0 <= map75 <= 1 else 0.0,
        "map50_95": map50_95,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class": per_class,
        "latency": latency,
        "pr_curve": pr_curve,
        "f1_curve": f1_curve,
        "confusion_matrix": confusion_matrix,
        "request": request,
    }
    if any(not math.isfinite(result[k]) or not 0 <= result[k] <= 1 for k in ("map50", "map50_95")):
        raise ValueError("Test evaluation did not produce finite metrics")
    write_json(Path(request["project"]) / "result.json", result)


if __name__ == "__main__":
    worker(sys.argv[1])
