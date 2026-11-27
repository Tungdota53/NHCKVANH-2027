"""Turnkey Cross-GPU Conformal VRAM Transfer Benchmark (Colab T4 / Kaggle T4/P100 / Local Dry-Run).

Evaluates transferring the pre-fitted RTX 3050 6GB Split Conformal VRAM Card
(configs/conformal_vram_card_rtx3050.json) onto an unseen target GPU via 24 isolated
subprocess probes (8x yolo11n, 8x yolo11s, 8x yolo26n), comparing:
  1) Zero-Shot transfer (source RTX 3050 card without target anchors)
  2) 2-Point Online Shrinkage Calibration (using 2 low/mid safe probes on target GPU)

Usage on Google Colab / Kaggle:
    python scripts/colab_cross_gpu_conformal_transfer.py --device 0

Local / CI dry-run verification without touching GPU:
    python scripts/colab_cross_gpu_conformal_transfer.py --dry-run-measurements runs/nckh_conformal_vram_study/measurements.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_pipeline.v2.conformal_vram import (  # noqa: E402
    _activation_proxy,
    calibrate_online_2point,
    predict_conformal_vram,
    verify_conformal_vram_card,
)
from research_pipeline.v2.model_features import architecture_key, describe_models  # noqa: E402
from yolo_tuner.adapters import export_yolo, subset  # noqa: E402
from yolo_tuner.pipeline import write_json  # noqa: E402
from research_pipeline.v2.data import discover  # noqa: E402


def build_24_cross_gpu_probes() -> list[dict[str, Any]]:
    """Return a stratified 24-config YOLO probe suite across 3 architectures (8 each)."""
    specs = [
        # --- weights/yolo11n.pt (8 configs) ---
        ("weights/yolo11n.pt", 320, 2, "fp16", 5, "AdamW"),
        ("weights/yolo11n.pt", 320, 8, "fp16", 0, "SGD"),
        ("weights/yolo11n.pt", 416, 4, "fp16", 0, "AdamW"),
        ("weights/yolo11n.pt", 480, 8, "fp16", 0, "AdamW"),
        ("weights/yolo11n.pt", 480, 8, "fp32", 0, "SGD"),
        ("weights/yolo11n.pt", 576, 8, "fp16", 5, "AdamW"),
        ("weights/yolo11n.pt", 640, 8, "fp16", 0, "AdamW"),
        ("weights/yolo11n.pt", 640, 8, "fp32", 0, "SGD"),
        # --- weights/yolo11s.pt (8 configs) ---
        ("weights/yolo11s.pt", 320, 2, "fp16", 5, "AdamW"),
        ("weights/yolo11s.pt", 320, 8, "fp16", 0, "SGD"),
        ("weights/yolo11s.pt", 416, 4, "fp16", 3, "AdamW"),
        ("weights/yolo11s.pt", 480, 4, "fp16", 0, "AdamW"),
        ("weights/yolo11s.pt", 480, 8, "fp16", 0, "AdamW"),
        ("weights/yolo11s.pt", 576, 6, "fp16", 3, "AdamW"),
        ("weights/yolo11s.pt", 640, 4, "fp16", 0, "SGD"),
        ("weights/yolo11s.pt", 640, 8, "fp32", 0, "SGD"),
        # --- weights/yolo26n.pt (8 configs) ---
        ("weights/yolo26n.pt", 320, 2, "fp16", 5, "AdamW"),
        ("weights/yolo26n.pt", 320, 8, "fp16", 0, "SGD"),
        ("weights/yolo26n.pt", 416, 4, "fp16", 2, "AdamW"),
        ("weights/yolo26n.pt", 480, 4, "fp16", 0, "SGD"),
        ("weights/yolo26n.pt", 480, 8, "fp16", 0, "AdamW"),
        ("weights/yolo26n.pt", 576, 4, "fp16", 0, "AdamW"),
        ("weights/yolo26n.pt", 640, 4, "fp16", 0, "AdamW"),
        ("weights/yolo26n.pt", 640, 8, "fp32", 0, "AdamW"),
    ]
    return [
        {
            "model": model,
            "imgsz": imgsz,
            "batch": batch,
            "accumulation": 1,
            "precision": prec,
            "optimizer": opt,
            "lr0": 0.001,
            "weight_decay": 0.0005,
            "freeze": freeze,
        }
        for model, imgsz, batch, prec, freeze, opt in specs
    ]


def _ensure_probe_dataset(data_arg: str | None, prep_dir: Path) -> tuple[Path, Path, dict[str, int], dict[str, int], list[str]]:
    """Use --data dataset if present, otherwise generate a minimal valid YOLO dataset for VRAM probing."""
    if data_arg and Path(data_arg).exists():
        whole = discover(str(data_arg), 101, "detection")
        sub = subset(whole, 32, 8)
        sub_dir = prep_dir / "subset_data"
        sub_yaml = export_yolo(sub, sub_dir)
        sub_manifest = prep_dir / "subset.json"
        write_json(sub_manifest, sub)
        return Path(sub_manifest), Path(sub_yaml), sub["counts"], whole["counts"], whole["names"]

    from PIL import Image

    synth_root = prep_dir / "synthetic_yolo_vram_probe"
    for split, count in (("train", 32), ("val", 8)):
        img_dir = synth_root / split / "images"
        lbl_dir = synth_root / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img_p = img_dir / f"img_{i:03d}.jpg"
            lbl_p = lbl_dir / f"img_{i:03d}.txt"
            if not img_p.exists():
                Image.new("RGB", (640, 640), color=(40 + i % 150, 90, 120)).save(img_p)
            if not lbl_p.exists():
                lbl_p.write_text("0 0.5 0.5 0.35 0.35\n", encoding="utf-8")
    yaml_p = synth_root / "data.yaml"
    yaml_p.write_text(
        f"path: {synth_root.as_posix()}\ntrain: train/images\nval: val/images\nnames:\n  0: object\n",
        encoding="utf-8",
    )
    whole = discover(str(yaml_p), 101, "detection")
    sub = subset(whole, 32, 8)
    sub_dir = prep_dir / "subset_data"
    sub_yaml = export_yolo(sub, sub_dir)
    sub_manifest = prep_dir / "subset.json"
    write_json(sub_manifest, sub)
    return Path(sub_manifest), Path(sub_yaml), sub["counts"], whole["counts"], whole["names"]


def _run_probe(idx: int, cfg: dict[str, Any], out_dir: Path, sub_manifest: Path, sub_yaml: Path, sub_counts: dict, full_counts: dict, device: str) -> dict[str, Any]:
    trial_dir = out_dir / "trials" / f"probe_{idx:02d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    req_path = trial_dir / "request.json"
    res_path = trial_dir / "result.json"
    log_path = trial_dir / "worker.log"
    req = {
        "attempt_id": uuid.uuid4().hex,
        "task": "detection",
        "config": cfg,
        "protocol_sha256": f"cross-gpu-probe-{idx}",
        "seed": 17,
        "mode": "search",
        "full_epochs": 5,
        "trial_epochs": 1,
        "resume": None,
        "manifest": str(sub_manifest),
        "device": str(device),
        "project": str(trial_dir),
        "counts": sub_counts,
        "full_counts": full_counts,
        "data": str(sub_yaml),
        "vocabulary": {},
        "max_tokens": 32,
        "data_loader": {"workers": 0, "cache_mb": 0.0, "pin_memory": "off"},
        "threads": 4,
        "logical_cpus": os.cpu_count() or 4,
        "full_counts_by_source": {"train": {}, "val": {}},
        "detection_metric": "map50",
    }
    write_json(req_path, req)
    env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    t0 = time.monotonic()
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            [sys.executable, "-m", "research_pipeline.v2.worker", str(req_path), str(res_path)],
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=240,
            check=False,
        )
    wall = time.monotonic() - t0
    raw = json.loads(res_path.read_text(encoding="utf-8")) if res_path.is_file() else {"status": "error", "error": f"exit={proc.returncode}"}
    return {
        "index": idx,
        "config": cfg,
        "status": raw.get("status"),
        "memory_mb": raw.get("memory_mb"),
        "peak_reserved_mb": raw.get("peak_reserved_mb"),
        "peak_allocated_mb": raw.get("peak_allocated_mb"),
        "quality": raw.get("quality"),
        "wall_seconds": round(wall, 3),
        "error": raw.get("error"),
    }


def evaluate_cross_gpu_transfer(
    card: dict[str, Any],
    measurements: list[dict[str, Any]],
    gpu_name: str,
    vram_free_mb: float,
    vram_total_mb: float,
    alphas: tuple[float, ...] = (0.10, 0.05),
) -> dict[str, Any]:
    """Compare Zero-shot vs 2-Point Online Shrinkage Calibration on unseen target measurements."""
    ok_records = [
        {
            "index": m.get("index"),
            "config": m["config"],
            "descriptor": m.get("descriptor"),
            "memory_mb": float(m["memory_mb"]),
        }
        for m in measurements
        if m.get("status") == "ok" and isinstance(m.get("memory_mb"), (int, float)) and math.isfinite(float(m["memory_mb"]))
    ]
    if len(ok_records) < 4:
        raise RuntimeError(f"Need at least 4 valid probes for transfer evaluation, got {len(ok_records)}")

    ordered = sorted(ok_records, key=lambda r: _activation_proxy(r["config"], r.get("descriptor"), task="detection"))
    safe_pool = [r for r in ordered if r["memory_mb"] <= 0.65 * vram_free_mb] or ordered
    anchors = [safe_pool[0], safe_pool[min(len(safe_pool) - 1, max(1, (2 * len(safe_pool)) // 3))]]
    anchor_indices = {id(a) for a in anchors}
    eval_pts = [r for r in ordered if id(r) not in anchor_indices]

    cal_2pt = calibrate_online_2point(card, anchors, task="detection")

    modes: dict[str, Any] = {}
    for mode_name, online_cal in (("zero_shot", None), ("online_2pt_calibrated", cal_2pt)):
        abs_errs: list[float] = []
        signed_errs: list[float] = []
        cov: dict[str, list[int]] = {str(a): [] for a in alphas}
        margins: dict[str, list[float]] = {str(a): [] for a in alphas}
        per_point = []
        for pt in eval_pts:
            pr95 = predict_conformal_vram(card, pt["config"], pt.get("descriptor"), alpha=0.05, online_cal=online_cal, task="detection")
            err = pt["memory_mb"] - pr95["point_mb"]
            abs_errs.append(abs(err))
            signed_errs.append(err)
            pt_entry = {
                "model": Path(pt["config"]["model"]).name,
                "imgsz": pt["config"]["imgsz"],
                "batch": pt["config"]["batch"],
                "precision": pt["config"]["precision"],
                "observed_mb": pt["memory_mb"],
                "point_mb": pr95["point_mb"],
            }
            for a in alphas:
                pr = predict_conformal_vram(card, pt["config"], pt.get("descriptor"), alpha=a, online_cal=online_cal, task="detection")
                cov[str(a)].append(1 if pt["memory_mb"] <= pr["upper_bound_mb"] else 0)
                margins[str(a)].append(pr["upper_bound_mb"] - pr["point_mb"])
                pt_entry[f"upper_{a:g}_mb"] = pr["upper_bound_mb"]
            per_point.append(pt_entry)

        modes[mode_name] = {
            "n_eval": len(eval_pts),
            "mae_mb": statistics.mean(abs_errs),
            "rmse_mb": math.sqrt(statistics.mean(e * e for e in signed_errs)),
            "bias_mb": statistics.mean(signed_errs),
            "coverage": {k: statistics.mean(v) for k, v in cov.items()},
            "covered_count": {k: f"{sum(v)}/{len(v)}" for k, v in cov.items()},
            "mean_margin_mb": {k: statistics.mean(v) for k, v in margins.items()},
            "per_point": per_point,
        }

    return {
        "schema_version": 1,
        "target_gpu": gpu_name,
        "vram_free_mb": vram_free_mb,
        "vram_total_mb": vram_total_mb,
        "source_card_metadata": card.get("metadata"),
        "source_card_sha256": card.get("sha256"),
        "n_total_measurements": len(measurements),
        "n_ok": len(ok_records),
        "anchors": [
            {
                "model": Path(a["config"]["model"]).name,
                "imgsz": a["config"]["imgsz"],
                "batch": a["config"]["batch"],
                "precision": a["config"]["precision"],
                "observed_mb": a["memory_mb"],
            }
            for a in anchors
        ],
        "calibration_2pt_params": cal_2pt,
        "modes": modes,
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    z = summary["modes"]["zero_shot"]
    c = summary["modes"]["online_2pt_calibrated"]
    lines = [
        f"# Cross-GPU Conformal VRAM Transfer Report: {summary['target_gpu']}",
        "",
        f"- Source Card: `configs/conformal_vram_card_rtx3050.json` (`sha256`: `{str(summary['source_card_sha256'])[:16]}...`)",
        f"- Target Hardware: `{summary['target_gpu']}` (Free={summary['vram_free_mb']:.1f} MiB / Total={summary['vram_total_mb']:.1f} MiB)",
        f"- Valid Probes: {summary['n_ok']}/{summary['n_total_measurements']} | Anchors = 2 | Held-out Target Eval = {z['n_eval']}",
        "",
        "| Transfer Mode | MAE (MiB) | RMSE (MiB) | Coverage (90%, $\\alpha=0.10$) | Coverage (95%, $\\alpha=0.05$) | Mean Margin 95% (MiB) |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| Zero-Shot (RTX 3050 Card directly) | {z['mae_mb']:.1f} | {z['rmse_mb']:.1f} | {z['coverage']['0.1']*100:.1f}% ({z['covered_count']['0.1']}) | {z['coverage']['0.05']*100:.1f}% ({z['covered_count']['0.05']}) | +{z['mean_margin_mb']['0.05']:.1f} |",
        f"| **2-Point Online Shrinkage Calibration** | **{c['mae_mb']:.1f}** | **{c['rmse_mb']:.1f}** | **{c['coverage']['0.1']*100:.1f}% ({c['covered_count']['0.1']})** | **{c['coverage']['0.05']*100:.1f}% ({c['covered_count']['0.05']})** | +{c['mean_margin_mb']['0.05']:.1f} |",
        "",
        "## Calibration Anchors",
    ]
    for a in summary["anchors"]:
        lines.append(f"- `{a['model']}` imgsz={a['imgsz']} B={a['batch']} {a['precision']} -> {a['observed_mb']:.1f} MiB")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-GPU Split Conformal VRAM Transfer (Colab T4 / Kaggle / Dry-Run).")
    parser.add_argument("--card", default=str(REPO_ROOT / "configs" / "conformal_vram_card_rtx3050.json"))
    parser.add_argument("--data", default=str(REPO_ROOT / "runs" / "datasets" / "african_wildlife_benchmark"))
    parser.add_argument("--output", default=str(REPO_ROOT / "runs" / "colab_cross_gpu_transfer"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--dry-run-measurements", default=None, help="Replay an existing measurements.json without running GPU probes.")
    args = parser.parse_args()

    card = json.loads(Path(args.card).read_text(encoding="utf-8"))
    if not verify_conformal_vram_card(card):
        raise ValueError(f"Conformal VRAM card SHA-256 verification failed: {args.card}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run_measurements:
        saved = json.loads(Path(args.dry_run_measurements).read_text(encoding="utf-8"))
        gpu_name = f"{saved['gpu']} (dry-run replay)"
        vram_free_mb = float(saved["vram_free_mb"])
        vram_total_mb = float(saved["vram_total_mb"])
        raw_m = saved["measurements"]
        if len(raw_m) > 24:
            by_arch: dict[str, list[dict[str, Any]]] = {}
            for rec in raw_m:
                by_arch.setdefault(Path(rec["config"]["model"]).name, []).append(rec)
            measurements = [r for grp in by_arch.values() for r in grp[:8]]
        else:
            measurements = raw_m
    else:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required unless --dry-run-measurements is provided.")
        free_b, total_b = torch.cuda.mem_get_info(0)
        gpu_name = torch.cuda.get_device_name(0)
        vram_free_mb = round(free_b / (1024 ** 2), 1)
        vram_total_mb = round(total_b / (1024 ** 2), 1)
        print(f"[Cross-GPU Transfer] Target GPU: {gpu_name} | Free={vram_free_mb:.1f} MiB / Total={vram_total_mb:.1f} MiB", flush=True)

        prep_dir = out_dir / "prep"
        prep_dir.mkdir(parents=True, exist_ok=True)
        sub_manifest, sub_yaml, sub_counts, full_counts, names = _ensure_probe_dataset(args.data, prep_dir)
        cfgs = build_24_cross_gpu_probes()
        descriptors = describe_models("detection", cfgs, names)

        measurements = []
        for idx, cfg in enumerate(cfgs):
            print(f"  [{idx + 1:02d}/{len(cfgs)}] {Path(cfg['model']).name} imgsz={cfg['imgsz']} B={cfg['batch']} {cfg['precision']} ... ", end="", flush=True)
            m = _run_probe(idx, cfg, out_dir, sub_manifest, sub_yaml, sub_counts, full_counts, device=str(args.device))
            m["descriptor"] = descriptors[architecture_key(cfg)]
            measurements.append(m)
            print(f"status={m['status']} | mem={m['memory_mb']} MiB | {m['wall_seconds']}s", flush=True)

        write_json(out_dir / "measurements.json", {"gpu": gpu_name, "vram_free_mb": vram_free_mb, "vram_total_mb": vram_total_mb, "measurements": measurements})

    summary = evaluate_cross_gpu_transfer(card, measurements, gpu_name=gpu_name, vram_free_mb=vram_free_mb, vram_total_mb=vram_total_mb)
    write_json(out_dir / "summary.json", summary)
    md = _render_markdown(summary)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
