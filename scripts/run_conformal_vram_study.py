"""Phase 2: Real RTX 3050 6GB Conformal VRAM profiling, LOMO + 2-pt transfer & OOM study."""

import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from yolo_tuner.adapters import export_yolo, subset
from yolo_tuner.pipeline import write_json
from yolo_tuner.research import digest
from research_pipeline.v2.conformal_vram import (
    calibrate_online_2point,
    evaluate_conformal_vram,
    fit_conformal_vram_card,
    predict_conformal_vram,
    verify_conformal_vram_card,
)
from research_pipeline.v2.data import discover
from research_pipeline.v2.model_features import architecture_key, describe_models


def build_probe_configs():
    """Return 28 realistic YOLO configs spanning 3 models, imgsz, batch, precision, freeze, optimizer."""
    specs = [
        # --- weights/yolo11n.pt (10 configs, safe -> high-VRAM -> OOM probe) ---
        ("weights/yolo11n.pt", 320, 2, "fp16", 0, "AdamW"),
        ("weights/yolo11n.pt", 320, 4, "fp16", 5, "AdamW"),
        ("weights/yolo11n.pt", 320, 8, "fp16", 0, "SGD"),
        ("weights/yolo11n.pt", 320, 16, "fp16", 0, "AdamW"),
        ("weights/yolo11n.pt", 480, 4, "fp16", 0, "AdamW"),
        ("weights/yolo11n.pt", 480, 8, "fp32", 0, "AdamW"),
        ("weights/yolo11n.pt", 640, 4, "fp16", 0, "AdamW"),
        ("weights/yolo11n.pt", 640, 8, "fp32", 0, "SGD"),
        ("weights/yolo11n.pt", 640, 16, "fp32", 0, "AdamW"),
        ("weights/yolo11n.pt", 800, 16, "fp32", 0, "AdamW"),
        # --- weights/yolo11s.pt (10 configs, medium -> high-VRAM -> real OOM probes) ---
        ("weights/yolo11s.pt", 320, 2, "fp16", 5, "AdamW"),
        ("weights/yolo11s.pt", 320, 4, "fp16", 0, "AdamW"),
        ("weights/yolo11s.pt", 320, 8, "fp32", 0, "SGD"),
        ("weights/yolo11s.pt", 480, 4, "fp16", 0, "AdamW"),
        ("weights/yolo11s.pt", 480, 8, "fp16", 0, "AdamW"),
        ("weights/yolo11s.pt", 640, 2, "fp16", 5, "AdamW"),
        ("weights/yolo11s.pt", 640, 4, "fp32", 0, "AdamW"),
        ("weights/yolo11s.pt", 640, 8, "fp32", 0, "AdamW"),
        ("weights/yolo11s.pt", 640, 16, "fp32", 0, "AdamW"),
        ("weights/yolo11s.pt", 800, 16, "fp32", 0, "AdamW"),
        # --- weights/yolo26n.pt (8 configs, distinct architecture LOMO transfer test) ---
        ("weights/yolo26n.pt", 320, 2, "fp16", 5, "AdamW"),
        ("weights/yolo26n.pt", 320, 4, "fp16", 0, "AdamW"),
        ("weights/yolo26n.pt", 320, 8, "fp32", 0, "AdamW"),
        ("weights/yolo26n.pt", 480, 4, "fp16", 0, "SGD"),
        ("weights/yolo26n.pt", 480, 8, "fp16", 0, "AdamW"),
        ("weights/yolo26n.pt", 640, 4, "fp16", 0, "AdamW"),
        ("weights/yolo26n.pt", 640, 8, "fp32", 0, "AdamW"),
        ("weights/yolo26n.pt", 800, 8, "fp32", 0, "AdamW"),
    ]
    out = []
    for model, imgsz, batch, prec, freeze, opt in specs:
        out.append({
            "model": model,
            "imgsz": imgsz,
            "batch": batch,
            "accumulation": 1,
            "precision": prec,
            "optimizer": opt,
            "lr0": 0.001,
            "weight_decay": 0.0005,
            "freeze": freeze,
        })
    return out


def run_isolated_probe(idx, cfg, folder, subset_manifest_path, subset_yaml_path, subset_counts, full_counts, device="0"):
    trial_dir = folder / "trials" / f"probe_{idx:02d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    req_path = trial_dir / "request.json"
    res_path = trial_dir / "result.json"
    log_path = trial_dir / "worker.log"
    req = {
        "attempt_id": uuid.uuid4().hex,
        "task": "detection",
        "config": cfg,
        "protocol_sha256": digest({"study": "phase2_conformal_vram", "idx": idx}),
        "seed": 17,
        "mode": "search",
        "full_epochs": 5,
        "trial_epochs": 1,
        "resume": None,
        "manifest": str(subset_manifest_path),
        "device": str(device),
        "project": str(trial_dir),
        "counts": subset_counts,
        "full_counts": full_counts,
        "data": str(subset_yaml_path),
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
            timeout=180,
        )
    wall = time.monotonic() - t0
    if res_path.is_file():
        raw = json.loads(res_path.read_text(encoding="utf-8"))
    else:
        raw = {"status": "error", "error": f"Worker exited {proc.returncode} without result.json"}
    return {
        "index": idx,
        "config": cfg,
        "status": raw.get("status"),
        "memory_mb": raw.get("memory_mb"),
        "peak_reserved_mb": raw.get("peak_reserved_mb"),
        "peak_allocated_mb": raw.get("peak_allocated_mb"),
        "quality": raw.get("quality"),
        "wall_seconds": round(wall, 3),
        "measured_loop_seconds": raw.get("measured_loop_seconds"),
        "error": raw.get("error"),
        "trial_dir": str(trial_dir),
    }


def _lomo_evaluation(ok_records, descriptors, alphas=(0.10, 0.05)):
    by_model = {}
    for rec in ok_records:
        by_model.setdefault(rec["config"]["model"], []).append(rec)
    lomo = {}
    for held_model, target_recs in sorted(by_model.items()):
        source_recs = [r for m, rs in sorted(by_model.items()) if m != held_model for r in rs]
        rng = random.Random(2027)
        shuffled = list(source_recs)
        rng.shuffle(shuffled)
        mid = len(shuffled) // 2
        card = fit_conformal_vram_card(shuffled[:mid], shuffled[mid:], alphas=alphas, ridge=1.0, task="detection")

        ordered_target = sorted(target_recs, key=lambda r: (r["memory_mb"], r["config"]["imgsz"], r["config"]["batch"]))
        anchors = (
            [ordered_target[0], ordered_target[min(len(ordered_target) - 1, max(1, len(ordered_target) // 3))]]
            if len(ordered_target) >= 3 else ordered_target[:2]
        )
        eval_pts = [r for r in ordered_target if r not in anchors] or ordered_target
        cal_2pt = calibrate_online_2point(card, anchors, task="detection") if len(anchors) >= 1 else None

        res_model = {"n_source": len(source_recs), "n_target_total": len(target_recs), "n_eval_unseen": len(eval_pts)}
        for mode_name, use_cal in (("zero_shot", None), ("online_2pt_calibrated", cal_2pt)):
            abs_errs = []
            cov = {str(a): [] for a in alphas}
            widths = {str(a): [] for a in alphas}
            for pt in eval_pts:
                pr0 = predict_conformal_vram(card, pt["config"], pt["descriptor"], alpha=alphas[-1], online_cal=use_cal, task="detection")
                abs_errs.append(abs(pt["memory_mb"] - pr0["point_mb"]))
                for a in alphas:
                    pr = predict_conformal_vram(card, pt["config"], pt["descriptor"], alpha=a, online_cal=use_cal, task="detection")
                    cov[str(a)].append(1 if pt["memory_mb"] <= pr["upper_bound_mb"] else 0)
                    widths[str(a)].append(pr["upper_bound_mb"] - pr["point_mb"])
            res_model[mode_name] = {
                "mae_mb": statistics.mean(abs_errs) if abs_errs else None,
                "coverage": {k: statistics.mean(v) if v else None for k, v in cov.items()},
                "mean_margin_mb": {k: statistics.mean(v) if v else None for k, v in widths.items()},
            }
        lomo[held_model] = res_model
    return lomo


def main():
    analyze_only = "--analyze-only" in sys.argv[1:]
    out_dir = REPO_ROOT / "runs" / "nckh_conformal_vram_study"
    out_dir.mkdir(parents=True, exist_ok=True)
    prep_dir = out_dir / "prep"
    prep_dir.mkdir(parents=True, exist_ok=True)

    if analyze_only and (out_dir / "measurements.json").is_file():
        saved = json.loads((out_dir / "measurements.json").read_text(encoding="utf-8"))
        gpu_name = saved["gpu"]
        vram_free_mb = saved["vram_free_mb"]
        vram_total_mb = saved["vram_total_mb"]
        measurements = saved["measurements"]
        descriptors = {architecture_key(m["config"]): m["descriptor"] for m in measurements}
        print(f"[Phase 2 analyze-only] Loaded {len(measurements)} probes for {gpu_name}", flush=True)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required for Phase 2 RTX 3050 Conformal VRAM study")
        free_b, total_b = torch.cuda.mem_get_info(0)
        gpu_name = torch.cuda.get_device_name(0)
        vram_free_mb = round(free_b / (1024 ** 2), 1)
        vram_total_mb = round(total_b / (1024 ** 2), 1)
        print(f"[Phase 2] GPU: {gpu_name} | Free={vram_free_mb:.1f} MiB / Total={vram_total_mb:.1f} MiB", flush=True)

        whole = discover(str(REPO_ROOT / "runs" / "datasets" / "african_wildlife_benchmark"), 101, "detection")
        sub = subset(whole, 48, 16)
        sub_dir = prep_dir / "subset_data"
        sub_yaml = export_yolo(sub, sub_dir)
        sub_manifest = prep_dir / "subset.json"
        write_json(sub_manifest, sub)

        cfgs = build_probe_configs()
        descriptors = describe_models("detection", cfgs, whole["names"])

        measurements = []
        for idx, cfg in enumerate(cfgs):
            print(
                f"  [{idx + 1:02d}/{len(cfgs)}] {Path(cfg['model']).name} imgsz={cfg['imgsz']} B={cfg['batch']} "
                f"{cfg['precision']} freeze={cfg['freeze']} {cfg['optimizer']} ... ",
                end="",
                flush=True,
            )
            m = run_isolated_probe(idx, cfg, out_dir, sub_manifest, sub_yaml, sub["counts"], whole["counts"], device="0")
            desc = descriptors[architecture_key(cfg)]
            m["descriptor"] = desc
            measurements.append(m)
            print(f"status={m['status']} | mem={m['memory_mb']} MiB | q={m['quality']} | {m['wall_seconds']}s", flush=True)

        write_json(out_dir / "measurements.json", {"gpu": gpu_name, "vram_free_mb": vram_free_mb, "vram_total_mb": vram_total_mb, "measurements": measurements})

    ok_records = [
        {"config": m["config"], "descriptor": m["descriptor"], "memory_mb": float(m["memory_mb"])}
        for m in measurements
        if m["status"] == "ok" and isinstance(m.get("memory_mb"), (int, float)) and math.isfinite(m["memory_mb"])
    ]
    oom_or_spill_records = [
        m for m in measurements
        if m["status"] == "oom" or (isinstance(m.get("memory_mb"), (int, float)) and float(m["memory_mb"]) > vram_free_mb)
    ]

    # Repeated 50x Split Conformal Evaluation on observed ok_records (plus edge-GPU ceiling 1800 MiB & WDDM spill ceiling)
    eval_split = evaluate_conformal_vram(ok_records, alphas=(0.10, 0.05), n_splits=50, seed=2027, task="detection", limit_mb=1800.0)
    eval_spill = evaluate_conformal_vram(ok_records, alphas=(0.10, 0.05), n_splits=50, seed=2027, task="detection", limit_mb=vram_free_mb)

    # Export canonical card using deterministic stratified split across models (fit 50% / cal 50%)
    fit_recs, cal_recs = [], []
    by_m = {}
    for rec in ok_records:
        by_m.setdefault(rec["config"]["model"], []).append(rec)
    for m_name, recs in sorted(by_m.items()):
        recs_s = sorted(recs, key=lambda r: (r["memory_mb"], r["config"]["imgsz"], r["config"]["batch"]))
        for k, r in enumerate(recs_s):
            (cal_recs if k % 2 == 1 else fit_recs).append(r)
    card = fit_conformal_vram_card(
        fit_recs,
        cal_recs,
        alphas=(0.10, 0.05),
        ridge=1.0,
        task="detection",
        metadata={"gpu": gpu_name, "vram_total_mb": vram_total_mb, "dataset": "african_wildlife_benchmark", "n_probes": len(ok_records)},
    )
    verify_conformal_vram_card(card)
    card_path = REPO_ROOT / "configs" / "conformal_vram_card_rtx3050.json"
    write_json(card_path, card)

    lomo = _lomo_evaluation(ok_records, descriptors, alphas=(0.10, 0.05))

    oom_check = []
    for om in oom_or_spill_records:
        pr = predict_conformal_vram(card, om["config"], om["descriptor"], alpha=0.05, task="detection")
        oom_check.append({
            "index": om["index"],
            "config": om["config"],
            "observed_status": om["status"],
            "observed_memory_mb": om["memory_mb"],
            "wall_seconds": om["wall_seconds"],
            "predicted_point_mb": pr["point_mb"],
            "upper_bound_95_mb": pr["upper_bound_mb"],
            "exceeds_vram_total": pr["upper_bound_mb"] > vram_total_mb,
            "exceeds_vram_free": pr["upper_bound_mb"] > vram_free_mb,
        })

    q05 = card["quantiles"]["0.05"]["q_add_mb"]
    q10 = card["quantiles"]["0.1"]["q_add_mb"]

    summary = {
        "gpu": gpu_name,
        "vram_free_mb": vram_free_mb,
        "vram_total_mb": vram_total_mb,
        "total_probes": len(measurements),
        "ok_count": len(ok_records),
        "oom_or_spill_count": len(oom_or_spill_records),
        "other_error_count": sum(1 for m in measurements if m["status"] not in {"ok", "oom"}),
        "conformal_card_path": str(card_path),
        "conformal_card_sha256": card["sha256"],
        "quantiles_summary": {"q_add_0.05_mb": q05, "q_add_0.1_mb": q10},
        "repeated_split_eval_1800": eval_split,
        "repeated_split_eval_vram_free": eval_spill,
        "lomo_transfer": lomo,
        "vram_spill_and_oom_screening": oom_check,
        "measured_ok_range_mb": [min(r["memory_mb"] for r in ok_records), max(r["memory_mb"] for r in ok_records)] if ok_records else None,
    }
    write_json(out_dir / "summary.json", summary)

    lines = [
        "# Phase 2 — Real RTX 3050 6GB Conformal VRAM Study",
        "",
        f"- GPU: `{gpu_name}` (`{vram_free_mb:.1f}` MiB free / `{vram_total_mb:.1f}` MiB total)",
        f"- Probes completed: `{len(ok_records)}/{len(measurements)}` OK, `{len(oom_or_spill_records)}` spilled past physical free VRAM (`>{vram_free_mb:.0f} MiB`) into WDDM shared memory",
        f"- Exported Card: `{card_path.relative_to(REPO_ROOT)}` (fit={len(fit_recs)}, cal={len(cal_recs)}, q_add@95={q05:.1f} MiB, q_add@90={q10:.1f} MiB)",
        "",
        "## In-Distribution Split Conformal (50x stratified splits)",
        "",
    ]
    for a_str, stats in sorted(eval_split.get("splits_summary", {}).items()):
        sp_stats = eval_spill.get("splits_summary", {}).get(a_str, {})
        lines.append(
            f"- alpha={a_str} (target {stats['target_coverage']:.0%}): empirical coverage = `{stats['empirical_coverage_mean']:.3f}` (±{stats['empirical_coverage_sd']:.3f}), "
            f"MAE = `{stats['mae_mb_mean']:.1f}` MiB, RMSE = `{stats['rmse_mb_mean']:.1f}` MiB, "
            f"mean conformal margin = `{stats['mean_interval_width_mb']:.1f}` MiB, "
            f"pruning recall@1800 = `{stats['oom_prevention_rate']}`, spill recall@{vram_free_mb:.0f} = `{sp_stats.get('oom_prevention_rate')}`"
        )
    lines += ["", "## Leave-One-Model-Out (LOMO) Transfer", "", "| Held-out Model | Mode | Unseen n | MAE (MiB) | Cov@90 | Cov@95 | Mean Margin@95 |", "|---|---|---:|---:|---:|---:|---:|"]
    for m_name, lstats in lomo.items():
        for mode in ("zero_shot", "online_2pt_calibrated"):
            ms = lstats[mode]
            lines.append(
                f"| {Path(m_name).name} | {mode} | {lstats['n_eval_unseen']} | "
                f"{ms['mae_mb']:.1f} | {ms['coverage']['0.1']:.3f} | {ms['coverage']['0.05']:.3f} | {ms['mean_margin_mb']['0.05']:.1f} |"
            )
    if oom_check:
        lines += ["", "## Real Physical VRAM Overflow / WDDM Thrashing Detection", "", "| Probe | Model | Img/B/Prec | Obs Mem (MiB) | Wall (s) | Pred Point (MiB) | 95% Upper (MiB) | Flagged > Free VRAM? |", "|---:|---|---|---:|---:|---:|---:|---|"]
        for oc in oom_check:
            c = oc["config"]
            lines.append(f"| #{oc['index']} | {Path(c['model']).name} | {c['imgsz']}/{c['batch']}/{c['precision']} | {oc['observed_memory_mb']:.1f} | {oc['wall_seconds']:.2f} | {oc['predicted_point_mb']:.1f} | {oc['upper_bound_95_mb']:.1f} | {oc['exceeds_vram_free']} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    from research_pipeline.v2.paper_export import export_conformal_vram_svg
    svg_path = export_conformal_vram_svg(measurements, card, out_dir / "conformal_vram_calibration.svg", vram_free_mb=vram_free_mb, edge_limit_mb=1800.0, lang="vi")
    print(f"\n[Phase 2 DONE] Summary written to {out_dir / 'summary.md'} | Plot: {svg_path}", flush=True)


if __name__ == "__main__":
    main()
