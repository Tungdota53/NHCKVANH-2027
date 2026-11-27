"""Paired 3-seed held-out Test evaluation: CC-MFBO selected optimum vs Default c00 across 3 domains.

Runs 3-seed (17, 29, 43) optimize + evaluate-v2 (--seed-scope all, --device 0) sequentially
for the 5 missing configs while reusing the already-completed Penn-Fudan c17 3-seed evaluation
at runs/nckh_2027_v2_eval_pennfudan. Supports idempotent resume and 1 clean retry on transient
Windows file-lock errors.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RIGOR_DIR = ROOT / "runs" / "nckh_2027_v2_rigor_pack"
EVAL_ROOT = RIGOR_DIR / "eval_3seed"
REUSE_PENNFUDAN_C17 = ROOT / "runs" / "nckh_2027_v2_eval_pennfudan"
SEEDS = [17, 29, 43]


@dataclass(frozen=True)
class EvalTarget:
    domain_key: str
    domain_label: str
    arm: str  # "cc_mfbo" | "default"
    config_id: str
    slug: str
    task: str
    data: str
    full_epochs: int
    detection_metric: str | None
    config: dict[str, Any]
    reuse_root: Path | None = None


TARGETS: list[EvalTarget] = [
    EvalTarget(
        domain_key="wildlife",
        domain_label="African Wildlife Detection (mAP@50)",
        arm="cc_mfbo",
        config_id="c14",
        slug="wildlife_c14_12ep",
        task="detection",
        data=str(ROOT / "runs" / "datasets" / "african_wildlife_benchmark" / "data.yaml"),
        full_epochs=12,
        detection_metric="map50",
        config={
            "model": "weights/yolo11s.pt",
            "imgsz": 480,
            "batch": 4,
            "accumulation": 1,
            "precision": "fp16",
            "optimizer": "AdamW",
            "lr0": 0.00018,
            "weight_decay": 0.0005,
            "freeze": 8,
        },
    ),
    EvalTarget(
        domain_key="wildlife",
        domain_label="African Wildlife Detection (mAP@50)",
        arm="default",
        config_id="c00",
        slug="wildlife_c00_12ep",
        task="detection",
        data=str(ROOT / "runs" / "datasets" / "african_wildlife_benchmark" / "data.yaml"),
        full_epochs=12,
        detection_metric="map50",
        config={
            "model": "weights/yolo11n.pt",
            "imgsz": 320,
            "batch": 8,
            "accumulation": 1,
            "precision": "fp16",
            "optimizer": "SGD",
            "lr0": 0.01,
            "weight_decay": 0.0005,
            "freeze": 10,
        },
    ),
    EvalTarget(
        domain_key="pennfudan",
        domain_label="Penn-Fudan Pedestrian Detection (mAP@50:95)",
        arm="cc_mfbo",
        config_id="c17",
        slug="pennfudan_c17_12ep",
        task="detection",
        data=str(ROOT / "runs" / "datasets" / "penn_fudan_research" / "data.yaml"),
        full_epochs=12,
        detection_metric="map50_95",
        config={
            "model": "weights/yolo11s.pt",
            "imgsz": 640,
            "batch": 4,
            "accumulation": 1,
            "precision": "fp16",
            "optimizer": "AdamW",
            "lr0": 0.0002,
            "weight_decay": 0.0005,
            "freeze": 6,
        },
        reuse_root=REUSE_PENNFUDAN_C17,
    ),
    EvalTarget(
        domain_key="pennfudan",
        domain_label="Penn-Fudan Pedestrian Detection (mAP@50:95)",
        arm="default",
        config_id="c00",
        slug="pennfudan_c00_12ep",
        task="detection",
        data=str(ROOT / "runs" / "datasets" / "penn_fudan_research" / "data.yaml"),
        full_epochs=12,
        detection_metric="map50_95",
        config={
            "model": "weights/yolo11n.pt",
            "imgsz": 320,
            "batch": 8,
            "accumulation": 1,
            "precision": "fp16",
            "optimizer": "SGD",
            "lr0": 0.01,
            "weight_decay": 0.0005,
            "freeze": 10,
        },
    ),
    EvalTarget(
        domain_key="cifar10",
        domain_label="CIFAR-10 Research CNN Classification (Top-1 Acc)",
        arm="cc_mfbo",
        config_id="c12",
        slug="cifar10_c12_20ep",
        task="classification",
        data=str(ROOT / "runs" / "datasets" / "cifar10_research"),
        full_epochs=20,
        detection_metric=None,
        config={
            "model": "cnn_classifier",
            "imgsz": 32,
            "batch": 16,
            "accumulation": 2,
            "precision": "fp32",
            "optimizer": "AdamW",
            "lr0": 0.001,
            "weight_decay": 0.01,
            "width": 64,
        },
    ),
    EvalTarget(
        domain_key="cifar10",
        domain_label="CIFAR-10 Research CNN Classification (Top-1 Acc)",
        arm="default",
        config_id="c00",
        slug="cifar10_c00_20ep",
        task="classification",
        data=str(ROOT / "runs" / "datasets" / "cifar10_research"),
        full_epochs=20,
        detection_metric=None,
        config={
            "model": "cnn_classifier",
            "imgsz": 32,
            "batch": 32,
            "accumulation": 1,
            "precision": "fp32",
            "optimizer": "AdamW",
            "lr0": 0.003,
            "weight_decay": 0.01,
            "width": 16,
        },
    ),
]


def _find_valid_evaluation(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = sorted(root.rglob("evaluation.json"))
    for p in reversed(candidates):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("status") == "ok" and d.get("evaluated_seeds") == SEEDS and d.get("multi_seed_summary"):
            return p
    return None


def _latest_run_dir(opt_parent: Path) -> Path:
    dirs = sorted(p for p in opt_parent.iterdir() if p.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No optimize run directory found in {opt_parent}")
    for d in reversed(dirs):
        if (d / "state.json").exists():
            return d
    return dirs[-1]


def _run_subprocess(cmd: list[str], label: str) -> None:
    print(f"[RUN] {label}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}")


def _ensure_evaluated(target: EvalTarget, device: str, budget_seconds: int) -> Path:
    if target.reuse_root is not None:
        reused = _find_valid_evaluation(target.reuse_root)
        if reused is None:
            raise FileNotFoundError(f"Expected reusable evaluation in {target.reuse_root}")
        print(f"[REUSE] {target.slug} -> {reused.relative_to(ROOT)}", flush=True)
        return reused

    target_dir = EVAL_ROOT / target.slug
    existing = _find_valid_evaluation(target_dir)
    if existing is not None:
        print(f"[SKIP] {target.slug} already evaluated -> {existing.relative_to(ROOT)}", flush=True)
        return existing

    target_dir.mkdir(parents=True, exist_ok=True)
    grid_path = target_dir / "grid_single.json"
    grid_path.write_text(json.dumps([target.config], indent=2), encoding="utf-8")

    last_err: Exception | None = None
    for attempt in range(2):
        opt_parent = target_dir / (f"opt_attempt_{attempt}" if attempt > 0 else "opt")
        opt_parent.mkdir(parents=True, exist_ok=True)
        cmd_opt = [
            sys.executable,
            "-m",
            "research_pipeline.v2",
            "optimize",
            "--data",
            target.data,
            "--task",
            target.task,
            "--device",
            str(device),
            "--output",
            str(opt_parent),
            "--grid",
            str(grid_path),
            "--full-epochs",
            str(target.full_epochs),
            "--seeds",
            *[str(s) for s in SEEDS],
            "--method",
            "default",
            "--budget-seconds",
            str(budget_seconds),
        ]
        if target.detection_metric:
            cmd_opt.extend(["--detection-metric", target.detection_metric])
        try:
            _run_subprocess(cmd_opt, f"{target.slug} optimize (attempt {attempt + 1})")
            run_dir = _latest_run_dir(opt_parent)
            cmd_eval = [
                sys.executable,
                "-m",
                "research_pipeline.v2",
                "evaluate-v2",
                "--run",
                str(run_dir),
                "--seed-scope",
                "all",
            ]
            _run_subprocess(cmd_eval, f"{target.slug} evaluate-v2 (attempt {attempt + 1})")
            ev = _find_valid_evaluation(run_dir)
            if ev is None:
                raise RuntimeError(f"evaluate-v2 did not produce valid 3-seed evaluation.json in {run_dir}")
            return ev
        except Exception as exc:
            last_err = exc
            print(f"[WARN] {target.slug} attempt {attempt + 1} failed: {exc}", flush=True)
            time.sleep(2.0)

    raise RuntimeError(f"All attempts failed for {target.slug}: {last_err}")


def _extract_summary(target: EvalTarget, eval_path: Path) -> dict[str, Any]:
    raw = json.loads(eval_path.read_text(encoding="utf-8"))
    run_dir = eval_path.parents[2]
    state_summary: dict[str, Any] = {}
    state_path = run_dir / "state.json"
    if state_path.exists():
        st = json.loads(state_path.read_text(encoding="utf-8"))
        full_outcomes: list[dict[str, Any]] = []
        for lst in (st.get("results") or {}).values():
            if isinstance(lst, list):
                full_outcomes.extend(
                    o for o in lst if isinstance(o, dict) and o.get("status") == "ok" and o.get("mode") in (None, "train", "full")
                )
        if full_outcomes:
            times = [
                float(o.get("wall_seconds") if o.get("wall_seconds") is not None else o.get("measured_loop_seconds"))
                for o in full_outcomes
                if o.get("wall_seconds") is not None or o.get("measured_loop_seconds") is not None
            ]
            mems = [float(o["memory_mb"]) for o in full_outcomes if o.get("memory_mb") is not None]
            vals = [float(o["quality"]) for o in full_outcomes if o.get("quality") is not None]
            state_summary = {
                "n_full_outcomes": len(full_outcomes),
                "mean_train_seconds": statistics.mean(times) if times else None,
                "max_memory_mb": max(mems) if mems else None,
                "val_quality_mean": statistics.mean(vals) if vals else None,
            }

    ms = raw.get("multi_seed_summary", {})
    per_seed = []
    psr = raw.get("per_seed_results")
    if isinstance(psr, dict):
        items = sorted(psr.items(), key=lambda kv: int(kv[0]))
    elif isinstance(psr, list):
        items = [(str(item.get("seed")), item) for item in psr]
    else:
        items = []

    for seed_str, item in items:
        row = {
            "seed": int(seed_str) if seed_str.isdigit() else item.get("seed"),
            "quality": item.get("quality"),
            "map50": item.get("map50"),
            "map50_95": item.get("map50_95"),
            "f1": item.get("f1"),
            "precision": item.get("precision"),
            "recall": item.get("recall"),
            "accuracy": item.get("accuracy"),
        }
        per_seed.append({k: v for k, v in row.items() if v is not None})

    lat = raw.get("latency", {}) if isinstance(raw.get("latency"), dict) else {}
    return {
        "domain_key": target.domain_key,
        "domain_label": target.domain_label,
        "arm": target.arm,
        "config_id": target.config_id,
        "config": target.config,
        "full_epochs": target.full_epochs,
        "primary_metric": raw.get("primary_metric"),
        "evaluation_path": str(eval_path.relative_to(ROOT)).replace("\\", "/"),
        "evaluated_seeds": raw.get("evaluated_seeds"),
        "multi_seed_summary": ms,
        "validation_vs_test": raw.get("validation_vs_test"),
        "latency": {
            "available": lat.get("available"),
            "mean_fps": lat.get("mean_fps"),
            "fps_sd": (lat.get("fps") or {}).get("sd") if isinstance(lat.get("fps"), dict) else None,
            "mean_total_ms_per_image": lat.get("mean_total_ms_per_image"),
            "total_ms_sd": (lat.get("total_ms_per_image") or {}).get("sd") if isinstance(lat.get("total_ms_per_image"), dict) else None,
        },
        "train_resource_summary": state_summary,
        "per_seed_results": per_seed,
    }


def _paired_comparison(domain_entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    opt = domain_entries["cc_mfbo"]
    base = domain_entries["default"]
    opt_seeds = {int(r["seed"]): r for r in opt["per_seed_results"]}
    base_seeds = {int(r["seed"]): r for r in base["per_seed_results"]}
    common_seeds = sorted(set(opt_seeds) & set(base_seeds))

    primary = opt.get("primary_metric") or "quality"
    deltas = [float(opt_seeds[s]["quality"]) - float(base_seeds[s]["quality"]) for s in common_seeds]
    delta_mean = statistics.mean(deltas)
    delta_sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    t_stat = (delta_mean / (delta_sd / math.sqrt(len(deltas)))) if delta_sd > 0 and len(deltas) > 1 else None

    opt_mean = float(opt["multi_seed_summary"]["quality"]["mean"])
    base_mean = float(base["multi_seed_summary"]["quality"]["mean"])
    rel_pct = ((opt_mean - base_mean) / base_mean * 100.0) if base_mean else None

    extra_metrics: dict[str, Any] = {}
    for metric in ("map50", "map50_95", "f1", "precision", "recall", "accuracy"):
        om = opt["multi_seed_summary"].get(metric)
        bm = base["multi_seed_summary"].get(metric)
        if isinstance(om, dict) and isinstance(bm, dict) and om.get("mean") is not None and bm.get("mean") is not None:
            extra_metrics[metric] = {
                "cc_mfbo_mean": float(om["mean"]),
                "cc_mfbo_sd": float(om.get("sd") or 0.0),
                "default_mean": float(bm["mean"]),
                "default_sd": float(bm.get("sd") or 0.0),
                "delta": float(om["mean"]) - float(bm["mean"]),
            }

    return {
        "domain_key": opt["domain_key"],
        "domain_label": opt["domain_label"],
        "primary_metric": primary,
        "full_epochs": opt["full_epochs"],
        "seeds": common_seeds,
        "cc_mfbo": opt,
        "default": base,
        "paired_delta_primary": {
            "mean": delta_mean,
            "sd": delta_sd,
            "relative_pct": rel_pct,
            "paired_t_df2": t_stat,
            "per_seed": {str(s): float(opt_seeds[s]["quality"]) - float(base_seeds[s]["quality"]) for s in common_seeds},
            "all_seeds_positive": all(d > 0 for d in deltas),
        },
        "metric_breakdown": extra_metrics,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paired 3-Seed Held-Out Test Validation: CC-MFBO vs Default (`c00`)",
        "",
        f"- Seeds: `{SEEDS}` (`seed-scope=all`, `rect=False`)",
        "",
        "| Domain | Primary Metric | Epochs | CC-MFBO (Selected) | Default (`c00`) | Paired $\\Delta$ (mean $\\pm$ sd) | Rel. Gain | All 3 Seeds $>$ 0 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for dom in report["domains"].values():
        opt = dom["cc_mfbo"]
        base = dom["default"]
        oms = opt["multi_seed_summary"]["quality"]
        bms = base["multi_seed_summary"]["quality"]
        pd = dom["paired_delta_primary"]
        rel = f"+{pd['relative_pct']:.2f}%" if pd["relative_pct"] is not None else "n/a"
        lines.append(
            f"| {dom['domain_label']} | `{dom['primary_metric']}` | {dom['full_epochs']} | "
            f"`{opt['config_id']}`: **{oms['mean']:.4f} $\\pm$ {oms['sd']:.4f}** | "
            f"`{base['config_id']}`: {bms['mean']:.4f} $\\pm$ {bms['sd']:.4f} | "
            f"**{pd['mean']:+.4f} $\\pm$ {pd['sd']:.4f}** | {rel} | {pd['all_seeds_positive']} |"
        )
    lines.append("")
    for dom in report["domains"].values():
        lines.append(f"## {dom['domain_label']}")
        for arm_key in ("cc_mfbo", "default"):
            arm = dom[arm_key]
            lat = arm["latency"]
            vt = arm.get("validation_vs_test") or {}
            tr = arm.get("train_resource_summary") or {}
            fps_str = f"{lat['mean_fps']:.2f} FPS ({lat['mean_total_ms_per_image']:.2f} ms)" if lat.get("available") else "n/a (CNN CPU/CUDA batch classifier)"
            val_str = f"{vt.get('validation_mean'):.4f}" if isinstance(vt.get("validation_mean"), (int, float)) else "n/a"
            gap_str = f"{vt.get('generalization_gap_test_minus_val'):+.4f}" if isinstance(vt.get("generalization_gap_test_minus_val"), (int, float)) else "n/a"
            mem_str = f"{tr.get('max_memory_mb'):.1f} MiB" if tr.get("max_memory_mb") is not None else "n/a"
            t_str = f"{tr.get('mean_train_seconds'):.1f}s" if tr.get("mean_train_seconds") is not None else "n/a"
            lines.append(
                f"- **{arm['arm']} (`{arm['config_id']}`)**: Test `{dom['primary_metric']}` = "
                f"{arm['multi_seed_summary']['quality']['mean']:.4f} ± {arm['multi_seed_summary']['quality']['sd']:.4f} | "
                f"Val = {val_str} (gap {gap_str}) | Inference = {fps_str} | Train/seed = {t_str}, Peak VRAM = {mem_str}"
            )
            lines.append(f"  - Eval JSON: `{arm['evaluation_path']}`")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paired 3-seed Test validation (CC-MFBO vs Default c00).")
    parser.add_argument("--device", default="0", help="CUDA device index (default 0)")
    parser.add_argument("--budget-seconds", type=int, default=1800, help="Budget seconds per 3-seed optimize call")
    args = parser.parse_args()

    RIGOR_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for target in TARGETS:
        eval_path = _ensure_evaluated(target, device=args.device, budget_seconds=args.budget_seconds)
        summary = _extract_summary(target, eval_path)
        grouped.setdefault(target.domain_key, {})[target.arm] = summary

    domains_report = {k: _paired_comparison(v) for k, v in grouped.items()}
    report = {
        "schema_version": 1,
        "seeds": SEEDS,
        "domains": domains_report,
    }
    json_path = RIGOR_DIR / "paired_test_comparison.json"
    md_path = RIGOR_DIR / "paired_test_comparison.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    print(f"[DONE] Saved {json_path} and {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
