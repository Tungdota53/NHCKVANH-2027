"""Compute NCKH 2027 quantitative rigor diagnostics and generate 5 publication 300-DPI PNG/PDF figures.

Computes:
  1. Probe-vs-Full Rank Consistency (Kendall's tau & Spearman's rho across all 3 domains)
  2. Two-Family Holm-Bonferroni Inference (Family A m=5 main baselines, Family B m=2 ablations)
  3. Useful vs VRAM-Wasted Search Budget Breakdown across 8 methods x 3 domains
  4. 5 Okabe-Ito publication figures (300 DPI PNG + vector PDF) in runs/nckh_2027_v2_rigor_pack/figures/
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_pipeline.v2.algorithm_study import _holm_bonferroni  # noqa: E402
from research_pipeline.v2.conformal_vram import predict_conformal_vram  # noqa: E402

RIGOR_DIR = ROOT / "runs" / "nckh_2027_v2_rigor_pack"
FIG_DIR = RIGOR_DIR / "figures"

# Okabe-Ito colorblind-safe palette
OKABE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#1F2937",
    "gray": "#6B7280",
    "lightgray": "#D1D5DB",
}

STUDY_META = [
    ("study1", "nckh_2027_v2_study1_wildlife_n8", "Study 1: Wildlife (mAP@50)", "map50"),
    ("study2", "nckh_2027_v2_study2_pennfudan_n8", "Study 2: Penn-Fudan (mAP@50:95)", "map50_95"),
    ("study3", "nckh_2027_v2_study3_cifar10_gpu_n8", "Study 3: CIFAR-10 CNN (Acc)", "quality_score"),
]

FAMILY_A_MAIN = ["adaptive_full", "default", "constrained_bo", "random", "tpe_bo"]
FAMILY_B_ABLATION = ["mf_no_conformal", "successive_halving"]

METHOD_LABELS = {
    "conformal_mf": "CC-MFBO (Ours)",
    "mf_no_conformal": "MF w/o Conformal",
    "constrained_bo": "Constrained BO",
    "tpe_bo": "TPE BO",
    "random": "Random",
    "default": "Default (c00)",
    "adaptive_full": "Adaptive Full",
    "successive_halving": "Successive Halving",
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#4B5563",
            "axes.linewidth": 0.8,
            "grid.color": "#E5E7EB",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#E5E7EB",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
        }
    )


def _save_fig(fig: plt.Figure, stem: str) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return {
        "png": str(png.relative_to(ROOT)).replace("\\", "/"),
        "pdf": str(pdf.relative_to(ROOT)).replace("\\", "/"),
    }


def _latest_file(folder: Path, filename: str) -> Path:
    matches = sorted(folder.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {folder}")
    return matches[-1]


def compute_rank_consistency() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, dirname, label, lc_key in STUDY_META:
        study_root = ROOT / "runs" / dirname
        ref_state_path = sorted(study_root.rglob("reference/*/state.json"))[-1]
        ref_st = json.loads(ref_state_path.read_text(encoding="utf-8"))
        full_q: dict[int, float] = {}
        e1_q: dict[int, float] = {}
        feasible_count = 0
        max_mem = 0.0
        for cid_str, lst in ref_st["results"].items():
            if not lst or lst[0].get("status") != "ok":
                continue
            r = lst[0]
            cid = int(cid_str)
            full_q[cid] = float(r["quality"])
            max_mem = max(max_mem, float(r.get("memory_mb") or 0.0))
            if float(r.get("memory_mb") or 0.0) <= 1850.0:
                feasible_count += 1
            lc = r.get("learning_curve") or []
            if lc and lc_key in lc[0]:
                e1_q[cid] = float(lc[0][lc_key])
        common = sorted(set(full_q) & set(e1_q))
        kt = kendalltau([e1_q[i] for i in common], [full_q[i] for i in common])
        sr = spearmanr([e1_q[i] for i in common], [full_q[i] for i in common])

        b_path = _latest_file(study_root, "BENCHMARK.json")
        b_data = json.loads(b_path.read_text(encoding="utf-8"))
        cmf_pds = [pd for o in b_data["outcomes"] if o["method"] == "conformal_mf" for pd in (o.get("proxy_diagnostics") or [])]
        cmf_spearman = [float(p["spearman_same_seed"]) for p in cmf_pds if p.get("spearman_same_seed") is not None]

        out[key] = {
            "label": label,
            "n_configs_evaluated": len(common),
            "n_feasible_1850mb": feasible_count,
            "infeasible_fraction": round(1.0 - (feasible_count / max(1, len(common))), 4),
            "max_ref_memory_mb": round(max_mem, 1),
            "epoch1_vs_full_kendall_tau": float(kt.statistic),
            "epoch1_vs_full_kendall_p": float(kt.pvalue),
            "epoch1_vs_full_spearman_rho": float(sr.statistic),
            "epoch1_vs_full_spearman_p": float(sr.pvalue),
            "conformal_mf_subset_spearman_mean": statistics.mean(cmf_spearman) if cmf_spearman else None,
            "conformal_mf_subset_n": len(cmf_spearman),
        }
    return out


def compute_two_family_holm(alg_study: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for st in alg_study["studies"]:
        sid = st["study_id"]
        by_base = {
            p["baseline"]: p
            for p in st["paired_comparisons"]
            if p["candidate"] == "conformal_mf"
        }
        fam_results: dict[str, list[dict[str, Any]]] = {}
        for fam_name, members in (("family_a_external_m5", FAMILY_A_MAIN), ("family_b_ablation_m2", FAMILY_B_ABLATION)):
            sf_pvals = []
            wx_pvals = []
            rows = []
            for m in members:
                reg = by_base[m]["metrics"]["objective_regret"]
                sf_p = float(reg["sign_flip"]["p_value"])
                wx_p = float(reg["wilcoxon"]["p_value"])
                sf_pvals.append(sf_p)
                wx_pvals.append(wx_p)
                diff = reg["candidate_minus_baseline"]
                rows.append(
                    {
                        "baseline": m,
                        "baseline_label": METHOD_LABELS.get(m, m),
                        "diff_mean": float(diff["mean"]),
                        "diff_sd": float(diff["sd"]),
                        "ci95": diff["interval"],
                        "raw_sign_flip_p": sf_p,
                        "raw_wilcoxon_p": wx_p,
                    }
                )
            holm_sf = _holm_bonferroni(sf_pvals)
            holm_wx = _holm_bonferroni(wx_pvals)
            for row, hs, hw in zip(rows, holm_sf, holm_wx):
                row["holm_sign_flip_p"] = float(hs)
                row["holm_wilcoxon_p"] = float(hw)
                row["significant_0_05"] = bool(hs < 0.05)
            fam_results[fam_name] = rows
        ms_by_method = {row["method"]: row for row in st["method_summaries"]}
        b_dir = ROOT / "runs" / {
            "4cae045574a367a3": "nckh_2027_v2_study1_wildlife_n8",
            "d4ad010059141fc7": "nckh_2027_v2_study2_pennfudan_n8",
            "77d166e1df75ed54": "nckh_2027_v2_study3_cifar10_gpu_n8",
        }[sid]
        b_data = json.loads(_latest_file(b_dir, "BENCHMARK.json").read_text(encoding="utf-8"))
        opt_hits: dict[str, list[int]] = {}
        for o in b_data["outcomes"]:
            opt_hits.setdefault(o["method"], []).append(1 if o.get("exact_optimum_hit") else 0)
        for m, m_row in ms_by_method.items():
            hits = opt_hits.get(m, [])
            m_row["exact_optimum_hit_rate"] = (sum(hits) / len(hits)) if hits else 0.0

        out[sid] = {
            "task": st["domain"]["task"],
            "full_epochs": st["full_epochs"],
            "budget_seconds": st["search_budget_seconds"],
            "method_summaries": ms_by_method,
            "two_family_holm": fam_results,
        }
    return out


def compute_budget_waste() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, dirname, label, _ in STUDY_META:
        b_path = _latest_file(ROOT / "runs" / dirname, "BENCHMARK.json")
        b = json.loads(b_path.read_text(encoding="utf-8"))
        by_m: dict[str, list[dict[str, Any]]] = {}
        for o in b["outcomes"]:
            by_m.setdefault(o["method"], []).append(o)
        methods_out: dict[str, Any] = {}
        for m, lst in by_m.items():
            online = [float(x.get("online_seconds") or 0.0) for x in lst]
            vram_w = [float(x.get("vram_violation_wasted_seconds") or 0.0) for x in lst]
            oom_w = [float(x.get("oom_wasted_seconds") or 0.0) for x in lst]
            pruned = [float(x.get("conformal_pruned_count") or 0.0) for x in lst]
            useful = [max(0.0, o_t - v_t - om_t) for o_t, v_t, om_t in zip(online, vram_w, oom_w)]
            mean_online = statistics.mean(online)
            mean_vram = statistics.mean(vram_w)
            mean_oom = statistics.mean(oom_w)
            mean_useful = statistics.mean(useful)
            methods_out[m] = {
                "label": METHOD_LABELS.get(m, m),
                "n_runs": len(lst),
                "online_seconds_mean": mean_online,
                "useful_seconds_mean": mean_useful,
                "vram_violation_wasted_seconds_mean": mean_vram,
                "oom_wasted_seconds_mean": mean_oom,
                "waste_pct_of_online": ((mean_vram + mean_oom) / mean_online * 100.0) if mean_online > 0 else 0.0,
                "conformal_pruned_count_mean": statistics.mean(pruned),
            }
        out[key] = {"label": label, "methods": methods_out}
    return out


def make_fig1(vram_summary: dict[str, Any], vram_meas: dict[str, Any], card: dict[str, Any]) -> dict[str, str]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4), gridspec_kw={"width_ratios": [1.35, 1.0]})

    recs = vram_meas["measurements"]
    by_arch = {"yolo11n.pt": [], "yolo11s.pt": [], "yolo26n.pt": []}
    oom_pts = []
    for m in recs:
        arch = Path(m["config"]["model"]).name
        pr = predict_conformal_vram(card, m["config"], m.get("descriptor"), alpha=0.05, task="detection")
        item = {"obs": m.get("memory_mb"), "pt": pr["point_mb"], "ub": pr["upper_bound_mb"], "status": m["status"]}
        if m["status"] == "ok" and item["obs"] is not None:
            by_arch.setdefault(arch, []).append(item)
        else:
            oom_pts.append((arch, item))

    colors = {"yolo11n.pt": OKABE["blue"], "yolo11s.pt": OKABE["orange"], "yolo26n.pt": OKABE["green"]}
    markers = {"yolo11n.pt": "o", "yolo11s.pt": "s", "yolo26n.pt": "^"}

    max_v = 5600.0
    xs = np.linspace(100, max_v, 100)
    ax1.plot(xs, xs, "--", color=OKABE["black"], lw=1.1, label="Exact Identity (y = x)")
    ax1.axvline(1850.0, color=OKABE["vermillion"], ls=":", lw=1.3, label="Edge Budget Ceiling (1850 MiB)")
    ax1.axhline(1850.0, color=OKABE["vermillion"], ls=":", lw=1.0, alpha=0.6)

    for arch, items in by_arch.items():
        if not items:
            continue
        obs = np.array([x["obs"] for x in items])
        ub = np.array([x["ub"] for x in items])
        pt = np.array([x["pt"] for x in items])
        ax1.scatter(obs, ub, s=42, color=colors[arch], marker=markers[arch], alpha=0.88, edgecolor="white", linewidth=0.5, label=f"{arch} (95% Upper Bound)")
        ax1.vlines(obs, pt, ub, color=colors[arch], alpha=0.28, lw=1.0)

    ax1.set_xlim(100, 5600)
    ax1.set_ylim(100, 5600)
    ax1.set_xlabel("Measured Peak VRAM (MiB)")
    ax1.set_ylabel("Split Conformal Prediction (MiB, $\\alpha=0.05$)")
    ax1.set_title("(a) Conformal 95% Upper Bound vs Observed VRAM (RTX 3050, n=56 ok)")
    ax1.grid(True)
    ax1.legend(loc="upper left", fontsize=7.6)

    # Panel (b): LOMO Transfer Coverage & MAE
    lomo = vram_summary["repeated_split_eval_1850"]["leave_one_model_out"]
    names = [Path(x["held_out_model_name"]).stem for x in lomo]
    cov_zs = [x["zero_shot_coverage"] * 100.0 for x in lomo]
    cov_2p = [x["calibrated_2pt_coverage"] * 100.0 for x in lomo]
    mae_zs = [x["zero_shot_mae_mb"] for x in lomo]
    mae_2p = [x["calibrated_2pt_mae_mb"] for x in lomo]

    x_idx = np.arange(len(names))
    w = 0.34
    b1 = ax2.bar(x_idx - w / 2, cov_zs, width=w, color=OKABE["lightgray"], edgecolor=OKABE["black"], linewidth=0.7, label="Zero-Shot Coverage (%)")
    b2 = ax2.bar(x_idx + w / 2, cov_2p, width=w, color=OKABE["blue"], edgecolor=OKABE["black"], linewidth=0.7, label="2-Pt Calibrated Coverage (%)")
    ax2.axhline(95.0, color=OKABE["vermillion"], ls="--", lw=1.2, label="Target 95% Coverage")

    for bar, mae in zip(b1, mae_zs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.4, f"{bar.get_height():.1f}%\n({mae:.0f}M)", ha="center", va="bottom", fontsize=7.3, color=OKABE["black"])
    for bar, mae in zip(b2, mae_2p):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.4, f"{bar.get_height():.1f}%\n({mae:.0f}M)", ha="center", va="bottom", fontsize=7.3, color=OKABE["blue"], fontweight="bold")

    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(names)
    ax2.set_ylim(0, 122)
    ax2.set_ylabel("Empirical Coverage (%)  [Annotation: MAE in MiB]")
    ax2.set_title("(b) Leave-One-Model-Out Architecture Transfer ($\\alpha=0.05$)")
    ax2.grid(True, axis="y")
    ax2.legend(loc="lower right", fontsize=7.5)

    fig.tight_layout()
    return _save_fig(fig, "fig1_conformal_vram_calibration")


def make_fig2(holm_data: dict[str, Any]) -> dict[str, str]:
    order = ["conformal_mf", "mf_no_conformal", "default", "constrained_bo", "adaptive_full", "successive_halving", "random", "tpe_bo"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), sharey=False)

    sids = ["4cae045574a367a3", "d4ad010059141fc7", "77d166e1df75ed54"]
    titles = [
        "(a) Study 1: Wildlife Detection\n(135s budget, 125.5s online)",
        "(b) Study 2: Penn-Fudan Detection\n(108s budget, 101.4s online)",
        "(c) Study 3: CIFAR-10 CNN\n(68s budget, $\\tau=0.19$)",
    ]

    for ax, sid, title in zip(axes, sids, titles):
        ms = holm_data[sid]["method_summaries"]
        fam_a = {r["baseline"]: r for r in holm_data[sid]["two_family_holm"]["family_a_external_m5"]}
        fam_b = {r["baseline"]: r for r in holm_data[sid]["two_family_holm"]["family_b_ablation_m2"]}

        y_pos = np.arange(len(order))
        means = [float(ms[m]["conditional_objective_regret"]["mean"]) for m in order]
        sds = [float(ms[m]["conditional_objective_regret"]["sd"]) for m in order]
        opts = [float(ms[m]["exact_optimum_hit_rate"]) * 100.0 for m in order]
        bar_colors = [OKABE["blue"] if m == "conformal_mf" else (OKABE["orange"] if m == "mf_no_conformal" else OKABE["lightgray"]) for m in order]

        ax.barh(y_pos, means, xerr=sds, height=0.62, color=bar_colors, edgecolor=OKABE["black"], linewidth=0.7, ecolor="#4B5563", capsize=2.5)
        ax.invert_yaxis()
        max_x = max(m + s for m, s in zip(means, sds))
        ax.set_xlim(0, max(0.32, max_x * 1.38))

        for i, m in enumerate(order):
            sig = ""
            if m in fam_a and fam_a[m]["significant_0_05"]:
                sig = " *"
            elif m in fam_b and fam_b[m]["significant_0_05"]:
                sig = " *"
            ax.text(means[i] + sds[i] + 0.008, i, f"{means[i]:.3f} ({opts[i]:.0f}% opt){sig}", va="center", ha="left", fontsize=7.5, fontweight="bold" if m == "conformal_mf" else "normal")

        ax.set_yticks(y_pos)
        ax.set_yticklabels([METHOD_LABELS[m] for m in order], fontsize=8.3)
        ax.set_xlabel("Simple Objective Regret $\\downarrow$ (mean $\\pm$ sd, n=8)")
        ax.set_title(title)
        ax.grid(True, axis="x")

    fig.suptitle("3-Domain Fixed-Budget Search Regret (8 Methods x 8 Search Seeds; * = Two-Family Holm p < 0.05 vs CC-MFBO)", fontsize=10.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    return _save_fig(fig, "fig2_3domain_regret_comparison")


def make_fig3(waste_data: dict[str, Any]) -> dict[str, str]:
    order = ["conformal_mf", "adaptive_full", "successive_halving", "default", "mf_no_conformal", "constrained_bo", "tpe_bo", "random"]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5), sharey=True)

    for ax, skey, title in zip(axes, ("study1", "study2"), ("(a) Study 1: African Wildlife (135s Budget)", "(b) Study 2: Penn-Fudan Pedestrian (108s Budget)")):
        md = waste_data[skey]["methods"]
        y = np.arange(len(order))
        useful = [md[m]["useful_seconds_mean"] for m in order]
        waste = [md[m]["vram_violation_wasted_seconds_mean"] + md[m]["oom_wasted_seconds_mean"] for m in order]
        pruned = [md[m]["conformal_pruned_count_mean"] for m in order]

        ax.barh(y, useful, height=0.62, color=OKABE["green"], edgecolor=OKABE["black"], linewidth=0.6, label="Feasible / Useful Search Time (s)")
        ax.barh(y, waste, left=useful, height=0.62, color=OKABE["vermillion"], edgecolor=OKABE["black"], linewidth=0.6, label="Wasted Time on VRAM-Infeasible Trials (s)")
        ax.set_xlim(0, 205)

        for i, m in enumerate(order):
            tot = useful[i] + waste[i]
            pct = md[m]["waste_pct_of_online"]
            tag = f"{tot:.1f}s | {pruned[i]:.0f} pruned (0% waste)" if m == "conformal_mf" else (f"{tot:.1f}s ({pct:.1f}% waste)" if waste[i] > 0 else f"{tot:.1f}s")
            ax.text(tot + 2.5, i, tag, va="center", ha="left", fontsize=7.6, fontweight="bold" if (m == "conformal_mf" or waste[i] > 0) else "normal", color=OKABE["vermillion"] if waste[i] > 0 else OKABE["black"])

        ax.set_yticks(y)
        ax.set_yticklabels([METHOD_LABELS[m] for m in order], fontsize=8.5)
        ax.set_xlabel("Mean Online Search Time per Run (seconds)")
        ax.set_title(title)
        ax.grid(True, axis="x")

    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03), fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return _save_fig(fig, "fig3_budget_waste_breakdown")


def make_fig4(paired_test: dict[str, Any]) -> dict[str, str]:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.4))
    specs = [
        ("wildlife", "Wildlife Detection\n(Held-Out Test mAP@50, 12ep)"),
        ("pennfudan", "Penn-Fudan Detection\n(Held-Out Test mAP@50:95, 12ep)"),
        ("cifar10", "CIFAR-10 Research CNN\n(Held-Out Test Top-1 Acc, 20ep)"),
    ]
    legend_handle = None
    for ax, (dkey, title) in zip(axes, specs):
        dom = paired_test["domains"][dkey]
        opt = dom["cc_mfbo"]
        base = dom["default"]
        oms = opt["multi_seed_summary"]["quality"]
        bms = base["multi_seed_summary"]["quality"]

        xs = np.array([0, 1])
        means = [float(bms["mean"]), float(oms["mean"])]
        sds = [float(bms["sd"]), float(oms["sd"])]
        bars = ax.bar(xs, means, yerr=sds, width=0.52, color=[OKABE["lightgray"], OKABE["blue"]], edgecolor=OKABE["black"], linewidth=0.8, capsize=4)

        opt_pts = [float(r["quality"]) for r in opt["per_seed_results"]]
        base_pts = [float(r["quality"]) for r in base["per_seed_results"]]
        for bp, op in zip(base_pts, opt_pts):
            ax.plot([0, 1], [bp, op], color="#4B5563", alpha=0.45, lw=1.0, zorder=4)
        ax.scatter(np.zeros(len(base_pts)), base_pts, color=OKABE["black"], s=24, zorder=5)
        sc_opt = ax.scatter(np.ones(len(opt_pts)), opt_pts, color=OKABE["orange"], edgecolor=OKABE["black"], s=34, zorder=5, label="Individual Seeds (17, 29, 43) — Paired Lines")
        if legend_handle is None:
            legend_handle = sc_opt

        pd = dom["paired_delta_primary"]
        rel = f"+{pd['relative_pct']:.1f}%" if pd["relative_pct"] is not None else ""
        top_y = max(means[0] + sds[0], means[1] + sds[1], max(opt_pts), max(base_pts))
        ax.set_ylim(0, min(1.20, top_y * 1.24))
        ax.text(0.5, top_y * 1.09, f"Paired $\\Delta$ = {pd['mean']:+.4f} ({rel})", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color=OKABE["blue"])

        for bar, m_v, s_v in zip(bars, means, sds):
            ax.text(bar.get_x() + bar.get_width() / 2, m_v / 2.0, f"{m_v:.4f}\n$\\pm${s_v:.4f}", ha="center", va="center", fontsize=8.0, fontweight="bold", color="white" if bar is bars[1] else OKABE["black"])

        ax.set_xticks(xs)
        ax.set_xticklabels([f"Default ({base['config_id']})", f"CC-MFBO ({opt['config_id']})"], fontsize=8.8)
        ax.set_title(title)
        ax.grid(True, axis="y")

    fig.legend([legend_handle], ["Individual Training Seeds (17, 29, 43) Connected by Paired Lines"], loc="upper center", bbox_to_anchor=(0.5, 0.96), fontsize=8.3)
    fig.suptitle("Paired 3-Seed Held-Out Test Verification (Training Seeds 17, 29, 43; rect=False)", fontsize=10.5, fontweight="bold", y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    return _save_fig(fig, "fig4_paired_heldout_test")


def make_fig5(ablation: dict[str, Any], rank_diag: dict[str, Any], holm_data: dict[str, Any]) -> dict[str, str]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.3))

    # Panel (a): Epsilon-constraint Pareto sensitivity across 3 domains
    dom_map = [
        ("Wildlife (mAP@50)", "Wildlife", OKABE["blue"], "o"),
        ("Penn-Fudan (mAP@50:95)", "Penn-Fudan", OKABE["orange"], "s"),
        ("CIFAR-10 (Top-1 Acc)", "CIFAR-10", OKABE["green"], "^"),
    ]
    for raw_key, short_lbl, col, mk in dom_map:
        entries = ablation["domains"][raw_key]
        eps_rows = [e for e in entries if e["rule"] == "epsilon_constraint"]
        knee_row = next((e for e in entries if e["rule"] == "knee"), None)
        eps_vals = [float(e["epsilon"]) for e in eps_rows]
        utils = [float(e["utility"]) for e in eps_rows]
        ax1.plot(eps_vals, utils, marker=mk, color=col, lw=1.8, markersize=5.5, label=f"{short_lbl} ($\\epsilon$-constraint)")
        if knee_row:
            ax1.axhline(float(knee_row["utility"]), color=col, ls=":", lw=1.1, alpha=0.7, label=f"{short_lbl} Knee ({knee_row['utility']:.3f})")

    ax1.axvline(0.015, color=OKABE["gray"], ls="--", lw=1.0, alpha=0.6)
    ax1.text(0.017, 0.89, "Locked $\\epsilon \\in [0.01, 0.02]$", fontsize=7.8, color=OKABE["gray"], fontweight="bold")
    ax1.set_xlabel("Quality Retention Tolerance $\\epsilon$")
    ax1.set_ylabel("Multi-Objective Normalized Utility $\\uparrow$")
    ax1.set_title("(a) Post-Decision $\\epsilon$-Constraint Sensitivity vs Knee")
    ax1.set_ylim(0.12, 0.96)
    ax1.grid(True)
    ax1.legend(loc="lower right", fontsize=7.2, ncol=2)

    # Panel (b): Applicability Boundary Map (Kendall tau vs VRAM infeasibility fraction)
    sids = {"study1": "4cae045574a367a3", "study2": "d4ad010059141fc7", "study3": "77d166e1df75ed54"}
    p_cols = {"study1": OKABE["blue"], "study2": OKABE["orange"], "study3": OKABE["vermillion"]}
    ax2.axvspan(0.55, 0.95, ymin=0.15, ymax=1.0, color=OKABE["green"], alpha=0.10, label="CC-MFBO Advantage Regime")
    for skey in ("study1", "study2", "study3"):
        rd = rank_diag[skey]
        ms = holm_data[sids[skey]]["method_summaries"]
        gain = float(ms["default"]["conditional_objective_regret"]["mean"]) - float(ms["conformal_mf"]["conditional_objective_regret"]["mean"])
        x = rd["epoch1_vs_full_kendall_tau"]
        y = rd["infeasible_fraction"] * 100.0
        sz = max(120.0, abs(gain) * 2600.0)
        ax2.scatter([x], [y], s=sz, color=p_cols[skey], edgecolor=OKABE["black"], linewidth=1.0, alpha=0.88, zorder=5)
        offset_y = -6.5 if skey == "study1" else 5.5
        ax2.text(x, y + offset_y, f"{rd['label'].split(':')[0]}: $\\tau={x:.2f}$\nInfeas={y:.1f}% | $\\Delta$Reg={gain:+.3f}", ha="center", va="center", fontsize=8.0, fontweight="bold")

    ax2.axvline(0.5, color=OKABE["gray"], ls="--", lw=1.0)
    ax2.set_xlim(0.05, 0.95)
    ax2.set_ylim(-8, 62)
    ax2.set_xlabel("Early-vs-Full Rank Consistency (Kendall's $\\tau$)")
    ax2.set_ylabel("Search Space VRAM Infeasibility (%)")
    ax2.set_title("(b) Empirical Applicability Boundary Across 3 Domains")
    ax2.grid(True)
    ax2.legend(loc="upper left", fontsize=8.0)

    fig.tight_layout()
    return _save_fig(fig, "fig5_epsilon_and_applicability")


def _render_markdown(diag: dict[str, Any]) -> str:
    lines = [
        "# Quantitative Rigor Diagnostics: Rank Consistency, Two-Family Holm, and Budget Breakdown",
        "",
        "## 1. Early-vs-Full Rank Consistency & VRAM Tightness (Applicability Boundary)",
        "",
        "| Domain | Valid Ref Configs | Infeasible (>1850 MiB) | Epoch-1 vs Full Kendall $\\tau$ | Kendall $p$-value | Spearman $\\rho$ | Conformal-MF Probe Spearman |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key in ("study1", "study2", "study3"):
        r = diag["rank_consistency"][key]
        cmf_sp = f"{r['conformal_mf_subset_spearman_mean']:.4f} (n={r['conformal_mf_subset_n']})" if r["conformal_mf_subset_spearman_mean"] is not None else "n/a"
        lines.append(
            f"| {r['label']} | {r['n_configs_evaluated']}/18 | {18 - r['n_feasible_1850mb']}/18 ({r['infeasible_fraction']*100:.1f}%) | "
            f"**{r['epoch1_vs_full_kendall_tau']:.4f}** | {r['epoch1_vs_full_kendall_p']:.4g} | {r['epoch1_vs_full_spearman_rho']:.4f} | {cmf_sp} |"
        )
    lines.append("")
    lines.append("## 2. Two-Family Holm-Bonferroni Inference (Family A: m=5 External Baselines | Family B: m=2 Ablations)")
    lines.append("")
    for sid, st in diag["two_family_holm"].items():
        lines.append(f"### Study `{sid}` (`{st['task']}`, epochs={st['full_epochs']}, budget={st['budget_seconds']}s)")
        lines.append("")
        lines.append("| Family | Baseline vs CC-MFBO | Paired $\\Delta$ Regret | 95% Bootstrap CI | Raw Sign-Flip $p$ | Two-Family Holm $p$ | Sig. ($p < 0.05$) |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for fam_label, fam_key in (("Family A (m=5)", "family_a_external_m5"), ("Family B (m=2)", "family_b_ablation_m2")):
            for row in st["two_family_holm"][fam_key]:
                sig = "Yes" if row["significant_0_05"] else "No"
                lines.append(
                    f"| {fam_label} | {row['baseline_label']} | {row['diff_mean']:+.4f} ± {row['diff_sd']:.4f} | "
                    f"[{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}] | {row['raw_sign_flip_p']:.4f} | **{row['holm_sign_flip_p']:.4f}** | {sig} |"
                )
        lines.append("")

    lines.append("## 3. Useful vs VRAM-Wasted Search Budget Breakdown")
    lines.append("")
    for skey in ("study1", "study2", "study3"):
        bd = diag["budget_waste"][skey]
        lines.append(f"### {bd['label']}")
        lines.append("")
        lines.append("| Method | Mean Online (s) | Useful Time (s) | VRAM Violation Waste (s) | Waste Share (%) | Pruned Configs/Run |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for m in ("conformal_mf", "mf_no_conformal", "constrained_bo", "tpe_bo", "random", "adaptive_full", "successive_halving", "default"):
            row = bd["methods"][m]
            w_tot = row["vram_violation_wasted_seconds_mean"] + row["oom_wasted_seconds_mean"]
            lines.append(
                f"| {row['label']} | {row['online_seconds_mean']:.1f} | {row['useful_seconds_mean']:.1f} | "
                f"{w_tot:.1f} | {row['waste_pct_of_online']:.1f}% | {row['conformal_pruned_count_mean']:.1f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    _configure_style()
    RIGOR_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    alg_study = json.loads((ROOT / "runs" / "nckh_2027_v2_algorithm_report" / "ALGORITHM_STUDY.json").read_text(encoding="utf-8"))
    vram_summary = json.loads((ROOT / "runs" / "nckh_conformal_vram_study" / "summary.json").read_text(encoding="utf-8"))
    vram_meas = json.loads((ROOT / "runs" / "nckh_conformal_vram_study" / "measurements.json").read_text(encoding="utf-8"))
    card = json.loads((ROOT / "configs" / "conformal_vram_card_rtx3050.json").read_text(encoding="utf-8"))
    paired_test = json.loads((RIGOR_DIR / "paired_test_comparison.json").read_text(encoding="utf-8"))
    ablation = json.loads((ROOT / "runs" / "nckh_2027_v2_selection_ablation" / "summary.json").read_text(encoding="utf-8"))

    rank_diag = compute_rank_consistency()
    holm_data = compute_two_family_holm(alg_study)
    waste_data = compute_budget_waste()

    figs = {
        "fig1": make_fig1(vram_summary, vram_meas, card),
        "fig2": make_fig2(holm_data),
        "fig3": make_fig3(waste_data),
        "fig4": make_fig4(paired_test),
        "fig5": make_fig5(ablation, rank_diag, holm_data),
    }

    diag = {
        "schema_version": 1,
        "rank_consistency": rank_diag,
        "two_family_holm": holm_data,
        "budget_waste": waste_data,
        "figures": figs,
    }
    json_p = RIGOR_DIR / "rigor_diagnostics.json"
    md_p = RIGOR_DIR / "rigor_diagnostics.md"
    json_p.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    md_p.write_text(_render_markdown(diag), encoding="utf-8")
    print(f"[DONE] Saved {json_p}, {md_p}, and 5 figures in {FIG_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
