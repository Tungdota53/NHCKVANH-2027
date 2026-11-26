"""Audited scientific paper exporter (LaTeX booktabs, CSV, Okabe-Ito vector SVG, NCKH Markdown).

Reads immutable run/study evidence, verifies integrity first, and writes derived
publication artifacts to a fresh output directory without mutating any run root.
Supports Vietnamese (vi) and English (en) exports alongside SHA-256 provenance.
"""

import csv
from dataclasses import replace
import html
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from research_pipeline.integrity import sha
from yolo_tuner.pipeline import write_json
from yolo_tuner.research import Objective, digest, nondominated
from .acquisition import (
    _fit_heteroscedastic_gp_loocv,
    _kernel_matrix,
    acquisition_spec,
    configuration_distances,
    full_observations,
)
from .diagnostics import anytime_trajectory, reference_utility_targets, spearman
from .schema import complete, detection_metric_for

OKABE_ITO = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#000000",  # black
    "#F0E442",  # yellow
)
DASH_PATTERNS = ("", "6,3", "2,2", "8,3,2,3")

SENSITIVITY_PROFILES = (
    ("Balanced (mặc định)", (0.60, 0.25, 0.15)),
    ("Ưu tiên độ chính xác (mAP50-first)", (0.80, 0.10, 0.10)),
    ("Thuần chất lượng (Pure Quality)", (1.00, 0.00, 0.00)),
    ("Ưu tiên tốc độ (Time-first)", (0.40, 0.50, 0.10)),
    ("Ưu tiên bộ nhớ (Memory-first)", (0.40, 0.10, 0.50)),
)

SENSITIVITY_PROFILES_BY_LANG = {
    "vi": SENSITIVITY_PROFILES,
    "en": (
        ("Balanced (default)", (0.60, 0.25, 0.15)),
        ("Accuracy-first (mAP50-first)", (0.80, 0.10, 0.10)),
        ("Pure Quality", (1.00, 0.00, 0.00)),
        ("Time-first", (0.40, 0.50, 0.10)),
        ("Memory-first", (0.40, 0.10, 0.50)),
    ),
}

_SVG_STRINGS = {
    "vi": {
        "pareto_title": "Mặt biên Pareto thực nghiệm: {metric} vs Thời gian Worker",
        "pareto_empty": "Chưa có cấu hình hoàn tất full-seeds để vẽ biểu đồ Pareto.",
        "pareto_x": "Thời gian trung bình mỗi lượt Full Worker (giây, thấp hơn tốt hơn)",
        "pareto_y": "{metric} (cao hơn tốt hơn)",
        "pareto_leg_sel": "Cấu hình chọn ★",
        "pareto_leg_par": "Mặt biên Pareto",
        "pareto_leg_feas": "Khả thi",
        "pareto_leg_other": "Chưa đủ / không khả thi",
        "lc_title": "Đường cong hội tụ theo Epoch ({metric} trung bình qua Seeds)",
        "lc_empty": "Chưa có learning curve full-training để hiển thị.",
        "lc_x": "Epoch huấn luyện",
        "any_opt_title": "Quỹ đạo hội tụ Anytime (chỉ tính cấu hình xác nhận đủ seeds)",
        "any_bm_title": "Đường cong hội tụ Anytime theo phương pháp (Mean Best Utility)",
        "any_empty": "Chưa có cấu hình xác nhận full-seeds theo thời gian thực để vẽ Anytime.",
        "any_x": "Thời gian thực thi online tích lũy (giây)",
        "any_y_opt": "Best Utility / Quality đã xác nhận",
        "any_y_bm": "Mean Best Confirmed Utility (cao hơn tốt hơn)",
        "any_best_u": "Best Confirmed Utility",
        "any_best_q": "Best Confirmed Quality",
        "any_ref_opt": "Reference Optimum",
        "any_target_95": "Mục tiêu 95% Utility",
        "pc_title": "Kết quả Test AP theo từng lớp đối tượng (mean ± SD)",
        "pc_x": "Độ chính xác trung bình (AP)",
        "pc_ap50": "Test AP@50",
        "pc_ap95": "Test AP@50:95",
        "pr_title": "Đường cong Precision-Recall trên tập Held-out Test",
        "pr_x": "Recall",
        "pr_y": "Precision",
        "pr_mean": "Trung bình (Mean PR)",
        "cm_title": "Ma trận nhầm lẫn chuẩn hóa trên tập Held-out Test",
        "cm_x": "Nhãn thực tế (True)",
        "cm_y": "Nhãn dự đoán (Predicted)",
        "bm_title": "So sánh Conditional Objective Regret giữa các phương pháp (thấp hơn tốt hơn)",
        "bm_null": "chưa có nghiệm hợp lệ (null regret)",
        "bm_any_fallback": "Tổng hợp hiệu quả tìm kiếm (Regret vs Thời gian online)",
    },
    "en": {
        "pareto_title": "Empirical Pareto Frontier: {metric} vs Worker Wall Time",
        "pareto_empty": "No full-seed confirmed configurations available for Pareto plot.",
        "pareto_x": "Mean Full-Worker Wall Time (seconds, lower is better)",
        "pareto_y": "{metric} (higher is better)",
        "pareto_leg_sel": "Selected ★",
        "pareto_leg_par": "Pareto Frontier",
        "pareto_leg_feas": "Feasible",
        "pareto_leg_other": "Incomplete / Infeasible",
        "lc_title": "Validation Learning Curves by Epoch ({metric}, Seed Mean)",
        "lc_empty": "No full-training learning curves available to render.",
        "lc_x": "Training Epoch",
        "any_opt_title": "Anytime Convergence Trajectory (Full-Seed Confirmed Only)",
        "any_bm_title": "Anytime Convergence by Method (Mean Best Confirmed Utility)",
        "any_empty": "No full-seed confirmed configurations observed along wall clock.",
        "any_x": "Cumulative Active Search Time (seconds)",
        "any_y_opt": "Confirmed Best Utility / Quality",
        "any_y_bm": "Mean Best Confirmed Utility (higher is better)",
        "any_best_u": "Best Confirmed Utility",
        "any_best_q": "Best Confirmed Quality",
        "any_ref_opt": "Reference Optimum",
        "any_target_95": "95% Target Utility",
        "pc_title": "Held-out Test Per-Class AP (mean ± SD)",
        "pc_x": "Average Precision (AP)",
        "pc_ap50": "Test AP@50",
        "pc_ap95": "Test AP@50:95",
        "pr_title": "Precision-Recall Curve on Held-out Test Set",
        "pr_x": "Recall",
        "pr_y": "Precision",
        "pr_mean": "Mean PR",
        "cm_title": "Normalized Confusion Matrix on Held-out Test Set",
        "cm_x": "True Class",
        "cm_y": "Predicted Class",
        "bm_title": "Conditional Objective Regret Across Methods (lower is better)",
        "bm_null": "no verified feasible selection (null regret)",
        "bm_any_fallback": "Search Efficiency Summary (Regret vs Online Time)",
    },
}


def _validate_lang(lang):
    if lang not in {"vi", "en"}:
        raise ValueError(f"Unsupported paper-export language {lang!r}; expected 'vi' or 'en'")
    return lang


def _tex_escape(value):
    text = str(value)
    replacements = (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _fmt(value, digits=4):
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _fmt_tex(value, digits=4, bold=False):
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "---"
    rendered = f"{value:.{digits}f}"
    return rf"\textbf{{{rendered}}}" if bold else rendered


def _write_csv(path, headers, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def hypervolume_3d(rows, objective):
    """Exact 3D dominated hypervolume in [0,1]^3 with nadir (1, 1, 1).

    Axes are normalized minimization losses:
      L_Q = clip(1 - quality_mean, 0, 1)
      L_T = clip(time_mean / time_scale, 0, 1)
      L_M = clip(memory_max / memory_scale, 0, 1)
    """
    points = []
    for row in rows:
        if row.get("state") != "confirmed_feasible":
            continue
        q, t, m = row.get("quality_mean"), row.get("time_mean"), row.get("memory_max")
        if any(v is None or not math.isfinite(v) for v in (q, t, m)):
            continue
        lq = min(1.0, max(0.0, 1.0 - float(q)))
        lt = min(1.0, max(0.0, float(t) / float(objective.time_scale)))
        lm = min(1.0, max(0.0, float(m) / float(objective.memory_scale)))
        if lq < 1.0 and lt < 1.0 and lm < 1.0:
            points.append((lq, lt, lm))
    if not points:
        return 0.0
    z_coords = sorted({p[2] for p in points} | {1.0})
    volume = 0.0
    for idx in range(len(z_coords) - 1):
        z_lo, z_hi = z_coords[idx], z_coords[idx + 1]
        dz = z_hi - z_lo
        if dz <= 0:
            continue
        active_2d = sorted((p[0], p[1]) for p in points if p[2] <= z_lo)
        if not active_2d:
            continue
        area = 0.0
        best_y = 1.0
        skyline = []
        for x, y in active_2d:
            if y < best_y:
                skyline.append((x, y))
                best_y = y
        for s_idx, (x, y) in enumerate(skyline):
            next_x = skyline[s_idx + 1][0] if s_idx + 1 < len(skyline) else 1.0
            if next_x > x and y < 1.0:
                area += (next_x - x) * (1.0 - y)
        volume += area * dz
    return float(min(1.0, max(0.0, volume)))


def surrogate_loocv(protocol, rows):
    """Leave-one-out diagnostic on observed complete configurations (no probe/test leakage)."""
    objective = Objective(**protocol["objective"])
    observations = full_observations(protocol, rows, objective)
    n = len(observations)
    if n < 3:
        return {
            "available": False,
            "observed_complete_configs": n,
            "reason": "need at least 3 complete configurations for leave-one-out surrogate diagnostic",
        }
    spec = acquisition_spec(protocol)
    distances, _ = configuration_distances(protocol)
    indices = [obs["index"] for obs in observations]
    targets = np.asarray([obs["quality_mean"] for obs in observations], dtype=float)

    if spec.get("version", 1) >= 2 or protocol.get("method") == "adaptive_ehvi":
        prior = float(targets.mean())
        sample_var = float(np.var(targets)) if n > 1 else 0.0
        fit = _fit_heteroscedastic_gp_loocv(protocol, spec, distances, observations, indices, targets, prior, sample_var)
        if fit.get("fitted"):
            eta = np.asarray(fit.get("noise_variances") or [0.0] * n, dtype=float)
            k_full = _kernel_matrix(distances, fit["selected_kernel"], fit["selected_ls"])
            gram = k_full[np.ix_(indices, indices)] + np.diag(fit["selected_ridge"] + eta)
            try:
                cf = cho_factor(gram, lower=True, check_finite=False)
                alpha = cho_solve(cf, targets - prior, check_finite=False)
                a_inv = cho_solve(cf, np.eye(n, dtype=float), check_finite=False)
                e_loo = alpha / np.maximum(1e-12, np.diag(a_inv))
                y_loo = np.clip(targets - e_loo, 0.0, 1.0)
                loocv_sp = spearman([float(v) for v in targets], [float(v) for v in y_loo])
            except (np.linalg.LinAlgError, ValueError):
                loocv_sp = None
            return {
                "available": True,
                "observed_complete_configs": n,
                "kernel": fit["selected_kernel"],
                "length_scale": fit["selected_ls"],
                "ridge": fit["selected_ridge"],
                "heteroscedastic": fit["heteroscedastic"],
                "loocv_mae": fit["loocv_mae"],
                "loocv_rmse": fit["loocv_rmse"],
                "loocv_spearman": loocv_sp,
                "scope": "post-run internal LOOCV on complete full-seed configurations; not out-of-domain calibration",
            }

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        kernel = np.exp(-0.5 * (distances / spec["length_scale"]) ** 2)
    preds = []
    for holdout_pos in range(n):
        train_mask = [i for i in range(n) if i != holdout_pos]
        train_idx = [indices[i] for i in train_mask]
        y_train = targets[train_mask]
        prior = float(y_train.mean())
        gram = kernel[np.ix_(train_idx, train_idx)] + spec["kernel_ridge"] * np.eye(len(train_idx))
        try:
            alpha = np.linalg.solve(gram, y_train - prior)
            pred = float(np.clip(prior + kernel[indices[holdout_pos], train_idx] @ alpha, 0.0, 1.0))
        except np.linalg.LinAlgError:
            pred = prior
        preds.append(pred)
    errors = [abs(float(y) - p) for y, p in zip(targets, preds)]
    sq_errors = [(float(y) - p) ** 2 for y, p in zip(targets, preds)]
    return {
        "available": True,
        "observed_complete_configs": n,
        "kernel": "rbf",
        "length_scale": float(spec["length_scale"]),
        "ridge": float(spec["kernel_ridge"]),
        "heteroscedastic": False,
        "loocv_mae": statistics.mean(errors),
        "loocv_rmse": math.sqrt(statistics.mean(sq_errors)),
        "loocv_spearman": spearman([float(v) for v in targets], preds),
        "scope": "post-run internal LOOCV on complete full-seed configurations; not out-of-domain calibration",
    }


def weight_sensitivity_table(rows, objective, lang="vi"):
    feasible = nondominated([r for r in rows if r.get("state") == "confirmed_feasible"])
    profiles = SENSITIVITY_PROFILES_BY_LANG.get(lang, SENSITIVITY_PROFILES)
    table = []
    for label, weights in profiles:
        modified = replace(objective, name="balanced", weights=tuple(weights))
        chosen = max(feasible, key=lambda r: (modified.value(r), r["config_id"]), default=None)
        table.append({
            "profile": label,
            "weights": list(weights),
            "selected_config_id": chosen["config_id"] if chosen else None,
            "quality_mean": chosen["quality_mean"] if chosen else None,
            "time_mean": chosen["time_mean"] if chosen else None,
            "memory_max": chosen["memory_max"] if chosen else None,
            "utility": modified.value(chosen) if chosen else None,
        })
    return table


def _overlap_area(box_a, box_b):
    ix = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    iy = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    return ix * iy


def _oob_area(box, bounds):
    bx0, by0, bx1, by1 = bounds
    total = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    inside_w = max(0.0, min(box[2], bx1) - max(box[0], bx0))
    inside_h = max(0.0, min(box[3], by1) - max(box[1], by0))
    return max(0.0, total - inside_w * inside_h)


def _place_point_labels(items, bounds):
    """Deterministic greedy bounding-box label placement avoiding points and sibling labels.

    items: list of dicts with keys cx, cy, radius, text, priority (lower = placed first).
    bounds: (x0, y0, x1, y1) plot box.
    Returns list of dicts matching input order: {text_x, text_y, anchor, leader, box, item}.
    """
    bx0, by0, bx1, by1 = bounds
    obstacles = [(it["cx"] - it["radius"] - 2.0, it["cy"] - it["radius"] - 2.0,
                  it["cx"] + it["radius"] + 2.0, it["cy"] + it["radius"] + 2.0) for it in items]
    directions = (
        (1.0, -0.8), (1.0, 0.0), (1.0, 0.8), (0.0, -1.0),
        (0.0, 1.0), (-1.0, -0.8), (-1.0, 0.0), (-1.0, 0.8),
    )
    radii = (10.0, 18.0, 28.0, 40.0, 54.0, 70.0)
    placed_boxes = list(obstacles)
    placements = [None] * len(items)

    order = sorted(range(len(items)), key=lambda i: (items[i]["priority"], items[i]["index"]))
    for orig_idx in order:
        it = items[orig_idx]
        cx, cy = it["cx"], it["cy"]
        w = max(20.0, len(it["text"]) * 6.4 + 4.0)
        h = 13.5
        best_score = None
        best_choice = None
        for dist in radii:
            for dx, dy in directions:
                norm = math.hypot(dx, dy) or 1.0
                ux, uy = dx / norm, dy / norm
                if ux > 0.25:
                    anchor = "start"
                    lx0 = cx + ux * dist
                elif ux < -0.25:
                    anchor = "end"
                    lx0 = cx + ux * dist - w
                else:
                    anchor = "middle"
                    lx0 = cx + ux * dist - w / 2.0
                ly_mid = cy + uy * dist
                ly0 = ly_mid - h / 2.0
                lx0 = min(max(lx0, bx0 + 2.0), max(bx0 + 2.0, bx1 - w - 2.0))
                ly0 = min(max(ly0, by0 + 2.0), max(by0 + 2.0, by1 - h - 2.0))
                box = (lx0, ly0, lx0 + w, ly0 + h)
                oob = _oob_area(box, bounds)
                overlap = sum(_overlap_area(box, ob) for ob in placed_boxes)
                center_x = (box[0] + box[2]) / 2.0
                center_y = (box[1] + box[3]) / 2.0
                d_sq = (center_x - cx) ** 2 + (center_y - cy) ** 2
                score = (1000.0 * oob + 100.0 * overlap + 0.05 * d_sq, dist)
                if best_score is None or score < best_score:
                    best_score = score
                    text_x = box[0] + 2.0 if anchor == "start" else (box[2] - 2.0 if anchor == "end" else center_x)
                    text_y = box[1] + 10.5
                    nearest_x = min(max(cx, box[0]), box[2])
                    nearest_y = min(max(cy, box[1]), box[3])
                    disp = math.hypot(nearest_x - cx, nearest_y - cy)
                    best_choice = {
                        "item": it,
                        "text_x": text_x,
                        "text_y": text_y,
                        "anchor": anchor,
                        "box": box,
                        "leader": (cx, cy, nearest_x, nearest_y) if disp > 14.0 else None,
                    }
        placed_boxes.append(best_choice["box"])
        placements[orig_idx] = best_choice
    return placements


def _svg_pareto(rows, selected_id, pareto_ids, metric_label, lang="vi"):
    s = _SVG_STRINGS[lang]
    width, height = 740, 485
    pad_l, pad_r, pad_t, pad_b = 76, 36, 50, 96
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    valid = [r for r in rows if r.get("quality_mean") is not None and r.get("time_mean") is not None]
    title = s["pareto_title"].format(metric=metric_label)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="27" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="15" font-weight="600" fill="#111827">{html.escape(title)}</text>',
    ]
    if not valid:
        lines.append(f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#d1d5db" stroke-width="1"/>')
        lines.append(f'<text x="{width / 2:.1f}" y="{pad_t + plot_h / 2:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="{OKABE_ITO[0]}">{html.escape(s["pareto_empty"])}</text>')
        lines.append("</svg>")
        return "\n".join(lines) + "\n"

    xs = [float(r["time_mean"]) for r in valid]
    ys = [float(r["quality_mean"]) for r in valid]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = max(1e-6, (x_max - x_min) * 1.18 if x_max > x_min else max(1.0, x_max * 0.2))
    y_span = max(1e-6, (y_max - y_min) * 1.18 if y_max > y_min else 0.1)
    x0 = max(0.0, x_min - 0.08 * x_span)
    y0 = max(0.0, y_min - 0.08 * y_span)
    y1 = min(1.02, y0 + y_span)
    y_span = max(1e-6, y1 - y0)

    def px(x):
        return pad_l + (x - x0) / x_span * plot_w

    def py(y):
        return pad_t + plot_h - (y - y0) / y_span * plot_h

    for step in range(5):
        frac = step / 4
        gy = pad_t + plot_h * (1 - frac)
        yv = y0 + frac * y_span
        lines.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l + plot_w}" y2="{gy:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{pad_l - 8}" y="{gy + 4:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#4b5563">{yv:.3f}</text>')
        gx = pad_l + plot_w * frac
        xv = x0 + frac * x_span
        lines.append(f'<line x1="{gx:.1f}" y1="{pad_t}" x2="{gx:.1f}" y2="{pad_t + plot_h}" stroke="#f3f4f6" stroke-width="1"/>')
        lines.append(f'<text x="{gx:.1f}" y="{pad_t + plot_h + 18}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#4b5563">{xv:.1f}s</text>')
    lines.append(f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#374151" stroke-width="1.2"/>')
    lines.append(f'<text x="{pad_l + plot_w / 2:.1f}" y="{pad_t + plot_h + 40}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="600" fill="#1f2937">{html.escape(s["pareto_x"])}</text>')
    y_label = s["pareto_y"].format(metric=metric_label)
    lines.append(f'<text x="20" y="{pad_t + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 20 {pad_t + plot_h / 2:.1f})" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="600" fill="#1f2937">{html.escape(y_label)}</text>')

    pareto_set = set(pareto_ids or [])
    pareto_pts = sorted(
        [(float(r["time_mean"]), float(r["quality_mean"])) for r in valid if r["config_id"] in pareto_set],
        key=lambda item: item[0],
    )
    if len(pareto_pts) >= 2:
        poly = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pareto_pts)
        lines.append(f'<polyline points="{poly}" fill="none" stroke="{OKABE_ITO[0]}" stroke-width="2.2" stroke-dasharray="5,4"/>')

    items = []
    for r in valid:
        cid = r["config_id"]
        cx, cy = px(float(r["time_mean"])), py(float(r["quality_mean"]))
        is_sel = cid == selected_id
        is_par = cid in pareto_set
        feasible = r.get("state") == "confirmed_feasible"
        color = OKABE_ITO[3] if is_sel else OKABE_ITO[0] if is_par else OKABE_ITO[2] if feasible else "#6b7280"
        radius = 7.0 if is_sel else 5.5 if is_par else 4.5
        priority = 0 if is_sel else (1 if is_par else (2 if feasible else 3))
        label = f"c{r['index']:02d}" + (" ★" if is_sel else "")
        items.append({
            "index": r["index"], "cx": cx, "cy": cy, "radius": radius,
            "color": color, "is_sel": is_sel, "text": label, "priority": priority,
        })
        lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius}" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')

    placements = _place_point_labels(items, (pad_l, pad_t, pad_l + plot_w, pad_t + plot_h))
    for pl in placements:
        it = pl["item"]
        if pl["leader"]:
            x1, y1, x2, y2 = pl["leader"]
            lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#9ca3af" stroke-width="0.9"/>')
        weight = "700" if it["is_sel"] else "500"
        lines.append(
            f'<text x="{pl["text_x"]:.1f}" y="{pl["text_y"]:.1f}" text-anchor="{pl["anchor"]}" '
            f'font-family="Segoe UI, Arial, sans-serif" font-size="10.5" font-weight="{weight}" fill="#111827">{html.escape(it["text"])}</text>'
        )

    # Bottom legend strip outside plot box
    leg_y = height - 18
    legend_entries = [
        (OKABE_ITO[3], s["pareto_leg_sel"]),
        (OKABE_ITO[0], s["pareto_leg_par"]),
        (OKABE_ITO[2], s["pareto_leg_feas"]),
        ("#6b7280", s["pareto_leg_other"]),
    ]
    lx = pad_l + 10
    for col, lbl in legend_entries:
        lines.append(f'<circle cx="{lx:.1f}" cy="{leg_y - 4:.1f}" r="5" fill="{col}" stroke="#ffffff" stroke-width="1"/>')
        lines.append(f'<text x="{lx + 10:.1f}" y="{leg_y:.1f}" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#1f2937">{html.escape(lbl)}</text>')
        lx += 34 + len(lbl) * 6.4

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _svg_learning_curves(curve_series, metric_label, lang="vi"):
    s = _SVG_STRINGS[lang]
    max_lbl = max((len(str(lbl)) for lbl, _ in curve_series), default=12)
    pad_l, pad_t, pad_b = 72, 50, 62
    pad_r = max(195, int(math.ceil(40 + max_lbl * 6.8 + 16)))
    legend_needed = pad_t + pad_b + max(180, len(curve_series) * 22 + 24)
    plot_w = 493
    width = pad_l + plot_w + pad_r
    height = max(420, legend_needed)
    plot_h = height - pad_t - pad_b
    title = s["lc_title"].format(metric=metric_label)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="27" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="15" font-weight="600" fill="#111827">{html.escape(title)}</text>',
    ]
    if not curve_series:
        lines.append(f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#d1d5db" stroke-width="1"/>')
        lines.append(f'<text x="{pad_l + plot_w / 2:.1f}" y="{pad_t + plot_h / 2:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="{OKABE_ITO[0]}">{html.escape(s["lc_empty"])}</text>')
        lines.append("</svg>")
        return "\n".join(lines) + "\n"

    all_pts = [pt for _, pts in curve_series for pt in pts]
    max_ep = max(ep for ep, _ in all_pts)
    min_ep = min(ep for ep, _ in all_pts)
    ep_span = max(1, max_ep - min_ep)
    ys = [val for _, val in all_pts]
    y0 = max(0.0, min(ys) - 0.05)
    y1 = min(1.0, max(ys) + 0.05)
    if y1 - y0 < 0.05:
        y0, y1 = max(0.0, y0 - 0.05), min(1.0, y1 + 0.05)
    y_span = max(1e-6, y1 - y0)

    def px(ep):
        return pad_l + (ep - min_ep) / ep_span * plot_w if max_ep > min_ep else pad_l + plot_w / 2

    def py(val):
        return pad_t + plot_h - (val - y0) / y_span * plot_h

    for step in range(5):
        frac = step / 4
        gy = pad_t + plot_h * (1 - frac)
        yv = y0 + frac * y_span
        lines.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l + plot_w}" y2="{gy:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{pad_l - 8}" y="{gy + 4:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#4b5563">{yv:.3f}</text>')
    epochs_sorted = sorted({ep for ep, _ in all_pts})
    stride = max(1, math.ceil(len(epochs_sorted) / 12))
    for ep in epochs_sorted[::stride]:
        gx = px(ep)
        lines.append(f'<text x="{gx:.1f}" y="{pad_t + plot_h + 18}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#4b5563">{ep}</text>')
    lines.append(f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#374151" stroke-width="1.2"/>')
    lines.append(f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 16}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="600" fill="#1f2937">{html.escape(s["lc_x"])}</text>')
    lines.append(f'<text x="20" y="{pad_t + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 20 {pad_t + plot_h / 2:.1f})" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="600" fill="#1f2937">{html.escape(metric_label)}</text>')

    for idx, (label, pts) in enumerate(curve_series):
        color = OKABE_ITO[idx % len(OKABE_ITO)]
        dash = DASH_PATTERNS[(idx // len(OKABE_ITO)) % len(DASH_PATTERNS)]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        poly = " ".join(f"{px(ep):.1f},{py(val):.1f}" for ep, val in pts)
        if len(pts) >= 2:
            lines.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.2"{dash_attr}/>')
        for ep, val in pts:
            lines.append(f'<circle cx="{px(ep):.1f}" cy="{py(val):.1f}" r="3.2" fill="{color}"/>')
        ly = pad_t + 16 + idx * 22
        lines.append(f'<line x1="{pad_l + plot_w + 12}" y1="{ly}" x2="{pad_l + plot_w + 34}" y2="{ly}" stroke="{color}" stroke-width="2.4"{dash_attr}/>')
        lines.append(f'<text x="{pad_l + plot_w + 40}" y="{ly + 4}" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#111827">{html.escape(label)}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _step_value_at(trajectory, t, key="best_utility"):
    val = None
    for pt in trajectory:
        if float(pt["active_seconds"]) <= t + 1e-12:
            val = pt.get(key)
        else:
            break
    return val


def _svg_anytime_optimize(traj_result, lang="vi"):
    s = _SVG_STRINGS[lang]
    max_leg = max(len(s["any_best_u"]), len(s["any_best_q"]))
    pad_l, pad_t, pad_b = 76, 50, 62
    pad_r = max(210, int(math.ceil(40 + max_leg * 6.8 + 16)))
    plot_w = 489
    width = pad_l + plot_w + pad_r
    height = 440
    plot_h = height - pad_t - pad_b
    pts = traj_result.get("trajectory") or []
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="27" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="15" font-weight="600" fill="#111827">{html.escape(s["any_opt_title"])}</text>',
    ]
    if not pts:
        lines.append(f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#d1d5db" stroke-width="1"/>')
        lines.append(f'<text x="{pad_l + plot_w / 2:.1f}" y="{pad_t + plot_h / 2:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12.5" fill="{OKABE_ITO[0]}">{html.escape(s["any_empty"])}</text>')
        lines.append("</svg>")
        return "\n".join(lines) + "\n"

    horizon = max(float(traj_result.get("horizon_seconds") or 1.0), float(pts[-1]["active_seconds"]), 1e-6)
    vals = [float(p["best_utility"]) for p in pts] + [float(p["best_quality"]) for p in pts]
    y_min, y_max = min(vals), max(vals)
    span = max(0.05, (y_max - y_min) * 1.2)
    y0 = y_min - 0.1 * span
    y1 = y0 + span
    y_span = max(1e-6, y1 - y0)

    def px(t):
        return pad_l + (t / horizon) * plot_w

    def py(v):
        return pad_t + plot_h - (v - y0) / y_span * plot_h

    for step in range(5):
        frac = step / 4
        gy = pad_t + plot_h * (1 - frac)
        yv = y0 + frac * y_span
        lines.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l + plot_w}" y2="{gy:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{pad_l - 8}" y="{gy + 4:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#4b5563">{yv:.3f}</text>')
        gx = pad_l + plot_w * frac
        tv = horizon * frac
        lines.append(f'<line x1="{gx:.1f}" y1="{pad_t}" x2="{gx:.1f}" y2="{pad_t + plot_h}" stroke="#f3f4f6" stroke-width="1"/>')
        lines.append(f'<text x="{gx:.1f}" y="{pad_t + plot_h + 18}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#4b5563">{tv:.1f}s</text>')
    lines.append(f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#374151" stroke-width="1.2"/>')
    lines.append(f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 16}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="600" fill="#1f2937">{html.escape(s["any_x"])}</text>')
    lines.append(f'<text x="20" y="{pad_t + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 20 {pad_t + plot_h / 2:.1f})" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="600" fill="#1f2937">{html.escape(s["any_y_opt"])}</text>')

    for series_idx, (key, label, dash) in enumerate((
        ("best_utility", s["any_best_u"], ""),
        ("best_quality", s["any_best_q"], "6,3"),
    )):
        color = OKABE_ITO[series_idx]
        step_coords = []
        for idx, pt in enumerate(pts):
            t_cur = float(pt["active_seconds"])
            v_cur = float(pt[key])
            if idx > 0:
                step_coords.append((t_cur, float(pts[idx - 1][key])))
            step_coords.append((t_cur, v_cur))
        step_coords.append((horizon, float(pts[-1][key])))
        poly = " ".join(f"{px(t):.1f},{py(v):.1f}" for t, v in step_coords)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.4"{dash_attr}/>')
        for pt in pts:
            lines.append(f'<circle cx="{px(float(pt["active_seconds"])):.1f}" cy="{py(float(pt[key])):.1f}" r="3.6" fill="{color}" stroke="#ffffff" stroke-width="1"/>')
        ly = pad_t + 20 + series_idx * 24
        lines.append(f'<line x1="{pad_l + plot_w + 12}" y1="{ly}" x2="{pad_l + plot_w + 34}" y2="{ly}" stroke="{color}" stroke-width="2.5"{dash_attr}/>')
        lines.append(f'<text x="{pad_l + plot_w + 40}" y="{ly + 4}" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#111827">{html.escape(label)}</text>')

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _svg_anytime_benchmark(method_trajectories, summary_rows, ref_opt_u, target_95, horizon, lang="vi"):
    s = _SVG_STRINGS[lang]
    methods = sorted(method_trajectories)
    summary_map = {r["method"]: r for r in summary_rows}
    legend_texts = [s["any_ref_opt"], s["any_target_95"]]
    for m in methods:
        sm = summary_map.get(m) or {}
        auarc_v = sm.get("auarc_mean")
        legend_texts.append(f"{m} (AUARC={auarc_v:.3f})" if isinstance(auarc_v, (int, float)) and math.isfinite(auarc_v) else str(m))
    for r in summary_rows:
        legend_texts.append(str(r.get("method", "")))
    max_leg = max((len(t) for t in legend_texts), default=20)
    pad_l, pad_t, pad_b = 76, 50, 62
    pad_r = max(225, int(math.ceil(40 + max_leg * 6.8 + 16)))
    plot_w = 489
    width = pad_l + plot_w + pad_r
    height = max(440, pad_t + pad_b + max(180, (len(methods) + 2) * 24 + 24))
    plot_h = height - pad_t - pad_b
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="27" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="15" font-weight="600" fill="#111827">{html.escape(s["any_bm_title"])}</text>',
    ]

    # Build step curves per method on union of confirmation timestamps
    series = []
    for method in methods:
        trajs = [t for t in method_trajectories[method] if t]
        if not trajs:
            continue
        t_grid = sorted({float(pt["active_seconds"]) for tr in trajs for pt in tr})
        pts = []
        for t in t_grid:
            vals = [_step_value_at(tr, t, "best_utility") for tr in trajs]
            vals = [float(v) for v in vals if v is not None]
            if vals:
                pts.append((t, statistics.mean(vals)))
        if pts:
            series.append((method, pts))

    if not series:
        # Synthetic study fallback (no raw run directories on disk)
        valid_s = [r for r in summary_rows if r.get("t_mean") is not None]
        if not valid_s:
            lines.append(f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#d1d5db" stroke-width="1"/>')
            lines.append(f'<text x="{pad_l + plot_w / 2:.1f}" y="{pad_t + plot_h / 2:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12.5" fill="{OKABE_ITO[0]}">{html.escape(s["any_empty"])}</text>')
            lines.append("</svg>")
            return "\n".join(lines) + "\n"
        max_t = max(float(r["t_mean"]) for r in valid_s) * 1.15
        regs = [float(r["regret"]) for r in valid_s if r.get("regret") is not None]
        max_r = max(regs, default=0.1) * 1.2
        max_t = max(max_t, 1.0)
        max_r = max(max_r, 0.05)
        for step in range(5):
            frac = step / 4
            gy = pad_t + plot_h * (1 - frac)
            yv = frac * max_r
            lines.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l + plot_w}" y2="{gy:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
            lines.append(f'<text x="{pad_l - 8}" y="{gy + 4:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#4b5563">{yv:.3f}</text>')
            gx = pad_l + plot_w * frac
            tv = frac * max_t
            lines.append(f'<line x1="{gx:.1f}" y1="{pad_t}" x2="{gx:.1f}" y2="{pad_t + plot_h}" stroke="#f3f4f6" stroke-width="1"/>')
            lines.append(f'<text x="{gx:.1f}" y="{pad_t + plot_h + 18}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#4b5563">{tv:.1f}s</text>')
        lines.append(f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#374151" stroke-width="1.2"/>')
        lines.append(f'<text x="20" y="{pad_t + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 20 {pad_t + plot_h / 2:.1f})" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="600" fill="#1f2937">Regret</text>')
        for idx, r in enumerate(valid_s):
            col = OKABE_ITO[idx % len(OKABE_ITO)]
            cx = pad_l + (float(r["t_mean"]) / max_t) * plot_w
            rv = float(r["regret"]) if r.get("regret") is not None else max_r * 0.9
            cy = pad_t + plot_h - (rv / max_r) * plot_h
            lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{col}" stroke="#ffffff" stroke-width="1.2"/>')
            ly = pad_t + 20 + idx * 22
            lines.append(f'<circle cx="{pad_l + plot_w + 18}" cy="{ly - 3}" r="5" fill="{col}"/>')
            lines.append(f'<text x="{pad_l + plot_w + 30}" y="{ly + 1}" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#111827">{html.escape(r["method"])}</text>')
        lines.append(f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 16}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="600" fill="#1f2937">{html.escape(s["bm_any_fallback"])}</text>')
        lines.append("</svg>")
        return "\n".join(lines) + "\n"

    max_h = max(float(horizon or 1.0), max(t for _, pts in series for t, _ in pts), 1e-6)
    all_y = [v for _, pts in series for _, v in pts]
    if ref_opt_u is not None:
        all_y.append(float(ref_opt_u))
    if target_95 is not None:
        all_y.append(float(target_95))
    y_min, y_max = min(all_y), max(all_y)
    span = max(0.05, (y_max - y_min) * 1.2)
    y0 = y_min - 0.1 * span
    y1 = y0 + span
    y_span = max(1e-6, y1 - y0)

    def px(t):
        return pad_l + (t / max_h) * plot_w

    def py(v):
        return pad_t + plot_h - (v - y0) / y_span * plot_h

    for step in range(5):
        frac = step / 4
        gy = pad_t + plot_h * (1 - frac)
        yv = y0 + frac * y_span
        lines.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l + plot_w}" y2="{gy:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{pad_l - 8}" y="{gy + 4:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#4b5563">{yv:.3f}</text>')
        gx = pad_l + plot_w * frac
        tv = max_h * frac
        lines.append(f'<line x1="{gx:.1f}" y1="{pad_t}" x2="{gx:.1f}" y2="{pad_t + plot_h}" stroke="#f3f4f6" stroke-width="1"/>')
        lines.append(f'<text x="{gx:.1f}" y="{pad_t + plot_h + 18}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#4b5563">{tv:.1f}s</text>')
    lines.append(f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#374151" stroke-width="1.2"/>')
    lines.append(f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 16}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="600" fill="#1f2937">{html.escape(s["any_x"])}</text>')
    lines.append(f'<text x="20" y="{pad_t + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 20 {pad_t + plot_h / 2:.1f})" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="600" fill="#1f2937">{html.escape(s["any_y_bm"])}</text>')

    ly_cursor = pad_t + 18
    if ref_opt_u is not None:
        ry = py(float(ref_opt_u))
        lines.append(f'<line x1="{pad_l}" y1="{ry:.1f}" x2="{pad_l + plot_w}" y2="{ry:.1f}" stroke="#000000" stroke-width="1.4" stroke-dasharray="4,4"/>')
        lines.append(f'<line x1="{pad_l + plot_w + 12}" y1="{ly_cursor}" x2="{pad_l + plot_w + 34}" y2="{ly_cursor}" stroke="#000000" stroke-width="1.6" stroke-dasharray="4,4"/>')
        lines.append(f'<text x="{pad_l + plot_w + 40}" y="{ly_cursor + 4}" font-family="Segoe UI, Arial, sans-serif" font-size="10.5" fill="#111827">{html.escape(s["any_ref_opt"])}</text>')
        ly_cursor += 22
    if target_95 is not None:
        ty = py(float(target_95))
        lines.append(f'<line x1="{pad_l}" y1="{ty:.1f}" x2="{pad_l + plot_w}" y2="{ty:.1f}" stroke="{OKABE_ITO[3]}" stroke-width="1.4" stroke-dasharray="2,3"/>')
        lines.append(f'<line x1="{pad_l + plot_w + 12}" y1="{ly_cursor}" x2="{pad_l + plot_w + 34}" y2="{ly_cursor}" stroke="{OKABE_ITO[3]}" stroke-width="1.6" stroke-dasharray="2,3"/>')
        lines.append(f'<text x="{pad_l + plot_w + 40}" y="{ly_cursor + 4}" font-family="Segoe UI, Arial, sans-serif" font-size="10.5" fill="#111827">{html.escape(s["any_target_95"])}</text>')
        ly_cursor += 24

    summary_map = {r["method"]: r for r in summary_rows}
    for idx, (method, pts) in enumerate(series):
        color = OKABE_ITO[idx % len(OKABE_ITO)]
        dash = DASH_PATTERNS[(idx // len(OKABE_ITO)) % len(DASH_PATTERNS)]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        coords = []
        for j, (t_cur, v_cur) in enumerate(pts):
            if j > 0:
                coords.append((t_cur, pts[j - 1][1]))
            coords.append((t_cur, v_cur))
        coords.append((max_h, pts[-1][1]))
        poly = " ".join(f"{px(t):.1f},{py(v):.1f}" for t, v in coords)
        lines.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.3"{dash_attr}/>')
        for t_cur, v_cur in pts:
            lines.append(f'<circle cx="{px(t_cur):.1f}" cy="{py(v_cur):.1f}" r="3.2" fill="{color}"/>')
        sm = summary_map.get(method) or {}
        auarc_v = sm.get("auarc_mean")
        lbl = f"{method} (AUARC={auarc_v:.3f})" if isinstance(auarc_v, (int, float)) and math.isfinite(auarc_v) else method
        lines.append(f'<line x1="{pad_l + plot_w + 12}" y1="{ly_cursor}" x2="{pad_l + plot_w + 34}" y2="{ly_cursor}" stroke="{color}" stroke-width="2.4"{dash_attr}/>')
        lines.append(f'<text x="{pad_l + plot_w + 40}" y="{ly_cursor + 4}" font-family="Segoe UI, Arial, sans-serif" font-size="10.5" fill="#111827">{html.escape(lbl)}</text>')
        ly_cursor += 22
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _svg_per_class_ap50(per_class_rows, lang="vi"):
    s = _SVG_STRINGS[lang]
    max_lbl = max((len(str(r["class_name"])) for r in per_class_rows), default=8)
    has_95 = any(r.get("ap50_95_mean") is not None for r in per_class_rows)
    pad_l = min(400, max(110, int(math.ceil(max_lbl * 7.2 + 28))))
    max_val_len = 13 if has_95 else 5
    pad_r = max(112 if has_95 else 75, int(math.ceil(8 + max_val_len * 6.8 + 16)))
    pad_t, pad_b = 52, 74
    plot_w = max(445, 740 - pad_l - pad_r)
    width = pad_l + plot_w + pad_r
    height = max(320, pad_t + pad_b + max(160, len(per_class_rows) * 44))
    plot_h = height - pad_t - pad_b
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="27" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="15" font-weight="600" fill="#111827">{html.escape(s["pc_title"])}</text>',
    ]
    for step in range(6):
        frac = step / 5
        gx = pad_l + plot_w * frac
        lines.append(f'<line x1="{gx:.1f}" y1="{pad_t}" x2="{gx:.1f}" y2="{pad_t + plot_h}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{gx:.1f}" y="{pad_t + plot_h + 18}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#4b5563">{frac:.1f}</text>')
    lines.append(f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#374151" stroke-width="1.2"/>')
    lines.append(f'<text x="{pad_l + plot_w / 2:.1f}" y="{pad_t + plot_h + 38}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="600" fill="#1f2937">{html.escape(s["pc_x"])}</text>')

    row_h = plot_h / max(1, len(per_class_rows))
    max_chars = max(12, int((pad_l - 22) // 7.2))
    for idx, r in enumerate(per_class_rows):
        y_top = pad_t + idx * row_h
        cy = y_top + row_h * 0.5
        ap50 = max(0.0, min(1.0, float(r.get("ap50_mean") or 0.0)))
        sd50 = float(r.get("ap50_sd") or 0.0)
        cls_txt = str(r["class_name"])
        if len(cls_txt) > max_chars:
            cls_txt = cls_txt[: max(1, max_chars - 3)] + "..."
        lines.append(f'<text x="{pad_l - 10}" y="{cy + 4:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="11.5" font-weight="600" fill="#111827">{html.escape(cls_txt)}</text>')
        if has_95 and r.get("ap50_95_mean") is not None:
            bh = min(14.0, row_h * 0.34)
            y50 = y_top + row_h * 0.16
            y95 = y50 + bh + 3.0
            bw50 = max(2.0, ap50 * plot_w)
            ap95 = max(0.0, min(1.0, float(r["ap50_95_mean"])))
            bw95 = max(2.0, ap95 * plot_w)
            lines.append(f'<rect x="{pad_l}" y="{y50:.1f}" width="{bw50:.1f}" height="{bh:.1f}" rx="2" fill="{OKABE_ITO[0]}"/>')
            lines.append(f'<rect x="{pad_l}" y="{y95:.1f}" width="{bw95:.1f}" height="{bh:.1f}" rx="2" fill="{OKABE_ITO[5]}"/>')
            if sd50 > 0:
                ex0 = pad_l + max(0.0, ap50 - sd50) * plot_w
                ex1 = pad_l + min(1.0, ap50 + sd50) * plot_w
                ey = y50 + bh / 2
                lines.append(f'<line x1="{ex0:.1f}" y1="{ey:.1f}" x2="{ex1:.1f}" y2="{ey:.1f}" stroke="#000000" stroke-width="1.3"/>')
            lines.append(f'<text x="{pad_l + max(bw50, bw95) + 8:.1f}" y="{y50 + bh * 0.82:.1f}" font-family="Segoe UI, Arial, sans-serif" font-size="10.5" fill="#1f2937">{ap50:.3f} / {ap95:.3f}</text>')
        else:
            bh = min(22.0, row_h * 0.56)
            y50 = cy - bh / 2
            bw50 = max(2.0, ap50 * plot_w)
            lines.append(f'<rect x="{pad_l}" y="{y50:.1f}" width="{bw50:.1f}" height="{bh:.1f}" rx="2" fill="{OKABE_ITO[0]}"/>')
            if sd50 > 0:
                ex0 = pad_l + max(0.0, ap50 - sd50) * plot_w
                ex1 = pad_l + min(1.0, ap50 + sd50) * plot_w
                lines.append(f'<line x1="{ex0:.1f}" y1="{cy:.1f}" x2="{ex1:.1f}" y2="{cy:.1f}" stroke="#000000" stroke-width="1.3"/>')
            lines.append(f'<text x="{pad_l + bw50 + 8:.1f}" y="{cy + 4:.1f}" font-family="Segoe UI, Arial, sans-serif" font-size="10.5" fill="#1f2937">{ap50:.3f}</text>')

    leg_y = height - 16
    lines.append(f'<rect x="{pad_l}" y="{leg_y - 10}" width="12" height="10" fill="{OKABE_ITO[0]}"/>')
    lines.append(f'<text x="{pad_l + 18}" y="{leg_y - 1}" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#111827">{html.escape(s["pc_ap50"])}</text>')
    if has_95:
        lines.append(f'<rect x="{pad_l + 120}" y="{leg_y - 10}" width="12" height="10" fill="{OKABE_ITO[5]}"/>')
        lines.append(f'<text x="{pad_l + 138}" y="{leg_y - 1}" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#111827">{html.escape(s["pc_ap95"])}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _downsample_indices(n, max_pts=201):
    if n <= max_pts:
        return list(range(n))
    step = (n - 1) / (max_pts - 1)
    idxs = sorted({min(n - 1, round(i * step)) for i in range(max_pts)})
    if idxs[0] != 0:
        idxs.insert(0, 0)
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    return idxs


def _svg_pr_curve(pr_curve, lang="vi"):
    s = _SVG_STRINGS[lang]
    recall = [float(x) for x in pr_curve["recall_grid"]]
    mean_p = [float(x) for x in pr_curve["mean_precision"]]
    per_cls = pr_curve.get("per_class") or []
    keep = _downsample_indices(min(len(recall), len(mean_p)), 201)
    leg_lens = [len(str(c.get("class_name", idx))) for idx, c in enumerate(per_cls)] + [len(s["pr_mean"])]
    max_leg = max(leg_lens, default=12)
    pad_l, pad_t, pad_b = 72, 50, 62
    pad_r = max(195, int(math.ceil(40 + max_leg * 6.8 + 16)))
    plot_w = 493
    width = pad_l + plot_w + pad_r
    height = max(440, pad_t + pad_b + max(180, (len(per_cls) + 1) * 22 + 24))
    plot_h = height - pad_t - pad_b

    def px(r):
        return pad_l + min(1.0, max(0.0, r)) * plot_w

    def py(p):
        return pad_t + plot_h - min(1.0, max(0.0, p)) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="27" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="15" font-weight="600" fill="#111827">{html.escape(s["pr_title"])}</text>',
    ]
    for step in range(6):
        frac = step / 5
        gy = pad_t + plot_h * (1 - frac)
        gx = pad_l + plot_w * frac
        lines.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l + plot_w}" y2="{gy:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<line x1="{gx:.1f}" y1="{pad_t}" x2="{gx:.1f}" y2="{pad_t + plot_h}" stroke="#f3f4f6" stroke-width="1"/>')
        lines.append(f'<text x="{pad_l - 8}" y="{gy + 4:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#4b5563">{frac:.1f}</text>')
        lines.append(f'<text x="{gx:.1f}" y="{pad_t + plot_h + 18}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#4b5563">{frac:.1f}</text>')
    lines.append(f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#374151" stroke-width="1.2"/>')
    lines.append(f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 16}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="600" fill="#1f2937">{html.escape(s["pr_x"])}</text>')
    lines.append(f'<text x="20" y="{pad_t + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 20 {pad_t + plot_h / 2:.1f})" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="600" fill="#1f2937">{html.escape(s["pr_y"])}</text>')

    ly = pad_t + 18
    for idx, cls_entry in enumerate(per_cls):
        p_vals = cls_entry.get("precision") or []
        if len(p_vals) < len(recall):
            continue
        col = OKABE_ITO[idx % (len(OKABE_ITO) - 1)]  # reserve #000000/#F0E442 carefully; cycle first 7
        if col == "#000000":
            col = OKABE_ITO[5]
        dash = DASH_PATTERNS[(idx // 6) % len(DASH_PATTERNS)]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        poly = " ".join(f"{px(recall[i]):.1f},{py(float(p_vals[i])):.1f}" for i in keep)
        lines.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="1.8" opacity="0.85"{dash_attr}/>')
        lines.append(f'<line x1="{pad_l + plot_w + 12}" y1="{ly}" x2="{pad_l + plot_w + 34}" y2="{ly}" stroke="{col}" stroke-width="2.0"{dash_attr}/>')
        lines.append(f'<text x="{pad_l + plot_w + 40}" y="{ly + 4}" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#111827">{html.escape(str(cls_entry.get("class_name", idx)))}</text>')
        ly += 22

    mean_poly = " ".join(f"{px(recall[i]):.1f},{py(mean_p[i]):.1f}" for i in keep)
    lines.append(f'<polyline points="{mean_poly}" fill="none" stroke="#000000" stroke-width="2.8"/>')
    lines.append(f'<line x1="{pad_l + plot_w + 12}" y1="{ly}" x2="{pad_l + plot_w + 34}" y2="{ly}" stroke="#000000" stroke-width="2.8"/>')
    lines.append(f'<text x="{pad_l + plot_w + 40}" y="{ly + 4}" font-family="Segoe UI, Arial, sans-serif" font-size="11" font-weight="700" fill="#111827">{html.escape(s["pr_mean"])}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _heat_color(val):
    v = min(1.0, max(0.0, float(val)))
    if v <= 0.5:
        t = v / 0.5
        r = round(255 + (86 - 255) * t)
        g = round(255 + (180 - 255) * t)
        b = round(255 + (233 - 255) * t)
    else:
        t = (v - 0.5) / 0.5
        r = round(86 + (0 - 86) * t)
        g = round(180 + (114 - 180) * t)
        b = round(233 + (178 - 233) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _svg_confusion_matrix(cm, lang="vi"):
    s = _SVG_STRINGS[lang]
    raw_labels = [str(lb) for lb in cm["labels"]]
    mat = cm["normalized"]
    n = len(raw_labels)
    cell = max(36, min(68, 360 // max(1, n)))
    max_lbl = max((len(lb) for lb in raw_labels), default=6)
    pad_l = min(340, max(96, int(math.ceil(max_lbl * 7.0 + 28))))
    max_cm_chars = max(12, int((pad_l - 24) // 7.0))
    labels = [lb if len(lb) <= max_cm_chars else lb[: max(1, max_cm_chars - 3)] + "..." for lb in raw_labels]
    eff_max_lbl = max((len(lb) for lb in labels), default=6)
    pad_r = 95
    pad_t = 52
    rotate_x = n > 4 or eff_max_lbl > 7
    pad_b = max(110, int(math.ceil(eff_max_lbl * 4.5 + 48))) if rotate_x else 72
    grid_size = n * cell
    width = max(520, pad_l + grid_size + pad_r)
    height = max(400, pad_t + grid_size + pad_b)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="27" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="15" font-weight="600" fill="#111827">{html.escape(s["cm_title"])}</text>',
    ]
    for r_idx in range(n):
        row = mat[r_idx] if r_idx < len(mat) else []
        y = pad_t + r_idx * cell
        lines.append(f'<text x="{pad_l - 8}" y="{y + cell / 2 + 4:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="10.5" fill="#1f2937">{html.escape(labels[r_idx])}</text>')
        for c_idx in range(n):
            raw_v = row[c_idx] if c_idx < len(row) else 0.0
            val = float(raw_v) if isinstance(raw_v, (int, float)) and math.isfinite(float(raw_v)) else 0.0
            x = pad_l + c_idx * cell
            fill = _heat_color(val)
            txt_col = "#ffffff" if val >= 0.55 else "#111827"
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#e5e7eb" stroke-width="0.8"/>')
            lines.append(f'<text x="{x + cell / 2:.1f}" y="{y + cell / 2 + 4:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="{10 if n <= 8 else 8.5}" fill="{txt_col}">{val:.2f}</text>')

    for c_idx in range(n):
        cx = pad_l + c_idx * cell + cell / 2
        cy = pad_t + grid_size + 16
        if rotate_x:
            lines.append(f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="end" transform="rotate(-35 {cx:.1f} {cy:.1f})" font-family="Segoe UI, Arial, sans-serif" font-size="10.5" fill="#1f2937">{html.escape(labels[c_idx])}</text>')
        else:
            lines.append(f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="10.5" fill="#1f2937">{html.escape(labels[c_idx])}</text>')

    lines.append(f'<rect x="{pad_l}" y="{pad_t}" width="{grid_size}" height="{grid_size}" fill="none" stroke="#374151" stroke-width="1.2"/>')
    lines.append(f'<text x="{pad_l + grid_size / 2:.1f}" y="{height - 14}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="600" fill="#1f2937">{html.escape(s["cm_x"])}</text>')
    lines.append(f'<text x="18" y="{pad_t + grid_size / 2:.1f}" text-anchor="middle" transform="rotate(-90 18 {pad_t + grid_size / 2:.1f})" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="600" fill="#1f2937">{html.escape(s["cm_y"])}</text>')

    # Discrete Okabe-Ito colorbar [0, 1]
    cb_x = pad_l + grid_size + 22
    cb_w = 16
    steps = 10
    step_h = grid_size / steps
    for st in range(steps):
        v = 1.0 - (st + 0.5) / steps
        lines.append(f'<rect x="{cb_x}" y="{pad_t + st * step_h:.1f}" width="{cb_w}" height="{step_h + 0.5:.1f}" fill="{_heat_color(v)}" stroke="none"/>')
    lines.append(f'<rect x="{cb_x}" y="{pad_t}" width="{cb_w}" height="{grid_size}" fill="none" stroke="#374151" stroke-width="1"/>')
    lines.append(f'<text x="{cb_x + cb_w + 6}" y="{pad_t + 8}" font-family="Segoe UI, Arial, sans-serif" font-size="10" fill="#374151">1.0</text>')
    lines.append(f'<text x="{cb_x + cb_w + 6}" y="{pad_t + grid_size / 2 + 4:.1f}" font-family="Segoe UI, Arial, sans-serif" font-size="10" fill="#374151">0.5</text>')
    lines.append(f'<text x="{cb_x + cb_w + 6}" y="{pad_t + grid_size}" font-family="Segoe UI, Arial, sans-serif" font-size="10" fill="#374151">0.0</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _svg_benchmark_bars(summaries, lang="vi"):
    s = _SVG_STRINGS[lang]
    max_m_len = max((len(str(r.get("method", ""))) for r in summaries), default=12)
    max_v_len = max(
        (len(f"{r['regret']:.5f} ({r['verified']})") if r.get("regret") is not None else 0 for r in summaries),
        default=14,
    )
    pad_l = max(195, int(math.ceil(max_m_len * 7.2 + 24)))
    pad_r = max(115, int(math.ceil(8 + max_v_len * 6.8 + 14)))
    pad_t, pad_b = 52, 46
    plot_w = max(435, 740 - pad_l - pad_r)
    width = pad_l + plot_w + pad_r
    height = max(320, 120 + 44 * len(summaries))
    plot_h = height - pad_t - pad_b
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="15" font-weight="600" fill="#111827">{html.escape(s["bm_title"])}</text>',
    ]
    vals = [float(r["regret"]) for r in summaries if r["regret"] is not None]
    max_v = max(vals, default=0.1)
    max_v = max(1e-4, max_v * 1.15)
    row_h = plot_h / max(1, len(summaries))
    lines.append(f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#d1d5db" stroke-width="1"/>')
    color_map = {
        "adaptive_full": OKABE_ITO[0],
        "adaptive_ehvi": OKABE_ITO[1],
        "adaptive_staged_seeds": OKABE_ITO[2],
    }
    for idx, row in enumerate(summaries):
        y = pad_t + idx * row_h + row_h * 0.2
        bh = row_h * 0.6
        lines.append(f'<text x="{pad_l - 10}" y="{y + bh * 0.68:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="11.5" font-weight="600" fill="#111827">{html.escape(row["method"])}</text>')
        if row["regret"] is None:
            lines.append(f'<text x="{pad_l + 10}" y="{y + bh * 0.68:.1f}" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#6b7280">{html.escape(s["bm_null"])}</text>')
        else:
            bw = max(2.0, (float(row["regret"]) / max_v) * plot_w)
            color = color_map.get(row["method"], OKABE_ITO[5] if str(row["method"]).startswith("adaptive_") else "#64748b")
            lines.append(f'<rect x="{pad_l}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3" fill="{color}"/>')
            lines.append(f'<text x="{pad_l + bw + 8:.1f}" y="{y + bh * 0.68:.1f}" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#1f2937">{row["regret"]:.5f} ({row["verified"]})</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _load_test_eval(folder):
    history_path = folder / "test_history.json"
    if history_path.is_file():
        entries = json.loads(history_path.read_text(encoding="utf-8"))
        for entry in reversed(entries):
            if entry.get("status") != "evaluated" or not entry.get("output"):
                continue
            raw_out = Path(entry["output"])
            candidates = [
                raw_out / "evaluation.json",
                folder / "evaluation" / raw_out.name / "evaluation.json",
                folder / "evaluation" / "test_summary.json",
            ]
            for cand in candidates:
                if cand.is_file():
                    return json.loads(cand.read_text(encoding="utf-8")), cand
    fallback = folder / "evaluation" / "test_summary.json"
    if fallback.is_file():
        return json.loads(fallback.read_text(encoding="utf-8")), fallback
    return None, None


def _normalize_per_class(test_eval):
    if not test_eval:
        return []
    if test_eval.get("per_class_summary"):
        return [
            {
                "class_id": c["class_id"],
                "class_name": c["class_name"],
                "ap50_mean": c.get("ap50_mean"),
                "ap50_sd": c.get("ap50_sd"),
                "ap50_95_mean": c.get("ap50_95_mean"),
                "ap50_95_sd": c.get("ap50_95_sd"),
            }
            for c in test_eval["per_class_summary"]
        ]
    if test_eval.get("per_class"):
        return [
            {
                "class_id": c.get("class_id", idx),
                "class_name": c.get("class_name", f"class_{idx}"),
                "ap50_mean": c.get("ap50"),
                "ap50_sd": None,
                "ap50_95_mean": c.get("ap50_95"),
                "ap50_95_sd": None,
            }
            for idx, c in enumerate(test_eval["per_class"])
        ]
    return []


def _write_provenance(out_dir, *, kind, source_path, lang, input_artifacts, protocol_identity):
    exported = {}
    for p in sorted(out_dir.iterdir(), key=lambda item: item.name):
        if p.is_file() and p.name != "PROVENANCE.json":
            exported[p.name] = {
                "path": p.name,
                "sha256": sha(p),
                "bytes": p.stat().st_size,
            }
    prov = {
        "schema_version": 1,
        "kind": kind,
        "source_path": str(source_path),
        "lang": lang,
        "protocol_identity": protocol_identity,
        "input_artifacts": input_artifacts,
        "exported_artifacts": exported,
        "training_started": False,
        "source_run_mutated": False,
    }
    prov["provenance_digest"] = digest(prov)
    prov_path = out_dir / "PROVENANCE.json"
    write_json(prov_path, prov)
    return prov_path, sorted(exported.keys())


def _export_optimize_run(folder, out_dir, lang="vi"):
    from .workflow import audit_run
    folder, protocol, rows, checked_attempts = audit_run(folder)
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    objective = Objective(**protocol["objective"])
    det_metric = detection_metric_for(protocol)
    metric_short = "mAP50" if det_metric == "map50" else "mAP50-95" if det_metric == "map50_95" else "Accuracy" if protocol["task"] == "classification" else "(BLEU4+chrF)/200"
    hv3d = hypervolume_3d(rows, objective)
    loocv = surrogate_loocv(protocol, rows)
    sensitivity = weight_sensitivity_table(rows, objective, lang=lang)
    det_diag = report.get("detection_diagnostics") or {}
    anytime = anytime_trajectory(folder, protocol)
    test_eval, test_eval_path = _load_test_eval(folder)

    # 1. CSV exports
    cand_headers = [
        "index", "config_id", "model", "imgsz", "batch", "accumulation", "precision",
        "optimizer", "lr0", "weight_decay", "freeze", "completed_seeds", "required_seeds",
        "state", f"val_{metric_short}_mean", f"val_{metric_short}_sd", "time_mean_s", "memory_max_mib",
        "utility", "in_pareto", "val_map50_95_at_best_mean", "val_precision_at_best_mean",
        "val_recall_at_best_mean", "val_f1_at_best_mean", "best_epoch_mean",
    ]
    pareto_set = set(report["decision"].get("pareto") or [])
    cand_csv_rows = []
    for r in rows:
        c = r["config"]
        d = det_diag.get(r["config_id"], {})
        cand_csv_rows.append([
            r["index"], r["config_id"], Path(str(c["model"])).name, c["imgsz"], c["batch"],
            c["accumulation"], c["precision"], c["optimizer"], c["lr0"], c["weight_decay"],
            c.get("freeze", 0), len(r["completed_seeds"]), len(r["required_seeds"]), r["state"],
            r["quality_mean"], r["quality_sd"], r["time_mean"], r["memory_max"], r["utility"],
            r["config_id"] in pareto_set,
            (d.get("map50_95_at_best") or {}).get("mean"),
            (d.get("precision_at_best") or {}).get("mean"),
            (d.get("recall_at_best") or {}).get("mean"),
            (d.get("f1_at_best") or {}).get("mean"),
            d.get("best_epoch_mean"),
        ])
    _write_csv(out_dir / "candidates.csv", cand_headers, cand_csv_rows)

    curve_headers = ["index", "config_id", "seed", "epoch", "primary_quality", "map50", "map50_95", "precision", "recall", "f1"]
    curve_rows = []
    curve_series = []
    seeds_required = set(protocol["training_seeds"])
    for r in rows:
        by_epoch = {}
        for run in r["runs"]:
            if run.get("seed") not in seeds_required or not complete(run, protocol["full_epochs"]):
                continue
            for pt in run.get("learning_curve") or []:
                ep = int(pt.get("epoch", 0))
                if det_metric in {"map50", "map50_95"}:
                    primary_val = pt.get(det_metric)
                else:
                    primary_val = next(
                        (pt[k] for k in ("quality", "quality_score", "accuracy") if pt.get(k) is not None),
                        None,
                    )
                if primary_val is not None and math.isfinite(float(primary_val)):
                    by_epoch.setdefault(ep, []).append(float(primary_val))
                curve_rows.append([
                    r["index"], r["config_id"], run["seed"], ep, primary_val,
                    pt.get("map50"), pt.get("map50_95"), pt.get("precision"), pt.get("recall"), pt.get("f1"),
                ])
        if by_epoch:
            pts = [(ep, statistics.mean(vals)) for ep, vals in sorted(by_epoch.items())]
            curve_series.append((f"c{r['index']:02d} ({Path(str(r['config']['model'])).stem})", pts))
    _write_csv(out_dir / "learning_curves.csv", curve_headers, curve_rows)

    acq_history = report.get("acquisition_history") or state.get("acquisition_history") or []
    acq_rows = []
    for entry in acq_history:
        surr = entry.get("surrogate") or {}
        sdiag = entry.get("surrogate_diagnostics") or surr
        chosen_score = (entry.get("scores") or {}).get(str(entry.get("index")), {})
        acq_rows.append([
            entry.get("step"), entry.get("index"), entry.get("reason"),
            surr.get("fitted", False), surr.get("observed_configurations", 0),
            chosen_score.get("score"), chosen_score.get("quality_mean_prediction"),
            chosen_score.get("quality_optimistic"), chosen_score.get("uncertainty"),
            chosen_score.get("remaining_all_seed_cost_seconds"), entry.get("remaining_seconds"),
            sdiag.get("kernel"), sdiag.get("length_scale"), sdiag.get("ridge"),
            sdiag.get("loocv_rmse"), sdiag.get("loocv_mae"), sdiag.get("heteroscedastic"),
        ])
    _write_csv(
        out_dir / "acquisition_history.csv",
        ["step", "chosen_index", "reason", "surrogate_fitted", "observed_full_configs", "acquisition_score",
         "predicted_quality_mean", "optimistic_quality", "uncertainty", "predicted_remaining_cost_s", "budget_remaining_s",
         "surrogate_kernel", "surrogate_length_scale", "surrogate_ridge", "surrogate_loocv_rmse", "surrogate_loocv_mae",
         "surrogate_heteroscedastic"],
        acq_rows,
    )

    _write_csv(
        out_dir / "weight_sensitivity.csv",
        ["profile", "w_quality", "w_time", "w_memory", "selected_config_id", f"val_{metric_short}_mean", "time_mean_s", "memory_max_mib", "utility"],
        [[s["profile"], *s["weights"], s["selected_config_id"], s["quality_mean"], s["time_mean"], s["memory_max"], s["utility"]] for s in sensitivity],
    )

    if test_eval:
        t_summary = test_eval.get("multi_seed_summary") or {}
        _write_csv(
            out_dir / "test_evaluation.csv",
            ["metric", "mean", "sd", "min", "max", "n"],
            [[k, v["mean"], v["sd"], v["min"], v["max"], v["n"]] for k, v in t_summary.items()],
        )
        if test_eval.get("per_class_summary"):
            _write_csv(
                out_dir / "test_per_class.csv",
                ["class_id", "class_name", "ap50_mean", "ap50_sd", "ap50_95_mean", "ap50_95_sd"],
                [[c["class_id"], c["class_name"], c["ap50_mean"], c["ap50_sd"], c["ap50_95_mean"], c["ap50_95_sd"]]
                 for c in test_eval["per_class_summary"]],
            )

    # 2. Vector SVG charts (Okabe-Ito colorblind-safe, deterministic XML)
    selected_id = report["decision"].get("selected_config_id")
    svg_files = ["pareto_front.svg", "learning_curves.svg", "anytime_convergence.svg"]
    (out_dir / "pareto_front.svg").write_text(_svg_pareto(rows, selected_id, pareto_set, f"Val {metric_short}", lang=lang), encoding="utf-8")
    (out_dir / "learning_curves.svg").write_text(_svg_learning_curves(curve_series, f"Val {metric_short}", lang=lang), encoding="utf-8")
    (out_dir / "anytime_convergence.svg").write_text(_svg_anytime_optimize(anytime, lang=lang), encoding="utf-8")

    per_class_norm = _normalize_per_class(test_eval)
    if per_class_norm:
        (out_dir / "per_class_ap50.svg").write_text(_svg_per_class_ap50(per_class_norm, lang=lang), encoding="utf-8")
        svg_files.append("per_class_ap50.svg")
    pr_curve = (test_eval or {}).get("pr_curve") or {}
    if pr_curve.get("available") and pr_curve.get("recall_grid") and pr_curve.get("mean_precision"):
        (out_dir / "pr_curve.svg").write_text(_svg_pr_curve(pr_curve, lang=lang), encoding="utf-8")
        svg_files.append("pr_curve.svg")
    cm = (test_eval or {}).get("confusion_matrix") or {}
    if cm.get("available") and cm.get("labels") and cm.get("normalized") and len(cm["labels"]) == len(cm["normalized"]):
        (out_dir / "confusion_matrix.svg").write_text(_svg_confusion_matrix(cm, lang=lang), encoding="utf-8")
        svg_files.append("confusion_matrix.svg")

    # 3. LaTeX booktabs tables (zero em-dash '—'; always '---')
    best_q = max((r["quality_mean"] for r in rows if r["state"] == "confirmed_feasible" and r["quality_mean"] is not None), default=None)
    best_u = max((r["utility"] for r in rows if r["utility"] is not None), default=None)
    is_en = lang == "en"
    tex = [
        "% Auto-generated by research_pipeline.v2 paper-export (raw-audited, zero manual edits)",
        "% Requires \\usepackage{booktabs} in your LaTeX preamble.",
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        (rf"\caption{{Confirmed candidate configurations ({_tex_escape(protocol['task'])}, primary metric: {_tex_escape(metric_short)}, $E={protocol['full_epochs']}$, $|S|={len(protocol['training_seeds'])}$).}}"
         if is_en else rf"\caption{{Kết quả xác nhận cấu hình huấn luyện ({_tex_escape(protocol['task'])}, metric chính: {_tex_escape(metric_short)}, $E={protocol['full_epochs']}$, $|S|={len(protocol['training_seeds'])}$).}}"),
        r"\label{tab:nckh_candidates}",
        r"\begin{tabular}{llrrlccccc}",
        r"\toprule",
        (rf"ID & Model & Img & Batch/Acc & LR & Seeds & Val {_tex_escape(metric_short)} ($\mu \pm \sigma$) & Time (s) & Memory (MiB) & $U(c)$ \\"
         if is_en else rf"ID & Mô hình & Ảnh & Batch/Acc & LR & Seeds & Val {_tex_escape(metric_short)} ($\mu \pm \sigma$) & Thời gian (s) & Bộ nhớ (MiB) & $U(c)$ \\"),
        r"\midrule",
    ]
    for r in rows:
        c = r["config"]
        is_sel = r["config_id"] == selected_id
        cid_tex = rf"\textbf{{c{r['index']:02d}$^\star$}}" if is_sel else f"c{r['index']:02d}"
        q_bold = best_q is not None and r["quality_mean"] == best_q and r["state"] == "confirmed_feasible"
        u_bold = best_u is not None and r["utility"] == best_u
        q_cell = "---" if r["quality_mean"] is None else (
            rf"\textbf{{{r['quality_mean']:.4f}}} $\pm$ {r['quality_sd']:.4f}" if q_bold else rf"{r['quality_mean']:.4f} $\pm$ {r['quality_sd']:.4f}"
        )
        tex.append(
            f"{cid_tex} & {_tex_escape(Path(str(c['model'])).name)} & {c['imgsz']} & "
            f"{c['batch']}/{c['accumulation']} & {c['lr0']} & {len(r['completed_seeds'])}/{len(r['required_seeds'])} & "
            f"{q_cell} & {_fmt_tex(r['time_mean'], 2)} & {_fmt_tex(r['memory_max'], 1)} & {_fmt_tex(r['utility'], 4, bold=u_bold)} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    if det_diag:
        tex += [
            r"\begin{table}[htbp]",
            r"\centering",
            (r"\caption{Detailed Object Detection validation metrics at Best Checkpoint (mean $\pm$ SD across training seeds).}"
             if is_en else r"\caption{Chỉ số chi tiết Object Detection tại Best Checkpoint trên tập Validation (trung bình $\pm$ độ lệch chuẩn qua các seeds).}"),
            r"\label{tab:nckh_detection_metrics}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"ID & $\mathrm{mAP}_{50}$ & $\mathrm{mAP}_{50:95}$ & Precision & Recall & F1-score & Best Epoch \\",
            r"\midrule",
        ]
        for r in rows:
            d = det_diag.get(r["config_id"])
            if not d:
                continue
            m50 = d["map50_at_best"]
            m95 = d["map50_95_at_best"]
            tex.append(
                f"c{r['index']:02d} & {_fmt_tex(m50['mean'], 4)} $\\pm$ {_fmt_tex(m50['sd'], 4)} & "
                f"{_fmt_tex(m95['mean'], 4)} $\\pm$ {_fmt_tex(m95['sd'], 4)} & "
                f"{_fmt_tex(d['precision_at_best']['mean'], 4)} & {_fmt_tex(d['recall_at_best']['mean'], 4)} & "
                f"{_fmt_tex(d['f1_at_best']['mean'], 4)} & {_fmt_tex(d['best_epoch_mean'], 1)} \\\\"
            )
        tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    tex += [
        r"\begin{table}[htbp]",
        r"\centering",
        (r"\caption{Post-hoc weight sensitivity analysis $(w_A, w_T, w_M)$ over confirmed full-seed feasible configurations.}"
         if is_en else r"\caption{Phân tích độ nhạy trọng số mục tiêu $(w_A, w_T, w_M)$ trên các cấu hình khả thi đã xác nhận full-seeds.}"),
        r"\label{tab:nckh_sensitivity}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        (rf"Preference Profile & Weights $(w_A, w_T, w_M)$ & Selected Config & Val {_tex_escape(metric_short)} & Time (s) & $U(c)$ \\"
         if is_en else rf"Kịch bản ưu tiên & Trọng số $(w_A, w_T, w_M)$ & Cấu hình chọn & Val {_tex_escape(metric_short)} & Thời gian (s) & $U(c)$ \\"),
        r"\midrule",
    ]
    for s in sensitivity:
        w_str = f"({s['weights'][0]:.2f}, {s['weights'][1]:.2f}, {s['weights'][2]:.2f})"
        tex.append(
            f"{_tex_escape(s['profile'])} & {w_str} & {_tex_escape(s['selected_config_id'] or '---')} & "
            f"{_fmt_tex(s['quality_mean'], 4)} & {_fmt_tex(s['time_mean'], 2)} & {_fmt_tex(s['utility'], 4)} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    if test_eval and test_eval.get("per_class_summary"):
        tex += [
            r"\begin{table}[htbp]",
            r"\centering",
            (r"\caption{Held-out Test per-class AP results (mean $\pm$ SD across locked seeds).}"
             if is_en else r"\caption{Kết quả AP theo từng lớp đối tượng trên tập Test độc lập (trung bình $\pm$ độ lệch chuẩn qua các seeds).}"),
            r"\label{tab:nckh_per_class_test}",
            r"\begin{tabular}{rlcc}",
            r"\toprule",
            (r"Class ID & Class Name & Test $\mathrm{AP}_{50}$ ($\mu \pm \sigma$) & Test $\mathrm{AP}_{50:95}$ ($\mu \pm \sigma$) \\"
             if is_en else r"Class ID & Tên lớp & Test $\mathrm{AP}_{50}$ ($\mu \pm \sigma$) & Test $\mathrm{AP}_{50:95}$ ($\mu \pm \sigma$) \\"),
            r"\midrule",
        ]
        for c in test_eval["per_class_summary"]:
            tex.append(
                f"{c['class_id']} & {_tex_escape(c['class_name'])} & "
                f"{_fmt_tex(c['ap50_mean'], 4)} $\\pm$ {_fmt_tex(c['ap50_sd'], 4)} & "
                f"{_fmt_tex(c['ap50_95_mean'], 4)} $\\pm$ {_fmt_tex(c['ap50_95_sd'], 4)} \\\\"
            )
        tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    latency = (test_eval or {}).get("latency") or {}
    if latency.get("available"):
        lat_total = latency.get("total_ms_per_image")
        lat_total_m = lat_total.get("mean") if isinstance(lat_total, dict) else lat_total
        lat_fps = latency.get("fps")
        lat_fps_m = lat_fps.get("mean") if isinstance(lat_fps, dict) else lat_fps
        lat_inf = latency.get("inference_ms")
        lat_inf_m = lat_inf.get("mean") if isinstance(lat_inf, dict) else lat_inf
        tex += [
            r"\begin{table}[htbp]",
            r"\centering",
            (r"\caption{Held-out Test inference latency benchmark (\texttt{rect=False}).}"
             if is_en else r"\caption{Đo lường độ trễ suy luận trên tập Test độc lập (\texttt{rect=False}).}"),
            r"\label{tab:nckh_test_latency}",
            r"\begin{tabular}{llccc}",
            r"\toprule",
            (r"Precision & Batch & Inference (ms/img) & Total (ms/img) & Throughput (FPS) \\"
             if is_en else r"Độ chính xác & Batch & Suy luận (ms/ảnh) & Tổng (ms/ảnh) & Tốc độ (FPS) \\"),
            r"\midrule",
            f"{_tex_escape(latency.get('precision', '---'))} & {latency.get('batch_size', '---')} & "
            f"{_fmt_tex(lat_inf_m, 3)} & {_fmt_tex(lat_total_m, 3)} & {_fmt_tex(lat_fps_m, 2)} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    (out_dir / "tables.tex").write_text("\n".join(tex), encoding="utf-8")

    # 4. Comprehensive NCKH Markdown Summary (keep "Held-out Test" and "LOOCV" in both languages)
    fitted_steps = sum(1 for e in acq_history if (e.get("surrogate") or {}).get("fitted") is True)
    last_diag = next((e.get("surrogate_diagnostics") for e in reversed(acq_history) if e.get("surrogate_diagnostics")), None)
    if is_en:
        md = [
            "# Scientific Experiment Summary Report (NCKH)",
            "",
            f"- **Raw-Audited Evidence Directory ({checked_attempts} worker attempts verified):** `{folder}`",
            f"- **Protocol SHA-256:** `{protocol['protocol_sha256']}`",
            f"- **Locked Task & Primary Metric:** `{protocol['task']}` — **{protocol['quality_definition']}**",
            f"- **Hardware:** `{protocol['hardware'].get('gpu') or protocol['hardware']['device']}` (Memory metric: `{protocol['memory_kind']}`)",
            f"- **Candidate Space & Confirmation Scale:** `{len(rows)}` candidates × `{len(protocol['training_seeds'])}` seeds (`{protocol['training_seeds']}`) × `{protocol['full_epochs']}` epochs",
            f"- **Search Method:** `{protocol['method']}` | **Granted Budget:** `{report['total_granted_budget_seconds']}s` | **Actual Online Session:** `{_fmt(report['online_session_seconds'], 2)}s`",
            f"- **Verification Status:** **`{report['decision']['verification_status']}`**",
            "",
            "## 1. Candidate Configuration Results",
            "",
            f"| Label | ID | Model | Img | Batch/Acc | LR | Freeze | Seeds | State | Val {metric_short} (mean±SD) | T mean (s) | M max (MiB) | Utility U |",
            "|---|---|---|---:|---|---:|---:|---|---|---:|---:|---:|---:|",
        ]
    else:
        md = [
            "# Báo cáo Tổng hợp Thực nghiệm Nghiên cứu Khoa học (NCKH)",
            "",
            f"- **Thư mục chứng cứ gốc (đã raw-audit {checked_attempts} lượt worker):** `{folder}`",
            f"- **Protocol SHA-256:** `{protocol['protocol_sha256']}`",
            f"- **Tác vụ & Metric chính khóa trước:** `{protocol['task']}` — **{protocol['quality_definition']}**",
            f"- **Phần cứng thực nghiệm:** `{protocol['hardware'].get('gpu') or protocol['hardware']['device']}` (Bộ nhớ đo: `{protocol['memory_kind']}`)",
            f"- **Không gian cấu hình & Quy mô xác nhận:** `{len(rows)}` ứng viên × `{len(protocol['training_seeds'])}` seeds (`{protocol['training_seeds']}`) × `{protocol['full_epochs']}` epochs",
            f"- **Phương pháp tìm kiếm:** `{protocol['method']}` | **Ngân sách cấp:** `{report['total_granted_budget_seconds']}s` | **Thời gian phiên online thực tế:** `{_fmt(report['online_session_seconds'], 2)}s`",
            f"- **Trạng thái chứng nhận:** **`{report['decision']['verification_status']}`**",
            "",
            "## 1. Bảng kết quả toàn bộ cấu hình ứng viên",
            "",
            f"| Ký hiệu | ID | Mô hình | Ảnh | Batch/Acc | LR | Freeze | Seeds | Trạng thái | Val {metric_short} (mean±SD) | T mean (s) | M max (MiB) | Utility U |",
            "|---|---|---|---:|---|---:|---:|---|---|---:|---:|---:|---:|",
        ]
    for r in rows:
        c = r["config"]
        mark = "**★**" if r["config_id"] == selected_id else ""
        md.append(
            f"| c{r['index']:02d} {mark} | `{r['config_id']}` | `{Path(str(c['model'])).name}` | {c['imgsz']} | "
            f"{c['batch']}/{c['accumulation']} | {c['lr0']} | {c.get('freeze', 0)} | "
            f"{len(r['completed_seeds'])}/{len(r['required_seeds'])} | `{r['state']}` | "
            f"{_fmt(r['quality_mean'], 4)}±{_fmt(r['quality_sd'], 4)} | {_fmt(r['time_mean'], 2)} | "
            f"{_fmt(r['memory_max'], 1)} | {_fmt(r['utility'], 4)} |"
        )
    if det_diag:
        md += [
            "",
            "## 2. Object Detection Metrics at Best Checkpoint (`best.pt`)" if is_en else "## 2. Bảng chỉ số Object Detection đa chiều tại Best Checkpoint (`best.pt`)",
            "",
            (f"Each seed's `best.pt` checkpoint aligns strictly with maximum `{protocol.get('detection_metric', 'map50')}`."
             if is_en else f"Checkpoint `best.pt` của từng seed được lưu khớp tuyệt đối với cực đại `{protocol.get('detection_metric', 'map50')}` (đã kiểm chứng `best_checkpoint_epoch == best_epoch`)."),
            "",
            ("| Label | mAP@50 (mean±SD) | mAP@50:95 (mean±SD) | Precision | Recall | F1 (mean per-seed) | Best Epoch (mean / range) |"
             if is_en else "| Ký hiệu | mAP@50 (mean±SD) | mAP@50:95 (mean±SD) | Precision | Recall | F1 (mean per-seed) | Best Epoch (mean / range) |"),
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for r in rows:
            d = det_diag.get(r["config_id"])
            if not d:
                continue
            ep_info = f"{d['best_epoch_mean']:.1f} ({d['best_epoch_min']}..{d['best_epoch_max']})" if d["best_epoch_mean"] is not None else "—"
            md.append(
                f"| c{r['index']:02d} | **{_fmt(d['map50_at_best']['mean'], 4)}**±{_fmt(d['map50_at_best']['sd'], 4)} | "
                f"{_fmt(d['map50_95_at_best']['mean'], 4)}±{_fmt(d['map50_95_at_best']['sd'], 4)} | "
                f"{_fmt(d['precision_at_best']['mean'], 4)} | {_fmt(d['recall_at_best']['mean'], 4)} | "
                f"{_fmt(d['f1_at_best']['mean'], 4)} | {ep_info} |"
            )

    md += [
        "",
        "## 3. Multi-Objective Pareto Front, 3D Hypervolume & Surrogate LOOCV Diagnostics" if is_en else "## 3. Phân tích Đa mục tiêu (Pareto Front & 3D Hypervolume) và Chẩn đoán Surrogate",
        "",
        (f"- **Observed Pareto Front:** `{report['decision']['pareto']}`"
         if is_en else f"- **Tập cấu hình không bị trội (Observed Pareto Front):** `{report['decision']['pareto']}`"),
        (f"- **3D Dominated Hypervolume (nadir `[1, 1, 1]`, `T_ref={objective.time_scale}s, M_ref={objective.memory_scale}MiB`):** **`{hv3d:.6f}`**"
         if is_en else f"- **3D Dominated Hypervolume (chuẩn hóa theo nadir `[1, 1, 1]` và `T_ref={objective.time_scale}s, M_ref={objective.memory_scale}MiB`):** **`{hv3d:.6f}`**"),
        (f"- **Total Acquisition Steps:** `{len(acq_history)}` (fitted steps `surrogate.fitted=true`: **`{fitted_steps}`**)"
         if is_en else f"- **Số bước Acquisition tổng cộng:** `{len(acq_history)}` (trong đó số bước thực sự fit Kernel Ridge `surrogate.fitted=true`: **`{fitted_steps}`**)"),
        (f"- **Confirmed Anytime Trajectory Points:** `{len(anytime['trajectory'])}` (horizon `{anytime['horizon_seconds']:.2f}s`)"
         if is_en else f"- **Số điểm quỹ đạo Anytime đã xác nhận full-seeds:** `{len(anytime['trajectory'])}` (chân trời `{anytime['horizon_seconds']:.2f}s`)"),
    ]
    if last_diag:
        md.append(
            f"- **Latest Online Surrogate Fit:** kernel=`{last_diag.get('kernel')}`, ls=`{last_diag.get('length_scale')}`, "
            f"ridge=`{last_diag.get('ridge')}`, heteroscedastic=`{last_diag.get('heteroscedastic')}`, "
            f"LOOCV RMSE=`{_fmt(last_diag.get('loocv_rmse'), 4)}`"
        )
    if loocv["available"]:
        md.append(
            (f"- **Internal Surrogate LOOCV (`{loocv['observed_complete_configs']}` complete configs, kernel=`{loocv.get('kernel', 'rbf')}`):** "
             if is_en else f"- **Chẩn đoán nội bộ Surrogate (LOOCV trên `{loocv['observed_complete_configs']}` cấu hình hoàn tất, kernel=`{loocv.get('kernel', 'rbf')}`):** ")
            + f"MAE = `{_fmt(loocv['loocv_mae'], 4)}`, RMSE = `{_fmt(loocv['loocv_rmse'], 4)}`, Spearman $\\rho$ = `{_fmt(loocv['loocv_spearman'], 4)}` "
            + ("*(descriptive in-domain LOOCV only; not an out-of-domain prediction interval)*."
               if is_en else "*(chỉ số mô tả hậu nghiệm trên tập đã đo, không phải khoảng dự đoán hiệu chuẩn ngoài miền)*.")
        )
    else:
        md.append(
            f"- **Internal Surrogate LOOCV:** Unavailable (`{loocv['reason']}`)."
            if is_en else f"- **Chẩn đoán nội bộ Kernel Ridge (LOOCV):** Chưa khả dụng (`{loocv['reason']}`)."
        )

    md += [
        "",
        "## 4. Post-hoc Weight Sensitivity Analysis" if is_en else "## 4. Phân tích Độ nhạy Trọng số Mục tiêu (Post-hoc Sensitivity)",
        "",
        ("Post-hoc analysis keeps all empirical measurements unchanged and never alters the locked primary selection:"
         if is_en else "Phân tích dưới đây giữ nguyên kết quả đo thực tế, không chạy huấn luyện mới và không thay đổi quyết định chính đã khóa:"),
        "",
        (f"| Profile | Weights (A, T, M) | Selected Config | Val {metric_short} | T mean (s) | M max (MiB) | Utility U |"
         if is_en else f"| Kịch bản | Trọng số (A, T, M) | Cấu hình chọn | Val {metric_short} | T mean (s) | M max (MiB) | Utility U |"),
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for s in sensitivity:
        md.append(
            f"| {s['profile']} | `{s['weights']}` | `{s['selected_config_id'] or '—'}` | "
            f"{_fmt(s['quality_mean'], 4)} | {_fmt(s['time_mean'], 2)} | {_fmt(s['memory_max'], 1)} | {_fmt(s['utility'], 4)} |"
        )

    if test_eval:
        gap = test_eval.get("validation_vs_test") or {}
        md += [
            "",
            "## 5. Independent Held-out Test Generalization Evaluation" if is_en else "## 5. Đánh giá Tổng quát hóa trên Tập Test Độc lập (Held-out Test)",
            "",
            (f"- **Evaluated Seeds:** `{test_eval.get('evaluated_seeds')}` (`test_used_for_selection = False`)"
             if is_en else f"- **Seeds đã đánh giá trên Test:** `{test_eval.get('evaluated_seeds')}` (`test_used_for_selection = False`)"),
            f"- **Validation {metric_short} (mean±SD):** `{_fmt(gap.get('validation_mean'), 4)} ± {_fmt(gap.get('validation_sd'), 4)}`",
            f"- **Held-out Test {metric_short} (mean±SD):** **`{_fmt(gap.get('test_mean'), 4)} ± {_fmt(gap.get('test_sd'), 4)}`**",
            f"- **Generalization Gap (`Test - Val`):** `{_fmt(gap.get('generalization_gap_test_minus_val'), 4)}`" if is_en else f"- **Độ lệch tổng quát hóa (`Test - Val`):** `{_fmt(gap.get('generalization_gap_test_minus_val'), 4)}`",
        ]
        t_sum = test_eval.get("multi_seed_summary") or {}
        if "map50" in t_sum:
            md.append(
                ("- **Companion Test Detection Metrics:** " if is_en else "- **Các chỉ số Test Detection đi kèm:** ")
                + f"`mAP@50 = {_fmt(t_sum['map50']['mean'], 4)}`, "
                + f"`mAP@75 = {_fmt((t_sum.get('map75') or {}).get('mean'), 4)}`, "
                + f"`mAP@50:95 = {_fmt((t_sum.get('map50_95') or {}).get('mean'), 4)}`, "
                + f"`Precision = {_fmt((t_sum.get('precision') or {}).get('mean'), 4)}`, "
                + f"`Recall = {_fmt((t_sum.get('recall') or {}).get('mean'), 4)}`, "
                + f"`F1 (mean per-seed) = {_fmt((t_sum.get('f1') or {}).get('mean'), 4)}`"
            )
        if latency.get("available"):
            lat_total = latency.get("total_ms_per_image")
            lat_total_m = lat_total.get("mean") if isinstance(lat_total, dict) else lat_total
            lat_fps = latency.get("fps")
            lat_fps_m = lat_fps.get("mean") if isinstance(lat_fps, dict) else lat_fps
            md.append(
                f"- **Held-out Test Inference Latency (`rect=False`):** `{_fmt(lat_total_m, 3)} ms/img` (`{_fmt(lat_fps_m, 2)} FPS`, precision=`{latency.get('precision')}`, batch=`{latency.get('batch_size')}`)"
            )
        f1_curve = test_eval.get("f1_curve") or {}
        if f1_curve.get("available"):
            md.append(
                f"- **F1-Confidence Curve Peak:** `max_mean_F1 = {_fmt(f1_curve.get('max_mean_f1'), 4)}` at `conf = {_fmt(f1_curve.get('best_confidence'), 3)}`"
            )
        if test_eval.get("per_class_summary"):
            md += [
                "",
                "### Held-out Test Per-Class AP" if is_en else "### Kết quả AP theo từng lớp đối tượng trên tập Test",
                "",
                ("| Class ID | Class Name | Test AP@50 (mean±SD) | Test AP@50:95 (mean±SD) |"
                 if is_en else "| Class ID | Tên lớp | Test AP@50 (mean±SD) | Test AP@50:95 (mean±SD) |"),
                "|---:|---|---:|---:|",
            ]
            for c in test_eval["per_class_summary"]:
                md.append(
                    f"| {c['class_id']} | `{c['class_name']}` | **{_fmt(c['ap50_mean'], 4)}**±{_fmt(c['ap50_sd'], 4)} | "
                    f"{_fmt(c['ap50_95_mean'], 4)}±{_fmt(c['ap50_95_sd'], 4)} |"
                )

    csv_files = ["`candidates.csv`", "`learning_curves.csv`", "`acquisition_history.csv`", "`weight_sensitivity.csv`"]
    if test_eval:
        csv_files.append("`test_evaluation.csv`")
        if test_eval.get("per_class_summary"):
            csv_files.append("`test_per_class.csv`")
    svg_md = ", ".join(f"`{name}`" for name in svg_files)
    if is_en:
        md += [
            "",
            "## 6. Threats to Validity",
            "",
            "- Optimality claims apply strictly within the **locked finite candidate space, epoch horizon, training seeds, dataset split, and hardware**; they do not extrapolate universally across unseen GPUs or datasets.",
            "- Standard deviation (SD) across training seeds captures initialization variance, not a cross-domain statistical confidence interval.",
            "- Surrogate uncertainty guides search exploration heuristically and is not a calibrated Bayesian posterior.",
            "",
            "## 7. Exported Publication Artifacts",
            "",
            "- `tables.tex`: LaTeX `booktabs` tables ready for manuscript inclusion.",
            f"- {', '.join(csv_files)}: UTF-8 tabular datasets.",
            f"- {svg_md}: Colorblind-safe Okabe-Ito vector SVG figures.",
            "- `PAPER_EVIDENCE.json`, `PROVENANCE.json`: Audited metrics and SHA-256 artifact provenance.",
            "",
        ]
    else:
        md += [
            "",
            "## 6. Giới hạn Hiệu lực Khoa học (Threats to Validity — cần nêu trong bài NCKH)",
            "",
            "- Kết luận tối ưu chỉ áp dụng trên **không gian ứng viên hữu hạn, chân trời epoch, tập seeds, dữ liệu và phần cứng đã khóa**; không khái quát hóa thành tối ưu toàn cục cho mọi GPU hoặc mọi tập dữ liệu.",
            "- Độ lệch chuẩn (SD) qua các training seeds phản ánh biến thiên thực nghiệm giữa các khởi tạo ngẫu nhiên, không phải khoảng tin cậy thống kê trên quần thể tập dữ liệu.",
            "- Uncertainty của Surrogate là đại lượng dẫn hướng tìm kiếm (heuristic exploration), không phải hậu nghiệm Bayes đã hiệu chuẩn xác suất.",
            "",
            "## 7. Danh mục Tệp Xuất bản Đi kèm",
            "",
            "- `tables.tex`: Bảng LaTeX chuẩn `booktabs` sẵn sàng chèn vào bài báo / báo cáo NCKH.",
            f"- {', '.join(csv_files)}: Dữ liệu bảng chuẩn UTF-8.",
            f"- {svg_md}: Biểu đồ vector chuẩn Okabe-Ito thân thiện người mù màu.",
            "- `PAPER_EVIDENCE.json`, `PROVENANCE.json`: Chứng cứ số liệu và bảng kiểm SHA-256 toàn bộ tệp xuất bản.",
            "",
        ]
    summary_md_path = out_dir / "PAPER_SUMMARY.md"
    summary_md_path.write_text("\n".join(md), encoding="utf-8")

    evidence = {
        "schema_version": 1,
        "kind": "optimize_run",
        "lang": lang,
        "source_path": str(folder),
        "protocol_sha256": protocol["protocol_sha256"],
        "checked_worker_attempts": checked_attempts,
        "task": protocol["task"],
        "detection_metric": det_metric,
        "quality_definition": protocol["quality_definition"],
        "verification_status": report["decision"]["verification_status"],
        "selected_config_id": selected_id,
        "hypervolume_3d": hv3d,
        "surrogate_loocv": loocv,
        "fitted_acquisition_steps": fitted_steps,
        "anytime_confirmed_count": anytime["confirmed_feasible_count"],
        "weight_sensitivity": sensitivity,
        "test_evaluation_included": test_eval is not None,
        "test_latency_included": bool(latency.get("available")),
        "test_curves_included": bool(pr_curve.get("available") or (cm.get("available"))),
        "training_started": False,
        "source_run_mutated": False,
    }
    evidence["sha256"] = digest(evidence)
    write_json(out_dir / "PAPER_EVIDENCE.json", evidence)

    input_artifacts = []
    for fname in ("protocol.json", "state.json", "events.jsonl", "report.json", "decision.locked.json", "dataset.json", "test_history.json"):
        p = folder / fname
        if p.is_file():
            input_artifacts.append({"role": fname, "path": str(p.resolve()), "sha256": sha(p)})
    if test_eval_path and test_eval_path.is_file():
        input_artifacts.append({"role": "test_evaluation.json", "path": str(test_eval_path.resolve()), "sha256": sha(test_eval_path)})
    prov_path, exported_files = _write_provenance(
        out_dir,
        kind="optimize_run",
        source_path=folder,
        lang=lang,
        input_artifacts=input_artifacts,
        protocol_identity={
            "protocol_sha256": protocol["protocol_sha256"],
            "dataset_sha256": protocol.get("dataset_sha256"),
            "subset_sha256": protocol.get("subset_sha256"),
            "source_files_digest": digest(protocol.get("source_files") or {}),
        },
    )
    return {
        "kind": "optimize_run",
        "lang": lang,
        "source_path": str(folder),
        "output_directory": str(out_dir),
        "summary_markdown": str(summary_md_path),
        "latex_tables": str(out_dir / "tables.tex"),
        "evidence_json": str(out_dir / "PAPER_EVIDENCE.json"),
        "provenance_json": str(prov_path),
        "exported_files": exported_files,
    }


def _export_benchmark_or_algorithm_study(folder, out_dir, lang="vi"):
    is_en = lang == "en"
    if (folder / "ALGORITHM_STUDY.json").is_file():
        data = json.loads((folder / "ALGORITHM_STUDY.json").read_text(encoding="utf-8"))
        expected = digest({k: v for k, v in data.items() if k != "sha256"})
        if data.get("sha256") != expected:
            raise ValueError("ALGORITHM_STUDY.json checksum mismatch")
        kind = "algorithm_study"
        studies = data["studies"]
        input_artifacts = [{"role": "ALGORITHM_STUDY.json", "path": str((folder / "ALGORITHM_STUDY.json").resolve()), "sha256": sha(folder / "ALGORITHM_STUDY.json")}]
        protocol_identity = {
            "analysis_manifest_sha256": data.get("analysis_manifest_sha256"),
            "study_ids": [s.get("study_id") for s in studies],
        }
    else:
        from . import benchmark
        manifest = json.loads((folder / "benchmark_protocol.json").read_text(encoding="utf-8"))
        if manifest.get("sha256") != digest({k: v for k, v in manifest.items() if k != "sha256"}):
            raise ValueError("Benchmark protocol checksum mismatch")
        boundary = folder / "decisions.locked.json"
        locked = json.loads(boundary.read_text(encoding="utf-8"))
        reference = benchmark._single_run(folder / "reference")
        report = benchmark.compare(locked["runs"], str(reference))
        kind = "benchmark_study"
        studies = [{
            "study_id": digest(str(folder))[:16],
            "study": str(folder),
            "reference": str(reference),
            "search_budget_seconds": locked.get("runs") and report["outcomes"][0].get("budget_seconds") if report.get("outcomes") else None,
            "outcomes": report["outcomes"],
            "summaries_raw": report["summaries"],
            "reference_cost_seconds": report["reference_cost_seconds"],
        }]
        input_artifacts = [
            {"role": "benchmark_protocol.json", "path": str((folder / "benchmark_protocol.json").resolve()), "sha256": sha(folder / "benchmark_protocol.json")},
            {"role": "decisions.locked.json", "path": str(boundary.resolve()), "sha256": sha(boundary)},
        ]
        protocol_identity = {
            "benchmark_sha256": manifest["sha256"],
            "methods": manifest.get("methods"),
            "search_seeds": manifest.get("search_seeds"),
        }

    outcome_rows = []
    summary_rows = []
    svg_bars = []
    method_trajectories = {}
    ref_opt_u = None
    target_95 = None
    horizon = None
    has_vram_safety_global = any(
        out.get("vram_violation_attempt_count") is not None
        for st in studies for out in st.get("outcomes", [])
    )

    tex = [
        "% Auto-generated by research_pipeline.v2 paper-export (benchmark/algorithm study)",
        "% Requires \\usepackage{booktabs} in your LaTeX preamble.",
        "",
    ]
    if is_en:
        md = [
            "# Algorithm Comparison Benchmark — Publication Summary",
            "",
            f"- **Evidence Source:** `{folder}` (`{kind}`)",
            "- **Protocol Principle:** All online search decisions were locked before running the exhaustive independent reference; failed searches retain `regret = null`, never zero.",
            "",
        ]
    else:
        md = [
            "# Báo cáo Đối chứng Thuật toán — Xuất bản NCKH",
            "",
            f"- **Nguồn chứng cứ:** `{folder}` (`{kind}`)",
            "- **Nguyên tắc:** Mọi quyết định tìm kiếm online đã khóa trước khi chạy tập đối chứng vét cạn (exhaustive independent reference); các lượt không tìm được cấu hình khả thi giữ `regret = null`, không gán bằng `0`.",
            "",
        ]

    for st in studies:
        sid = st["study_id"]
        ref_path = st.get("reference")
        if ref_opt_u is None or target_95 is None:
            if st.get("reference_optimal_utility") is not None:
                ref_opt_u = float(st["reference_optimal_utility"])
                if st.get("target_utility_95") is not None:
                    target_95 = float(st["target_utility_95"])
            if (ref_opt_u is None or target_95 is None) and ref_path and Path(ref_path).exists():
                u_opt, t_95 = reference_utility_targets(ref_path)
                if ref_opt_u is None and u_opt is not None:
                    ref_opt_u = u_opt
                if target_95 is None and t_95 is not None:
                    target_95 = t_95
        for out in st.get("outcomes", []):
            row_out = [
                sid, out["method"], out["search_seed"], out["selected"], out["feasible"],
                out.get("objective_regret"), out.get("quality_regret"),
                out.get("exact_optimum_hit"), out.get("pareto_hit"),
                out.get("online_seconds"), out.get("overrun_seconds"),
                out.get("auarc"), out.get("time_to_target_95"),
            ]
            if has_vram_safety_global:
                row_out += [
                    out.get("vram_violation_attempt_count"),
                    out.get("vram_violation_wasted_seconds"),
                    out.get("conformal_pruned_count"),
                ]
            outcome_rows.append(row_out)
            traj_pts = out.get("anytime_trajectory")
            if traj_pts is None and out.get("run"):
                run_p = Path(out["run"])
                if (run_p / "protocol.json").is_file() and (run_p / "state.json").is_file() and (run_p / "events.jsonl").is_file():
                    t_res = anytime_trajectory(run_p, reference=ref_path if ref_path and Path(ref_path).exists() else None)
                    traj_pts = t_res["trajectory"]
                    if ref_opt_u is None and t_res.get("reference_optimal_utility") is not None:
                        ref_opt_u = t_res["reference_optimal_utility"]
                        target_95 = t_res["target_utility_95"]
                    if horizon is None or t_res["horizon_seconds"] > horizon:
                        horizon = t_res["horizon_seconds"]
            if traj_pts:
                method_trajectories.setdefault(out["method"], []).append(traj_pts)
            if out.get("budget_seconds") is not None:
                horizon = max(float(out["budget_seconds"]), horizon or 0.0)

        raw_list = st.get("summaries_raw")
        if raw_list is None:
            raw_list = []
            for m in st.get("method_summaries", []):
                reg = m["conditional_objective_regret"]
                m_outs = [o for o in st.get("outcomes", []) if o.get("method") == m["method"]]
                q_regs = [float(o["quality_regret"]) for o in m_outs if isinstance(o.get("quality_regret"), (int, float)) and math.isfinite(o["quality_regret"])]
                opt_hits = [bool(o["exact_optimum_hit"]) for o in m_outs if o.get("exact_optimum_hit") is not None]
                par_hits = [bool(o["pareto_hit"]) for o in m_outs if o.get("pareto_hit") is not None]
                auarcs = [float(o["auarc"]) for o in m_outs if isinstance(o.get("auarc"), (int, float)) and math.isfinite(o["auarc"])]
                ttts = [float(o["time_to_target_95"]) for o in m_outs if isinstance(o.get("time_to_target_95"), (int, float)) and math.isfinite(o["time_to_target_95"])]
                v_viols = [float(o["vram_violation_attempt_count"]) for o in m_outs if isinstance(o.get("vram_violation_attempt_count"), (int, float)) and math.isfinite(o["vram_violation_attempt_count"])]
                v_waste = [float(o["vram_violation_wasted_seconds"]) for o in m_outs if isinstance(o.get("vram_violation_wasted_seconds"), (int, float)) and math.isfinite(o["vram_violation_wasted_seconds"])]
                c_prun = [float(o["conformal_pruned_count"]) for o in m_outs if isinstance(o.get("conformal_pruned_count"), (int, float)) and math.isfinite(o["conformal_pruned_count"])]
                raw_list.append({
                    "method": m["method"],
                    "feasible_count": m["verified_selection_count"],
                    "n": m["search_repetitions"],
                    "conditional_regret_mean": reg["mean"],
                    "conditional_regret_sd": reg["sd"],
                    "conditional_quality_regret_mean": statistics.mean(q_regs) if q_regs else None,
                    "exact_optimum_hit_rate": statistics.mean(opt_hits) if opt_hits else None,
                    "pareto_hit_rate": statistics.mean(par_hits) if par_hits else None,
                    "auarc_mean": statistics.mean(auarcs) if auarcs else None,
                    "time_to_target_95_mean": statistics.mean(ttts) if ttts else None,
                    "online_seconds_mean": m["online_seconds"]["mean"],
                    "vram_violation_attempts_mean": statistics.mean(v_viols) if v_viols else None,
                    "vram_violation_wasted_seconds_mean": statistics.mean(v_waste) if v_waste else None,
                    "conformal_pruned_mean": statistics.mean(c_prun) if c_prun else None,
                })

        has_auarc = any(s.get("auarc_mean") is not None for s in raw_list)
        has_vram_col = any(s.get("vram_violation_attempts_mean") is not None for s in raw_list)
        if is_en:
            hdr_md = "| Method | Verified | Conditional U Regret (mean±SD) | Quality Regret mean | Opt Hit | Pareto Hit | Online Time (s)"
            sep_md = "|---|---:|---:|---:|---:|---:|---:"
        else:
            hdr_md = "| Phương pháp | Nghiệm hợp lệ | Conditional U Regret (mean±SD) | Quality Regret mean | Tỷ lệ trúng Optimum | Tỷ lệ trúng Pareto | Thời gian Online (s)"
            sep_md = "|---|---:|---:|---:|---:|---:|---:"
        if has_auarc:
            hdr_md += " | AUARC mean | TTT-95 (s)"
            sep_md += "|---:|---:"
        if has_vram_col:
            hdr_md += " | VRAM-Viol | Wasted (s) | Pruned" if is_en else " | Vi phạm VRAM | Lãng phí (s) | Conformal lọc"
            sep_md += "|---:|---:|---:"
        md += [f"## Study `{sid}`", "", hdr_md + " |", sep_md + "|"]

        col_spec = "lcccccc" + ("c" if has_auarc else "") + ("ccc" if has_vram_col else "")
        tex_hdr = (
            (r"Method & Verified & U Regret ($\mu \pm \sigma$) & Quality Regret & Opt Hit & Pareto Hit"
             if is_en else r"Phương pháp & Hợp lệ & U Regret ($\mu \pm \sigma$) & Quality Regret & Trúng Opt & Trúng Pareto")
            + (r" & AUARC" if has_auarc else "")
            + (r" & Online (s)" if True else "")
            + ((r" & VRAM-Viol & Wasted (s) & Pruned" if is_en else r" & Vi phạm VRAM & Lãng phí (s) & Conformal lọc") if has_vram_col else "")
            + r" \\"
        )
        tex += [
            r"\begin{table}[htbp]",
            r"\centering",
            (rf"\caption{{Budget-constrained hyperparameter optimization benchmark (Study {_tex_escape(sid)}).}}"
             if is_en else rf"\caption{{Đối chứng các phương pháp tối ưu cấu hình có ràng buộc ngân sách (Study {_tex_escape(sid)}).}}"),
            rf"\label{{tab:nckh_benchmark_{_tex_escape(sid)}}}",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            tex_hdr,
            r"\midrule",
        ]
        for s in raw_list:
            verified_str = f"{s['feasible_count']}/{s['n']}"
            reg_m, reg_sd = s.get("conditional_regret_mean"), s.get("conditional_regret_sd")
            q_reg = s.get("conditional_quality_regret_mean")
            opt_hit = s.get("exact_optimum_hit_rate")
            par_hit = s.get("pareto_hit_rate")
            auarc_m = s.get("auarc_mean")
            ttt_m = s.get("time_to_target_95_mean")
            t_mean = s.get("online_seconds_mean")
            v_viol = s.get("vram_violation_attempts_mean")
            v_waste = s.get("vram_violation_wasted_seconds_mean")
            c_prun = s.get("conformal_pruned_mean")
            s_row = [sid, s["method"], s["feasible_count"], s["n"], reg_m, reg_sd, q_reg, opt_hit, par_hit, t_mean, auarc_m, ttt_m]
            if has_vram_safety_global:
                s_row += [v_viol, v_waste, c_prun]
            summary_rows.append(s_row)
            if len(svg_bars) < 12:
                svg_bars.append({"method": s["method"], "regret": reg_m, "verified": verified_str, "t_mean": t_mean, "auarc_mean": auarc_m})
            opt_s = "—" if opt_hit is None else f"{opt_hit:.1%}"
            par_s = "—" if par_hit is None else f"{par_hit:.1%}"
            opt_tex = "---" if opt_hit is None else _tex_escape(f"{opt_hit:.1%}")
            par_tex = "---" if par_hit is None else _tex_escape(f"{par_hit:.1%}")
            line_md = f"| `{s['method']}` | {verified_str} | {_fmt(reg_m, 5)}±{_fmt(reg_sd, 5)} | {_fmt(q_reg, 5)} | {opt_s} | {par_s} | {_fmt(t_mean, 2)}"
            if has_auarc:
                line_md += f" | {_fmt(auarc_m, 5)} | {_fmt(ttt_m, 2)}"
            if has_vram_col:
                line_md += f" | {_fmt(v_viol, 2)} | {_fmt(v_waste, 2)} | {_fmt(c_prun, 1)}"
            md.append(line_md + " |")

            reg_tex = "---" if reg_m is None else (rf"{reg_m:.4f} $\pm$ {reg_sd:.4f}" if reg_sd is not None else f"{reg_m:.4f}")
            line_tex = f"{_tex_escape(s['method'])} & {verified_str} & {reg_tex} & {_fmt_tex(q_reg, 4)} & {opt_tex} & {par_tex}"
            if has_auarc:
                line_tex += f" & {_fmt_tex(auarc_m, 4)}"
            line_tex += f" & {_fmt_tex(t_mean, 2)}"
            if has_vram_col:
                line_tex += f" & {_fmt_tex(v_viol, 2)} & {_fmt_tex(v_waste, 1)} & {_fmt_tex(c_prun, 1)}"
            tex.append(line_tex + r" \\")
        tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        md.append("")

        paired_list = st.get("paired_comparisons") or []
        has_stats = any("holm" in m for p in paired_list for m in (p.get("metrics") or {}).values())
        if has_stats:
            md += [
                f"### Paired Non-Parametric Tests (Study `{sid}`)" if is_en else f"### Kiểm định Phi tham số Ghép cặp (Study `{sid}`)",
                "",
                st.get("paired_nonparametric_scope", "Conditional on the observed dataset, hardware, locked budget, and paired search_seeds; not a cross-domain confidence interval."),
                "",
                "| Candidate / Baseline | Metric | Paired n | Wilcoxon p | Sign-flip p | Holm adj. p |",
                "|---|---|---:|---:|---:|---:|",
            ]
            for pair in paired_list:
                for metric, values in (pair.get("metrics") or {}).items():
                    if "holm" not in values:
                        continue
                    md.append(
                        f"| `{pair['candidate']}` vs `{pair['baseline']}` | `{metric}` | {values['paired_seed_count']} | "
                        f"{values['wilcoxon']['p_value']:.4f} | {values['sign_flip']['p_value']:.4f} | {values['holm']['adjusted_p_value']:.4f} |"
                    )
            md.append("")

    out_headers = [
        "study_id", "method", "search_seed", "selected_config_id", "feasible", "objective_regret",
        "quality_regret", "exact_optimum_hit", "pareto_hit", "online_seconds", "overrun_seconds",
        "auarc", "time_to_target_95",
    ]
    sum_headers = [
        "study_id", "method", "feasible_count", "repetitions", "conditional_regret_mean",
        "conditional_regret_sd", "conditional_quality_regret_mean", "exact_optimum_hit_rate",
        "pareto_hit_rate", "online_seconds_mean", "auarc_mean", "time_to_target_95_mean",
    ]
    if has_vram_safety_global:
        out_headers += ["vram_violation_attempt_count", "vram_violation_wasted_seconds", "conformal_pruned_count"]
        sum_headers += ["vram_violation_attempts_mean", "vram_violation_wasted_seconds_mean", "conformal_pruned_mean"]

    _write_csv(out_dir / "benchmark_outcomes.csv", out_headers, outcome_rows)
    _write_csv(out_dir / "method_summary.csv", sum_headers, summary_rows)
    (out_dir / "tables.tex").write_text("\n".join(tex), encoding="utf-8")
    (out_dir / "benchmark_regret.svg").write_text(_svg_benchmark_bars(svg_bars, lang=lang), encoding="utf-8")
    (out_dir / "anytime_convergence.svg").write_text(
        _svg_anytime_benchmark(method_trajectories, svg_bars, ref_opt_u, target_95, horizon, lang=lang),
        encoding="utf-8",
    )
    summary_md_path = out_dir / "PAPER_SUMMARY.md"
    summary_md_path.write_text("\n".join(md), encoding="utf-8")
    evidence = {
        "schema_version": 1,
        "kind": kind,
        "lang": lang,
        "source_path": str(folder),
        "study_count": len(studies),
        "training_started": False,
        "source_run_mutated": False,
    }
    evidence["sha256"] = digest(evidence)
    write_json(out_dir / "PAPER_EVIDENCE.json", evidence)

    prov_path, exported_files = _write_provenance(
        out_dir,
        kind=kind,
        source_path=folder,
        lang=lang,
        input_artifacts=input_artifacts,
        protocol_identity=protocol_identity,
    )
    return {
        "kind": kind,
        "lang": lang,
        "source_path": str(folder),
        "output_directory": str(out_dir),
        "summary_markdown": str(summary_md_path),
        "latex_tables": str(out_dir / "tables.tex"),
        "evidence_json": str(out_dir / "PAPER_EVIDENCE.json"),
        "provenance_json": str(prov_path),
        "exported_files": exported_files,
    }


def export_conformal_vram_svg(measurements, card, out_path, *, vram_free_mb=None, edge_limit_mb=1800.0, lang="vi"):
    """Render Okabe-Ito vector plot of Observed vs Split Conformal VRAM predictions + 95% upper bounds."""
    from .conformal_vram import predict_conformal_vram
    is_en = lang == "en"
    W, H = 760, 480
    pad_l, pad_r, pad_t, pad_b = 76, 36, 54, 68
    pw, ph = W - pad_l - pad_r, H - pad_t - pad_b

    pts = []
    for m in measurements or []:
        if m.get("status") != "ok" or not isinstance(m.get("memory_mb"), (int, float)) or not math.isfinite(m["memory_mb"]):
            continue
        c = m["config"]
        pr = predict_conformal_vram(card, c, m.get("descriptor"), alpha=0.05, task=card.get("task", "detection"))
        pts.append({
            "model": Path(str(c.get("model", ""))).name or "model",
            "obs": float(m["memory_mb"]),
            "pt": float(pr["point_mb"]),
            "ub": float(pr["upper_bound_mb"]),
        })
    if not pts:
        raise ValueError("No valid VRAM points to plot conformal_vram_calibration.svg")

    vmax = max([p["obs"] for p in pts] + [p["ub"] for p in pts] + ([float(vram_free_mb)] if vram_free_mb else [0.0])) * 1.06
    def sx(v):
        return pad_l + (v / max(1.0, vmax)) * pw
    def sy(v):
        return pad_t + ph - (v / max(1.0, vmax)) * ph

    models_order = sorted({p["model"] for p in pts})
    colors = {m_name: OKABE_ITO[idx % len(OKABE_ITO)] for idx, m_name in enumerate(models_order)}

    title_s = "Split Conformal VRAM: Observed vs Point Prediction & 95% Upper Bound" if is_en else "Split Conformal VRAM: Thực đo vs Dự báo Điểm & Cận trên 95%"
    xlab = "Predicted Point VRAM (MiB)" if is_en else "VRAM Dự báo Điểm (MiB)"
    ylab = "Observed / Conformal Upper Bound VRAM (MiB)" if is_en else "VRAM Thực đo & Cận trên Conformal 95% (MiB)"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{W/2:.1f}" y="28" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="bold" fill="#111827">{html.escape(title_s)}</text>',
        f'<rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>',
    ]
    n_ticks = 5
    for k in range(n_ticks + 1):
        val = vmax * k / n_ticks
        gx, gy = sx(val), sy(val)
        parts.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l+pw}" y2="{gy:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l-8}" y="{gy+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="10" fill="#475569">{val:.0f}</text>')
        parts.append(f'<text x="{gx:.1f}" y="{pad_t+ph+16}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#475569">{val:.0f}</text>')

    # Diagonal y=x
    parts.append(f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(vmax):.1f}" y2="{sy(vmax):.1f}" stroke="#475569" stroke-width="1.3" stroke-dasharray="5,4"/>')
    if edge_limit_mb and edge_limit_mb < vmax:
        gy_e = sy(edge_limit_mb)
        parts.append(f'<line x1="{pad_l}" y1="{gy_e:.1f}" x2="{pad_l+pw}" y2="{gy_e:.1f}" stroke="#E69F00" stroke-width="1.4" stroke-dasharray="6,3"/>')
        parts.append(f'<text x="{pad_l+6}" y="{gy_e-4:.1f}" font-family="sans-serif" font-size="10" fill="#b45309">Edge Ceiling {edge_limit_mb:.0f} MiB</text>')
    if vram_free_mb and vram_free_mb < vmax:
        gy_f = sy(vram_free_mb)
        parts.append(f'<line x1="{pad_l}" y1="{gy_f:.1f}" x2="{pad_l+pw}" y2="{gy_f:.1f}" stroke="#D55E00" stroke-width="1.5" stroke-dasharray="3,3"/>')
        parts.append(f'<text x="{pad_l+6}" y="{gy_f-4:.1f}" font-family="sans-serif" font-size="10" fill="#b91c1c">Physical Free VRAM {vram_free_mb:.0f} MiB (WDDM spill)</text>')

    for p in pts:
        col = colors[p["model"]]
        px, py_obs, py_ub, py_pt = sx(p["pt"]), sy(p["obs"]), sy(p["ub"]), sy(p["pt"])
        parts.append(f'<line x1="{px:.1f}" y1="{py_pt:.1f}" x2="{px:.1f}" y2="{py_ub:.1f}" stroke="{col}" stroke-width="1.4" stroke-opacity="0.55"/>')
        parts.append(f'<line x1="{px-4:.1f}" y1="{py_ub:.1f}" x2="{px+4:.1f}" y2="{py_ub:.1f}" stroke="{col}" stroke-width="1.8"/>')
        parts.append(f'<circle cx="{px:.1f}" cy="{py_obs:.1f}" r="4.2" fill="{col}" stroke="#ffffff" stroke-width="0.9"/>')

    # Legend
    lx = pad_l + pw - 210
    ly = pad_t + 14
    parts.append(f'<rect x="{lx}" y="{ly}" width="200" height="{24 + 18*len(models_order)}" rx="4" fill="#ffffff" fill-opacity="0.92" stroke="#cbd5e1"/>')
    parts.append(f'<text x="{lx+8}" y="{ly+15}" font-family="sans-serif" font-size="10" font-weight="bold" fill="#1e293b">● Observed  ⊤ Conformal 95%</text>')
    for idx, m_name in enumerate(models_order):
        cy = ly + 31 + idx * 18
        parts.append(f'<circle cx="{lx+14}" cy="{cy-3}" r="4" fill="{colors[m_name]}"/>')
        parts.append(f'<text x="{lx+24}" y="{cy}" font-family="sans-serif" font-size="10" fill="#334155">{html.escape(m_name)}</text>')

    parts.append(f'<text x="{pad_l + pw/2:.1f}" y="{H - 18}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1e293b">{html.escape(xlab)}</text>')
    parts.append(f'<text x="18" y="{pad_t + ph/2:.1f}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1e293b" transform="rotate(-90 18 {pad_t + ph/2:.1f})">{html.escape(ylab)}</text>')
    parts.append("</svg>")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return out


def export_paper_artifacts(run_path, output_directory, lang="vi"):
    _validate_lang(lang)
    folder = Path(run_path).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Run/study directory does not exist: {folder}")
    out_dir = Path(output_directory).resolve()
    if out_dir.exists():
        raise ValueError("Paper export directory already exists; choose a new directory to preserve prior artifacts")
    out_dir.mkdir(parents=True, exist_ok=False)
    if (folder / "protocol.json").is_file() and (folder / "report.json").is_file():
        return _export_optimize_run(folder, out_dir, lang=lang)
    if (folder / "benchmark_protocol.json").is_file() or (folder / "ALGORITHM_STUDY.json").is_file():
        return _export_benchmark_or_algorithm_study(folder, out_dir, lang=lang)
    raise ValueError("Unsupported directory for paper-export: expected a completed v2 optimize run, benchmark study, or algorithm-report folder")
