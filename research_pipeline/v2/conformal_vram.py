"""Physics-residual Locally-Weighted Split Conformal VRAM predictor and shrinkage 2-point calibration."""

import math
import random
import statistics

import numpy as np

from yolo_tuner.research import digest
from .model_features import descriptor_for, family_scale, model_identity, trainable_parameters_for

CONFORMAL_SCHEMA_VERSION = 1
_RIDGE_GRID = (0.1, 0.5, 1.0, 5.0, 15.0)


def _safe_float(value, default=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return float(default)
    return float(value)


def _static_components(config, descriptor=None, total_p=None, train_p=None):
    desc = descriptor or {}
    if total_p is None:
        total_p = _safe_float(desc.get("parameters"), 2_600_000.0)
    if train_p is None:
        layers = max(1, int(_safe_float(desc.get("layers"), 24.0)))
        freeze = max(0, int(_safe_float(config.get("freeze"), 0.0)))
        train_p = float(total_p) * max(0.05, 1.0 - min(freeze, layers) / layers)
    precision = str(config.get("precision", "fp32") or "fp32").lower()
    optimizer = str(config.get("optimizer", "AdamW") or "AdamW")
    is_amp = precision in {"fp16", "bf16"}
    weight_bytes = (6.0 if is_amp else 4.0) * float(total_p)
    grad_bytes = 4.0 * float(train_p)
    opt_bytes = (8.0 if optimizer == "AdamW" else 4.0) * float(train_p)
    static_mb = (weight_bytes + grad_bytes + opt_bytes) / (1024.0 ** 2)
    return float(static_mb), float(total_p), float(train_p)


def _activation_proxy(config, descriptor=None, task="detection"):
    desc = descriptor or {}
    batch = max(1.0, _safe_float(config.get("batch"), 1.0))
    imgsz = max(1.0, _safe_float(config.get("imgsz"), 320.0 if task == "detection" else 32.0))
    precision = str(config.get("precision", "fp32") or "fp32").lower()
    prec_factor = 2.0 if precision == "fp32" else 1.0
    if task == "detection":
        layers = max(1.0, _safe_float(desc.get("layers"), 24.0))
        freeze = max(0.0, _safe_float(config.get("freeze"), 0.0))
        freeze_factor = max(0.35, 1.0 - 1.2 * (min(freeze, layers) / layers))
        flops = _safe_float(desc.get("gflops_reference_32"), 0.0)
        params = _safe_float(desc.get("parameters"), 0.0)
        if flops > 0.0:
            model_factor = (flops / 0.01653) ** 0.55 * (layers / 24.0) ** 0.25
        elif params > 0.0:
            model_factor = (params / 2_600_000.0) ** 0.50 * (layers / 24.0) ** 0.25
        else:
            _, scale = family_scale(config)
            model_factor = max(1.0, float(scale or 1)) ** 1.2
    else:
        width = max(4.0, _safe_float(config.get("width"), 16.0))
        hidden = max(4.0, _safe_float(config.get("hidden"), 256.0)) if task == "captioning" else 0.0
        model_factor = (width / 16.0) * (1.0 + 0.15 * hidden / 256.0)
        freeze_factor = 1.0
    return float(batch * ((imgsz / 320.0) ** 2) * prec_factor * model_factor * freeze_factor)


def vram_feature_vector(config, descriptor=None, task="detection", protocol=None):
    """Extract physics-interaction residual features invariant to gradient accumulation.

    Avoids raw collinear parameter/FLOP intercepts that distort cross-architecture LOMO transfer.
    """
    desc = descriptor
    if desc is None and protocol is not None:
        desc = descriptor_for(protocol, config)
    desc = desc or {}
    if protocol is not None:
        total_p, train_p, _ = trainable_parameters_for(protocol, config)
        static_mb, total_p, train_p = _static_components(config, desc, total_p=total_p, train_p=train_p)
    else:
        static_mb, total_p, train_p = _static_components(config, desc)
    act_proxy = _activation_proxy(config, desc, task=task)
    precision = str(config.get("precision", "fp32") or "fp32").lower()
    is_fp32 = float(precision == "fp32")
    optimizer = str(config.get("optimizer", "AdamW") or "AdamW")
    layers = max(1.0, _safe_float(desc.get("layers"), 24.0))
    freeze = max(0.0, _safe_float(config.get("freeze"), 0.0))
    feats = [
        1.0,
        act_proxy,
        act_proxy * is_fp32,
        is_fp32,
        float(optimizer == "SGD"),
        min(1.0, freeze / layers),
    ]
    return feats, float(static_mb), float(act_proxy)


def _split_conformal_quantile(scores, alpha):
    """Exact finite-sample split conformal quantile ceil((n+1)*(1-alpha))/n."""
    vals = sorted(float(s) for s in scores if math.isfinite(s))
    n = len(vals)
    if n == 0:
        return 0.0
    level = math.ceil((n + 1) * (1.0 - float(alpha))) / float(n)
    level = min(1.0, max(0.0, level))
    idx = min(n - 1, max(0, math.ceil(level * n) - 1))
    return float(vals[idx])


def _fit_ridge(X, y, ridge=1.0):
    n, d = X.shape
    means = np.mean(X[:, 1:], axis=0) if n > 0 else np.zeros(d - 1, dtype=float)
    stds = np.std(X[:, 1:], axis=0) if n > 0 else np.ones(d - 1, dtype=float)
    stds = np.where(stds < 1e-8, 1.0, stds)
    Xs = np.ones_like(X, dtype=float)
    if d > 1:
        Xs[:, 1:] = (X[:, 1:] - means) / stds
    reg = np.eye(d, dtype=float) * float(ridge)
    reg[0, 0] = 0.0  # Do not shrink intercept.
    A = Xs.T @ Xs + reg
    b = Xs.T @ y
    try:
        coef = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(A, b, rcond=None)[0]
    return coef.astype(float), means.astype(float), stds.astype(float)


def _select_ridge_loocv(X, y, candidates=_RIDGE_GRID):
    """Select Ridge penalty on the fit split ONLY using exact PRESS / LOOCV MSE."""
    n, d = X.shape
    if n < 5:
        return 1.0
    means = np.mean(X[:, 1:], axis=0)
    stds = np.std(X[:, 1:], axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)
    Xs = np.ones_like(X, dtype=float)
    if d > 1:
        Xs[:, 1:] = (X[:, 1:] - means) / stds

    best_ridge = float(candidates[0])
    best_mse = math.inf
    for lam in candidates:
        reg = np.eye(d, dtype=float) * float(lam)
        reg[0, 0] = 0.0
        A = Xs.T @ Xs + reg
        try:
            A_inv = np.linalg.inv(A)
            coef = A_inv @ (Xs.T @ y)
        except np.linalg.LinAlgError:
            continue
        yhat = Xs @ coef
        # Exact hat diagonal h_i = x_i^T (X^T X + reg)^{-1} x_i
        h = np.sum((Xs @ A_inv) * Xs, axis=1)
        denom = np.maximum(1e-3, 1.0 - h)
        e_loo = (y - yhat) / denom
        mse = float(np.mean(e_loo ** 2))
        if math.isfinite(mse) and mse < best_mse:
            best_mse = mse
            best_ridge = float(lam)
    return best_ridge


def _eval_ridge(feats, coef, means, stds):
    x = np.asarray(feats, dtype=float)
    xs = np.ones_like(x)
    if len(x) > 1:
        xs[1:] = (x[1:] - means) / stds
    return float(np.dot(xs, coef))


def _fit_local_scale(fit_preds, fit_abs_res, sigma_floor=120.0):
    """Fit monotone heteroscedastic residual scale sigma(pred) = max(floor, c0 + c1 * pred) on fit split."""
    if not fit_preds:
        return {"sigma_floor": float(sigma_floor), "c0": float(sigma_floor), "c1": 0.05}
    mean_p = statistics.mean(fit_preds)
    mean_r = statistics.mean(fit_abs_res)
    var_p = sum((p - mean_p) ** 2 for p in fit_preds)
    if len(fit_preds) >= 5 and var_p > 100.0:
        slope = sum((p - mean_p) * (r - mean_r) for p, r in zip(fit_preds, fit_abs_res)) / var_p
        c1 = min(0.25, max(0.03, float(slope)))
    else:
        c1 = 0.06
    c0 = max(float(sigma_floor), 1.10 * (mean_r - c1 * mean_p))
    return {"sigma_floor": float(sigma_floor), "c0": float(c0), "c1": float(c1)}


def _eval_scale(pred_mb, scale_params):
    sp = scale_params or {}
    floor = float(sp.get("sigma_floor", 120.0))
    c0 = float(sp.get("c0", floor))
    c1 = float(sp.get("c1", 0.06))
    return float(max(floor, c0 + c1 * max(0.0, float(pred_mb))))


def fit_conformal_vram_card(fit_records, cal_records, alphas=(0.10, 0.05), ridge="auto", task="detection", metadata=None):
    """Fit physics-residual Ridge + Locally-Weighted Split Conformal quantiles on strictly disjoint splits."""
    valid_fit = [r for r in fit_records if isinstance(r.get("memory_mb"), (int, float)) and not isinstance(r.get("memory_mb"), bool) and math.isfinite(r["memory_mb"]) and r["memory_mb"] >= 0]
    valid_cal = [r for r in cal_records if isinstance(r.get("memory_mb"), (int, float)) and not isinstance(r.get("memory_mb"), bool) and math.isfinite(r["memory_mb"]) and r["memory_mb"] >= 0]
    if len(valid_fit) < 2:
        raise ValueError("Need at least 2 valid non-OOM fit records for Conformal VRAM card")
    if not valid_cal:
        raise ValueError("Need at least 1 valid non-OOM calibration record for Split Conformal card")

    X_fit, y_res_fit, static_fit = [], [], []
    for rec in valid_fit:
        fv, static_mb, _ = vram_feature_vector(rec["config"], rec.get("descriptor"), task=task)
        X_fit.append(fv)
        static_fit.append(static_mb)
        y_res_fit.append(max(0.0, float(rec["memory_mb"]) - static_mb))
    X_arr = np.asarray(X_fit, dtype=float)
    y_arr = np.asarray(y_res_fit, dtype=float)
    eff_ridge = _select_ridge_loocv(X_arr, y_arr) if ridge == "auto" else float(ridge)
    coef, means, stds = _fit_ridge(X_arr, y_arr, ridge=eff_ridge)

    fit_preds, fit_abs_res = [], []
    for fv, st_mb, rec in zip(X_fit, static_fit, valid_fit):
        p_mb = st_mb + max(0.0, _eval_ridge(fv, coef, means, stds))
        fit_preds.append(p_mb)
        fit_abs_res.append(abs(float(rec["memory_mb"]) - p_mb))
    scale_params = _fit_local_scale(fit_preds, fit_abs_res, sigma_floor=120.0)

    scores_add, scores_rel, scores_norm = [], [], []
    cal_errors = []
    for rec in valid_cal:
        fv, static_mb, _ = vram_feature_vector(rec["config"], rec.get("descriptor"), task=task)
        pred_res = max(0.0, _eval_ridge(fv, coef, means, stds))
        pred_mb = static_mb + pred_res
        obs_mb = float(rec["memory_mb"])
        err = obs_mb - pred_mb
        sigma_c = _eval_scale(pred_mb, scale_params)
        scores_add.append(err)
        scores_rel.append(err / max(1024.0, pred_mb))
        scores_norm.append(err / sigma_c)
        cal_errors.append(abs(err))

    quantiles = {}
    for alpha in alphas:
        key = f"{float(alpha):.4f}".rstrip("0").rstrip(".")
        quantiles[key] = {
            "alpha": float(alpha),
            "target_coverage": float(1.0 - float(alpha)),
            "q_norm": float(max(0.0, _split_conformal_quantile(scores_norm, alpha))),
            "q_add_mb": float(max(0.0, _split_conformal_quantile(scores_add, alpha))),
            "q_rel": float(max(0.0, _split_conformal_quantile(scores_rel, alpha))),
            "q_add_signed_mb": float(_split_conformal_quantile(scores_add, alpha)),
        }

    card = {
        "schema_version": CONFORMAL_SCHEMA_VERSION,
        "kind": "split_conformal_physics_residual_vram",
        "task": task,
        "ridge": float(eff_ridge),
        "scale_floor_mb": 1024.0,
        "scale_params": scale_params,
        "calibration_scores": [float(s) for s in sorted(scores_norm)],
        "n_fit": len(valid_fit),
        "n_cal": len(valid_cal),
        "cal_mae_mb": float(statistics.mean(cal_errors)),
        "feature_means": [float(v) for v in means],
        "feature_stds": [float(v) for v in stds],
        "coefficients": [float(v) for v in coef],
        "quantiles": quantiles,
        "model_identities": sorted({model_identity(r["config"]) for r in valid_fit + valid_cal}),
        "metadata": metadata or {},
    }
    card["sha256"] = digest(card)
    return card


def verify_conformal_vram_card(card):
    if not isinstance(card, dict) or card.get("schema_version") != CONFORMAL_SCHEMA_VERSION:
        raise ValueError("Invalid or unsupported Conformal VRAM card schema")
    expected = card.get("sha256")
    actual = digest({k: v for k, v in card.items() if k != "sha256"})
    if expected != actual:
        raise ValueError("Conformal VRAM card checksum mismatch")
    return True


def calibrate_online_2point(card, anchor_observations, task="detection", protocol=None):
    """Shrinkage-stabilized online calibration using 1-5 local anchor measurements."""
    anchors = []
    for item in anchor_observations or []:
        cfg = item.get("config")
        obs = item.get("memory_mb")
        if cfg is None or isinstance(obs, bool) or not isinstance(obs, (int, float)) or not math.isfinite(obs):
            continue
        desc = item.get("descriptor")
        fv, static_mb, act_proxy = vram_feature_vector(cfg, desc, task=task, protocol=protocol)
        if card is not None:
            coef = np.asarray(card["coefficients"], dtype=float)
            means = np.asarray(card["feature_means"], dtype=float)
            stds = np.asarray(card["feature_stds"], dtype=float)
            base_pred = static_mb + max(0.0, _eval_ridge(fv, coef, means, stds))
        else:
            base_pred = static_mb + 200.0 + 50.0 * act_proxy
        anchors.append({
            "obs_mb": float(obs),
            "static_mb": static_mb,
            "base_pred_mb": float(base_pred),
            "pred_res": max(1.0, float(base_pred) - static_mb),
            "obs_res": max(0.0, float(obs) - static_mb),
            "err_mb": float(obs) - float(base_pred),
        })
    if not anchors:
        return None
    if len(anchors) == 1:
        shift = 0.75 * anchors[0]["err_mb"]
        return {"mode": "1-point-shift", "n_anchors": 1, "scale": 1.0, "shift_mb": float(shift)}

    xs = [a["pred_res"] for a in anchors]
    ys = [a["obs_res"] for a in anchors]
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    span_x = (max(xs) - min(xs)) if len(xs) >= 2 else 0.0
    if var_x > 100.0 and span_x >= 300.0:
        cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        # Centered Bayesian slope shrinkage toward 1.0 preserves unbiased shift across anchors
        prior_w = 15_000.0
        slope = (cov_xy + prior_w * 1.0) / (var_x + prior_w)
        scale = min(1.85, max(0.85, slope))
        shift = mean_y - scale * mean_x
        if all(math.isfinite(v) for v in (scale, shift)):
            return {"mode": "2-point-affine", "n_anchors": len(anchors), "scale": float(scale), "shift_mb": float(shift)}
    med_shift = statistics.median(a["err_mb"] for a in anchors)
    return {"mode": "median-shift", "n_anchors": len(anchors), "scale": 1.0, "shift_mb": float(med_shift)}


def _base_point_and_sigma(card, config, descriptor=None, online_cal=None, task="detection", protocol=None):
    fv, static_mb, act_proxy = vram_feature_vector(config, descriptor, task=task, protocol=protocol)
    coef = np.asarray(card["coefficients"], dtype=float)
    means = np.asarray(card["feature_means"], dtype=float)
    stds = np.asarray(card["feature_stds"], dtype=float)
    raw_res = max(0.0, _eval_ridge(fv, coef, means, stds))
    raw_point_mb = max(static_mb, static_mb + raw_res)

    if online_cal:
        scale = float(online_cal.get("scale", 1.0))
        shift = float(online_cal.get("shift_mb", 0.0))
        cal_res = max(0.0, scale * raw_res + shift)
        point_mb = max(static_mb, static_mb + cal_res)
        basis_cal = f"; calibrated({online_cal.get('mode')}, k={online_cal.get('n_anchors')})"
    else:
        point_mb = raw_point_mb
        basis_cal = ""
    # Evaluate local uncertainty scale at max(raw_point_mb, point_mb) so shrinkage scaling never under-covers
    scale_params = card.get("scale_params")
    sigma_c = _eval_scale(max(raw_point_mb, point_mb), scale_params) if scale_params else max(float(card.get("scale_floor_mb", 1024.0)), point_mb)
    return point_mb, sigma_c, static_mb, act_proxy, basis_cal


def predict_conformal_vram(card, config, descriptor=None, alpha=0.05, online_cal=None, task="detection", protocol=None):
    """Return point estimate and Locally-Weighted Split Conformal upper bound."""
    point_mb, sigma_c, static_mb, act_proxy, basis_cal = _base_point_and_sigma(
        card, config, descriptor=descriptor, online_cal=online_cal, task=task, protocol=protocol
    )
    quantiles = card.get("quantiles") or {}
    akey = f"{float(alpha):.4f}".rstrip("0").rstrip(".")
    qinfo = quantiles.get(akey)
    if qinfo is None and quantiles:
        qinfo = min(quantiles.values(), key=lambda q: abs(float(q.get("alpha", 1.0)) - float(alpha)))

    if qinfo and "q_norm" in qinfo and card.get("scale_params"):
        margin_mb = max(0.0, float(qinfo["q_norm"])) * sigma_c
    else:
        q_rel = float(qinfo.get("q_rel", 0.0)) if qinfo else 0.0
        scale_floor = float(card.get("scale_floor_mb", 1024.0))
        margin_mb = max(0.0, q_rel) * max(scale_floor, point_mb)

    upper_mb = point_mb + margin_mb
    return {
        "point_mb": float(point_mb),
        "upper_bound_mb": float(upper_mb),
        "sigma_mb": float(sigma_c),
        "static_model_state_mb": float(static_mb),
        "activation_proxy": float(act_proxy),
        "conformal_q_mb": float(margin_mb),
        "alpha": float(alpha),
        "target_coverage": float(1.0 - float(alpha)),
        "basis": f"physics_residual_ridge + lw_split_conformal(n_fit={card.get('n_fit')}, n_cal={card.get('n_cal')}, alpha={alpha}){basis_cal}",
    }


def conformal_feasibility_probability(card, config, limit_mb, descriptor=None, online_cal=None, task="detection", protocol=None):
    """Return smoothed Split Conformal predictive feasibility P(V(c) <= limit_mb)."""
    if card is None or limit_mb is None:
        return 1.0
    if isinstance(limit_mb, bool) or not isinstance(limit_mb, (int, float)) or not math.isfinite(limit_mb):
        return 1.0
    point_mb, sigma_c, _, _, _ = _base_point_and_sigma(
        card, config, descriptor=descriptor, online_cal=online_cal, task=task, protocol=protocol
    )
    scores = card.get("calibration_scores")
    if isinstance(scores, list) and scores:
        n_cal = len(scores)
        count_ok = sum(1 for s in scores if (point_mb + float(s) * sigma_c) <= float(limit_mb) + 1e-9)
        return float((count_ok + 1.0) / (n_cal + 1.0))
    # Legacy card fallback via stored alpha quantile
    pr = predict_conformal_vram(card, config, descriptor=descriptor, alpha=0.05, online_cal=online_cal, task=task, protocol=protocol)
    return 1.0 if pr["upper_bound_mb"] <= float(limit_mb) else 0.05


def _select_lomo_anchors(sorted_held):
    """Select 2 informative non-spill anchors across the operational range (low & upper-mid safe)."""
    safe = [r for r in sorted_held if float(r["memory_mb"]) <= 2400.0]
    pool = safe if len(safe) >= 3 else sorted_held
    if len(pool) < 2:
        return list(sorted_held[:2])
    mid_hi = min(len(pool) - 1, max(1, (2 * len(pool)) // 3))
    a0, a1 = pool[0], pool[mid_hi]
    if a0 is a1 and len(pool) >= 2:
        a1 = pool[-1]
    return [a0, a1]


def evaluate_conformal_vram(records, alphas=(0.10, 0.05), n_splits=50, seed=2027, task="detection", limit_mb=None):
    """Evaluate empirical coverage, MAE/RMSE, OOM prevention, and LOMO + 2-pt calibration transfer."""
    valid = [r for r in records if isinstance(r.get("memory_mb"), (int, float)) and not isinstance(r.get("memory_mb"), bool) and math.isfinite(r["memory_mb"]) and r["memory_mb"] >= 0]
    if len(valid) < 8:
        raise ValueError("Need at least 8 valid VRAM records to evaluate Split Conformal predictor")

    rng = random.Random(seed)
    n_total = len(valid)
    if n_total >= 45:
        n_fit = int(round(0.42 * n_total))
        n_cal = max(20, int(round(0.38 * n_total)))
    else:
        n_fit = max(3, int(round(0.45 * n_total)))
        n_cal = max(2, int(round(0.30 * n_total)))
    if n_fit + n_cal >= n_total:
        n_fit = max(2, n_total - 4)
        n_cal = max(2, n_total - n_fit - 2)

    split_stats = {str(a): {"coverages": [], "maes": [], "rmses": [], "max_unders": [], "widths": [], "oom_prevent": [], "false_reject": []} for a in alphas}
    for _ in range(n_splits):
        idxs = list(range(n_total))
        rng.shuffle(idxs)
        fit_r = [valid[i] for i in idxs[:n_fit]]
        cal_r = [valid[i] for i in idxs[n_fit:n_fit + n_cal]]
        test_r = [valid[i] for i in idxs[n_fit + n_cal:]]
        card = fit_conformal_vram_card(fit_r, cal_r, alphas=alphas, ridge="auto", task=task)
        for alpha in alphas:
            key = str(alpha)
            covered, sq_errs, abs_errs, unders, widths = [], [], [], [], []
            actual_over, pred_over_on_actual, pred_over_on_safe, actual_safe = 0, 0, 0, 0
            for rec in test_r:
                obs = float(rec["memory_mb"])
                pr = predict_conformal_vram(card, rec["config"], rec.get("descriptor"), alpha=alpha, task=task)
                pt, ub = pr["point_mb"], pr["upper_bound_mb"]
                covered.append(obs <= ub + 1e-9)
                abs_errs.append(abs(obs - pt))
                sq_errs.append((obs - pt) ** 2)
                unders.append(max(0.0, obs - ub))
                widths.append(ub - pt)
                if limit_mb is not None:
                    if obs > limit_mb:
                        actual_over += 1
                        if ub > limit_mb:
                            pred_over_on_actual += 1
                    else:
                        actual_safe += 1
                        if ub > limit_mb:
                            pred_over_on_safe += 1
            split_stats[key]["coverages"].append(statistics.mean(covered))
            split_stats[key]["maes"].append(statistics.mean(abs_errs))
            split_stats[key]["rmses"].append(math.sqrt(statistics.mean(sq_errs)))
            split_stats[key]["max_unders"].append(max(unders))
            split_stats[key]["widths"].append(statistics.mean(widths))
            if limit_mb is not None:
                if actual_over > 0:
                    split_stats[key]["oom_prevent"].append(pred_over_on_actual / actual_over)
                if actual_safe > 0:
                    split_stats[key]["false_reject"].append(pred_over_on_safe / actual_safe)

    summary_splits = {}
    for alpha in alphas:
        key = str(alpha)
        st = split_stats[key]
        summary_splits[key] = {
            "alpha": float(alpha),
            "target_coverage": float(1.0 - float(alpha)),
            "empirical_coverage_mean": float(statistics.mean(st["coverages"])),
            "empirical_coverage_sd": float(statistics.stdev(st["coverages"])) if len(st["coverages"]) > 1 else 0.0,
            "mae_mb_mean": float(statistics.mean(st["maes"])),
            "rmse_mb_mean": float(statistics.mean(st["rmses"])),
            "mean_interval_width_mb": float(statistics.mean(st["widths"])),
            "mean_max_underestimation_mb": float(statistics.mean(st["max_unders"])),
            "oom_prevention_rate": float(statistics.mean(st["oom_prevent"])) if st["oom_prevent"] else None,
            "false_rejection_rate": float(statistics.mean(st["false_reject"])) if st["false_reject"] else None,
        }

    # Leave-One-Model-Out cross-architecture transfer (zero-shot vs 2-point online calibration)
    by_model = {}
    for rec in valid:
        by_model.setdefault(model_identity(rec["config"]), []).append(rec)
    lomo = []
    if len(by_model) >= 2:
        for held_id, held_recs in by_model.items():
            src_fit, src_cal = [], []
            for mid, rs in sorted(by_model.items()):
                if mid == held_id:
                    continue
                rs_s = sorted(rs, key=lambda x: float(x["memory_mb"]))
                for k_idx, rec in enumerate(rs_s):
                    (src_cal if k_idx % 2 == 1 else src_fit).append(rec)
            if len(src_fit) < 3 or len(src_cal) < 2 or len(held_recs) < 4:
                continue
            card = fit_conformal_vram_card(src_fit, src_cal, alphas=alphas, ridge="auto", task=task)
            sorted_held = sorted(held_recs, key=lambda r: _activation_proxy(r["config"], r.get("descriptor"), task=task))
            anchors = _select_lomo_anchors(sorted_held)
            eval_held = [r for r in sorted_held if r not in anchors]
            if not eval_held:
                continue
            cal_2pt = calibrate_online_2point(card, anchors, task=task)
            zero_errs, cal_errs, zero_cov, cal_cov = [], [], [], []
            alpha_ref = min(alphas)
            for rec in eval_held:
                obs = float(rec["memory_mb"])
                p0 = predict_conformal_vram(card, rec["config"], rec.get("descriptor"), alpha=alpha_ref, task=task)
                p1 = predict_conformal_vram(card, rec["config"], rec.get("descriptor"), alpha=alpha_ref, online_cal=cal_2pt, task=task)
                zero_errs.append(abs(obs - p0["point_mb"]))
                cal_errs.append(abs(obs - p1["point_mb"]))
                zero_cov.append(obs <= p0["upper_bound_mb"] + 1e-9)
                cal_cov.append(obs <= p1["upper_bound_mb"] + 1e-9)
            lomo.append({
                "held_out_model_identity": held_id,
                "held_out_model_name": str(held_recs[0]["config"].get("model", "")),
                "n_eval": len(eval_held),
                "zero_shot_mae_mb": float(statistics.mean(zero_errs)),
                "calibrated_2pt_mae_mb": float(statistics.mean(cal_errs)),
                "zero_shot_coverage": float(statistics.mean(zero_cov)),
                "calibrated_2pt_coverage": float(statistics.mean(cal_cov)),
                "alpha": float(alpha_ref),
                "calibration_params": cal_2pt,
            })

    return {
        "n_records": n_total,
        "n_splits": n_splits,
        "split_sizes": {"fit": n_fit, "cal": n_cal, "test": n_total - n_fit - n_cal},
        "splits_summary": summary_splits,
        "leave_one_model_out": lomo,
    }
