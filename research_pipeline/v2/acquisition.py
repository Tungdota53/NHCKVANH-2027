"""Finite-grid, cost-aware optimistic FULL-confirmation acquisition.

Kernel ridge is a quality point surrogate, not a Bayesian posterior or a safety
certificate. Leverage/distance/residual uncertainty is deliberately uncalibrated.
Only audited-style complete full-data/all-seed outcomes enter its quality fit;
neither probe quality nor a forecast's quality prior is a training target.
"""

import math
import statistics

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr

from yolo_tuner.research import Objective, nondominated
from .model_features import descriptor_for, family_scale, model_identity
from .schema import complete, config_id


DEFAULT_SPEC = {"version": 1, "kernel_ridge": 0.05, "length_scale": 1.0,
                "exploration": 0.25, "novelty_weight": 0.05}

_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(21)
_GH_SQRT2 = math.sqrt(2.0)
_GH_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)


def acquisition_spec(protocol):
    """Read predeclared design choices; these defaults are not universal optima."""
    spec = {**DEFAULT_SPEC, **protocol.get("acquisition", {})}
    if type(spec["version"]) is not int or spec["version"] not in {1, 2}:
        raise ValueError("Unsupported acquisition specification")
    for key in ("kernel_ridge", "length_scale", "exploration", "novelty_weight"):
        value = spec[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Acquisition {key} must be finite and nonnegative")
        if key in {"kernel_ridge", "length_scale"} and value <= 0:
            raise ValueError(f"Acquisition {key} must be positive")
    if "kernel" in spec and spec["kernel"] not in {"auto", "matern52", "rbf"}:
        raise ValueError("Acquisition kernel must be auto, matern52, or rbf")
    for grid_key in ("length_scale_grid", "ridge_grid"):
        if grid_key in spec:
            grid = spec[grid_key]
            if (not isinstance(grid, (list, tuple)) or not grid
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in grid)):
                raise ValueError(f"Acquisition {grid_key} must be a non-empty list of positive finite numbers")
    return spec


def finite(value, default=0.0, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return float(default)
    return float(value) if value > 0 or not positive else float(default)


def bounded(value):
    """Keep arithmetic receipts finite even for extreme-but-finite user scales."""
    return min(1e100, max(-1e100, value)) if not math.isnan(value) else 0.0


def median(values):
    """Median without overflowing the average of two large finite costs."""
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else statistics.mean((ordered[middle - 1], ordered[middle]))


def configuration_distances(protocol):
    """Mixed numeric/categorical RBF inputs normalized by the LOCKED grid.

    No measured quality determines normalization. Constant fields are removed;
    numeric squared differences and categorical mismatch each have unit weight.
    Model identity is content-based for local checkpoints, not a path ordinal.
    """
    numeric, categories = [], []
    names = ("imgsz", "batch", "accumulation", "width", "hidden", "feature_grid",
             "lr0", "weight_decay", "freeze", "dropout", "label_smoothing",
             "warmup_epochs", "mosaic", "close_mosaic",
             "parameters", "layers", "gflops_reference_32", "model_scale")
    for config in protocol["configs"]:
        descriptor = descriptor_for(protocol, config)
        family, scale = family_scale(config)
        values = []
        for name in names:
            if name in {"parameters", "layers", "gflops_reference_32"}:
                value = descriptor.get(name, 0)
            elif name == "model_scale":
                value = scale
            elif name == "mosaic":
                value = config.get(name, 1.0)
            else:
                value = config.get(name, 0)
            value = finite(value)
            values.append(math.log(max(value, 1e-12)) if name == "lr0" else
                          value if name in {"dropout", "label_smoothing", "freeze", "mosaic", "model_scale"} else math.log1p(max(0, value)))
        numeric.append(values)
        categories.append((model_identity(config), str(family), str(config.get("encoder", "none")),
                           str(config.get("precision", "fp32")), str(config.get("optimizer", "AdamW")),
                           str(bool(config.get("pretrained", descriptor.get("pretrained_checkpoint", False)))),
                           str(bool(config.get("cos_lr", False)))))
    if not numeric:
        return np.zeros((0, 0)), {"numeric_fields": [], "categorical_fields": [], "normalization": "locked finite-grid ranges"}
    x = np.asarray(numeric, dtype=float)
    spans = x.max(axis=0) - x.min(axis=0)
    active = spans > 1e-12
    scaled = (x[:, active] - x[:, active].min(axis=0)) / spans[active] if active.any() else np.zeros((len(x), 0))
    distance_squared = np.sum((scaled[:, None, :] - scaled[None, :, :]) ** 2, axis=2)
    category_names = ("model_content_identity", "model_family", "encoder", "precision", "optimizer", "pretrained", "cos_lr")
    active_categories = [i for i in range(len(category_names)) if len({row[i] for row in categories}) > 1]
    for column in active_categories:
        values = np.asarray([row[column] for row in categories])
        distance_squared += values[:, None] != values[None, :]
    dimensions = int(active.sum()) + len(active_categories)
    if dimensions:
        distance_squared /= dimensions
    return np.sqrt(np.clip(distance_squared, 0, 1)), {
        "numeric_fields": [name for name, yes in zip(names, active) if yes],
        "categorical_fields": [category_names[i] for i in active_categories],
        "normalization": "locked finite-grid ranges; mean numeric squared difference / categorical mismatch",
        "dimensions": dimensions}


def full_observations(protocol, rows, objective):
    """Recompute targets from complete workers, never trust partial row means."""
    seeds = set(protocol["training_seeds"])
    observed = {}
    for row in rows:
        index = row.get("index")
        if (type(index) is not int or not 0 <= index < len(protocol["configs"])
                or row.get("complete") is not True
                or row.get("state") not in {"confirmed_feasible", "confirmed_constraint_failure"}
                or row.get("config") != protocol["configs"][index]):
            continue
        runs = {r["seed"]: r for r in row.get("runs", [])
                if r.get("seed") in seeds and complete(r, protocol["full_epochs"])}
        if set(runs) != seeds:
            continue
        qualities = [r["quality"] for r in runs.values()]
        quality_var = statistics.variance(qualities) if len(qualities) >= 2 else 0.0
        quality_mean_var = quality_var / len(qualities) if len(qualities) >= 2 else 0.0
        target = {"complete": True, "quality_mean": statistics.mean(qualities), "quality_min": min(qualities),
                  "quality_var": float(quality_var), "quality_mean_var": float(quality_mean_var),
                  "time_mean": statistics.mean(r["worker_wall_seconds"] for r in runs.values()),
                  "time_max": max(r["worker_wall_seconds"] for r in runs.values()),
                  "memory_max": max(r["memory_mb"] for r in runs.values())}
        observed[index] = {"index": index, "config_id": config_id(protocol["configs"][index]),
                           **target, "feasible": objective.feasible(target),
                           "utility": bounded(objective.value(target))}
    return [observed[i] for i in sorted(observed)]


def _kernel_matrix(distances, kernel_name, length_scale):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaled = distances / float(length_scale)
        if kernel_name == "rbf":
            return np.exp(-0.5 * (scaled ** 2))
        if kernel_name == "matern52":
            r = math.sqrt(5.0) * scaled
            return (1.0 + r + (r ** 2) / 3.0) * np.exp(-r)
    raise ValueError(f"Unsupported kernel: {kernel_name}")


def _gaussian_ei(mu, sigma, target):
    if sigma <= 1e-12:
        return max(0.0, float(mu - target))
    z = (float(mu) - float(target)) / float(sigma)
    val = float(sigma) * (z * float(ndtr(z)) + math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi))
    return max(0.0, val) if math.isfinite(val) else 0.0


def _area_2d(points):
    pts = [(min(1.0, max(0.0, float(t))), min(1.0, max(0.0, float(m))))
           for t, m in points if t > 0.0 and m > 0.0]
    if not pts:
        return 0.0
    pts.sort(key=lambda p: (p[0], p[1]), reverse=True)
    area = 0.0
    max_m = 0.0
    for t, m in pts:
        if m > max_m:
            area += (m - max_m) * t
            max_m = m
    return area


def _exact_3d_ehvi(mu_q, sigma_q, time_s, mem_mb, front_points, t_scale, m_scale):
    b_t = min(1.0, max(0.0, 1.0 - float(time_s) / float(t_scale)))
    b_m = min(1.0, max(0.0, 1.0 - float(mem_mb) / float(m_scale)))
    if b_t <= 0.0 or b_m <= 0.0:
        return 0.0
    xs = [0.0] + sorted({q for q, _, _ in front_points if 0.0 < q < 1.0}) + [1.0]
    ehvi = 0.0
    for r in range(len(xs) - 1):
        x_lo, x_hi = xs[r], xs[r + 1]
        if x_hi <= x_lo:
            continue
        active = [(t, m) for q, t, m in front_points if q >= x_hi - 1e-12]
        gain_2d = max(0.0, _area_2d(active + [(b_t, b_m)]) - _area_2d(active))
        if gain_2d > 0.0:
            slice_ei = max(0.0, _gaussian_ei(mu_q, sigma_q, x_lo) - _gaussian_ei(mu_q, sigma_q, x_hi))
            ehvi += gain_2d * slice_ei
    return float(max(0.0, ehvi))


def _rule_evaluator(protocol, objective, observations, q_star_override=None):
    rule = protocol.get("selection_rule", "weighted")
    weights = [w / sum(objective.weights) for w in objective.weights]
    t_scale = float(objective.time_scale)
    m_scale = float(objective.memory_scale)

    def losses_of(q, t, m):
        return (1.0 - float(q), float(t) / t_scale, float(m) / m_scale)

    def tcheb_val(q, t, m):
        xs = losses_of(q, t, m)
        return -max(w * x for w, x in zip(weights, xs)) - 1e-6 * sum(w * x for w, x in zip(weights, xs))

    if rule == "weighted":
        return (lambda q, t, m: bounded(objective.value({"quality_mean": q, "time_mean": t, "memory_max": m}))), None
    if rule == "epsilon_constraint":
        eps = max(0.0, float(protocol.get("quality_tolerance", 0.03)))
        q_star = max((float(obs["quality_mean"]) for obs in observations if obs.get("feasible")), default=0.0)
        if q_star_override is not None:
            q_star = max(q_star, float(q_star_override))
        wt, wm = weights[1], weights[2]
        denom_tm = wt + wm
        def eps_val(q, t, m):
            _, tl, ml = losses_of(q, t, m)
            c_cost = (wt * tl + wm * ml) / denom_tm if denom_tm > 1e-12 else 0.5 * (tl + ml)
            gap = max(0.0, q_star - float(q))
            if gap <= eps + 1e-12:
                return bounded(1.0 - (c_cost / (1.0 + max(0.0, c_cost))) + 1e-4 * float(q))
            return bounded(-gap - 0.1 * (c_cost / (1.0 + max(0.0, c_cost))))
        return eps_val, None
    if rule == "tchebycheff":
        return tcheb_val, None
    if rule == "topsis":
        def topsis_val(q, t, m):
            a, tl, ml = losses_of(q, t, m)
            xs = (a, tl / (1.0 + max(0.0, tl)), ml / (1.0 + max(0.0, ml)))
            good = math.sqrt(sum((w * x) ** 2 for w, x in zip(weights, xs)))
            bad = math.sqrt(sum((w * (1.0 - x)) ** 2 for w, x in zip(weights, xs)))
            return bad / (good + bad) if (good + bad) else 0.5
        return topsis_val, None
    feasible = [obs for obs in observations if obs["feasible"]]
    front = nondominated(feasible) if feasible else []
    extremes = [min(front, key=lambda r: (losses_of(r["quality_mean"], r["time_mean"], r["memory_max"])[axis], r["config_id"])) for axis in range(3)] if front else []
    if len({r["config_id"] for r in extremes}) < 3:
        return tcheb_val, "degenerate Pareto extremes; declared tchebycheff fallback"
    a, b, c = [losses_of(r["quality_mean"], r["time_mean"], r["memory_max"]) for r in extremes]
    u, v = [b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)]
    normal = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
    length = math.sqrt(sum(x * x for x in normal))
    intercept = sum(x * y for x, y in zip(normal, a))
    if length < 1e-12 or abs(intercept) < 1e-12:
        return tcheb_val, "degenerate knee plane; declared tchebycheff fallback"
    direction = 1.0 if intercept > 0 else -1.0
    return (lambda q, t, m: direction * (intercept - sum(x * y for x, y in zip(normal, losses_of(q, t, m)))) / length), None


def _fit_heteroscedastic_gp_loocv(protocol, spec, distances, observations, train_indices, y, prior, sample_variance):
    n_configs = len(protocol["configs"])
    count = len(observations)
    kernel_opt = spec.get("kernel", "auto")
    default_kernel = "matern52" if kernel_opt == "auto" else kernel_opt
    heteroscedastic = len(protocol["training_seeds"]) >= 2
    eta = np.asarray(
        [min(0.25, max(0.0, float(obs.get("quality_mean_var", 0.0)))) if heteroscedastic else 0.0 for obs in observations],
        dtype=float,
    )

    if count == 0:
        means = [0.5] * n_configs
        stds = [0.5] * n_configs
        leverages = [1.0] * n_configs
        return {
            "fitted": False, "fit_warning": None,
            "selected_kernel": default_kernel, "selected_ls": float(spec["length_scale"]),
            "selected_ridge": float(spec["kernel_ridge"]), "loocv_mae": None, "loocv_rmse": None,
            "heteroscedastic": heteroscedastic, "noise_variances": [], "grid_size_evaluated": 0,
            "residual_variance": 0.0, "means": means, "stds": stds, "leverages": leverages,
        }

    if count == 1:
        selected_ls = float(spec["length_scale"])
        selected_ridge = float(spec["kernel_ridge"])
        k_full = _kernel_matrix(distances, default_kernel, selected_ls)
        denom = 1.0 + selected_ridge + float(eta[0])
        means, stds, leverages = [], [], []
        for j in range(n_configs):
            k_j0 = float(k_full[j, train_indices[0]])
            lev = float(np.clip(1.0 - (k_j0 * k_j0) / max(1e-12, denom), 0.0, 1.0))
            means.append(float(np.clip(prior, 0.0, 1.0)))
            leverages.append(lev)
            stds.append(float(math.sqrt(max(1e-4, 0.25 * lev))))
        return {
            "fitted": False, "fit_warning": None,
            "selected_kernel": default_kernel, "selected_ls": selected_ls,
            "selected_ridge": selected_ridge, "loocv_mae": None, "loocv_rmse": None,
            "heteroscedastic": heteroscedastic, "noise_variances": [float(v) for v in eta],
            "grid_size_evaluated": 0, "residual_variance": 0.0,
            "means": means, "stds": stds, "leverages": leverages,
        }

    kernels = ("matern52", "rbf") if kernel_opt == "auto" else (kernel_opt,)
    ls_grid = tuple(float(v) for v in spec["length_scale_grid"]) if "length_scale_grid" in spec else tuple(sorted({0.25, 0.5, 1.0, 2.0, float(spec["length_scale"])}))
    ridge_grid = tuple(float(v) for v in spec["ridge_grid"]) if "ridge_grid" in spec else tuple(sorted({1e-3, 1e-2, 5e-2, 1e-1, 5e-1, float(spec["kernel_ridge"])}))
    y_centered = y - prior
    eye_n = np.eye(count, dtype=float)
    best = None
    evaluated_count = 0
    for kernel_name in kernels:
        for ls in ls_grid:
            k_full = _kernel_matrix(distances, kernel_name, ls)
            if not np.isfinite(k_full).all():
                continue
            k_train = k_full[np.ix_(train_indices, train_indices)]
            for ridge in ridge_grid:
                diag_noise = ridge + eta
                gram = k_train + np.diag(diag_noise)
                try:
                    cf = cho_factor(gram, lower=True, check_finite=False)
                    alpha = cho_solve(cf, y_centered, check_finite=False)
                    a_inv = cho_solve(cf, eye_n, check_finite=False)
                except (np.linalg.LinAlgError, ValueError):
                    continue
                if not (np.isfinite(alpha).all() and np.isfinite(a_inv).all()):
                    continue
                diag_inv = np.maximum(1e-12, np.diag(a_inv))
                e_loo = alpha / diag_inv
                y_loo = np.clip(y - e_loo, 0.0, 1.0)
                rmse = float(math.sqrt(float(np.mean((y - y_loo) ** 2))))
                mae = float(np.mean(np.abs(y - y_loo)))
                if not (math.isfinite(rmse) and math.isfinite(mae)):
                    continue
                evaluated_count += 1
                sel_key = (round(rmse, 12), round(mae, 12), 0 if kernel_name == "matern52" else 1, float(ls), float(ridge))
                if best is None or sel_key < best[0]:
                    best = (sel_key, kernel_name, float(ls), float(ridge), rmse, mae, k_full, cf, alpha)

    if best is None:
        return {
            "fitted": False,
            "fit_warning": "numerically unavailable kernel solve: deterministic novelty/cost fallback",
            "selected_kernel": default_kernel, "selected_ls": float(spec["length_scale"]),
            "selected_ridge": float(spec["kernel_ridge"]), "loocv_mae": None, "loocv_rmse": None,
            "heteroscedastic": heteroscedastic, "noise_variances": [float(v) for v in eta],
            "grid_size_evaluated": evaluated_count, "residual_variance": 0.0,
            "means": [float(np.clip(prior, 0.0, 1.0))] * n_configs,
            "stds": [float(math.sqrt(max(1e-4, sample_variance)))] * n_configs,
            "leverages": [1.0] * n_configs,
        }

    _, sel_kernel, sel_ls, sel_ridge, loocv_rmse, loocv_mae, k_full, cf, alpha = best
    try:
        projected = cho_solve(cf, k_full[train_indices, :], check_finite=False)
        if not np.isfinite(projected).all():
            raise np.linalg.LinAlgError("nonfinite projection")
    except (np.linalg.LinAlgError, ValueError):
        return {
            "fitted": False,
            "fit_warning": "numerically unavailable kernel solve: deterministic novelty/cost fallback",
            "selected_kernel": sel_kernel, "selected_ls": sel_ls, "selected_ridge": sel_ridge,
            "loocv_mae": None, "loocv_rmse": None, "heteroscedastic": heteroscedastic,
            "noise_variances": [float(v) for v in eta], "grid_size_evaluated": evaluated_count,
            "residual_variance": 0.0,
            "means": [float(np.clip(prior, 0.0, 1.0))] * n_configs,
            "stds": [float(math.sqrt(max(1e-4, sample_variance)))] * n_configs,
            "leverages": [1.0] * n_configs,
        }

    residuals = y - (prior + k_full[np.ix_(train_indices, train_indices)] @ alpha)
    residual_variance = float(np.mean(residuals ** 2))
    sigma_f2 = max(sample_variance, loocv_rmse ** 2, float(np.mean(eta)), 1e-4)
    means, stds, leverages = [], [], []
    for j in range(n_configs):
        cross = k_full[j, train_indices]
        mean_j = float(np.clip(prior + cross @ alpha, 0.0, 1.0))
        lev_j = float(np.clip(1.0 - cross @ projected[:, j], 0.0, 1.0))
        std_j = float(math.sqrt(max(1e-8, sigma_f2 * lev_j)))
        means.append(mean_j)
        leverages.append(lev_j)
        stds.append(std_j)
    return {
        "fitted": True, "fit_warning": None,
        "selected_kernel": sel_kernel, "selected_ls": sel_ls, "selected_ridge": sel_ridge,
        "loocv_mae": loocv_mae, "loocv_rmse": loocv_rmse,
        "heteroscedastic": heteroscedastic, "noise_variances": [float(v) for v in eta],
        "grid_size_evaluated": evaluated_count, "residual_variance": residual_variance,
        "means": means, "stds": stds, "leverages": leverages,
    }


def rank_candidates(protocol, rows, forecasts, candidates=None, order=None,
                    use_cost=True, use_uncertainty=True, use_diversity=True,
                    acquisition_mode="auto", probe_priors=None, feasibility_weights=None):
    """Rank unfinished full confirmations; the coordinator owns admission/budget.

    candidates accepts indices or {index, missing_seeds/missing_seed_count}.
    forecasts accepts integer/string index keys with CostModel's point/planning
    seconds and predicted_memory_mb. Unknown forecasts never reject candidates:
    finite median placeholders and explicit reasons keep the fallback auditable.
    """
    spec = acquisition_spec(protocol)
    objective = Objective(**protocol["objective"])
    observations = full_observations(protocol, rows, objective)
    trained = {row["index"] for row in observations}
    row_map = {row["index"]: row for row in rows}
    if order is None:
        import random
        order = list(range(len(protocol["configs"])))
        random.Random(protocol["search_seed"]).shuffle(order)
    priority = {index: position for position, index in enumerate(order)}
    if candidates is None:
        candidates = [i for i in range(len(protocol["configs"])) if i not in trained]
    missing = {}
    for entry in candidates:
        index = entry if type(entry) is int else entry["index"]
        if type(index) is not int or not 0 <= index < len(protocol["configs"]):
            raise ValueError("Acquisition candidate outside locked grid")
        if index in trained:
            continue
        done = set(row_map.get(index, {}).get("completed_seeds", []))
        count = len(set(protocol["training_seeds"]) - done)
        if isinstance(entry, dict):
            if "missing_seeds" in entry:
                count = len(entry["missing_seeds"])
            count = entry.get("missing_seed_count", count)
        if type(count) is not int or count < 1:
            raise ValueError("Acquisition needs a positive missing-seed count")
        missing[index] = count
    distances, feature_metadata = configuration_distances(protocol)
    count = len(observations)
    train_indices = [row["index"] for row in observations]
    y = np.asarray([row["quality_mean"] for row in observations], dtype=float)
    prior = float(y.mean()) if count else .5
    sample_variance = float(np.var(y)) if count else 0.0
    incumbent = max((row["utility"] for row in observations if row["feasible"]), default=None)

    known_costs = []
    known_memories = [row["memory_max"] for row in observations]

    def worker_time(index, forecast):
        # Partial quality is never a surrogate label. Its already completed
        # full workers may nevertheless measure the cost of remaining seeds.
        partial = [run["worker_wall_seconds"] for run in row_map.get(index, {}).get("runs", [])
                   if run.get("seed") in protocol["training_seeds"] and complete(run, protocol["full_epochs"])]
        if partial:
            return statistics.mean(partial), "same-config mean completed full-worker cost"
        point = finite(forecast.get("point_seconds"), -1, positive=True)
        if point > 0:
            return point, "forecast point worker cost"
        return finite(forecast.get("planning_seconds"), -1, positive=True), "planning worker-cost fallback"

    for index in missing:
        forecast = forecasts.get(index, forecasts.get(str(index), {}))
        time, _ = worker_time(index, forecast)
        if time > 0:
            known_costs.append(time)
        memory = finite(forecast.get("predicted_memory_mb"), -1)
        if memory >= 0:
            known_memories.append(memory)
    default_time = median(known_costs) if known_costs else median(row["time_mean"] for row in observations) if observations else 1.0
    default_time = max(1e-9, default_time)
    default_memory = median(known_memories) if known_memories else objective.memory_scale

    effective_mode = "ehvi" if (acquisition_mode == "ehvi" or protocol.get("method") == "adaptive_ehvi") else "rule"
    use_v2 = spec["version"] >= 2 or effective_mode == "ehvi"

    if not use_v2:
        fitted = count >= 2
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            kernel = np.exp(-.5 * (distances / spec["length_scale"]) ** 2)
        alpha, projected, residual_variance, fit_warning = None, None, 0.0, None
        if fitted:
            gram = kernel[np.ix_(train_indices, train_indices)] + spec["kernel_ridge"] * np.eye(count)
            try:
                solved = np.linalg.solve(gram, np.column_stack((y - prior, kernel[:, train_indices].T)))
                if not np.isfinite(solved).all():
                    raise np.linalg.LinAlgError("nonfinite ridge solution")
                alpha, projected = solved[:, 0], solved[:, 1:].T
                residuals = y - (prior + kernel[np.ix_(train_indices, train_indices)] @ alpha)
                with np.errstate(over="ignore", invalid="ignore"):
                    residual_variance = float(np.mean(residuals ** 2))
                if not math.isfinite(residual_variance):
                    raise np.linalg.LinAlgError("nonfinite residual variance")
            except np.linalg.LinAlgError:
                fitted, fit_warning = False, "numerically unavailable kernel solve: deterministic novelty/cost fallback"
                residual_variance = 0.0
        scores = {}
        for index, need in missing.items():
            novelty = float(min(distances[index, train_indices])) if count else 1.0
            leverage = 1.0
            mean = prior
            if fitted:
                cross = kernel[index, train_indices]
                mean = float(np.clip(prior + cross @ alpha, 0, 1))
                leverage = float(np.clip(1 - cross @ projected[index], 0, 1))
            uncertainty = min(1.0, math.sqrt(leverage) + novelty * math.sqrt(sample_variance + residual_variance))
            quality_bonus = spec["exploration"] * uncertainty if use_uncertainty else 0.0
            optimistic = min(1.0, mean + quality_bonus)
            forecast = forecasts.get(index, forecasts.get(str(index), {}))
            time, cost_basis = worker_time(index, forecast)
            unknown_time = time <= 0
            time = default_time if unknown_time else time
            memory = finite(forecast.get("predicted_memory_mb"), -1)
            unknown_memory = memory < 0
            memory = default_memory if unknown_memory else memory
            optimistic_utility = bounded(objective.value({"quality_mean": optimistic, "time_mean": time, "memory_max": memory}))
            improvement = max(0, optimistic_utility - (incumbent if incumbent is not None else 0))
            diversity_bonus = spec["novelty_weight"] * novelty if use_diversity else 0.0
            numerator = max(1e-12, improvement + diversity_bonus) if fitted else 1.0 + diversity_bonus
            remaining_cost = max(1e-9, bounded(time * need))
            resource_excess = ((max(0, memory / objective.max_memory - 1) if objective.max_memory else 0)
                               + (max(0, time / objective.max_time - 1) if objective.max_time else 0))
            planning_penalty = 1 / (1 + resource_excess)
            score = bounded(numerator * planning_penalty / (remaining_cost if use_cost else 1.0))
            reasons = ["uncalibrated heuristic; not a Bayesian probability or upper bound"]
            if fit_warning:
                reasons.append(fit_warning)
            elif not fitted:
                reasons.append("fewer than two complete configurations: deterministic novelty/cost fallback")
            if unknown_time:
                reasons.append("unknown worker cost: median/default planning placeholder; coordinator must verify affordability")
            if unknown_memory:
                reasons.append("unknown memory: median/default planning placeholder, not feasibility evidence")
            if resource_excess:
                reasons.append("predicted resource constraint excess penalized heuristically, not rejected")
            scores[str(index)] = {"index": index, "score": float(score), "quality_mean_prediction": float(mean),
                                  "quality_optimistic": float(optimistic), "quality_bonus": float(quality_bonus),
                                  "uncertainty": float(uncertainty), "leverage": float(leverage), "novelty": novelty,
                                  "diversity_bonus": float(diversity_bonus), "optimistic_utility": float(optimistic_utility),
                                  "optimistic_improvement": float(improvement), "worker_cost_seconds": float(time),
                                  "cost_basis": cost_basis,
                                  "missing_seed_count": need, "remaining_all_seed_cost_seconds": float(remaining_cost),
                                  "memory_prediction_mb": float(memory), "unknown_cost": unknown_time,
                                  "unknown_memory": unknown_memory, "planning_resource_penalty": float(planning_penalty),
                                  "reasons": reasons, "quality_measured": False, "safety_guarantee": False}
        ranking = sorted(missing, key=lambda index: (-scores[str(index)]["score"], priority.get(index, len(priority) + index)))
        return {"ranking": ranking, "scores": scores,
                "surrogate": {"specification": spec, "fitted": fitted, "observed_configurations": count,
                              "training_indices": train_indices, "training_quality_mean": [float(value) for value in y],
                              "target": "complete full-data/full-horizon/all-required-seed mean validation quality only",
                              "quality_prior": prior, "residual_variance": residual_variance,
                              "sample_variance": sample_variance, "incumbent_utility": incumbent,
                              "fit_warning": fit_warning,
                              "features": feature_metadata,
                              "use_cost": bool(use_cost), "use_uncertainty": bool(use_uncertainty), "use_diversity": bool(use_diversity),
                              "uncertainty_definition": "clipped sqrt(kernel leverage) + nearest-distance * sqrt(observed quality variance + ridge residual variance)",
                              "calibration": "uncalibrated heuristic; no probabilistic or optimum guarantee",
                              "parameter_basis": "prelocked design choices, not learned universal optimal constants",
                              "limitations": ["Quality can be misleading at a short fixed horizon.",
                                              "Kernel geometry and uncertainty are uncalibrated, not a Bayesian posterior.",
                                              "Unknown cost/memory placeholders do not establish feasibility.",
                                              "No-diversity removes the explicit novelty bonus, not feature distances used by the surrogate.",
                                              "No test, reference or subset quality is an acquisition input."]}}

    n_configs = len(protocol["configs"])
    probe_items = (
        sorted((int(k), float(v)) for k, v in probe_priors.items()
               if 0 <= int(k) < n_configs and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v))
        if isinstance(probe_priors, dict) and probe_priors else []
    )
    if len(probe_items) >= 2:
        raw_vals = [p[1] for p in probe_items]
        pm = statistics.mean(raw_vals)
        ps = statistics.stdev(raw_vals)
        target_sd = max(ps, 0.075)
        rho = min(3.0, target_sd / max(1e-4, ps)) if ps > 1e-5 else 1.0
        probe_items = [
            (idx_p, float(min(0.98, max(0.01, pm + rho * (val_p - pm)))))
            for idx_p, val_p in probe_items
        ]
    m_probe = None
    lev_probe = None
    if probe_items:
        p_idx = [p[0] for p in probe_items]
        p_q = np.asarray([p[1] for p in probe_items], dtype=float)
        p_mean = float(np.mean(p_q))
        try:
            k_p = _kernel_matrix(distances, "matern52", 0.90)
            gram_p = k_p[np.ix_(p_idx, p_idx)] + 0.08 * np.eye(len(p_idx))
            solved_p = np.linalg.solve(gram_p, np.column_stack((p_q - p_mean, k_p[:, p_idx].T)))
            if np.isfinite(solved_p).all():
                alpha_p, proj_p = solved_p[:, 0], solved_p[:, 1:].T
                m_probe = np.clip(p_mean + k_p[:, p_idx] @ alpha_p, 0.0, 1.0)
                lev_probe = [float(np.clip(1.0 - float(k_p[j, p_idx] @ proj_p[j]), 0.0, 1.0)) for j in range(n_configs)]
        except (np.linalg.LinAlgError, ValueError):
            m_probe = None
        if m_probe is None:
            m_probe = np.full(n_configs, p_mean, dtype=float)
            for idx_p, q_p in probe_items:
                m_probe[idx_p] = float(np.clip(q_p, 0.0, 1.0))
            lev_probe = [0.2 if j in p_idx else 1.0 for j in range(n_configs)]

    if m_probe is not None and count == 0:
        p_var = float(np.var([p[1] for p in probe_items])) if len(probe_items) > 1 else 0.03
        gp = {
            "fitted": True, "fit_warning": None,
            "selected_kernel": "matern52_mf_probe", "selected_ls": 0.90,
            "selected_ridge": 0.08, "loocv_mae": None, "loocv_rmse": None,
            "heteroscedastic": False, "noise_variances": [], "grid_size_evaluated": 1,
            "residual_variance": p_var,
            "means": [float(v) for v in m_probe],
            "stds": [float(math.sqrt(max(1e-4, min(0.05, max(0.015, p_var)) * lv))) for lv in lev_probe],
            "leverages": lev_probe,
        }
    elif m_probe is not None and count == 1:
        p_var = float(np.var([p[1] for p in probe_items])) if len(probe_items) > 1 else 0.03
        delta_shift = float(np.mean(y - m_probe[train_indices]))
        m_prior_vec = np.clip(m_probe + delta_shift, 0.0, 1.0)
        gp = _fit_heteroscedastic_gp_loocv(protocol, spec, distances, observations, train_indices, y, prior, sample_variance)
        lev_joint = [min(float(lf), 0.5 * (float(lf) + float(lp))) for lf, lp in zip(gp["leverages"], lev_probe)]
        sig2_mf = min(0.04, max(0.012, p_var))
        gp["means"] = [float(v) for v in m_prior_vec]
        gp["leverages"] = lev_joint
        gp["stds"] = [float(math.sqrt(max(1e-4, sig2_mf * lj))) for lj in lev_joint]
        gp["selected_kernel"] = f"{gp['selected_kernel']}_mf_prior"
    elif m_probe is not None and count >= 2:
        delta_shift = float(np.mean(y - m_probe[train_indices]))
        m_prior_vec = np.clip(m_probe + delta_shift, 0.0, 1.0)
        y_res = y - m_prior_vec[train_indices]
        y_shifted = np.clip(0.5 + y_res, 0.0, 1.0)
        res_var = float(np.var(y_shifted))
        res_gp = _fit_heteroscedastic_gp_loocv(protocol, spec, distances, observations, train_indices, y_shifted, 0.5, res_var)
        gp = dict(res_gp)
        gp["selected_kernel"] = f"{res_gp['selected_kernel']}_mf_residual"
        gp["means"] = [float(np.clip(m_prior_vec[j] + (res_gp["means"][j] - 0.5), 0.0, 1.0)) for j in range(n_configs)]
    else:
        gp = _fit_heteroscedastic_gp_loocv(protocol, spec, distances, observations, train_indices, y, prior, sample_variance)
    fitted = gp["fitted"]
    fit_warning = gp["fit_warning"]
    residual_variance = gp["residual_variance"]

    selection_rule = protocol.get("selection_rule", "weighted")
    q_star_override = max((gp["means"][idx] for idx in missing), default=None) if m_probe is not None else None
    rule_fn, rule_fallback = _rule_evaluator(protocol, objective, observations, q_star_override=q_star_override)
    feasible_obs = [obs for obs in observations if obs["feasible"]]
    ref_rows = feasible_obs if feasible_obs else observations
    target_u = incumbent if incumbent is not None else max((row["utility"] if row["utility"] is not None else objective.value(row) for row in observations), default=0.0)
    target_r = max((rule_fn(r["quality_mean"], r["time_mean"], r["memory_max"]) for r in ref_rows), default=0.0)

    def _eps_post_gain(q_pt, t_pt, m_pt):
        eps_tol = max(0.0, float(protocol.get("quality_tolerance", 0.03)))
        q_obs_max = max((float(obs["quality_mean"]) for obs in ref_rows), default=0.0)
        weights_n = [w / sum(objective.weights) for w in objective.weights]
        wt, wm = weights_n[1], weights_n[2]
        denom_tm = wt + wm
        def _c_cost(t_v, m_v):
            tl = float(t_v) / float(objective.time_scale)
            ml = float(m_v) / float(objective.memory_scale)
            return (wt * tl + wm * ml) / denom_tm if denom_tm > 1e-12 else 0.5 * (tl + ml)
        if not ref_rows:
            q_anchor = float(q_star_override) if q_star_override is not None else float(q_pt)
            gap0 = max(0.0, q_anchor - float(q_pt))
            if gap0 > eps_tol + 1e-12:
                return 0.0
            cc = _c_cost(t_pt, m_pt)
            return max(0.0, 0.25 + 0.75 * float(q_pt) - 0.10 * (cc / (1.0 + max(0.0, cc))))
        q_world = max(q_obs_max, float(q_pt))
        def _s_at(qv, tv, mv):
            cc = _c_cost(tv, mv)
            gap = max(0.0, q_world - float(qv))
            if gap <= eps_tol + 1e-12:
                return bounded(1.0 - (cc / (1.0 + max(0.0, cc))) + 1e-4 * float(qv))
            return bounded(-gap - 0.1 * (cc / (1.0 + max(0.0, cc))))
        s_cand = _s_at(q_pt, t_pt, m_pt)
        s_inc = max(_s_at(r["quality_mean"], r["time_mean"], r["memory_max"]) for r in ref_rows)
        q_break = max(0.0, float(q_pt) - q_obs_max)
        return max(0.0, s_cand - s_inc) + q_break

    all_times = [obs["time_mean"] for obs in observations] + known_costs + [default_time]
    all_memories = [obs["memory_max"] for obs in observations] + known_memories + [default_memory]
    max_all_t = max(all_times) if all_times else float(objective.time_scale)
    max_all_m = max(all_memories) if all_memories else float(objective.memory_scale)
    eff_t_scale = max(float(objective.time_scale), 1.25 * max_all_t) if max_all_t > float(objective.time_scale) else float(objective.time_scale)
    eff_m_scale = max(float(objective.memory_scale), 1.25 * max_all_m) if max_all_m > float(objective.memory_scale) else float(objective.memory_scale)
    pareto_obs = nondominated(feasible_obs) if feasible_obs else []
    front_points = [
        (min(1.0, max(0.0, float(obs["quality_mean"]))),
         min(1.0, max(0.0, 1.0 - float(obs["time_mean"]) / eff_t_scale)),
         min(1.0, max(0.0, 1.0 - float(obs["memory_max"]) / eff_m_scale)))
        for obs in pareto_obs
    ]
    weights = [w / sum(objective.weights) for w in objective.weights]
    w_q = weights[0]

    scores = {}
    for index, need in missing.items():
        novelty = float(min(distances[index, train_indices])) if count else 1.0
        mean = gp["means"][index]
        std_j = gp["stds"][index]
        leverage = gp["leverages"][index]
        uncertainty = min(1.0, std_j)
        eff_std = std_j if use_uncertainty else 0.0
        quality_bonus = spec["exploration"] * uncertainty if use_uncertainty else 0.0
        optimistic = min(1.0, mean + quality_bonus)

        forecast = forecasts.get(index, forecasts.get(str(index), {}))
        time, cost_basis = worker_time(index, forecast)
        unknown_time = time <= 0
        time = default_time if unknown_time else time
        memory = finite(forecast.get("predicted_memory_mb"), -1)
        unknown_memory = memory < 0
        memory = default_memory if unknown_memory else memory

        optimistic_utility = bounded(objective.value({"quality_mean": optimistic, "time_mean": time, "memory_max": memory}))
        improvement = max(0.0, optimistic_utility - (incumbent if incumbent is not None else 0.0))

        if objective.min_quality > 0:
            if use_uncertainty and std_j > 1e-12:
                p_feas_q = float(ndtr((mean - float(objective.min_quality)) / std_j))
            else:
                p_feas_q = 1.0 if mean >= objective.min_quality else 0.05
        else:
            p_feas_q = 1.0

        ehvi = _exact_3d_ehvi(mean, eff_std, time, memory, front_points, eff_t_scale, eff_m_scale)
        if effective_mode == "ehvi":
            ucb_ehvi = _exact_3d_ehvi(optimistic, 0.0, time, memory, front_points, eff_t_scale, eff_m_scale)
            ei_val = ehvi
            ucb_imp = max(0.0, ucb_ehvi - ehvi)
            gain = (ehvi + 0.1 * ucb_imp) * p_feas_q
        elif selection_rule == "weighted":
            mu_u = bounded(objective.value({"quality_mean": mean, "time_mean": time, "memory_max": memory}))
            sigma_u = w_q * eff_std
            ei_val = _gaussian_ei(mu_u, sigma_u, target_u)
            ucb_u = bounded(objective.value({"quality_mean": optimistic, "time_mean": time, "memory_max": memory}))
            ucb_imp = max(0.0, ucb_u - target_u)
            gain = (ei_val + 0.25 * ucb_imp) * p_feas_q
        else:
            if selection_rule == "epsilon_constraint":
                def _imp_at(q_pt):
                    return _eps_post_gain(q_pt, time, memory)
            else:
                def _imp_at(q_pt):
                    return max(0.0, rule_fn(q_pt, time, memory) - target_r)
            if use_uncertainty and std_j > 1e-12:
                ei_val = 0.0
                for node, weight in zip(_GH_NODES, _GH_WEIGHTS):
                    q_pt = min(1.0, max(0.0, mean + _GH_SQRT2 * std_j * float(node)))
                    ei_val += float(weight) * _imp_at(q_pt)
                ei_val *= _GH_INV_SQRT_PI
            else:
                ei_val = _imp_at(mean)
            ucb_imp = _imp_at(optimistic)
            gain = (ei_val + 0.25 * ucb_imp + 0.5 * ehvi) * p_feas_q

        diversity_bonus = (
            spec["novelty_weight"] * novelty * (0.15 if count == 0 else min(1.0, gain / 0.02))
            if (use_diversity and m_probe is not None)
            else (spec["novelty_weight"] * novelty if use_diversity else 0.0)
        )
        numerator = max(1e-12, gain + diversity_bonus) if (fitted or count == 1) else (1.0 + diversity_bonus)
        remaining_cost = max(1e-9, bounded(time * need))
        resource_excess = ((max(0.0, memory / objective.max_memory - 1.0) if objective.max_memory else 0.0)
                           + (max(0.0, time / objective.max_time - 1.0) if objective.max_time else 0.0))
        planning_penalty = 1.0 / (1.0 + resource_excess)
        feas_w = float(feasibility_weights.get(index, 1.0)) if isinstance(feasibility_weights, dict) else 1.0
        feas_w = min(1.0, max(1e-4, feas_w))
        score = bounded(feas_w * numerator * planning_penalty / (remaining_cost if use_cost else 1.0))

        reasons = ["heteroscedastic GP/adaptive kernel ridge LOOCV surrogate; not a statistical safety guarantee"]
        if m_probe is not None:
            reasons.append("multi-fidelity probe prior residual GP adjustment; final selection still requires full confirmations")
        if feasibility_weights is not None:
            reasons.append(f"feasibility-weighted expected utility p_feas={feas_w:.3f}")
        if fit_warning:
            reasons.append(fit_warning)
        elif not fitted:
            reasons.append("fewer than two complete configurations: deterministic prior/novelty/cost fallback")
        if unknown_time:
            reasons.append("unknown worker cost: median/default planning placeholder; coordinator must verify affordability")
        if unknown_memory:
            reasons.append("unknown memory: median/default planning placeholder, not feasibility evidence")
        if resource_excess:
            reasons.append("predicted resource constraint excess penalized heuristically, not rejected")
        if rule_fallback:
            reasons.append(rule_fallback)
        scores[str(index)] = {
            "index": index, "score": float(score), "quality_mean_prediction": float(mean),
            "quality_optimistic": float(optimistic), "quality_bonus": float(quality_bonus),
            "uncertainty": float(uncertainty), "leverage": float(leverage), "novelty": novelty,
            "diversity_bonus": float(diversity_bonus), "optimistic_utility": float(optimistic_utility),
            "optimistic_improvement": float(improvement), "ei": float(ei_val), "ehvi": float(ehvi),
            "ucb_improvement": float(ucb_imp), "p_quality_feasible": float(p_feas_q),
            "selection_rule": selection_rule, "worker_cost_seconds": float(time),
            "cost_basis": cost_basis, "missing_seed_count": need,
            "remaining_all_seed_cost_seconds": float(remaining_cost),
            "memory_prediction_mb": float(memory), "unknown_cost": unknown_time,
            "unknown_memory": unknown_memory, "planning_resource_penalty": float(planning_penalty),
            "conformal_feasibility_weight": float(feas_w),
            "reasons": reasons, "quality_measured": False, "safety_guarantee": False,
        }

    ranking = sorted(missing, key=lambda index: (-scores[str(index)]["score"], priority.get(index, len(priority) + index)))
    surrogate_diagnostics = {
        "kernel": gp["selected_kernel"],
        "length_scale": float(gp["selected_ls"]),
        "ridge": float(gp["selected_ridge"]),
        "loocv_mae": gp["loocv_mae"],
        "loocv_rmse": gp["loocv_rmse"],
        "heteroscedastic": gp["heteroscedastic"],
        "noise_variances": gp["noise_variances"],
        "grid_size_evaluated": gp["grid_size_evaluated"],
        "predictions": {
            str(idx): {"mean": float(gp["means"][idx]), "std": float(gp["stds"][idx])}
            for idx in range(len(protocol["configs"]))
        },
    }
    return {
        "ranking": ranking, "scores": scores,
        "surrogate_diagnostics": surrogate_diagnostics,
        "surrogate": {
            "specification": spec, "fitted": fitted, "observed_configurations": count,
            "training_indices": train_indices, "training_quality_mean": [float(value) for value in y],
            "target": "complete full-data/full-horizon/all-required-seed mean validation quality only",
            "quality_prior": prior, "residual_variance": residual_variance,
            "sample_variance": sample_variance, "incumbent_utility": incumbent,
            "fit_warning": fit_warning, "features": feature_metadata,
            "kernel": gp["selected_kernel"], "length_scale": float(gp["selected_ls"]),
            "ridge": float(gp["selected_ridge"]), "loocv_mae": gp["loocv_mae"],
            "loocv_rmse": gp["loocv_rmse"], "heteroscedastic": gp["heteroscedastic"],
            "selection_rule": selection_rule, "acquisition_mode": effective_mode,
            "use_cost": bool(use_cost), "use_uncertainty": bool(use_uncertainty), "use_diversity": bool(use_diversity),
            "uncertainty_definition": "heteroscedastic GP posterior std sqrt(sigma_f^2 * leverage) via exact analytic LOOCV",
            "calibration": "LOOCV-selected empirical kernel heuristic; no probabilistic or optimum guarantee",
            "parameter_basis": "prelocked finite kernel/length_scale/ridge grid selected deterministically by analytic LOOCV",
            "limitations": ["Quality can be misleading at a short fixed horizon.",
                            "Surrogate uncertainty is conditioned on observed seeds/configs, not an out-of-domain guarantee.",
                            "Unknown cost/memory placeholders do not establish feasibility.",
                            "No-diversity removes the explicit novelty bonus, not feature distances used by the surrogate.",
                            "No test, reference or subset quality is an acquisition input."],
        },
    }
