"""Finite-grid, cost-aware optimistic FULL-confirmation acquisition.

Kernel ridge is a quality point surrogate, not a Bayesian posterior or a safety
certificate. Leverage/distance/residual uncertainty is deliberately uncalibrated.
Only audited-style complete full-data/all-seed outcomes enter its quality fit;
neither probe quality nor a forecast's quality prior is a training target.
"""

import math
import statistics

import numpy as np

from yolo_tuner.research import Objective
from .model_features import descriptor_for, family_scale, model_identity
from .schema import complete


DEFAULT_SPEC = {"version": 1, "kernel_ridge": 0.05, "length_scale": 1.0,
                "exploration": 0.25, "novelty_weight": 0.05}


def acquisition_spec(protocol):
    """Read predeclared design choices; these defaults are not universal optima."""
    spec = {**DEFAULT_SPEC, **protocol.get("acquisition", {})}
    if type(spec["version"]) is not int or spec["version"] != 1:
        raise ValueError("Unsupported acquisition specification")
    for key in ("kernel_ridge", "length_scale", "exploration", "novelty_weight"):
        value = spec[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Acquisition {key} must be finite and nonnegative")
        if key in {"kernel_ridge", "length_scale"} and value <= 0:
            raise ValueError(f"Acquisition {key} must be positive")
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
             "parameters", "layers", "gflops_reference_32", "model_scale")
    for config in protocol["configs"]:
        descriptor = descriptor_for(protocol, config)
        family, scale = family_scale(config)
        values = []
        for name in names:
            value = descriptor.get(name, 0) if name in {"parameters", "layers", "gflops_reference_32"} else scale if name == "model_scale" else config.get(name, 0)
            value = finite(value)
            values.append(math.log(max(value, 1e-12)) if name == "lr0" else
                          value if name in {"dropout", "label_smoothing", "freeze", "model_scale"} else math.log1p(max(0, value)))
        numeric.append(values)
        categories.append((model_identity(config), str(family), str(config.get("encoder", "none")),
                           str(config.get("precision", "fp32")), str(config.get("optimizer", "AdamW")),
                           str(bool(config.get("pretrained", descriptor.get("pretrained_checkpoint", False))))))
    if not numeric:
        return np.zeros((0, 0)), {"numeric_fields": [], "categorical_fields": [], "normalization": "locked finite-grid ranges"}
    x = np.asarray(numeric, dtype=float)
    spans = x.max(axis=0) - x.min(axis=0)
    active = spans > 1e-12
    scaled = (x[:, active] - x[:, active].min(axis=0)) / spans[active] if active.any() else np.zeros((len(x), 0))
    distance_squared = np.sum((scaled[:, None, :] - scaled[None, :, :]) ** 2, axis=2)
    category_names = ("model_content_identity", "model_family", "encoder", "precision", "optimizer", "pretrained")
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
        target = {"complete": True, "quality_mean": statistics.mean(qualities), "quality_min": min(qualities),
                  "time_mean": statistics.mean(r["worker_wall_seconds"] for r in runs.values()),
                  "time_max": max(r["worker_wall_seconds"] for r in runs.values()),
                  "memory_max": max(r["memory_mb"] for r in runs.values())}
        observed[index] = {"index": index, **target, "feasible": objective.feasible(target),
                           "utility": bounded(objective.value(target))}
    return [observed[i] for i in sorted(observed)]


def rank_candidates(protocol, rows, forecasts, candidates=None, order=None,
                    use_cost=True, use_uncertainty=True, use_diversity=True):
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
    fitted = count >= 2
    train_indices = [row["index"] for row in observations]
    y = np.asarray([row["quality_mean"] for row in observations], dtype=float)
    prior = float(y.mean()) if count else .5
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        kernel = np.exp(-.5 * (distances / spec["length_scale"]) ** 2)
    alpha, projected, residual_variance, fit_warning = None, None, 0.0, None
    if fitted:
        gram = kernel[np.ix_(train_indices, train_indices)] + spec["kernel_ridge"] * np.eye(count)
        try:
            # Solve all leverage RHS together; do not repeat an O(n^3) solve for
            # every candidate. No stochastic model fitting or random restarts.
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
