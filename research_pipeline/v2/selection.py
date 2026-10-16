"""Predeclared decision rules over verified full measurements, never proxies."""

import math

from .schema import decision
from yolo_tuner.research import nondominated

RULES = ("weighted", "tchebycheff", "topsis", "knee")


def rule_scores(rows, protocol):
    feasible = [r for r in rows if r["state"] == "confirmed_feasible"]
    rule = protocol.get("selection_rule", "weighted")
    if rule not in RULES:
        raise ValueError("Unsupported selection rule")
    obj = protocol["objective"]
    weights = [w / sum(obj["weights"]) for w in obj["weights"]]
    losses = {r["config_id"]: (1 - r["quality_mean"], r["time_mean"] / obj["time_scale"], r["memory_max"] / obj["memory_scale"]) for r in feasible}
    if rule == "weighted":
        return {r["config_id"]: r["utility"] for r in feasible}, None
    if rule == "tchebycheff":
        return {k: -max(w * x for w, x in zip(weights, xs)) - 1e-6 * sum(w * x for w, x in zip(weights, xs)) for k, xs in losses.items()}, None
    if rule == "topsis":
        # Fixed anchors, bounded resource costs; no oracle min/max normalization.
        scores = {}
        for key, (a, t, m) in losses.items():
            xs = (a, t / (1 + t), m / (1 + m))
            good = math.sqrt(sum((w * x) ** 2 for w, x in zip(weights, xs)))
            bad = math.sqrt(sum((w * (1 - x)) ** 2 for w, x in zip(weights, xs)))
            scores[key] = bad / (good + bad) if good + bad else 0.5
        return scores, None
    # 3D knee: signed distance toward the ideal from the plane through the
    # three single-axis extremes of the OBSERVED feasible Pareto front.
    front = nondominated(feasible)
    extremes = [min(front, key=lambda r: (losses[r["config_id"]][axis], r["config_id"])) for axis in range(3)] if front else []
    if len({r["config_id"] for r in extremes}) < 3:
        return rule_scores(rows, {**protocol, "selection_rule": "tchebycheff"})[0], "degenerate Pareto extremes; declared tchebycheff fallback"
    a, b, c = [losses[r["config_id"]] for r in extremes]
    u, v = [b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)]
    normal = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
    length = math.sqrt(sum(x * x for x in normal))
    intercept = sum(x * y for x, y in zip(normal, a))
    if length < 1e-12 or abs(intercept) < 1e-12:
        return rule_scores(rows, {**protocol, "selection_rule": "tchebycheff"})[0], "degenerate knee plane; declared tchebycheff fallback"
    direction = 1 if intercept > 0 else -1
    return {r["config_id"]: direction * (intercept - sum(x * y for x, y in zip(normal, losses[r["config_id"]]))) / length for r in front}, None


def select(rows, protocol):
    result = decision(rows)
    if "selection_rule" not in protocol:
        return result  # Preserve historical schema4 publications byte-for-value.
    scores, fallback = rule_scores(rows, protocol)
    candidates = [r for r in rows if r["config_id"] in scores]
    if protocol.get("scheduler", {}).get("policy_version", 1) >= 2:
        # Zero weights can tie a dominated point with its dominator. Exclude
        # such points without changing the declared objective or its scores.
        candidates = nondominated(candidates)
    best = max(candidates, key=lambda r: (scores[r["config_id"]], r["config_id"]), default=None)
    result.update(selected_index=best["index"] if best else None, selected_config_id=best["config_id"] if best else None,
                  selection_rule=protocol["selection_rule"], selection_score=scores[best["config_id"]] if best else None,
                  selection_fallback=fallback)
    return result


def presets(rows, protocol):
    feasible = [r for r in rows if r["state"] == "confirmed_feasible"]
    if protocol.get("scheduler", {}).get("policy_version", 1) >= 2:
        feasible = nondominated(feasible)
    balanced = select(rows, protocol)["selected_config_id"]
    objectives = {"Accuracy": lambda r: r["quality_mean"], "Time": lambda r: -r["time_mean"], "Memory": lambda r: -r["memory_max"]}
    chosen = {name: max(feasible, key=lambda r: (metric(r), r["config_id"]), default=None) for name, metric in objectives.items()}
    chosen["Balanced"] = next((r for r in feasible if r["config_id"] == balanced), None)
    return {name: {"config_id": r["config_id"], "config": r["config"], "validation_mean": r["quality_mean"],
                   "time_mean_seconds": r["time_mean"], "memory_max_mib": r["memory_max"],
                   "full_seeds": r["completed_seeds"], "basis": "observed full confirmations; never predicted/test"} if r else None for name, r in chosen.items()}
