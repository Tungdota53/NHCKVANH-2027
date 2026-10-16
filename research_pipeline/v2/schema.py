"""Canonical protocol and strict complete-seed measurement semantics."""

from dataclasses import asdict
import math
from pathlib import Path
import statistics

from yolo_tuner.research import Objective, digest, nondominated

TASKS = {"detection", "classification", "captioning"}
METRICS = {"detection": "best validation mAP50-95 within declared horizon",
           "classification": "final epoch validation accuracy",
           "captioning": "best validation generated (BLEU4+chrF)/200 within declared horizon"}


def config_id(config):
    from .space import canonical
    return digest(canonical(config))[:12]


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def validate_protocol(p):
    if p.get("schema_version") != 4 or p.get("adapter") != "research_v2":
        raise ValueError("Not a v2 protocol")
    if p["task"] not in TASKS:
        raise ValueError("Unsupported research task")
    seeds = p["training_seeds"]
    if not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Required training seeds must be distinct integers")
    for name in ("budget_seconds", "trial_timeout", "full_epochs"):
        positive(p[name], name)
    if type(p["full_epochs"]) is not int or not 1 <= p["probe_epochs"] <= p["full_epochs"]:
        raise ValueError("Invalid full/probe horizon")
    positive(p["scheduler"]["cost_margin"], "cost_margin")
    if p["scheduler"]["cost_margin"] < 1 or not 0 <= p["scheduler"]["probe_fraction"] <= 0.5:
        raise ValueError("Invalid scheduler margin/probe fraction")
    positive(p["scheduler"]["cleanup_seconds"], "cleanup_seconds")
    if "policy_version" in p["scheduler"] and (type(p["scheduler"]["policy_version"]) is not int or p["scheduler"]["policy_version"] not in {1, 2}):
        raise ValueError("Unknown scheduler policy version")
    Objective(**p["objective"])
    if not p["configs"] or len(p["config_ids"]) != len(p["configs"]) or len(set(p["config_ids"])) != len(p["configs"]):
        raise ValueError("Empty/duplicate candidate space")
    if p["protocol_sha256"] != digest({k: v for k, v in p.items() if k != "protocol_sha256"}):
        raise ValueError("Protocol checksum mismatch")
    if "selection_rule" in p:
        from .selection import RULES
        if p["selection_rule"] not in RULES:
            raise ValueError("Unknown selection rule")
    from .scheduler import ADAPTIVE_METHODS
    if p.get("method") in ADAPTIVE_METHODS and p.get("selection_rule", "weighted") != "weighted":
        raise ValueError("adaptive_full acquisition currently supports weighted objective only; use baseline methods for other rules")
    if p.get("method") in ADAPTIVE_METHODS and "acquisition" not in p:
        raise ValueError("Adaptive method needs a locked acquisition specification")
    if "acquisition" in p:
        spec = p["acquisition"]
        if spec.get("version") != 1:
            raise ValueError("Unknown acquisition policy")
        for key in ("kernel_ridge", "length_scale"):
            positive(spec[key], key)
        for key in ("exploration", "novelty_weight"):
            value = spec[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
    if "rungs" in p["scheduler"]:
        rungs = p["scheduler"]["rungs"]
        if not rungs or any(type(e) is not int or not 1 <= e <= p["full_epochs"] for e in rungs) or sorted(set(rungs)) != rungs:
            raise ValueError("Rungs must be distinct increasing epochs within full horizon")
    if "execution_policy" in p:
        policy = p["execution_policy"]
        if policy.get("version") != 2 or policy.get("integrity") != "stat_cached_between_trials_full_hash_boundaries":
            raise ValueError("Unknown execution/integrity policy")
        loader = policy["data_loader"]
        workers = loader["workers"]
        if workers != "auto" and (type(workers) is not int or not 0 <= workers <= 8):
            raise ValueError("workers must be auto or an integer in 0..8")
        cache = loader["cache_mb"]
        if isinstance(cache, bool) or not isinstance(cache, (int, float)) or not math.isfinite(cache) or cache < 0:
            raise ValueError("Image cache must be finite and nonnegative")
        if loader["pin_memory"] not in {"auto", "on", "off"}:
            raise ValueError("pin_memory must be auto, on or off")


def complete(result, epochs):
    return (result.get("status") == "ok" and result.get("mode") == "train"
            and result.get("epochs_completed") == epochs
            and result.get("optimizer_steps", 0) > 0
            and result.get("training_loss_finite") is True
            and result.get("actual_config_verified") is True
            and result.get("checkpoint_sha256") is not None
            and all(isinstance(result.get(k), (float, int)) and math.isfinite(result[k]) for k in ("quality", "memory_mb", "worker_wall_seconds"))
            and 0 <= result["quality"] <= 1 and result["memory_mb"] >= 0 and result["worker_wall_seconds"] >= 0)


def measurements(configs, results, seeds, epochs, objective):
    rows = []
    for i, config in enumerate(configs):
        attempts = results.get(str(i), [])
        valid = {r["seed"]: r for r in attempts if r["seed"] in seeds and complete(r, epochs)}
        values = list(valid.values())
        row = {"index": i, "config_id": config_id(config), "config": config,
               "complete": set(valid) == set(seeds), "completed_seeds": sorted(valid),
               "required_seeds": seeds, "quality_mean": None, "quality_min": None,
               "time_mean": None, "time_max": None, "memory_max": None,
               "quality_sd": None, "runs": attempts, "utility": None, "state": "unresolved"}
        if values:
            q, t, m = [[r[k] for r in values] for k in ("quality", "worker_wall_seconds", "memory_mb")]
            row.update(quality_mean=statistics.mean(q), quality_min=min(q),
                       quality_sd=statistics.stdev(q) if len(q) > 1 else 0.0,
                       time_mean=statistics.mean(t), time_max=max(t), memory_max=max(m))
        if row["complete"]:
            row["state"] = "confirmed_feasible" if objective.feasible(row) else "confirmed_constraint_failure"
            if objective.feasible(row):
                row["utility"] = objective.value(row)
        elif any(r.get("status") == "oom" and r.get("mode") == "train" and r["seed"] in seeds for r in attempts):
            row["state"] = "observed_required_seed_oom"
        rows.append(row)
    return rows


def decision(rows):
    feasible = [r for r in rows if r["state"] == "confirmed_feasible"]
    best = max(feasible, key=lambda r: (r["utility"], r["config_id"])) if feasible else None
    # Require complete measurements even for failed constraints. OOM/timeouts do
    # not get a free optimum certificate in this conservative implementation.
    exhaustive = all(r["complete"] for r in rows)
    status = "empirical_optimal_in_declared_space" if best and exhaustive else "fully_trained_best_observed" if best else "no_verified_feasible_selection"
    return {"selected_index": best["index"] if best else None,
            "selected_config_id": best["config_id"] if best else None,
            "verification_status": status, "space_complete": exhaustive,
            "pareto": [r["config_id"] for r in nondominated(feasible)],
            "selection_basis": "full-data/full-horizon/all-required-seeds only; no test or oracle",
            "application_quality_established": False}


def lock_protocol(value):
    p = dict(value)
    p["schema_version"], p["adapter"] = 4, "research_v2"
    p["config_ids"] = [config_id(c) for c in p["configs"]]
    p["protocol_sha256"] = digest(p)
    validate_protocol(p)
    return p
