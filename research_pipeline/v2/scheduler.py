"""All-seed confirmations under ONE session budget, without oracle access."""

import math
import random
import time

from yolo_tuner.research import Objective
from .predictors import CostModel
from .model_features import descriptor_for, family_scale
from .schema import complete, decision, measurements
from .selection import select

ADAPTIVE_METHODS = ("adaptive_full", "adaptive_no_cost", "adaptive_no_uncertainty", "adaptive_no_diversity")
METHODS = (*ADAPTIVE_METHODS, "reserve", "random", "default", "profile_all", "random_proxy", "quality_sh", "utility_sh", "no_reserve", "successive_halving", "prior_full")


def diversified_order(configs, seed):
    order = list(range(len(configs)))
    random.Random(seed).shuffle(order)
    groups = {}
    for i in order:
        c = configs[i]
        groups.setdefault((c["model"], c["lr0"], c["imgsz"]), []).append(i)
    output = []
    while any(groups.values()):
        for group in groups.values():
            if group:
                output.append(group.pop())
    return output


def exploration_fits(remaining, action_cost, reserved_cost, cleanup):
    return action_cost is not None and all(math.isfinite(v) and v >= 0 for v in (remaining, action_cost, reserved_cost, cleanup)) and action_cost + reserved_cost + cleanup <= remaining


def small_first_key(protocol, index):
    """Measured architecture descriptors first; metadata fallback is not a cost bound."""
    config = protocol["configs"][index]
    descriptor = descriptor_for(protocol, config)
    parameters = descriptor.get("parameters")
    known = isinstance(parameters, (int, float)) and math.isfinite(parameters) and parameters > 0
    family, scale = family_scale(config)
    size = parameters if known else config.get("width", scale or 16)
    return (not known, size, config["imgsz"], config["batch"], family, scale)


def run_search(protocol, evaluate, remaining, ledger, initial=None, on_change=None, elapsed=None):
    """evaluate(index, seed, epochs, full, worker_seconds) has no oracle/test inputs."""
    configs, seeds = protocol["configs"], protocol["training_seeds"]
    method = protocol["method"]
    if method not in METHODS:
        raise ValueError("Unknown v2 search method")
    objective = Objective(**protocol["objective"])
    state = initial or {"results": {str(i): [] for i in range(len(configs))}, "probes": [],
                        "actions": [], "ordering": diversified_order(configs, protocol["search_seed"])}
    cost_model = CostModel(protocol)
    for rows in state["results"].values():
        for result in rows:
            if method not in ADAPTIVE_METHODS or complete(result, protocol["full_epochs"]):
                cost_model.update(result["index"], result)
    if method not in ADAPTIVE_METHODS:
        for result in state["probes"]:
            cost_model.update(result["index"], result)
    cleanup = protocol["scheduler"]["cleanup_seconds"]
    protected = protocol["scheduler"].get("policy_version", 1) >= 2
    probe_spent = (state.get("probe_budget_spent_seconds", sum(r["worker_wall_seconds"] for r in state["probes"]))
                   if protected else sum(r["worker_wall_seconds"] for r in state["probes"]))
    order = list(state["ordering"])
    if method == "default":
        order = [0]
    elif method in {"random", "random_proxy", "quality_sh", "utility_sh", "profile_all"}:
        random.Random(protocol["search_seed"]).shuffle(order)

    def save():
        if on_change:
            on_change(state)

    def attempt(i, seed, full, seconds=None, epochs=None):
        nonlocal probe_spent
        started_remaining = remaining()
        started_elapsed = elapsed() if elapsed is not None else None
        def elapsed_charge():
            return max(0, elapsed() - started_elapsed) if elapsed is not None else max(0, started_remaining - remaining())
        available = min(remaining() - cleanup, protocol["trial_timeout"])
        if seconds is not None:
            available = min(available, seconds)
        if available < 0.05:
            return None
        prediction = cost_model.estimate(i)
        try:
            result = evaluate(i, seed, protocol["full_epochs"] if full else epochs or protocol["probe_epochs"], full, available)
        except TimeoutError:
            if protected and not full:
                probe_spent += elapsed_charge()
                state["probe_budget_spent_seconds"] = probe_spent
            ledger.emit("candidate_unresolved", index=i, reason="budget exhausted during preflight, no worker started")
            return None
        if protected and not full:
            # Charge measured controller/preflight time too, not merely the
            # worker loop. Saved cumulative charge survives workflow resume.
            probe_spent += max(result["worker_wall_seconds"], elapsed_charge())
            state["probe_budget_spent_seconds"] = probe_spent
        result.update(index=i, seed=seed, mode="train" if full else "search", scheduling_prediction=prediction)
        if method not in ADAPTIVE_METHODS or full and complete(result, protocol["full_epochs"]):
            cost_model.update(i, result)
        if full:
            state["results"][str(i)].append(result)
        else:
            state["probes"].append(result)
        state["actions"].append({"index": i, "seed": seed, "full": full, "attempt_id": result.get("attempt_id"), "status": result["status"]})
        save()
        return result

    def missing(i):
        done = {r["seed"] for r in state["results"][str(i)] if complete(r, protocol["full_epochs"])}
        return [s for s in seeds if s not in done]

    def confirm(i):
        for seed in missing(i):
            rows = state["results"][str(i)]
            if any(r.get("status") in {"oom", "config_mismatch", "training_evidence_missing"} for r in rows):
                return
            cap = None
            if protected and (method == "successive_halving" or method in ADAPTIVE_METHODS):
                forecast = cost_model.estimate(i)
                plan = forecast["planning_seconds"]
                need = len(missing(i))
                if plan is not None and plan * need + cleanup > remaining():
                    ledger.emit("candidate_unresolved", index=i, reason="all-seed finalizer forecast exceeds remaining budget", forecast=forecast)
                    return
                if plan is None:
                    # With no evidence no policy can promise affordability.
                    # Share the unknown bootstrap budget across required seeds
                    # instead of letting one unknown worker consume it all.
                    cap = max(0, (remaining() - cleanup) / need)
                    ledger.emit("candidate_cost_unavailable", index=i, reason="unknown full cost; equal-share bootstrap cap is heuristic, not a guarantee", worker_cap_seconds=cap)
            result = attempt(i, seed, True, cap)
            if result is None or not complete(result, protocol["full_epochs"]):
                return

    if method in ADAPTIVE_METHODS:
        from .acquisition import rank_candidates
        # Each action is one complete E x missing-seeds confirmation, never a
        # low-fidelity quality rung. A timeout can be retried in a later resume,
        # but never in an unbounded loop during the same execution.
        attempted = set()
        history = state.setdefault("acquisition_history", [])
        while remaining() > cleanup:
            rows = measurements(configs, state["results"], seeds, protocol["full_epochs"], objective)
            eligible, deferred = [], []
            forecasts = {i: cost_model.estimate(i) for i in range(len(configs))}
            for i in order:
                prior = state["results"][str(i)]
                if i in attempted or not missing(i) or any(r.get("status") in {"oom", "config_mismatch", "training_evidence_missing"} for r in prior):
                    continue
                plan = forecasts[i]["planning_seconds"]
                censored = [r["worker_wall_seconds"] for r in prior if r.get("status") == "timeout" and r.get("worker_wall_seconds", 0) > 0]
                if censored:
                    # A killed worker gives only a time lower bound. Never let
                    # resume turn it into evidence that this full run is cheap.
                    lower = max(censored)
                    forecasts[i] = {**forecasts[i], "point_seconds": max(lower, forecasts[i]["point_seconds"] or 0),
                                    "planning_seconds": max(lower * protocol["scheduler"]["cost_margin"], plan or 0),
                                    "timeout_lower_bound_seconds": lower,
                                    "basis": forecasts[i]["basis"] + "; observed timeout lower bound, not completed full cost"}
                    plan = forecasts[i]["planning_seconds"]
                if plan is not None and plan * len(missing(i)) + cleanup > remaining():
                    deferred.append(i)
                    continue
                eligible.append({"index": i, "missing_seeds": missing(i)})
            if not eligible:
                ledger.emit("adaptive_stopped", reason="no remaining all-seed action has affordable forecast", deferred=deferred, upper_bound_guaranteed=False)
                break
            # Preserve previous partial full-data investment before acquiring
            # another configuration. Failed/incomplete observations never fit Q.
            partial = [c for c in eligible if any(complete(r, protocol["full_epochs"]) for r in state["results"][str(c["index"])])]
            if partial:
                chosen = partial[0]["index"]
                acquisition = {"ranking": [c["index"] for c in partial], "scores": {},
                               "surrogate": {"reason": "finish partially confirmed configuration first; no proxy quality input"}}
                reason = "partial_confirmation_priority"
            elif not any(row["complete"] for row in rows):
                # Cold start needs one measured incumbent. Seed order breaks
                # equal-metadata ties; no future validation/ref/table is read.
                chosen = min((c["index"] for c in eligible), key=lambda i: (small_first_key(protocol, i), order.index(i)))
                acquisition = {"ranking": [chosen], "scores": {},
                               "surrogate": {"reason": "cold start; metadata-small first and equal-share unknown worker cap"}}
                reason = "bootstrap_full_confirmation"
            else:
                acquisition = rank_candidates(protocol, rows, forecasts, candidates=eligible, order=order,
                                              use_cost=method != "adaptive_no_cost",
                                              use_uncertainty=method != "adaptive_no_uncertainty",
                                              use_diversity=method != "adaptive_no_diversity")
                chosen = acquisition["ranking"][0]
                reason = "adaptive_optimistic_acquisition"
            entry = {"step": len(history) + 1, "index": chosen, "reason": reason,
                     "missing_seeds": missing(chosen), "remaining_seconds": remaining(),
                     "deferred_unaffordable": deferred, **acquisition}
            history.append(entry)
            ledger.emit("adaptive_acquisition", **entry)
            save()
            attempted.add(chosen)
            confirm(chosen)
        # Avoid the generic one-pass scheduler; adaptive has already reacquired
        # after every complete configuration and stored its decision trace.
        order = []
    if method == "prior_full":
        # Scheduling only. Missing/OOD predictions keep metadata order; every
        # selected candidate still requires all full-data seeds.
        def prior_score(i):
            prediction = cost_model.estimate(i)
            if prediction.get("predicted_quality") is None or prediction["predicted_memory_mb"] is None:
                return -math.inf
            return objective.value({"quality_mean": prediction["predicted_quality"], "time_mean": prediction["point_seconds"], "memory_max": prediction["predicted_memory_mb"]})
        order.sort(key=prior_score, reverse=True)
    if method == "successive_halving":
        cohort = list(order)
        if protected:
            cohort.sort(key=lambda i: small_first_key(protocol, i))
            # An unknown-cost all-grid sweep can exhaust the session before
            # the first usable result. Bootstrap one complete incumbent first;
            # never present this fallback as a completed halving experiment.
            if cohort and not any(cost_model.estimate(i)["planning_seconds"] is not None for i in cohort):
                bootstrap = next((i for i in cohort if missing(i)), None)
                if bootstrap is not None:
                    ledger.emit("sh_bootstrap_confirmation", index=bootstrap, reason="no full-cost forecast; prioritize an all-seed incumbent before exploration", upper_bound_guaranteed=False)
                    confirm(bootstrap)

        def finalizer_reserve():
            plans = [(i, cost_model.estimate(i)["planning_seconds"]) for i in cohort if missing(i)]
            if any(plan is None for _, plan in plans):
                return None
            # Any retained candidate could win the rung: protect the most
            # expensive remaining all-seed finalizer, not just the cheapest.
            return max((plan * len(missing(i)) for i, plan in plans), default=0)

        def probe_forecast(i, epoch):
            observations = [r for r in state["probes"] if r["index"] == i and r["status"] == "ok" and r.get("epochs_completed", 0) > 0]
            margin = protocol["scheduler"]["cost_margin"]
            if observations:
                return max(r["worker_wall_seconds"] * max(1, epoch / r["epochs_completed"]) for r in observations) * margin
            estimate = cost_model.estimate(i)["planning_seconds"]
            if estimate is None:
                return None
            startup = max((max(0, r["worker_wall_seconds"] - r.get("measured_loop_seconds", 0))
                           for rs in cost_model.fulls.values() for r in rs), default=cleanup)
            return max(startup, estimate * epoch / protocol["full_epochs"])

        rungs = protocol["scheduler"].get("rungs", [protocol["probe_epochs"]])
        if protected and not any(missing(i) for i in cohort):
            rungs = []
        for epoch in rungs:
            if protected:
                pending = [i for i in cohort if not any(r["index"] == i and r.get("epochs_completed") == epoch and r["status"] == "ok" for r in state["probes"])]
                forecasts = [probe_forecast(i, epoch) for i in pending]
                reserve = finalizer_reserve()
                allowance = protocol["budget_seconds"] * protocol["scheduler"]["probe_fraction"] - probe_spent
                known = reserve is not None and all(v is not None for v in forecasts)
                planned = sum(forecasts) if known else None
                if pending and (not known or planned > allowance or not exploration_fits(remaining(), planned, reserve, cleanup)):
                    ledger.emit("rung_deferred", epochs=epoch, cohort=cohort,
                                reason="unknown cost" if not known else "probe cap or all-seed finalizer reserve does not fit",
                                planned_probe_seconds=planned, probe_allowance_seconds=max(0, allowance),
                                reserved_finalizer_seconds=reserve, upper_bound_guaranteed=False)
                    break
            measured = []
            for i in cohort:
                previous = next((r for r in reversed(state["probes"]) if r["index"] == i and r.get("epochs_completed") == epoch and r["status"] == "ok"), None)
                cap = None
                if protected and previous is None:
                    reserve = finalizer_reserve()
                    allowance = protocol["budget_seconds"] * protocol["scheduler"]["probe_fraction"] - probe_spent
                    cap = min(allowance, remaining() - cleanup - (reserve or 0)) if reserve is not None else 0
                    forecast = probe_forecast(i, epoch)
                    if forecast is None or cap < forecast:
                        break  # Actual elapsed costs may invalidate the rung plan.
                result = previous or attempt(i, seeds[0], False, cap, epoch)
                if result and result["status"] == "ok" and result.get("epochs_completed") == epoch and result.get("quality") is not None:
                    measured.append((i, result))
            if len(measured) != len(cohort):
                # Never compare mixed fidelity or pretend an incomplete rung
                # proves a discarded config worse. Keep the prior cohort.
                ledger.emit("rung_incomplete", epochs=epoch, cohort=cohort, completed=len(measured))
                break
            measured.sort(key=lambda pair: (pair[1]["quality"], -pair[0]), reverse=True)
            cohort = [i for i, _ in measured[:max(1, math.ceil(len(cohort) / 2))]]
            ledger.emit("rung_completed", epochs=epoch, retained=cohort, ranked=[i for i, _ in measured])
            save()
        order = cohort
    profile_methods = {"profile_all", "random_proxy", "quality_sh", "utility_sh"}
    if method in profile_methods:
        existing = {r["index"] for r in state["probes"]}
        for i in order:
            if i in existing:
                continue
            result = attempt(i, seeds[0], False)
            if result is None or result["status"] == "timeout":
                break
        successful = {r["index"]: r for r in state["probes"] if r["status"] == "ok" and r.get("quality") is not None}
        def score(i):
            result = successful.get(i)
            if not result:
                return -math.inf
            if method != "utility_sh":
                return result["quality"]
            forecast = cost_model.estimate(i)
            row = {"quality_mean": result["quality"], "time_mean": forecast["point_seconds"], "memory_max": result["memory_mb"]}
            return objective.value(row)
        if method in {"random_proxy", "quality_sh", "utility_sh"}:
            order.sort(key=score, reverse=True)
        if method in {"quality_sh", "utility_sh"}:
            # One predeclared rung, restart finalizer. No fake Hyperband label.
            order = order[:max(1, math.ceil(len(order) / 2))]

    if method == "reserve":
        # A priori small-model/image order, not historical oracle quality.
        order.sort(key=(lambda i: small_first_key(protocol, i)) if protected else (lambda i: (configs[i].get("width", 16), configs[i]["imgsz"], -configs[i]["batch"])))
        # Restore a partially confirmed incumbent FIRST on workflow resume.
        partial = [i for i in order if state["results"][str(i)] and missing(i)]
        order = partial + [i for i in order if i not in partial]
        # A single optional startup-aware probe, never an all-grid sweep.
        # It can warn that all-seed confirmation is unaffordable; that is an
        # unresolved scheduling decision, NOT proof of physical infeasibility.
        if not any(state["results"].values()) and not state["probes"]:
            cap = min(protocol["budget_seconds"] * protocol["scheduler"]["probe_fraction"], remaining() - cleanup)
            if cap >= cleanup and order:
                result = attempt(order[0], seeds[0], False, cap)
                if result and not protected:
                    probe_spent += result["worker_wall_seconds"]
    for i in order:
        if not missing(i):
            continue
        if remaining() <= cleanup:
            break
        prediction = cost_model.estimate(i)
        estimate = prediction["planning_seconds"]
        need = len(missing(i))
        rows = measurements(configs, state["results"], seeds, protocol["full_epochs"], objective)
        has_incumbent = decision(rows)["selected_index"] is not None
        if (method == "reserve" or protected and method == "successive_halving") and estimate is not None and estimate * need + cleanup > remaining():
            ledger.emit("candidate_unresolved", index=i, reason="estimated confirmations exceed remaining budget", forecast=prediction)
            continue
        if method == "reserve" and has_incumbent and i not in cost_model.fulls and i not in cost_model.probes:
            cap = protocol["budget_seconds"] * protocol["scheduler"]["probe_fraction"] - probe_spent
            # No all-grid probes. The reserve protects all missing seeds for THIS finalist.
            reserve = (estimate or 0) * need
            # A sub-startup cap would only kill the import before learning any
            # resource information. Use observed startup, not a GPU-name table.
            startup_floor = max((r["worker_wall_seconds"] - r.get("measured_loop_seconds", 0)
                                 for rs in cost_model.fulls.values() for r in rs), default=cleanup)
            if cap >= max(cleanup, startup_floor) and exploration_fits(remaining(), min(cap, estimate or cap), reserve, cleanup):
                result = attempt(i, seeds[0], False, cap)
                if result and not protected:
                    probe_spent += result["worker_wall_seconds"]
                prediction = cost_model.estimate(i)
                estimate = prediction["planning_seconds"]
                if estimate is not None and estimate * need + cleanup > remaining():
                    ledger.emit("candidate_unresolved", index=i, reason="post-probe confirmation reserve does not fit")
                    continue
        if method == "no_reserve":
            # First seed of every candidate before remaining seeds: intentional
            # ablation of complete-incumbent scheduling, not a production policy.
            first = missing(i)[0]
            attempt(i, first, True)
        else:
            confirm(i)
    if method == "no_reserve":
        for i in order:
            confirm(i)
    rows = measurements(configs, state["results"], seeds, protocol["full_epochs"], objective)
    state["decision"] = select(rows, protocol)
    state["cost_predictions"] = {str(i): cost_model.estimate(i) for i in range(len(configs))}
    save()
    return state
