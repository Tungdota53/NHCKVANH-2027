"""Predeclared scheduler ablations, without access to reference outcomes."""

import random

from yolo_tuner.core import Deadline
from yolo_tuner.research import nondominated, proxy_row, run_search, measured_quality

ABLATIONS = ("adaptive_no_profile", "adaptive_no_strata")
METHODS = ("default", "random", "random_proxy", "successive_halving", "pareto_racing", "adaptive_verified", *ABLATIONS)


def stratified_order(order, configs):
    groups = {}
    for index in order:
        groups.setdefault(configs[index]["lr0"], []).append(index)
    output = []
    while any(groups.values()):
        for group in groups.values():
            if group:
                output.append(group.pop(0))
    return output


def search(method, configs, evaluate, objective, seconds, rungs, epochs, seed, clock=None):
    if method not in ABLATIONS:
        kwargs = {"clock": clock} if clock else {}
        return run_search(method, configs, evaluate, objective, seconds, rungs, epochs, seed, **kwargs)
    deadline = Deadline(seconds, **({"clock": clock} if clock else {}))
    order = list(range(len(configs)))
    random.Random(seed).shuffle(order)
    history, confirmations, estimates = [], [], {}

    def observe(index, full):
        if deadline.remaining < 0.1:
            return None
        trial_epochs = epochs if full else max(rungs)
        result = evaluate(index, trial_epochs, deadline.remaining, full)
        row = {"config_id": f"c{index:02d}", "epochs": trial_epochs, "full_data": full,
               "status": result.get("status"), "quality": measured_quality(result),
               "estimated_time": result.get("worker_wall_seconds") if full else result.get("estimated_full_seconds"),
               "memory": result.get("memory_mb"), "worker_seconds": result.get("worker_wall_seconds"),
               "complete": result.get("status") == "ok" and result.get("epochs_completed") == trial_epochs,
               "result": result}
        history.append(row)
        return row

    if method == "adaptive_no_strata":
        for index in order:
            row = observe(index, False)
            if row and row["complete"]:
                estimates[index] = row["estimated_time"]
            if row is None or row["status"] == "timeout":
                break
    schedule = stratified_order(order, configs) if method == "adaptive_no_profile" else order
    for index in schedule:
        estimate = estimates.get(index)
        if estimate is not None and deadline.remaining < estimate * 1.25 + 2:
            continue
        row = observe(index, True)
        if row and row["complete"] and row["quality"] is not None:
            confirmations.append(row)
        if row is None or row["status"] == "timeout":
            break
    feasible = [r for r in confirmations if objective.feasible(proxy_row(r))]
    selected = max(feasible, key=lambda r: objective.value(proxy_row(r)))["config_id"] if feasible else None
    return {"method": method, "seed": seed, "selected": selected,
            "finalists": [r["config_id"] for r in nondominated([proxy_row(r) for r in feasible])],
            "history": history, "search_seconds": deadline.elapsed, "budget_seconds": seconds,
            "budget_overrun_seconds": max(0, deadline.elapsed - seconds),
            "selection_basis": "completed full-data confirmations only; predeclared component ablation"}
