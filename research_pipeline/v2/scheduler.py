"""All-seed confirmations under ONE session budget, without oracle access."""

import math
import random
import statistics
import time

from yolo_tuner.research import Objective
from .predictors import CostModel
from .model_features import descriptor_for, family_scale
from .schema import complete, config_id, decision, measurements
from .selection import select

ADAPTIVE_METHODS = (
    "adaptive_full",
    "adaptive_ehvi",
    "adaptive_staged_seeds",
    "adaptive_no_cost",
    "adaptive_no_uncertainty",
    "adaptive_no_diversity",
)
METHODS = (*ADAPTIVE_METHODS, "conformal_mf", "mf_no_conformal", "constrained_bo", "tpe_bo", "reserve", "random", "default", "profile_all", "random_proxy", "quality_sh", "utility_sh", "no_reserve", "successive_halving", "prior_full")


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


def _effective_memory_limit(objective, protocol):
    """Return (limit_mb, limit_kind) from objective.max_memory or 0.98 * vram_free_mb fallback."""
    if objective.max_memory is not None:
        return float(objective.max_memory), "max_memory"
    vram_free = protocol.get("hardware", {}).get("vram_free_mb")
    if (isinstance(vram_free, (int, float)) and not isinstance(vram_free, bool)
            and math.isfinite(vram_free) and vram_free > 0):
        return float(vram_free) * 0.98, "vram_free"
    return None, None


def _memory_risk(forecast, objective, protocol, prior_runs=None):
    """Classify predicted or observed memory risk relative to objective/VRAM limits."""
    limit, limit_kind = _effective_memory_limit(objective, protocol)
    if limit is not None and prior_runs:
        obs_mems = [float(r["memory_mb"]) for r in prior_runs
                    if r.get("mode") == "train" and r.get("status") == "ok"
                    and isinstance(r.get("memory_mb"), (int, float)) and not isinstance(r.get("memory_mb"), bool)
                    and math.isfinite(r["memory_mb"])]
        if obs_mems and max(obs_mems) > limit:
            risk_level = "observed_seed_exceeded_max_memory" if limit_kind == "max_memory" else "observed_seed_exceeded_vram_free"
            return risk_level, limit, max(obs_mems)
    pred_mb = forecast.get("predicted_memory_mb")
    if pred_mb is None or isinstance(pred_mb, bool) or not isinstance(pred_mb, (int, float)) or not math.isfinite(pred_mb):
        return "unknown", None, None
    pred_mb = float(pred_mb)
    risk_margin = float(protocol["scheduler"].get("memory_risk_margin", 1.10))
    if limit is not None and pred_mb > limit:
        risk_level = "hard_exceeds_max_memory" if limit_kind == "max_memory" else "hard_exceeds_vram_free"
        return risk_level, limit, pred_mb
    if limit is not None and pred_mb * risk_margin > limit:
        risk_level = "soft_near_max_memory" if limit_kind == "max_memory" else "soft_near_vram_free"
        return risk_level, limit, pred_mb
    return "safe", limit, pred_mb


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
    policy_version = protocol["scheduler"].get("policy_version", 1)
    protected = policy_version >= 2
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
            if protected and (method in {"successive_halving", "conformal_mf", "mf_no_conformal", "constrained_bo"} or method in ADAPTIVE_METHODS):
                forecast = cost_model.estimate(i)
                has_conformal_card = (method == "conformal_mf" and protocol.get("conformal_vram_card") is not None)
                if policy_version >= 3 and method not in {"mf_no_conformal", "constrained_bo"} and not has_conformal_card:
                    risk_level, limit_mb, meas_mb = _memory_risk(forecast, objective, protocol, rows)
                    if risk_level in {"hard_exceeds_max_memory", "hard_exceeds_vram_free", "observed_seed_exceeded_max_memory", "observed_seed_exceeded_vram_free"}:
                        reason = ("observed seed memory exceeds effective memory limit; OOM/constraint risk guard"
                                  if risk_level.startswith("observed_seed_")
                                  else "predicted memory exceeds effective memory limit; OOM/constraint risk guard")
                        ledger.emit("candidate_unresolved", index=i, reason=reason, risk_level=risk_level,
                                    limit_mb=limit_mb, measured_or_predicted_mb=meas_mb, forecast=forecast)
                        return
                plan = forecast["planning_seconds"]
                need = len(missing(i))
                has_complete_inc = any(
                    complete(rr, protocol["full_epochs"]) and (
                        objective.max_memory is None or float(rr.get("memory_mb") or 0.0) <= float(objective.max_memory)
                    )
                    for k_rows in state["results"].values() for rr in k_rows
                )
                if method == "conformal_mf" and has_conformal_card and has_complete_inc:
                    cap = max(0.0, (remaining() - cleanup) / need)
                else:
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
        partial_deferred = set()
        emitted_memory_deferred = {d["index"] for d in state.get("memory_risk_deferred", []) if isinstance(d, dict) and "index" in d}
        history = state.setdefault("acquisition_history", [])
        acq_mode = "ehvi" if method == "adaptive_ehvi" else "auto"

        def _is_competitive(chosen_idx, result, rows_now):
            sel = select(rows_now, protocol)
            best_idx = sel.get("selected_index")
            if best_idx is None:
                return True
            best_row = rows_now[best_idx]
            seed1_row = {
                "quality_mean": result["quality"],
                "quality_min": result["quality"],
                "time_mean": result["worker_wall_seconds"],
                "time_max": result["worker_wall_seconds"],
                "memory_max": result["memory_mb"],
            }
            if not objective.feasible(seed1_row):
                return False
            inc_u = best_row["utility"]
            inc_q = best_row["quality_mean"]
            inc_sd = best_row.get("quality_sd") or 0.0
            seed1_u = objective.value(seed1_row)
            rule = protocol.get("selection_rule", "weighted")
            if rule == "weighted":
                return (seed1_u >= inc_u - max(0.02, 0.05 * abs(inc_u))) or (result["quality"] >= inc_q - max(0.05, 2.0 * inc_sd))
            if objective.min_quality > 0 and result["quality"] < max(0.0, objective.min_quality - 0.05):
                return False
            return ((seed1_u >= inc_u - 0.03)
                    or (result["quality"] >= inc_q - max(0.05, 2.0 * inc_sd))
                    or (result["worker_wall_seconds"] < best_row["time_mean"] * 0.90)
                    or (result["memory_mb"] < best_row["memory_max"] * 0.90))

        while remaining() > cleanup:
            rows = measurements(configs, state["results"], seeds, protocol["full_epochs"], objective)
            has_complete_incumbent = any(row["complete"] for row in rows)
            eligible, deferred, memory_deferred = [], [], []
            forecasts = {i: cost_model.estimate(i) for i in range(len(configs))}
            for i in order:
                prior = state["results"][str(i)]
                if i in attempted or not missing(i) or any(r.get("status") in {"oom", "config_mismatch", "training_evidence_missing"} for r in prior):
                    continue
                if policy_version >= 3:
                    risk_level, limit_mb, meas_mb = _memory_risk(forecasts[i], objective, protocol, prior)
                    if risk_level in {"hard_exceeds_max_memory", "hard_exceeds_vram_free", "observed_seed_exceeded_max_memory", "observed_seed_exceeded_vram_free"}:
                        rec = {
                            "index": i, "config_id": config_id(configs[i]),
                            "predicted_memory_mb": meas_mb, "max_memory_mb": limit_mb,
                            "risk_level": risk_level, "memory_basis": forecasts[i].get("memory_basis"),
                            "upper_bound_guaranteed": False,
                        }
                        memory_deferred.append(rec)
                        if i not in emitted_memory_deferred:
                            emitted_memory_deferred.add(i)
                            state.setdefault("memory_risk_deferred", []).append(rec)
                            ledger.emit("candidate_memory_risk_deferred", **rec)
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
                has_done_seed = any(complete(r, protocol["full_epochs"]) for r in prior)
                need_count = 1 if (method == "adaptive_staged_seeds" and has_complete_incumbent and not has_done_seed and len(missing(i)) >= 2) else len(missing(i))
                if plan is not None and plan * need_count + cleanup > remaining():
                    deferred.append(i)
                    continue
                eligible.append({"index": i, "missing_seeds": missing(i)})
            if not eligible:
                stop_kwargs = {"reason": "no remaining all-seed action has affordable forecast", "deferred": deferred, "upper_bound_guaranteed": False}
                if policy_version >= 3:
                    stop_kwargs["deferred_memory_risk"] = [d["index"] for d in memory_deferred]
                ledger.emit("adaptive_stopped", **stop_kwargs)
                break
            if policy_version >= 3:
                safe_eligible = [c for c in eligible if _memory_risk(forecasts[c["index"]], objective, protocol, state["results"][str(c["index"])])[0] in {"safe", "unknown"}]
                candidates_for_acq = safe_eligible if safe_eligible else eligible
            else:
                candidates_for_acq = eligible

            if method == "adaptive_staged_seeds" and len(seeds) >= 2:
                if not has_complete_incumbent:
                    partial = [c for c in eligible if any(complete(r, protocol["full_epochs"]) for r in state["results"][str(c["index"])])]
                    if partial:
                        chosen = partial[0]["index"]
                        acquisition = {"ranking": [c["index"] for c in partial], "scores": {},
                                       "surrogate": {"reason": "finish partially confirmed configuration first; no proxy quality input"}}
                        reason = "partial_confirmation_priority"
                    else:
                        chosen = min((c["index"] for c in candidates_for_acq), key=lambda i: (small_first_key(protocol, i), order.index(i)))
                        acquisition = {"ranking": [chosen], "scores": {},
                                       "surrogate": {"reason": "cold start; metadata-small first and equal-share unknown worker cap"}}
                        reason = "bootstrap_full_confirmation"
                    entry = {"step": len(history) + 1, "index": chosen, "reason": reason,
                             "missing_seeds": missing(chosen), "remaining_seconds": remaining(),
                             "deferred_unaffordable": deferred, **acquisition}
                    if policy_version >= 3:
                        entry["deferred_memory_risk"] = [d["index"] for d in memory_deferred]
                    history.append(entry)
                    ledger.emit("adaptive_acquisition", **entry)
                    save()
                    attempted.add(chosen)
                    confirm(chosen)
                    continue

                active_partial = [c for c in eligible if c["index"] not in partial_deferred and any(complete(r, protocol["full_epochs"]) for r in state["results"][str(c["index"])])]
                unstarted_acq = [c for c in candidates_for_acq if not any(complete(r, protocol["full_epochs"]) for r in state["results"][str(c["index"])])]
                if active_partial:
                    chosen = active_partial[0]["index"]
                    acquisition = {"ranking": [c["index"] for c in active_partial], "scores": {},
                                   "surrogate": {"reason": "finish partially confirmed configuration first; no proxy quality input"}}
                    reason = "staged_seed_completion"
                    entry = {"step": len(history) + 1, "index": chosen, "reason": reason,
                             "missing_seeds": missing(chosen), "remaining_seconds": remaining(),
                             "deferred_unaffordable": deferred, **acquisition}
                    if policy_version >= 3:
                        entry["deferred_memory_risk"] = [d["index"] for d in memory_deferred]
                    history.append(entry)
                    ledger.emit("adaptive_acquisition", **entry)
                    save()
                    attempted.add(chosen)
                    confirm(chosen)
                    continue

                if unstarted_acq:
                    acquisition = rank_candidates(protocol, rows, forecasts, candidates=unstarted_acq, order=order,
                                                  use_cost=True, use_uncertainty=True, use_diversity=True,
                                                  acquisition_mode=acq_mode)
                    chosen = acquisition["ranking"][0]
                    reason = "adaptive_staged_seed_1"
                    entry = {"step": len(history) + 1, "index": chosen, "reason": reason,
                             "missing_seeds": missing(chosen), "remaining_seconds": remaining(),
                             "deferred_unaffordable": deferred, **acquisition}
                    if policy_version >= 3:
                        entry["deferred_memory_risk"] = [d["index"] for d in memory_deferred]
                    history.append(entry)
                    ledger.emit("adaptive_acquisition", **entry)
                    save()

                    fc_chosen = forecasts[chosen]
                    cap = None
                    if protected:
                        if fc_chosen["planning_seconds"] is None:
                            need = len(missing(chosen))
                            cap = max(0, (remaining() - cleanup) / need)
                            ledger.emit("candidate_cost_unavailable", index=chosen, reason="unknown full cost; equal-share bootstrap cap is heuristic, not a guarantee", worker_cap_seconds=cap)
                    seed_1 = missing(chosen)[0]
                    result = attempt(chosen, seed_1, True, cap)
                    if result is None or not complete(result, protocol["full_epochs"]):
                        attempted.add(chosen)
                        continue
                    updated_fc = cost_model.estimate(chosen)
                    limit_mb, limit_kind = _effective_memory_limit(objective, protocol)
                    obs_mem = result.get("memory_mb")
                    violated_mem = (limit_mb is not None and isinstance(obs_mem, (int, float))
                                    and not isinstance(obs_mem, bool) and math.isfinite(obs_mem) and float(obs_mem) > limit_mb)
                    violated_time = objective.max_time is not None and result["worker_wall_seconds"] > objective.max_time
                    if violated_mem or violated_time:
                        ledger.emit("adaptive_staged_stopped_infeasible", index=chosen, seed=seed_1,
                                    memory_mb=result["memory_mb"], worker_wall_seconds=result["worker_wall_seconds"],
                                    memory_limit_mb=limit_mb, memory_limit_kind=limit_kind)
                        attempted.add(chosen)
                        continue
                    need_rem = len(missing(chosen))
                    plan_rem = updated_fc["planning_seconds"]
                    if need_rem > 0 and plan_rem is not None and plan_rem * need_rem + cleanup > remaining():
                        ledger.emit("adaptive_staged_deferred_unaffordable", index=chosen, remaining_seeds=need_rem,
                                    planning_seconds=plan_rem, remaining_seconds=remaining())
                        attempted.add(chosen)
                        continue
                    if not _is_competitive(chosen, result, rows) and len(unstarted_acq) > 1:
                        partial_deferred.add(chosen)
                        ledger.emit("adaptive_staged_deferred_weak", index=chosen, seed1_quality=result["quality"],
                                    seed1_wall_seconds=result["worker_wall_seconds"], seed1_memory_mb=result["memory_mb"])
                    else:
                        attempted.add(chosen)
                        confirm(chosen)
                    continue

                drain_pool = [c for c in eligible if c["index"] in partial_deferred]
                if not drain_pool:
                    drain_pool = eligible
                def _seed1_utility(entry_c):
                    idx = entry_c["index"]
                    comp = [r for r in state["results"][str(idx)] if complete(r, protocol["full_epochs"])]
                    if not comp:
                        return (-math.inf, -idx)
                    u = objective.value({"quality_mean": comp[0]["quality"], "time_mean": comp[0]["worker_wall_seconds"], "memory_max": comp[0]["memory_mb"]})
                    return (u, -idx)
                drain_sorted = sorted(drain_pool, key=_seed1_utility, reverse=True)
                chosen = drain_sorted[0]["index"]
                acquisition = {"ranking": [c["index"] for c in drain_sorted], "scores": {},
                               "surrogate": {"reason": "drain deferred staged candidates with remaining budget"}}
                reason = "staged_seed_drain_completion"
                entry = {"step": len(history) + 1, "index": chosen, "reason": reason,
                         "missing_seeds": missing(chosen), "remaining_seconds": remaining(),
                         "deferred_unaffordable": deferred, **acquisition}
                if policy_version >= 3:
                    entry["deferred_memory_risk"] = [d["index"] for d in memory_deferred]
                history.append(entry)
                ledger.emit("adaptive_acquisition", **entry)
                save()
                attempted.add(chosen)
                confirm(chosen)
                continue

            # Preserve previous partial full-data investment before acquiring
            # another configuration. Failed/incomplete observations never fit Q.
            partial = [c for c in eligible if any(complete(r, protocol["full_epochs"]) for r in state["results"][str(c["index"])])]
            if partial:
                chosen = partial[0]["index"]
                acquisition = {"ranking": [c["index"] for c in partial], "scores": {},
                               "surrogate": {"reason": "finish partially confirmed configuration first; no proxy quality input"}}
                reason = "partial_confirmation_priority"
            elif not has_complete_incumbent:
                # Cold start needs one measured incumbent. Seed order breaks
                # equal-metadata ties; no future validation/ref/table is read.
                chosen = min((c["index"] for c in candidates_for_acq), key=lambda i: (small_first_key(protocol, i), order.index(i)))
                acquisition = {"ranking": [chosen], "scores": {},
                               "surrogate": {"reason": "cold start; metadata-small first and equal-share unknown worker cap"}}
                reason = "bootstrap_full_confirmation"
            else:
                acquisition = rank_candidates(protocol, rows, forecasts, candidates=candidates_for_acq, order=order,
                                              use_cost=method != "adaptive_no_cost",
                                              use_uncertainty=method != "adaptive_no_uncertainty",
                                              use_diversity=method != "adaptive_no_diversity",
                                              acquisition_mode=acq_mode)
                chosen = acquisition["ranking"][0]
                reason = "adaptive_optimistic_acquisition"
            entry = {"step": len(history) + 1, "index": chosen, "reason": reason,
                     "missing_seeds": missing(chosen), "remaining_seconds": remaining(),
                     "deferred_unaffordable": deferred, **acquisition}
            if policy_version >= 3:
                entry["deferred_memory_risk"] = [d["index"] for d in memory_deferred]
            history.append(entry)
            ledger.emit("adaptive_acquisition", **entry)
            save()
            attempted.add(chosen)
            confirm(chosen)
        # Avoid the generic one-pass scheduler; adaptive has already reacquired
        # after every complete configuration and stored its decision trace.
        order = []
    def _run_tpe_loop(excluded=None):
        # Deterministic Tree-Structured Parzen Estimator loop over candidate hyperparameters.
        from .selection import rule_scores
        excluded_set = set(excluded or ())
        def _tpe_feats(idx):
            c = configs[idx]
            d = descriptor_for(protocol, c)
            p_cnt = float(d.get("parameters") or 2_600_000.0)
            return [
                math.log(max(1.0, float(c.get("imgsz", 32)))),
                math.log(max(1.0, float(c.get("batch", 1)))),
                math.log(max(1e-8, float(c.get("lr0", 1e-3)))),
                math.log1p(p_cnt),
                float(c.get("precision") == "fp16"),
                float(c.get("freeze", 0) or 0),
                float(c.get("optimizer") == "SGD"),
                math.log1p(float(c.get("width", 16) or 16)),
            ]
        raw_vecs = [_tpe_feats(i) for i in range(len(configs))]
        n_dim = len(raw_vecs[0]) if raw_vecs else 0
        norm_vecs = []
        for idx in range(len(configs)):
            vec = []
            for d_i in range(n_dim):
                col = [raw_vecs[j][d_i] for j in range(len(configs))]
                lo, hi = min(col), max(col)
                vec.append((raw_vecs[idx][d_i] - lo) / (hi - lo) if hi > lo + 1e-12 else 0.0)
            norm_vecs.append(vec)

        n_startup = max(2, min(3, max(1, len(configs) // 4)))
        attempted = set()
        while remaining() > cleanup:
            rows = measurements(configs, state["results"], seeds, protocol["full_epochs"], objective)
            scores_map, _ = rule_scores(rows, protocol)
            obs_scores = {}
            for r in rows:
                idx = r["index"]
                if r["state"] == "confirmed_feasible":
                    obs_scores[idx] = float(scores_map.get(r["config_id"], r["utility"] or 0.0))
                elif r["complete"] and r.get("quality_mean") is not None:
                    obs_scores[idx] = -10.0 + float(r["quality_mean"])
                elif any(ar.get("status") == "oom" for ar in r["runs"]):
                    obs_scores[idx] = -100.0

            candidates_left = [i for i in order if i not in attempted and i not in excluded_set and missing(i)
                               and not any(ar.get("status") in {"oom", "config_mismatch", "training_evidence_missing"} for ar in state["results"][str(i)])]
            affordable = []
            for i in candidates_left:
                plan = cost_model.estimate(i)["planning_seconds"]
                if plan is None or plan * len(missing(i)) + cleanup <= remaining():
                    affordable.append(i)
            if not affordable:
                break

            if len(obs_scores) < n_startup:
                chosen = affordable[0]
                tpe_reason = "tpe_startup_random"
            else:
                ranked_obs = sorted(obs_scores, key=lambda i: (obs_scores[i], -i), reverse=True)
                n_good = max(1, min(len(ranked_obs) - 1, int(math.ceil(0.25 * len(ranked_obs)))))
                good_idx = ranked_obs[:n_good]
                bad_idx = ranked_obs[n_good:]

                def _log_kde(x_vec, group_indices):
                    if not group_indices:
                        return 0.0
                    log_p = 0.0
                    m_g = float(len(group_indices))
                    for d_i in range(n_dim):
                        vals = [norm_vecs[g][d_i] for g in group_indices]
                        std_d = max(0.18, statistics.stdev(vals) if len(vals) > 1 else 0.35)
                        mix = sum(math.exp(-0.5 * ((x_vec[d_i] - v) / std_d) ** 2) / (std_d * 2.5066282746310002) for v in vals) / m_g
                        log_p += math.log(max(1e-12, mix))
                    return log_p

                best_tpe = [(_log_kde(norm_vecs[i], good_idx) - _log_kde(norm_vecs[i], bad_idx), -order.index(i), i) for i in affordable]
                best_tpe.sort(reverse=True)
                chosen = best_tpe[0][2]
                tpe_reason = "tpe_ei_kde_acquisition"

            ledger.emit("tpe_bo_step", index=chosen, reason=tpe_reason, n_observed=len(obs_scores), remaining_seconds=remaining())
            attempted.add(chosen)
            confirm(chosen)

    if method == "tpe_bo":
        _run_tpe_loop()
        order = []

    if method == "constrained_bo":
        # Gelbart et al. (2014) Constrained Expected Improvement (cEI):
        # Independent quality GP (GP-LOOCV) * latent log-memory GP constraint feasibility probability.
        from scipy.special import ndtr
        from .acquisition import _kernel_matrix, configuration_distances, rank_candidates

        distances, _ = configuration_distances(protocol)
        limit_mb, _ = _effective_memory_limit(objective, protocol)
        attempted_cbo = set()
        while remaining() > cleanup:
            rows = measurements(configs, state["results"], seeds, protocol["full_epochs"], objective)
            forecasts = {i: cost_model.estimate(i) for i in range(len(configs))}
            affordable = []
            for i in order:
                if i in attempted_cbo or not missing(i):
                    continue
                prior = state["results"][str(i)]
                if any(ar.get("status") in {"oom", "config_mismatch", "training_evidence_missing"} for ar in prior):
                    continue
                plan = forecasts[i].get("planning_seconds")
                if plan is None or plan * len(missing(i)) + cleanup <= remaining():
                    affordable.append(i)
            if not affordable:
                break

            mem_indices, mem_logs = [], []
            for i_m, r_list in state["results"].items():
                idx_m = int(i_m)
                vals = [
                    float(ar["memory_mb"]) for ar in r_list
                    if isinstance(ar.get("memory_mb"), (int, float))
                    and not isinstance(ar.get("memory_mb"), bool)
                    and math.isfinite(ar["memory_mb"]) and float(ar["memory_mb"]) > 0
                ]
                if vals:
                    mem_indices.append(idx_m)
                    mem_logs.append(math.log(max(vals)))
                elif limit_mb is not None and any(ar.get("status") == "oom" for ar in r_list):
                    mem_indices.append(idx_m)
                    mem_logs.append(math.log(max(1.0, 1.35 * limit_mb)))

            p_mem_gauss = {i: 1.0 for i in range(len(configs))}
            if limit_mb is not None and len(mem_indices) >= 2:
                import numpy as _np
                y_lm = _np.asarray(mem_logs, dtype=float)
                m_lm = float(_np.mean(y_lm))
                k_m = _kernel_matrix(distances, "matern52", 0.90)
                gram_m = k_m[_np.ix_(mem_indices, mem_indices)] + 0.05 * _np.eye(len(mem_indices))
                try:
                    solved_m = _np.linalg.solve(gram_m, _np.column_stack((y_lm - m_lm, k_m[:, mem_indices].T)))
                    alpha_m, proj_m = solved_m[:, 0], solved_m[:, 1:].T
                    s2_m = max(0.02, float(_np.var(y_lm)))
                    log_lim = math.log(max(1.0, float(limit_mb)))
                    for j in range(len(configs)):
                        mu_j = m_lm + float(k_m[j, mem_indices] @ alpha_m)
                        lev_j = float(_np.clip(1.0 - float(k_m[j, mem_indices] @ proj_m[j]), 0.0, 1.0))
                        sig_j = max(0.08, math.sqrt(max(1e-6, s2_m * lev_j)))
                        p_mem_gauss[j] = float(ndtr((log_lim - mu_j) / sig_j))
                except _np.linalg.LinAlgError:
                    pass

            n_complete = sum(1 for r in rows if r["complete"])
            if len(attempted_cbo) < 2:
                chosen = affordable[0]
                cbo_reason = "constrained_bo_cold_start"
            elif n_complete == 0:
                chosen = max(affordable, key=lambda idx: (p_mem_gauss[idx], -order.index(idx)))
                cbo_reason = "constrained_bo_memory_recovery"
            else:
                acq = rank_candidates(
                    protocol, rows, forecasts,
                    candidates=[{"index": i, "missing_seeds": missing(i)} for i in affordable],
                    order=order, use_cost=True, use_uncertainty=True, use_diversity=True,
                    acquisition_mode="rule", feasibility_weights=p_mem_gauss,
                )
                chosen = acq["ranking"][0]
                cbo_reason = "constrained_bo_cei"
            ledger.emit("constrained_bo_step", index=chosen, reason=cbo_reason, p_feasible=p_mem_gauss[chosen], remaining_seconds=remaining())
            attempted_cbo.add(chosen)
            confirm(chosen)
        order = []

    if method in {"conformal_mf", "mf_no_conformal"}:
        from .acquisition import configuration_distances, rank_candidates
        from .conformal_vram import calibrate_online_2point, conformal_feasibility_probability, predict_conformal_vram
        from .predictors import observed_cross_memory

        use_conformal = method == "conformal_mf"
        card = protocol.get("conformal_vram_card") if use_conformal else None
        limit_mb, limit_kind = _effective_memory_limit(objective, protocol)
        conformal_alpha = float(protocol.get("scheduler", {}).get("conformal_alpha", 0.05))
        task_name = protocol.get("task", "detection")
        pruned_set = {rec["index"] for rec in state.get("conformal_pruned", []) if isinstance(rec, dict) and "index" in rec}
        distances, _ = configuration_distances(protocol)

        def _record_pruned(idx, stage, pred_point, pred_upper, basis_str):
            if idx in pruned_set:
                return
            pruned_set.add(idx)
            rec = {
                "index": idx,
                "config_id": config_id(configs[idx]),
                "stage": stage,
                "predicted_point_mb": pred_point,
                "upper_bound_mb": pred_upper,
                "limit_mb": limit_mb,
                "limit_kind": limit_kind,
                "basis": basis_str,
            }
            state.setdefault("conformal_pruned", []).append(rec)
            ledger.emit("conformal_vram_pruned", **rec)

        p_feas_map = {idx: 1.0 for idx in range(len(configs))}

        def _refresh_conformal_bounds(stage_tag, current_online_cal=None):
            if not (use_conformal and card is not None and limit_mb is not None):
                return
            for idx in range(len(configs)):
                desc_i = descriptor_for(protocol, configs[idx])
                pred = predict_conformal_vram(
                    card, configs[idx], desc_i,
                    alpha=conformal_alpha, online_cal=current_online_cal, task=task_name, protocol=protocol,
                )
                p_feas_map[idx] = conformal_feasibility_probability(
                    card, configs[idx], limit_mb, descriptor=desc_i, online_cal=current_online_cal, task=task_name, protocol=protocol,
                )
                if pred["upper_bound_mb"] > limit_mb and idx not in pruned_set:
                    if not any(r["index"] == idx for r in state["probes"]) and not any(complete(r, protocol["full_epochs"]) for r in state["results"][str(idx)]):
                        _record_pruned(idx, stage_tag, pred["point_mb"], pred["upper_bound_mb"], pred["basis"])

        # Stage 0: Unified Conformal VRAM Pre-Filter & Predictive Feasibility Assessment
        _refresh_conformal_bounds("stage0_offline_conformal", None)

        # Collect existing VRAM anchors for resume compatibility
        online_anchors = []
        for pr in state["probes"]:
            if pr.get("status") == "ok" and isinstance(pr.get("memory_mb"), (int, float)) and math.isfinite(pr["memory_mb"]):
                online_anchors.append({
                    "config": configs[pr["index"]],
                    "descriptor": descriptor_for(protocol, configs[pr["index"]]),
                    "memory_mb": float(pr["memory_mb"]),
                })
        online_cal = calibrate_online_2point(card, online_anchors, task=task_name, protocol=protocol) if (use_conformal and card is not None and online_anchors) else None
        if online_cal is not None:
            _refresh_conformal_bounds("stage0_resume_calibrated_conformal", online_cal)

        # Stage 1: Seed-Stratified Facility-Location (K-Medoid) Multi-Fidelity Screening
        safe_candidates = [i for i in range(len(configs)) if i not in pruned_set]
        if protocol["probe_epochs"] >= protocol["full_epochs"] or len(safe_candidates) <= 2:
            target_probes = 0
        else:
            target_probes = min(
                4 if (task_name != "detection" and len(safe_candidates) >= 12) else 3 if len(safe_candidates) >= 9 else 2,
                max(1, (len(safe_candidates) + 1) // 2),
            )

        if use_conformal and card is not None and safe_candidates:
            rng_probe = random.Random(protocol["search_seed"])
            sorted_safe = sorted(safe_candidates, key=lambda i: (small_first_key(protocol, i), order.index(i)))
            fast_tier = sorted_safe[:max(2, (len(sorted_safe) + 2) // 3)]
            p0 = rng_probe.choice(fast_tier)
            probe_order = [p0]
            rem_pool = [i for i in sorted_safe if i != p0]
            while rem_pool:
                next_p = min(
                    rem_pool,
                    key=lambda j: (
                        sum(min(float(distances[u, c]) for c in probe_order + [j]) for u in safe_candidates)
                        - 0.08 * float(all(configs[j].get("model") != configs[c].get("model") for c in probe_order))
                        + 1e-3 * rng_probe.random()
                    ),
                )
                probe_order.append(next_p)
                rem_pool.remove(next_p)
        else:
            probe_order = [i for i in order if i not in pruned_set]

        mf_probe_fraction = min(max(float(protocol["scheduler"].get("probe_fraction", 0.1)), 0.20), 0.28)
        probe_allowance = protocol["budget_seconds"] * mf_probe_fraction
        existing_probes = {r["index"]: r for r in state["probes"]}
        stage1_probes_done = sum(1 for r in state["probes"] if r.get("status") == "ok")

        for idx in probe_order:
            if idx in existing_probes or idx in pruned_set:
                continue
            if stage1_probes_done >= target_probes:
                break
            if use_conformal and limit_mb is not None and online_anchors and card is None and len(online_anchors) >= 2:
                merged_obs = {}
                for pr in state["probes"]:
                    if pr.get("status") == "ok" and isinstance(pr.get("memory_mb"), (int, float)):
                        merged_obs.setdefault(pr["index"], []).append(pr)
                for k_idx, k_rows in state["results"].items():
                    ok_rows = [r for r in k_rows if r.get("status") == "ok" and isinstance(r.get("memory_mb"), (int, float))]
                    if ok_rows:
                        merged_obs.setdefault(int(k_idx), []).extend(ok_rows)
                mem_est = observed_cross_memory(protocol, merged_obs, idx)
                if mem_est and mem_est["point_mb"] * 1.12 > limit_mb:
                    _record_pruned(idx, "stage1_online_physical_safety", mem_est["point_mb"], mem_est["point_mb"] * 1.12, mem_est["basis"])
                    p_feas_map[idx] = 0.05
                    continue

            if probe_spent >= probe_allowance:
                break
            if stage1_probes_done >= 1 and remaining() <= cleanup + 0.65 * protocol["budget_seconds"]:
                break
            cap = min(max(cleanup, probe_allowance - probe_spent + cleanup), remaining() - cleanup)
            if cap < cleanup:
                break
            res = attempt(idx, seeds[0], False, cap, protocol["probe_epochs"])
            if res is None:
                break
            if res.get("status") == "ok" and isinstance(res.get("memory_mb"), (int, float)) and math.isfinite(res["memory_mb"]):
                stage1_probes_done += 1
                if use_conformal and card is not None:
                    online_anchors.append({
                        "config": configs[idx],
                        "descriptor": descriptor_for(protocol, configs[idx]),
                        "memory_mb": float(res["memory_mb"]),
                    })
                    online_cal = calibrate_online_2point(card, online_anchors, task=task_name, protocol=protocol)
                    _refresh_conformal_bounds("stage1_online_calibrated_conformal", online_cal)

        # Collect screened probe outcomes & continuous multi-fidelity probe prior
        probe_by_idx = {}
        for r in state["probes"]:
            probe_by_idx[r["index"]] = r
        valid_probes = []
        failed_probes = set()
        for idx, r in probe_by_idx.items():
            if idx in pruned_set:
                continue
            mem = r.get("memory_mb")
            q = r.get("quality")
            if r.get("status") == "ok" and isinstance(q, (int, float)) and math.isfinite(q) and isinstance(mem, (int, float)) and math.isfinite(mem):
                if limit_mb is not None and use_conformal and float(mem) > limit_mb:
                    failed_probes.add(idx)
                else:
                    valid_probes.append((idx, r))
            elif r.get("status") in {"oom", "config_mismatch", "training_evidence_missing"}:
                failed_probes.add(idx)

        probe_priors = {idx: float(r["quality"]) for idx, r in valid_probes}
        if use_conformal:
            ledger.emit(
                "conformal_mf_stage_summary",
                mode="cc_mfbo_unified",
                pruned=sorted(pruned_set),
                probed=sorted(probe_priors),
                target_probes=target_probes,
                limit_mb=limit_mb,
                limit_kind=limit_kind,
                online_calibration=online_cal,
            )
        save()

        # Stage 2: Unified Multi-Fidelity Residual GP-LOOCV + Conformal cEUI Confirmation Loop
        safe_order = [i for i in probe_order if i not in pruned_set and i not in failed_probes]
        attempted_full = set()
        while remaining() > cleanup:
            rows = measurements(configs, state["results"], seeds, protocol["full_epochs"], objective)
            obs_full_w = [
                float(rr["worker_wall_seconds"])
                for k_rows in state["results"].values()
                for rr in k_rows
                if complete(rr, protocol["full_epochs"]) and rr.get("worker_wall_seconds")
            ]
            obs_probe_w = {
                r["index"]: float(r["worker_wall_seconds"])
                for r in state["probes"]
                if r.get("status") == "ok" and r.get("worker_wall_seconds")
            }
            med_probe_w = statistics.median(obs_probe_w.values()) if obs_probe_w else None
            med_full_w = statistics.median(obs_full_w) if obs_full_w else None
            n_complete = sum(1 for r in rows if r["complete"])

            forecasts = {}
            eligible_full = []
            for idx in safe_order:
                if idx in attempted_full or idx in pruned_set or not missing(idx):
                    continue
                prior = state["results"][str(idx)]
                if any(ar.get("status") in {"oom", "config_mismatch", "training_evidence_missing"} for ar in prior):
                    continue
                fc = cost_model.estimate(idx)
                if any(complete(ar, protocol["full_epochs"]) for ar in prior):
                    mf_fc = fc
                else:
                    raw_pt = fc.get("point_seconds")
                    if med_full_w is not None:
                        ratio = (float(raw_pt) / med_full_w) if (raw_pt and med_full_w > 0) else 1.0
                        pt_s = med_full_w * min(1.55, max(0.70, ratio ** 0.30))
                        if idx in obs_probe_w and med_probe_w and med_probe_w > 0:
                            pt_s *= (obs_probe_w[idx] / med_probe_w) ** 0.5
                    elif obs_probe_w:
                        p_w = obs_probe_w.get(idx, med_probe_w)
                        pt_s = p_w * max(1.5, (protocol["full_epochs"] / max(1, protocol["probe_epochs"])) * 0.65)
                    else:
                        pt_s = raw_pt
                    if pt_s is not None:
                        mf_fc = dict(fc, point_seconds=float(pt_s), planning_seconds=float(1.15 * pt_s))
                    else:
                        mf_fc = fc
                forecasts[idx] = mf_fc
                if use_conformal and limit_mb is not None:
                    if card is not None:
                        desc_c = descriptor_for(protocol, configs[idx])
                        pred_c = predict_conformal_vram(card, configs[idx], desc_c, alpha=conformal_alpha, online_cal=online_cal, task=task_name, protocol=protocol)
                        p_feas_map[idx] = conformal_feasibility_probability(card, configs[idx], limit_mb, descriptor=desc_c, online_cal=online_cal, task=task_name, protocol=protocol)
                        if pred_c["upper_bound_mb"] > limit_mb:
                            _record_pruned(idx, "stage2_online_conformal", pred_c["point_mb"], pred_c["upper_bound_mb"], pred_c["basis"])
                            continue
                    else:
                        risk_lvl, _, _ = _memory_risk(mf_fc, objective, protocol, prior)
                        if risk_lvl in {"hard_exceeds_max_memory", "hard_exceeds_vram_free", "observed_seed_exceeded_max_memory", "observed_seed_exceeded_vram_free"}:
                            continue
                if use_conformal and card is not None:
                    if n_complete >= 1 and obs_full_w:
                        if 0.75 * min(obs_full_w) * len(missing(idx)) + cleanup > remaining():
                            continue
                else:
                    plan = mf_fc.get("planning_seconds")
                    if plan is not None and plan * len(missing(idx)) + cleanup > remaining():
                        continue
                eligible_full.append({"index": idx, "missing_seeds": missing(idx)})
            if not eligible_full:
                break
            if n_complete == 0 and not probe_priors:
                chosen = eligible_full[0]["index"]
                reason = "stage2_cold_bootstrap"
                acq_payload = {"ranking": [chosen], "scores": {}}
            else:
                acq_payload = rank_candidates(
                    protocol, rows, forecasts, candidates=eligible_full, order=safe_order,
                    use_cost=True, use_uncertainty=True, use_diversity=True, acquisition_mode="rule",
                    probe_priors=probe_priors if probe_priors else None,
                    feasibility_weights=p_feas_map if use_conformal else None,
                )
                chosen = acq_payload["ranking"][0]
                reason = "stage2_mf_residual_gp_ceui"
            ledger.emit("conformal_mf_acquisition", index=chosen, reason=reason, p_feas=p_feas_map.get(chosen, 1.0), remaining_seconds=remaining(), ranking=acq_payload.get("ranking"))
            attempted_full.add(chosen)
            confirm(chosen)
            if use_conformal and card is not None:
                ok_full = [r for r in state["results"][str(chosen)] if complete(r, protocol["full_epochs"])]
                if ok_full:
                    online_anchors.append({
                        "config": configs[chosen],
                        "descriptor": descriptor_for(protocol, configs[chosen]),
                        "memory_mb": float(max(r["memory_mb"] for r in ok_full)),
                    })
                    online_cal = calibrate_online_2point(card, online_anchors, task=task_name, protocol=protocol)
                    _refresh_conformal_bounds("stage2_online_conformal", online_cal)
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
