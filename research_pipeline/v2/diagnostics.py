"""Reference-only diagnostic analysis. Never called by online acquisition."""

import json
import math
from pathlib import Path
import statistics

from yolo_tuner.research import Objective
from .ledger import read_events
from .schema import complete, config_id, measurements
from .selection import select


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result, start = [0.0] * len(values), 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for i in order[start:end]:
            result[i] = (start + end - 1) / 2
        start = end
    return result


def spearman(a, b):
    if len(a) != len(b) or len(a) < 2:
        return None
    a, b = ranks(a), ranks(b)
    am, bm = sum(a) / len(a), sum(b) / len(b)
    denom = math.sqrt(sum((x - am) ** 2 for x in a) * sum((x - bm) ** 2 for x in b))
    return sum((x - am) * (y - bm) for x, y in zip(a, b)) / denom if denom else None


def first_full_seconds(folder, protocol, rows):
    good = {r["attempt_id"]: r for row in rows for r in row["runs"] if complete(r, protocol["full_epochs"]) and r.get("attempt_id")}
    seen = {}
    for event in read_events(folder / "events.jsonl"):
        if event["kind"] == "trial_completed" and event["payload"]["attempt_id"] in good:
            r = good[event["payload"]["attempt_id"]]
            seen.setdefault(r["index"], set()).add(r["seed"])
            if set(protocol["training_seeds"]) <= seen[r["index"]]:
                return event["active_seconds"]
    return None


def proxy_reference_diagnostics(state, protocol, reference):
    epochs = sorted({r.get("epochs_completed", 0) for r in state["probes"] if r.get("status") == "ok"})
    output = []
    for epoch in epochs:
        proxies = {r["index"]: r for r in state["probes"] if r.get("status") == "ok" and r.get("epochs_completed") == epoch and r.get("quality") is not None}
        paired = []
        for i, proxy in proxies.items():
            full = next((r for r in reversed(reference[i]["runs"]) if r["seed"] == proxy["seed"] and complete(r, protocol["full_epochs"])), None)
            if full:
                paired.append((proxy["quality"], full["quality"]))
        output.append({"subset_epochs": epoch, "paired_configs": len(paired),
                       "spearman_same_seed": spearman([q for q, _ in paired], [q for _, q in paired]),
                       "mae_same_seed": sum(abs(q - a) for q, a in paired) / len(paired) if paired else None,
                       "basis": "independent reference full-data/horizon, same seed; not online selection input"})
    return output


def _load_reference_rows(reference):
    if reference is None:
        return None
    if isinstance(reference, (list, tuple)):
        return list(reference)
    ref_dir = Path(reference).resolve()
    state_path = ref_dir / "state.json"
    proto_path = ref_dir / "protocol.json"
    if state_path.is_file() and proto_path.is_file():
        ref_prot = json.loads(proto_path.read_text(encoding="utf-8"))
        ref_state = json.loads(state_path.read_text(encoding="utf-8"))
        return measurements(
            ref_prot["configs"],
            ref_state.get("results", {}),
            ref_prot["training_seeds"],
            ref_prot["full_epochs"],
            Objective(**ref_prot["objective"]),
        )
    report_path = ref_dir / "report.json"
    if report_path.is_file():
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        candidates = rep.get("candidates")
        if isinstance(candidates, list):
            return candidates
    return None


def _target_95_from_bounds(u_opt, u_min):
    if u_opt > 0.0:
        return 0.95 * u_opt
    if u_min < u_opt:
        return u_min + 0.95 * (u_opt - u_min)
    return u_opt


def reference_utility_targets(reference):
    """Return (u_opt, target_95) from an exhaustive reference run/rows, or (None, None)."""
    ref_rows = _load_reference_rows(reference)
    if not (ref_rows and all(r.get("complete") for r in ref_rows)):
        return None, None
    ref_feasible = [r for r in ref_rows if r.get("utility") is not None]
    if not ref_feasible:
        return None, None
    u_opt = float(max(ref_feasible, key=lambda r: (float(r["utility"]), r["config_id"]))["utility"])
    u_min = min(float(r["utility"]) for r in ref_feasible)
    return u_opt, _target_95_from_bounds(u_opt, u_min)


def anytime_trajectory(run_dir, protocol=None, reference=None):
    """Confirmed full-data, full-horizon, all-required-seeds anytime convergence trajectory.

    Attributes ``trial_completed`` events strictly by ``attempt_id`` against
    ``state["results"]``. Probe trials and partial-seed completions are never
    included. Returns AUARC (normalized by effective horizon) and Time-to-Target
    (95% of reference optimal utility) when an exhaustive reference is supplied.
    """
    folder = Path(run_dir).resolve()
    if protocol is None:
        protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    horizon = max(float(protocol["budget_seconds"]), 1e-9)
    objective = Objective(**protocol["objective"])
    required_seeds = list(protocol["training_seeds"])
    required_seeds_set = set(required_seeds)
    full_epochs = int(protocol["full_epochs"])

    ref_rows = _load_reference_rows(reference)
    ref_ready = False
    ref_map = {}
    ref_best_q = None
    u_opt = None
    target_95 = None
    r0 = None
    if ref_rows and all(r.get("complete") for r in ref_rows):
        ref_feasible = [r for r in ref_rows if r.get("utility") is not None]
        if ref_feasible:
            ref_ready = True
            ref_map = {r["config_id"]: r for r in ref_rows}
            ref_best = max(ref_feasible, key=lambda r: (float(r["utility"]), r["config_id"]))
            ref_best_q = max(ref_feasible, key=lambda r: (float(r["quality_mean"]), r["config_id"]))
            u_opt = float(ref_best["utility"])
            u_min = min(float(r["utility"]) for r in ref_feasible)
            target_95 = _target_95_from_bounds(u_opt, u_min)
            r0 = (u_opt - u_min) if (len(ref_feasible) > 1 and u_opt > u_min) else max(abs(u_opt), 1.0)

    trajectory = []
    confirmed_total_count = 0
    time_to_target_95 = None
    state_path = folder / "state.json"
    events_path = folder / "events.jsonl"
    if state_path.is_file() and events_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        results = state.get("results", {})
        valid_by_aid = {}
        for idx, _cfg in enumerate(protocol["configs"]):
            for r in results.get(str(idx), []):
                aid = r.get("attempt_id")
                seed = r.get("seed")
                if aid and seed in required_seeds_set and complete(r, full_epochs):
                    valid_by_aid[aid] = (idx, seed, r)

        seen = {}
        confirmed_indices = set()
        confirmed_feasible_rows = []
        best_u = None
        best_q = None
        for event in read_events(events_path):
            if event.get("kind") != "trial_completed":
                continue
            aid = (event.get("payload") or {}).get("attempt_id")
            if aid not in valid_by_aid:
                continue
            idx, seed, r = valid_by_aid[aid]
            seen.setdefault(idx, {})[seed] = r
            if idx in confirmed_indices or set(seen[idx]) != required_seeds_set:
                continue
            confirmed_indices.add(idx)
            confirmed_total_count += 1

            vals = [seen[idx][s] for s in required_seeds]
            q_vals = [float(v["quality"]) for v in vals]
            t_vals = [float(v["worker_wall_seconds"]) for v in vals]
            m_vals = [float(v["memory_mb"]) for v in vals]
            cfg = protocol["configs"][idx]
            cid = config_id(cfg)
            row = {
                "index": idx,
                "config_id": cid,
                "config": cfg,
                "complete": True,
                "completed_seeds": sorted(required_seeds),
                "required_seeds": required_seeds,
                "quality_mean": statistics.mean(q_vals),
                "quality_min": min(q_vals),
                "quality_sd": statistics.stdev(q_vals) if len(q_vals) > 1 else 0.0,
                "time_mean": statistics.mean(t_vals),
                "time_max": max(t_vals),
                "memory_max": max(m_vals),
                "runs": vals,
            }
            if not objective.feasible(row):
                continue
            u_val = float(objective.value(row))
            q_val = float(row["quality_mean"])
            row["state"] = "confirmed_feasible"
            row["utility"] = u_val
            confirmed_feasible_rows.append(row)
            best_u = u_val if best_u is None else max(best_u, u_val)
            best_q = q_val if best_q is None else max(best_q, q_val)

            if "selection_rule" in protocol:
                inc_cid = select(confirmed_feasible_rows, protocol)["selected_config_id"]
            else:
                inc_cid = max(confirmed_feasible_rows, key=lambda x: (x["utility"], x["config_id"]))["config_id"]

            t_sec = max(0.0, float(event["active_seconds"]))
            regret_t = None
            q_regret_t = None
            if ref_ready:
                oracle = ref_map.get(inc_cid)
                if oracle is not None and oracle.get("utility") is not None:
                    ref_u_t = float(oracle["utility"])
                    regret_t = max(0.0, u_opt - ref_u_t)
                    q_regret_t = float(ref_best_q["quality_mean"]) - float(oracle["quality_mean"])
                    if time_to_target_95 is None and ref_u_t >= target_95:
                        time_to_target_95 = t_sec

            trajectory.append({
                "active_seconds": t_sec,
                "config_index": idx,
                "config_id": cid,
                "selected_config_id": inc_cid,
                "config_utility": u_val,
                "config_quality": q_val,
                "best_utility": best_u,
                "best_quality": best_q,
                "regret": regret_t,
                "quality_regret": q_regret_t,
            })

    effective_horizon = max(horizon, trajectory[-1]["active_seconds"]) if trajectory else horizon
    raw_area = None
    auarc = None
    if ref_ready and trajectory and all(pt["regret"] is not None for pt in trajectory):
        raw_area = r0 * trajectory[0]["active_seconds"]
        for j in range(len(trajectory) - 1):
            dt = max(0.0, trajectory[j + 1]["active_seconds"] - trajectory[j]["active_seconds"])
            raw_area += trajectory[j]["regret"] * dt
        raw_area += trajectory[-1]["regret"] * max(0.0, effective_horizon - trajectory[-1]["active_seconds"])
        auarc = raw_area / effective_horizon

    return {
        "trajectory": trajectory,
        "auarc": auarc,
        "auarc_raw_seconds": raw_area,
        "time_to_target_95": time_to_target_95,
        "target_utility_95": target_95,
        "reference_optimal_utility": u_opt,
        "horizon_seconds": effective_horizon,
        "confirmed_feasible_count": len(trajectory),
        "confirmed_total_count": confirmed_total_count,
        "basis": "confirmed full-data/full-horizon/all-required-seeds trial_completed events only; zero proxy/partial-seed leakage",
    }
