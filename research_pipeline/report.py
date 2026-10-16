"""Selection evidence, fixed-objective sensitivity and reproducible reports."""

from dataclasses import replace
import json
from pathlib import Path
import statistics

from yolo_tuner.research import Objective, aggregate, nondominated, score_decisions, reference_cost
from yolo_tuner.pipeline import write_json

SCENARIOS = {"balanced": [0.6, 0.25, 0.15], "accuracy_first": [0.8, 0.1, 0.1],
             "time_first": [0.4, 0.5, 0.1], "memory_first": [0.4, 0.1, 0.5]}


def sensitivity(reference, objective, scenarios):
    feasible = [r for r in reference if objective.feasible(r)]
    output = []
    for name, weights in scenarios.items():
        modified = replace(objective, name="balanced", weights=tuple(weights))
        chosen = max(feasible, key=modified.value) if feasible else None
        output.append({"scenario": name, "weights": weights,
                       "selected": chosen["config_id"] if chosen else None,
                       "utility": modified.value(chosen) if chosen else None,
                       "basis": "same measured outcomes; sensitivity only, not a new statistical superiority test"})
    return output


def summarize(protocol, decisions, results, elapsed, prior=None):
    objective = Objective(**protocol["objective"])
    reference = [aggregate(f"c{i:02d}", results[f"c{i:02d}"], protocol["training_seeds"])
                 for i in range(len(protocol["configs"]))]
    complete = all(r["complete"] for r in reference)
    feasible = [r for r in reference if objective.feasible(r)]
    best = max(feasible, key=objective.value) if feasible else None
    clean = protocol["measurement_environment"] == "uncontended_preflight"
    verified = "empirical_optimal_in_declared_space" if complete and best else "fully_trained_best_observed" if best else "no_verified_feasible_selection"
    if not clean:
        verified = "contended_smoke_not_certified"
    online = protocol["mode"] == "online_verification"
    _, scores, summaries = score_decisions(decisions, reference, objective) if not online else (None, [], [])
    if not complete:
        for row in scores:
            for key in ("objective_regret", "quality_regret", "recovered_hypervolume_ratio"):
                row[key] = None
        for row in summaries:
            for key in ("objective_regret", "quality_regret", "recovered_hypervolume_ratio"):
                row[key + "_mean"] = row[key + "_sd"] = None
    candidate_rows = []
    for row in reference:
        failures = [r["status"] for r in row["runs"] if not r.get("full_completed")]
        candidate_rows.append({"config_id": row["config_id"], "config": protocol["configs"][int(row["config_id"][1:])],
                               "state": "confirmed_feasible" if objective.feasible(row) else "confirmed_constraint_failure" if row["complete"] else "unresolved",
                               "observed_failure_statuses": failures, "completed_seeds": [r["training_seed"] for r in row["runs"] if r.get("full_completed")],
                               "required_seeds": protocol["training_seeds"], "utility": objective.value(row) if objective.feasible(row) else None,
                               "measurement": {k: v for k, v in row.items() if k != "runs"}})
    ranked = sorted(feasible, key=objective.value, reverse=True)
    selection_evidence = None
    if best:
        probes = [r for d in decisions for r in d["history"] if r["config_id"] == best["config_id"] and r["complete"] and not r["full_data"]]
        selection_evidence = {"config_id": best["config_id"], "config": protocol["configs"][int(best["config_id"][1:])],
                              "selection_origin": "full_online_confirmations" if online else "offline_full_reference_not_free_online_search",
                              "utility": objective.value(best), "runner_up": ranked[1]["config_id"] if len(ranked) > 1 else None,
                              "utility_margin": objective.value(best) - objective.value(ranked[1]) if len(ranked) > 1 else None,
                              "validation_map50_95_mean": best["quality_mean"], "validation_map50_95_sd": best["quality_sd"],
                              "training_worker_seconds_mean": best["time_mean"], "measured_memory_mb": best["memory_max"],
                              "memory_kind": "CUDA peak reserved allocator" if protocol["hardware"]["device"] not in {"cpu", "mps"} else "process RSS; NOT VRAM",
                              "predicted_full_seconds": statistics.mean(r["estimated_time"] for r in probes) if probes else None,
                              "predicted_memory_mb": max((r["memory"] for r in probes), default=None),
                              "prediction_basis": "subset profile/extrapolation, uncalibrated; no confidence interval" if probes else "not available",
                              "test_map50_95": None, "test_status": "not_evaluated_for_selection",
                              "full_training_verification": "passed_observed_required_seeds" if clean else "smoke_only_contended",
                              "oom_in_chosen_full_runs": False,
                              "optimality_verification": verified, "seed_variation_not_confidence_interval": True}
    cost = sum(reference_cost(r) for rows in results.values() for r in rows)
    return {"schema_version": 3, "adapter": "yolo_research", "mode": protocol["mode"], "protocol": protocol,
            "decisions": decisions, "reference": reference, "reference_complete": complete,
            "reference_status": "complete_finite_empirical_reference" if complete else "partial_reference_not_certified",
            "verification_status": verified, "empirical_best": best["config_id"] if best and clean else None,
            "observed_best": best["config_id"] if best else None, "candidates": candidate_rows,
            "pareto": [r["config_id"] for r in nondominated(feasible)], "recommendation": selection_evidence,
            "scores": scores, "summaries": summaries,
            "weight_sensitivity": sensitivity(reference, objective, protocol["weight_scenarios"]),
            "elapsed_seconds": elapsed, "offline_reference_seconds": 0 if online else cost,
            "online_confirmation_seconds": cost if online else None,
            "previous_active_seconds": prior,
            "limitations": ["Empirical finite-grid optimum only; unobserved seeds/hardware/configurations remain unproved.",
                            "Resource forecasts are heuristic, not calibrated safety/time bounds.",
                            "Repeated seeds are observed variation, not a confidence interval or a significant superiority claim.",
                            "Full reference cost is separate from online search; test is not used for selection.",
                            "Concurrent non-detected workloads/thermal/power may still affect wall time; preflight is not isolation attestation."]}


def markdown(report):
    p = report["protocol"]
    lines = ["# Nghiên cứu lựa chọn cấu hình detection", "", f"Status: **{report['verification_status']}**.",
             f"Dataset: {p['dataset_source']}; device: {p['hardware'].get('gpu') or 'CPU'}.",
             f"Grid: {len(p['configs'])}; seeds: {p['training_seeds']}; horizon: {p['full_epochs']}epoch.",
             f"Environment: {p['measurement_environment']}; validation metric: mAP50-95, not percent captions correct.",
             "U=wA*mean(A)-wT*mean(T)/Tref-wM*max(M)/Mref; parameters frozen before search.",
             f"Objective: {p['objective']}", "", "## Tất cả cấu hình và cơ sở lựa chọn", "",
             "| ID | Model | Image | Batch / accum | Precision | LR / decay | Full seeds | State | A mean±SD | T mean(s) | M max(MiB) | U |",
             "|---|---|---:|---|---|---|---|---|---:|---:|---:|---:|"]
    def fmt(v):
        return "—" if v is None else f"{v:.6f}"
    for row in report["candidates"]:
        c, m = row["config"], row["measurement"]
        lines.append(f"| {row['config_id']} | {Path(c['model']).name} | {c['imgsz']} | {c['batch']} / {c['accumulation']} | {c['precision']} | {c['lr0']} / {c['weight_decay']} | {len(row['completed_seeds'])}/{len(row['required_seeds'])} | {row['state']} | {fmt(m['quality_mean'])}±{fmt(m.get('quality_sd'))} | {fmt(m['time_mean'])} | {fmt(m['memory_max'])} | {fmt(row['utility'])} |")
    lines += ["", f"Pareto: {report['pareto']}", "", "## Recommendation (không có số liệu giả)", ""]
    if report["recommendation"]:
        lines += ["```json", json.dumps(report["recommendation"], ensure_ascii=False, indent=2), "```"]
    else:
        lines += ["Chưa có cấu hình đủ bằng chứng/khả thi để chọn."]
    lines += ["", "## So sánh phương pháp", "", "| Method | Feasible | Regret mean | Actual search mean(s) | OOM |", "|---|---:|---:|---:|---:|"]
    for row in report["summaries"]:
        lines.append(f"| {row['method']} | {row['feasibility_rate']:.2%} | {fmt(row['objective_regret_mean'])} | {fmt(row['search_seconds_mean'])} | {row['oom_count']} |")
    if report["mode"] == "online_verification":
        lines += ["Không có oracle độc lập ở production; không suy ra superiority hoặc regret0 từ chính selection này."]
    lines += ["", "## Sensitivity (không thay selection chính)", "", "| Scenario | Weights A/T/M | Selected | U |", "|---|---|---|---:|"]
    for row in report["weight_sensitivity"]:
        lines.append(f"| {row['scenario']} | {row['weights']} | {row['selected']} | {fmt(row['utility'])} |")
    lines += ["", f"Elapsed active: {report['elapsed_seconds']}; offline oracle seconds: {report['offline_reference_seconds']}; online confirmations: {report['online_confirmation_seconds']}.",
              "Profile/startup/failures consume online budget; resume has additional explicitly recorded cost.", "", "## Giới hạn", ""]
    lines += [f"- {s}" for s in report["limitations"]]
    return "\n".join(lines) + "\n"


def publish(folder, protocol, decisions, results, elapsed, prior=None):
    report = summarize(protocol, decisions, results, elapsed, prior)
    write_json(folder / "study.json", report)
    (folder / "RESEARCH.md").write_text(markdown(report), encoding="utf-8")
    r = report["recommendation"]
    if report["empirical_best"]:
        write_json(folder / "selected.json", {"schema_version": 3, "backend": "yolo_research", "task": "detection",
                   "status": report["verification_status"], "trainable": True, "config": r["config"],
                   "device": protocol["hardware"]["device"], "full_epochs": protocol["full_epochs"],
                   "seed": protocol["training_seeds"][0], "research_provenance": str(folder / "study.json"),
                   "manifest": str(folder / "dataset.json"), "data": str(folder / "full_data" / "data.yaml"),
                   "protocol_sha256": protocol["protocol_sha256"], "optimality_scope": report["verification_status"]})
    elif (folder / "selected.json").exists():
        old = json.loads((folder / "selected.json").read_text(encoding="utf-8"))
        write_json(folder / "selected.previous.json", old)
        write_json(folder / "selected.json", {**old, "trainable": False, "status": report["verification_status"]})
    write_json(folder / "status.json", {"phase": "complete", "reference_complete": report["reference_complete"], "verification_status": report["verification_status"]})
    print(f"Status: {report['verification_status']} | selected: {report['empirical_best']} | Pareto: {report['pareto']}", flush=True)
    print(f"Report: {folder / 'RESEARCH.md'}", flush=True)
    return 0 if report["empirical_best"] else 3
