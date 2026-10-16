"""Rebuildable report: recommendations are separate from proxies and oracle."""

import json
import sys
from pathlib import Path

from yolo_tuner.pipeline import write_json
from yolo_tuner.research import Objective
from .schema import measurements, decision, complete
from .ledger import read_events, worker_cost
from .terminal import summary_view
from .selection import select, presets, rule_scores


def make_report(protocol, state, events):
    modern = protocol.get("execution_policy", {}).get("version", 0) >= 2
    rows = measurements(protocol["configs"], state["results"], protocol["training_seeds"], protocol["full_epochs"], Objective(**protocol["objective"]))
    selected = select(rows, protocol)
    clean = protocol["measurement_environment"] == "uncontended_preflight"
    if not clean:
        selected.update(verification_status="contended_smoke_not_certified", selected_index=None, selected_config_id=None)
    last = next((e for e in reversed(events) if e["kind"] == "selection_committed"), None)
    # One terminal cumulative clock per execution, not commit + post-commit
    # export error counted twice. Works for preserved early-v2 journals too.
    sessions, terminal, committed = [], None, False
    for event in events:
        if event["kind"] in {"session_started", "resume_requested"}:
            if terminal is not None:
                sessions.append(terminal)
            terminal, committed = None, False
        elif event["kind"] == "selection_committed":
            terminal, committed = event["payload"]["session_seconds"], True
        elif modern and event["kind"] == "publication_finished" and committed:
            terminal = event["payload"]["session_seconds"]
        elif event["kind"] == "session_interrupted" and not committed:
            terminal = event["payload"]["session_seconds"]
    if terminal is not None:
        sessions.append(terminal)
    online = sum(sessions) if last else None
    additional_budget = sum(e["payload"]["additional_budget_seconds"] for e in events if e["kind"] == "resume_requested")
    total_budget = protocol["budget_seconds"] + additional_budget
    ranked = sorted((r for r in rows if r["utility"] is not None), key=lambda r: r["utility"], reverse=True)
    recommendation = None
    index = selected["selected_index"]
    if index is not None:
        row = rows[index]
        recommendation = {"config": row["config"], "validation_mean": row["quality_mean"], "validation_sd_not_ci": row["quality_sd"],
                          "full_worker_seconds_mean": row["time_mean"], "memory_max_mb": row["memory_max"], "memory_kind": protocol["memory_kind"],
                          "completed_seeds": row["completed_seeds"], "full_epochs": protocol["full_epochs"], "utility": row["utility"],
                          "runner_up_margin": ranked[0]["utility"] - ranked[1]["utility"] if len(ranked) > 1 else None,
                          "future_seed_or_hardware_guaranteed": False, "test_metric": None, "test_used_for_selection": False,
                          "prediction": state.get("cost_predictions", {}).get(str(index)), "optimality_scope": selected["verification_status"]}
    report = {"schema_version": 4, "adapter": "research_v2", "protocol_sha256": protocol["protocol_sha256"],
            "method": protocol["method"], "task": protocol["task"], "decision": selected, "candidates": rows,
            "recommendation": recommendation, "online_session_seconds": online,
            "budget_seconds": protocol["budget_seconds"], "total_granted_budget_seconds": total_budget,
            "budget_overrun_seconds": max(0, online - total_budget) if online is not None else None,
            "all_attempt_worker_seconds": worker_cost(events),
            "proxy_worker_seconds": sum(r["worker_wall_seconds"] for r in state["probes"]),
            "preparation_seconds": protocol["preparation_seconds"],
            "reference_cost_seconds": online if protocol["mode"] == "offline_reference" else 0,
            "oracle_used_for_selection": False, "objective_regret": None,
            "ledger_event_count": len(events), "ledger_head": events[-1]["sha256"] if events else None,
            "additional_resume_budget_seconds": additional_budget,
            "limitations": ["Finite declared space and required observed seeds only; no universal optimum.",
                            "Forecasts have heuristic margins, not guaranteed bounds; no quality surrogate is used.",
                            "Preflight detects known ML processes, not all contention/thermal effects.",
                            "Only observed hardware is validated; CPU RSS is not VRAM.",
                            "Small-domain engineering benchmarks are exploratory, not statistical superiority."]}
    if "selection_rule" in protocol:
        scores, fallback = rule_scores(rows, protocol)
        report.update(selection_rule=protocol["selection_rule"], selection_scores=scores, selection_fallback=fallback,
                      presets=presets(rows, protocol),
                      pareto_measurements=[{k: r[k] for k in ("config_id", "config", "quality_mean", "quality_sd", "time_mean", "memory_max", "completed_seeds")} for r in rows if r["config_id"] in selected["pareto"]])
        if recommendation:
            row = rows[index]
            recommendation["runner_up_margin"] = None if protocol["selection_rule"] != "weighted" else recommendation["runner_up_margin"]
            recommendation["selection_score"] = selected["selection_score"]
            recommendation["prediction_before_first_full"] = next((r.get("scheduling_prediction") for r in row["runs"] if r["mode"] == "train"), None)
    if modern:
        finalizations = [e["payload"] for e in events if e["kind"] == "finalization_verified"]
        report.update(execution_policy=protocol["execution_policy"],
                      finalization_seconds=sum(e["seconds"] for e in finalizations),
                      integrity_diagnostics=finalizations[-1].get("integrity") if finalizations else None,
                      timing_boundary="after primary report/CLI export when publication_finished exists; final timing receipt refresh excluded",
                      hard_deadline_guaranteed=False)
        report["limitations"][1] = "Forecast margins are heuristic. Quality priors may order search but never replace measured full confirmations."
        report["limitations"] += ["Metadata cache is not anti-tamper attestation; full content hashes are mandatory at execution boundaries.",
                                  "Finalization reserve is measured/heuristic; slow filesystems and mandatory audit can exceed a soft end-to-end budget."]
    if "acquisition" in protocol:
        report.update(acquisition_policy=protocol["acquisition"], acquisition_history=state.get("acquisition_history", []),
                      acquisition_uses_full_confirmations_only=True, uncertainty_calibrated=False)
        report["limitations"] += ["Adaptive quality model uses complete full-data/all-seed outcomes only; optimism/novelty are heuristic, not statistical upper bounds.",
                                  "No early-quality pruning in adaptive_full does not guarantee exploration of every slow starter before budget exhaustion."]
    return report


def markdown(report, protocol):
    def f(value):
        return "—" if value is None else f"{value:.5f}"
    lines = ["# Cấu hình huấn luyện — pipeline v2", "", f"Trạng thái: {report['decision']['verification_status']}",
             f"Dataset: {protocol['dataset_source']}", f"Hardware: {protocol['hardware'].get('gpu') or 'CPU'}; task={protocol['task']}; method={protocol['method']}.",
             f"Horizon={protocol['full_epochs']}; required seeds={protocol['training_seeds']}; shared budget={protocol['budget_seconds']}s.",
             f"Metric: {protocol['quality_definition']}. Bộ nhớ: {protocol['memory_kind']}.",
             f"Objective khóa trước: {protocol['objective']}", "", "## Tất cả ứng viên", "",
             "| ID | Model | Image | Batch/accum | LR | Full seeds | State | Val mean±SD | Worker T(s) | M max(MiB) | U |",
             "|---|---|---:|---|---:|---|---|---:|---:|---:|---:|"]
    for r in report["candidates"]:
        c = r["config"]
        lines.append(f"| {r['config_id']} | {Path(c['model']).name} | {c['imgsz']} | {c['batch']}/{c['accumulation']} | {c['lr0']} | {len(r['completed_seeds'])}/{len(r['required_seeds'])} | {r['state']} | {f(r['quality_mean'])}±{f(r['quality_sd'])} | {f(r['time_mean'])} | {f(r['memory_max'])} | {f(r['utility'])} |")
    lines += ["", "## Recommendation", "", "```json", json.dumps(report["recommendation"], ensure_ascii=False, indent=2), "```",
              "", f"Pareto: {report['decision']['pareto']}",
              f"Online session: {report['online_session_seconds']}s; budget overrun: {report['budget_overrun_seconds']}s; all worker attempts: {report['all_attempt_worker_seconds']}s; proxy workers: {report['proxy_worker_seconds']}s.",
              "Không cộng lại worker time vào session time; worker spans nằm bên trong session. Preparation đã nằm trong online budget nếu chạy optimize --data; prepared-start công bố preparation riêng.",
              "Test không dùng để chọn model/config/checkpoint; SD không là confidence interval.", "", "## Giới hạn", ""]
    lines += [f"- {s}" for s in report["limitations"]]
    if "acquisition_policy" in report:
        lines += ["", "## Thuật toán chọn phép thử", "", f"Acquisition khóa trước: {report['acquisition_policy']}",
                  "Chỉ quality từ cấu hình đủ full-data/horizon/all-seeds huấn luyện surrogate. Không dùng test, reference hoặc quality epoch sớm để loại cấu hình.",
                  "Uncertainty/affordability là heuristic; chưa có guarantee thống kê hoặc tối ưu toàn cục.", "",
                  "| Step | Config index | Reason | Remaining(s) |", "|---|---:|---|---:|"]
        for entry in report["acquisition_history"]:
            lines.append(f"| {entry['step']} | {entry['index']} | {entry['reason']} | {entry['remaining_seconds']:.3f} |")
    if "presets" in report:
        lines += ["", "## Quy tắc và presets", "", f"Rule: {report['selection_rule']}; fallback: {report['selection_fallback']}",
                  "Các presets chỉ dùng full confirmations thỏa constraints; không dùng để chọn lại objective sau khi xem test.", "",
                  "| Preset | ID | Val mean | T mean(s) | M max(MiB) |", "|---|---|---:|---:|---:|"]
        for name, preset in report["presets"].items():
            if preset:
                lines.append(f"| {name} | {preset['config_id']} | {f(preset['validation_mean'])} | {f(preset['time_mean_seconds'])} | {f(preset['memory_max_mib'])} |")
            else:
                lines.append(f"| {name} | chưa có full-confirmed feasible config | — | — | — |")
    return "\n".join(lines) + "\n"


def publish(folder, protocol, state, display=True):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    events = read_events(folder / "events.jsonl")
    report = make_report(protocol, state, events)
    write_json(folder / "report.json", report)
    if "presets" in report:
        write_json(folder / "presets.json", {"protocol_sha256": protocol["protocol_sha256"], "full_epochs": protocol["full_epochs"],
                                            "memory_kind": protocol["memory_kind"], "presets": report["presets"], "test_used": False})
        write_json(folder / "pareto.json", {"protocol_sha256": protocol["protocol_sha256"], "observed_only": True, "rows": report["pareto_measurements"]})
    (folder / "REPORT.md").write_text(markdown(report, protocol), encoding="utf-8")
    index = report["decision"]["selected_index"]
    if index is not None:
        row = report["candidates"][index]
        seed = protocol["training_seeds"][0]
        checkpoint = next(r["best_checkpoint"] for r in reversed(row["runs"]) if r["seed"] == seed and complete(r, protocol["full_epochs"]))
        write_json(folder / "selected.json", {"schema_version": 4, "backend": "research_v2", "trainable": True,
                   "config": row["config"], "index": index, "task": protocol["task"], "seed": seed,
                   "checkpoint": checkpoint, "full_epochs": protocol["full_epochs"], "run": str(folder),
                   "protocol_sha256": protocol["protocol_sha256"], "status": report["decision"]["verification_status"]})
    write_json(folder / "status.json", {"phase": "complete", **report["decision"]})
    if display:
        summary_view(folder, protocol, report)
        print("  Test không dùng để chọn cấu hình; chưa tự động đánh giá test.", flush=True)
    return report
