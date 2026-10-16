"""Cold-process paired benchmark; decisions locked BEFORE independent reference."""

import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from yolo_tuner.pipeline import new_session, write_json
from yolo_tuner.research import digest
from .workflow import audit_run, source_map, idle
from .selection import select, rule_scores
from .diagnostics import first_full_seconds, proxy_reference_diagnostics
from .ledger import read_events
from . import __version__


def child_command(*arguments):
    """Do not assume the repository-only train_config.py exists in a wheel."""
    return [sys.executable, "-m", "research_pipeline.v2", *map(str, arguments)]


def _single_run(directory):
    candidates = sorted(p for p in Path(directory).iterdir() if p.is_dir() and (p / "protocol.json").is_file())
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one benchmark child run in {directory}; found {len(candidates)}")
    return candidates[0].resolve()


def _finished_run(path):
    """A stale 'search' flag or unjournaled crash is never declared complete."""
    folder = Path(path).resolve()
    events = read_events(folder / "events.jsonl")
    started = {e["payload"]["attempt_id"] for e in events if e["kind"] == "trial_started"}
    completed = {e["payload"]["attempt_id"] for e in events if e["kind"] == "trial_completed"}
    if started - completed:
        raise ValueError(f"Benchmark child has unknown crash cost / unjournaled worker: {folder}. Preserve evidence; cannot safely resume or finalize.")
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    if status.get("phase") != "complete":
        raise ValueError(f"Benchmark child is not complete: {folder}. This command only rebuilds reports, never starts training.")
    return audit_run(folder)


def compare(runs, reference_path, allow_controller_patch=False):
    _, reference_protocol, reference, _ = audit_run(reference_path)
    ref = {r["config_id"]: r for r in reference}
    ref_complete = all(r["complete"] for r in reference)
    feasible = [r for r in reference if r["utility"] is not None]
    best = max(feasible, key=lambda r: r["utility"]) if feasible and ref_complete else None
    scores, _ = rule_scores(reference, reference_protocol)
    reference_decision = select(reference, reference_protocol)
    outcomes = []
    source_differences = {}
    for path in runs:
        folder, p, rows, checked = audit_run(path)
        for key in ("dataset_content_id", "task", "full_epochs", "training_seeds", "objective", "runtime_versions", "memory_kind"):
            if p[key] != reference_protocol[key]:
                raise ValueError(f"Benchmark domain/protocol mismatch: {key}")
        if p.get("selection_rule", "weighted") != reference_protocol.get("selection_rule", "weighted"):
            raise ValueError("Benchmark selection-rule mismatch")
        if p.get("execution_policy") != reference_protocol.get("execution_policy"):
            raise ValueError("Benchmark execution-policy mismatch")
        if p.get("acquisition") != reference_protocol.get("acquisition"):
            raise ValueError("Benchmark acquisition-policy mismatch")
        different = {k for k in p["source_files"] | reference_protocol["source_files"] if p["source_files"].get(k) != reference_protocol["source_files"].get(k)}
        if different:
            permitted = {"research_pipeline/v2/workflow.py", "research_pipeline/v2/benchmark.py"}
            if not allow_controller_patch or not different <= permitted:
                raise ValueError("Source differs; explicit disclosed controller-only patch required; training-source changes never allowed")
            source_differences[str(folder)] = sorted(different)
        if p["config_ids"] != reference_protocol["config_ids"]:
            raise ValueError("Candidate space mismatch")
        for key in ("device", "gpu_uuid", "host_label", "torch_version", "driver_version"):
            if p["hardware"].get(key) != reference_protocol["hardware"].get(key):
                raise ValueError(f"Physical hardware identity differs: {key}")
        selected = json.loads((folder / "decision.locked.json").read_text(encoding="utf-8"))
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        key = selected["selected_config_id"]
        oracle_row = ref.get(key)
        regret = best["utility"] - oracle_row["utility"] if best and oracle_row and oracle_row["utility"] is not None else None
        rule_best = reference_decision["selected_config_id"]
        rule_regret = scores[rule_best] - scores[key] if ref_complete and rule_best in scores and key in scores and p.get("selection_rule", "weighted") != "knee" else None
        state_path = folder / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"probes": []}
        outcomes.append({"run": str(folder), "method": p["method"], "search_seed": p["search_seed"],
                         "selected": key, "feasible": key is not None, "reference_feasibility_disagreement": key is not None and (not oracle_row or oracle_row["utility"] is None),
                         "objective_regret": regret, "selection_rule_regret": rule_regret, "online_seconds": report["online_session_seconds"],
                         "budget_seconds": p["budget_seconds"], "overrun_seconds": report["budget_overrun_seconds"],
                         "checked_attempts": checked, "time_to_first_full_config_seconds": first_full_seconds(folder, p, rows),
                         "proxy_diagnostics": proxy_reference_diagnostics(state, p, reference)})
    summaries = []
    for method in dict.fromkeys(r["method"] for r in outcomes):
        rows = [r for r in outcomes if r["method"] == method]
        regrets = [r["objective_regret"] for r in rows if r["objective_regret"] is not None]
        summaries.append({"method": method, "n": len(rows), "feasible_count": sum(r["feasible"] for r in rows),
                          "feasibility_rate": statistics.mean(r["feasible"] for r in rows),
                          "conditional_regret_mean": statistics.mean(regrets) if regrets else None,
                          "conditional_regret_sd": statistics.stdev(regrets) if len(regrets) > 1 else None,
                          "regret_available_count": len(regrets), "online_seconds_mean": statistics.mean(r["online_seconds"] for r in rows),
                          "failed_selection_count": sum(not r["feasible"] for r in rows)})
    return {"schema_version": 1, "scope": "exploratory paired repeated search on one domain/hardware; not general superiority",
            "outcomes": outcomes, "summaries": summaries, "reference_complete": ref_complete,
            "reference": reference_path, "reference_selected": best["config_id"] if best else None,
            "reference_cost_seconds": json.loads((Path(reference_path) / "report.json").read_text(encoding="utf-8"))["online_session_seconds"],
            "regret_rule": "objective_regret compares weighted U on the same independent reference; rule-regret also provided except population-dependent knee; failures null",
            "selection_rule": reference_protocol.get("selection_rule", "weighted"),
            "reference_rule_selected": reference_decision["selected_config_id"],
            "source_differences": source_differences,
            "source_comparison": "explicit controller-only compatibility patch disclosed; all worker/runner/model/metric sources identical" if source_differences else "same exact source mapping",
            "statistical_superiority_established": False, "unseen_gpu_validated": False}


def benchmark(args):
    idle()
    folder = new_session(args.output)
    preparation_start = time.monotonic()
    if args.max_memory is None:
        # Independent diagnostic process exits before all matched cold runs.
        info = json.loads(subprocess.check_output(child_command("doctor", "--device", args.device, "--json"), encoding="utf-8"))
        hw = info["hardware"]
        if not hw.get("ml_runtime_ready"):
            raise ValueError("No working ML runtime for benchmark")
        args.max_memory = (hw.get("vram_free_mb") or hw["ram_available_mb"]) * 0.8
    sources = source_map()
    manifest = {"sources": sources, "arguments": vars(args), "grid_sha256": None,
                "methods": args.methods, "search_seeds": args.search_seeds,
                "training_seeds": args.seeds, "test_used": False, "phase": "locked_before_search"}
    if args.grid:
        from research_pipeline.integrity import sha
        manifest["grid_sha256"] = sha(args.grid)
    manifest["external_shared_preparation_seconds"] = time.monotonic() - preparation_start
    manifest["sha256"] = digest(manifest)
    write_json(folder / "benchmark_protocol.json", manifest)
    common = ["--data", args.data, "--task", args.task, "--device", args.device,
              "--space", args.space, "--full-epochs", str(args.full_epochs), "--probe-epochs", str(args.probe_epochs),
              "--train-subset", str(args.train_subset), "--val-subset", str(args.val_subset), "--split-seed", str(args.split_seed),
              "--trial-timeout", str(args.trial_timeout), "--probe-fraction", str(args.probe_fraction),
              "--cost-margin", str(args.cost_margin), "--cleanup-seconds", str(args.cleanup_seconds),
              "--time-scale", str(args.time_scale), "--memory-scale", str(args.memory_scale),
              "--min-quality", str(args.min_quality), "--weights", *map(str, args.weights), "--seeds", *map(str, args.seeds)]
    common += ["--selection-rule", args.selection_rule]
    common += ["--acquisition-ridge", str(getattr(args, "acquisition_ridge", 0.05)),
               "--acquisition-length-scale", str(getattr(args, "acquisition_length_scale", 1.0)),
               "--acquisition-exploration", str(getattr(args, "acquisition_exploration", 0.25)),
               "--acquisition-novelty", str(getattr(args, "acquisition_novelty", 0.05))]
    common += ["--workers", str(getattr(args, "workers", "auto")),
               "--image-cache-mb", str(getattr(args, "image_cache_mb", 64.0)),
               "--pin-memory", str(getattr(args, "pin_memory", "auto"))]
    if args.rungs:
        common += ["--rungs", *map(str, args.rungs)]
    for flag, value in (("--grid", args.grid), ("--precision", args.precision), ("--max-memory", args.max_memory), ("--max-time", args.max_time), ("--predictor", args.predictor)):
        if value is not None:
            common += [flag, str(value)]
    if args.models:
        common += ["--models", *args.models]
    paths = []
    for repeat, seed in enumerate(args.search_seeds):
        methods = args.methods[repeat % len(args.methods):] + args.methods[:repeat % len(args.methods)]
        for method in methods:
            if source_map() != sources:
                raise ValueError("Source changed during benchmark; no mixed-version comparison")
            output = folder / method / f"repeat_{seed}"
            cmd = child_command("optimize", *common, "--method", method, "--search-seed", str(seed),
                                "--budget-seconds", str(args.budget_seconds), "--output", str(output))
            # No GPU training concurrency: one child at a time, matched cold process.
            subprocess.run(cmd, check=True)
            run = _single_run(output)
            audit_run(run)
            paths.append(str(run))
            write_json(folder / "search_runs.partial.json", paths)
    locked = {"benchmark_sha256": manifest["sha256"], "runs": paths,
              "decision_hashes": {path: digest(json.loads((Path(path) / "decision.locked.json").read_text(encoding="utf-8"))) for path in paths}}
    boundary = folder / "decisions.locked.json"
    write_json(boundary, locked)
    output = folder / "reference"
    subprocess.run(child_command("optimize", *common, "--method", "random", "--search-seed", "123456",
                                 "--mode", "offline_reference", "--reference-boundary", str(boundary),
                                 "--budget-seconds", str(args.reference_budget_seconds), "--output", str(output)), check=True)
    reference = str(_single_run(output))
    report = compare(paths, reference)
    report["external_shared_preparation_seconds"] = manifest["external_shared_preparation_seconds"]
    return _publish_benchmark(folder, report)


def _publish_benchmark(folder, report):
    write_json(folder / "BENCHMARK.json", report)
    lines = ["# Benchmark nghiên cứu v2", "", report["scope"], "",
             "| Method | Feasible | Conditional regret mean | Actual online seconds mean |", "|---|---:|---:|---:|"]
    for r in report["summaries"]:
        lines.append(f"| {r['method']} | {r['feasible_count']}/{r['n']} | {r['conditional_regret_mean']} (available {r['regret_available_count']}) | {r['online_seconds_mean']:.2f} |")
    lines += ["", f"Independent full reference complete: {report['reference_complete']}; reference cost {report['reference_cost_seconds']}s.",
              "Failed choices have null regret, not zero. Decisions locked before reference; no test evaluation in this benchmark.",
              "Repeated searches on one dataset/GPU do not establish statistical superiority or unseen-GPU reliability."]
    (folder / "BENCHMARK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Benchmark: {folder / 'BENCHMARK.md'}", flush=True)
    return report


def finalize_benchmark(path, reference_path=None):
    """Audit an existing finished study and rebuild derived reports, without ML."""
    folder = Path(path).resolve()
    manifest = json.loads((folder / "benchmark_protocol.json").read_text(encoding="utf-8"))
    if manifest.get("sha256") != digest({k: v for k, v in manifest.items() if k != "sha256"}):
        raise ValueError("Benchmark protocol checksum mismatch")
    boundary = folder / "decisions.locked.json"
    locked = json.loads(boundary.read_text(encoding="utf-8"))
    paths = locked.get("runs", [])
    if (locked.get("benchmark_sha256") != manifest["sha256"] or not paths or len(paths) != len(set(paths))
            or set(paths) != set(locked.get("decision_hashes", {}))):
        raise ValueError("Benchmark decision boundary differs from protocol")
    expected = {(method, seed) for method in manifest["methods"] for seed in manifest["search_seeds"]}
    observed = set()
    for path in paths:
        run = Path(path).resolve()
        if not run.is_relative_to(folder):
            raise ValueError("Benchmark online run lies outside declared study")
        _, p, _, _ = _finished_run(run)
        identity = (p["method"], p["search_seed"])
        if identity in observed or identity not in expected or p["mode"] != "online":
            raise ValueError("Benchmark online method/repeat boundary mismatch")
        observed.add(identity)
        if p["source_files"] != manifest["sources"]:
            raise ValueError("Benchmark child sources differ from locked study")
        decision = json.loads((run / "decision.locked.json").read_text(encoding="utf-8"))
        if digest(decision) != locked["decision_hashes"][path]:
            raise ValueError("Benchmark locked online decision changed")
    if observed != expected:
        raise ValueError("Benchmark missing online method/repeat; incomplete study cannot be finalized")
    reference = Path(reference_path).resolve() if reference_path else _single_run(folder / "reference")
    _, p, rows, _ = _finished_run(reference)
    if not rows or not all(row["complete"] for row in rows):
        raise ValueError("Independent reference has not completed every configuration/seed; no reference optimum or benchmark finalization")
    from research_pipeline.integrity import sha
    if (p["mode"] != "offline_reference" or p.get("reference_boundary_sha256") != sha(boundary)
            or Path(p.get("reference_boundary") or "").resolve() != boundary
            or p["source_files"] != manifest["sources"]):
        raise ValueError("Independent reference does not match this locked benchmark boundary/source")
    report = compare(paths, str(reference))
    report.update(external_shared_preparation_seconds=manifest.get("external_shared_preparation_seconds", 0),
                  report_generation={"implementation_version": __version__, "analysis_only": True, "training_started": False})
    return _publish_benchmark(folder, report)
