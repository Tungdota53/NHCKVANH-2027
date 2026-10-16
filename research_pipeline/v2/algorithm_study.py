"""Post-decision, raw-audited algorithm evidence; never an online search input.

Search-repeat uncertainty is conditional on the observed dataset, machine and
independent reference. It is not a confidence interval over datasets or GPUs.
"""

from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics

from research_pipeline.integrity import sha
from yolo_tuner.pipeline import write_json
from yolo_tuner.research import digest
from . import benchmark


BASELINES = ("random", "reserve", "default", "successive_halving")
MIN_BOOTSTRAP_REPEATS = 5


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _finite(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or positive and value <= 0:
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return float(value)


def descriptive(values):
    values = [_finite(value, "measurement") for value in values]
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "sd": statistics.stdev(values) if len(values) > 1 else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def _quantile(ordered, probability):
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[min(lower + 1, len(ordered) - 1)] * fraction


def bootstrap_mean(values, *, seed=2027, samples=2000, confidence=0.95):
    """Seeded percentile CI for search repeats, not independent domain evidence."""
    values = [_finite(value, "bootstrap value") for value in values]
    if type(seed) is not int or type(samples) is not int or not 100 <= samples <= 20000:
        raise ValueError("Bootstrap requires an integer seed and 100..20000 samples")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("Bootstrap confidence must be finite and in (0,1)")
    result = {**descriptive(values), "confidence": confidence, "seed": seed, "resamples": samples,
              "interval": None, "status": "insufficient_search_repeats",
              "minimum_repeats": MIN_BOOTSTRAP_REPEATS,
              "unit": "paired search seed within one fixed study, not training seeds/datasets/GPUs",
              "assumption": "independent search repetitions conditional on this fixed domain/reference; not verified by seed labels",
              "coverage_guaranteed": False}
    if len(values) >= MIN_BOOTSTRAP_REPEATS:
        rng = random.Random(seed)
        means = sorted(statistics.mean(values[rng.randrange(len(values))] for _ in values) for _ in range(samples))
        tail = (1 - confidence) / 2
        result.update(status="exploratory_conditional_percentile_bootstrap", interval=[_quantile(means, tail), _quantile(means, 1 - tail)])
    return result


def _hardware_domain(hardware):
    device = str(hardware.get("device"))
    if device == "cpu":
        host = {"host_label": hardware.get("host_label"), "cpu": hardware.get("cpu")}
        if not host["host_label"] or not host["cpu"]:
            raise ValueError("CPU study needs an identified host label and CPU description")
        return {"kind": "cpu_host", "identity": digest(host), **host,
                "identity_basis": "host label plus CPU description; not unique physical silicon"}
    if device == "mps":
        host = {"host_label": hardware.get("host_label"), "cpu": hardware.get("cpu")}
        if not host["host_label"]:
            raise ValueError("MPS study needs an identified host")
        return {"kind": "mps_host_unverified_physical_gpu", "identity": digest(host), **host}
    uuid = hardware.get("gpu_uuid")
    if not uuid or not hardware.get("gpu_identity_verified", bool(uuid)):
        raise ValueError("GPU study requires a verified physical GPU UUID; names/ordinals do not identify a GPU")
    return {"kind": "physical_gpu", "identity": uuid, "gpu_uuid": uuid,
            "gpu": hardware.get("gpu"), "gpu_instance_uuid": hardware.get("gpu_instance_uuid")}


def _method_summary(outcomes, settings):
    summaries = []
    for method in sorted({row["method"] for row in outcomes}):
        rows = [row for row in outcomes if row["method"] == method]
        regrets = [row["objective_regret"] for row in rows if row["objective_regret"] is not None]
        summaries.append({"method": method, "search_repetitions": len(rows),
                          "verified_selection_count": sum(bool(row["feasible"]) for row in rows),
                          "no_verified_selection_count": sum(not row["feasible"] for row in rows),
                          "reference_feasibility_disagreement_count": sum(row["reference_feasibility_disagreement"] for row in rows),
                          "feasibility_rate": statistics.mean(bool(row["feasible"]) for row in rows),
                          "conditional_objective_regret": bootstrap_mean(regrets, **settings),
                          "regret_unavailable_count": len(rows) - len(regrets),
                          "online_seconds": descriptive([row["online_seconds"] for row in rows]),
                          "budget_overrun_seconds": descriptive([row["overrun_seconds"] for row in rows]),
                          "budget_overrun_count": sum(row["overrun_seconds"] > 0 for row in rows),
                          "time_to_first_full_config_seconds": descriptive([row["time_to_first_full_config_seconds"] for row in rows if row["time_to_first_full_config_seconds"] is not None])})
    return summaries


def _paired(outcomes, settings):
    by_method = defaultdict(dict)
    for row in outcomes:
        by_method[row["method"]][row["search_seed"]] = row
    comparisons = []
    for method in sorted(by_method):
        for baseline in BASELINES:
            if method == baseline or baseline not in by_method:
                continue
            left, right = by_method[method], by_method[baseline]
            if set(left) != set(right):
                raise ValueError("Methods must have exactly paired search seeds")
            seeds = sorted(left)
            metrics = {}
            for metric in ("objective_regret", "online_seconds", "time_to_first_full_config_seconds", "feasible"):
                pairs = [(seed, left[seed][metric], right[seed][metric]) for seed in seeds
                         if left[seed][metric] is not None and right[seed][metric] is not None]
                differences = [float(a) - float(b) for _, a, b in pairs]
                metrics[metric] = {"direction": "positive favors candidate" if metric == "feasible" else "negative favors candidate",
                                   "paired_seed_count": len(pairs), "paired_seeds": [seed for seed, _, _ in pairs],
                                   "unavailable_pairs": len(seeds) - len(pairs),
                                   "candidate_minus_baseline": bootstrap_mean(differences, **settings)}
            comparisons.append({"candidate": method, "baseline": baseline, "search_seeds": seeds,
                                "candidate_no_verified_selection": sum(not row["feasible"] for row in left.values()),
                                "baseline_no_verified_selection": sum(not row["feasible"] for row in right.values()),
                                "conditional_regret_warning": "Only seeds with available regret for BOTH methods are compared; failures remain separate, never regret zero.",
                                "metrics": metrics})
    return comparisons


def _audit_study(folder, reference_path, required_methods, settings):
    folder = Path(folder).resolve()
    manifest = _read(folder / "benchmark_protocol.json")
    if manifest.get("sha256") != digest({key: value for key, value in manifest.items() if key != "sha256"}):
        raise ValueError("Benchmark protocol checksum mismatch")
    if manifest.get("phase") != "locked_before_search":
        raise ValueError("Benchmark protocol was not declared locked before search")
    if manifest.get("test_used") is not False:
        raise ValueError("Study must declare no test used in search/reference decisions")
    methods, seeds = manifest["methods"], manifest["search_seeds"]
    if (not methods or any(not isinstance(method, str) for method in methods) or len(set(methods)) != len(methods)
            or not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError("Locked methods and search seeds must be nonempty and distinct")
    if not set(required_methods) <= set(methods):
        raise ValueError("Study is missing required method(s)")
    boundary = folder / "decisions.locked.json"
    locked = _read(boundary)
    paths = locked.get("runs", [])
    if (locked.get("benchmark_sha256") != manifest["sha256"] or not paths or len(set(paths)) != len(paths)
            or set(paths) != set(locked.get("decision_hashes", {}))):
        raise ValueError("Benchmark decision boundary mismatch")
    reference = Path(reference_path).resolve() if reference_path else benchmark._single_run(folder / "reference")
    _, ref_protocol, ref_rows, reference_checked = benchmark._finished_run(reference)
    if not ref_rows or not all(row["complete"] for row in ref_rows):
        raise ValueError("Independent exhaustive reference is incomplete; no study analysis")
    if (ref_protocol.get("mode") != "offline_reference" or ref_protocol.get("reference_boundary_sha256") != sha(boundary)
            or Path(ref_protocol.get("reference_boundary") or "").resolve() != boundary
            or ref_protocol["source_files"] != manifest["sources"]):
        raise ValueError("Independent reference boundary/source mismatch")
    if ref_protocol.get("measurement_environment") != "uncontended_preflight":
        raise ValueError("Reference was not measured under the declared uncontended preflight")
    if manifest.get("training_seeds", ref_protocol["training_seeds"]) != ref_protocol["training_seeds"]:
        raise ValueError("Reference training seeds differ from locked benchmark")
    declared = manifest.get("arguments", {})
    for key in ("full_epochs", "probe_epochs", "split_seed", "trial_timeout"):
        if key in declared and declared[key] != ref_protocol.get(key):
            raise ValueError(f"Reference differs from locked benchmark: {key}")
    if declared.get("seeds", ref_protocol["training_seeds"]) != ref_protocol["training_seeds"]:
        raise ValueError("Reference training seeds differ from locked benchmark arguments")
    hardware = _hardware_domain(ref_protocol["hardware"])
    reference_report = _read(reference / "report.json")
    reference_seconds = _finite(reference_report["online_session_seconds"], "Measured exhaustive reference cost", positive=True)
    expected = {(method, seed) for method in methods for seed in seeds}
    seen, checked, protocols, all_run_paths = set(), reference_checked, [], set()
    comparable_fields = ("dataset_content_id", "task", "config_ids", "full_epochs", "training_seeds", "objective", "selection_rule",
                         "runtime_versions", "memory_kind", "execution_policy", "acquisition", "subset_sha256", "split_seed", "probe_epochs", "trial_timeout", "scheduler")
    for path in paths:
        run = Path(path).resolve()
        if not run.is_relative_to(folder):
            raise ValueError("Online run lies outside the locked benchmark study")
        _, protocol, _, count = benchmark._finished_run(run)
        checked += count
        identity = (protocol["method"], protocol["search_seed"])
        if identity not in expected or identity in seen or protocol.get("mode") != "online":
            raise ValueError("Method/repeat does not match locked study")
        seen.add(identity)
        all_run_paths.add(str(run))
        if protocol["source_files"] != manifest["sources"]:
            raise ValueError("Online source differs from locked study")
        if digest(_read(run / "decision.locked.json")) != locked["decision_hashes"][path]:
            raise ValueError("Online decision changed after reference boundary")
        for key in comparable_fields:
            if protocol.get(key) != ref_protocol.get(key):
                raise ValueError(f"Matched algorithm study protocol mismatch: {key}")
        for key in ("device", "gpu_uuid", "gpu_instance_uuid", "host_label", "cpu", "torch_version", "driver_version"):
            if protocol["hardware"].get(key) != ref_protocol["hardware"].get(key):
                raise ValueError(f"Matched algorithm study device mismatch: {key}")
        if protocol.get("measurement_environment") != "uncontended_preflight":
            raise ValueError("Online run was not measured under the declared uncontended preflight")
        budget = _finite(protocol["budget_seconds"], "Search budget", positive=True)
        if not budget < reference_seconds:
            raise ValueError("Search budget must be smaller than measured exhaustive reference cost; this study is not a budget-constrained comparison")
        run_report = _read(run / "report.json")
        granted = _finite(run_report.get("total_granted_budget_seconds", budget), "Total granted search budget", positive=True)
        if granted != budget or run_report.get("additional_resume_budget_seconds", 0) != 0:
            raise ValueError("Additional resume grants change the locked search budget; not matched equal-budget evidence")
        protocols.append(protocol)
    if seen != expected:
        raise ValueError("Missing method/search-seed pair in benchmark study")
    budgets = {protocol["budget_seconds"] for protocol in protocols}
    if len(budgets) != 1:
        raise ValueError("Methods/repeats have different search budgets")
    budget = float(next(iter(budgets)))
    declared_budget = declared.get("budget_seconds")
    if declared_budget is not None and _finite(declared_budget, "Locked benchmark budget", positive=True) != budget:
        raise ValueError("Online budget differs from benchmark protocol")
    comparison = benchmark.compare(paths, str(reference))
    if comparison.get("reference_complete") is not True:
        raise ValueError("Independent exhaustive reference is incomplete")
    outcomes = comparison["outcomes"]
    if len(outcomes) != len(expected) or {(row["method"], row["search_seed"]) for row in outcomes} != expected:
        raise ValueError("Audited comparison method/repeat mismatch")
    for row in outcomes:
        _finite(row["online_seconds"], "Actual online seconds")
        _finite(row["overrun_seconds"], "Budget overrun seconds")
        if row["online_seconds"] < 0 or row["overrun_seconds"] < 0:
            raise ValueError("Measured algorithm costs must be nonnegative")
        if not row["feasible"] and row["objective_regret"] is not None:
            raise ValueError("No verified selection must have null regret, never zero")
        if row["objective_regret"] is not None:
            _finite(row["objective_regret"], "Objective regret")
        if row["budget_seconds"] != budget:
            raise ValueError("Audited comparison budget mismatch")
    domain = {"dataset_content_id": ref_protocol["dataset_content_id"], "task": ref_protocol["task"], "hardware": hardware}
    study = {"study": str(folder), "study_id": digest(str(folder))[:16], "benchmark_sha256": manifest["sha256"],
             "domain": domain, "domain_id": digest(domain)[:16], "methods": methods, "missing_baselines": [method for method in BASELINES if method not in methods],
             "search_seeds": seeds, "training_seeds_not_independent_repeats": ref_protocol["training_seeds"],
             "full_epochs": ref_protocol["full_epochs"], "config_count": len(ref_protocol["config_ids"]),
             "objective": ref_protocol["objective"], "selection_rule": ref_protocol.get("selection_rule", "weighted"),
             "reference": str(reference), "reference_complete": True, "reference_feasible_config_count": sum(row["utility"] is not None for row in ref_rows),
             "raw_audited_attempts": checked, "search_budget_seconds": budget, "measured_exhaustive_reference_seconds": reference_seconds,
             "search_budget_to_reference_cost_ratio": budget / reference_seconds,
             "budget_comparison": "declared search budget is smaller than this measured independent reference cost, not a prediction or runtime guarantee",
             "outcomes": outcomes, "method_summaries": _method_summary(outcomes, settings), "paired_comparisons": _paired(outcomes, settings),
             "any_actual_budget_overrun": any(row["overrun_seconds"] > 0 for row in outcomes),
             "statistical_superiority_established": False}
    return study, all_run_paths | {str(reference)}


def _summary(studies):
    datasets = sorted({study["domain"]["dataset_content_id"] for study in studies})
    hardware = {digest(study["domain"]["hardware"]): study["domain"]["hardware"] for study in studies}
    gpus = sorted({item["gpu_uuid"] for item in hardware.values() if item["kind"] == "physical_gpu"})
    cpus = {item["identity"]: item for item in hardware.values() if item["kind"] == "cpu_host"}
    totals = []
    for method in sorted({row["method"] for study in studies for row in study["outcomes"]}):
        rows = [row for study in studies for row in study["outcomes"] if row["method"] == method]
        per_study = [row for study in studies for row in study["method_summaries"] if row["method"] == method]
        totals.append({"method": method, "study_count": len(per_study), "observed_search_repetitions": len(rows),
                       "verified_selection_count": sum(bool(row["feasible"]) for row in rows),
                       "no_verified_selection_count": sum(not row["feasible"] for row in rows),
                       "regret_available_count": sum(row["objective_regret"] is not None for row in rows),
                       "reference_feasibility_disagreement_count": sum(row["reference_feasibility_disagreement"] for row in rows),
                       "budget_overrun_count": sum(row["overrun_seconds"] > 0 for row in rows),
                       "pooled_regret_or_domain_ci": None,
                       "aggregation_scope": "counts only; heterogeneous objectives/domains and reused seeds are not pooled as iid regret observations"})
    return {"study_count": len(studies), "unique_dataset_content_ids": datasets, "unique_dataset_count": len(datasets),
            "unique_observed_domain_count": len({study["domain_id"] for study in studies}),
            "physical_gpu_uuids": gpus, "physical_gpu_count": len(gpus), "cpu_hosts": list(cpus.values()), "cpu_host_count": len(cpus),
            "mps_host_count_not_verified_physical_gpus": len({item["identity"] for item in hardware.values() if item["kind"].startswith("mps_")}),
            "method_totals": totals, "cross_domain_confidence_interval": None,
            "independence_warning": "Search seeds and training seeds on one dataset are not independent datasets; CPU and CUDA on one host are not two physical GPUs. No cross-domain resampling or superiority inference is performed."}


def markdown(report):
    lines = ["# Algorithm study — audited post-decision evidence", "", report["scope"], "",
             f"Studies: {report['summary']['study_count']}; unique datasets: {report['summary']['unique_dataset_count']}; observed domains: {report['summary']['unique_observed_domain_count']}; physical GPUs: {report['summary']['physical_gpu_count']}; CPU hosts: {report['summary']['cpu_host_count']}.",
             "Training seeds are not search repetitions. Search repetitions are not independent datasets. Failed selections have null regret, never zero.",
             "", "## Observed totals (counts only)", "", "| Method | Search repetitions | Verified selections | No verified selection | Available regrets |", "|---|---:|---:|---:|---:|"]
    for row in report["summary"]["method_totals"]:
        lines.append(f"| {row['method']} | {row['observed_search_repetitions']} | {row['verified_selection_count']} | {row['no_verified_selection_count']} | {row['regret_available_count']} |")
    for study in report["studies"]:
        lines += ["", f"## Study {study['study_id']}", "", f"Path: {study['study']}", f"Dataset: {study['domain']['dataset_content_id']}; hardware: {study['domain']['hardware']}",
                  f"Configs={study['config_count']}; epochs={study['full_epochs']}; training seeds={study['training_seeds_not_independent_repeats']}; search seeds={study['search_seeds']}.",
                  f"Locked search budget={study['search_budget_seconds']}s; measured exhaustive reference={study['measured_exhaustive_reference_seconds']}s; ratio={study['search_budget_to_reference_cost_ratio']:.4f}. Not a runtime guarantee.",
                  f"Actual budget overrun observed: {study['any_actual_budget_overrun']}; missing baselines: {study['missing_baselines']}.", "",
                  "| Method | Verified / repetitions | Conditional regret mean | SD | Search-repeat CI / status |", "|---|---:|---:|---:|---|"]
        for row in study["method_summaries"]:
            regret = row["conditional_objective_regret"]
            lines.append(f"| {row['method']} | {row['verified_selection_count']}/{row['search_repetitions']} | {regret['mean']} | {regret['sd']} | {regret['interval']} / {regret['status']} (n={regret['n']}) |")
        lines += ["", "### Paired differences: candidate minus baseline", "", "Negative regret/time differences favor candidate; positive feasibility differences favor candidate.", "",
                  "| Candidate / baseline | Metric | Paired n | Mean difference | Search-repeat CI / status |", "|---|---|---:|---:|---|"]
        for pair in study["paired_comparisons"]:
            for metric, values in pair["metrics"].items():
                difference = values["candidate_minus_baseline"]
                lines.append(f"| {pair['candidate']} / {pair['baseline']} | {metric} | {difference['n']} | {difference['mean']} | {difference['interval']} / {difference['status']} |")
    lines += ["", "## Limitations", ""] + [f"- {limitation}" for limitation in report["limitations"]]
    return "\n".join(lines) + "\n"


def analyze_studies(manifest_path, output_directory=None):
    """Read a schema-1 manifest; optionally create NEW JSON/Markdown evidence.

    Manifest: {"schema_version": 1, "studies": [{"path": "study", "reference":
    "optional reference run"}], "required_methods": [], "bootstrap_seed": 2027,
    "bootstrap_samples": 2000, "confidence": 0.95}. Paths are manifest-relative.
    This function does not train, fit predictors, tune objectives, or rewrite any
    benchmark. Existing output directories are rejected.
    """
    manifest_path = Path(manifest_path).resolve()
    manifest = _read(manifest_path)
    entries = manifest.get("studies")
    if manifest.get("schema_version") != 1 or not isinstance(entries, list) or not entries:
        raise ValueError("Algorithm study manifest needs schema_version=1 and nonempty studies")
    output = Path(output_directory).resolve() if output_directory is not None else None
    if output and output.exists():
        raise ValueError("Analysis output directory already exists; preserve previous evidence and choose a new directory")
    required = manifest.get("required_methods", [])
    if not isinstance(required, list) or any(not isinstance(method, str) for method in required) or len(set(required)) != len(required):
        raise ValueError("required_methods must be a list of distinct method names")
    settings = {"seed": manifest.get("bootstrap_seed", 2027), "samples": manifest.get("bootstrap_samples", 2000), "confidence": manifest.get("confidence", 0.95)}
    bootstrap_mean([], **settings)  # Validate settings before reading raw trials.
    studies, seen_studies, seen_runs = [], set(), set()
    for entry in entries:
        if isinstance(entry, str):
            entry = {"path": entry}
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("Each study must provide a path")
        path = (manifest_path.parent / entry["path"]).resolve()
        reference = (manifest_path.parent / entry["reference"]).resolve() if entry.get("reference") else None
        if path in seen_studies:
            raise ValueError("Duplicate benchmark study; cannot inflate repeat/domain counts")
        seen_studies.add(path)
        study, runs = _audit_study(path, reference, required, settings)
        if seen_runs & runs:
            raise ValueError("Benchmark studies reuse underlying runs/reference; not separate experimental evidence")
        seen_runs.update(runs)
        studies.append(study)
    report = {"schema_version": 1, "analysis_manifest": str(manifest_path), "analysis_manifest_sha256": sha(manifest_path),
              "scope": "Exploratory, raw-audited, equal-declared-budget algorithm comparisons conditional on observed domains and independent exhaustive references; not universal superiority.",
              "analysis_only": True, "training_started": False, "predictor_fit_or_objective_tuning_performed": False,
              "reference_used_for_online_selection": False, "test_used_for_selection": False,
              "bootstrap": {**settings, "minimum_repeats": MIN_BOOTSTRAP_REPEATS, "unit": "search repetitions within each study only"},
              "studies": studies, "summary": _summary(studies), "statistical_superiority_established": False,
              "unseen_hardware_generalization_established": False,
              "limitations": ["All selections were locked before their independent exhaustive reference; references are analysis-only, never predictor-fit or acquisition input.",
                              "Regret uses weighted utility on the same independent reference. A knee/TOPSIS/Tchebycheff selection can have nonzero weighted regret without being incorrectly implemented.",
                              "Feasibility failures and reference-feasibility disagreements remain separate from conditional regret; no failure gets regret zero.",
                              "At least five usable search repetitions are required for an exploratory percentile bootstrap interval; this threshold does not establish statistical power or verify independence.",
                              "Intervals condition on this one observed reference and fixed training seeds; reference/training variability and domain shift are not resampled.",
                              "Paired conditional regret excludes pairs where either regret is unavailable; report feasibility alongside it to avoid survivor bias.",
                              "Multiple baseline/metric comparisons are exploratory with no multiplicity-adjusted significance or automatic winner declaration.",
                              "Budget restriction compares locked search budgets with measured reference wall cost, not a guaranteed lower bound on exhaustive training or a hard real-time guarantee.",
                              "Observed budget overruns, source/runtime matching and dataset/GPU/CPU counts are disclosed; seed counts cannot replace independent domains.",
                              "Counts across heterogeneous studies are descriptive only; there is no pooled iid regret interval, cross-domain confidence interval, or universal optimum claim."]}
    report["sha256"] = digest(report)
    if output:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "ALGORITHM_STUDY.json", report)
        (output / "ALGORITHM_STUDY.md").write_text(markdown(report), encoding="utf-8")
    return report
