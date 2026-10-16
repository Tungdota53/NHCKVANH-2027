"""Protocol preparation, real execution, integrity-first resume and raw audit."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import re
import time

import psutil
from yolo_tuner.adapters import export_yolo, subset
from .data import discover
from yolo_tuner.pipeline import new_session, runtime_info, write_json
from yolo_tuner.research import Objective, digest, verify_dataset_files
from yolo_tuner.verification import content_identity
from research_pipeline.integrity import ROOT, freeze_sources, hardware_changes, runtime, sha
from research_pipeline.study import exclusive_run, require_idle
from . import __version__
from .ledger import Ledger, read_events
from .schema import METRICS, TASKS, complete, lock_protocol, validate_protocol
from .space import build_space
from .model_features import describe_models, descriptor_for
from .data_features import describe_dataset
from .scheduler import run_search
from .runner import run_worker
from .report import publish, make_report
from .file_integrity import VerifiedFileCache
from .budget import FinalizationReserve
from .terminal import title, say, pairs, duration, command


def source_map():
    files = [*(ROOT / "research_pipeline").rglob("*.py"), *(ROOT / "yolo_tuner").glob("*.py")]
    # A source checkout includes this compatibility launcher; a wheel need not.
    if (ROOT / "train_config.py").is_file():
        files.append(ROOT / "train_config.py")
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}


def idle():
    require_idle()
    own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        if p.pid in own or "python" not in (p.info["name"] or "").lower():
            continue
        command = " ".join(p.info["cmdline"] or [])
        if "research_pipeline.v2.worker" in command or re.search(r"(?:train_config\.py|-m\s+research_pipeline\.v2)\s+(optimize|benchmark|resume|train-v2)\b", command):
            raise ValueError(f"Other v2 training process active: PID {p.pid}; refusing resource measurements")


def prepare(args, origin=None):
    title("CHUẨN BỊ LƯỢT CHẠY", "Đang kiểm tra dữ liệu, phần cứng và khóa protocol...")
    folder = new_session(args.output)
    ledger = Ledger(folder / "events.jsonl", origin=origin)
    ledger.emit("workflow_entered", command=args.command, budget_seconds=args.budget_seconds)
    write_json(folder / "status.json", {"phase": "preparing"})
    with ledger.span("discovery_hardware_and_export"):
        idle()
        manifest = discover(args.data, args.split_seed, args.task)
        if manifest["task"] not in TASKS or not manifest["validation_ready"]:
            raise ValueError("V2 needs supported supervised labels and independent validation; no fabricated targets")
        hw = runtime_info(args.device)
        hw["host_label"] = platform.node()
        if not hw["ml_runtime_ready"]:
            raise ValueError("No working ML runtime")
        sampled = subset(manifest, args.train_subset, args.val_subset)
        write_json(folder / "dataset.json", manifest)
        write_json(folder / "subset.json", sampled)
        configs = build_space(manifest["task"], hw, folder, args.grid, args.space, args.models, args.precision)
        descriptors = describe_models(manifest["task"], configs, len(manifest["names"]))
        dataset_descriptor = describe_dataset(manifest)
        for config in configs:
            layers = descriptor_for({"model_descriptors": descriptors}, config).get("layers")
            if manifest["task"] == "detection" and layers and config["freeze"] >= layers:
                raise ValueError("Freeze would remove every trainable model block")
        if manifest["task"] == "detection":
            export_yolo(manifest, folder / "full_data")
            export_yolo(sampled, folder / "subset_data")
    memory = args.max_memory if args.max_memory is not None else (hw.get("vram_free_mb") or hw["ram_available_mb"]) * 0.8
    objective = Objective("balanced", args.min_quality, args.max_time, memory, args.time_scale, args.memory_scale, tuple(args.weights))
    counts = manifest["captions"] if manifest["task"] == "captioning" else manifest["counts"]
    predictor = json.loads(Path(args.predictor).read_text(encoding="utf-8")) if args.predictor else None
    if predictor and predictor["sha256"] != digest({k: v for k, v in predictor.items() if k != "sha256"}):
        raise ValueError("Predictor card checksum mismatch")
    p = {"implementation_version": __version__, "mode": getattr(args, "mode", "online"), "method": args.method,
         "execution_policy": {"version": 2,
                              "integrity": "stat_cached_between_trials_full_hash_boundaries",
                              "budget": "end_to_end_soft_deadline_with_measured_finalization_reserve",
                              "data_loader": {"workers": getattr(args, "workers", "auto"),
                                              "cache_mb": getattr(args, "image_cache_mb", 64.0),
                                              "pin_memory": getattr(args, "pin_memory", "auto")}},
         "selection_rule": getattr(args, "selection_rule", "weighted"), "model_descriptors": descriptors, "dataset_descriptor": dataset_descriptor,
         "run_directory": str(folder), "task": manifest["task"], "dataset_source": manifest["source"],
         "dataset_sha256": digest(manifest), "dataset_content_id": content_identity(manifest), "subset_sha256": digest(sampled),
         "hardware": hw, "runtime_versions": runtime(), "configs": configs,
         "training_seeds": args.seeds, "split_seed": args.split_seed, "search_seed": args.search_seed,
         "full_epochs": args.full_epochs, "probe_epochs": args.probe_epochs, "trial_timeout": args.trial_timeout,
         "budget_seconds": args.budget_seconds, "objective": asdict(objective), "example_counts": counts, "num_classes": len(manifest["names"]),
         "quality_definition": METRICS[manifest["task"]],
         "determinism_policy": "strict torch deterministic algorithms; caption tiny encoder uses explicit adaptive-cell means for deterministic pooling backward",
         "memory_kind": "cuda_reserved_allocator_mib" if hw["device"] not in {"cpu", "mps"} else "process_tree_rss_mib_not_vram",
         "scheduler": {"policy_version": 2, "probe_fraction": args.probe_fraction, "cost_margin": args.cost_margin, "cleanup_seconds": args.cleanup_seconds,
                       "rungs": getattr(args, "rungs", None) or [e for e in (1, 2, 4) if e <= args.full_epochs]},
         "predictor": predictor, "source_files": source_map(),
         "model_files": {c["model"]: sha(c["model"]) for c in configs if Path(c["model"]).is_file()},
         "data_files": {p.relative_to(folder).as_posix(): sha(p) for d in ("full_data", "subset_data") for p in sorted((folder / d).rglob("*")) if p.is_file()},
         "preparation_seconds": ledger.elapsed, "measurement_environment": "uncontended_preflight",
         "test_policy": "never used for config/checkpoint/objective selection; historical datasets are not blind",
         "development_history": "0.5/0.6 findings and Penn-Fudan/Fashion/CIFAR/ViECAP informed policy; new local trials are development, not unseen-domain proof",
         "reference_boundary": getattr(args, "reference_boundary", None)}
    # Locked for every method in a study so baseline/reference comparison never
    # changes the representation or tuning choices after seeing outcomes.
    p["acquisition"] = {"version": 1, "kernel_ridge": getattr(args, "acquisition_ridge", 0.05),
                        "length_scale": getattr(args, "acquisition_length_scale", 1.0),
                        "exploration": getattr(args, "acquisition_exploration", 0.25),
                        "novelty_weight": getattr(args, "acquisition_novelty", 0.05)}
    if p["mode"] == "offline_reference" and not p["reference_boundary"]:
        raise ValueError("Independent reference requires previously locked search decisions")
    if p["reference_boundary"]:
        boundary = Path(p["reference_boundary"])
        locked = json.loads(boundary.read_text(encoding="utf-8"))
        if not locked.get("runs") or set(locked["runs"]) != set(locked.get("decision_hashes", {})):
            raise ValueError("Reference boundary must identify audited locked online decisions")
        from .schema import config_id
        for path in locked["runs"]:
            prior_folder, prior, _, _ = audit_run(path)
            d = json.loads((prior_folder / "decision.locked.json").read_text(encoding="utf-8"))
            if digest(d) != locked["decision_hashes"][path] or prior["mode"] != "online":
                raise ValueError("Reference boundary decision identity differs")
            if prior["dataset_content_id"] != p["dataset_content_id"] or prior["config_ids"] != [config_id(c) for c in configs]:
                raise ValueError("Reference boundary uses different data/space")
            for key in ("full_epochs", "training_seeds", "objective", "runtime_versions", "selection_rule", "execution_policy", "scheduler"):
                if digest(prior[key]) != digest(p[key]):
                    raise ValueError(f"Reference boundary differs: {key}")
        p["reference_boundary_sha256"] = sha(boundary)
    protocol = lock_protocol(p)
    freeze_sources(folder, protocol["source_files"])
    write_json(folder / "protocol.json", protocol)
    ledger.emit("protocol_locked", protocol_sha256=protocol["protocol_sha256"])
    write_json(folder / "status.json", {"phase": "prepared"})
    pairs([("Thư mục mới", folder), ("Task / thiết bị", f"{manifest['task']} / {hw.get('gpu') or hw['device']}"),
           ("Kế hoạch", f"{len(configs)} cấu hình × {len(args.seeds)} seeds × {args.full_epochs} epoch"),
           ("Ngân sách chung", duration(args.budget_seconds)), ("Protocol", folder / "protocol.json")])
    say("  Theo dõi ở cửa sổ khác: " + command(["status", "--run", str(folder)]))
    return folder, protocol, ledger


def verify_live(folder, protocol, cache=None, force=False):
    validate_protocol(protocol)
    def verify(path, expected, error):
        if cache is not None:
            cache.verify(path, expected, force=force)
        elif sha(path) != expected:
            raise ValueError(error)
    for name, expected in protocol["source_files"].items():
        verify(ROOT / name, expected, f"Source changed: {name}; new protocol required")
    if runtime() != protocol["runtime_versions"]:
        raise ValueError("Runtime changed; new protocol required")
    manifest = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    sampled = json.loads((folder / "subset.json").read_text(encoding="utf-8"))
    if digest(manifest) != protocol["dataset_sha256"] or digest(sampled) != protocol["subset_sha256"]:
        raise ValueError("Dataset/subset manifest changed")
    if cache is None:
        verify_dataset_files(manifest)
    else:
        for records in manifest["splits"].values():
            for record in records:
                if record.get("sha256"):
                    verify(record["image"], record["sha256"], "Dataset image changed")
    for name, expected in protocol["model_files"].items():
        verify(name, expected, "Model bytes changed")
    for name, expected in protocol["data_files"].items():
        verify(folder / name, expected, "Exported data changed")
    if protocol.get("reference_boundary") and sha(protocol["reference_boundary"]) != protocol["reference_boundary_sha256"]:
        raise ValueError("Pre-reference decision lock changed")
    return manifest, sampled


def audit_run(path, check_published=True):
    folder = Path(path).resolve()
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    validate_protocol(protocol)
    events = read_events(folder / "events.jsonl")
    for name, expected in protocol["source_files"].items():
        if sha(folder / "source_snapshot" / name) != expected:
            raise ValueError("Historical source snapshot changed")
    for name, expected in (("dataset.json", protocol["dataset_sha256"]), ("subset.json", protocol["subset_sha256"])):
        if digest(json.loads((folder / name).read_text(encoding="utf-8"))) != expected:
            raise ValueError("Historical manifest changed")
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    completions = {e["payload"]["attempt_id"]: e["payload"] for e in events if e["kind"] == "trial_completed"}
    attempts = [r for rs in state["results"].values() for r in rs] + state["probes"]
    seen = set()
    for result in attempts:
        attempt_id = result["attempt_id"]
        if attempt_id in seen or attempt_id not in completions:
            raise ValueError("Duplicate attempt or missing ledger completion")
        seen.add(attempt_id)
        trial = Path(result["trial_directory"]).resolve()
        if not trial.is_relative_to(folder):
            raise ValueError("Trial outside declared run")
        event = completions[attempt_id]
        if sha(trial / "parent_result.json") != event["parent_sha256"]:
            raise ValueError("Parent outcome differs from event ledger")
        parent = json.loads((trial / "parent_result.json").read_text(encoding="utf-8"))
        if any(result.get(k) != v for k, v in parent.items()):
            raise ValueError("State outcome differs from parent raw evidence")
        if sha(trial / "request.json") != parent["request_sha256"] or sha(trial / "result.json") != parent["raw_sha256"]:
            raise ValueError("Original request/raw result changed")
        req = json.loads((trial / "request.json").read_text(encoding="utf-8"))
        index = result["index"]
        if (req["config"] != protocol["configs"][index] or req["protocol_sha256"] != protocol["protocol_sha256"]
                or req["seed"] not in protocol["training_seeds"] or req["seed"] != result["seed"]
                or req["mode"] != result["mode"] or req["full_epochs"] != protocol["full_epochs"] or req["resume"] is not None):
            raise ValueError("Worker request differs from protocol")
        if "execution_policy" in protocol and req.get("data_loader") != protocol["execution_policy"]["data_loader"]:
            raise ValueError("Worker data-loader request differs from locked execution policy")
        full = result["mode"] == "train"
        expected_manifest = folder / ("dataset.json" if full else "subset.json")
        allowed_epochs = [protocol["full_epochs"]] if full else protocol["scheduler"].get("rungs", [protocol["probe_epochs"]]) if protocol["method"] == "successive_halving" else [protocol["probe_epochs"]]
        if Path(req["manifest"]).resolve() != expected_manifest or req["trial_epochs"] not in allowed_epochs:
            raise ValueError("Hidden proxy data/horizon in confirmation")
        if protocol["task"] == "detection" and Path(req["data"]).resolve() != folder / ("full_data" if full else "subset_data") / "data.yaml":
            raise ValueError("Hidden subset export")
        if complete(result, protocol["full_epochs"]):
            checkpoint = Path(result["best_checkpoint"]).resolve()
            if not checkpoint.is_relative_to(trial) or sha(checkpoint) != result["checkpoint_sha256"]:
                raise ValueError("Checkpoint identity changed")
        if event["worker_seconds"] != result["worker_wall_seconds"] or event["status"] != result["status"]:
            raise ValueError("Ledger time/status mismatch")
    if seen != set(completions):
        raise ValueError("Completed worker omitted from state; reconcile before resume")
    rebuilt = make_report(protocol, state, events)
    commits = [e for e in events if e["kind"] == "selection_committed"]
    published = rebuilt
    if commits:
        commit = commits[-1]
        committed_state = json.loads((folder / commit["payload"]["state_path"]).read_text(encoding="utf-8"))
        if digest(committed_state) != commit["payload"]["state_sha256"]:
            raise ValueError("Committed state changed")
        cutoff = commit["sequence"] + 1
        if protocol.get("execution_policy", {}).get("version", 0) >= 2:
            for following in events[cutoff:]:
                if following["kind"] in {"resume_requested", "session_started"}:
                    break
                cutoff = following["sequence"] + 1
        published = make_report(protocol, committed_state, events[:cutoff])
        if commit["payload"]["decision"] != published["decision"]:
            raise ValueError("Journaled selection differs from committed full evidence")
        status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        if status["phase"] == "complete" and digest(state) != digest(committed_state):
            raise ValueError("Completed state differs from committed state")
    if check_published and (folder / "decision.locked.json").exists():
        locked = json.loads((folder / "decision.locked.json").read_text(encoding="utf-8"))
        if published["decision"] != locked:
            raise ValueError("Locked decision differs from full outcomes")
    if check_published and (folder / "report.json").exists():
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        keys = ["decision", "candidates", "recommendation", "online_session_seconds", "all_attempt_worker_seconds"]
        if "selection_rule" in protocol:
            keys += ["selection_rule", "selection_scores", "selection_fallback", "presets", "pareto_measurements"]
        for key in keys:
            if published[key] != report[key]:
                raise ValueError(f"Published report changed: {key}")
        if "acquisition" in protocol:
            for key in ("acquisition_policy", "acquisition_history", "acquisition_uses_full_confirmations_only", "uncertainty_calibrated"):
                if published[key] != report[key]:
                    raise ValueError(f"Published acquisition trace changed: {key}")
        if "selection_rule" in protocol:
            expected = {"presets.json": {"protocol_sha256": protocol["protocol_sha256"], "full_epochs": protocol["full_epochs"],
                                         "memory_kind": protocol["memory_kind"], "presets": published["presets"], "test_used": False},
                        "pareto.json": {"protocol_sha256": protocol["protocol_sha256"], "observed_only": True, "rows": published["pareto_measurements"]}}
            for name, contents in expected.items():
                if (folder / name).exists() and json.loads((folder / name).read_text(encoding="utf-8")) != contents:
                    raise ValueError(f"Published artifact changed: {name}")
        if (folder / "selected.json").exists():
            selection = json.loads((folder / "selected.json").read_text(encoding="utf-8"))
            index = published["decision"]["selected_index"]
            if (index is None or selection["index"] != index or selection["config"] != protocol["configs"][index]
                    or selection["protocol_sha256"] != protocol["protocol_sha256"] or selection["run"] != str(folder)):
                raise ValueError("Selection artifact differs from audited recommendation")
    return folder, protocol, rebuilt["candidates"], len(seen)


def execute(folder, protocol, ledger, initial=None, resume_budget=None, charge_preparation=True):
    with exclusive_run(folder):
        idle()
        modern = protocol.get("execution_policy", {}).get("version", 0) >= 2
        cache = VerifiedFileCache() if modern else None
        boundary_start = time.monotonic()
        verify_live(folder, protocol, cache=cache, force=True)
        boundary_seconds = time.monotonic() - boundary_start
        start_elapsed = ledger.prior if initial is not None or not charge_preparation else 0.0
        budget = resume_budget if resume_budget is not None else protocol["budget_seconds"]
        final_budget = FinalizationReserve(protocol["scheduler"]["cleanup_seconds"], boundary_seconds)
        if initial:
            prior_attempts = [r for rs in initial["results"].values() for r in rs] + initial["probes"]
            if modern:
                for result in prior_attempts:
                    final_budget.observe(result)
        def finalization_reserve():
            if not modern:
                return protocol["scheduler"]["cleanup_seconds"]
            return final_budget.seconds()
        def wall_remaining():
            return budget - (ledger.elapsed - start_elapsed)
        def remaining():
            # Scheduler subtracts its declared cleanup again; reserve only the
            # additional observed boundary/audit cost here, never double-count it.
            return wall_remaining() - max(0, finalization_reserve() - protocol["scheduler"]["cleanup_seconds"])
        write_json(folder / "status.json", {"phase": "search"})
        ledger.emit("session_started", additional_budget_seconds=budget, resume=initial is not None, preparation_charged_online=charge_preparation)
        title("TIẾP TỤC TÌM CẤU HÌNH" if initial is not None else "BẮT ĐẦU TÌM CẤU HÌNH",
              "PROBE: subset để đo chi phí. FULL: toàn bộ train/val, đủ epoch; không dùng test.")
        def evaluate(index, seed, epochs, full, seconds):
            say(f"\n  Budget lần thực thi còn: {duration(wall_remaining())}")
            before = time.monotonic()
            idle()
            whole, sampled = verify_live(folder, protocol, cache=cache)
            seconds = min(seconds - (time.monotonic() - before), remaining() - protocol["scheduler"]["cleanup_seconds"])
            if seconds <= 0.05:
                raise TimeoutError("Budget exhausted during integrity/preflight")
            count = len(list((folder / "trials").glob("attempt_*")))
            result = run_worker(folder / "trials" / f"attempt_{count + 1:05d}", protocol, whole if full else sampled, full, index, seed, epochs, seconds, ledger)
            if modern:
                final_budget.observe(result)
            return result
        def save(state):
            write_json(folder / "state.json", state)
        try:
            committed = False
            state = run_search(protocol, evaluate, remaining, ledger, initial, save, elapsed=lambda: ledger.elapsed)
            final_start = time.monotonic()
            if modern:
                ledger.emit("finalization_started", **final_budget.summary(), wall_remaining_seconds=wall_remaining())
            verify_live(folder, protocol, cache=cache, force=True)
            audit_run(folder, check_published=False)
            if modern:
                ledger.emit("finalization_verified", seconds=time.monotonic() - final_start, integrity=cache.stats())
            session_seconds = ledger.elapsed - start_elapsed
            state_hash = digest(state)
            state_path = "commits/" + state_hash + ".json"
            (folder / "commits").mkdir(exist_ok=True)
            write_json(folder / state_path, state)
            ledger.emit("selection_committed", session_seconds=session_seconds, original_budget_seconds=protocol["budget_seconds"],
                        this_execution_budget_seconds=budget, decision=state["decision"], state_sha256=state_hash, state_path=state_path)
            committed = True
            write_json(folder / "decision.locked.json", state["decision"])
            report = publish(folder, protocol, state)
            if modern:
                ledger.emit("publication_finished", session_seconds=ledger.elapsed - start_elapsed,
                            boundary="after primary report/CLI export; final timing receipt refresh excluded",
                            commit_session_seconds=session_seconds)
                # Refresh timing receipt silently. Timing never claims the
                # self-referential final receipt write is inside its own clock.
                report = publish(folder, protocol, state, display=False)
            return report
        except BaseException as exc:
            ledger.emit("report_export_failed" if committed else "session_interrupted", error=type(exc).__name__, session_seconds=ledger.elapsed - start_elapsed)
            write_json(folder / "status.json", {"phase": "selection_committed_report_error" if committed else "interrupted", "error": str(exc)})
            raise


def resume_run(path, seconds, origin=None):
    # Audit BEFORE skipping any already complete job, including no-op resumes.
    folder = Path(path).resolve()
    with exclusive_run(folder):
        reconcile_state(folder)
    folder, protocol, _, checked = audit_run(path)
    hw = runtime_info(protocol["hardware"]["device"])
    hw["host_label"] = platform.node()
    changes = hardware_changes(protocol["hardware"], hw)
    if changes:
        raise ValueError(f"Resume hardware/runtime changed/unverified: {changes}")
    verify_live(folder, protocol)
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    ledger = Ledger(folder / "events.jsonl", origin=origin)
    ledger.emit("resume_requested", audited_attempts=checked, additional_budget_seconds=seconds, downtime_charged=False)
    return execute(folder, protocol, ledger, state, seconds)


def reconcile_state(folder):
    """Recover ONLY journaled immutable completed outcomes missing from state.

    Unknown elapsed cost after hard parent/machine crash is not invented. Such
    sessions require investigation/new protocol rather than zero-cost replay.
    """
    events = read_events(folder / "events.jsonl")
    completed = {e["payload"]["attempt_id"]: e["payload"] for e in events if e["kind"] == "trial_completed"}
    started = {e["payload"]["attempt_id"] for e in events if e["kind"] == "trial_started"}
    if started - set(completed):
        raise ValueError("Unjournaled worker completion/cost after hard interruption; preserve evidence and investigate, no zero-cost replay")
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    state_path = folder / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
        "results": {str(i): [] for i in range(len(protocol["configs"]))}, "probes": [], "actions": [],
        "ordering": __import__("research_pipeline.v2.scheduler", fromlist=["diversified_order"]).diversified_order(protocol["configs"], protocol["search_seed"])}
    existing = {r["attempt_id"] for rows in state["results"].values() for r in rows} | {r["attempt_id"] for r in state["probes"]}
    missing = set(completed) - existing
    if not missing:
        return
    write_json(folder / ("state.before_reconcile." + digest(state)[:12] + ".json"), state)
    for attempt in sorted(missing, key=lambda a: next(e["sequence"] for e in events if e["kind"] == "trial_completed" and e["payload"]["attempt_id"] == a)):
        event = completed[attempt]
        trial = Path(event["directory"]).resolve()
        if not trial.is_relative_to(folder) or sha(trial / "parent_result.json") != event["parent_sha256"]:
            raise ValueError("Cannot reconcile altered/out-of-run parent outcome")
        result = json.loads((trial / "parent_result.json").read_text(encoding="utf-8"))
        if result["attempt_id"] != attempt:
            raise ValueError("Reconcile identity mismatch")
        if result["mode"] == "train":
            state["results"][str(result["index"])].append(result)
        else:
            state["probes"].append(result)
            if protocol["scheduler"].get("policy_version", 1) >= 2:
                # A crash before scheduler.save may have lost controller/probe
                # charge. Never replay that exploration at zero cost: exhaust
                # its quota conservatively, while permitting full confirmation.
                quota = protocol["budget_seconds"] * protocol["scheduler"]["probe_fraction"]
                state["probe_budget_spent_seconds"] = max(quota, state.get("probe_budget_spent_seconds", 0))
                state["probe_budget_recovery"] = "reconciled probe with unknown controller charge; exploration quota conservatively exhausted"
        state["actions"].append({"attempt_id": attempt, "reconciled": True, "status": result["status"]})
    write_json(state_path, state)
    # Full raw audit is performed immediately by resume before any worker.
