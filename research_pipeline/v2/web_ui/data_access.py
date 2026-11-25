"""Read-only data access, telemetry, audit/export wrappers, and safe job launcher."""

import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from urllib.parse import quote
import uuid

import psutil

from ..scheduler import METHODS
from ..schema import TASKS
from ..selection import RULES
from .security import resolve_contained_path, resolve_dataset_input

REPO_ROOT = Path(__file__).resolve().parents[3]
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")
_STATIC_TORCH_INFO = None
_STATIC_TORCH_LOCK = threading.Lock()
_JOBS_LOCK = threading.Lock()
_JOBS = {}


def _read_json(path: Path, *, optional: bool = True):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        if optional:
            return None
        raise


def _torch_static_info():
    global _STATIC_TORCH_INFO
    with _STATIC_TORCH_LOCK:
        if _STATIC_TORCH_INFO is None:
            info = {"torch_version": None, "cuda_available": False}
            try:
                import torch
                info["torch_version"] = str(torch.__version__)
                info["cuda_available"] = bool(torch.cuda.is_available())
            except Exception:
                pass
            _STATIC_TORCH_INFO = info
        return dict(_STATIC_TORCH_INFO)


def hardware_snapshot():
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(str(REPO_ROOT))
    cpu_pct = psutil.cpu_percent(interval=None)
    tinfo = _torch_static_info()
    gpus = []
    driver_version = None
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            drv = pynvml.nvmlSystemGetDriverVersion()
            driver_version = drv.decode("utf-8", "replace") if isinstance(drv, bytes) else str(drv)
        except Exception:
            driver_version = None
        count = int(pynvml.nvmlDeviceGetCount())
        for idx in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            gpu = {"index": idx, "driver_version": driver_version}
            try:
                name = pynvml.nvmlDeviceGetName(handle)
                gpu["name"] = name.decode("utf-8", "replace") if isinstance(name, bytes) else str(name)
            except Exception:
                gpu["name"] = f"GPU {idx}"
            try:
                uid = pynvml.nvmlDeviceGetUUID(handle)
                gpu["uuid"] = uid.decode("utf-8", "replace") if isinstance(uid, bytes) else str(uid)
            except Exception:
                gpu["uuid"] = None
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu["vram_total_mb"] = round(mem.total / (1024.0 ** 2), 2)
                gpu["vram_used_mb"] = round(mem.used / (1024.0 ** 2), 2)
                gpu["vram_free_mb"] = round(mem.free / (1024.0 ** 2), 2)
            except Exception:
                gpu["vram_total_mb"] = gpu["vram_used_mb"] = gpu["vram_free_mb"] = None
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu["gpu_utilization_percent"] = float(util.gpu)
                gpu["memory_utilization_percent"] = float(util.memory)
            except Exception:
                gpu["gpu_utilization_percent"] = gpu["memory_utilization_percent"] = None
            try:
                gpu["temperature_c"] = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                gpu["temperature_c"] = None
            try:
                gpu["power_watts"] = round(float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0, 2)
            except Exception:
                gpu["power_watts"] = None
            try:
                gpu["power_limit_watts"] = round(float(pynvml.nvmlDeviceGetPowerManagementLimit(handle)) / 1000.0, 2)
            except Exception:
                gpu["power_limit_watts"] = None
            gpus.append(gpu)
    except Exception:
        gpus = []
    return {
        "mode": "cuda" if (tinfo["cuda_available"] and gpus) else "cpu",
        "cuda_available": tinfo["cuda_available"],
        "torch_version": tinfo["torch_version"],
        "driver_version": driver_version,
        "cpu_percent": float(cpu_pct),
        "logical_cpus": int(psutil.cpu_count(logical=True) or 1),
        "ram_total_mb": round(vm.total / (1024.0 ** 2), 2),
        "ram_used_mb": round(vm.used / (1024.0 ** 2), 2),
        "ram_available_mb": round(vm.available / (1024.0 ** 2), 2),
        "ram_percent": float(vm.percent),
        "disk_total_mb": round(disk.total / (1024.0 ** 2), 2),
        "disk_free_mb": round(disk.free / (1024.0 ** 2), 2),
        "gpus": gpus,
    }


def _is_within_real(root_real: Path, path: Path) -> bool:
    try:
        return Path(os.path.realpath(str(path))).is_relative_to(root_real)
    except OSError:
        return False


def _contained_eval_dir(run_dir: Path, run_dir_real: Path, runs_root_real: Path | None, raw_out) -> Path | None:
    if not raw_out or not isinstance(raw_out, str):
        return None
    try:
        cand = resolve_contained_path(run_dir, raw_out, allow_absolute_inside=True, must_exist=True)
    except (PermissionError, FileNotFoundError, ValueError, OSError):
        return None
    if not _is_within_real(run_dir_real, cand):
        return None
    if runs_root_real is not None and not _is_within_real(runs_root_real, cand):
        return None
    if not cand.is_dir():
        return None
    return cand


def _load_test_eval(run_dir: Path, runs_root: Path | None = None):
    run_dir_resolved = Path(run_dir).resolve()
    run_dir_real = Path(os.path.realpath(str(run_dir_resolved)))
    runs_root_real = Path(os.path.realpath(str(Path(runs_root).resolve()))) if runs_root is not None else None

    history_path = run_dir_resolved / "test_history.json"
    history = (
        _read_json(history_path) or []
        if _is_within_real(run_dir_real, history_path)
        and (runs_root_real is None or _is_within_real(runs_root_real, history_path))
        else []
    )
    if not isinstance(history, list):
        history = []

    if history:
        last = history[-1]
        out_raw = last.get("output") if isinstance(last, dict) else None
        out_dir = _contained_eval_dir(run_dir_resolved, run_dir_real, runs_root_real, out_raw)
        if out_dir is not None:
            ev_file = out_dir / "evaluation.json"
            if _is_within_real(run_dir_real, ev_file) and (runs_root_real is None or _is_within_real(runs_root_real, ev_file)):
                ev = _read_json(ev_file)
                if isinstance(ev, dict):
                    return ev, history

    ev_summary_path = run_dir_resolved / "evaluation" / "test_summary.json"
    if _is_within_real(run_dir_real, ev_summary_path) and (runs_root_real is None or _is_within_real(runs_root_real, ev_summary_path)):
        ev_summary = _read_json(ev_summary_path)
        if isinstance(ev_summary, dict):
            return ev_summary, history

    eval_root = run_dir_resolved / "evaluation"
    if eval_root.is_dir() and not eval_root.is_symlink() and _is_within_real(run_dir_real, eval_root):
        subdirs = sorted(
            (
                p for p in eval_root.iterdir()
                if p.is_dir()
                and not p.is_symlink()
                and _is_within_real(run_dir_real, p)
                and (runs_root_real is None or _is_within_real(runs_root_real, p))
            ),
            reverse=True,
        )
        for sub in subdirs:
            ev_file = sub / "evaluation.json"
            if _is_within_real(run_dir_real, ev_file) and (runs_root_real is None or _is_within_real(runs_root_real, ev_file)):
                ev = _read_json(ev_file)
                if isinstance(ev, dict):
                    return ev, history
    return None, history


def scan_runs(runs_root):
    root = Path(runs_root).resolve()
    root_real = Path(os.path.realpath(str(root)))
    if not root.is_dir():
        return {"runs_dir": str(root), "runs": [], "benchmarks": [], "algorithm_studies": []}

    runs = []
    benchmarks = []
    studies = []
    skip_dirs = {"trials", "sources", "source_snapshot", "evaluation", "full_data", "subset_data", "commits", "__pycache__"}

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        cur = Path(dirpath)
        if not _is_within_real(root_real, cur):
            dirnames[:] = []
            continue
        dirnames[:] = [
            d for d in sorted(dirnames)
            if d not in skip_dirs and not (cur / d).is_symlink() and _is_within_real(root_real, cur / d)
        ]
        fset = set(filenames)
        rel = cur.relative_to(root).as_posix()
        if rel == ".":
            rel = ""

        if "ALGORITHM_STUDY.json" in fset:
            st = _read_json(cur / "ALGORITHM_STUDY.json") or {}
            studies.append({
                "rel_path": rel,
                "name": cur.name,
                "kind": "algorithm_study",
                "study_count": len(st.get("studies") or []),
                "sha256": st.get("sha256"),
                "mtime": (cur / "ALGORITHM_STUDY.json").stat().st_mtime,
            })

        if "BENCHMARK.json" in fset:
            bm = _read_json(cur / "BENCHMARK.json") or {}
            bp = _read_json(cur / "benchmark_protocol.json") or {}
            benchmarks.append({
                "rel_path": rel,
                "name": cur.name,
                "kind": "benchmark",
                "task": bp.get("task") or bm.get("task") or "?",
                "methods": [s.get("method") for s in (bm.get("summaries") or []) if isinstance(s, dict)],
                "selection_rule": bm.get("selection_rule") or bp.get("selection_rule"),
                "reference_complete": bm.get("reference_complete"),
                "outcome_count": len(bm.get("outcomes") or []),
                "mtime": (cur / "BENCHMARK.json").stat().st_mtime,
            })

        if "protocol.json" in fset:
            prot = _read_json(cur / "protocol.json")
            if not isinstance(prot, dict):
                continue
            adapter = prot.get("adapter")
            if adapter not in {"research_v2", "yolo_research"}:
                continue
            status = _read_json(cur / "status.json") or {}
            rep = _read_json(cur / "report.json") or {}
            rec = rep.get("recommendation") or {}
            dec = rep.get("decision") or status
            ev, _ = _load_test_eval(cur, runs_root=root)
            kind = "v2_deployment" if adapter == "research_v2" and prot.get("mode") == "deployment" else "v2" if adapter == "research_v2" else "research"
            runs.append({
                "rel_path": rel,
                "name": cur.name,
                "kind": kind,
                "mode": prot.get("mode", "online"),
                "task": prot.get("task", "?"),
                "method": prot.get("method", "?"),
                "selection_rule": prot.get("selection_rule", "weighted"),
                "detection_metric": prot.get("detection_metric"),
                "phase": status.get("phase", "prepared"),
                "verification_status": dec.get("verification_status"),
                "selected_config_id": dec.get("selected_config_id"),
                "best_quality": rec.get("validation_mean"),
                "utility": rec.get("utility"),
                "online_session_seconds": rep.get("online_session_seconds"),
                "budget_seconds": prot.get("budget_seconds"),
                "has_evaluation": ev is not None,
                "protocol_sha256": prot.get("protocol_sha256"),
                "mtime": (cur / "protocol.json").stat().st_mtime,
            })
        elif "training_report.json" in fset:
            tr = _read_json(cur / "training_report.json") or {}
            runs.append({
                "rel_path": rel,
                "name": cur.name,
                "kind": "legacy_training",
                "mode": "legacy",
                "task": tr.get("task") or (tr.get("dataset") or {}).get("task", "?"),
                "method": "legacy",
                "selection_rule": "weighted",
                "phase": "complete",
                "verification_status": None,
                "selected_config_id": None,
                "best_quality": None,
                "utility": None,
                "online_session_seconds": None,
                "budget_seconds": None,
                "has_evaluation": False,
                "protocol_sha256": None,
                "mtime": (cur / "training_report.json").stat().st_mtime,
            })

    runs.sort(key=lambda r: (r["mtime"], r["rel_path"]), reverse=True)
    benchmarks.sort(key=lambda r: (r["mtime"], r["rel_path"]), reverse=True)
    studies.sort(key=lambda r: (r["mtime"], r["rel_path"]), reverse=True)
    return {
        "runs_dir": str(root),
        "runs": runs,
        "benchmarks": benchmarks,
        "algorithm_studies": studies,
    }


def load_run_detail(runs_root, rel_path: str):
    root = Path(runs_root).resolve()
    run_dir = resolve_contained_path(root, rel_path, allow_absolute_inside=False, must_exist=True)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {rel_path}")
    rel = run_dir.relative_to(root).as_posix()

    protocol = _read_json(run_dir / "protocol.json")
    status = _read_json(run_dir / "status.json") or {}
    report = _read_json(run_dir / "report.json")
    state = _read_json(run_dir / "state.json") or {}
    if report is None and isinstance(protocol, dict) and protocol.get("adapter") == "research_v2" and state:
        try:
            from ..ledger import read_events
            from ..report import make_report
            events = read_events(run_dir / "events.jsonl") if (run_dir / "events.jsonl").is_file() else []
            report = make_report(protocol, state, events)
        except Exception:
            report = None

    anytime = None
    hv3d = None
    loocv = None
    if isinstance(protocol, dict) and protocol.get("adapter") == "research_v2" and protocol.get("mode") != "deployment":
        try:
            from ..diagnostics import anytime_trajectory
            anytime = anytime_trajectory(run_dir, protocol)
        except Exception:
            anytime = None
        if isinstance(report, dict) and isinstance(report.get("candidates"), list):
            try:
                from yolo_tuner.research import Objective
                from ..paper_export import hypervolume_3d, surrogate_loocv
                hv3d = hypervolume_3d(report["candidates"], Objective(**protocol["objective"]))
                loocv = surrogate_loocv(protocol, report["candidates"])
            except Exception:
                pass

    evaluation, test_history = _load_test_eval(run_dir, runs_root=root)

    artifacts = []
    for rel_file in (
        "REPORT.md", "report.json", "protocol.json", "status.json",
        "pareto.json", "presets.json", "selected.json",
        "evaluation/test_summary.json",
    ):
        if (run_dir / rel_file).is_file():
            artifacts.append(f"{rel}/{rel_file}" if rel else rel_file)

    return {
        "rel_path": rel,
        "protocol": protocol,
        "status": status,
        "report": report,
        "state": {
            "results": state.get("results", {}),
            "probes": state.get("probes", []),
            "acquisition_history": state.get("acquisition_history", []),
            "cost_predictions": state.get("cost_predictions", {}),
            "memory_risk_deferred": state.get("memory_risk_deferred", []),
            "decision": state.get("decision"),
        } if state else None,
        "anytime": anytime,
        "hypervolume_3d": hv3d,
        "surrogate_loocv": loocv,
        "evaluation": evaluation,
        "test_history": test_history,
        "artifacts": artifacts,
    }


def load_evaluation(runs_root, rel_path: str):
    root = Path(runs_root).resolve()
    run_dir = resolve_contained_path(root, rel_path, allow_absolute_inside=False, must_exist=True)
    evaluation, history = _load_test_eval(run_dir, runs_root=root)
    return {
        "run": rel_path,
        "available": evaluation is not None,
        "evaluation": evaluation,
        "test_history": history,
    }


def _snapshot_jobs():
    with _JOBS_LOCK:
        items = []
        for jid, job in list(_JOBS.items()):
            proc = job.get("process")
            rc = proc.poll() if proc is not None else job.get("returncode")
            job["returncode"] = rc
            items.append({
                "job_id": jid,
                "mode": job["mode"],
                "pid": job.get("pid"),
                "alive": rc is None,
                "returncode": rc,
                "output": job.get("output"),
                "cli_preview": job.get("cli_preview"),
                "started_at": job.get("started_at"),
            })
        items.sort(key=lambda x: x["started_at"] or 0.0, reverse=True)
        return items


def load_live_status(runs_root, rel_path: str | None = None):
    jobs = _snapshot_jobs()
    payload = {"jobs": jobs, "run": None}
    if not rel_path:
        return payload

    run_dir = resolve_contained_path(runs_root, rel_path, allow_absolute_inside=False, must_exist=True)
    protocol = _read_json(run_dir / "protocol.json") or {}
    status = _read_json(run_dir / "status.json") or {}
    state = _read_json(run_dir / "state.json") or {}

    live_workers = []
    latest_trial = None
    latest_log_lines = []
    trials_dir = run_dir / "trials"
    if trials_dir.is_dir():
        attempt_dirs = sorted((p for p in trials_dir.iterdir() if p.is_dir()), reverse=True)
        for tdir in attempt_dirs:
            prec = _read_json(tdir / "process.json") or {}
            res = _read_json(tdir / "result.json") or {}
            req = _read_json(tdir / "request.json") or {}
            if latest_trial is None:
                latest_trial = {
                    "attempt_id": tdir.name,
                    "index": req.get("index", res.get("index")),
                    "seed": req.get("seed", res.get("seed")),
                    "mode": req.get("mode", res.get("mode")),
                    "epochs_completed": res.get("epochs_completed", 0),
                    "trial_epochs": req.get("trial_epochs", protocol.get("full_epochs")),
                    "quality": res.get("quality_score") if res.get("quality_score") is not None else res.get("quality"),
                    "memory_mb": res.get("memory_mb") or res.get("peak_reserved_mb") or res.get("peak_rss_mb"),
                    "status": prec.get("status") or res.get("status"),
                }
                log_path = tdir / "worker.log"
                if log_path.is_file():
                    try:
                        raw_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-120:]
                        latest_log_lines = [_ANSI_RE.sub("", ln) for ln in raw_lines]
                    except OSError:
                        latest_log_lines = []
            if prec.get("status") == "running" and prec.get("pid"):
                try:
                    proc = psutil.Process(int(prec["pid"]))
                    if (prec.get("create_time") is not None
                            and abs(proc.create_time() - prec["create_time"]) < 0.01
                            and proc.is_running()
                            and proc.status() != psutil.STATUS_ZOMBIE):
                        live_workers.append({
                            "pid": int(prec["pid"]),
                            "attempt_id": tdir.name,
                            "epochs_completed": res.get("epochs_completed", 0),
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, TypeError):
                    pass

    elapsed_s = None
    if (run_dir / "events.jsonl").is_file():
        try:
            from ..ledger import read_events
            evs = read_events(run_dir / "events.jsonl")
            if evs:
                elapsed_s = evs[-1].get("active_seconds")
        except Exception:
            elapsed_s = None

    budget_s = protocol.get("budget_seconds")
    rem_s = max(0.0, float(budget_s) - float(elapsed_s)) if isinstance(budget_s, (int, float)) and isinstance(elapsed_s, (int, float)) else None

    candidates_summary = []
    configs = protocol.get("configs") or []
    results = state.get("results") or {}
    for idx, cfg in enumerate(configs):
        runs_for_idx = results.get(str(idx), [])
        best_q = None
        last_status = "pending"
        last_curve = []
        for r in runs_for_idx:
            last_status = r.get("status", last_status)
            q = r.get("quality")
            if isinstance(q, (int, float)) and math.isfinite(q):
                best_q = q if best_q is None else max(best_q, q)
            if r.get("learning_curve"):
                last_curve = r["learning_curve"]
        candidates_summary.append({
            "index": idx,
            "model": Path(str(cfg.get("model", "?"))).name,
            "imgsz": cfg.get("imgsz"),
            "batch": cfg.get("batch"),
            "lr0": cfg.get("lr0"),
            "attempts": len(runs_for_idx),
            "best_quality": best_q,
            "status": last_status,
            "learning_curve": last_curve,
        })

    payload["run"] = {
        "rel_path": rel_path,
        "phase": status.get("phase"),
        "task": protocol.get("task"),
        "method": protocol.get("method"),
        "budget_seconds": budget_s,
        "elapsed_seconds": elapsed_s,
        "remaining_seconds": rem_s,
        "live_workers": live_workers,
        "latest_trial": latest_trial,
        "worker_log": latest_log_lines,
        "candidates": candidates_summary,
    }
    return payload


def load_benchmark_detail(runs_root, rel_path: str):
    target = resolve_contained_path(runs_root, rel_path, allow_absolute_inside=False, must_exist=True)
    if (target / "BENCHMARK.json").is_file():
        bm = _read_json(target / "BENCHMARK.json", optional=False)
        bp = _read_json(target / "benchmark_protocol.json")
        return {"kind": "benchmark", "rel_path": rel_path, "benchmark": bm, "benchmark_protocol": bp}
    if (target / "ALGORITHM_STUDY.json").is_file():
        study = _read_json(target / "ALGORITHM_STUDY.json", optional=False)
        return {"kind": "algorithm_study", "rel_path": rel_path, "algorithm_study": study}
    raise FileNotFoundError(f"No BENCHMARK.json or ALGORITHM_STUDY.json found in {rel_path}")


def list_workspace_resources():
    from ..wizard import scan_datasets, scan_weights
    datasets = scan_datasets(REPO_ROOT / "runs" / "datasets")
    weights = scan_weights(REPO_ROOT / "weights")
    configs_dir = REPO_ROOT / "configs"
    configs = []
    if configs_dir.is_dir():
        for p in sorted(configs_dir.glob("*.json"), key=lambda x: x.name.lower()):
            configs.append({"name": p.name, "path": str(p.relative_to(REPO_ROOT).as_posix())})
    return {
        "datasets": datasets,
        "weights": weights,
        "configs": configs,
        "methods": list(METHODS),
        "rules": list(RULES),
        "tasks": sorted(TASKS),
    }


def run_inspect_action(data_path: str, task: str = "auto", device: str = "cpu"):
    ds_path = resolve_dataset_input(data_path)
    if task not in {"auto", *TASKS}:
        raise ValueError(f"Invalid task: {task}")
    from ..data import audit_dataset, discover
    from yolo_tuner.pipeline import runtime_info
    audit = audit_dataset(str(ds_path), task)
    try:
        data = discover(str(ds_path), 101, task)
    except ValueError as exc:
        data = {
            "task": audit["task"],
            "source": audit["source"],
            "counts": audit["counts"],
            "captions": {},
            "validation_ready": False,
            "warnings": [str(exc)],
        }
    hw = runtime_info(device)
    return {
        "dataset": {
            "task": data.get("task"),
            "source": data.get("source"),
            "counts": data.get("counts"),
            "captions": data.get("captions"),
            "validation_ready": data.get("validation_ready"),
            "warnings": data.get("warnings"),
            "audit": audit,
        },
        "hardware": hw,
    }


def run_audit_action(runs_root, rel_path: str):
    target = resolve_contained_path(runs_root, rel_path, allow_absolute_inside=False, must_exist=True)
    if (target / "protocol.json").is_file():
        from ..workflow import audit_run
        folder, protocol, candidates, count = audit_run(target)
        checks = [
            {"name": "source_snapshot_sha256", "passed": True, "detail": f"{len(protocol.get('source_files', {}))} source files verified"},
            {"name": "dataset_and_subset_sha256", "passed": True, "detail": protocol.get("dataset_sha256", "")[:16]},
            {"name": "events_ledger_hash_chain", "passed": True, "detail": f"{count} attempt(s) linked"},
            {"name": "request_result_checkpoint_sha256", "passed": True, "detail": "Raw worker evidence intact"},
            {"name": "committed_state_and_decision_parity", "passed": True, "detail": "Rebuilt decision/report matches locked files"},
        ]
        return {
            "ok": True,
            "kind": "v2_run",
            "run": rel_path,
            "folder": str(folder),
            "checked_attempts": count,
            "candidate_count": len(candidates),
            "protocol_sha256": protocol.get("protocol_sha256"),
            "checks": checks,
        }
    if (target / "BENCHMARK.json").is_file():
        bm = _read_json(target / "BENCHMARK.json", optional=False)
        from ..workflow import audit_run
        root = Path(runs_root).resolve()
        root_real = Path(os.path.realpath(str(root)))
        checked_runs = 0
        containment_failures = []
        contained_dirs = []
        for row in bm.get("outcomes") or []:
            rpath = row.get("run") if isinstance(row, dict) else None
            if not rpath:
                continue
            try:
                contained = resolve_contained_path(root, str(rpath), allow_absolute_inside=True, must_exist=False)
            except PermissionError as exc:
                containment_failures.append(f"{rpath}: {exc}")
                continue
            if not _is_within_real(root_real, contained):
                containment_failures.append(f"{rpath}: escapes runs_root")
                continue
            if contained.is_dir():
                contained_dirs.append(contained)
        if containment_failures:
            return {
                "ok": False,
                "kind": "benchmark",
                "run": rel_path,
                "checked_attempts": 0,
                "checks": [
                    {"name": "containment", "passed": False, "detail": "; ".join(containment_failures)},
                ],
            }
        for contained in contained_dirs:
            audit_run(contained)
            checked_runs += 1
        return {
            "ok": True,
            "kind": "benchmark",
            "run": rel_path,
            "checked_attempts": checked_runs,
            "checks": [
                {"name": "containment", "passed": True, "detail": f"{len(contained_dirs)} child path(s) contained within runs_root"},
                {"name": "benchmark_outcomes_audit", "passed": True, "detail": f"{checked_runs} child runs audited"},
            ],
        }
    if (target / "ALGORITHM_STUDY.json").is_file():
        from yolo_tuner.research import digest
        study = _read_json(target / "ALGORITHM_STUDY.json", optional=False)
        expected = study.get("sha256")
        actual = digest({k: v for k, v in study.items() if k != "sha256"})
        if expected != actual:
            raise ValueError("ALGORITHM_STUDY.json checksum mismatch")
        return {
            "ok": True,
            "kind": "algorithm_study",
            "run": rel_path,
            "checked_attempts": len(study.get("studies") or []),
            "sha256": actual,
            "checks": [{"name": "algorithm_study_sha256", "passed": True, "detail": actual[:16]}],
        }
    raise FileNotFoundError(f"No auditable v2 run, benchmark, or algorithm study at {rel_path}")


def run_paper_export_action(runs_root, run_rel: str, output_path: str, lang: str = "vi"):
    if not isinstance(run_rel, str):
        raise ValueError("run must be a string")
    if not isinstance(output_path, str):
        raise ValueError("output must be a string")
    if not isinstance(lang, str) or lang not in {"vi", "en"}:
        raise ValueError("lang must be 'vi' or 'en'")
    root = Path(runs_root).resolve()
    run_dir = resolve_contained_path(root, run_rel, allow_absolute_inside=False, must_exist=True)
    out_dir = resolve_contained_path(root, output_path, allow_absolute_inside=True, must_exist=False)
    from ..paper_export import export_paper_artifacts
    summary = export_paper_artifacts(run_dir, out_dir, lang=lang)
    out_rel = out_dir.relative_to(root).as_posix()
    exported_files = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            rel_f = p.relative_to(root).as_posix()
            exported_files.append({
                "name": p.relative_to(out_dir).as_posix(),
                "rel_path": rel_f,
                "url": f"/api/file?path={quote(rel_f, safe='/')}",
            })
    provenance = _read_json(out_dir / "PROVENANCE.json")
    return {
        "ok": True,
        "output_rel": out_rel,
        "summary": summary,
        "provenance": provenance,
        "files": exported_files,
    }


def _parse_distinct_int_list(raw_val, field_name: str, default: list[int]) -> list[int]:
    if raw_val is None:
        return list(default)
    if isinstance(raw_val, bool):
        raise ValueError(f"{field_name} cannot be a boolean")
    if isinstance(raw_val, str):
        tokens = [s.strip() for s in raw_val.replace(",", " ").split() if s.strip()]
        if not tokens:
            raise ValueError(f"{field_name} must be non-empty")
        parsed = []
        for tok in tokens:
            if not re.fullmatch(r"-?[0-9]+", tok):
                raise ValueError(f"{field_name} must contain integers, got {tok!r}")
            parsed.append(int(tok))
    elif isinstance(raw_val, (list, tuple)):
        if not raw_val:
            raise ValueError(f"{field_name} must be non-empty")
        parsed = []
        for item in raw_val:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(f"{field_name} must contain integers, got {item!r}")
            parsed.append(int(item))
    else:
        raise ValueError(f"Invalid {field_name}: {raw_val!r}")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{field_name} must contain distinct integers")
    return parsed


def launch_job_action(runs_root, payload: dict):
    if not isinstance(payload, dict):
        raise ValueError("Invalid JSON payload")
    if payload.get("confirmed") is not True:
        raise ValueError("Explicit confirmation (confirmed: true) is required before launching")
    raw_mode = payload.get("mode")
    mode = "optimize" if raw_mode is None or raw_mode == "" else raw_mode
    if not isinstance(mode, str) or mode not in {"optimize", "plan-search", "benchmark"}:
        raise ValueError(f"Unsupported launch mode: {raw_mode!r}")

    raw_data = payload.get("data")
    if not isinstance(raw_data, str) or not raw_data.strip():
        raise ValueError("Dataset path must be a non-empty string")
    if raw_data.strip().startswith("-"):
        raise ValueError("Dataset path cannot start with '-'")
    data_path = str(resolve_dataset_input(raw_data))

    raw_output = payload.get("output")
    if raw_output is None or raw_output == "":
        output_raw = f"runs/{mode}_ui"
    elif not isinstance(raw_output, str):
        raise ValueError(f"Invalid output: {raw_output!r}")
    else:
        output_raw = raw_output.strip() or f"runs/{mode}_ui"
    if output_raw.startswith("-"):
        raise ValueError("Output path cannot start with '-'")
    root = Path(runs_root).resolve()
    # Allow relative paths like "runs/ui_exp" when runs_root is workspace runs/ or arbitrary temp runs_dir.
    if not Path(output_raw).is_absolute() and output_raw.startswith("runs/") and root.name != "runs":
        output_raw = output_raw[5:]
    out_dir = resolve_contained_path(root, output_raw, allow_absolute_inside=True, must_exist=False)

    task = payload.get("task") if "task" in payload and payload.get("task") is not None else "auto"
    if not isinstance(task, str) or task not in ({"auto"} | TASKS):
        raise ValueError(f"Invalid task: {task!r}")

    raw_device = payload.get("device") if "device" in payload else None
    if raw_device is None or raw_device == "":
        device = "auto"
    elif not isinstance(raw_device, str):
        raise ValueError(f"Invalid device: {raw_device!r}")
    else:
        device = raw_device.strip()
    if not device or device.startswith("-") or not re.fullmatch(r"(?:auto|cpu|mps|(?:cuda:)?[0-9]+)", device):
        raise ValueError(f"Invalid device: {raw_device!r}")

    raw_space = payload.get("space")
    space = "core" if raw_space is None or raw_space == "" else raw_space
    if not isinstance(space, str) or space not in {"compact", "core", "research"}:
        raise ValueError(f"Invalid space: {raw_space!r}")
    raw_method = payload.get("method")
    method = "adaptive_full" if raw_method is None or raw_method == "" else raw_method
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError(f"Invalid method: {raw_method!r}")
    raw_rule = payload.get("selection_rule")
    rule = "weighted" if raw_rule is None or raw_rule == "" else raw_rule
    if not isinstance(rule, str) or rule not in RULES:
        raise ValueError(f"Invalid selection_rule: {raw_rule!r}")
    raw_det_metric = payload.get("detection_metric")
    det_metric = "map50" if raw_det_metric is None or raw_det_metric == "" else raw_det_metric
    if not isinstance(det_metric, str) or det_metric not in {"map50", "map50_95"}:
        raise ValueError(f"Invalid detection_metric: {raw_det_metric!r}")

    raw_epochs = payload["full_epochs"] if "full_epochs" in payload and payload["full_epochs"] is not None else 10
    if isinstance(raw_epochs, bool):
        raise ValueError("full_epochs cannot be a boolean")
    if isinstance(raw_epochs, float) and not raw_epochs.is_integer():
        raise ValueError("full_epochs must be an integer")
    try:
        full_epochs = int(raw_epochs.strip()) if isinstance(raw_epochs, str) else int(raw_epochs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid full_epochs: {raw_epochs!r}") from exc
    if full_epochs <= 0:
        raise ValueError("full_epochs must be > 0")

    raw_budget = payload["budget_seconds"] if "budget_seconds" in payload and payload["budget_seconds"] is not None else 300.0
    if isinstance(raw_budget, bool):
        raise ValueError("budget_seconds cannot be a boolean")
    try:
        budget_seconds = float(raw_budget)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid budget_seconds: {raw_budget!r}") from exc
    if not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError("budget_seconds must be finite and > 0")

    seeds = _parse_distinct_int_list(
        payload["seeds"] if "seeds" in payload else None,
        "seeds",
        [17, 29, 43],
    )

    raw_weights = payload["weights"] if "weights" in payload and payload["weights"] is not None else [0.6, 0.25, 0.15]
    if isinstance(raw_weights, bool):
        raise ValueError("weights cannot be a boolean")
    if isinstance(raw_weights, str):
        tokens = [w.strip() for w in raw_weights.replace(",", " ").split() if w.strip()]
        try:
            weights = [float(w) for w in tokens]
        except ValueError as exc:
            raise ValueError(f"Invalid weights: {raw_weights!r}") from exc
    elif isinstance(raw_weights, (list, tuple)):
        weights = []
        for w in raw_weights:
            if isinstance(w, bool) or not isinstance(w, (int, float)):
                raise ValueError(f"weights entries must be numbers, got {w!r}")
            weights.append(float(w))
    else:
        raise ValueError(f"Invalid weights: {raw_weights!r}")
    if len(weights) != 3 or any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("weights must contain 3 finite non-negative numbers with positive sum")

    cli_args = [
        mode,
        "--data", data_path,
        "--task", task,
        "--device", device,
        "--space", space,
        "--full-epochs", str(full_epochs),
        "--budget-seconds", str(budget_seconds),
        "--seeds", *[str(int(s)) for s in seeds],
        "--selection-rule", rule,
        "--weights", *[str(float(w)) for w in weights],
        "--detection-metric", det_metric,
        "--output", str(out_dir),
    ]
    if mode == "benchmark":
        raw_methods = payload["methods"] if "methods" in payload and payload["methods"] is not None else ["adaptive_full", "adaptive_ehvi", "random"]
        if isinstance(raw_methods, bool):
            raise ValueError("methods cannot be a boolean")
        if isinstance(raw_methods, str):
            methods = [m.strip() for m in raw_methods.replace(",", " ").split() if m.strip()]
        elif isinstance(raw_methods, (list, tuple)):
            methods = []
            for m in raw_methods:
                if not isinstance(m, str) or not m.strip():
                    raise ValueError(f"Invalid benchmark method: {m!r}")
                methods.append(m.strip())
        else:
            raise ValueError(f"Invalid methods: {raw_methods!r}")
        if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
            raise ValueError("methods must be non-empty, distinct, and valid METHODS entries")

        search_seeds = _parse_distinct_int_list(
            payload["search_seeds"] if "search_seeds" in payload else None,
            "search_seeds",
            [17, 29, 43],
        )

        raw_ref = payload["reference_budget_seconds"] if "reference_budget_seconds" in payload and payload["reference_budget_seconds"] is not None else max(budget_seconds * 2, 600.0)
        if isinstance(raw_ref, bool):
            raise ValueError("reference_budget_seconds cannot be a boolean")
        try:
            ref_budget = float(raw_ref)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid reference_budget_seconds: {raw_ref!r}") from exc
        if not math.isfinite(ref_budget) or ref_budget <= 0:
            raise ValueError("reference_budget_seconds must be finite and > 0")

        cli_args += ["--methods", *methods, "--search-seeds", *[str(int(s)) for s in search_seeds], "--reference-budget-seconds", str(ref_budget)]
    else:
        cli_args += ["--method", method]

    raw_models = payload.get("models")
    if raw_models is not None and raw_models != [] and raw_models != ():
        if isinstance(raw_models, bool):
            raise ValueError("models cannot be a boolean")
        models: list[str] = []
        if isinstance(raw_models, str):
            if raw_models.strip():
                models = [m.strip() for m in raw_models.replace(";", ",").split(",") if m.strip()]
                if not models:
                    raise ValueError("models must contain at least one non-empty model entry when provided")
        elif isinstance(raw_models, (list, tuple)):
            for m in raw_models:
                if not isinstance(m, str) or not m.strip():
                    raise ValueError(f"Invalid model entry: {m!r}")
                models.append(m.strip())
        else:
            raise ValueError(f"Invalid models: {raw_models!r}")
        if any(m.startswith("-") for m in models):
            raise ValueError("Model entries cannot start with '-'")
        if models:
            cli_args += ["--models", *models]
    raw_grid = payload.get("grid")
    if raw_grid is not None and raw_grid != "":
        if not isinstance(raw_grid, str):
            raise ValueError(f"Invalid grid: {raw_grid!r}")
        grid_str = raw_grid.strip()
        if grid_str.startswith("-"):
            raise ValueError("Grid path cannot start with '-'")
        if grid_str:
            cli_args += ["--grid", str(resolve_dataset_input(grid_str))]

    cmd = [sys.executable, "-m", "research_pipeline.v2", *cli_args]
    preview = " ".join(cmd)
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "mode": mode,
            "pid": proc.pid,
            "process": proc,
            "returncode": None,
            "output": str(out_dir),
            "cli_preview": preview,
            "started_at": time.time(),
        }
    return {
        "launched": True,
        "job_id": job_id,
        "pid": proc.pid,
        "output": str(out_dir),
        "cli_preview": preview,
    }


def _terminate_pid_safely(pid: int, expected_create_time: float | None = None) -> bool:
    try:
        proc = psutil.Process(int(pid))
        if expected_create_time is not None and abs(proc.create_time() - expected_create_time) >= 0.05:
            return False
        cmdline = " ".join(proc.cmdline() or [])
        if "research_pipeline" not in cmdline:
            return False
        from yolo_tuner.runner import stop_process_tree
        stop_process_tree(proc)
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, TypeError):
        return False


def stop_job_action(runs_root, payload: dict | None = None):
    payload = payload or {}
    job_id = payload.get("job_id")
    run_rel = payload.get("run")

    if job_id:
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
        if job and job.get("process") is not None and job["process"].poll() is None:
            from yolo_tuner.runner import stop_process_tree
            stop_process_tree(job["process"])
            job["returncode"] = job["process"].poll()
            return {"stopped": True, "status": "stopped_job", "job_id": job_id, "pid": job.get("pid")}

    if run_rel:
        run_dir = resolve_contained_path(runs_root, str(run_rel), allow_absolute_inside=False, must_exist=True)
        stopped_pids = []
        trials_dir = run_dir / "trials"
        if trials_dir.is_dir():
            for pfile in trials_dir.glob("*/process.json"):
                rec = _read_json(pfile) or {}
                if rec.get("status") == "running" and rec.get("pid"):
                    if _terminate_pid_safely(rec["pid"], rec.get("create_time")):
                        stopped_pids.append(int(rec["pid"]))
        lock_file = run_dir / "run.lock"
        if lock_file.is_file():
            lrec = _read_json(lock_file) or {}
            if lrec.get("pid") and _terminate_pid_safely(lrec["pid"], lrec.get("create_time")):
                stopped_pids.append(int(lrec["pid"]))
        if stopped_pids:
            return {"stopped": True, "status": "stopped_run_processes", "pids": stopped_pids}

    return {"stopped": False, "status": "no_active_process"}
