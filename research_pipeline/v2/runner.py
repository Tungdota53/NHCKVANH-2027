"""Budget-deadline subprocess runner; each attempt has immutable raw outcomes."""

from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import psutil
from yolo_tuner.pipeline import write_json, language_policy
from yolo_tuner.runner import stop_process_tree
from yolo_tuner.research import digest
from research_pipeline.integrity import sha
from .terminal import TrialProgress, wait_worker, say, duration, number


def run_worker(folder, protocol, data, full, index, seed, epochs, seconds, ledger):
    folder.mkdir(parents=True, exist_ok=True)
    config = protocol["configs"][index]
    task = protocol["task"]
    root = Path(protocol["run_directory"])
    whole = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    vocab, maximum = language_policy(whole) if task == "captioning" else ({}, 32)
    req = {"attempt_id": uuid.uuid4().hex, "task": task, "config": config,
           "protocol_sha256": protocol["protocol_sha256"], "seed": seed, "mode": "train" if full else "search",
           "full_epochs": protocol["full_epochs"], "trial_epochs": epochs, "resume": None,
           "manifest": str(root / ("dataset.json" if full else "subset.json")), "device": protocol["hardware"]["device"],
           "project": str(folder), "counts": data["counts"], "full_counts": whole["captions"] if task == "captioning" else whole["counts"],
           "data": str(root / ("full_data" if full else "subset_data") / "data.yaml"),
           "vocabulary": vocab, "max_tokens": maximum,
           "data_loader": protocol.get("execution_policy", {}).get("data_loader", {}),
           "threads": min(4, protocol["hardware"].get("logical_cpus") or 1),
           "logical_cpus": protocol["hardware"].get("logical_cpus") or 1,
           "full_counts_by_source": {s: dict(Counter({})) for s in ("train", "val")}}
    if task == "captioning":
        from yolo_tuner.vision_worker import tokens
        common = Counter(" ".join(tokens(c)) for r in whole["splits"]["train"] for c in r["captions"])
        req["caption_baseline"] = common.most_common(1)[0][0] if common else ""
    if task != "detection":
        for split in req["full_counts_by_source"]:
            counts = Counter()
            for row in whole["splits"][split]:
                counts[row["source"]] += len(row["captions"]) if task == "captioning" else 1
            req["full_counts_by_source"][split] = dict(counts)
    write_json(folder / "request.json", req)
    ledger.emit("trial_started", attempt_id=req["attempt_id"], index=index, seed=seed, full=full, epochs=epochs, timeout_seconds=seconds)
    progress = TrialProgress(folder, index, seed, epochs, full, config, seconds)
    start, timeout, interrupted = time.monotonic(), False, None
    env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    with (folder / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "research_pipeline.v2.worker", str(folder / "request.json"), str(folder / "result.json")],
                                   stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            created = psutil.Process(process.pid).create_time()
        except psutil.NoSuchProcess:
            created = None
        record = {"pid": process.pid, "create_time": created, "status": "running", "attempt_id": req["attempt_id"]}
        write_json(folder / "process.json", record)
        try:
            wait_worker(process, max(0.01, seconds - (time.monotonic() - start)), progress)
        except subprocess.TimeoutExpired:
            timeout = True
            stop_process_tree(process)
        except BaseException as exc:
            stop_process_tree(process)
            interrupted = exc
        finally:
            write_json(folder / "process.json", {**record, "status": "terminated" if timeout else "exited", "returncode": process.poll()})
    output = folder / "result.json"
    if output.exists():
        try:
            raw = json.loads(output.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeError):
            raw = {"status": "error", "error": "Worker result is corrupt; original bytes preserved"}
    else:
        raw = {"status": "error", "error": "Worker exited without result"}
        write_json(output, raw)
    result = {**raw, "config": config, "seed": seed, "mode": req["mode"], "index": index,
              "worker_wall_seconds": time.monotonic() - start, "trial_directory": str(folder),
              "attempt_id": req["attempt_id"], "request_sha256": sha(folder / "request.json"), "raw_sha256": sha(output)}
    if timeout:
        result.update(status="timeout", error="Worker wall deadline exceeded; not a full confirmation")
    elif interrupted:
        result.update(status="interrupted", error=type(interrupted).__name__)
    elif process.returncode != 0:
        result.update(status="error", error=f"Worker exit {process.returncode}")
    phases = result.get("telemetry", {}).get("phases", {})
    result["unattributed_startup_and_parent_seconds"] = max(0, result["worker_wall_seconds"] - sum(p["wall_seconds"] for p in phases.values()))
    write_json(folder / "parent_result.json", result)
    ledger.emit("trial_completed", attempt_id=req["attempt_id"], index=index, seed=seed, full=full,
                status=result["status"], worker_seconds=result["worker_wall_seconds"],
                parent_sha256=sha(folder / "parent_result.json"), directory=str(folder))
    say(f"  Kết thúc c{index:02d} seed={seed} {'FULL' if full else 'PROBE'}: {result['status']} | Q={number(result.get('quality'))} | T={duration(result['worker_wall_seconds'])} | M={number(result.get('memory_mb'), 1)} MiB")
    if result["status"] != "ok":
        say(f"  Lý do: {result.get('error', 'xem worker.log')} | {folder / 'worker.log'}")
    if interrupted:
        raise interrupted
    return result
