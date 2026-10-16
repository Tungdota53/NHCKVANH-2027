"""Frozen source/runtime/dataset/model and original-worker artifact checks."""

import hashlib
import json
from pathlib import Path
import platform
from importlib.metadata import version

from yolo_tuner.research import digest, verify_protocol, verify_dataset_files

ROOT = Path(__file__).resolve().parent.parent


def sha(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def sources():
    paths = [*sorted((ROOT / "research_pipeline").glob("*.py")), *sorted((ROOT / "yolo_tuner").glob("*.py")), ROOT / "train_config.py"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def runtime():
    return {**{name: version(name) for name in ("torch", "torchvision", "ultralytics", "Pillow", "psutil", "PyYAML")},
            "python": platform.python_version(), "os": platform.platform()}


def hardware_changes(recorded, current):
    changes = [key for key in ("device", "gpu_uuid", "gpu", "cpu", "torch_version", "torch_cuda_version", "driver_version", "host_label")
               if recorded.get(key) != current.get(key)]
    if recorded["device"] not in {"cpu", "mps"} or current["device"] not in {"cpu", "mps"}:
        if not recorded.get("gpu_identity_verified", bool(recorded.get("gpu_uuid"))) or not current.get("gpu_identity_verified", bool(current.get("gpu_uuid"))):
            changes.append("physical GPU identity unverified")
    return changes


def freeze_sources(folder, mapping):
    for name in mapping:
        destination = folder / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())


def verify_live(protocol, manifest, check_runtime=True, folder=None):
    verify_protocol(protocol)
    for name, expected in protocol["source_files"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Research source changed: {name}; preserve study and use its frozen snapshot")
    if check_runtime and runtime() != protocol["runtime_versions"]:
        raise ValueError("Research runtime changed; use a new protocol")
    if digest(manifest) != protocol["dataset_sha256"]:
        raise ValueError("Dataset manifest changed")
    verify_dataset_files(manifest)
    for model, expected in protocol["model_files"].items():
        if sha(model) != expected:
            raise ValueError(f"Model file changed: {model}")
    if folder is not None:
        for name, expected in protocol.get("data_files", {}).items():
            if sha(folder / name) != expected:
                raise ValueError(f"Exported training data changed: {name}")


def verify_raw(folder, protocol, decisions, reference):
    """Audit historical evidence against snapshots, not today's source version."""
    verify_protocol(protocol)
    manifest = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    if digest(manifest) != protocol["dataset_sha256"]:
        raise ValueError("Manifest differs from frozen protocol")
    if digest(decisions) != digest(json.loads((folder / "decisions.locked.json").read_text(encoding="utf-8"))):
        raise ValueError("Search decisions changed")
    for name, expected in protocol["source_files"].items():
        if sha(folder / "source_snapshot" / name) != expected:
            raise ValueError(f"Source snapshot changed: {name}")
    expected_ids = {f"c{i:02d}" for i in range(len(protocol["configs"]))}
    if len(reference) != len(expected_ids) or {r["config_id"] for r in reference} != expected_ids:
        raise ValueError("Missing/duplicate configuration rows")
    checked = 0
    results = [r for row in reference for r in row["runs"]]
    results += [o["result"] for d in decisions for o in d["history"]]
    pending = list(results)
    while pending:
        result = pending.pop()
        old = result.get("previous_attempts", [])
        results.extend(old)
        pending.extend(old)
    seen = set()
    for result in results:
        trial = Path(result["log"]).parent
        if not trial.resolve().is_relative_to(folder):
            raise ValueError("Worker artifact lies outside declared study")
        if str(trial) in seen:
            continue
        seen.add(str(trial))
        req = json.loads((trial / "request.json").read_text(encoding="utf-8"))
        raw = json.loads((trial / "result.json").read_text(encoding="utf-8"))
        if req["config"] not in protocol["configs"] or req["seed"] not in protocol["training_seeds"]:
            raise ValueError("Original request differs from declared grid/seed")
        if req["full_epochs"] != protocol["full_epochs"] or req.get("resume") is not None:
            raise ValueError("Original scheduler horizon/resume differs")
        mode = req["mode"]
        if mode not in {"train", "search"}:
            raise ValueError("Unknown worker mode")
        expected_manifest = folder / ("dataset.json" if mode == "train" else "subset.json")
        expected_data = folder / ("full_data" if mode == "train" else "subset_data") / "data.yaml"
        if Path(req["manifest"]).resolve() != expected_manifest or Path(req["data"]).resolve() != expected_data:
            raise ValueError("Actual data request differs from frozen split")
        if raw.get("config") != req["config"]:
            raise ValueError("Raw config differs from request")
        parent = json.loads((trial / "parent_result.json").read_text(encoding="utf-8"))
        for key, value in parent.items():
            if result.get(key) != value:
                raise ValueError(f"Parent worker outcome changed: {key}")
        for key in ("quality_score", "best_quality", "memory_mb", "epochs_completed", "actual", "checkpoint_sha256"):
            if raw.get(key) != result.get(key):
                raise ValueError(f"Raw result changed: {key}")
        if result.get("worker_wall_seconds", 0) < raw.get("measured_loop_seconds", 0):
            raise ValueError("Worker time smaller than measured train loop")
        if result.get("full_completed"):
            if req["mode"] != "train" or req["trial_epochs"] != protocol["full_epochs"] or raw["status"] != "ok":
                raise ValueError("Full-data verification uses incomplete/proxy request")
            if Path(req["manifest"]).resolve() != folder / "dataset.json" or raw["epochs_completed"] != protocol["full_epochs"]:
                raise ValueError("Full-data/horizon identity differs")
        if result["status"] == "ok":
            from .runner import actual_matches
            if raw.get("status") != "ok" or not actual_matches(req["config"], raw.get("actual", {})):
                raise ValueError("Successful result has mismatched actual parameters/status")
            if req["mode"] == "train" and sha(raw["best_checkpoint"]) != raw["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed after full-training")
            if protocol.get("require_optimizer_evidence"):
                from .runner import training_evidence_complete
                if not training_evidence_complete(raw):
                    raise ValueError("Successful training lacks optimizer/finite-loss evidence")
        checked += 1
    from yolo_tuner.research import measured_quality
    for decision in decisions:
        for observation in decision["history"]:
            result = observation["result"]
            request = json.loads((Path(result["log"]).parent / "request.json").read_text(encoding="utf-8"))
            if (request["seed"] != decision["seed"] or request["config"] != protocol["configs"][int(observation["config_id"][1:])]
                    or observation["epochs"] != request["trial_epochs"] or observation["full_data"] != (request["mode"] == "train")
                    or observation["quality"] != measured_quality(result) or observation["memory"] != result.get("memory_mb")
                    or observation["complete"] != (result["status"] == "ok" and result.get("epochs_completed") == observation["epochs"])):
                raise ValueError("Locked search observation differs from its worker outcome")
    return checked
