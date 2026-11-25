"""Shared opaque-box E2E fixture builders, CLI runners, and R1 baseline constants."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw

from research_pipeline.integrity import freeze_sources, runtime, sha
from research_pipeline.v2.ledger import Ledger
from research_pipeline.v2.report import make_report, publish
from research_pipeline.v2.schema import lock_protocol
from research_pipeline.v2.workflow import source_map
from yolo_tuner.pipeline import write_json
from yolo_tuner.research import digest


REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

ALGORITHM_010_EXPECTED_SHA256 = "d246015b106c70a1e0cb6a509f8b065c6bb4758debe0b65f89f86168e3b255f5"
ALGORITHM_010_MANIFEST_FILE_SHA256 = "8c60b2f0cbb9c6815e6691c128cc35e76acc9177d33e5b958ae47a47a0adeee8"

YOLO_TUNER_SHA256 = {
    "yolo_tuner/__init__.py": "f6480181b410ae7419022ced63bf80fa90137f233ebcf01e098c1ecdc05fa137",
    "yolo_tuner/__main__.py": "3129adff95e0f38b87ab34cddfc27082c7892bcaec267ff5d40f160fdf55b44a",
    "yolo_tuner/adapters.py": "3558b1fb025b31829654f3b01b2e6708b138803b884a11f15ef3cd195b0adb55",
    "yolo_tuner/caption_features.py": "db5e3c1a154676a8b6d0f2b06d3884dc49293608628ea9af1dbdccbe67365902",
    "yolo_tuner/caption_worker.py": "cf450f5a92d0b7e9cf1bf45cb61e98b1528611bb028a7d914583e7f24bc9c389",
    "yolo_tuner/captioning.py": "4c97e7c80a3307aa623678bb1bf51d280e6d7193d9ca111cb3192805143d373d",
    "yolo_tuner/cli.py": "bf724a353ae02ea5bd8e9bc934c7b47983fb328d509b536a125f97f89b801cb9",
    "yolo_tuner/cli_ui.py": "98658b2446fd3d8ccd03e4c3a89b39a289180775ff91a6a1d3d915baaf9088a6",
    "yolo_tuner/completion.py": "5ab85a243602b6fb30db952f3c6e18781fdaf049b0bb30eae8ca83e9194b7934",
    "yolo_tuner/core.py": "98b40891bc4a98af4c880f4433560bae74f3277e3f4b6d4cf51192579ea902a0",
    "yolo_tuner/dataset.py": "6331fdb4c7f2cfe700766ba4c06f7535345a8a7e958401df0776107f8102ad25",
    "yolo_tuner/offline.py": "c52e55f8f19a972c121800afa503f87f8e7a14e485d5d89ab9c52a062a61b984",
    "yolo_tuner/pipeline.py": "5259fe9ccee116eec9e1523337b5d3113775efe66e41e5a073435a73a664e80c",
    "yolo_tuner/research.py": "038fd9df2bb68eb4d9b3d6dfbee0e05089ed519d3374ac0dabc35b5072435c17",
    "yolo_tuner/runner.py": "08187113ee87bc8dbaa4d53061646e118f289e06c67a6abc0362ac38f54c9bd0",
    "yolo_tuner/verification.py": "5e6ae75eabdc84d778738942ffdb265a92433eaab455283726531c8d0744ec95",
    "yolo_tuner/vision_models.py": "e48e0e79510bb211e31a628d32e128ad05843c566130719c48fd14c778d1a6e4",
    "yolo_tuner/vision_worker.py": "fdc33ee3c3cfeb59847ae858517c7545084cabe7b50f60f858c670ea81eaafc5",
    "yolo_tuner/worker.py": "bec86bfd46ec4ff67c86da9e0b43ac1f75dbb32519ce3b61140d5d5b115a8a07",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(65536):
            h.update(chunk)
    return h.hexdigest()


def assert_zero_ansi(text: str) -> None:
    match = ANSI_ESCAPE_RE.search(text or "")
    if match is not None:
        raise AssertionError(f"Unexpected ANSI escape sequence found: {match.group(0)!r}")


def run_cli(
    args: list[str],
    *,
    timeout: int = 120,
    entry: str = "module",
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run CLI in an isolated offline UTF-8 subprocess using sys.executable."""
    if entry == "script":
        cmd = [sys.executable, str(REPO_ROOT / "train_config.py"), *args]
    else:
        cmd = [sys.executable, "-m", "research_pipeline.v2", *args]
    env = dict(os.environ)
    env["YOLO_AUTOINSTALL"] = "false"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _write_unique_rgb_png(path: Path, size: int, idx: int, cls_id: int) -> None:
    """Create a deterministic unique-pixel PNG so partition() never drops splits."""
    bg = ((17 + idx * 29) % 240, (31 + idx * 43) % 240, (53 + idx * 61) % 240)
    im = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(im)
    box_color = (220, 40, 40) if cls_id == 0 else (40, 220, 40)
    pad = max(2, size // 4)
    draw.rectangle([pad, pad, size - pad, size - pad], fill=box_color)
    # Deterministic unique pixel marker encoding idx
    px = idx % size
    py = (idx * 7) % size
    im.putpixel((px, py), ((idx * 19 + 5) % 255, (idx * 37 + 11) % 255, (idx * 71 + 13) % 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="PNG")


def make_mini_yolo_dataset(
    root: Path,
    n_train: int = 4,
    n_val: int = 2,
    n_test: int = 2,
    size: int = 64,
) -> Path:
    """Build a valid offline miniature YOLO detection dataset with unique-pixel PNGs."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    global_idx = 0
    for split, count in (("train", n_train), ("val", n_val), ("test", n_test)):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            cls_id = global_idx % 2
            stem = f"{split}_{i:03d}"
            _write_unique_rgb_png(img_dir / f"{stem}.png", size, global_idx, cls_id)
            (lbl_dir / f"{stem}.txt").write_text(
                f"{cls_id} 0.500000 0.500000 0.500000 0.500000\n",
                encoding="utf-8",
            )
            global_idx += 1
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        "\n".join([
            f"path: {root.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "- red_box",
            "- green_box",
            "",
        ]),
        encoding="utf-8",
    )
    return yaml_path


def make_dirty_yolo_dataset(root: Path, *, include_bad_boxes: bool = True) -> Path:
    """Build a YOLO detection dataset containing controlled defects for Phase 2 inspect/dedup."""
    root = Path(root)
    yaml_path = make_mini_yolo_dataset(root, n_train=6, n_val=2, n_test=2, size=64)
    # 1) Intra-split SHA-256 duplicate in train
    shutil.copy2(root / "images" / "train" / "train_000.png", root / "images" / "train" / "train_dup.png")
    (root / "labels" / "train" / "train_dup.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    # 2) Cross-split leakage (train -> val & train -> test)
    shutil.copy2(root / "images" / "train" / "train_001.png", root / "images" / "val" / "val_leak.png")
    (root / "labels" / "val" / "val_leak.txt").write_text("1 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    shutil.copy2(root / "images" / "train" / "train_002.png", root / "images" / "test" / "test_leak.png")
    (root / "labels" / "test" / "test_leak.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    if include_bad_boxes:
        # 3) Missing label file in train
        _write_unique_rgb_png(root / "images" / "train" / "train_missing_lbl.png", 64, 900, 0)
        # 4) Empty label file in train
        _write_unique_rgb_png(root / "images" / "train" / "train_empty_lbl.png", 64, 901, 0)
        (root / "labels" / "train" / "train_empty_lbl.txt").write_text("", encoding="utf-8")
        # 5) Out-of-bounds bbox, degenerate bbox (w<=0, h<=0), invalid class_id >= nc, plus small/medium/large boxes
        _write_unique_rgb_png(root / "images" / "train" / "train_bad_boxes.png", 64, 902, 0)
        (root / "labels" / "train" / "train_bad_boxes.txt").write_text(
            "\n".join([
                "0 1.250000 0.500000 0.400000 0.400000",   # OOB
                "0 0.500000 0.500000 0.000000 0.200000",   # degenerate w=0
                "0 0.500000 0.500000 0.200000 -0.100000",  # degenerate h<0
                "5 0.500000 0.500000 0.200000 0.200000",   # class_id >= nc (nc=2)
                "0 0.100000 0.100000 0.050000 0.050000",   # small box (area < 0.01)
                "0 0.500000 0.500000 0.250000 0.250000",   # medium box
                "0 0.500000 0.500000 0.800000 0.800000",   # large box
            ]) + "\n",
            encoding="utf-8",
        )
    # 6) Extra class-0 samples for class imbalance
    for extra in range(3):
        idx = 950 + extra
        _write_unique_rgb_png(root / "images" / "train" / f"train_imbal_{extra}.png", 64, idx, 0)
        (root / "labels" / "train" / f"train_imbal_{extra}.txt").write_text(
            "0 0.500000 0.500000 0.040000 0.040000\n", encoding="utf-8"
        )
    return yaml_path


def make_mini_cls_dataset(
    root: Path,
    n_train: int = 8,
    n_val: int = 4,
    n_test: int = 4,
    size: int = 16,
) -> Path:
    """Build an ultra-fast ImageFolder classification dataset for multi-method CPU E2E runs."""
    root = Path(root)
    classes = ("cat_a", "cat_b")
    global_idx = 0
    for split, total in (("train", n_train), ("val", n_val), ("test", n_test)):
        for i in range(total):
            cls_id = i % 2
            img_path = root / split / classes[cls_id] / f"{split}_{i:03d}.png"
            _write_unique_rgb_png(img_path, size, global_idx, cls_id)
            global_idx += 1
    return root


def _build_audited_cls_run(
    run_dir: Path,
    *,
    dataset_source: str,
    sources: dict[str, str],
    configs: list[dict],
    training_seeds: list[int],
    method: str,
    search_seed: int,
    mode: str,
    budget_seconds: float,
    qualities: dict[int, float],
    times: dict[int, float],
    memories: dict[int, float],
    selected_indices: list[int],
    reference_boundary: Path | None = None,
    reference_boundary_sha256: str | None = None,
) -> Path:
    """Create a complete disk-audited v2 classification run passing audit_run() without mocks."""
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    freeze_sources(run_dir, sources)

    dataset = {
        "schema_version": 1,
        "task": "classification",
        "source": dataset_source,
        "seed": 101,
        "names": ["cat_a", "cat_b"],
        "splits": {"train": [{"image": "t0.png"}], "val": [{"image": "v0.png"}], "test": []},
        "counts": {"train": 8, "val": 4, "test": 0},
        "captions": {"train": 0, "val": 0, "test": 0},
        "validation_ready": True,
        "warnings": [],
    }
    subset = {
        **dataset,
        "splits": {"train": [{"image": "t0.png"}], "val": [{"image": "v0.png"}], "test": []},
        "counts": {"train": 4, "val": 2, "test": 0},
    }
    write_json(run_dir / "dataset.json", dataset)
    write_json(run_dir / "subset.json", subset)

    hw = {
        "device": "cpu",
        "host_label": "e2e-synthetic-host",
        "cpu": "e2e-synthetic-cpu",
        "torch_version": runtime()["torch"],
        "driver_version": None,
        "ml_runtime_ready": True,
        "ram_available_mb": 4096.0,
    }
    raw_protocol = {
        "implementation_version": "1.0.0",
        "mode": mode,
        "method": method,
        "execution_policy": {
            "version": 2,
            "integrity": "stat_cached_between_trials_full_hash_boundaries",
            "budget": "end_to_end_soft_deadline_with_measured_finalization_reserve",
            "data_loader": {"workers": 0, "cache_mb": 0.0, "pin_memory": "off"},
        },
        "selection_rule": "weighted",
        "model_descriptors": {},
        "dataset_descriptor": {},
        "run_directory": str(run_dir),
        "task": "classification",
        "dataset_source": dataset["source"],
        "dataset_sha256": digest(dataset),
        "dataset_content_id": "synthetic-n5-dataset-id",
        "subset_sha256": digest(subset),
        "hardware": hw,
        "runtime_versions": runtime(),
        "configs": configs,
        "training_seeds": training_seeds,
        "split_seed": 101,
        "search_seed": search_seed,
        "full_epochs": 2,
        "probe_epochs": 1,
        "trial_timeout": 60.0,
        "budget_seconds": float(budget_seconds),
        "objective": {
            "name": "balanced",
            "min_quality": 0.0,
            "max_time": None,
            "max_memory": 2048.0,
            "time_scale": 60.0,
            "memory_scale": 512.0,
            "weights": [0.6, 0.25, 0.15],
        },
        "example_counts": dataset["counts"],
        "num_classes": 2,
        "quality_definition": "final epoch validation accuracy",
        "determinism_policy": "synthetic audited fixture",
        "memory_kind": "process_tree_rss_mib_not_vram",
        "scheduler": {
            "policy_version": 2,
            "probe_fraction": 0.1,
            "cost_margin": 1.25,
            "cleanup_seconds": 1.0,
            "rungs": [1, 2],
        },
        "predictor": None,
        "source_files": sources,
        "model_files": {},
        "data_files": {},
        "preparation_seconds": 0.1,
        "measurement_environment": "uncontended_preflight",
        "test_policy": "never used for config/checkpoint/objective selection",
        "development_history": "e2e synthetic n5 benchmark fixture",
        "reference_boundary": str(reference_boundary) if reference_boundary else None,
        "acquisition": {
            "version": 1,
            "kernel_ridge": 0.05,
            "length_scale": 1.0,
            "exploration": 0.25,
            "novelty_weight": 0.05,
        },
    }
    if reference_boundary_sha256 is not None:
        raw_protocol["reference_boundary_sha256"] = reference_boundary_sha256
    protocol = lock_protocol(raw_protocol)
    write_json(run_dir / "protocol.json", protocol)

    ledger = Ledger(run_dir / "events.jsonl")
    ledger.emit("session_started", method=method, search_seed=search_seed)

    results = {str(i): [] for i in range(len(configs))}
    attempt_num = 0
    total_wall = 0.0
    for idx in selected_indices:
        for seed in training_seeds:
            attempt_num += 1
            attempt_id = f"attempt_{attempt_num:05d}"
            trial_dir = run_dir / "trials" / attempt_id
            trial_dir.mkdir(parents=True, exist_ok=True)
            ckpt = trial_dir / "best.pt"
            ckpt.write_bytes(f"synthetic-ckpt-{idx}-{seed}".encode("utf-8"))
            req = {
                "config": configs[idx],
                "protocol_sha256": protocol["protocol_sha256"],
                "seed": seed,
                "mode": "train",
                "full_epochs": 2,
                "trial_epochs": 2,
                "resume": None,
                "data_loader": protocol["execution_policy"]["data_loader"],
                "manifest": str((run_dir / "dataset.json").resolve()),
            }
            write_json(trial_dir / "request.json", req)
            wall = float(times[idx])
            total_wall += wall
            raw_res = {
                "attempt_id": attempt_id,
                "index": idx,
                "seed": seed,
                "mode": "train",
                "status": "ok",
                "epochs_completed": 2,
                "optimizer_steps": 4,
                "training_loss_finite": True,
                "actual_config_verified": True,
                "best_checkpoint": str(ckpt),
                "checkpoint_sha256": sha(ckpt),
                "quality": float(qualities[idx]),
                "memory_mb": float(memories[idx]),
                "worker_wall_seconds": wall,
                "measured_loop_seconds": round(wall * 0.8, 4),
                "trial_directory": str(trial_dir),
            }
            write_json(trial_dir / "result.json", raw_res)
            parent_res = {
                **raw_res,
                "request_sha256": sha(trial_dir / "request.json"),
                "raw_sha256": sha(trial_dir / "result.json"),
            }
            write_json(trial_dir / "parent_result.json", parent_res)
            ledger.emit("trial_started", attempt_id=attempt_id, index=idx, seed=seed, mode="train")
            ledger.emit(
                "trial_completed",
                attempt_id=attempt_id,
                parent_sha256=sha(trial_dir / "parent_result.json"),
                worker_seconds=wall,
                status="ok",
            )
            results[str(idx)].append(parent_res)

    state = {
        "results": results,
        "probes": [],
        "actions": [],
        "ordering": list(range(len(configs))),
    }
    from research_pipeline.v2.ledger import read_events
    draft = make_report(protocol, state, read_events(run_dir / "events.jsonl"))
    state["decision"] = draft["decision"]
    write_json(run_dir / "state.json", state)
    state_hash = digest(state)
    state_rel = f"commits/{state_hash}.json"
    (run_dir / "commits").mkdir(exist_ok=True)
    write_json(run_dir / state_rel, state)
    ledger.emit(
        "selection_committed",
        state_path=state_rel,
        state_sha256=state_hash,
        decision=draft["decision"],
        session_seconds=max(1.0, total_wall),
        original_budget_seconds=float(budget_seconds),
        this_execution_budget_seconds=float(budget_seconds),
    )
    write_json(run_dir / "decision.locked.json", draft["decision"])
    publish(run_dir, protocol, state, display=False)
    write_json(run_dir / "status.json", {"phase": "complete"})
    return run_dir


def make_synthetic_n5_study_dir(
    root: Path,
    *,
    seeds: tuple[int, ...] = (11, 17, 23, 29, 37),
    failed_pair: tuple[str, int] | None = None,
    zero_diff: bool = False,
) -> tuple[Path, Path]:
    """Create a genuine disk-audited n>=5 study directory + manifest for algorithm-report."""
    root = Path(root).resolve()
    study_dir = root / "study_n5"
    study_dir.mkdir(parents=True, exist_ok=True)
    sources = source_map()
    configs = [
        {
            "model": "cnn_classifier",
            "imgsz": 32,
            "batch": 16,
            "accumulation": 2,
            "precision": "fp32",
            "optimizer": "AdamW",
            "lr0": 0.001,
            "weight_decay": 0.01,
            "width": 16,
        },
        {
            "model": "cnn_classifier",
            "imgsz": 32,
            "batch": 16,
            "accumulation": 2,
            "precision": "fp32",
            "optimizer": "AdamW",
            "lr0": 0.0003,
            "weight_decay": 0.01,
            "width": 16,
        },
    ]
    qualities = {0: 0.90, 1: 0.60}
    times = {0: 5.0, 1: 8.0}
    memories = {0: 120.0, 1: 140.0}
    methods = ["adaptive_full", "random", "default"]
    training_seeds = [17]
    budget = 25.0

    manifest_study = {
        "methods": methods,
        "search_seeds": list(seeds),
        "training_seeds": training_seeds,
        "sources": sources,
        "test_used": False,
        "arguments": {
            "full_epochs": 2,
            "probe_epochs": 1,
            "split_seed": 101,
            "trial_timeout": 60.0,
            "seeds": training_seeds,
            "budget_seconds": budget,
        },
        "phase": "locked_before_search",
    }
    manifest_study["sha256"] = digest(manifest_study)
    write_json(study_dir / "benchmark_protocol.json", manifest_study)

    run_paths = []
    decision_hashes = {}
    for m_idx, method in enumerate(methods):
        for s_idx, seed in enumerate(seeds):
            run_dir = study_dir / method / f"repeat_{seed}"
            if failed_pair == (method, seed):
                chosen = []
            elif zero_diff:
                chosen = [0]
            elif method == "adaptive_full":
                chosen = [0]
            elif method == "random":
                chosen = [1] if s_idx < 4 else [0, 1]
            else:
                chosen = [1]
            _build_audited_cls_run(
                run_dir,
                dataset_source=str(study_dir / "synthetic_data"),
                sources=sources,
                configs=configs,
                training_seeds=training_seeds,
                method=method,
                search_seed=seed,
                mode="online",
                budget_seconds=budget,
                qualities=qualities,
                times=times,
                memories=memories,
                selected_indices=chosen,
            )
            dec = json.loads((run_dir / "decision.locked.json").read_text(encoding="utf-8"))
            run_paths.append(str(run_dir))
            decision_hashes[str(run_dir)] = digest(dec)

    boundary = study_dir / "decisions.locked.json"
    write_json(boundary, {
        "benchmark_sha256": manifest_study["sha256"],
        "runs": run_paths,
        "decision_hashes": decision_hashes,
    })

    ref_dir = study_dir / "reference" / "exhaustive"
    _build_audited_cls_run(
        ref_dir,
        dataset_source=str(study_dir / "synthetic_data"),
        sources=sources,
        configs=configs,
        training_seeds=training_seeds,
        method="random",
        search_seed=9999,
        mode="offline_reference",
        budget_seconds=120.0,
        qualities=qualities,
        times={0: 25.0, 1: 25.0},
        memories=memories,
        selected_indices=[0, 1],
        reference_boundary=boundary,
        reference_boundary_sha256=sha(boundary),
    )

    study_manifest = root / "studies_manifest.json"
    write_json(study_manifest, {
        "schema_version": 1,
        "studies": [{"path": str(study_dir)}],
        "required_methods": ["adaptive_full", "random"],
        "bootstrap_seed": 2027,
        "bootstrap_samples": 200,
        "confidence": 0.95,
    })
    return study_dir, study_manifest


build_audited_cls_run = _build_audited_cls_run


def make_det_protocol(
    configs: list[dict],
    *,
    method: str = "adaptive_full",
    selection_rule: str = "weighted",
    acquisition_version: int = 2,
    policy_version: int = 3,
    training_seeds: list[int] | None = None,
    budget_seconds: float = 120.0,
    max_memory: float = 2048.0,
    max_time: float | None = None,
    full_epochs: int = 2,
    cost_margin: float = 1.25,
    device: str = "cpu",
    model_descriptors: dict | None = None,
    dataset_descriptor: dict | None = None,
) -> dict:
    """Build a valid locked detection protocol dictionary for opaque-box API/scheduler tests."""
    seeds = training_seeds if training_seeds is not None else [17, 29]
    hw = {
        "device": device,
        "host_label": "e2e-host",
        "cpu": "e2e-cpu",
        "torch_version": runtime()["torch"],
        "driver_version": None,
        "ml_runtime_ready": True,
        "ram_available_mb": 4096.0,
    }
    raw = {
        "implementation_version": "1.0.0",
        "mode": "online",
        "method": method,
        "execution_policy": {
            "version": 2,
            "integrity": "stat_cached_between_trials_full_hash_boundaries",
            "budget": "end_to_end_soft_deadline_with_measured_finalization_reserve",
            "data_loader": {"workers": 0, "cache_mb": 0.0, "pin_memory": "off"},
        },
        "selection_rule": selection_rule,
        "model_descriptors": model_descriptors or {},
        "dataset_descriptor": dataset_descriptor or {},
        "run_directory": str(REPO_ROOT / "runs" / "_ephemeral_e2e"),
        "task": "detection",
        "dataset_source": "synthetic",
        "dataset_sha256": "0" * 64,
        "dataset_content_id": "synthetic-det-id",
        "subset_sha256": "1" * 64,
        "hardware": hw,
        "runtime_versions": runtime(),
        "configs": configs,
        "training_seeds": seeds,
        "split_seed": 101,
        "search_seed": 17,
        "full_epochs": full_epochs,
        "probe_epochs": 1,
        "trial_timeout": 120.0,
        "budget_seconds": float(budget_seconds),
        "objective": {
            "name": "balanced",
            "min_quality": 0.0,
            "max_time": max_time,
            "max_memory": max_memory,
            "time_scale": 60.0,
            "memory_scale": 512.0,
            "weights": [0.6, 0.25, 0.15],
        },
        "example_counts": {"train": 16, "val": 8, "test": 8},
        "num_classes": 2,
        "detection_metric": "map50",
        "quality_definition": "best validation mAP50 within declared horizon",
        "determinism_policy": "e2e fixture",
        "memory_kind": "cuda_reserved_allocator_mib" if device != "cpu" else "process_tree_rss_mib_not_vram",
        "scheduler": {
            "policy_version": policy_version,
            "probe_fraction": 0.1,
            "cost_margin": cost_margin,
            "cleanup_seconds": 1.0,
            "rungs": [1, full_epochs] if full_epochs > 1 else [1],
        },
        "predictor": None,
        "source_files": source_map(),
        "model_files": {},
        "data_files": {},
        "preparation_seconds": 0.1,
        "measurement_environment": "uncontended_preflight",
        "test_policy": "never used for selection",
        "development_history": "e2e",
        "reference_boundary": None,
        "acquisition": {
            "version": acquisition_version,
            "kernel_ridge": 0.05,
            "length_scale": 1.0,
            "exploration": 0.25,
            "novelty_weight": 0.05,
        },
    }
    return lock_protocol(raw)


def launch_ui_server(
    runs_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    timeout: float = 15.0,
    entry: str = "module",
) -> tuple[subprocess.Popen[str], str]:
    """Start `ui` server subprocess bound to loopback and return (process, base_url)."""
    import queue
    import threading
    import time

    if entry == "script":
        cmd = [sys.executable, str(REPO_ROOT / "train_config.py"), "ui", "--host", host, "--port", str(port), "--runs-dir", str(runs_dir)]
    else:
        cmd = [sys.executable, "-m", "research_pipeline.v2", "ui", "--host", host, "--port", str(port), "--runs-dir", str(runs_dir)]
    env = dict(os.environ)
    env["YOLO_AUTOINSTALL"] = "false"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    q: queue.Queue[str] = queue.Queue()
    lines: list[str] = []

    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            q.put(line)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    deadline = time.monotonic() + timeout
    url_re = re.compile(r"http://127\.0\.0\.1:(\d+)")
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"UI server exited prematurely ({proc.returncode}): {''.join(lines)}")
        try:
            line = q.get(timeout=0.1)
        except queue.Empty:
            continue
        m = url_re.search(line)
        if m:
            return proc, f"http://127.0.0.1:{m.group(1)}"
    stop_ui_server(proc)
    raise TimeoutError(f"Timed out waiting for UI server URL; output: {''.join(lines)}")


def stop_ui_server(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def http_get(url: str, timeout: float = 10.0) -> tuple[int, dict[str, str], str]:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, dict(resp.headers.items()), body
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers.items()), err.read().decode("utf-8", errors="replace")


def http_post(
    url: str,
    *,
    data: bytes = b"{}",
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> tuple[int, dict[str, str], str]:
    import urllib.error
    import urllib.request

    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, dict(resp.headers.items()), body
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers.items()), err.read().decode("utf-8", errors="replace")


def extract_nav_links(html_text: str) -> list[str]:
    links = re.findall(r'href=["\']([^"\'#]+)["\']', html_text, flags=re.IGNORECASE)
    seen = []
    for link in links:
        if link.startswith("/") and not link.startswith("//") and link not in seen:
            seen.append(link)
    return seen


def extract_csrf_token(html_text: str) -> str | None:
    patterns = (
        r'<meta[^>]+name=["\']csrf-token["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']csrf-token["\']',
        r'name=["\']csrf_token["\'][^>]+value=["\']([^"\']+)["\']',
        r'csrf[_-]?token["\']?\s*[:=]\s*["\']([A-Za-z0-9_.\-=+/]+)["\']',
    )
    for pat in patterns:
        m = re.search(pat, html_text, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def extract_external_urls(html_text: str) -> list[str]:
    urls = re.findall(r'(?:src|href)=["\'](https?://[^"\']+)["\']', html_text, flags=re.IGNORECASE)
    return [u for u in urls if not u.startswith("http://127.0.0.1") and not u.startswith("http://localhost")]


def offline_yolo_val(
    ckpt_path: Path | str,
    data_yaml: Path | str,
    *,
    imgsz: int = 64,
    batch: int = 2,
) -> tuple[float, float]:
    """Run Ultralytics YOLO(ckpt).val(rect=False) offline in an isolated subprocess."""
    probe = (
        "import json, tempfile\n"
        "from ultralytics import YOLO\n"
        "from ultralytics.utils import SETTINGS\n"
        "SETTINGS.update({'sync': False})\n"
        "with tempfile.TemporaryDirectory() as tmp_dir:\n"
        f"    r = YOLO({str(ckpt_path)!r}).val("
        f"data={str(data_yaml)!r}, split='test', imgsz={int(imgsz)}, batch={int(batch)}, "
        "rect=False, device='cpu', workers=0, plots=False, verbose=False, project=tmp_dir, name='val')\n"
        "    print(json.dumps({'map50': float(r.box.map50), 'map50_95': float(r.box.map)}))\n"
    )
    env = dict(os.environ)
    env["YOLO_AUTOINSTALL"] = "false"
    env["HTTP_PROXY"] = "http://127.0.0.1:9"
    env["HTTPS_PROXY"] = "http://127.0.0.1:9"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Offline YOLO.val probe failed: {proc.stderr}\n{proc.stdout}")
    last_line = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")][-1]
    parsed = json.loads(last_line)
    return float(parsed["map50"]), float(parsed["map50_95"])


def _lookup_dotted(obj: object, dotted: str) -> object:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def find_audit_dict(payload: dict) -> dict:
    if isinstance(payload.get("dataset"), dict) and isinstance(payload["dataset"].get("audit"), dict):
        return payload["dataset"]["audit"]
    if isinstance(payload.get("audit"), dict):
        return payload["audit"]
    if isinstance(payload, dict) and payload:
        return payload
    raise AssertionError(f"Missing structured audit dict in payload keys={list(payload.keys()) if isinstance(payload, dict) else type(payload)}")


def audit_field(audit: dict, *paths: str) -> object:
    for path in paths:
        if path in audit and audit[path] is not None:
            return audit[path]
        val = _lookup_dotted(audit, path)
        if val is not None:
            return val
    raise AssertionError(f"Missing expected audit field among {paths} in {list(audit.keys())}")


def audit_count(audit: dict, *paths: str) -> int:
    val = audit_field(audit, *paths)
    if isinstance(val, bool):
        raise AssertionError(f"Expected integer/list count for {paths}, got bool {val!r}")
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, list):
        return len(val)
    if isinstance(val, dict):
        if "sha256_leak_count" in val or "group_id_leak_count" in val:
            s_cnt = val.get("sha256_leak_count", 0)
            g_cnt = val.get("group_id_leak_count", 0)
            if (
                isinstance(s_cnt, (int, float))
                and not isinstance(s_cnt, bool)
                and isinstance(g_cnt, (int, float))
                and not isinstance(g_cnt, bool)
            ):
                return int(s_cnt) + int(g_cnt)
        for k in ("count", "total", "num", "n", "total_duplicate_files", "sha256_leak_count", "group_id_leak_count"):
            if isinstance(val.get(k), (int, float)) and not isinstance(val.get(k), bool):
                return int(val[k])
        if all(isinstance(v, list) for v in val.values()):
            return sum(len(v) for v in val.values())
    raise AssertionError(f"Could not extract numeric count from {val!r} for {paths}")


def extract_stat_p_value(pair: dict, metric_name: str, kind: str) -> float:
    metric_block = (pair.get("metrics") or {}).get(metric_name) or {}
    key_map = {
        "wilcoxon": (
            "wilcoxon_p_value", "wilcoxon_p", "wilcoxon_pvalue",
            "wilcoxon.p_value", "wilcoxon.p", "wilcoxon_signed_rank.p_value",
        ),
        "sign_flip": (
            "sign_flip_p_value", "sign_flip_p", "exact_sign_flip_p_value",
            "permutation_p_value", "sign_flip.p_value", "exact_sign_flip.p_value",
        ),
        "holm": (
            "holm_adjusted_p_value", "holm_p_value", "holm_bonferroni_p_value",
            "adjusted_p_value", "holm.p_value", "holm.adjusted_p_value",
        ),
    }
    for container in (metric_block, pair):
        for candidate in key_map[kind]:
            val = container.get(candidate) if "." not in candidate else _lookup_dotted(container, candidate)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return float(val)
    raise AssertionError(f"Missing {kind} p-value for {metric_name} in keys={list(metric_block.keys())}")


def create_dir_link_or_junction(target_dir: Path, link_dir: Path) -> str:
    """Create a directory symlink, falling back to unprivileged Windows NTFS junction."""
    target_dir = Path(target_dir).resolve()
    link_dir = Path(link_dir)
    try:
        os.symlink(target_dir, link_dir, target_is_directory=True)
        return "symlink"
    except OSError:
        if os.name != "nt":
            raise
        res = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link_dir), str(target_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0 or not link_dir.exists():
            raise RuntimeError(f"mklink /J failed: {res.stderr}\n{res.stdout}")
        return "junction"


def remove_dir_link_or_junction(link_dir: Path) -> None:
    link_dir = Path(link_dir)
    if not link_dir.exists() and not link_dir.is_symlink():
        return
    if link_dir.is_symlink():
        link_dir.unlink()
    elif os.name == "nt":
        subprocess.run(["cmd", "/c", "rmdir", str(link_dir)], capture_output=True, check=False)


def discover_ui_surface(base_url: str) -> dict:
    """Discover views, static assets, API routes, and CSRF token from served HTML/JS."""
    _, _, root_html = http_get(base_url + "/")
    nav_links = [lnk for lnk in extract_nav_links(root_html) if not lnk.startswith("/api/")]
    asset_links = re.findall(r'(?:src|href)=["\'](/[^"\'#?]+\.(?:js|css))["\']', root_html, flags=re.IGNORECASE)
    corpus = [root_html]
    for link in dict.fromkeys(nav_links[:8]):
        c, _, body = http_get(base_url + link)
        if c == 200:
            corpus.append(body)
    for asset in dict.fromkeys(asset_links):
        c, _, body = http_get(base_url + asset)
        if c == 200:
            corpus.append(body)
    joined = "\n".join(corpus)
    api_routes = sorted(set(re.findall(r'(/api/[a-zA-Z0-9_/\-]+)', joined)))
    post_routes = sorted(
        set(
            re.findall(
                r'["\'](/(?:[a-zA-Z0-9_\-]+/)*(?:audit|paper[-_]?export|paper|export|launch|stop|inspect|start|delete)[a-zA-Z0-9_/\-]*)["\']',
                joined,
                flags=re.IGNORECASE,
            )
            + re.findall(r'<form[^>]+action=["\'](/[^"\'#?]+)["\']', joined, flags=re.IGNORECASE)
            + re.findall(r'fetch\s*\(\s*["\'](/[^"\'#?]+)["\']', joined, flags=re.IGNORECASE)
        )
    )
    token = None
    for text in corpus:
        token = extract_csrf_token(text)
        if token:
            break
    return {
        "root_html": root_html,
        "nav_links": list(dict.fromkeys(nav_links)),
        "asset_links": list(dict.fromkeys(asset_links)),
        "api_routes": api_routes,
        "post_routes": post_routes,
        "csrf_token": token,
        "corpus": joined,
    }


