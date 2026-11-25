"""Phase 3 tests: Anytime Convergence, Paired Non-Parametric Stats, and Publication Export."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
import warnings
import xml.etree.ElementTree as ET

from research_pipeline.integrity import sha
from research_pipeline.v2 import algorithm_study, benchmark, cli, diagnostics, paper_export
from research_pipeline.v2.ledger import Ledger, read_events
from research_pipeline.v2.report import make_report
from research_pipeline.v2.schema import config_id, lock_protocol
from yolo_tuner.pipeline import write_json
from yolo_tuner.research import Objective, digest


REPO_ROOT = Path(__file__).resolve().parent.parent
ALGORITHM_010_EXPECTED_SHA256 = "d246015b106c70a1e0cb6a509f8b065c6bb4758debe0b65f89f86168e3b255f5"
OKABE_ITO_SET = {c.lower() for c in paper_export.OKABE_ITO}


def _runtime_versions():
    return {"python": "3.11", "torch": "2.5"}


def _cls_config(lr0=0.001, width=16, batch=16):
    return {
        "model": "cnn_classifier",
        "imgsz": 32,
        "batch": batch,
        "accumulation": 2,
        "precision": "fp32",
        "optimizer": "AdamW",
        "lr0": lr0,
        "weight_decay": 0.01,
        "width": width,
    }


def _det_config(lr0=0.001, imgsz=64, batch=2):
    return {
        "model": "yolo11n.pt",
        "imgsz": imgsz,
        "batch": batch,
        "accumulation": 1,
        "precision": "fp32",
        "optimizer": "AdamW",
        "lr0": lr0,
        "weight_decay": 0.0005,
        "freeze": 0,
    }


def _build_run(
    run_dir: Path,
    *,
    configs: list[dict],
    training_seeds: list[int],
    method: str,
    search_seed: int,
    mode: str,
    budget_seconds: float,
    sources: dict,
    qualities: dict[int, float],
    times: dict[int, float],
    memories: dict[int, float],
    trial_events: list[tuple[int, int, float]] | None = None,
    omit_event_payload_fields: bool = False,
    probe_configs: list[int] | None = None,
    partial_configs: dict[int, list[int]] | None = None,
    full_epochs: int = 2,
    implementation_version: str = "1.0.0",
    selection_rule: str = "weighted",
    acquisition_version: int = 2,
    objective_kwargs: dict | None = None,
    reference_boundary: Path | None = None,
    reference_boundary_sha256: str | None = None,
    task: str = "classification",
    detection_metric: str | None = None,
    test_eval: dict | None = None,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = run_dir / "source_snapshot"
    snap_dir.mkdir(parents=True, exist_ok=True)
    real_sources = {}
    for s_name, s_val in sources.items():
        s_file = snap_dir / s_name
        s_file.parent.mkdir(parents=True, exist_ok=True)
        s_file.write_text(f"# snapshot {s_name} {s_val}\n", encoding="utf-8")
        real_sources[s_name] = sha(s_file)

    ds_manifest = {"task": task, "source": "synthetic", "counts": {"train": 16, "val": 8, "test": 8}, "splits": {}}
    write_json(run_dir / "dataset.json", ds_manifest)
    write_json(run_dir / "subset.json", ds_manifest)
    ds_digest = digest(ds_manifest)
    ds_sha = sha(run_dir / "dataset.json")
    sub_sha = sha(run_dir / "subset.json")

    hw = {
        "device": "cpu",
        "host_label": "p3-host",
        "cpu": "p3-cpu",
        "torch_version": "2.5",
        "driver_version": None,
        "ml_runtime_ready": True,
        "ram_available_mb": 4096.0,
    }
    obj = {
        "name": "balanced",
        "min_quality": 0.0,
        "max_time": None,
        "max_memory": 2048.0,
        "time_scale": 60.0,
        "memory_scale": 512.0,
        "weights": [0.6, 0.25, 0.15],
        **(objective_kwargs or {}),
    }
    raw_proto = {
        "implementation_version": implementation_version,
        "mode": mode,
        "method": method,
        "execution_policy": {
            "version": 2,
            "integrity": "stat_cached_between_trials_full_hash_boundaries",
            "budget": "end_to_end_soft_deadline_with_measured_finalization_reserve",
            "data_loader": {"workers": 0, "cache_mb": 0.0, "pin_memory": "off"},
        },
        "selection_rule": selection_rule,
        "run_directory": str(run_dir),
        "task": task,
        "dataset_source": "synthetic",
        "dataset_sha256": ds_digest,
        "dataset_content_id": "synthetic-p3-ds",
        "subset_sha256": ds_digest,
        "hardware": hw,
        "runtime_versions": _runtime_versions(),
        "configs": configs,
        "training_seeds": list(training_seeds),
        "split_seed": 101,
        "search_seed": search_seed,
        "full_epochs": full_epochs,
        "probe_epochs": 1,
        "trial_timeout": 120.0,
        "budget_seconds": float(budget_seconds),
        "objective": obj,
        "example_counts": {"train": 16, "val": 8, "test": 8},
        "quality_definition": (
            "best validation mAP50 within declared horizon" if task == "detection" else "final epoch validation accuracy"
        ),
        "determinism_policy": "test",
        "memory_kind": "process_tree_rss_mib_not_vram",
        "scheduler": {
            "policy_version": 3,
            "probe_fraction": 0.1,
            "cost_margin": 1.25,
            "cleanup_seconds": 1.0,
            "rungs": [1, full_epochs] if full_epochs > 1 else [1],
        },
        "predictor": None,
        "source_files": real_sources,
        "model_files": {},
        "data_files": {},
        "preparation_seconds": 0.1,
        "measurement_environment": "uncontended_preflight",
        "test_policy": "never used for selection",
        "development_history": "phase3 test",
        "reference_boundary": str(reference_boundary) if reference_boundary else None,
        "reference_boundary_sha256": reference_boundary_sha256,
        "acquisition": {
            "version": acquisition_version,
            "kernel_ridge": 0.05,
            "length_scale": 1.0,
            "exploration": 0.25,
            "novelty_weight": 0.05,
        },
    }
    if task == "detection":
        raw_proto["detection_metric"] = detection_metric or "map50"
    protocol = lock_protocol(raw_proto)
    write_json(run_dir / "protocol.json", protocol)

    clock_now = [0.0]
    ledger = Ledger(run_dir / "events.jsonl", clock=lambda: clock_now[0], origin=0.0)
    ledger.emit("session_started", mode=mode, method=method)

    results = {str(i): [] for i in range(len(configs))}
    probes = []
    attempt_counter = 0

    def _add_trial(idx: int, seed: int, is_full: bool, t_done: float, epochs_done: int, status: str = "ok"):
        nonlocal attempt_counter
        attempt_counter += 1
        aid = f"attempt_{attempt_counter:05d}"
        trial_dir = run_dir / f"trial_{attempt_counter:05d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        ckpt = trial_dir / "best.pt"
        ckpt.write_bytes(f"ckpt-{idx}-{seed}-{is_full}".encode("utf-8"))
        ckpt_sha = sha(ckpt)
        manifest_file = run_dir / ("dataset.json" if is_full else "subset.json")
        req_payload = {
            "attempt_id": aid,
            "index": idx,
            "config": configs[idx],
            "seed": seed,
            "trial_epochs": epochs_done,
            "full_epochs": full_epochs,
            "full_data": is_full,
            "mode": "train" if is_full else "search",
            "resume": None,
            "manifest": str(manifest_file),
            "dataset": str(manifest_file),
            "dataset_sha256": ds_sha if is_full else sub_sha,
            "protocol_sha256": protocol["protocol_sha256"],
            "source_files": real_sources,
            "data_loader": protocol["execution_policy"]["data_loader"],
        }
        if task == "detection":
            req_payload["detection_metric"] = protocol["detection_metric"]
            req_payload["data"] = str(run_dir / ("full_data" if is_full else "subset_data") / "data.yaml")
        req_path = trial_dir / "request.json"
        write_json(req_path, req_payload)
        req_sha = sha(req_path)
        q_val = float(qualities[idx])
        w_sec = float(times[idx])
        m_mb = float(memories[idx])
        lc = []
        for ep in range(1, epochs_done + 1):
            frac = 0.7 + 0.3 * (ep / max(1, epochs_done))
            pt = {"epoch": ep, "quality": round(q_val * frac, 6), "accuracy": round(q_val * frac, 6)}
            if task == "detection":
                pt.update({
                    "map50": round(q_val * frac, 6),
                    "map50_95": round(q_val * frac * 0.7, 6),
                    "precision": round(min(1.0, q_val * 1.02), 6),
                    "recall": round(q_val * 0.95, 6),
                    "f1": round(q_val * 0.96, 6),
                })
            lc.append(pt)
        res_payload = {
            "attempt_id": aid,
            "index": idx,
            "seed": seed,
            "full": is_full,
            "mode": "train" if is_full else "search",
            "status": status,
            "epochs_completed": epochs_done,
            "optimizer_steps": 10 * epochs_done,
            "training_loss_finite": True,
            "actual_config_verified": True,
            "checkpoint_sha256": ckpt_sha,
            "best_checkpoint": str(ckpt),
            "best_checkpoint_epoch": epochs_done,
            "best_epoch": epochs_done,
            "quality": q_val,
            "memory_mb": m_mb,
            "worker_wall_seconds": w_sec,
            "learning_curve": lc,
            "trial_directory": str(trial_dir),
            "request_path": str(req_path),
            "request_sha256": req_sha,
            "result_path": str(trial_dir / "result.json"),
            "directory": str(trial_dir),
        }
        if task == "detection":
            res_payload.update({
                "primary_metric": protocol["detection_metric"],
                "map50": q_val,
                "map50_95": round(q_val * 0.7, 6),
                "precision": round(min(1.0, q_val * 1.02), 6),
                "recall": round(q_val * 0.95, 6),
                "f1": round(q_val * 0.96, 6),
            })
        res_path = trial_dir / "result.json"
        write_json(res_path, res_payload)
        res_payload["raw_sha256"] = sha(res_path)
        parent_path = trial_dir / "parent_result.json"
        write_json(parent_path, res_payload)
        parent_sha = sha(parent_path)

        ev_payload = {"attempt_id": aid, "parent_sha256": parent_sha, "worker_seconds": w_sec, "status": status}
        if not omit_event_payload_fields:
            ev_payload.update({"index": idx, "seed": seed, "full": is_full, "directory": str(trial_dir)})
        clock_now[0] = max(clock_now[0], float(t_done) - 0.5)
        ledger.emit("trial_started", attempt_id=aid, index=idx, seed=seed, full=is_full)
        clock_now[0] = max(clock_now[0], float(t_done))
        ledger.emit("trial_completed", **ev_payload)
        if is_full:
            results[str(idx)].append(res_payload)
        else:
            probes.append(res_payload)

    # Optional probe trials (must NEVER leak into anytime_trajectory)
    t_cursor = 1.0
    for p_idx in (probe_configs or []):
        _add_trial(p_idx, training_seeds[0], False, t_cursor, 1)
        t_cursor += 1.0

    # Optional partial seeds (must NEVER leak into anytime_trajectory)
    for idx, p_seeds in (partial_configs or {}).items():
        for sd in p_seeds:
            _add_trial(idx, sd, True, t_cursor, full_epochs)
            t_cursor += 1.0

    # Full-seed trials
    if trial_events is not None:
        for idx, sd, t_done in trial_events:
            _add_trial(idx, sd, True, t_done, full_epochs)

    from research_pipeline.v2.schema import measurements
    from research_pipeline.v2.selection import select

    rows = measurements(configs, results, training_seeds, full_epochs, Objective(**obj))
    decision = select(rows, protocol)
    acq_hist = []
    full_obs_indices = [r["index"] for r in rows if r["complete"]]
    if len(full_obs_indices) >= 3:
        acq_hist.append({
            "step": 1,
            "index": full_obs_indices[-1],
            "reason": "acquisition_v2",
            "remaining_seconds": max(0.0, budget_seconds - t_cursor),
            "surrogate": {"fitted": True, "observed_configurations": len(full_obs_indices)},
            "surrogate_diagnostics": {
                "kernel": "matern52",
                "length_scale": 0.5,
                "ridge": 0.01,
                "loocv_mae": 0.02,
                "loocv_rmse": 0.03,
                "heteroscedastic": len(training_seeds) >= 2,
            },
            "scores": {
                str(full_obs_indices[-1]): {
                    "score": 0.42,
                    "quality_mean_prediction": qualities[full_obs_indices[-1]],
                    "quality_optimistic": min(1.0, qualities[full_obs_indices[-1]] + 0.03),
                    "uncertainty": 0.04,
                    "remaining_all_seed_cost_seconds": times[full_obs_indices[-1]],
                }
            },
        })

    state = {
        "results": results,
        "probes": probes,
        "acquisition_history": acq_hist,
    }
    write_json(run_dir / "state.json", state)
    write_json(run_dir / "decision.locked.json", decision)
    write_json(run_dir / "status.json", {"phase": "complete", "protocol": protocol["protocol_sha256"]})
    ledger.emit(
        "selection_committed",
        state_path="state.json",
        state_sha256=digest(state),
        decision=decision,
        session_seconds=max(1.0, float(clock_now[0])),
    )
    report = make_report(protocol, state, read_events(run_dir / "events.jsonl"))
    write_json(run_dir / "report.json", report)

    if test_eval is not None:
        ev_dir = run_dir / "evaluation" / "eval_001"
        ev_dir.mkdir(parents=True, exist_ok=True)
        write_json(ev_dir / "evaluation.json", test_eval)
        write_json(run_dir / "test_history.json", [{"status": "evaluated", "output": str(ev_dir)}])

    return protocol, rows


def _extract_pareto_label_boxes(svg_text: str):
    root = ET.fromstring(svg_text)
    boxes = []
    for elem in root.iter():
        if not elem.tag.endswith("text"):
            continue
        txt = (elem.text or "").strip()
        if not re.match(r"^c\d{2}(\s*★)?$", txt):
            continue
        x = float(elem.attrib["x"])
        y = float(elem.attrib["y"])
        anchor = elem.attrib.get("text-anchor", "start")
        w = max(20.0, len(txt) * 6.4)
        h = 12.0
        if anchor == "start":
            x0 = x
        elif anchor == "end":
            x0 = x - w
        else:
            x0 = x - w / 2.0
        boxes.append((txt, x0, y - 10.0, x0 + w, y - 10.0 + h))
    return boxes


class Phase3StatsAndExportTests(unittest.TestCase):
    def test_anytime_trajectory_attempt_id_attribution_and_no_proxy_or_partial_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfgs = [_cls_config(lr0=lr) for lr in (0.001, 0.0003, 0.003)]
            sources = {"dummy.py": "1"}
            ref_dir = root / "ref"
            _build_run(
                ref_dir,
                configs=cfgs,
                training_seeds=[17, 29],
                method="random",
                search_seed=123456,
                mode="offline_reference",
                budget_seconds=200.0,
                sources=sources,
                qualities={0: 0.70, 1: 0.90, 2: 0.80},
                times={0: 5.0, 1: 6.0, 2: 7.0},
                memories={0: 100.0, 1: 100.0, 2: 100.0},
                trial_events=[(0, 17, 5.0), (0, 29, 10.0), (1, 17, 16.0), (1, 29, 22.0), (2, 17, 29.0), (2, 29, 36.0)],
            )
            run_dir = root / "online"
            # Config 2 has probe + partial seed (only seed 17) — must NOT enter trajectory!
            # Config 0 confirms at t=12.0, Config 1 (reference optimal) confirms at t=24.0.
            prot, _ = _build_run(
                run_dir,
                configs=cfgs,
                training_seeds=[17, 29],
                method="adaptive_staged_seeds",
                search_seed=11,
                mode="online",
                budget_seconds=40.0,
                sources=sources,
                qualities={0: 0.70, 1: 0.90, 2: 0.99},
                times={0: 5.0, 1: 6.0, 2: 4.0},
                memories={0: 100.0, 1: 100.0, 2: 100.0},
                probe_configs=[2],
                partial_configs={2: [17]},
                trial_events=[(0, 17, 8.0), (0, 29, 12.0), (1, 17, 18.0), (1, 29, 24.0)],
                omit_event_payload_fields=True,
            )
            res = diagnostics.anytime_trajectory(run_dir, prot, reference=ref_dir)
            self.assertEqual(res["confirmed_total_count"], 2)
            self.assertEqual(res["confirmed_feasible_count"], 2)
            self.assertEqual([pt["config_index"] for pt in res["trajectory"]], [0, 1])
            self.assertEqual([pt["active_seconds"] for pt in res["trajectory"]], [12.0, 24.0])
            utils = [pt["best_utility"] for pt in res["trajectory"]]
            self.assertEqual(utils, sorted(utils))
            self.assertAlmostEqual(res["time_to_target_95"], 24.0, places=6)
            self.assertIsNotNone(res["auarc"])
            self.assertGreater(res["auarc"], 0.0)

    def test_anytime_trajectory_negative_utility_ttt95_and_missing_files_graceful(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfgs = [_cls_config(lr0=lr) for lr in (0.001, 0.0003)]
            sources = {"dummy.py": "1"}
            # Make all utilities negative (high time penalty relative to small quality)
            obj_kw = {"time_scale": 10.0, "memory_scale": 100.0, "weights": [0.1, 0.6, 0.3]}
            ref_dir = root / "ref"
            _build_run(
                ref_dir,
                configs=cfgs,
                training_seeds=[17],
                method="random",
                search_seed=123456,
                mode="offline_reference",
                budget_seconds=100.0,
                sources=sources,
                qualities={0: 0.2, 1: 0.5},
                times={0: 20.0, 1: 10.0},
                memories={0: 200.0, 1: 100.0},
                trial_events=[(0, 17, 10.0), (1, 17, 20.0)],
                objective_kwargs=obj_kw,
            )
            run_dir = root / "online"
            prot, _ = _build_run(
                run_dir,
                configs=cfgs,
                training_seeds=[17],
                method="adaptive_full",
                search_seed=11,
                mode="online",
                budget_seconds=30.0,
                sources=sources,
                qualities={0: 0.2, 1: 0.5},
                times={0: 20.0, 1: 10.0},
                memories={0: 200.0, 1: 100.0},
                trial_events=[(0, 17, 8.0), (1, 17, 18.0)],
                objective_kwargs=obj_kw,
            )
            res = diagnostics.anytime_trajectory(run_dir, prot, reference=ref_dir)
            self.assertLess(res["reference_optimal_utility"], 0.0)
            self.assertLess(res["target_utility_95"], 0.0)
            self.assertGreater(res["target_utility_95"], min(p["config_utility"] for p in res["trajectory"]))
            self.assertAlmostEqual(res["time_to_target_95"], 18.0, places=6)
            self.assertGreaterEqual(res["auarc"], 0.0)

            # Graceful fallback when state.json & events.jsonl do not exist (mocked benchmark dirs)
            mock_dir = root / "mock_only"
            mock_dir.mkdir()
            empty_res = diagnostics.anytime_trajectory(mock_dir, prot, reference=ref_dir)
            self.assertEqual(empty_res["trajectory"], [])
            self.assertIsNone(empty_res["auarc"])
            self.assertIsNone(empty_res["time_to_target_95"])

    def test_benchmark_compare_gates_010_and_includes_anytime_for_v11_plus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfgs = [_cls_config(lr0=lr) for lr in (0.001, 0.0003)]
            sources = {"dummy.py": "1"}
            ref_dir = root / "ref"
            _build_run(
                ref_dir,
                configs=cfgs,
                training_seeds=[17],
                method="random",
                search_seed=123456,
                mode="offline_reference",
                budget_seconds=100.0,
                sources=sources,
                qualities={0: 0.6, 1: 0.9},
                times={0: 5.0, 1: 5.0},
                memories={0: 100.0, 1: 100.0},
                trial_events=[(0, 17, 5.0), (1, 17, 10.0)],
                implementation_version="0.11.0",
                task="detection",
                detection_metric="map50",
                acquisition_version=1,
            )
            run_dir = root / "online"
            _build_run(
                run_dir,
                configs=cfgs,
                training_seeds=[17],
                method="adaptive_full",
                search_seed=11,
                mode="online",
                budget_seconds=25.0,
                sources=sources,
                qualities={0: 0.6, 1: 0.9},
                times={0: 5.0, 1: 5.0},
                memories={0: 100.0, 1: 100.0},
                trial_events=[(0, 17, 5.0), (1, 17, 10.0)],
                implementation_version="0.11.0",
                task="detection",
                detection_metric="map50",
                acquisition_version=1,
            )
            with patch("research_pipeline.v2.benchmark.audit_run", side_effect=lambda p: (
                Path(p).resolve(),
                json.loads((Path(p) / "protocol.json").read_text(encoding="utf-8")),
                json.loads((Path(p) / "report.json").read_text(encoding="utf-8"))["candidates"],
                2,
            )):
                comp = benchmark.compare([str(run_dir)], str(ref_dir))
            self.assertIn("auarc", comp["outcomes"][0])
            self.assertIn("anytime_trajectory", comp["outcomes"][0])
            self.assertIn("auarc_mean", comp["summaries"][0])
            bm_out = root / "bm_out"
            bm_out.mkdir()
            with redirect_stdout(io.StringIO()):
                benchmark._publish_benchmark(bm_out, comp)
            bm_md = (bm_out / "BENCHMARK.md").read_text(encoding="utf-8")
            self.assertIn("AUARC mean", bm_md)
            self.assertIn("TTT-95 mean", bm_md)

            # Verify 0.10.0 protocol (extended == False) omits anytime keys
            ref_010 = root / "ref_010"
            _build_run(
                ref_010,
                configs=cfgs,
                training_seeds=[17],
                method="random",
                search_seed=123456,
                mode="offline_reference",
                budget_seconds=100.0,
                sources=sources,
                qualities={0: 0.6, 1: 0.9},
                times={0: 5.0, 1: 5.0},
                memories={0: 100.0, 1: 100.0},
                trial_events=[(0, 17, 5.0), (1, 17, 10.0)],
                implementation_version="0.10.0",
                task="classification",
                acquisition_version=1,
            )
            run_010 = root / "online_010"
            _build_run(
                run_010,
                configs=cfgs,
                training_seeds=[17],
                method="adaptive_full",
                search_seed=11,
                mode="online",
                budget_seconds=25.0,
                sources=sources,
                qualities={0: 0.6, 1: 0.9},
                times={0: 5.0, 1: 5.0},
                memories={0: 100.0, 1: 100.0},
                trial_events=[(0, 17, 5.0), (1, 17, 10.0)],
                implementation_version="0.10.0",
                task="classification",
                acquisition_version=1,
            )
            with patch("research_pipeline.v2.benchmark.audit_run", side_effect=lambda p: (
                Path(p).resolve(),
                json.loads((Path(p) / "protocol.json").read_text(encoding="utf-8")),
                json.loads((Path(p) / "report.json").read_text(encoding="utf-8"))["candidates"],
                2,
            )):
                comp_010 = benchmark.compare([str(run_010)], str(ref_010))
            self.assertNotIn("auarc", comp_010["outcomes"][0])
            self.assertNotIn("anytime_trajectory", comp_010["outcomes"][0])
            self.assertNotIn("auarc_mean", comp_010["summaries"][0])

    def test_algorithm_010_sha256_preserved_and_nonparametric_stats_gated_at_n5(self):
        rep_010 = algorithm_study.analyze_studies(REPO_ROOT / "configs" / "algorithm_010_studies.json")
        self.assertEqual(rep_010["sha256"], ALGORITHM_010_EXPECTED_SHA256)
        for st in rep_010["studies"]:
            self.assertNotIn("paired_nonparametric_scope", st)
            for pair in st["paired_comparisons"]:
                for m in pair["metrics"].values():
                    self.assertNotIn("wilcoxon", m)
                    self.assertNotIn("sign_flip", m)
                    self.assertNotIn("holm", m)

        # Unit checks on statistical helpers: all-zero diffs produce zero warnings and p=1.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wx_zero = algorithm_study._paired_wilcoxon([0.0, 0.0, 0.0, 0.0, 0.0])
            sf_zero = algorithm_study._exact_sign_flip([0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(len(caught), 0, f"Unexpected warnings: {[str(w.message) for w in caught]}")
        self.assertEqual(wx_zero["p_value"], 1.0)
        self.assertEqual(sf_zero["p_value"], 1.0)

        # Exact sign-flip for n=5 strictly negative diffs: 2 / 32 = 0.0625
        sf_neg = algorithm_study._exact_sign_flip([-0.1, -0.2, -0.3, -0.4, -0.5])
        self.assertEqual(sf_neg["method"], "exact")
        self.assertEqual(sf_neg["permutations"], 32)
        self.assertAlmostEqual(sf_neg["p_value"], 2.0 / 32.0, places=9)

        # Monte Carlo fallback for n=18 > 16
        sf_mc = algorithm_study._exact_sign_flip([-0.1] * 18, seed=2027)
        self.assertEqual(sf_mc["method"], "monte_carlo")
        self.assertLess(sf_mc["p_value"], 0.01)

        # Holm-Bonferroni monotonicity & step-down exactness
        adj = algorithm_study._holm_bonferroni([0.04, 0.01, 0.03])
        self.assertAlmostEqual(adj[1], 0.03, places=9)
        self.assertAlmostEqual(adj[2], 0.06, places=9)
        self.assertAlmostEqual(adj[0], 0.06, places=9)

        # Partial failure reduces paired count below 5 for one baseline while keeping >= 5 for another
        outcomes = []
        for s_idx, seed in enumerate((11, 17, 23, 29, 37)):
            outcomes.append({"method": "adaptive_full", "search_seed": seed, "feasible": True, "objective_regret": 0.01 * (s_idx + 1), "online_seconds": 10.0, "time_to_first_full_config_seconds": 4.0})
            outcomes.append({"method": "random", "search_seed": seed, "feasible": True, "objective_regret": 0.10 * (s_idx + 1), "online_seconds": 15.0, "time_to_first_full_config_seconds": 6.0})
            fail_def = seed == 11
            outcomes.append({"method": "default", "search_seed": seed, "feasible": not fail_def, "objective_regret": None if fail_def else 0.20, "online_seconds": 12.0, "time_to_first_full_config_seconds": None if fail_def else 8.0})
        paired = algorithm_study._paired(outcomes, {"seed": 2027, "samples": 100, "confidence": 0.95})
        p_rand = next(p for p in paired if p["baseline"] == "random")
        p_def = next(p for p in paired if p["baseline"] == "default")
        self.assertIn("wilcoxon", p_rand["metrics"]["objective_regret"])
        self.assertIn("sign_flip", p_rand["metrics"]["objective_regret"])
        self.assertIn("holm", p_rand["metrics"]["objective_regret"])
        self.assertNotIn("wilcoxon", p_def["metrics"]["objective_regret"])
        self.assertNotIn("holm", p_def["metrics"]["objective_regret"])
        self.assertIn("wilcoxon", p_def["metrics"]["online_seconds"])

    def test_paper_export_all_6_svgs_okabe_ito_non_overlapping_lang_vi_en_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Clustered identical/near coordinates to stress-test Pareto label collision solver
            cfgs = [_det_config(lr0=lr, imgsz=sz) for lr in (0.001, 0.0003) for sz in (64, 96)]
            sources = {"dummy.py": "1"}
            test_eval = {
                "evaluated_seeds": [17, 29],
                "multi_seed_summary": {
                    "map50": {"mean": 0.88, "sd": 0.01, "min": 0.87, "max": 0.89, "n": 2},
                    "map75": {"mean": 0.72, "sd": 0.02, "min": 0.70, "max": 0.74, "n": 2},
                    "map50_95": {"mean": 0.61, "sd": 0.01, "min": 0.60, "max": 0.62, "n": 2},
                    "precision": {"mean": 0.89, "sd": 0.01, "min": 0.88, "max": 0.90, "n": 2},
                    "recall": {"mean": 0.84, "sd": 0.02, "min": 0.82, "max": 0.86, "n": 2},
                    "f1": {"mean": 0.86, "sd": 0.01, "min": 0.85, "max": 0.87, "n": 2},
                },
                "validation_vs_test": {"validation_mean": 0.87, "validation_sd": 0.01, "test_mean": 0.88, "test_sd": 0.01, "generalization_gap_test_minus_val": 0.01},
                "per_class_summary": [
                    {"class_id": 0, "class_name": "buffalo", "ap50_mean": 0.90, "ap50_sd": 0.02, "ap50_95_mean": 0.65, "ap50_95_sd": 0.01},
                    {"class_id": 1, "class_name": "elephant", "ap50_mean": 0.86, "ap50_sd": 0.01, "ap50_95_mean": 0.57, "ap50_95_sd": 0.02},
                ],
                "latency": {"available": True, "precision": "fp32", "batch_size": 2, "inference_ms": {"mean": 3.5}, "total_ms_per_image": {"mean": 4.2}, "fps": {"mean": 238.1}},
                "pr_curve": {
                    "available": True,
                    "recall_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
                    "mean_precision": [1.0, 0.96, 0.91, 0.84, 0.62],
                    "per_class": [
                        {"class_id": 0, "class_name": "buffalo", "precision": [1.0, 0.98, 0.93, 0.86, 0.65]},
                        {"class_id": 1, "class_name": "elephant", "precision": [1.0, 0.94, 0.89, 0.82, 0.59]},
                    ],
                },
                "f1_curve": {"available": True, "best_confidence": 0.45, "max_mean_f1": 0.87},
                "confusion_matrix": {
                    "available": True,
                    "labels": ["buffalo", "elephant", "background"],
                    "normalized": [[0.90, 0.05, 0.05], [0.04, 0.88, 0.08], [0.06, 0.07, 0.87]],
                },
            }
            run_dir = root / "opt_run"
            prot, rows = _build_run(
                run_dir,
                configs=cfgs,
                training_seeds=[17, 29],
                method="adaptive_ehvi",
                search_seed=17,
                mode="online",
                budget_seconds=60.0,
                sources=sources,
                qualities={0: 0.80, 1: 0.80, 2: 0.85, 3: 0.89},
                times={0: 10.0, 1: 10.0, 2: 12.0, 3: 15.0},
                memories={0: 200.0, 1: 200.0, 2: 220.0, 3: 240.0},
                trial_events=[(0, 17, 5.0), (0, 29, 10.0), (1, 17, 15.0), (1, 29, 20.0), (2, 17, 25.0), (2, 29, 30.0), (3, 17, 35.0), (3, 29, 40.0)],
                task="detection",
                detection_metric="map50",
                acquisition_version=2,
                test_eval=test_eval,
            )
            loocv = paper_export.surrogate_loocv(prot, rows)
            self.assertTrue(loocv["available"])
            self.assertIn(loocv["kernel"], {"matern52", "rbf"})
            self.assertTrue(loocv["heteroscedastic"])

            before_hashes = {p.relative_to(run_dir).as_posix(): sha(p) for p in sorted(run_dir.rglob("*")) if p.is_file()}
            out_en_1 = root / "out_en_1"
            out_en_2 = root / "out_en_2"
            out_vi = root / "out_vi"
            with redirect_stdout(io.StringIO()):
                rc_en = cli.main(argv=["paper-export", "--run", str(run_dir), "--output", str(out_en_1), "--lang", "en"])
                rc_vi = cli.main(argv=["paper-export", "--run", str(run_dir), "--output", str(out_vi), "--lang", "vi"])
            self.assertEqual(rc_en, 0)
            self.assertEqual(rc_vi, 0)
            paper_export.export_paper_artifacts(run_dir, out_en_2, lang="en")
            after_hashes = {p.relative_to(run_dir).as_posix(): sha(p) for p in sorted(run_dir.rglob("*")) if p.is_file()}
            self.assertEqual(before_hashes, after_hashes)

            with self.assertRaises(ValueError):
                paper_export.export_paper_artifacts(run_dir, root / "bad_lang", lang="fr")

            expected_svgs = (
                "pareto_front.svg",
                "learning_curves.svg",
                "anytime_convergence.svg",
                "per_class_ap50.svg",
                "pr_curve.svg",
                "confusion_matrix.svg",
            )
            for svg_name in expected_svgs:
                p1 = out_en_1 / svg_name
                p2 = out_en_2 / svg_name
                self.assertTrue(p1.is_file(), f"Missing {svg_name}")
                b1 = p1.read_bytes()
                b2 = p2.read_bytes()
                self.assertEqual(b1, b2, f"Non-deterministic SVG: {svg_name}")
                txt = b1.decode("utf-8")
                ET.fromstring(txt)
                if svg_name != "confusion_matrix.svg":
                    self.assertTrue(any(col in txt.lower() for col in OKABE_ITO_SET), f"Missing Okabe-Ito color in {svg_name}")

            # Verify zero overlapping label boxes on clustered Pareto points (c00 and c01 share exact coords)
            boxes = _extract_pareto_label_boxes((out_en_1 / "pareto_front.svg").read_text(encoding="utf-8"))
            self.assertEqual(len(boxes), 4)
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    _, ax0, ay0, ax1, ay1 = boxes[i]
                    _, bx0, by0, bx1, by1 = boxes[j]
                    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
                    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
                    self.assertEqual(ix * iy, 0.0, f"Overlapping Pareto labels: {boxes[i]} vs {boxes[j]}")

            # Check LaTeX & Markdown localization
            tex_en = (out_en_1 / "tables.tex").read_text(encoding="utf-8")
            tex_vi = (out_vi / "tables.tex").read_text(encoding="utf-8")
            self.assertIn(r"\toprule", tex_en)
            self.assertIn(r"\toprule", tex_vi)
            self.assertNotIn("—", tex_en)
            self.assertNotIn("—", tex_vi)
            self.assertIn("tab:nckh_test_latency", tex_en)
            self.assertNotEqual(tex_en, tex_vi)

            md_en = (out_en_1 / "PAPER_SUMMARY.md").read_text(encoding="utf-8")
            md_vi = (out_vi / "PAPER_SUMMARY.md").read_text(encoding="utf-8")
            for md_txt in (md_en, md_vi):
                self.assertIn("Held-out Test", md_txt)
                self.assertIn("LOOCV", md_txt)
                self.assertIn("238.10 FPS", md_txt)
                self.assertIn("max_mean_F1 = 0.8700", md_txt)
            self.assertNotEqual(md_en, md_vi)

            # Verify PROVENANCE.json hashes every file excluding itself
            prov = json.loads((out_en_1 / "PROVENANCE.json").read_text(encoding="utf-8"))
            self.assertNotIn("PROVENANCE.json", prov["exported_artifacts"])
            self.assertIn("PAPER_EVIDENCE.json", prov["exported_artifacts"])
            for fname, meta in prov["exported_artifacts"].items():
                self.assertEqual(meta["sha256"], sha(out_en_1 / fname))

            # Verify historical real 0.11.0 run anytime_trajectory if present on disk
            hist_bm = REPO_ROOT / "runs" / "african_wildlife_nckh_benchmark" / "20261118_173050_553132"
            if hist_bm.is_dir():
                bm_json = json.loads((hist_bm / "BENCHMARK.json").read_text(encoding="utf-8"))
                ref_p = Path(bm_json["reference"])
                if not ref_p.is_absolute():
                    ref_p = REPO_ROOT / ref_p
                if ref_p.is_dir():
                    auarcs = {}
                    for outc in bm_json["outcomes"]:
                        run_p = Path(outc["run"])
                        if not run_p.is_absolute():
                            run_p = REPO_ROOT / run_p
                        if run_p.is_dir():
                            tr = diagnostics.anytime_trajectory(run_p, reference=ref_p)
                            self.assertIsNotNone(tr["auarc"])
                            auarcs[outc["method"]] = tr["auarc"]
                    if {"adaptive_full", "default", "random"} <= set(auarcs):
                        self.assertLess(auarcs["adaptive_full"], auarcs["random"])

    def test_paper_export_benchmark_and_n5_algorithm_study_with_anytime_and_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bm_dir = root / "bm_study"
            bm_dir.mkdir()
            cfgs = [_cls_config(lr0=lr) for lr in (0.001, 0.0003)]
            sources = {"dummy.py": "1"}
            ref_run = bm_dir / "reference" / "ref_run"
            _build_run(
                ref_run,
                configs=cfgs,
                training_seeds=[17],
                method="random",
                search_seed=123456,
                mode="offline_reference",
                budget_seconds=100.0,
                sources=sources,
                qualities={0: 0.6, 1: 0.9},
                times={0: 5.0, 1: 5.0},
                memories={0: 100.0, 1: 100.0},
                trial_events=[(0, 17, 5.0), (1, 17, 10.0)],
                task="detection",
                detection_metric="map50",
            )
            run_a = bm_dir / "adaptive_full" / "repeat_11" / "run_0"
            _build_run(
                run_a,
                configs=cfgs,
                training_seeds=[17],
                method="adaptive_full",
                search_seed=11,
                mode="online",
                budget_seconds=30.0,
                sources=sources,
                qualities={0: 0.6, 1: 0.9},
                times={0: 5.0, 1: 5.0},
                memories={0: 100.0, 1: 100.0},
                trial_events=[(0, 17, 4.0), (1, 17, 8.0)],
                task="detection",
                detection_metric="map50",
            )
            run_r = bm_dir / "random" / "repeat_11" / "run_0"
            _build_run(
                run_r,
                configs=cfgs,
                training_seeds=[17],
                method="random",
                search_seed=11,
                mode="online",
                budget_seconds=30.0,
                sources=sources,
                qualities={0: 0.6, 1: 0.9},
                times={0: 5.0, 1: 5.0},
                memories={0: 100.0, 1: 100.0},
                trial_events=[(0, 17, 12.0), (1, 17, 20.0)],
                task="detection",
                detection_metric="map50",
            )
            ref_proto = json.loads((ref_run / "protocol.json").read_text(encoding="utf-8"))
            manifest = {
                "sources": ref_proto["source_files"],
                "methods": ["adaptive_full", "random"],
                "search_seeds": [11],
                "training_seeds": [17],
                "test_used": False,
                "phase": "locked_before_search",
            }
            manifest["sha256"] = digest(manifest)
            write_json(bm_dir / "benchmark_protocol.json", manifest)
            write_json(bm_dir / "decisions.locked.json", {
                "benchmark_sha256": manifest["sha256"],
                "runs": [str(run_a), str(run_r)],
                "decision_hashes": {
                    str(run_a): digest(json.loads((run_a / "decision.locked.json").read_text(encoding="utf-8"))),
                    str(run_r): digest(json.loads((run_r / "decision.locked.json").read_text(encoding="utf-8"))),
                },
            })
            out_bm = root / "paper_bm"
            summary = paper_export.export_paper_artifacts(bm_dir, out_bm, lang="en")
            self.assertEqual(summary["kind"], "benchmark_study")
            for name in ("anytime_convergence.svg", "benchmark_regret.svg", "PROVENANCE.json", "PAPER_EVIDENCE.json"):
                self.assertTrue((out_bm / name).is_file(), name)
            ET.fromstring((out_bm / "anytime_convergence.svg").read_text(encoding="utf-8"))
            ET.fromstring((out_bm / "benchmark_regret.svg").read_text(encoding="utf-8"))
            self.assertIn("AUARC", (out_bm / "tables.tex").read_text(encoding="utf-8"))
            self.assertNotIn("—", (out_bm / "tables.tex").read_text(encoding="utf-8"))

            # Algorithm study with n>=5 paired comparisons -> renders Wilcoxon/Sign-flip/Holm in PAPER_SUMMARY.md
            alg_dir = root / "alg_n5"
            alg_dir.mkdir()
            outcomes = []
            for idx, sd in enumerate((11, 17, 23, 29, 37)):
                outcomes.append({"method": "adaptive_full", "search_seed": sd, "selected": "c01", "feasible": True, "reference_feasibility_disagreement": False, "objective_regret": 0.01 * idx, "quality_regret": 0.01, "exact_optimum_hit": True, "pareto_hit": True, "online_seconds": 10.0, "overrun_seconds": 0.0, "time_to_first_full_config_seconds": 4.0})
                outcomes.append({"method": "random", "search_seed": sd, "selected": "c00", "feasible": True, "reference_feasibility_disagreement": False, "objective_regret": 0.1 + 0.02 * idx, "quality_regret": 0.1, "exact_optimum_hit": False, "pareto_hit": False, "online_seconds": 15.0, "overrun_seconds": 0.0, "time_to_first_full_config_seconds": 8.0})
            paired = algorithm_study._paired(outcomes, {"seed": 2027, "samples": 100, "confidence": 0.95})
            alg_payload = {
                "schema_version": 1,
                "studies": [{
                    "study_id": "n5study01",
                    "outcomes": outcomes,
                    "method_summaries": algorithm_study._method_summary(outcomes, {"seed": 2027, "samples": 100, "confidence": 0.95}),
                    "paired_comparisons": paired,
                    "paired_nonparametric_scope": "Conditional on observed dataset, hardware, and paired search_seeds.",
                }],
            }
            alg_payload["sha256"] = digest(alg_payload)
            write_json(alg_dir / "ALGORITHM_STUDY.json", alg_payload)
            out_alg = root / "paper_alg"
            paper_export.export_paper_artifacts(alg_dir, out_alg, lang="en")
            ET.fromstring((out_alg / "anytime_convergence.svg").read_text(encoding="utf-8"))
            alg_md = (out_alg / "PAPER_SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn("Wilcoxon p", alg_md)
            self.assertIn("Sign-flip p", alg_md)
            self.assertIn("Holm adj. p", alg_md)

    def _assert_svg_texts_in_viewbox(self, svg_text: str):
        root = ET.fromstring(svg_text)
        width = float(root.attrib["width"])
        height = float(root.attrib["height"])
        for elem in root.iter():
            if not elem.tag.endswith("text"):
                continue
            if "transform" in elem.attrib:
                continue
            txt = elem.text or ""
            x = float(elem.attrib["x"])
            y = float(elem.attrib["y"])
            fs = float(elem.attrib.get("font-size", "11"))
            est_w = len(txt) * (fs * 0.62)
            anchor = elem.attrib.get("text-anchor", "start")
            if anchor == "start":
                x0, x1 = x, x + est_w
            elif anchor == "end":
                x0, x1 = x - est_w, x
            else:
                x0, x1 = x - est_w / 2.0, x + est_w / 2.0
            self.assertGreaterEqual(x0, 0.0, f"Left viewBox clip for {txt!r}: x0={x0:.1f} < 0 in width={width}")
            self.assertLessEqual(x1, width, f"Right viewBox clip for {txt!r}: x1={x1:.1f} > width={width}")
            self.assertGreaterEqual(y - fs, 0.0, f"Top viewBox clip for {txt!r}: y={y}")
            self.assertLessEqual(y, height, f"Bottom viewBox clip for {txt!r}: y={y} > height={height}")

    def test_exact_sign_flip_scale_invariant_and_float_cancellation(self):
        tiny_pos = algorithm_study._exact_sign_flip([1e-13] * 5)
        self.assertEqual(tiny_pos["status"], "ok")
        self.assertAlmostEqual(tiny_pos["p_value"], 0.0625, places=12)

        tiny_mixed = algorithm_study._exact_sign_flip([2e-13, 3e-13, -1e-13, 4e-13, 5e-13])
        self.assertEqual(tiny_mixed["status"], "ok")
        self.assertAlmostEqual(tiny_mixed["p_value"], 0.125, places=12)

        float_cancel = algorithm_study._exact_sign_flip([0.1 + 0.2 - 0.3, 1.0, -1.0, 0.25, -0.25])
        self.assertEqual(float_cancel["status"], "ok")
        self.assertAlmostEqual(float_cancel["p_value"], 1.0, places=12)

        zeros = algorithm_study._exact_sign_flip([0.0] * 5)
        self.assertEqual(zeros["status"], "all_zero_differences")
        self.assertEqual(zeros["p_value"], 1.0)

    def test_benchmark_anytime_svg_renders_reference_and_target_when_trajectory_prepopulated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bm_dir = root / "bm_v11"
            bm_dir.mkdir()
            cfgs = [_cls_config(lr0=lr) for lr in (0.001, 0.0003)]
            sources = {"dummy.py": "1"}
            obj_kw = {"time_scale": 10.0, "memory_scale": 100.0, "weights": [0.1, 0.6, 0.3]}
            ref_run = bm_dir / "reference" / "ref_run"
            _build_run(
                ref_run,
                configs=cfgs,
                training_seeds=[17],
                method="random",
                search_seed=123456,
                mode="offline_reference",
                budget_seconds=100.0,
                sources=sources,
                qualities={0: 0.2, 1: 0.5},
                times={0: 20.0, 1: 10.0},
                memories={0: 200.0, 1: 100.0},
                trial_events=[(0, 17, 10.0), (1, 17, 20.0)],
                implementation_version="0.11.0",
                task="detection",
                detection_metric="map50",
                acquisition_version=1,
                objective_kwargs=obj_kw,
            )
            run_a = bm_dir / "adaptive_full" / "repeat_17" / "run_0"
            _build_run(
                run_a,
                configs=cfgs,
                training_seeds=[17],
                method="adaptive_full",
                search_seed=17,
                mode="online",
                budget_seconds=40.0,
                sources=sources,
                qualities={0: 0.2, 1: 0.5},
                times={0: 20.0, 1: 10.0},
                memories={0: 200.0, 1: 100.0},
                trial_events=[(0, 17, 8.0), (1, 17, 18.0)],
                implementation_version="0.11.0",
                task="detection",
                detection_metric="map50",
                acquisition_version=1,
                objective_kwargs=obj_kw,
            )
            ref_proto = json.loads((ref_run / "protocol.json").read_text(encoding="utf-8"))
            manifest = {
                "sources": ref_proto["source_files"],
                "methods": ["adaptive_full"],
                "search_seeds": [17],
                "training_seeds": [17],
                "test_used": False,
                "phase": "locked_before_search",
            }
            manifest["sha256"] = digest(manifest)
            write_json(bm_dir / "benchmark_protocol.json", manifest)
            write_json(bm_dir / "decisions.locked.json", {
                "benchmark_sha256": manifest["sha256"],
                "runs": [str(run_a)],
                "decision_hashes": {
                    str(run_a): digest(json.loads((run_a / "decision.locked.json").read_text(encoding="utf-8"))),
                },
            })
            out_en = root / "out_en"
            out_vi = root / "out_vi"
            paper_export.export_paper_artifacts(bm_dir, out_en, lang="en")
            paper_export.export_paper_artifacts(bm_dir, out_vi, lang="vi")
            svg_en = (out_en / "anytime_convergence.svg").read_text(encoding="utf-8")
            svg_vi = (out_vi / "anytime_convergence.svg").read_text(encoding="utf-8")
            self.assertIn("Reference Optimum", svg_en)
            self.assertIn("95% Target Utility", svg_en)
            self.assertIn("Reference Optimum", svg_vi)
            self.assertIn("Mục tiêu 95% Utility", svg_vi)
            self._assert_svg_texts_in_viewbox(svg_en)
            self._assert_svg_texts_in_viewbox(svg_vi)

    def test_svg_viewbox_bounds_for_long_labels_and_boundary_ap50_and_synthetic_fallback(self):
        lc_svg = paper_export._svg_learning_curves(
            [
                ("c00 (0ebbc80d4a76_yolo11n_long_name)", [(1, 0.52), (2, 0.74)]),
                ("c01 (85a76fe86dd8_yolo11s_extra_model)", [(1, 0.61), (2, 0.85)]),
            ],
            "Val mAP50",
            lang="en",
        )
        self._assert_svg_texts_in_viewbox(lc_svg)

        pc_svg = paper_export._svg_per_class_ap50(
            [
                {"class_id": 0, "class_name": "very_long_endangered_species_name_01", "ap50_mean": 1.0, "ap50_sd": 0.01, "ap50_95_mean": 0.999},
                {"class_id": 1, "class_name": "extreme_" + "x" * 70, "ap50_mean": 0.85, "ap50_sd": 0.0, "ap50_95_mean": 0.62},
            ],
            lang="en",
        )
        self.assertIn("very_long_endangered_species_name_01", pc_svg)
        self.assertIn("1.000 / 0.999", pc_svg)
        self._assert_svg_texts_in_viewbox(pc_svg)

        pr_svg = paper_export._svg_pr_curve(
            {
                "recall_grid": [0.0, 0.5, 1.0],
                "mean_precision": [1.0, 0.9, 0.7],
                "per_class": [
                    {"class_id": 0, "class_name": "very_long_endangered_species_name_01", "precision": [1.0, 0.92, 0.75]},
                ],
            },
            lang="vi",
        )
        self._assert_svg_texts_in_viewbox(pr_svg)

        cm_svg = paper_export._svg_confusion_matrix(
            {
                "labels": ["very_long_endangered_species_name_01", "background"],
                "normalized": [[0.92, 0.08], [0.10, 0.90]],
            },
            lang="en",
        )
        self._assert_svg_texts_in_viewbox(cm_svg)

        any_fallback_svg = paper_export._svg_anytime_benchmark(
            {},
            [
                {"method": "adaptive_staged_seeds", "t_mean": 12.5, "regret": 0.02},
                {"method": "random", "t_mean": 20.0, "regret": 0.14},
            ],
            None,
            None,
            None,
            lang="en",
        )
        self.assertIn("Regret", any_fallback_svg)
        self.assertIn("0.0s", any_fallback_svg)
        self._assert_svg_texts_in_viewbox(any_fallback_svg)


if __name__ == "__main__":
    unittest.main()
