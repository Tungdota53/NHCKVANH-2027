"""Tier 5 adversarial & hardening regression tests (M5-A Iterations 2–4)."""

from contextlib import redirect_stdout
import csv
import http.client
import io
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import zlib

from research_pipeline.integrity import sha
from research_pipeline.v2 import acquisition, data, paper_export, predictors, selection, terminal
from research_pipeline.v2.ledger import Ledger, read_events
from research_pipeline.v2.report import make_report
from research_pipeline.v2.schema import lock_protocol, measurements
from research_pipeline.v2.web_ui import create_server, data_access
from yolo_tuner.pipeline import write_json
from yolo_tuner.research import Objective, digest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_png_bytes(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Create a minimal valid RGB PNG without external dependencies."""
    r, g, b = rgb
    raw_row = b"\x00" + bytes([r, g, b]) * width
    raw = raw_row * height

    def chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _build_audited_run_with_quality_score_curves(run_dir: Path, *, task: str) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = run_dir / "source_snapshot"
    snap_dir.mkdir(parents=True, exist_ok=True)
    src_file = snap_dir / "worker.py"
    src_file.write_text("# snapshot\n", encoding="utf-8")
    sources = {"worker.py": sha(src_file)}

    ds_manifest = {"task": task, "source": "synthetic", "counts": {"train": 16, "val": 8, "test": 8}, "splits": {}}
    write_json(run_dir / "dataset.json", ds_manifest)
    write_json(run_dir / "subset.json", ds_manifest)
    ds_digest = digest(ds_manifest)
    ds_sha = sha(run_dir / "dataset.json")

    configs = [
        {
            "model": "tiny_cnn" if task == "classification" else "tiny_cap",
            "imgsz": 32,
            "batch": 8,
            "accumulation": 1,
            "precision": "fp32",
            "optimizer": "AdamW",
            "lr0": 0.001,
            "weight_decay": 0.0005,
            "width": 16,
        },
        {
            "model": "tiny_cnn" if task == "classification" else "tiny_cap",
            "imgsz": 32,
            "batch": 8,
            "accumulation": 1,
            "precision": "fp32",
            "optimizer": "AdamW",
            "lr0": 0.002,
            "weight_decay": 0.0005,
            "width": 24,
        },
    ]
    raw_proto = {
        "implementation_version": "1.0.0",
        "mode": "online",
        "method": "adaptive_full",
        "execution_policy": {
            "version": 2,
            "integrity": "stat_cached_between_trials_full_hash_boundaries",
            "budget": "end_to_end_soft_deadline_with_measured_finalization_reserve",
            "data_loader": {"workers": 0, "cache_mb": 0.0, "pin_memory": "off"},
        },
        "selection_rule": "weighted",
        "run_directory": str(run_dir),
        "task": task,
        "dataset_source": "synthetic",
        "dataset_sha256": ds_digest,
        "dataset_content_id": f"tier5-{task}",
        "subset_sha256": ds_digest,
        "hardware": {
            "device": "cpu",
            "host_label": "tier5-host",
            "cpu": "tier5-cpu",
            "torch_version": "2.5",
            "driver_version": None,
            "ml_runtime_ready": True,
            "ram_available_mb": 4096.0,
        },
        "runtime_versions": {"python": "3.11", "torch": "2.5"},
        "configs": configs,
        "training_seeds": [17],
        "split_seed": 101,
        "search_seed": 17,
        "full_epochs": 2,
        "probe_epochs": 1,
        "trial_timeout": 120.0,
        "budget_seconds": 120.0,
        "objective": {
            "name": "balanced",
            "min_quality": 0.0,
            "max_time": None,
            "max_memory": 2048.0,
            "time_scale": 60.0,
            "memory_scale": 512.0,
            "weights": [0.6, 0.25, 0.15],
        },
        "example_counts": {"train": 16, "val": 8, "test": 8},
        "quality_definition": "final epoch validation accuracy" if task == "classification" else "best validation generated (BLEU4+chrF)/200 within declared horizon",
        "determinism_policy": "tier5",
        "memory_kind": "process_tree_rss_mib_not_vram",
        "scheduler": {
            "policy_version": 3,
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
        "test_policy": "never used for selection",
        "development_history": "tier5 test",
        "reference_boundary": None,
        "reference_boundary_sha256": None,
        "acquisition": {
            "version": 2,
            "kernel_ridge": 0.05,
            "length_scale": 1.0,
            "exploration": 0.25,
            "novelty_weight": 0.05,
        },
    }
    protocol = lock_protocol(raw_proto)
    write_json(run_dir / "protocol.json", protocol)

    clock_now = [0.0]
    ledger = Ledger(run_dir / "events.jsonl", clock=lambda: clock_now[0], origin=0.0)
    ledger.emit("session_started", mode="online", method="adaptive_full")

    results = {"0": [], "1": []}
    for idx, q_final in ((0, 0.72), (1, 0.84)):
        aid = f"attempt_{idx + 1:05d}"
        t_dir = run_dir / f"trial_{idx + 1:05d}"
        t_dir.mkdir(parents=True, exist_ok=True)
        ckpt = t_dir / "best.pt"
        ckpt.write_bytes(f"ckpt-{idx}".encode("utf-8"))
        req = {
            "attempt_id": aid,
            "index": idx,
            "config": configs[idx],
            "seed": 17,
            "trial_epochs": 2,
            "full_epochs": 2,
            "full_data": True,
            "mode": "train",
            "resume": None,
            "manifest": str(run_dir / "dataset.json"),
            "dataset": str(run_dir / "dataset.json"),
            "dataset_sha256": ds_sha,
            "protocol_sha256": protocol["protocol_sha256"],
            "source_files": sources,
            "data_loader": protocol["execution_policy"]["data_loader"],
        }
        req_path = t_dir / "request.json"
        write_json(req_path, req)
        # Emit ONLY quality_score (mirroring vision_worker.py:218 & caption_worker.py:234)
        lc = [
            {"epoch": 1, "validation_loss": 0.61, "quality_score": round(q_final * 0.75, 4)},
            {"epoch": 2, "validation_loss": 0.42, "quality_score": q_final},
        ]
        res = {
            "attempt_id": aid,
            "index": idx,
            "seed": 17,
            "full": True,
            "mode": "train",
            "status": "ok",
            "epochs_completed": 2,
            "optimizer_steps": 12,
            "training_loss_finite": True,
            "actual_config_verified": True,
            "checkpoint_sha256": sha(ckpt),
            "best_checkpoint": str(ckpt),
            "best_checkpoint_epoch": 2,
            "best_epoch": 2,
            "quality": q_final,
            "quality_score": q_final,
            "memory_mb": 256.0 + idx * 32.0,
            "worker_wall_seconds": 3.0 + idx * 1.5,
            "measured_loop_seconds": 2.4 + idx * 1.2,
            "learning_curve": lc,
            "trial_directory": str(t_dir),
            "request_path": str(req_path),
            "request_sha256": sha(req_path),
            "result_path": str(t_dir / "result.json"),
            "directory": str(t_dir),
        }
        res_path = t_dir / "result.json"
        write_json(res_path, res)
        res["raw_sha256"] = sha(res_path)
        parent_path = t_dir / "parent_result.json"
        write_json(parent_path, res)

        clock_now[0] += 1.0
        ledger.emit("trial_started", attempt_id=aid, index=idx, seed=17, full=True)
        clock_now[0] += 3.0 + idx * 1.5
        ledger.emit(
            "trial_completed",
            attempt_id=aid,
            parent_sha256=sha(parent_path),
            worker_seconds=res["worker_wall_seconds"],
            status="ok",
            index=idx,
            seed=17,
            full=True,
            directory=str(t_dir),
        )
        results[str(idx)].append(res)

    state = {"results": results, "probes": [], "acquisition_history": []}
    rows = measurements(configs, results, [17], 2, Objective(**raw_proto["objective"]))
    dec = selection.select(rows, protocol)
    write_json(run_dir / "state.json", state)
    write_json(run_dir / "decision.locked.json", dec)
    write_json(run_dir / "status.json", {"phase": "complete", "protocol": protocol["protocol_sha256"]})
    ledger.emit(
        "selection_committed",
        state_path="state.json",
        state_sha256=digest(state),
        decision=dec,
        session_seconds=max(1.0, float(clock_now[0])),
    )
    report = make_report(protocol, state, read_events(run_dir / "events.jsonl"))
    write_json(run_dir / "report.json", report)
    return run_dir


class _MockPopen:
    def __init__(self, pid: int = 424242):
        self.pid = pid

    def poll(self):
        return None


class Tier5AdversarialTests(unittest.TestCase):
    def test_paper_export_classification_and_captioning_learning_curves_populated_from_quality_score(self):
        for task in ("classification", "captioning"):
            with self.subTest(task=task):
                with tempfile.TemporaryDirectory() as td:
                    td_path = Path(td)
                    run_dir = _build_audited_run_with_quality_score_curves(td_path / f"run_{task}", task=task)
                    out_dir = td_path / f"paper_{task}"
                    summary = paper_export.export_paper_artifacts(run_dir, out_dir, lang="en")
                    self.assertTrue((out_dir / "learning_curves.csv").is_file())
                    self.assertTrue((out_dir / "learning_curves.svg").is_file())

                    with (out_dir / "learning_curves.csv").open("r", encoding="utf-8", newline="") as fh:
                        rows = list(csv.DictReader(fh))
                    self.assertEqual(len(rows), 4)
                    for row in rows:
                        val = float(row["primary_quality"])
                        self.assertGreater(val, 0.4)

                    svg_text = (out_dir / "learning_curves.svg").read_text(encoding="utf-8")
                    self.assertNotIn("No full-training learning curves available to render.", svg_text)
                    self.assertIn("<polyline", svg_text)
                    ET.fromstring(svg_text)
                    self.assertIn("learning_curves.svg", summary["exported_files"])

    def test_charts_js_reads_active_seconds_and_quality_score_and_app_js_handles_multistudy(self):
        charts_js = (REPO_ROOT / "research_pipeline" / "v2" / "web_ui" / "static" / "charts.js").read_text(encoding="utf-8")
        app_js = (REPO_ROOT / "research_pipeline" / "v2" / "web_ui" / "static" / "app.js").read_text(encoding="utf-8")

        # Anytime convergence reads active_seconds first
        self.assertIn("p.active_seconds ?? p.t_seconds ?? p.time_seconds ?? 0", charts_js)
        # Learning curves fall back to quality_score for classification/captioning
        self.assertIn("finite(pt[yKey]) ? pt[yKey] : finite(pt.quality_score) ? pt.quality_score : finite(pt.quality) ? pt.quality : null", charts_js)
        self.assertIn("finite(pt[yKey]) ? pt[yKey] : finite(pt.quality_score) ? pt.quality_score : finite(pt.quality) ? pt.quality : 0", charts_js)

        # View 5 iterates across all st.studies entries instead of hardcoding st.studies[0]
        self.assertNotIn("st.studies[0]", app_js)
        self.assertIn("studies.forEach", app_js)
        # Benchmark preview mirrors selected #launch-method
        self.assertIn("p.methods", app_js)
        self.assertIn("selectedMethod", app_js)

    def test_terminal_candidate_status_confirmed_constraint_failure_and_oom_badges(self):
        self.assertEqual(terminal._candidate_status({"state": "confirmed_constraint_failure", "completed_seeds": [17, 29], "runs": []}), ("FAIL", "danger"))
        self.assertEqual(terminal._candidate_status({"state": "infeasible", "completed_seeds": [17], "runs": []}), ("FAIL", "danger"))
        self.assertEqual(terminal._candidate_status({"state": "observed_required_seed_oom", "completed_seeds": [], "runs": []}), ("OOM", "danger"))
        self.assertEqual(terminal._candidate_status({"state": "confirmed_feasible", "completed_seeds": [17], "runs": []}), ("OK", "success"))
        self.assertEqual(terminal._candidate_status({"state": "unresolved", "completed_seeds": [17], "runs": []}), ("PARTIAL", "warn"))
        self.assertEqual(terminal._candidate_status({"state": "unresolved", "completed_seeds": [], "runs": []}), ("UNRES", "muted"))

    def test_webui_launch_rejects_flag_injection_bools_nonfinite_and_invalid_methods(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            runs_dir = td_path / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            data_yaml = td_path / "ds.yaml"
            data_yaml.write_text("names: [a]\n", encoding="utf-8")

            server, _, csrf = create_server(host="127.0.0.1", port=0, runs_dir=runs_dir)
            port = server.server_address[1]
            t = threading.Thread(target=server.serve_forever, daemon=True)
            t.start()
            try:
                def post_launch(payload_obj):
                    body = json.dumps(payload_obj, allow_nan=True).encode("utf-8")
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                    conn.request(
                        "POST",
                        "/api/launch",
                        body=body,
                        headers={"Content-Type": "application/json", "X-CSRF-Token": csrf.token},
                    )
                    resp = conn.getresponse()
                    raw = resp.read().decode("utf-8", "replace")
                    conn.close()
                    return resp.status, json.loads(raw)

                base = {
                    "confirmed": True,
                    "mode": "optimize",
                    "data": str(data_yaml),
                    "device": "cpu",
                    "space": "core",
                    "method": "adaptive_full",
                    "selection_rule": "weighted",
                    "detection_metric": "map50",
                    "full_epochs": 5,
                    "budget_seconds": 120.0,
                    "seeds": [17, 29],
                    "weights": [0.6, 0.25, 0.15],
                    "output": "valid_out",
                }

                with patch.object(data_access.subprocess, "Popen", return_value=_MockPopen()) as popen_mock:
                    code, ok_res = post_launch(base)
                    self.assertEqual(code, 200, ok_res)
                    self.assertTrue(ok_res["launched"])
                    self.assertEqual(popen_mock.call_count, 1)

                    bm_ok = {
                        **base,
                        "mode": "benchmark",
                        "methods": ["adaptive_full", "adaptive_ehvi", "random"],
                        "search_seeds": [17, 29, 43],
                        "reference_budget_seconds": 300.0,
                        "output": "valid_bm",
                    }
                    code_bm, bm_res = post_launch(bm_ok)
                    self.assertEqual(code_bm, 200, bm_res)
                    self.assertEqual(popen_mock.call_count, 2)

                    bad_cases = [
                        ("device_flag", {**base, "device": "--help"}),
                        ("device_invalid", {**base, "device": "cuda:abc"}),
                        ("device_int", {**base, "device": 0}),
                        ("models_flag", {**base, "models": ["--budget-seconds", "99999"]}),
                        ("models_str_flag", {**base, "models": "--help"}),
                        ("models_bool", {**base, "models": True}),
                        ("models_empty_entry", {**base, "models": [""]}),
                        ("models_whitespace_list_entry", {**base, "models": ["   "]}),
                        ("models_comma_only", {**base, "models": ","}),
                        ("models_semicolon_only", {**base, "models": ";"}),
                        ("models_mixed_delims_spaces", {**base, "models": " ; , "}),
                        ("epochs_bool_true", {**base, "full_epochs": True}),
                        ("epochs_bool_false", {**base, "full_epochs": False}),
                        ("epochs_zero", {**base, "full_epochs": 0}),
                        ("epochs_float", {**base, "full_epochs": 2.5}),
                        ("budget_bool_true", {**base, "budget_seconds": True}),
                        ("budget_bool_false", {**base, "budget_seconds": False}),
                        ("budget_neg", {**base, "budget_seconds": -10}),
                        ("budget_inf", {**base, "budget_seconds": float("inf")}),
                        ("seeds_dup", {**base, "seeds": [17, 17]}),
                        ("seeds_empty", {**base, "seeds": []}),
                        ("seeds_bool_elem", {**base, "seeds": [True, 17]}),
                        ("seeds_bool", {**base, "seeds": True}),
                        ("weights_bool_elem", {**base, "weights": [True, 0.25, 0.15]}),
                        ("weights_zero_sum", {**base, "weights": [0.0, 0.0, 0.0]}),
                        ("weights_len", {**base, "weights": [0.5, 0.5]}),
                        ("task_flag", {**base, "task": "--bad"}),
                        ("bm_bad_method", {**bm_ok, "methods": ["not_a_real_method"]}),
                        ("bm_dup_method", {**bm_ok, "methods": ["adaptive_full", "adaptive_full"]}),
                        ("bm_empty_methods", {**bm_ok, "methods": []}),
                        ("bm_dup_search_seeds", {**bm_ok, "search_seeds": [17, 17]}),
                        ("bm_bool_search_seeds", {**bm_ok, "search_seeds": [False]}),
                        ("bm_bool_ref_budget", {**bm_ok, "reference_budget_seconds": True}),
                        ("bm_neg_ref_budget", {**bm_ok, "reference_budget_seconds": -5}),
                        ("output_bool_true", {**base, "output": True}),
                        ("output_bool_false", {**base, "output": False}),
                        ("output_dict", {**base, "output": {"a": 1}}),
                        ("output_int", {**base, "output": 123}),
                        ("output_list", {**base, "output": ["x"]}),
                        ("output_flag", {**base, "output": "-bad"}),
                        ("grid_bool_true", {**base, "grid": True}),
                        ("grid_bool_false", {**base, "grid": False}),
                        ("grid_int", {**base, "grid": 123}),
                        ("grid_list", {**base, "grid": ["x"]}),
                        ("grid_dict", {**base, "grid": {"a": 1}}),
                        ("grid_flag", {**base, "grid": "-bad"}),
                        ("data_bool", {**base, "data": True}),
                        ("data_dict", {**base, "data": {"a": 1}}),
                        ("data_int", {**base, "data": 123}),
                        ("data_flag", {**base, "data": "-bad"}),
                        ("mode_bool_false", {**base, "mode": False}),
                        ("space_bool_false", {**base, "space": False}),
                        ("method_bool_false", {**base, "method": False}),
                        ("rule_bool_false", {**base, "selection_rule": False}),
                        ("metric_bool_false", {**base, "detection_metric": False}),
                        ("unconfirmed", {**base, "confirmed": False}),
                    ]
                    for label, payload in bad_cases:
                        with self.subTest(case=label):
                            bad_code, bad_res = post_launch(payload)
                            self.assertEqual(bad_code, 400, f"{label} returned {bad_code}: {bad_res}")
                    # Valid explicit grid file path still succeeds
                    grid_json = td_path / "grid.json"
                    grid_json.write_text("[]\n", encoding="utf-8")
                    code_grid, grid_res = post_launch({**base, "grid": str(grid_json), "output": "valid_grid_out"})
                    self.assertEqual(code_grid, 200, grid_res)
                    self.assertTrue(grid_res["launched"])
                    # Whitespace-only or empty-list models cleanly omit --models (no dangling --models flag)
                    code_ws_models, ws_res = post_launch({**base, "models": "   ", "output": "valid_ws_models"})
                    self.assertEqual(code_ws_models, 200, ws_res)
                    self.assertNotIn("--models", popen_mock.call_args.args[0])
                    code_empty_models, empty_res = post_launch({**base, "models": [], "output": "valid_empty_models"})
                    self.assertEqual(code_empty_models, 200, empty_res)
                    self.assertNotIn("--models", popen_mock.call_args.args[0])
                    code_list_models, list_res = post_launch({**base, "models": "yolo11n.pt; yolo11s.pt", "output": "valid_multi_models"})
                    self.assertEqual(code_list_models, 200, list_res)
                    launched_cmd = popen_mock.call_args.args[0]
                    m_idx = launched_cmd.index("--models")
                    self.assertEqual(launched_cmd[m_idx + 1 : m_idx + 3], ["yolo11n.pt", "yolo11s.pt"])
                    # Ensure zero invalid payloads ever reached Popen (only base, bm_ok, valid_grid_out, and 3 valid models cases)
                    self.assertEqual(popen_mock.call_count, 6)
            finally:
                server.shutdown()
                server.server_close()

    def test_dedup_dataset_same_stem_different_extension_preserves_distinct_labels(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src = td_path / "src_ds"
            dst = td_path / "dst_ds"
            for rel in ("images/train/sub1", "images/train/sub2", "images/val", "labels/train/sub1", "labels/train/sub2", "labels/val"):
                (src / rel).mkdir(parents=True, exist_ok=True)

            # Distinct image bytes & distinct labels with identical stem "dup" and different extensions / casing
            (src / "images/train/sub1/dup.png").write_bytes(_make_png_bytes(8, 8, (10, 20, 30)))
            (src / "labels/train/sub1/dup.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            (src / "images/train/sub2/dup.jpg").write_bytes(_make_png_bytes(8, 8, (200, 40, 10)))
            (src / "labels/train/sub2/dup.txt").write_text("1 0.3 0.3 0.2 0.2\n", encoding="utf-8")
            (src / "images/val/val_only.png").write_bytes(_make_png_bytes(8, 8, (30, 180, 90)))
            (src / "labels/val/val_only.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            (src / "data.yaml").write_text(
                f"path: {src.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: cat\n  1: dog\n",
                encoding="utf-8",
            )

            res = data.dedup_dataset(src / "data.yaml", dst, task="detection")
            self.assertEqual(res["kept_count"], 3)
            train_imgs = sorted((dst / "images" / "train").iterdir())
            train_lbls = sorted((dst / "labels" / "train").iterdir())
            self.assertEqual(len(train_imgs), 2)
            self.assertEqual(len(train_lbls), 2)
            self.assertEqual({p.stem for p in train_imgs}, {p.stem for p in train_lbls})
            lbl_contents = {p.read_text(encoding="utf-8").strip() for p in train_lbls}
            self.assertEqual(lbl_contents, {"0 0.5 0.5 0.4 0.4", "1 0.3 0.3 0.2 0.2"})

    def test_predictors_observed_cross_cost_ignores_bool_loop_and_wall_seconds(self):
        cfg = {
            "model": "tiny_cnn",
            "imgsz": 32,
            "batch": 8,
            "accumulation": 1,
            "precision": "fp32",
            "optimizer": "AdamW",
            "lr0": 0.001,
            "weight_decay": 0.0005,
            "width": 16,
        }
        proto = {"task": "classification", "configs": [cfg], "hardware": {"device": "cpu"}}
        fulls_bool_only = {0: [{"worker_wall_seconds": 5.0, "measured_loop_seconds": True}]}
        self.assertIsNone(predictors.observed_cross_cost(proto, fulls_bool_only, 0))

        fulls_bool_wall = {0: [{"worker_wall_seconds": True, "measured_loop_seconds": 0.5}]}
        self.assertIsNone(predictors.observed_cross_cost(proto, fulls_bool_wall, 0))

        fulls_mixed = {
            0: [
                {"worker_wall_seconds": 10.0, "measured_loop_seconds": True},
                {"worker_wall_seconds": 10.0, "measured_loop_seconds": 8.0},
            ]
        }
        est = predictors.observed_cross_cost(proto, fulls_mixed, 0)
        self.assertIsNotNone(est)
        self.assertAlmostEqual(est["cost_components"]["reference_loop_seconds"], 8.0, places=6)
        self.assertAlmostEqual(est["cost_components"]["observed_non_loop_overhead_seconds"], 2.0, places=6)

    def test_webui_security_traversal_csrf_and_multistudy_benchmark_api(self):
        with tempfile.TemporaryDirectory() as td:
            runs_dir = Path(td) / "runs"
            study_dir = runs_dir / "alg_study_multi"
            study_dir.mkdir(parents=True, exist_ok=True)
            (study_dir / "ALGORITHM_STUDY.json").write_text(
                json.dumps({
                    "schema_version": 2,
                    "studies": [
                        {
                            "study": "runs/study_alpha",
                            "study_id": "alpha123",
                            "domain": {"task": "detection"},
                            "method_summaries": [{"method": "adaptive_full", "search_repetitions": 5, "feasibility_rate": 1.0, "conditional_objective_regret": {"mean": 0.01}, "online_seconds": {"mean": 45.0}}],
                            "paired_comparisons": [],
                        },
                        {
                            "study": "runs/study_beta",
                            "study_id": "beta456",
                            "domain": {"task": "classification"},
                            "method_summaries": [{"method": "adaptive_ehvi", "search_repetitions": 5, "feasibility_rate": 1.0, "conditional_objective_regret": {"mean": 0.02}, "online_seconds": {"mean": 32.0}}],
                            "paired_comparisons": [],
                        },
                    ],
                }),
                encoding="utf-8",
            )
            (runs_dir / "secret.json").write_text('{"secret": 1}', encoding="utf-8")

            server, _, _ = create_server(host="127.0.0.1", port=0, runs_dir=runs_dir)
            port = server.server_address[1]
            t = threading.Thread(target=server.serve_forever, daemon=True)
            t.start()
            try:
                for bad_path in (
                    "/api/file?path=%5C%5C127.0.0.1%5Cc%24%5Csecret.json",
                    "/api/file?path=C:secret.json",
                    "/api/file?path=%252e%252e/secret.json",
                    "/api/file?path=alg_study_multi/ALGORITHM_STUDY.json%00.txt",
                ):
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                    conn.request("GET", bad_path)
                    resp = conn.getresponse()
                    resp.read()
                    conn.close()
                    self.assertIn(resp.status, (400, 403), f"Expected 400/403 for {bad_path}, got {resp.status}")

                # POST without CSRF token must return 403
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                conn.request("POST", "/api/launch", body=b"{}", headers={"Content-Type": "application/json"})
                resp = conn.getresponse()
                resp.read()
                conn.close()
                self.assertEqual(resp.status, 403)

                # Multi-study benchmark endpoint preserves all studies
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                conn.request("GET", "/api/benchmark?path=alg_study_multi")
                resp = conn.getresponse()
                bm_payload = json.loads(resp.read().decode("utf-8"))
                conn.close()
                self.assertEqual(resp.status, 200)
                self.assertEqual(bm_payload["kind"], "algorithm_study")
                self.assertEqual(len(bm_payload["algorithm_study"]["studies"]), 2)
            finally:
                server.shutdown()
                server.server_close()

    def test_objective_weight_sensitivity_and_zero_template_bias_across_datasets(self):
        configs = [
            {"model": "yolo11n.pt", "imgsz": 64, "batch": 4, "accumulation": 1, "precision": "fp32", "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.0005, "freeze": 0},
            {"model": "yolo11n.pt", "imgsz": 128, "batch": 8, "accumulation": 1, "precision": "fp32", "optimizer": "AdamW", "lr0": 0.002, "weight_decay": 0.0005, "freeze": 5},
            {"model": "yolo11s.pt", "imgsz": 192, "batch": 8, "accumulation": 1, "precision": "fp32", "optimizer": "SGD", "lr0": 0.01, "weight_decay": 0.0005, "freeze": 0},
        ]
        # Candidate 0 is fast & light, Candidate 2 is highest quality & heavier
        rows_ds_a = [
            {"index": 0, "config_id": "c0", "config": configs[0], "complete": True, "state": "confirmed_feasible", "quality_mean": 0.70, "time_mean": 5.0, "memory_max": 200.0, "utility": Objective(weights=[0.9, 0.05, 0.05], time_scale=60.0, memory_scale=1024.0).value({"quality_mean": 0.70, "time_mean": 5.0, "memory_max": 200.0})},
            {"index": 2, "config_id": "c2", "config": configs[2], "complete": True, "state": "confirmed_feasible", "quality_mean": 0.92, "time_mean": 55.0, "memory_max": 950.0, "utility": Objective(weights=[0.9, 0.05, 0.05], time_scale=60.0, memory_scale=1024.0).value({"quality_mean": 0.92, "time_mean": 55.0, "memory_max": 950.0})},
        ]
        proto_quality = {"selection_rule": "weighted", "scheduler": {"policy_version": 3}, "objective": {"weights": [0.9, 0.05, 0.05], "time_scale": 60.0, "memory_scale": 1024.0}}
        proto_fast = {"selection_rule": "weighted", "scheduler": {"policy_version": 3}, "objective": {"weights": [0.1, 0.45, 0.45], "time_scale": 60.0, "memory_scale": 1024.0}}
        obj_fast = Objective(weights=[0.1, 0.45, 0.45], time_scale=60.0, memory_scale=1024.0)
        rows_fast = [{**r, "utility": obj_fast.value(r)} for r in rows_ds_a]
        self.assertEqual(selection.select(rows_ds_a, proto_quality)["selected_index"], 2)
        self.assertEqual(selection.select(rows_fast, proto_fast)["selected_index"], 0)

        # Zero template bias: flipping observed dataset outcomes flips acquisition ranking
        base_proto = {
            "task": "detection",
            "method": "adaptive_ehvi",
            "selection_rule": "weighted",
            "configs": configs,
            "training_seeds": [17, 29],
            "search_seed": 17,
            "full_epochs": 4,
            "budget_seconds": 120.0,
            "objective": {"name": "balanced", "min_quality": 0.0, "max_time": 120.0, "max_memory": 2048.0, "time_scale": 60.0, "memory_scale": 1024.0, "weights": [0.6, 0.25, 0.15]},
            "acquisition": {"version": 2, "kernel_ridge": 0.05, "length_scale": 1.0, "exploration": 0.25, "novelty_weight": 0.05},
        }
        obj_bm = Objective(**base_proto["objective"])
        state_a = {
            "0": [{"seed": 17, "mode": "train", "status": "ok", "epochs_completed": 4, "optimizer_steps": 8, "training_loss_finite": True, "actual_config_verified": True, "checkpoint_sha256": "a", "quality": 0.90, "worker_wall_seconds": 8.0, "memory_mb": 200.0},
                  {"seed": 29, "mode": "train", "status": "ok", "epochs_completed": 4, "optimizer_steps": 8, "training_loss_finite": True, "actual_config_verified": True, "checkpoint_sha256": "b", "quality": 0.91, "worker_wall_seconds": 8.0, "memory_mb": 200.0}],
            "2": [{"seed": 17, "mode": "train", "status": "ok", "epochs_completed": 4, "optimizer_steps": 8, "training_loss_finite": True, "actual_config_verified": True, "checkpoint_sha256": "c", "quality": 0.40, "worker_wall_seconds": 25.0, "memory_mb": 800.0},
                  {"seed": 29, "mode": "train", "status": "ok", "epochs_completed": 4, "optimizer_steps": 8, "training_loss_finite": True, "actual_config_verified": True, "checkpoint_sha256": "d", "quality": 0.42, "worker_wall_seconds": 25.0, "memory_mb": 800.0}],
        }
        state_b = {
            "0": [{"seed": 17, "mode": "train", "status": "ok", "epochs_completed": 4, "optimizer_steps": 8, "training_loss_finite": True, "actual_config_verified": True, "checkpoint_sha256": "a", "quality": 0.35, "worker_wall_seconds": 8.0, "memory_mb": 200.0},
                  {"seed": 29, "mode": "train", "status": "ok", "epochs_completed": 4, "optimizer_steps": 8, "training_loss_finite": True, "actual_config_verified": True, "checkpoint_sha256": "b", "quality": 0.37, "worker_wall_seconds": 8.0, "memory_mb": 200.0}],
            "2": [{"seed": 17, "mode": "train", "status": "ok", "epochs_completed": 4, "optimizer_steps": 8, "training_loss_finite": True, "actual_config_verified": True, "checkpoint_sha256": "c", "quality": 0.93, "worker_wall_seconds": 12.0, "memory_mb": 400.0},
                  {"seed": 29, "mode": "train", "status": "ok", "epochs_completed": 4, "optimizer_steps": 8, "training_loss_finite": True, "actual_config_verified": True, "checkpoint_sha256": "d", "quality": 0.94, "worker_wall_seconds": 12.0, "memory_mb": 400.0}],
        }
        rows_a = measurements(configs, state_a, [17, 29], 4, obj_bm)
        rows_b = measurements(configs, state_b, [17, 29], 4, obj_bm)
        forecasts = {1: {"point_seconds": 10.0, "planning_seconds": 10.0, "predicted_memory_mb": 300.0}}
        rank_a = acquisition.rank_candidates(base_proto, rows_a, forecasts, candidates=[1])
        rank_b = acquisition.rank_candidates(base_proto, rows_b, forecasts, candidates=[1])
        pred_a = rank_a["scores"]["1"]["quality_mean_prediction"]
        pred_b = rank_b["scores"]["1"]["quality_mean_prediction"]
        self.assertNotAlmostEqual(pred_a, pred_b, places=3)

    def test_charts_js_anytime_negative_and_one_epoch_learning_curve_rendering(self):
        charts_path = REPO_ROOT / "research_pipeline" / "v2" / "web_ui" / "static" / "charts.js"
        charts_js = charts_path.read_text(encoding="utf-8")
        # renderAnytimeChart must not clamp Y-axis to [0, 1.02]
        anytime_block = charts_js.split("function renderAnytimeChart", 1)[1].split("function renderPerClassChart", 1)[0]
        self.assertNotIn("Math.max(0, Math.min(...utils)", anytime_block)
        self.assertNotIn("Math.min(1.02, Math.max(...utils)", anytime_block)
        # renderLearningCurvesChart must emit <circle> markers for single-epoch runs
        lc_block = charts_js.split("function renderLearningCurvesChart", 1)[1].split("function renderSurrogateTraceChart", 1)[0]
        self.assertIn("<circle", lc_block)

        node_bin = shutil.which("node")
        if node_bin:
            script = """
const fs = require("fs");
const vm = require("vm");
const code = fs.readFileSync(process.argv[1], "utf8");
const sandbox = { window: {} };
vm.runInNewContext(code, sandbox);
const C = sandbox.window.NCKHCharts;
const elNeg = { innerHTML: "" };
C.renderAnytimeChart(elNeg, {
  trajectory: [
    { active_seconds: 5.0, best_utility: -0.765 },
    { active_seconds: 12.0, best_utility: -0.250 },
  ],
});
const elHigh = { innerHTML: "" };
C.renderAnytimeChart(elHigh, {
  trajectory: [
    { active_seconds: 5.0, best_utility: 1.50 },
    { active_seconds: 12.0, best_utility: 2.20 },
  ],
});
const elLc1 = { innerHTML: "" };
C.renderLearningCurvesChart(
  elLc1,
  { "0": [{ mode: "train", seed: 17, learning_curve: [{ epoch: 1, quality_score: 0.72 }] }] },
  0,
  "classification"
);
process.stdout.write(JSON.stringify({ neg: elNeg.innerHTML, high: elHigh.innerHTML, lc1: elLc1.innerHTML }));
"""
            proc = subprocess.run(
                [node_bin, "-e", script, str(charts_path)],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )
            out = json.loads(proc.stdout)
            for key in ("neg", "high"):
                svg_root = ET.fromstring(out[key])
                circles = svg_root.findall(".//circle")
                self.assertEqual(len(circles), 2)
                for c in circles:
                    cy = float(c.attrib["cy"])
                    self.assertGreaterEqual(cy, 20.0)
                    self.assertLessEqual(cy, 256.0)
                y_ticks = [t.text for t in svg_root.findall(".//text") if t.attrib.get("text-anchor") == "end"]
                self.assertEqual(len(y_ticks), 5)
                self.assertGreater(len(set(y_ticks)), 1, f"Collapsed Y ticks for {key}: {y_ticks}")

            lc_root = ET.fromstring(out["lc1"])
            lc_circles = lc_root.findall(".//circle")
            self.assertEqual(len(lc_circles), 1)
            cx = float(lc_circles[0].attrib["cx"])
            cy = float(lc_circles[0].attrib["cy"])
            self.assertGreaterEqual(cx, 58.0)
            self.assertLessEqual(cx, 616.0)
            self.assertGreaterEqual(cy, 20.0)
            self.assertLessEqual(cy, 276.0)

    def test_webui_paper_export_special_chars_urls_resolve_http_200(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            runs_dir = td_path / "runs"
            _build_audited_run_with_quality_score_curves(runs_dir / "run_alpha", task="classification")

            server, _, csrf = create_server(host="127.0.0.1", port=0, runs_dir=runs_dir)
            port = server.server_address[1]
            t = threading.Thread(target=server.serve_forever, daemon=True)
            t.start()
            try:
                body = json.dumps({
                    "run": "run_alpha",
                    "output": "paper_exports/exp+1#2&3 4",
                    "lang": "en",
                }).encode("utf-8")
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
                conn.request(
                    "POST",
                    "/api/paper-export",
                    body=body,
                    headers={"Content-Type": "application/json", "X-CSRF-Token": csrf.token},
                )
                resp = conn.getresponse()
                payload = json.loads(resp.read().decode("utf-8"))
                conn.close()
                self.assertEqual(resp.status, 200, payload)
                self.assertTrue(payload["files"])

                for item in payload["files"]:
                    url = item["url"]
                    self.assertIn("%2B", url)
                    self.assertIn("%23", url)
                    self.assertIn("%26", url)
                    self.assertIn("%20", url)

                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                    conn.request("GET", url)
                    r_preview = conn.getresponse()
                    data_preview = r_preview.read()
                    conn.close()
                    self.assertEqual(r_preview.status, 200, f"Preview failed for {url}")
                    self.assertGreater(len(data_preview), 0)

                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                    conn.request("GET", f"{url}&download=1")
                    r_dl = conn.getresponse()
                    data_dl = r_dl.read()
                    cd_hdr = r_dl.getheader("Content-Disposition") or ""
                    conn.close()
                    self.assertEqual(r_dl.status, 200, f"Download failed for {url}&download=1")
                    self.assertIn("attachment;", cd_hdr)
                    self.assertEqual(data_dl, data_preview)
            finally:
                server.shutdown()
                server.server_close()

    def test_charts_js_benchmark_null_regret_never_coerced_to_zero(self):
        charts_path = REPO_ROOT / "research_pipeline" / "v2" / "web_ui" / "static" / "charts.js"
        charts_js = charts_path.read_text(encoding="utf-8")
        bm_block = charts_js.split("function renderBenchmarkChart", 1)[1].split("window.NCKHCharts", 1)[0]
        self.assertIn(".filter(finite)", bm_block)
        self.assertIn("regret=null", bm_block)
        self.assertNotIn("finite(r.conditional_regret_mean) ? r.conditional_regret_mean : 0", bm_block)

        node_bin = shutil.which("node")
        if node_bin:
            script = """
const fs = require("fs");
const vm = require("vm");
const code = fs.readFileSync(process.argv[1], "utf8");
const sandbox = { window: {} };
vm.runInNewContext(code, sandbox);
const C = sandbox.window.NCKHCharts;
const elMixed = { innerHTML: "" };
C.renderBenchmarkChart(elMixed, [
  { method: "random", conditional_regret_mean: null },
  { method: "adaptive_full", conditional_regret_mean: 0.02 },
]);
const elAllNull = { innerHTML: "" };
C.renderBenchmarkChart(elAllNull, [
  { method: "random", conditional_regret_mean: null },
  { method: "default", conditional_regret_mean: undefined },
]);
const elZero = { innerHTML: "" };
C.renderBenchmarkChart(elZero, [
  { method: "adaptive_full", conditional_regret_mean: 0.0 },
]);
const elPerClassNull = { innerHTML: "" };
C.renderPerClassChart(elPerClassNull, [
  { class_name: "zebra", ap50: null, ap50_95: null },
  { class_name: "rhino", ap50: 0.75, ap50_95: 0.5 },
]);
const elCurveNull = { innerHTML: "" };
C.renderCurveChart(elCurveNull, [0.0, 0.5, 1.0], [0.9, null, 0.5], [{ class_name: "zebra", precision: [0.8, null, 0.4] }], "Recall", "Precision");
process.stdout.write(JSON.stringify({
  mixed: elMixed.innerHTML,
  all_null: elAllNull.innerHTML,
  zero: elZero.innerHTML,
  per_class: elPerClassNull.innerHTML,
  curve: elCurveNull.innerHTML,
}));
"""
            proc = subprocess.run(
                [node_bin, "-e", script, str(charts_path)],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )
            out = json.loads(proc.stdout)

            # Mixed null + finite: never emit regret=0.000000; only 1 data bar (+1 axis frame rect)
            self.assertNotIn("regret=0.000000", out["mixed"])
            self.assertIn("regret=null", out["mixed"])
            self.assertIn("adaptive_full: regret=0.020000", out["mixed"])
            mixed_root = ET.fromstring(out["mixed"])
            self.assertEqual(len(mixed_root.findall(".//rect")), 2)
            self.assertIn("null", [t.text for t in mixed_root.findall(".//text")])

            # All-null benchmark: 0 data bars (only axis frame rect), valid finite Y-ticks
            self.assertNotIn("regret=0.000000", out["all_null"])
            self.assertNotIn("NaN", out["all_null"])
            null_root = ET.fromstring(out["all_null"])
            self.assertEqual(len(null_root.findall(".//rect")), 1)
            y_ticks = [t.text for t in null_root.findall(".//text") if t.attrib.get("text-anchor") == "end"]
            self.assertEqual(len(y_ticks), 5)
            self.assertTrue(all(t != "NaN" for t in y_ticks))

            # Genuine feasible zero regret still reports regret=0.000000
            self.assertIn("adaptive_full: regret=0.000000", out["zero"])
            ET.fromstring(out["zero"])

            # Per-class and Curve charts do not coerce null to 0.0000 or emit NaN
            self.assertNotIn("AP50=0.0000", out["per_class"])
            self.assertIn("rhino: AP50=0.7500", out["per_class"])
            ET.fromstring(out["per_class"])
            self.assertNotIn("NaN", out["curve"])
            ET.fromstring(out["curve"])


if __name__ == "__main__":
    unittest.main()
