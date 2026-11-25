"""Tier 1 Feature Coverage: 85 isolated requirement-driven tests (5 per F1-F17)."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import unittest

from research_pipeline.v2 import acquisition as acq_mod
from research_pipeline.v2 import algorithm_study as study_mod
from research_pipeline.v2 import diagnostics as diag_mod
from research_pipeline.v2 import paper_export as paper_mod
from research_pipeline.v2 import predictors as pred_mod
from research_pipeline.v2 import scheduler as sched_mod
from research_pipeline.v2 import schema as schema_mod
from research_pipeline.v2 import selection as sel_mod
from research_pipeline.v2 import space as space_mod
from research_pipeline.v2 import terminal as term_mod
from research_pipeline.v2.data import discover
from research_pipeline.v2.ledger import Ledger, read_events
from research_pipeline.v2.model_features import describe_models, trainable_parameters_for
from research_pipeline.v2.report import make_report
from research_pipeline.v2.workflow import audit_run, source_map
from yolo_tuner.pipeline import runtime_info, write_json
from yolo_tuner.research import Objective

from tests_e2e._e2e_fixtures import (
    ALGORITHM_010_EXPECTED_SHA256,
    REPO_ROOT,
    YOLO_TUNER_SHA256,
    assert_zero_ansi,
    audit_count,
    audit_field,
    build_audited_cls_run,
    discover_ui_surface,
    extract_csrf_token,
    extract_external_urls,
    extract_nav_links,
    extract_stat_p_value,
    find_audit_dict,
    http_get,
    launch_ui_server,
    make_det_protocol,
    make_dirty_yolo_dataset,
    make_mini_cls_dataset,
    make_mini_yolo_dataset,
    make_synthetic_n5_study_dir,
    offline_yolo_val,
    run_cli,
    sha256_file,
    stop_ui_server,
)


def _det_cfg(
    *,
    model: str = "yolo11n.yaml",
    imgsz: int = 160,
    batch: int = 4,
    accumulation: int = 2,
    precision: str = "fp32",
    optimizer: str = "AdamW",
    lr0: float = 0.001,
    freeze: int = 0,
) -> dict:
    return {
        "model": model,
        "imgsz": imgsz,
        "batch": batch,
        "accumulation": accumulation,
        "precision": precision,
        "optimizer": optimizer,
        "lr0": lr0,
        "weight_decay": 0.0005,
        "freeze": freeze,
    }


def _trial_outcome(
    attempt_id: str,
    index: int,
    seed: int,
    epochs: int,
    quality: float,
    wall_s: float,
    loop_s: float,
    mem_mb: float,
) -> dict:
    return {
        "attempt_id": attempt_id,
        "index": index,
        "seed": seed,
        "mode": "train",
        "status": "ok",
        "epochs_completed": epochs,
        "optimizer_steps": 10,
        "training_loss_finite": True,
        "actual_config_verified": True,
        "checkpoint_sha256": f"ckpt-{attempt_id}",
        "quality": quality,
        "memory_mb": mem_mb,
        "worker_wall_seconds": wall_s,
        "measured_loop_seconds": loop_s,
    }


class TestTier1Features(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._shared_tmp = tempfile.TemporaryDirectory()
        cls._shared_root = Path(cls._shared_tmp.name)
        cls._f10_cache: tuple[Path, Path, dict] | None = None

    @classmethod
    def tearDownClass(cls):
        cls._shared_tmp.cleanup()

    # =========================================================================
    # F1: R1 Legacy SHA-256 & Historical Audits (5 tests)
    # =========================================================================
    def test_f01_yolo_tuner_19_files_sha256_preserved(self):
        for rel, expected in YOLO_TUNER_SHA256.items():
            self.assertEqual(sha256_file(REPO_ROOT / rel), expected, rel)

    def test_f01_algorithm_010_studies_api_hash_preserved(self):
        rep = study_mod.analyze_studies(REPO_ROOT / "configs" / "algorithm_010_studies.json")
        self.assertEqual(rep["sha256"], ALGORITHM_010_EXPECTED_SHA256)

    def test_f01_algorithm_010_studies_cli_hash_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "alg010"
            proc = run_cli(["algorithm-report", "--manifest", "configs/algorithm_010_studies.json", "--output", str(out)])
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
            data = json.loads((out / "ALGORITHM_STUDY.json").read_text(encoding="utf-8"))
            self.assertEqual(data["sha256"], ALGORITHM_010_EXPECTED_SHA256)

    def test_f01_historical_run_011_benchmark_audit_passes(self):
        bench_dir = REPO_ROOT / "runs" / "african_wildlife_nckh_benchmark" / "20261118_173050_553132"
        self.assertTrue(bench_dir.is_dir())
        checked = 0
        for p_file in sorted(bench_dir.rglob("protocol.json")):
            run_dir = p_file.parent
            if json.loads((run_dir / "status.json").read_text(encoding="utf-8")).get("phase") == "complete":
                audit_run(run_dir)
                checked += 1
        self.assertGreaterEqual(checked, 4)

    def test_f01_historical_run_report_parity(self):
        run_dir = REPO_ROOT / "runs" / "nckh_011_map50_verification"
        protos = sorted(run_dir.rglob("protocol.json"))
        self.assertTrue(protos)
        folder, prot, _, _ = audit_run(protos[0].parent)
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        events = read_events(folder / "events.jsonl")
        rebuilt = make_report(prot, state, events)
        saved = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(rebuilt["decision"], saved["decision"])

    # =========================================================================
    # F2: R1 Training Signature Exclusions (5 tests)
    # =========================================================================
    def test_f02_web_ui_modules_excluded_from_training_signature(self):
        sources = source_map()
        base = pred_mod.training_signature({"source_files": dict(sources)})
        changed = dict(sources)
        changed["research_pipeline/v2/web_ui/server.py"] = "a" * 64
        changed["research_pipeline/v2/web_ui/app.py"] = "b" * 64
        self.assertEqual(base, pred_mod.training_signature({"source_files": changed}))

    def test_f02_web_ui_submodules_and_report_modules_excluded_from_training_signature(self):
        sources = source_map()
        base = pred_mod.training_signature({"source_files": dict(sources)})
        changed = dict(sources)
        changed["research_pipeline/v2/web_ui/routes.py"] = "c" * 64
        changed["research_pipeline/v2/web_ui/templates/base.html"] = "d" * 64
        changed["research_pipeline/v2/web_ui/static/app.js"] = "e" * 64
        changed["research_pipeline/v2/paper_export.py"] = "f" * 64
        changed["research_pipeline/v2/report.py"] = "1" * 64
        changed["research_pipeline/v2/diagnostics.py"] = "2" * 64
        self.assertEqual(base, pred_mod.training_signature({"source_files": changed}))
        self.assertEqual(len(base), 64)

    def test_f02_wizard_display_excluded_from_training_signature(self):
        sources = source_map()
        base = pred_mod.training_signature({"source_files": dict(sources)})
        changed = dict(sources)
        changed["research_pipeline/v2/wizard.py"] = "d" * 64
        self.assertEqual(base, pred_mod.training_signature({"source_files": changed}))

    def test_f02_worker_source_change_invalidates_training_signature(self):
        sources = source_map()
        base = pred_mod.training_signature({"source_files": dict(sources)})
        changed = dict(sources)
        changed["research_pipeline/v2/worker.py"] = "e" * 64
        self.assertNotEqual(base, pred_mod.training_signature({"source_files": changed}))

    def test_f02_execution_policy_changes_training_signature_and_card_compatibility(self):
        sources = source_map()
        p1 = {
            "source_files": sources,
            "execution_policy": {"version": 2, "data_loader": {"workers": 0, "cache_mb": 0.0, "pin_memory": "off"}},
        }
        p2 = {
            "source_files": sources,
            "execution_policy": {"version": 2, "data_loader": {"workers": 2, "cache_mb": 64.0, "pin_memory": "on"}},
        }
        self.assertNotEqual(pred_mod.training_signature(p1), pred_mod.training_signature(p2))

    # =========================================================================
    # F3: R2 Cross-Config VRAM Predictor (`observed_cross_memory`) (5 tests)
    # =========================================================================
    def test_f03_observed_cross_memory_none_without_full_observations(self):
        prot = make_det_protocol([_det_cfg(imgsz=160), _det_cfg(imgsz=320)], acquisition_version=1)
        self.assertIsNone(pred_mod.observed_cross_memory(prot, {}, 1))

    def test_f03_observed_cross_memory_returns_decomposed_components(self):
        cfgs = [_det_cfg(imgsz=160, batch=2), _det_cfg(imgsz=320, batch=4)]
        descs = describe_models("detection", cfgs, 2)
        prot = make_det_protocol(cfgs, acquisition_version=1, model_descriptors=descs)
        fulls = {0: [_trial_outcome("a1", 0, 17, 2, 0.5, 4.0, 3.0, 800.0)]}
        est = pred_mod.observed_cross_memory(prot, fulls, 1)
        self.assertIsInstance(est, dict)
        for key in ("point_mb", "base_allocator_mb", "static_model_state_mb", "activation_mb", "basis"):
            self.assertIn(key, est)
        self.assertIs(est["upper_bound_guaranteed"], False)
        self.assertGreater(est["point_mb"], 800.0)

    def test_f03_activation_mb_scales_with_batch_imgsz_independent_of_accumulation(self):
        cfgs = [
            _det_cfg(imgsz=160, batch=2, accumulation=4),
            _det_cfg(imgsz=320, batch=4, accumulation=1),
            _det_cfg(imgsz=320, batch=4, accumulation=8),
        ]
        descs = describe_models("detection", cfgs, 2)
        prot = make_det_protocol(cfgs, acquisition_version=1, model_descriptors=descs)
        fulls = {0: [_trial_outcome("a1", 0, 17, 2, 0.5, 4.0, 3.0, 900.0)]}
        e1 = pred_mod.observed_cross_memory(prot, fulls, 1)
        e2 = pred_mod.observed_cross_memory(prot, fulls, 2)
        self.assertAlmostEqual(e1["activation_mb"], e2["activation_mb"], places=6)
        self.assertGreater(e1["activation_mb"], 0.0)

    def test_f03_cost_model_estimate_populates_vram_fields(self):
        cfgs = [_det_cfg(imgsz=160, batch=2), _det_cfg(imgsz=320, batch=4)]
        descs = describe_models("detection", cfgs, 2)
        prot = make_det_protocol(cfgs, acquisition_version=1, model_descriptors=descs)
        cm = pred_mod.CostModel(prot)
        cm.update(0, _trial_outcome("a1", 0, 17, 2, 0.5, 4.0, 3.0, 750.0))
        est = cm.estimate(1)
        self.assertIsNotNone(est["predicted_memory_mb"])
        self.assertIn("memory_basis", est)
        self.assertIn("memory_components", est)
        self.assertIs(est["upper_bound_guaranteed"], False)

    def test_f03_scheduler_oom_risk_guard_steers_away_from_high_vram(self):
        cfgs = [
            _det_cfg(imgsz=160, batch=2, lr0=0.001),
            _det_cfg(imgsz=640, batch=16, lr0=0.001),
            _det_cfg(imgsz=160, batch=2, lr0=0.0003),
        ]
        descs = describe_models("detection", cfgs, 2)
        prot = make_det_protocol(
            cfgs,
            method="adaptive_full",
            acquisition_version=2,
            training_seeds=[17],
            budget_seconds=60.0,
            max_memory=1000.0,
            model_descriptors=descs,
        )
        with tempfile.TemporaryDirectory() as td:
            ledger = Ledger(Path(td) / "events.jsonl")
            clock = [0.0]
            executed = []

            def eval_fn(idx, seed, epochs, full, seconds):
                executed.append(idx)
                clock[0] += 2.0
                mem = 900.0 if idx in (0, 2) else 4500.0
                return _trial_outcome(f"a{len(executed)}", idx, seed, epochs, 0.6 + 0.05 * idx, 2.0, 1.5, mem)

            state = sched_mod.run_search(prot, eval_fn, lambda: 60.0 - clock[0], ledger, elapsed=lambda: clock[0])
            self.assertNotIn(1, executed)
            self.assertIn(0, executed)
            self.assertIsInstance(state["results"]["0"], list)

    # =========================================================================
    # F4: R2 Cost Predictor Upgrade (`observed_cross_cost`) (5 tests)
    # =========================================================================
    def test_f04_observed_cross_cost_separates_non_loop_and_loop(self):
        cfgs = [_det_cfg(imgsz=160), _det_cfg(imgsz=320)]
        prot = make_det_protocol(cfgs, acquisition_version=1)
        fulls = {0: [_trial_outcome("a1", 0, 17, 2, 0.5, 10.0, 6.0, 500.0)]}
        est = pred_mod.observed_cross_cost(prot, fulls, 1)
        self.assertEqual(est["cost_components"]["observed_non_loop_overhead_seconds"], 4.0)
        self.assertGreater(est["cost_components"]["scaled_loop_seconds"], 6.0)

    def test_f04_cost_model_estimate_separates_non_loop_and_loop_seconds(self):
        cfgs = [_det_cfg(imgsz=160), _det_cfg(imgsz=320)]
        prot = make_det_protocol(cfgs, acquisition_version=1, cost_margin=1.25)
        cm = pred_mod.CostModel(prot)
        cm.update(0, _trial_outcome("a1", 0, 17, 2, 0.5, 10.0, 6.0, 500.0))
        est = cm.estimate(1)
        self.assertIn("non_loop_seconds", est)
        self.assertIn("loop_seconds", est)
        self.assertIn("effective_margin", est)
        self.assertIn("online_error_ratio", est)

    def test_f04_non_loop_overhead_not_multiplied_by_cost_margin(self):
        cfgs = [_det_cfg(imgsz=160), _det_cfg(imgsz=160, lr0=0.0003)]
        prot = make_det_protocol(cfgs, acquisition_version=1, cost_margin=1.5)
        cm = pred_mod.CostModel(prot)
        cm.update(0, _trial_outcome("a1", 0, 17, 2, 0.5, 20.0, 4.0, 500.0))
        est = cm.estimate(1)
        self.assertAlmostEqual(est["non_loop_seconds"], 16.0, places=3)
        # planning_seconds should NOT inflate the 16s fixed overhead by 1.5x (which would be >= 30s)
        self.assertLess(est["planning_seconds"], 16.0 + 4.0 * 1.5 + 1e-6)

    def test_f04_online_error_ratio_updates_across_completed_trials(self):
        cfgs = [
            _det_cfg(imgsz=160, lr0=0.001),
            _det_cfg(imgsz=160, lr0=0.0003),
            _det_cfg(imgsz=320, lr0=0.001),
            _det_cfg(imgsz=320, lr0=0.0003),
        ]
        prot = make_det_protocol(cfgs, acquisition_version=1, policy_version=3)
        cm = pred_mod.CostModel(prot)
        cm.update(0, _trial_outcome("a1", 0, 17, 2, 0.5, 10.0, 6.0, 500.0))
        e1 = cm.estimate(1)
        r1, m1 = e1["online_error_ratio"], e1["effective_margin"]
        cm.update(1, _trial_outcome("a2", 1, 17, 2, 0.6, 12.0, 8.0, 500.0))
        e2 = cm.estimate(2)
        r2, m2 = e2["online_error_ratio"], e2["effective_margin"]
        cm.update(2, _trial_outcome("a3", 2, 17, 2, 0.65, 22.0, 18.0, 500.0))
        e3 = cm.estimate(3)
        r3, m3 = e3["online_error_ratio"], e3["effective_margin"]
        self.assertIsNone(r1)
        self.assertIsNotNone(r2)
        self.assertIsNotNone(r3)
        self.assertNotEqual(r2, r3)
        self.assertNotEqual(m1, m2)
        self.assertNotEqual(m2, m3)
        self.assertTrue(math.isfinite(m3))

    def test_f04_smoothed_cost_improves_late_budget_utilization_without_overrun(self):
        cfgs = [_det_cfg(imgsz=160, lr0=lr) for lr in (0.001, 0.0003, 0.003)]
        prot = make_det_protocol(cfgs, method="adaptive_full", acquisition_version=2, policy_version=3, training_seeds=[17], budget_seconds=50.0, cost_margin=1.25)
        with tempfile.TemporaryDirectory() as td:
            ledger = Ledger(Path(td) / "events.jsonl")
            clock = [0.0]

            def eval_fn(idx, seed, epochs, full, seconds):
                # 14s non-loop overhead + 2s loop = 16s total; 3 configs = 48s <= 50s
                clock[0] += 16.0
                return _trial_outcome(f"a{idx}", idx, seed, epochs, 0.5 + 0.1 * idx, 16.0, 2.0, 500.0)

            state = sched_mod.run_search(prot, eval_fn, lambda: 50.0 - clock[0], ledger, elapsed=lambda: clock[0])
            completed = sum(1 for rs in state["results"].values() if rs)
            self.assertEqual(completed, 3)
            self.assertLessEqual(clock[0], 50.0)

    # =========================================================================
    # F5: R2 Deterministic Heteroscedastic GP / Adaptive Kernel Ridge (5 tests)
    # =========================================================================
    def test_f05_rank_candidates_returns_surrogate_diagnostics_schema(self):
        cfgs = [_det_cfg(lr0=lr) for lr in (0.001, 0.0003, 0.003, 0.0001)]
        prot = make_det_protocol(cfgs, acquisition_version=2, training_seeds=[17, 29])
        results = {
            "0": [_trial_outcome("a1", 0, 17, 2, 0.70, 5.0, 3.0, 500.0), _trial_outcome("a2", 0, 29, 2, 0.74, 5.0, 3.0, 500.0)],
            "1": [_trial_outcome("a3", 1, 17, 2, 0.60, 5.0, 3.0, 500.0), _trial_outcome("a4", 1, 29, 2, 0.62, 5.0, 3.0, 500.0)],
        }
        rows = schema_mod.measurements(cfgs, results, [17, 29], 2, Objective(**prot["objective"]))
        forecasts = {i: {"point_seconds": 5.0, "planning_seconds": 6.0, "predicted_memory_mb": 500.0} for i in range(4)}
        out = acq_mod.rank_candidates(prot, rows, forecasts)
        self.assertIn("surrogate_diagnostics", out)
        diag = out["surrogate_diagnostics"]
        self.assertIn(diag["kernel"], ("matern52", "rbf"))
        self.assertGreater(diag["length_scale"], 0.0)
        self.assertGreater(diag["ridge"], 0.0)
        self.assertTrue(math.isfinite(diag["loocv_mae"]))
        self.assertTrue(math.isfinite(diag["loocv_rmse"]))
        self.assertIn("heteroscedastic", diag)
        self.assertIn("predictions", diag)

    def test_f05_surrogate_diagnostics_100_percent_deterministic(self):
        cfgs = [_det_cfg(lr0=lr) for lr in (0.001, 0.0003, 0.003, 0.0001)]
        prot = make_det_protocol(cfgs, acquisition_version=2, training_seeds=[17, 29])
        results = {
            "0": [_trial_outcome("a1", 0, 17, 2, 0.70, 5.0, 3.0, 500.0), _trial_outcome("a2", 0, 29, 2, 0.76, 5.0, 3.0, 500.0)],
            "1": [_trial_outcome("a3", 1, 17, 2, 0.55, 5.0, 3.0, 500.0), _trial_outcome("a4", 1, 29, 2, 0.57, 5.0, 3.0, 500.0)],
        }
        rows = schema_mod.measurements(cfgs, results, [17, 29], 2, Objective(**prot["objective"]))
        forecasts = {i: {"point_seconds": 5.0, "planning_seconds": 6.0, "predicted_memory_mb": 500.0} for i in range(4)}
        out1 = acq_mod.rank_candidates(prot, rows, forecasts)
        out2 = acq_mod.rank_candidates(prot, rows, forecasts)
        self.assertEqual(out1["ranking"], out2["ranking"])
        self.assertEqual(out1["surrogate_diagnostics"], out2["surrogate_diagnostics"])

    def test_f05_heteroscedastic_noise_enabled_with_multi_seed_variance(self):
        cfgs = [_det_cfg(lr0=lr) for lr in (0.001, 0.0003, 0.003)]
        prot = make_det_protocol(cfgs, acquisition_version=2, training_seeds=[17, 29])
        results = {
            "0": [_trial_outcome("a1", 0, 17, 2, 0.50, 5.0, 3.0, 500.0), _trial_outcome("a2", 0, 29, 2, 0.90, 5.0, 3.0, 500.0)],
            "1": [_trial_outcome("a3", 1, 17, 2, 0.70, 5.0, 3.0, 500.0), _trial_outcome("a4", 1, 29, 2, 0.71, 5.0, 3.0, 500.0)],
        }
        rows = schema_mod.measurements(cfgs, results, [17, 29], 2, Objective(**prot["objective"]))
        forecasts = {i: {"point_seconds": 5.0, "planning_seconds": 6.0, "predicted_memory_mb": 500.0} for i in range(3)}
        out = acq_mod.rank_candidates(prot, rows, forecasts)
        self.assertTrue(out["surrogate_diagnostics"]["heteroscedastic"])

    def test_f05_loocv_selects_finite_positive_hyperparameters_and_predictions(self):
        cfgs = [_det_cfg(lr0=lr) for lr in (0.0001, 0.0003, 0.001, 0.003, 0.01)]
        prot_auto = make_det_protocol(cfgs, acquisition_version=2, training_seeds=[17])
        prot_rbf = make_det_protocol(cfgs, acquisition_version=2, training_seeds=[17])
        prot_rbf["acquisition"]["kernel"] = "rbf"
        prot_m52 = make_det_protocol(cfgs, acquisition_version=2, training_seeds=[17])
        prot_m52["acquisition"]["kernel"] = "matern52"
        results = {
            "0": [_trial_outcome("a0", 0, 17, 2, 0.45, 5.0, 3.0, 500.0)],
            "1": [_trial_outcome("a1", 1, 17, 2, 0.80, 5.0, 3.0, 500.0)],
            "2": [_trial_outcome("a2", 2, 17, 2, 0.65, 5.0, 3.0, 500.0)],
            "3": [_trial_outcome("a3", 3, 17, 2, 0.72, 5.0, 3.0, 500.0)],
        }
        rows = schema_mod.measurements(cfgs, results, [17], 2, Objective(**prot_auto["objective"]))
        forecasts = {i: {"point_seconds": 5.0, "planning_seconds": 6.0, "predicted_memory_mb": 500.0} for i in range(5)}
        out = acq_mod.rank_candidates(prot_auto, rows, forecasts)
        diag = out["surrogate_diagnostics"]
        self.assertGreaterEqual(diag.get("grid_size_evaluated", 2), 2)
        preds = diag["predictions"]
        self.assertIn("4", preds)
        self.assertIn("mean", preds["4"])
        self.assertIn("std", preds["4"])
        self.assertGreaterEqual(preds["4"]["std"], 0.0)
        self.assertEqual(acq_mod.rank_candidates(prot_rbf, rows, forecasts)["surrogate_diagnostics"]["kernel"], "rbf")
        self.assertEqual(acq_mod.rank_candidates(prot_m52, rows, forecasts)["surrogate_diagnostics"]["kernel"], "matern52")

    def test_f05_scheduler_persists_surrogate_diagnostics_in_acquisition_history(self):
        cfgs = [_det_cfg(lr0=lr) for lr in (0.001, 0.0003, 0.003, 0.0001)]
        prot = make_det_protocol(cfgs, method="adaptive_full", acquisition_version=2, training_seeds=[17], budget_seconds=60.0)
        with tempfile.TemporaryDirectory() as td:
            ledger = Ledger(Path(td) / "events.jsonl")
            clock = [0.0]

            def eval_fn(idx, seed, epochs, full, seconds):
                clock[0] += 3.0
                return _trial_outcome(f"a{idx}", idx, seed, epochs, 0.5 + 0.1 * idx, 3.0, 2.0, 500.0)

            state = sched_mod.run_search(prot, eval_fn, lambda: 60.0 - clock[0], ledger, elapsed=lambda: clock[0])
            diags = [step.get("surrogate_diagnostics") for step in state["acquisition_history"] if "surrogate_diagnostics" in step]
            self.assertTrue(diags)

    # =========================================================================
    # F6: R2 Acquisition v2 & Rule Consistency (5 tests)
    # =========================================================================
    def test_f06_acquisition_version_1_and_2_both_valid_in_schema(self):
        cfgs = [_det_cfg(lr0=lr) for lr in (0.001, 0.0003, 0.003, 0.0001)]
        p1 = make_det_protocol(cfgs, acquisition_version=1, training_seeds=[17])
        p2 = make_det_protocol(cfgs, acquisition_version=2, training_seeds=[17])
        schema_mod.validate_protocol(p1)
        schema_mod.validate_protocol(p2)
        self.assertEqual(acq_mod.acquisition_spec(p1)["version"], 1)
        self.assertEqual(acq_mod.acquisition_spec(p2)["version"], 2)
        results = {
            "0": [_trial_outcome("a0", 0, 17, 2, 0.60, 5.0, 3.0, 400.0)],
            "1": [_trial_outcome("a1", 1, 17, 2, 0.72, 5.0, 3.0, 400.0)],
        }
        rows = schema_mod.measurements(cfgs, results, [17], 2, Objective(**p1["objective"]))
        fc = {2: {"point_seconds": 4.0, "planning_seconds": 5.0, "predicted_memory_mb": 400.0},
              3: {"point_seconds": 12.0, "planning_seconds": 15.0, "predicted_memory_mb": 400.0}}
        out_v1 = acq_mod.rank_candidates(p1, rows, fc)
        self.assertNotIn("surrogate_diagnostics", out_v1)
        self.assertEqual(out_v1["ranking"], acq_mod.rank_candidates(p1, rows, fc)["ranking"])
        self.assertNotIn("ehvi", out_v1["scores"]["2"])

    def test_f06_acquisition_v2_ei_per_cost_ranking(self):
        cfgs = [_det_cfg(lr0=lr) for lr in (0.001, 0.0003, 0.003, 0.0001)]
        prot = make_det_protocol(cfgs, acquisition_version=2, selection_rule="weighted", training_seeds=[17])
        results = {
            "0": [_trial_outcome("a0", 0, 17, 2, 0.60, 5.0, 3.0, 500.0)],
            "1": [_trial_outcome("a1", 1, 17, 2, 0.70, 5.0, 3.0, 500.0)],
        }
        rows = schema_mod.measurements(cfgs, results, [17], 2, Objective(**prot["objective"]))
        forecasts = {
            2: {"point_seconds": 4.0, "planning_seconds": 5.0, "predicted_memory_mb": 400.0},
            3: {"point_seconds": 40.0, "planning_seconds": 50.0, "predicted_memory_mb": 400.0},
        }
        out = acq_mod.rank_candidates(prot, rows, forecasts)
        self.assertEqual(out["ranking"][0], 2)

    def test_f06_adaptive_full_unlocked_for_all_four_selection_rules_in_schema(self):
        for rule in ("weighted", "tchebycheff", "topsis", "knee"):
            prot = make_det_protocol([_det_cfg(), _det_cfg(lr0=0.0003)], method="adaptive_full", selection_rule=rule, acquisition_version=2)
            schema_mod.validate_protocol(prot)

    def test_f06_cli_plan_search_unlocks_adaptive_full_with_topsis_knee_tchebycheff(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            yaml_path = make_mini_yolo_dataset(root / "ds")
            for rule in ("tchebycheff", "topsis", "knee"):
                proc = run_cli([
                    "plan-search",
                    "--data", str(yaml_path),
                    "--device", "cpu",
                    "--space", "compact",
                    "--method", "adaptive_full",
                    "--selection-rule", rule,
                    "--full-epochs", "1",
                    "--seeds", "17",
                    "--budget-seconds", "30",
                    "--output", str(root / f"out_{rule}"),
                ])
                self.assertEqual(proc.returncode, 0, msg=f"rule={rule}: {proc.stderr}{proc.stdout}")

    def test_f06_rank_candidates_respects_selection_rule_consistency(self):
        cfgs = [
            _det_cfg(imgsz=160, lr0=0.001),
            _det_cfg(imgsz=160, lr0=0.0003),
            _det_cfg(imgsz=160, lr0=0.003),
            _det_cfg(imgsz=320, lr0=0.0001),
            _det_cfg(imgsz=320, lr0=0.005),
        ]
        # 3-point non-degenerate 3D Pareto front:
        # 0: highest quality; 1: lowest time; 2: lowest memory
        results = {
            "0": [_trial_outcome("a0", 0, 17, 2, 0.90, 20.0, 15.0, 900.0)],
            "1": [_trial_outcome("a1", 1, 17, 2, 0.65, 3.0, 2.0, 600.0)],
            "2": [_trial_outcome("a2", 2, 17, 2, 0.72, 10.0, 7.0, 250.0)],
        }
        forecasts = {
            3: {"point_seconds": 4.0, "planning_seconds": 5.0, "predicted_memory_mb": 350.0},
            4: {"point_seconds": 15.0, "planning_seconds": 18.0, "predicted_memory_mb": 950.0},
        }
        rule_score_vectors = {}
        for rule in ("weighted", "tchebycheff", "topsis", "knee"):
            prot = make_det_protocol(cfgs, method="adaptive_full", selection_rule=rule, acquisition_version=2, training_seeds=[17])
            rows = schema_mod.measurements(cfgs, results, [17], 2, Objective(**prot["objective"]))
            out = acq_mod.rank_candidates(prot, rows, forecasts)
            self.assertEqual(set(out["ranking"]), {3, 4})
            for idx in out["ranking"]:
                self.assertEqual(out["scores"][str(idx)]["selection_rule"], rule)
                if rule == "knee":
                    self.assertNotIn("tchebycheff fallback", " ".join(out["scores"][str(idx)].get("reasons", [])))
            ranked_scores = [out["scores"][str(idx)]["score"] for idx in out["ranking"]]
            self.assertEqual(ranked_scores, sorted(ranked_scores, reverse=True))
            self.assertGreater(ranked_scores[0], ranked_scores[-1], f"rule={rule} failed strict score discrimination: {ranked_scores}")
            rule_score_vectors[rule] = tuple(round(float(out["scores"][str(idx)]["score"]), 12) for idx in (3, 4))
        self.assertGreater(
            len(set(rule_score_vectors.values())),
            1,
            f"All 4 rules produced identical score vectors: {rule_score_vectors}",
        )

    # =========================================================================
    # F7: R2 `adaptive_ehvi` & `adaptive_staged_seeds` (5 tests)
    # =========================================================================
    def test_f07_methods_include_adaptive_ehvi_and_adaptive_staged_seeds(self):
        self.assertIn("adaptive_ehvi", sched_mod.METHODS)
        self.assertIn("adaptive_staged_seeds", sched_mod.METHODS)
        for legacy_m in ("adaptive_full", "random", "reserve", "successive_halving", "default"):
            self.assertIn(legacy_m, sched_mod.METHODS)

    def test_f07_adaptive_ehvi_executes_and_selects_valid_pareto_candidate(self):
        cfgs = [_det_cfg(imgsz=s, lr0=lr) for s, lr in ((160, 0.001), (320, 0.0003), (160, 0.003), (320, 0.005))]
        prot = make_det_protocol(cfgs, method="adaptive_ehvi", acquisition_version=2, training_seeds=[17], budget_seconds=60.0)
        with tempfile.TemporaryDirectory() as td:
            ledger = Ledger(Path(td) / "events.jsonl")
            clock = [0.0]

            def eval_fn(idx, seed, epochs, full, seconds):
                clock[0] += 4.0
                return _trial_outcome(f"a{idx}", idx, seed, epochs, 0.5 + 0.1 * idx, 4.0, 2.5, 400.0 + 50 * idx)

            state = sched_mod.run_search(prot, eval_fn, lambda: 60.0 - clock[0], ledger, elapsed=lambda: clock[0])
            rows = schema_mod.measurements(cfgs, state["results"], [17], 2, Objective(**prot["objective"]))
            self.assertTrue(any(r["complete"] for r in rows))

    def test_f07_adaptive_staged_seeds_evaluates_seed1_before_full_confirmation(self):
        cfgs = [
            _det_cfg(imgsz=160, lr0=0.001),
            _det_cfg(imgsz=160, lr0=0.0001),
            _det_cfg(imgsz=160, lr0=0.0003),
            _det_cfg(imgsz=160, lr0=0.003),
        ]
        quals = {0: 0.65, 1: 0.20, 2: 0.25, 3: 0.90}

        # 1) Full budget: seed 17 evaluated across deferred weak & leading candidates BEFORE deferred weak candidates get seed 29
        prot = make_det_protocol(cfgs, method="adaptive_staged_seeds", acquisition_version=2, training_seeds=[17, 29], budget_seconds=45.0)
        with tempfile.TemporaryDirectory() as td:
            ledger = Ledger(Path(td) / "events.jsonl")
            clock = [0.0]
            calls = []

            def eval_fn(idx, seed, epochs, full, seconds):
                calls.append((idx, seed))
                clock[0] += 4.0
                return _trial_outcome(f"a{len(calls)}", idx, seed, epochs, quals[idx], 4.0, 2.5, 500.0)

            state = sched_mod.run_search(prot, eval_fn, lambda: 45.0 - clock[0], ledger, elapsed=lambda: clock[0])
            self.assertGreaterEqual(len(calls), 8)
            for cand_idx in (1, 2, 3):
                self.assertIn((cand_idx, 17), calls)
            self.assertLess(calls.index((2, 17)), calls.index((1, 29)))
            self.assertLess(calls.index((3, 17)), calls.index((1, 29)))
            self.assertLess(calls.index((3, 17)), calls.index((2, 29)))
            # Leading seed-1 candidate 3 gets seed-2 confirmation BEFORE weaker deferred candidates 1 & 2
            self.assertLess(calls.index((3, 29)), calls.index((1, 29)))
            self.assertLess(calls.index((3, 29)), calls.index((2, 29)))
            rows = schema_mod.measurements(cfgs, state["results"], [17, 29], 2, Objective(**prot["objective"]))
            self.assertTrue(any(r["complete"] for r in rows))

        # 2) Tight budget: leading seed-1 candidate 3 completes both seeds while weak 1 & 2 stop at seed 1
        prot_tight = make_det_protocol(cfgs, method="adaptive_staged_seeds", acquisition_version=2, training_seeds=[17, 29], budget_seconds=27.0)
        with tempfile.TemporaryDirectory() as td:
            ledger_t = Ledger(Path(td) / "events.jsonl")
            clock_t = [0.0]
            calls_t = []

            def eval_tight(idx, seed, epochs, full, seconds):
                calls_t.append((idx, seed))
                clock_t[0] += 4.0
                return _trial_outcome(f"t{len(calls_t)}", idx, seed, epochs, quals[idx], 4.0, 2.5, 500.0)

            state_t = sched_mod.run_search(prot_tight, eval_tight, lambda: 27.0 - clock_t[0], ledger_t, elapsed=lambda: clock_t[0])
            self.assertEqual(state_t["decision"]["selected_index"], 3)
            self.assertIn((3, 29), calls_t)
            self.assertNotIn((1, 29), calls_t)
            self.assertNotIn((2, 29), calls_t)

    def test_f07_staged_partial_seeds_never_enter_full_observations_or_selection(self):
        cfgs = [_det_cfg(lr0=lr) for lr in (0.001, 0.0003)]
        prot = make_det_protocol(cfgs, method="adaptive_staged_seeds", acquisition_version=2, training_seeds=[17, 29])
        results = {
            "0": [_trial_outcome("a0", 0, 17, 2, 0.99, 4.0, 2.5, 500.0)],  # partial: only seed 17
            "1": [_trial_outcome("a1", 1, 17, 2, 0.70, 4.0, 2.5, 500.0), _trial_outcome("a2", 1, 29, 2, 0.72, 4.0, 2.5, 500.0)],
        }
        obj = Objective(**prot["objective"])
        rows = schema_mod.measurements(cfgs, results, [17, 29], 2, obj)
        full_obs = acq_mod.full_observations(prot, rows, obj)
        self.assertEqual([o["index"] for o in full_obs], [1])
        sel = sel_mod.select(rows, prot)
        self.assertEqual(sel["selected_index"], 1)

    def test_f07_legacy_methods_still_accepted_by_plan_search_cli(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            yaml_path = make_mini_yolo_dataset(root / "ds")
            for method in ("random", "reserve", "successive_halving", "default"):
                proc = run_cli([
                    "plan-search",
                    "--data", str(yaml_path),
                    "--device", "cpu",
                    "--space", "compact",
                    "--method", method,
                    "--full-epochs", "1",
                    "--seeds", "17",
                    "--budget-seconds", "30",
                    "--output", str(root / f"out_{method}"),
                ])
                self.assertEqual(proc.returncode, 0, msg=f"{method}: {proc.stderr}")

    # =========================================================================
    # F8: R3 Preflight Data Quality & Anti-Leakage Audit (5 tests)
    # =========================================================================
    def test_f08_inspect_clean_mini_yolo_returns_structured_audit_zero_issues(self):
        with tempfile.TemporaryDirectory() as td:
            yaml_path = make_mini_yolo_dataset(Path(td) / "clean")
            proc = run_cli(["inspect", "--data", str(yaml_path), "--device", "cpu", "--json"])
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            payload = json.loads(proc.stdout)
            audit = find_audit_dict(payload)
            self.assertEqual(audit_count(audit, "intra_split_duplicates_count", "intra_split_duplicates", "duplicates.intra_split_count", "duplicates.intra_split"), 0)
            self.assertEqual(audit_count(audit, "cross_split_leakage_count", "cross_split_leakage", "leakage.cross_split_count", "leakage.cross_split"), 0)

    def test_f08_inspect_dirty_yolo_reports_duplicates_and_cross_split_leakage(self):
        with tempfile.TemporaryDirectory() as td:
            yaml_path = make_dirty_yolo_dataset(Path(td) / "dirty")
            proc = run_cli(["inspect", "--data", str(yaml_path), "--device", "cpu", "--json"])
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
            audit = find_audit_dict(json.loads(proc.stdout))
            self.assertGreaterEqual(audit_count(audit, "intra_split_duplicates_count", "intra_split_duplicates", "duplicates.intra_split_count", "duplicates.intra_split"), 1)
            self.assertGreaterEqual(audit_count(audit, "cross_split_leakage_count", "cross_split_leakage", "leakage.cross_split_count", "leakage.cross_split"), 2)

    def test_f08_inspect_dirty_yolo_reports_bbox_label_imbalance_and_sizes(self):
        with tempfile.TemporaryDirectory() as td:
            yaml_path = make_dirty_yolo_dataset(Path(td) / "dirty")
            proc = run_cli(["inspect", "--data", str(yaml_path), "--device", "cpu", "--json"])
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
            audit = find_audit_dict(json.loads(proc.stdout))
            self.assertGreaterEqual(audit_count(audit, "missing_labels_count", "missing_labels", "labels.missing_count", "labels.missing"), 1)
            self.assertGreaterEqual(audit_count(audit, "empty_labels_count", "empty_labels", "labels.empty_count", "labels.empty"), 1)
            self.assertGreaterEqual(audit_count(audit, "out_of_bounds_boxes_count", "out_of_bounds_boxes", "boxes.out_of_bounds_count", "boxes.out_of_bounds"), 1)
            self.assertGreaterEqual(audit_count(audit, "degenerate_boxes_count", "degenerate_boxes", "boxes.degenerate_count", "boxes.degenerate"), 2)
            self.assertGreaterEqual(audit_count(audit, "invalid_class_ids_count", "invalid_class_ids", "boxes.invalid_class_ids_count", "labels.invalid_class_ids"), 1)
            imb = audit_field(audit, "class_imbalance_ratio", "imbalance_ratio", "class_distribution.imbalance_ratio", "classes.imbalance_ratio")
            self.assertGreater(float(imb), 1.0)
            sizes = audit_field(audit, "object_size_distribution", "box_size_distribution", "size_distribution", "boxes.size_distribution")
            self.assertIsInstance(sizes, dict)
            size_bins = sizes.get("normalized_area", sizes.get("coco_pixel_area", sizes))
            for sz_k in ("small", "medium", "large"):
                self.assertIn(sz_k, size_bins)
                self.assertGreaterEqual(size_bins[sz_k], 1)

    def test_f08_inspect_audit_json_exports_file_read_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ds_root = root / "dirty"
            yaml_path = make_dirty_yolo_dataset(ds_root)
            audit_out = root / "data_audit.json"
            proc = run_cli(["inspect", "--data", str(yaml_path), "--device", "cpu", "--audit-json", str(audit_out), "--json"])
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
            self.assertTrue(audit_out.is_file())
            saved = find_audit_dict(json.loads(audit_out.read_text(encoding="utf-8")))
            self.assertGreaterEqual(audit_count(saved, "cross_split_leakage_count", "cross_split_leakage", "leakage.cross_split_count", "leakage.cross_split"), 2)

    def test_f08_dedup_dataset_creates_clean_copy_without_mutating_raw(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ds_root = root / "dirty"
            yaml_path = make_dirty_yolo_dataset(ds_root)
            before = {p.relative_to(ds_root).as_posix(): sha256_file(p) for p in sorted(ds_root.rglob("*")) if p.is_file()}
            clean_dir = root / "cleaned"
            proc = run_cli(["dedup-dataset", "--data", str(yaml_path), "--output", str(clean_dir)])
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
            after = {p.relative_to(ds_root).as_posix(): sha256_file(p) for p in sorted(ds_root.rglob("*")) if p.is_file()}
            self.assertEqual(before, after)
            clean_yaml = clean_dir / "data.yaml"
            self.assertTrue(clean_yaml.is_file())
            re_inspect = run_cli(["inspect", "--data", str(clean_yaml), "--device", "cpu", "--json"])
            self.assertEqual(re_inspect.returncode, 0, msg=re_inspect.stderr)
            clean_audit = find_audit_dict(json.loads(re_inspect.stdout))
            self.assertEqual(audit_count(clean_audit, "intra_split_duplicates_count", "intra_split_duplicates", "duplicates.intra_split_count", "duplicates.intra_split"), 0)
            self.assertEqual(audit_count(clean_audit, "cross_split_leakage_count", "cross_split_leakage", "leakage.cross_split_count", "leakage.cross_split"), 0)

    # =========================================================================
    # F9: R3 Controlled Search Space & Local .pt Support (5 tests)
    # =========================================================================
    def test_f09_space_supports_controlled_research_hyperparameters(self):
        hw = runtime_info("cpu")
        with tempfile.TemporaryDirectory() as td:
            grid = Path(td) / "grid.json"
            write_json(grid, [{
                "imgsz": 64, "batch": 2, "accumulation": 1,
                "cos_lr": True, "warmup_epochs": 1.5, "mosaic": 0.5, "close_mosaic": 2,
            }])
            cfgs = space_mod.build_space("detection", hw, Path(td) / "work", grid=str(grid))
            self.assertEqual(len(cfgs), 1)
            self.assertIs(cfgs[0]["cos_lr"], True)
            self.assertEqual(cfgs[0]["warmup_epochs"], 1.5)
            self.assertEqual(cfgs[0]["mosaic"], 0.5)
            self.assertEqual(cfgs[0]["close_mosaic"], 2)

    def test_f09_config_id_unchanged_when_optional_m2_keys_omitted(self):
        base = _det_cfg()
        self.assertEqual(schema_mod.config_id(base), "1d888424d09b")
        hist_p = sorted((REPO_ROOT / "runs" / "nckh_011_map50_verification").rglob("protocol.json"))[0]
        hist_prot = json.loads(hist_p.read_text(encoding="utf-8"))
        hist_rep = json.loads((hist_p.parent / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(schema_mod.config_id(hist_prot["configs"][0]), "b54709addd30")
        self.assertEqual(hist_rep["candidates"][0]["config_id"], "b54709addd30")
        with_opt = {**base, "cos_lr": True, "warmup_epochs": 1.0}
        self.assertNotEqual(schema_mod.config_id(base), schema_mod.config_id(with_opt))

    def test_f09_core_preset_auto_explores_freeze5_with_local_pt(self):
        hw = runtime_info("cpu")
        pt = str(REPO_ROOT / "weights" / "yolo11n.pt")
        with tempfile.TemporaryDirectory() as td:
            cfgs = space_mod.build_space("detection", hw, Path(td) / "work", preset="core", models=[pt], precision="fp32")
            freezes = {c["freeze"] for c in cfgs}
            self.assertIn(0, freezes)
            self.assertIn(5, freezes)

    def test_f09_describe_models_extracts_trainable_parameters_for_pt_checkpoints(self):
        c0 = _det_cfg(model=str(REPO_ROOT / "weights" / "yolo11n.pt"), imgsz=64, freeze=0)
        c5 = _det_cfg(model=str(REPO_ROOT / "weights" / "yolo11n.pt"), imgsz=64, freeze=5)
        c26 = _det_cfg(model=str(REPO_ROOT / "weights" / "yolo26n.pt"), imgsz=64, freeze=0)
        c8 = _det_cfg(model="yolov8n.yaml", imgsz=64, freeze=0)
        descs = describe_models("detection", [c0, c5, c26, c8], classes=4)
        self.assertEqual(len(descs), 3)  # yolo11n.pt, yolo26n.pt, yolov8n.yaml
        for d in descs.values():
            self.assertIn("trainable_parameters", d)
            self.assertGreater(d["parameters"], 0)
            self.assertGreater(d["trainable_parameters"], 0)
            self.assertGreater(d["layers"], 0)
            self.assertGreater(d["gflops_reference_32"], 0.0)
        prot = {"task": "detection", "model_descriptors": descs}
        _, tp0, _ = trainable_parameters_for(prot, c0)
        _, tp5, _ = trainable_parameters_for(prot, c5)
        self.assertLess(tp5, tp0)

    def test_f09_acquisition_distances_include_controlled_hparams(self):
        c1 = {**_det_cfg(imgsz=64), "cos_lr": False, "mosaic": 0.0, "warmup_epochs": 0.0, "close_mosaic": 0}
        c2 = {**_det_cfg(imgsz=64), "cos_lr": True, "mosaic": 1.0, "warmup_epochs": 3.0, "close_mosaic": 5}
        prot = make_det_protocol([c1, c2], acquisition_version=1)
        dists, meta = acq_mod.configuration_distances(prot)
        self.assertGreater(float(dists[0, 1]), 0.0)
        self.assertGreater(meta["dimensions"], 0)

    # =========================================================================
    # F10: R3 Deep Held-Out Test Evaluation (5 tests)
    # =========================================================================
    @classmethod
    def _ensure_shared_f10_eval(cls) -> tuple[Path, Path, dict]:
        if cls._f10_cache is not None:
            return cls._f10_cache
        root = cls._shared_root / "f10_shared"
        yaml_path = make_mini_yolo_dataset(root / "ds", n_train=4, n_val=2, n_test=2, size=64)
        grid = root / "grid.json"
        write_json(grid, [{"imgsz": 64, "batch": 2, "accumulation": 1, "lr0": 0.001, "freeze": 0}])
        out = root / "runs"
        opt = run_cli([
            "optimize", "--data", str(yaml_path), "--device", "cpu",
            "--models", str(REPO_ROOT / "weights" / "yolo11n.pt"),
            "--grid", str(grid), "--full-epochs", "1", "--seeds", "17",
            "--budget-seconds", "90", "--output", str(out),
        ], timeout=180)
        if opt.returncode != 0:
            raise AssertionError(opt.stderr + opt.stdout)
        run_dir = [p for p in out.iterdir() if p.is_dir() and (p / "protocol.json").is_file()][0]
        ev = run_cli(["evaluate-v2", "--run", str(run_dir), "--seed-scope", "all"], timeout=180)
        if ev.returncode != 0:
            raise AssertionError(ev.stderr + ev.stdout)
        eval_dirs = sorted(p for p in (run_dir / "evaluation").iterdir() if p.is_dir())
        eval_json = json.loads((eval_dirs[-1] / "evaluation.json").read_text(encoding="utf-8"))
        cls._f10_cache = (run_dir, eval_dirs[-1], eval_json)
        return cls._f10_cache

    def test_f10_evaluate_v2_records_latency_and_fps(self):
        _, _, ev = self._ensure_shared_f10_eval()
        lat = ev["latency"]
        for k in ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms_per_image", "fps", "batch_size", "precision", "warmup_batches"):
            self.assertIn(k, lat)
        total_ms = lat["total_ms_per_image"]["mean"] if isinstance(lat["total_ms_per_image"], dict) else lat["total_ms_per_image"]
        fps_val = lat["fps"]["mean"] if isinstance(lat["fps"], dict) else lat["fps"]
        self.assertGreater(total_ms, 0.0)
        self.assertGreater(fps_val, 0.0)

    def test_f10_evaluate_v2_exports_pr_curve_json(self):
        _, _, ev = self._ensure_shared_f10_eval()
        self.assertIn("pr_curve", ev)
        pr = ev["pr_curve"]
        self.assertIsInstance(pr, dict)
        self.assertTrue(pr)

        def _flatten_numbers(obj) -> list[float]:
            out: list[float] = []
            if isinstance(obj, (int, float)) and not isinstance(obj, bool):
                val = float(obj)
                if math.isfinite(val):
                    out.append(val)
            elif isinstance(obj, dict):
                for v in obj.values():
                    out.extend(_flatten_numbers(v))
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    out.extend(_flatten_numbers(item))
            return out

        nums = _flatten_numbers(pr)
        self.assertTrue(nums, f"pr_curve contained no finite numeric values: {pr}")
        self.assertTrue(all(0.0 <= x <= 1.0 for x in nums), f"pr_curve values out of [0, 1]: {nums}")

    def test_f10_evaluate_v2_exports_f1_curve_json(self):
        _, _, ev = self._ensure_shared_f10_eval()
        self.assertIn("f1_curve", ev)
        self.assertIsInstance(ev["f1_curve"], dict)
        self.assertTrue(ev["f1_curve"])

    def test_f10_evaluate_v2_exports_normalized_confusion_matrix(self):
        _, _, ev = self._ensure_shared_f10_eval()
        self.assertIn("confusion_matrix", ev)
        cm = ev["confusion_matrix"]
        self.assertIs(cm.get("available"), True)
        self.assertIn("normalized", cm)
        self.assertTrue("raw" in cm or "sum_raw" in cm, f"Expected raw/sum_raw in confusion_matrix: {list(cm.keys())}")
        raw_mat = cm["sum_raw"] if "sum_raw" in cm else cm["raw"]
        raw_sum = sum(float(val) for row in raw_mat for val in row)
        self.assertGreater(raw_sum, 0.0, f"Expected non-zero raw confusion matrix mass, got {raw_mat}")
        per_seed = ev.get("per_seed_results") or {}
        self.assertTrue(per_seed)
        for seed_name, seed_res in per_seed.items():
            self.assertIn("raw", seed_res["confusion_matrix"])
            seed_raw_sum = sum(float(val) for row in seed_res["confusion_matrix"]["raw"] for val in row)
            self.assertGreater(seed_raw_sum, 0.0, f"Seed {seed_name} confusion matrix raw sum was 0.0")
        norm = cm["normalized"]
        self.assertIsInstance(norm, list)
        self.assertTrue(norm)
        norm_sum = sum(float(val) for row in norm for val in row)
        self.assertGreater(norm_sum, 0.0, f"Expected non-zero normalized confusion matrix mass, got {norm}")
        for row in norm:
            for val in row:
                self.assertTrue(math.isfinite(float(val)))
                self.assertGreaterEqual(float(val), 0.0)
                self.assertLessEqual(float(val), 1.0)
        n_cols = len(raw_mat[0])
        populated_cols = 0
        for col in range(n_cols):
            col_raw_sum = sum(float(raw_mat[row_i][col]) for row_i in range(len(raw_mat)))
            col_norm_sum = sum(float(norm[row_i][col]) for row_i in range(len(norm)))
            if col_raw_sum > 0.0:
                populated_cols += 1
                self.assertAlmostEqual(col_norm_sum, 1.0, places=6)
        self.assertGreater(populated_cols, 0)

    def test_f10_evaluate_v2_rect_false_parity_within_1e_12(self):
        run_dir, _, ev = self._ensure_shared_f10_eval()
        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        ckpt = state["results"]["0"][0]["best_checkpoint"]
        direct_map50, direct_map50_95 = offline_yolo_val(
            ckpt, run_dir / "full_data" / "data.yaml", imgsz=64, batch=2
        )
        self.assertLess(abs(ev["map50"] - direct_map50), 1e-12)
        self.assertLess(abs(ev["map50_95"] - direct_map50_95), 1e-12)

    # =========================================================================
    # F11: R4 Anytime Convergence, AUARC & Time-to-Target (5 tests)
    # =========================================================================
    def test_f11_anytime_trajectory_returns_required_schema(self):
        with tempfile.TemporaryDirectory() as td:
            _, manifest = make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29, 37))
            study_dir = Path(td) / "study_n5"
            run_dir = study_dir / "adaptive_full" / "repeat_11"
            ref_dir = study_dir / "reference" / "exhaustive"
            prot = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
            res = diag_mod.anytime_trajectory(run_dir, prot, reference=ref_dir)
            self.assertIn("trajectory", res)
            self.assertIn("auarc", res)
            self.assertIn("time_to_target_95", res)
            self.assertTrue(res["trajectory"])

    def test_f11_anytime_trajectory_monotonic_non_decreasing(self):
        with tempfile.TemporaryDirectory() as td:
            make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29, 37))
            run_dir = Path(td) / "study_n5" / "adaptive_full" / "repeat_11"
            prot = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
            res = diag_mod.anytime_trajectory(run_dir, prot)
            utils = [pt["best_utility"] for pt in res["trajectory"]]
            self.assertEqual(utils, sorted(utils))

    def test_f11_anytime_trajectory_ignores_probe_or_partial_trials(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "partial_probe_run"
            cfgs = [{
                "model": "cnn_classifier", "imgsz": 32, "batch": 16, "accumulation": 2,
                "precision": "fp32", "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.01, "width": 16,
            }]
            build_audited_cls_run(
                run_dir,
                dataset_source="syn",
                sources=source_map(),
                configs=cfgs,
                training_seeds=[17, 29],
                method="adaptive_full",
                search_seed=11,
                mode="online",
                budget_seconds=30.0,
                qualities={0: 0.8},
                times={0: 4.0},
                memories={0: 100.0},
                selected_indices=[],
            )
            prot = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
            res = diag_mod.anytime_trajectory(run_dir, prot)
            self.assertEqual(res["trajectory"], [])

    def test_f11_auarc_finite_and_non_negative_with_reference(self):
        with tempfile.TemporaryDirectory() as td:
            make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29, 37))
            study_dir = Path(td) / "study_n5"
            run_dir = study_dir / "random" / "repeat_11"
            ref_dir = study_dir / "reference" / "exhaustive"
            prot = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
            res = diag_mod.anytime_trajectory(run_dir, prot, reference=ref_dir)
            self.assertIsNotNone(res["auarc"])
            self.assertGreaterEqual(res["auarc"], 0.0)

    def test_f11_time_to_target_95_computed_when_optimal_reached(self):
        with tempfile.TemporaryDirectory() as td:
            make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29, 37))
            study_dir = Path(td) / "study_n5"
            run_dir = study_dir / "adaptive_full" / "repeat_11"
            ref_dir = study_dir / "reference" / "exhaustive"
            prot = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
            res = diag_mod.anytime_trajectory(run_dir, prot, reference=ref_dir)
            self.assertIsNotNone(res["time_to_target_95"])
            self.assertGreaterEqual(res["time_to_target_95"], 0.0)

    # =========================================================================
    # F12: R4 Conditional Paired Non-Parametric Statistics (5 tests)
    # =========================================================================
    def test_f12_schema1_010_study_preserves_exact_hash_when_n_lt_5(self):
        rep = study_mod.analyze_studies(REPO_ROOT / "configs" / "algorithm_010_studies.json")
        self.assertEqual(rep["sha256"], ALGORITHM_010_EXPECTED_SHA256)

    def test_f12_n5_study_emits_paired_wilcoxon_signed_rank(self):
        with tempfile.TemporaryDirectory() as td:
            _, manifest = make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29, 37))
            rep = study_mod.analyze_studies(manifest)
            pair = rep["studies"][0]["paired_comparisons"][0]
            pval = extract_stat_p_value(pair, "objective_regret", "wilcoxon")
            self.assertGreaterEqual(pval, 0.0)
            self.assertLessEqual(pval, 1.0)

    def test_f12_n5_study_emits_exact_sign_flip_permutation_p_value(self):
        with tempfile.TemporaryDirectory() as td:
            _, manifest = make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29, 37))
            rep = study_mod.analyze_studies(manifest)
            pair = rep["studies"][0]["paired_comparisons"][0]
            pval = extract_stat_p_value(pair, "objective_regret", "sign_flip")
            self.assertGreaterEqual(pval, 0.0)
            self.assertLessEqual(pval, 1.0)

    def test_f12_n5_study_emits_holm_bonferroni_adjusted_p_values(self):
        with tempfile.TemporaryDirectory() as td:
            _, manifest = make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29, 37))
            rep = study_mod.analyze_studies(manifest)
            pairs = rep["studies"][0]["paired_comparisons"]
            self.assertGreaterEqual(len(pairs), 2)
            for pair in pairs:
                pval = extract_stat_p_value(pair, "objective_regret", "holm")
                self.assertGreaterEqual(pval, 0.0)
                self.assertLessEqual(pval, 1.0)

    def test_f12_n5_study_includes_explicit_conditional_validity_statement(self):
        with tempfile.TemporaryDirectory() as td:
            _, manifest = make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29, 37))
            out_dir = Path(td) / "alg_out"
            rep = study_mod.analyze_studies(manifest, out_dir)
            md_text = (out_dir / "ALGORITHM_STUDY.md").read_text(encoding="utf-8")
            self.assertFalse(rep["statistical_superiority_established"])
            self.assertIsNone(rep["summary"]["cross_domain_confidence_interval"])
            for mt in rep["summary"]["method_totals"]:
                self.assertIsNone(mt["pooled_regret_or_domain_ci"])
            self.assertIn("observed", md_text.lower())
            self.assertIn("not a confidence interval", rep["summary"]["independence_warning"].lower() + " " + " ".join(rep.get("limitations", [])).lower())

    # =========================================================================
    # F13: R4 Publication-Ready `paper-export` (5 tests)
    # =========================================================================
    def test_f13_paper_export_creates_provenance_json_and_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            _, _ = make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29, 37))
            run_dir = Path(td) / "study_n5" / "adaptive_full" / "repeat_11"
            out = Path(td) / "paper_en"
            summary = paper_mod.export_paper_artifacts(run_dir, out, lang="en")
            prov_path = out / "PROVENANCE.json"
            self.assertTrue(prov_path.is_file())
            self.assertTrue((out / "PAPER_EVIDENCE.json").is_file())
            self.assertTrue(Path(summary["summary_markdown"]).is_file())
            prov = json.loads(prov_path.read_text(encoding="utf-8"))
            self.assertIsInstance(prov, dict)
            # Verify ALL exported artifacts exist and have matching SHA-256 entries in PROVENANCE.json
            verified_by_path: dict[Path, str] = {}

            def _is_sha256(val: object) -> bool:
                return isinstance(val, str) and len(val) == 64 and all(ch in "0123456789abcdef" for ch in val.lower())

            def _walk_check(obj):
                if isinstance(obj, dict):
                    path_val = obj.get("path") or obj.get("file") or obj.get("filename") or obj.get("name")
                    if isinstance(path_val, str) and _is_sha256(obj.get("sha256")):
                        raw_p = Path(path_val)
                        cand = raw_p if raw_p.is_absolute() else (out / raw_p)
                        self.assertTrue(cand.is_file(), f"PROVENANCE.json references missing file: {path_val}")
                        actual = sha256_file(cand)
                        self.assertEqual(actual, obj["sha256"].lower(), f"Hash mismatch for {path_val}")
                        verified_by_path[cand.resolve()] = actual
                        return
                    for k, v in obj.items():
                        if _is_sha256(v) and ("." in k or "/" in k or "\\" in k):
                            raw_p = Path(k)
                            cand = raw_p if raw_p.is_absolute() else (out / raw_p)
                            self.assertTrue(cand.is_file(), f"PROVENANCE.json references missing file: {k}")
                            actual = sha256_file(cand)
                            self.assertEqual(actual, v.lower(), f"Hash mismatch for {k}")
                            verified_by_path[cand.resolve()] = actual
                        else:
                            _walk_check(v)
                elif isinstance(obj, list):
                    for item in obj:
                        _walk_check(item)

            _walk_check(prov)
            svgs = sorted(out.glob("*.svg"))
            self.assertTrue(svgs, f"No exported SVG figures found in {out}")
            required_files = [out / "tables.tex", out / "PAPER_SUMMARY.md", *svgs]
            for req in required_files:
                self.assertTrue(req.is_file(), f"Missing core exported artifact: {req}")
                self.assertIn(req.resolve(), verified_by_path, f"Missing PROVENANCE.json SHA-256 entry for {req.name}")
                self.assertEqual(verified_by_path[req.resolve()], sha256_file(req))

    def test_f13_paper_export_lang_en_and_vi_booktabs_tables_tex(self):
        with tempfile.TemporaryDirectory() as td:
            make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29, 37))
            run_dir = Path(td) / "study_n5" / "adaptive_full" / "repeat_11"
            out_en = Path(td) / "paper_en"
            out_vi = Path(td) / "paper_vi"
            proc_en = run_cli(["paper-export", "--run", str(run_dir), "--output", str(out_en), "--lang", "en"])
            proc_vi = run_cli(["paper-export", "--run", str(run_dir), "--output", str(out_vi), "--lang", "vi"])
            self.assertEqual(proc_en.returncode, 0, msg=proc_en.stderr)
            self.assertEqual(proc_vi.returncode, 0, msg=proc_vi.stderr)
            tex_en = (out_en / "tables.tex").read_text(encoding="utf-8")
            tex_vi = (out_vi / "tables.tex").read_text(encoding="utf-8")
            self.assertIn(r"\toprule", tex_en)
            self.assertIn(r"\toprule", tex_vi)
            self.assertNotEqual(tex_en, tex_vi)

    def test_f13_paper_export_generates_okabe_ito_vector_svgs(self):
        okabe_ito_hexes = ("#0072b2", "#e69f00", "#009e73", "#d55e00", "#cc79a7", "#56b4e9", "#f0e442", "#000000")
        geom_tags = ("<path", "<circle", "<rect", "<line", "<polyline", "<polygon")
        with tempfile.TemporaryDirectory() as td:
            make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29, 37))
            run_dir = Path(td) / "study_n5" / "adaptive_full" / "repeat_11"
            out = Path(td) / "paper_svg"
            paper_mod.export_paper_artifacts(run_dir, out, lang="en")
            for svg_name in ("pareto_front.svg", "learning_curves.svg", "anytime_convergence.svg"):
                svg_path = out / svg_name
                self.assertTrue(svg_path.is_file(), svg_name)
                txt = svg_path.read_text(encoding="utf-8").lower()
                self.assertIn("<svg", txt)
                self.assertTrue(any(tag in txt for tag in geom_tags), f"No geometric elements in {svg_name}")
                self.assertTrue(any(hx in txt for hx in okabe_ito_hexes), f"No Okabe-Ito color in {svg_name}")

    def test_f13_paper_export_generates_eval_svgs_when_evaluation_exists(self):
        run_dir, _, _ = self._ensure_shared_f10_eval()
        out = self._shared_root / "f13_paper_eval"
        proc = run_cli(["paper-export", "--run", str(run_dir), "--output", str(out), "--lang", "en"])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
        for svg_name in ("per_class_ap50.svg", "pr_curve.svg", "confusion_matrix.svg"):
            svg_path = out / svg_name
            self.assertTrue(svg_path.is_file(), svg_name)
            raw_svg = svg_path.read_text(encoding="utf-8")
            txt = raw_svg.lower()
            self.assertIn("<svg", txt)
            self.assertTrue(any(tag in txt for tag in ("<path", "<circle", "<rect", "<line", "<polyline", "<polygon")))
            if svg_name == "confusion_matrix.svg":
                cell_vals = [float(m) for m in re.findall(r"<text[^>]*>\s*([0-9]+\.[0-9]{2})\s*</text>", raw_svg)]
                cell_fills = re.findall(r'<rect[^>]*fill="(#[0-9a-fA-F]{6})"[^>]*stroke="#e5e7eb"', raw_svg)
                self.assertTrue(
                    any(v > 0.0 for v in cell_vals) or any(f.lower() != "#ffffff" for f in cell_fills),
                    f"confusion_matrix.svg rendered a blank/all-zero matrix: vals={cell_vals}, fills={cell_fills}",
                )

    def test_f13_paper_export_does_not_mutate_source_run_dir(self):
        with tempfile.TemporaryDirectory() as td:
            make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29, 37))
            run_dir = Path(td) / "study_n5" / "adaptive_full" / "repeat_11"
            before = {p.relative_to(run_dir).as_posix(): sha256_file(p) for p in sorted(run_dir.rglob("*")) if p.is_file()}
            paper_mod.export_paper_artifacts(run_dir, Path(td) / "paper_out", lang="vi")
            after = {p.relative_to(run_dir).as_posix(): sha256_file(p) for p in sorted(run_dir.rglob("*")) if p.is_file()}
            self.assertEqual(before, after)

    # =========================================================================
    # F14: R5 Tier 1: Rich Terminal UI & Upgraded Wizard (5 tests)
    # =========================================================================
    def test_f14_cli_no_color_flag_emits_zero_ansi(self):
        for cmd in (["--no-color", "doctor", "--device", "cpu"], ["--no-color", "runs", "--root", "runs", "--limit", "3"]):
            proc = run_cli(cmd)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
            assert_zero_ansi(proc.stdout)
            assert_zero_ansi(proc.stderr)

    def test_f14_non_tty_stdout_automatically_emits_zero_ansi(self):
        proc = run_cli(["doctor", "--device", "cpu"])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        assert_zero_ansi(proc.stdout)

    def test_f14_terminal_trial_progress_and_badges_render_cleanly(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            write_json(td_path / "result.json", {"epochs_completed": 3, "quality": 0.72, "memory_mb": 420.0})
            buf = io.StringIO()
            with redirect_stdout(buf):
                tp = term_mod.TrialProgress(
                    td_path, 2, 17, 5, True,
                    {"model": "yolo11n.pt", "imgsz": 64, "batch": 4, "precision": "fp32"},
                    60.0,
                )
                tp.update()
            rendered = buf.getvalue()
            self.assertIn("c02", rendered)
            self.assertIn("17", rendered)
            self.assertIn("5", rendered)
            assert_zero_ansi(rendered)

            # Verify all 4 semantic status badges across summary_view / audit CLI
            make_synthetic_n5_study_dir(td_path, seeds=(11, 17, 23, 29, 37))
            run_dir = td_path / "study_n5" / "adaptive_full" / "repeat_11"
            prot = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
            rep = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))

            rep_opt = {**rep, "decision": {**rep["decision"], "verification_status": "empirical_optimal_in_declared_space"}}
            rep_best = {**rep, "decision": {**rep["decision"], "verification_status": "fully_trained_best_observed"}}
            rep_oom = {
                **rep,
                "candidates": [
                    *rep.get("candidates", []),
                    {
                        "index": 99,
                        "config_id": "deadbeef0001",
                        "config": prot["configs"][0],
                        "complete": False,
                        "completed_seeds": [],
                        "state": "unresolved",
                        "runs": [{"seed": 17, "status": "oom", "mode": "train", "epochs_completed": 0}],
                    },
                ],
            }
            for label, r_obj, expected_badge in (
                ("opt", rep_opt, "EMPIRICAL OPTIMAL"),
                ("best", rep_best, "BEST OBSERVED"),
                ("oom", rep_oom, "OOM"),
            ):
                b_buf = io.StringIO()
                with redirect_stdout(b_buf):
                    term_mod.summary_view(run_dir, prot, r_obj)
                b_out = b_buf.getvalue()
                assert_zero_ansi(b_out)
                self.assertIn(expected_badge.lower(), b_out.lower(), f"Missing badge {expected_badge} for {label}")

            aud = run_cli(["audit", "--run", str(run_dir)])
            self.assertEqual(aud.returncode, 0, msg=aud.stderr + aud.stdout)
            assert_zero_ansi(aud.stdout)
            self.assertIn("AUDIT PASS", aud.stdout.upper())

    def test_f14_terminal_candidate_table_highlights_star_and_pareto(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_dir = REPO_ROOT / "runs" / "nckh_011_map50_verification"
            p_file = sorted(run_dir.rglob("protocol.json"))[0]
            folder = p_file.parent
            prot = json.loads(p_file.read_text(encoding="utf-8"))
            rep = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            term_mod.summary_view(folder, prot, rep)
        output = buf.getvalue()
        assert_zero_ansi(output)
        self.assertIn("★", output)
        self.assertIn("pareto", output.lower())

    def test_f14_wizard_scans_datasets_weights_and_suggests_scales(self):
        from research_pipeline.v2 import wizard as wiz_mod
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            yaml_path = make_mini_yolo_dataset(ws / "ds")
            out_dir = ws / "wiz_out"
            dispatched = []
            # Walk menu -> [1] optimize wizard -> enter dataset, cpu, 10 min, 1 epoch, core, 1 seed,
            # local .pt (accept defaults), advanced=y, map50, topsis, adaptive_full, weights,
            # auto memory, no time limit, accept suggested time_scale & memory_scale, output,
            # action=2 (plan-search), confirm=y -> exit menu [0]
            scripted = "\n".join([
                "1",
                str(yaml_path),
                "cpu",
                "10",
                "1",
                "1",
                "2",
                "1",
                "",
                "y",
                "map50",
                "topsis",
                "adaptive_full",
                "0.6 0.25 0.15",
                "",
                "",
                "",
                "",
                str(out_dir),
                "2",
                "y",
                "0",
            ]) + "\n"
            buf = io.StringIO()
            old_stdin = sys.stdin
            try:
                sys.stdin = io.StringIO(scripted)
                with redirect_stdout(buf):
                    rc = wiz_mod.menu(lambda argv: dispatched.append(argv) or 0, root=str(ws))
            finally:
                sys.stdin = old_stdin
            self.assertEqual(rc, 0)
            wiz_out = buf.getvalue()
            assert_zero_ansi(wiz_out)
            self.assertIn("benchmark", wiz_out.lower())
            self.assertIn("ui", wiz_out.lower())
            self.assertIn(".pt", wiz_out.lower())
            self.assertTrue("params" in wiz_out.lower() or "gflops" in wiz_out.lower())
            self.assertTrue(dispatched)
            cmd = dispatched[0]
            self.assertEqual(cmd[0], "plan-search")
            self.assertIn("--time-scale", cmd)
            self.assertIn("--memory-scale", cmd)
            self.assertIn("--models", cmd)

    # =========================================================================
    # F15: R5 Tier 2: Offline Local Web Dashboard Studio (5 tests)
    # =========================================================================
    def test_f15_ui_binds_loopback_ephemeral_port_and_serves_root(self):
        with tempfile.TemporaryDirectory() as td:
            proc, base_url = launch_ui_server(Path(td), port=0)
            try:
                self.assertTrue(base_url.startswith("http://127.0.0.1:"))
                code, _, body = http_get(base_url + "/")
                self.assertEqual(code, 200)
                self.assertIn("<html", body.lower())
            finally:
                stop_ui_server(proc)

    def test_f15_ui_serves_all_6_functional_views_http_200(self):
        with tempfile.TemporaryDirectory() as td:
            proc, base_url = launch_ui_server(Path(td), port=0)
            try:
                code, _, root_html = http_get(base_url + "/")
                self.assertEqual(code, 200)
                links = [lnk for lnk in extract_nav_links(root_html) if not lnk.startswith("/api/")]
                distinct_views = {"/"} | set(links)
                self.assertGreaterEqual(len(distinct_views), 6, f"Expected 6 distinct UI views, found {sorted(distinct_views)}")
                for link in sorted(distinct_views):
                    c, _, _ = http_get(base_url + link)
                    self.assertEqual(c, 200, f"GET {link} returned {c}")
            finally:
                stop_ui_server(proc)

    def test_f15_ui_is_100_percent_offline_zero_external_cdn_urls(self):
        with tempfile.TemporaryDirectory() as td:
            proc, base_url = launch_ui_server(Path(td), port=0)
            try:
                surf = discover_ui_surface(base_url)
                self.assertEqual(extract_external_urls(surf["root_html"]), [])
                for link in surf["nav_links"][:8]:
                    _, _, body = http_get(base_url + link)
                    self.assertEqual(extract_external_urls(body), [], f"External URL in {link}")
                for asset in surf["asset_links"]:
                    _, _, abody = http_get(base_url + asset)
                    self.assertEqual(extract_external_urls(abody), [], f"External URL in {asset}")
            finally:
                stop_ui_server(proc)

    def test_f15_ui_api_returns_real_run_data_matching_report_json(self):
        with tempfile.TemporaryDirectory() as td:
            runs_dir = Path(td) / "runs"
            make_synthetic_n5_study_dir(runs_dir, seeds=(11, 17, 23, 29, 37))
            sample_run = runs_dir / "study_n5" / "adaptive_full" / "repeat_11"
            expected_rep = json.loads((sample_run / "report.json").read_text(encoding="utf-8"))
            proc, base_url = launch_ui_server(runs_dir, port=0)
            try:
                surf = discover_ui_surface(base_url)
                matched = False
                for route in surf["api_routes"] + ["/api/runs"]:
                    if "{" in route or "<" in route:
                        continue
                    code, _, body = http_get(base_url + route)
                    if code == 200 and expected_rep["protocol_sha256"] in body:
                        matched = True
                        break
                if not matched:
                    # Also check root/runs HTML views if SSR-rendered
                    for link in ["/"] + surf["nav_links"][:6]:
                        code, _, body = http_get(base_url + link)
                        if code == 200 and expected_rep["protocol_sha256"] in body:
                            matched = True
                            break
                self.assertTrue(matched, f"protocol_sha256 not found in any discovered UI API/HTML response: {surf['api_routes']}")
            finally:
                stop_ui_server(proc)

    def test_f15_ui_read_only_never_mutates_run_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            runs_dir = Path(td) / "runs"
            make_synthetic_n5_study_dir(runs_dir, seeds=(11, 17, 23, 29, 37))
            before = {p.relative_to(runs_dir).as_posix(): sha256_file(p) for p in sorted(runs_dir.rglob("*")) if p.is_file()}
            proc, base_url = launch_ui_server(runs_dir, port=0)
            try:
                surf = discover_ui_surface(base_url)
                for link in surf["nav_links"][:8]:
                    http_get(base_url + link)
                for route in surf["api_routes"]:
                    if "{" not in route and "<" not in route:
                        http_get(base_url + route)
            finally:
                stop_ui_server(proc)
            after = {p.relative_to(runs_dir).as_posix(): sha256_file(p) for p in sorted(runs_dir.rglob("*")) if p.is_file()}
            self.assertEqual(before, after)

    # =========================================================================
    # F16: R6 Offline .venv + .venv-cpu E2E Discovery & Cleanliness (5 tests)
    # =========================================================================
    def test_f16_cli_version_reports_1_0_0(self):
        proc = run_cli(["--version"])
        self.assertEqual(proc.returncode, 0)
        self.assertIn("1.0.0", proc.stdout)

    def test_f16_cli_help_lists_dedup_dataset_and_ui_commands(self):
        proc = run_cli(["--help"])
        self.assertEqual(proc.returncode, 0)
        self.assertIn("dedup-dataset", proc.stdout)
        self.assertIn("ui", proc.stdout)

    def test_f16_doctor_json_passes_offline_without_network(self):
        proc = run_cli(["doctor", "--device", "cpu", "--json"], env_overrides={"HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9"})
        self.assertEqual(proc.returncode, 0)
        self.assertTrue(json.loads(proc.stdout)["hardware"]["ml_runtime_ready"])

    def test_f16_pyproject_toml_version_is_1_0_0(self):
        txt = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertRegex(txt, r'version\s*=\s*"1\.0\.0"')

    def test_f16_docs_and_legacy_help_work_offline(self):
        proc = run_cli(["legacy-help"])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        assert_zero_ansi(proc.stdout)

    # =========================================================================
    # F17: R6 Full Workflow Parity & Non-Mutation Invariants (5 tests)
    # =========================================================================
    def test_f17_full_cls_optimize_status_audit_lifecycle_cpu(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cls_dir = make_mini_cls_dataset(root / "cls")
            out = root / "runs"
            opt = run_cli([
                "optimize", "--data", str(cls_dir), "--task", "classification",
                "--device", "cpu", "--space", "compact", "--full-epochs", "1",
                "--seeds", "17", "--budget-seconds", "60", "--output", str(out),
            ], timeout=120)
            self.assertEqual(opt.returncode, 0, msg=opt.stderr + opt.stdout)
            run_dir = [p for p in out.iterdir() if p.is_dir() and (p / "protocol.json").is_file()][0]
            st = run_cli(["status", "--run", str(run_dir), "--json"])
            self.assertEqual(st.returncode, 0)
            self.assertEqual(json.loads(st.stdout)["saved_status"]["phase"], "complete")
            aud = run_cli(["audit", "--run", str(run_dir)])
            self.assertEqual(aud.returncode, 0, msg=aud.stderr + aud.stdout)

    def test_f17_train_v2_deploys_selected_preset_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cls_dir = make_mini_cls_dataset(root / "cls")
            out = root / "runs"
            opt = run_cli([
                "optimize", "--data", str(cls_dir), "--task", "classification",
                "--device", "cpu", "--space", "compact", "--full-epochs", "1",
                "--seeds", "17", "--budget-seconds", "60", "--output", str(out),
            ], timeout=120)
            self.assertEqual(opt.returncode, 0, msg=opt.stderr + opt.stdout)
            run_dir = [p for p in out.iterdir() if p.is_dir() and (p / "protocol.json").is_file()][0]
            before = {p.relative_to(run_dir).as_posix(): sha256_file(p) for p in sorted(run_dir.rglob("*")) if p.is_file()}
            dep_out = root / "deploy"
            tr = run_cli([
                "train-v2", "--selection", str(run_dir / "selected.json"),
                "--device", "cpu", "--epochs", "1", "--budget-seconds", "60",
                "--output", str(dep_out), "--preset", "Balanced",
            ], timeout=120)
            self.assertEqual(tr.returncode, 0, msg=tr.stderr + tr.stdout)
            after = {p.relative_to(run_dir).as_posix(): sha256_file(p) for p in sorted(run_dir.rglob("*")) if p.is_file()}
            self.assertEqual(before, after)

    def test_f17_resume_preserves_protocol_and_audits_cleanly(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cls_dir = make_mini_cls_dataset(root / "cls")
            out = root / "runs"
            opt = run_cli([
                "optimize", "--data", str(cls_dir), "--task", "classification",
                "--device", "cpu", "--space", "compact", "--full-epochs", "1",
                "--seeds", "17", "--budget-seconds", "30", "--output", str(out),
            ], timeout=120)
            self.assertEqual(opt.returncode, 0, msg=opt.stderr + opt.stdout)
            run_dir = [p for p in out.iterdir() if p.is_dir() and (p / "protocol.json").is_file()][0]
            prot_before = sha256_file(run_dir / "protocol.json")
            res = run_cli(["resume", "--run", str(run_dir), "--additional-budget-seconds", "20"], timeout=120)
            self.assertEqual(res.returncode, 0, msg=res.stderr + res.stdout)
            self.assertEqual(prot_before, sha256_file(run_dir / "protocol.json"))
            audit_run(run_dir)

    def test_f17_benchmark_cli_and_benchmark_report_cpu(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cls_dir = make_mini_cls_dataset(root / "cls")
            out = root / "bench"
            bm = run_cli([
                "benchmark", "--data", str(cls_dir), "--task", "classification",
                "--device", "cpu", "--space", "compact", "--full-epochs", "1",
                "--seeds", "17", "--search-seeds", "17", "--methods", "adaptive_full", "default",
                "--budget-seconds", "30", "--reference-budget-seconds", "60", "--output", str(out),
            ], timeout=180)
            self.assertEqual(bm.returncode, 0, msg=bm.stderr + bm.stdout)
            study_dir = [p for p in out.iterdir() if p.is_dir() and (p / "benchmark_protocol.json").is_file()][0]
            rep = run_cli(["benchmark-report", "--run", str(study_dir)], timeout=120)
            self.assertEqual(rep.returncode, 0, msg=rep.stderr + rep.stdout)
            self.assertTrue((study_dir / "BENCHMARK.json").is_file())

    def test_f17_audit_run_detects_tampered_result_json(self):
        with tempfile.TemporaryDirectory() as td:
            make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29, 37))
            run_dir = Path(td) / "study_n5" / "adaptive_full" / "repeat_11"
            res_file = run_dir / "trials" / "attempt_00001" / "result.json"
            data = json.loads(res_file.read_text(encoding="utf-8"))
            data["quality"] = 0.99999
            write_json(res_file, data)
            with self.assertRaises(ValueError):
                audit_run(run_dir)
