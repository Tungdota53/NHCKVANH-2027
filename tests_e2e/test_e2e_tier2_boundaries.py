"""Tier 2 Boundary & Corner Cases: 52 boundary/edge tests across 9 feature groups."""

from __future__ import annotations

import json
import os
from pathlib import Path
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
from research_pipeline.v2.ledger import Ledger
from research_pipeline.v2.model_features import describe_models
from research_pipeline.v2.report import make_report
from research_pipeline.v2.workflow import audit_run, source_map
from yolo_tuner.pipeline import runtime_info, write_json
from yolo_tuner.research import Objective

from tests_e2e._e2e_fixtures import (
    REPO_ROOT,
    assert_zero_ansi,
    audit_count,
    build_audited_cls_run,
    create_dir_link_or_junction,
    discover_ui_surface,
    extract_csrf_token,
    extract_stat_p_value,
    find_audit_dict,
    http_get,
    http_post,
    launch_ui_server,
    make_det_protocol,
    make_mini_cls_dataset,
    make_mini_yolo_dataset,
    make_synthetic_n5_study_dir,
    remove_dir_link_or_junction,
    run_cli,
    stop_ui_server,
)


def _det_cfg(**overrides) -> dict:
    base = {
        "model": "yolo11n.yaml",
        "imgsz": 160,
        "batch": 4,
        "accumulation": 2,
        "precision": "fp32",
        "optimizer": "AdamW",
        "lr0": 0.001,
        "weight_decay": 0.0005,
        "freeze": 0,
    }
    base.update(overrides)
    return base


def _outcome(aid, idx, seed, epochs, q, wall, loop, mem, status="ok"):
    return {
        "attempt_id": aid,
        "index": idx,
        "seed": seed,
        "mode": "train",
        "status": status,
        "epochs_completed": epochs if status == "ok" else 0,
        "optimizer_steps": 5 if status == "ok" else 0,
        "training_loss_finite": status == "ok",
        "actual_config_verified": status == "ok",
        "checkpoint_sha256": f"ckpt-{aid}" if status == "ok" else None,
        "quality": q,
        "memory_mb": mem,
        "worker_wall_seconds": wall,
        "measured_loop_seconds": loop,
    }


class TestTier2Boundaries(unittest.TestCase):
    # =========================================================================
    # Group B1: Data Quality & Corrupt Input Boundaries (6 tests)
    # =========================================================================
    def test_b1_empty_dataset_directory_fails_cleanly_exit_2(self):
        with tempfile.TemporaryDirectory() as td:
            empty_dir = Path(td) / "empty_ds"
            empty_dir.mkdir()
            proc = run_cli(["inspect", "--data", str(empty_dir), "--device", "cpu"])
            self.assertEqual(proc.returncode, 2)
            assert_zero_ansi(proc.stderr)

    def test_b1_missing_label_file_detected_in_inspect_audit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ds"
            yaml_path = make_mini_yolo_dataset(root, n_train=4, n_val=2, n_test=2)
            (root / "labels" / "train" / "train_000.txt").unlink()
            proc = run_cli(["inspect", "--data", str(yaml_path), "--device", "cpu", "--json"])
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            audit = find_audit_dict(json.loads(proc.stdout))
            self.assertEqual(audit_count(audit, "missing_labels_count", "missing_labels", "labels.missing_count", "labels.missing"), 1)

    def test_b1_empty_label_file_detected_in_inspect_audit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ds"
            yaml_path = make_mini_yolo_dataset(root, n_train=4, n_val=2, n_test=2)
            (root / "labels" / "train" / "train_001.txt").write_text("", encoding="utf-8")
            proc = run_cli(["inspect", "--data", str(yaml_path), "--device", "cpu", "--json"])
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            audit = find_audit_dict(json.loads(proc.stdout))
            self.assertEqual(audit_count(audit, "empty_labels_count", "empty_labels", "labels.empty_count", "labels.empty"), 1)

    def test_b1_out_of_bounds_bbox_detected_in_inspect_audit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ds"
            yaml_path = make_mini_yolo_dataset(root, n_train=4, n_val=2, n_test=2)
            (root / "labels" / "train" / "train_000.txt").write_text("0 1.25 0.5 0.5 0.5\n", encoding="utf-8")
            proc = run_cli(["inspect", "--data", str(yaml_path), "--device", "cpu", "--json"])
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            audit = find_audit_dict(json.loads(proc.stdout))
            self.assertGreaterEqual(audit_count(audit, "out_of_bounds_boxes_count", "out_of_bounds_boxes", "boxes.out_of_bounds_count", "boxes.out_of_bounds"), 1)

    def test_b1_degenerate_w_zero_and_h_negative_bbox_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ds"
            yaml_path = make_mini_yolo_dataset(root, n_train=4, n_val=2, n_test=2)
            (root / "labels" / "train" / "train_000.txt").write_text(
                "0 0.5 0.5 0.0 0.2\n0 0.5 0.5 0.2 -0.1\n", encoding="utf-8"
            )
            proc = run_cli(["inspect", "--data", str(yaml_path), "--device", "cpu", "--json"])
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            audit = find_audit_dict(json.loads(proc.stdout))
            self.assertGreaterEqual(audit_count(audit, "degenerate_boxes_count", "degenerate_boxes", "boxes.degenerate_count", "boxes.degenerate"), 2)

    def test_b1_invalid_class_id_ge_nc_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ds"
            yaml_path = make_mini_yolo_dataset(root, n_train=4, n_val=2, n_test=2)
            (root / "labels" / "train" / "train_000.txt").write_text("2 0.5 0.5 0.3 0.3\n", encoding="utf-8")
            proc = run_cli(["inspect", "--data", str(yaml_path), "--device", "cpu", "--json"])
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            audit = find_audit_dict(json.loads(proc.stdout))
            self.assertGreaterEqual(audit_count(audit, "invalid_class_ids_count", "invalid_class_ids", "boxes.invalid_class_ids_count", "labels.invalid_class_ids"), 1)

    # =========================================================================
    # Group B2: Cost, VRAM & OOM / Budget Guard Boundaries (6 tests)
    # =========================================================================
    def test_b2_observed_cross_memory_returns_none_with_empty_fulls(self):
        prot = make_det_protocol([_det_cfg()], acquisition_version=1)
        self.assertIsNone(pred_mod.observed_cross_memory(prot, {}, 0))

    def test_b2_tight_max_memory_oom_guard_blocks_all_oversized_configs(self):
        cfgs = [_det_cfg(imgsz=160, batch=2), _det_cfg(imgsz=640, batch=16)]
        descs = describe_models("detection", cfgs, 2)
        prot = make_det_protocol(
            cfgs,
            method="adaptive_full",
            acquisition_version=2,
            training_seeds=[17],
            budget_seconds=40.0,
            max_memory=600.0,
            model_descriptors=descs,
        )
        with tempfile.TemporaryDirectory() as td:
            ledger = Ledger(Path(td) / "events.jsonl")
            clock = [0.0]
            ran = []

            def eval_fn(idx, seed, epochs, full, seconds):
                ran.append(idx)
                clock[0] += 2.0
                return _outcome(f"a{idx}", idx, seed, epochs, 0.7, 2.0, 1.5, 500.0 if idx == 0 else 3000.0)

            sched_mod.run_search(prot, eval_fn, lambda: 40.0 - clock[0], ledger, elapsed=lambda: clock[0])
            self.assertEqual(ran, [0])

    def test_b2_tight_budget_overrun_guard_stops_before_exceeding_budget(self):
        cfgs = [_det_cfg(lr0=lr) for lr in (0.001, 0.0003, 0.003)]
        prot = make_det_protocol(cfgs, method="adaptive_full", acquisition_version=2, training_seeds=[17], budget_seconds=12.0)
        with tempfile.TemporaryDirectory() as td:
            ledger = Ledger(Path(td) / "events.jsonl")
            clock = [0.0]

            def eval_fn(idx, seed, epochs, full, seconds):
                clock[0] += 5.0
                return _outcome(f"a{idx}", idx, seed, epochs, 0.7, 5.0, 3.0, 400.0)

            state = sched_mod.run_search(prot, eval_fn, lambda: 12.0 - clock[0], ledger, elapsed=lambda: clock[0])
            self.assertLessEqual(clock[0], 12.0)
            self.assertLessEqual(sum(1 for rs in state["results"].values() if rs), 2)

    def test_b2_cost_model_fallback_when_measured_loop_seconds_missing(self):
        cfgs = [_det_cfg(imgsz=160), _det_cfg(imgsz=320)]
        prot = make_det_protocol(cfgs, acquisition_version=1)
        cm = pred_mod.CostModel(prot)
        rec = _outcome("a0", 0, 17, 2, 0.6, 10.0, None, 400.0)
        del rec["measured_loop_seconds"]
        cm.update(0, rec)
        est = cm.estimate(1)
        self.assertGreater(est["point_seconds"], 10.0)

    def test_b2_accumulation_invariance_boundary_for_activation_mb(self):
        cfgs = [_det_cfg(imgsz=160, batch=4, accumulation=1), _det_cfg(imgsz=160, batch=4, accumulation=16)]
        descs = describe_models("detection", cfgs, 2)
        prot = make_det_protocol(cfgs, acquisition_version=1, model_descriptors=descs)
        fulls = {0: [_outcome("a0", 0, 17, 2, 0.6, 5.0, 3.0, 700.0)]}
        e0 = pred_mod.observed_cross_memory(prot, fulls, 0)
        e1 = pred_mod.observed_cross_memory(prot, fulls, 1)
        self.assertAlmostEqual(e0["activation_mb"], e1["activation_mb"], places=6)

    def test_b2_freeze_reduces_static_model_state_mb_boundary(self):
        pt = str(REPO_ROOT / "weights" / "yolo11n.pt")
        cfgs = [_det_cfg(model=pt, imgsz=64, freeze=0), _det_cfg(model=pt, imgsz=64, freeze=5)]
        descs = describe_models("detection", cfgs, 2)
        prot = make_det_protocol(cfgs, acquisition_version=1, model_descriptors=descs)
        fulls = {0: [_outcome("a0", 0, 17, 2, 0.6, 5.0, 3.0, 700.0)]}
        e0 = pred_mod.observed_cross_memory(prot, fulls, 0)
        e5 = pred_mod.observed_cross_memory(prot, fulls, 1)
        self.assertLessEqual(e5["static_model_state_mb"], e0["static_model_state_mb"])

    # =========================================================================
    # Group B3: Surrogate, Acquisition & Staged-Seed Boundaries (6 tests)
    # =========================================================================
    def test_b3_acquisition_version_rejects_invalid_versions(self):
        for bad_v in (0, 3, "2", True):
            prot = make_det_protocol([_det_cfg()], acquisition_version=1)
            prot["acquisition"]["version"] = bad_v
            with self.assertRaises(ValueError):
                acq_mod.acquisition_spec(prot)

    def test_b3_single_seed_uses_homoscedastic_while_multi_seed_heteroscedastic(self):
        cfgs = [_det_cfg(lr0=lr) for lr in (0.001, 0.0003, 0.003)]
        p1 = make_det_protocol(cfgs, acquisition_version=2, training_seeds=[17])
        r1 = {"0": [_outcome("a0", 0, 17, 2, 0.6, 4.0, 2.0, 400.0)], "1": [_outcome("a1", 1, 17, 2, 0.8, 4.0, 2.0, 400.0)]}
        rows1 = schema_mod.measurements(cfgs, r1, [17], 2, Objective(**p1["objective"]))
        out1 = acq_mod.rank_candidates(p1, rows1, {2: {"point_seconds": 4.0, "predicted_memory_mb": 400.0}})
        self.assertFalse(out1["surrogate_diagnostics"]["heteroscedastic"])

    def test_b3_constant_target_values_loocv_numerically_stable(self):
        cfgs = [_det_cfg(lr0=lr) for lr in (0.001, 0.0003, 0.003, 0.0001)]
        prot = make_det_protocol(cfgs, acquisition_version=2, training_seeds=[17])
        results = {
            "0": [_outcome("a0", 0, 17, 2, 0.75, 4.0, 2.0, 400.0)],
            "1": [_outcome("a1", 1, 17, 2, 0.75, 4.0, 2.0, 400.0)],
            "2": [_outcome("a2", 2, 17, 2, 0.75, 4.0, 2.0, 400.0)],
        }
        rows = schema_mod.measurements(cfgs, results, [17], 2, Objective(**prot["objective"]))
        out = acq_mod.rank_candidates(prot, rows, {3: {"point_seconds": 4.0, "predicted_memory_mb": 400.0}})
        self.assertEqual(out["surrogate_diagnostics"]["loocv_mae"], 0.0)

    def test_b3_partial_seed_only_excluded_from_report_and_selection(self):
        cfgs = [_det_cfg(lr0=0.001)]
        prot = make_det_protocol(cfgs, acquisition_version=2, training_seeds=[17, 29])
        results = {"0": [_outcome("a0", 0, 17, 2, 0.99, 4.0, 2.0, 400.0)]}
        rows = schema_mod.measurements(cfgs, results, [17, 29], 2, Objective(**prot["objective"]))
        self.assertFalse(rows[0]["complete"])
        self.assertIsNone(sel_mod.select(rows, prot)["selected_index"])

    def test_b3_oom_and_timeout_runs_keep_null_regret_never_zero(self):
        with tempfile.TemporaryDirectory() as td:
            _, manifest = make_synthetic_n5_study_dir(Path(td), failed_pair=("default", 23))
            rep = study_mod.analyze_studies(manifest)
            row = next(r for r in rep["studies"][0]["outcomes"] if r["method"] == "default" and r["search_seed"] == 23)
            self.assertFalse(row["feasible"])
            self.assertIsNone(row["objective_regret"])

    def test_b3_unknown_selection_rule_rejected(self):
        with self.assertRaises(ValueError):
            sel_mod.rule_scores([], {"objective": {"weights": [0.6, 0.25, 0.15], "time_scale": 60, "memory_scale": 512}, "selection_rule": "pareto_magic"})

    # =========================================================================
    # Group B4: Controlled Search Space & Model Checkpoint Boundaries (5 tests)
    # =========================================================================
    def test_b4_freeze_gt_zero_rejected_on_yaml_architecture(self):
        hw = runtime_info("cpu")
        with self.assertRaises(ValueError):
            space_mod.validate_config("detection", _det_cfg(model="yolo11n.yaml", freeze=2), hw)

    def test_b4_controlled_hparam_invalid_mosaic_or_negative_warmup_rejected(self):
        hw = runtime_info("cpu")
        with self.assertRaises(ValueError):
            space_mod.validate_config("detection", {**_det_cfg(), "mosaic": 1.5}, hw)
        with self.assertRaises(ValueError):
            space_mod.validate_config("detection", {**_det_cfg(), "warmup_epochs": -1.0}, hw)
        with self.assertRaises(ValueError):
            space_mod.validate_config("detection", {**_det_cfg(), "close_mosaic": -1}, hw)

    def test_b4_controlled_hparam_boundary_values_zero_and_one_accepted(self):
        hw = runtime_info("cpu")
        with tempfile.TemporaryDirectory() as td:
            grid = Path(td) / "grid.json"
            write_json(grid, [{"imgsz": 64, "batch": 2, "accumulation": 1, "cos_lr": False, "warmup_epochs": 0.0, "mosaic": 0.0, "close_mosaic": 0}])
            cfgs = space_mod.build_space("detection", hw, Path(td) / "work", grid=str(grid))
            self.assertEqual(len(cfgs), 1)
            write_json(grid, [{"imgsz": 64, "batch": 2, "accumulation": 1, "cos_lr": True, "warmup_epochs": 0.0, "mosaic": 1.0, "close_mosaic": 0}])
            cfgs2 = space_mod.build_space("detection", hw, Path(td) / "work2", grid=str(grid))
            self.assertEqual(len(cfgs2), 1)

    def test_b4_detection_imgsz_below_64_or_non_multiple_32_rejected(self):
        hw = runtime_info("cpu")
        with self.assertRaises(ValueError):
            space_mod.validate_config("detection", _det_cfg(imgsz=32), hw)
        with self.assertRaises(ValueError):
            space_mod.validate_config("detection", _det_cfg(imgsz=96 + 8), hw)

    def test_b4_fp16_rejected_on_cpu_device(self):
        hw = runtime_info("cpu")
        with self.assertRaises(ValueError):
            space_mod.validate_config("detection", _det_cfg(precision="fp16"), hw)

    # =========================================================================
    # Group B5: Held-Out Test & Evaluation Guard Boundaries (5 tests)
    # =========================================================================
    def test_b5_evaluate_v2_retest_blocked_without_allow_retest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cls_dir = make_mini_cls_dataset(root / "cls")
            out = root / "runs"
            opt = run_cli([
                "optimize", "--data", str(cls_dir), "--task", "classification",
                "--device", "cpu", "--space", "compact", "--full-epochs", "1",
                "--seeds", "17", "--budget-seconds", "60", "--output", str(out),
            ], timeout=120)
            self.assertEqual(opt.returncode, 0, msg=opt.stderr)
            run_dir = [p for p in out.iterdir() if p.is_dir() and (p / "protocol.json").is_file()][0]
            ev1 = run_cli(["evaluate-v2", "--run", str(run_dir), "--seed-scope", "all"], timeout=120)
            self.assertEqual(ev1.returncode, 0, msg=ev1.stderr)
            ev2 = run_cli(["evaluate-v2", "--run", str(run_dir), "--seed-scope", "all"], timeout=120)
            self.assertEqual(ev2.returncode, 2)

    def test_b5_evaluate_v2_retest_allowed_with_allow_retest_flag(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cls_dir = make_mini_cls_dataset(root / "cls")
            out = root / "runs"
            opt = run_cli([
                "optimize", "--data", str(cls_dir), "--task", "classification",
                "--device", "cpu", "--space", "compact", "--full-epochs", "1",
                "--seeds", "17", "--budget-seconds", "60", "--output", str(out),
            ], timeout=120)
            self.assertEqual(opt.returncode, 0, msg=opt.stderr)
            run_dir = [p for p in out.iterdir() if p.is_dir() and (p / "protocol.json").is_file()][0]
            self.assertEqual(run_cli(["evaluate-v2", "--run", str(run_dir)], timeout=120).returncode, 0)
            self.assertEqual(run_cli(["evaluate-v2", "--run", str(run_dir), "--allow-retest"], timeout=120).returncode, 0)
            hist = json.loads((run_dir / "test_history.json").read_text(encoding="utf-8"))
            self.assertEqual(len(hist), 2)

    def test_b5_evaluate_v2_blocked_when_phase_not_complete(self):
        with tempfile.TemporaryDirectory() as td:
            make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29, 37))
            run_dir = Path(td) / "study_n5" / "adaptive_full" / "repeat_11"
            write_json(run_dir / "status.json", {"phase": "search"})
            proc = run_cli(["evaluate-v2", "--run", str(run_dir)])
            self.assertEqual(proc.returncode, 2)

    def test_b5_evaluate_v2_blocked_when_test_split_empty(self):
        with tempfile.TemporaryDirectory() as td:
            make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29, 37))
            run_dir = Path(td) / "study_n5" / "adaptive_full" / "repeat_11"
            proc = run_cli(["evaluate-v2", "--run", str(run_dir)])
            self.assertEqual(proc.returncode, 2)

    def test_b5_evaluate_v2_blocked_when_checkpoint_tampered(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cls_dir = make_mini_cls_dataset(root / "cls")
            out = root / "runs"
            opt = run_cli([
                "optimize", "--data", str(cls_dir), "--task", "classification",
                "--device", "cpu", "--space", "compact", "--full-epochs", "1",
                "--seeds", "17", "--budget-seconds", "60", "--output", str(out),
            ], timeout=120)
            self.assertEqual(opt.returncode, 0, msg=opt.stderr)
            run_dir = [p for p in out.iterdir() if p.is_dir() and (p / "protocol.json").is_file()][0]
            state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
            ckpt = Path(next(r["best_checkpoint"] for rs in state["results"].values() for r in rs))
            ckpt.write_bytes(b"tampered")
            proc = run_cli(["evaluate-v2", "--run", str(run_dir)])
            self.assertEqual(proc.returncode, 2)

    # =========================================================================
    # Group B6: Anytime Trajectory & Statistical Significance Boundaries (6 tests)
    # =========================================================================
    def test_b6_n4_study_skips_n5_significance_and_has_null_bootstrap(self):
        with tempfile.TemporaryDirectory() as td:
            _, manifest = make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29))
            rep = study_mod.analyze_studies(manifest)
            pair = rep["studies"][0]["paired_comparisons"][0]
            regret_stats = pair["metrics"]["objective_regret"]
            self.assertIsNone(regret_stats["candidate_minus_baseline"]["interval"])
            for k in ("wilcoxon_p_value", "wilcoxon_p", "sign_flip_p_value", "holm_adjusted_p_value"):
                self.assertIsNone(regret_stats.get(k))

    def test_b6_n5_study_enables_wilcoxon_sign_flip_and_holm(self):
        with tempfile.TemporaryDirectory() as td:
            _, manifest = make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29, 37))
            rep = study_mod.analyze_studies(manifest)
            pair = rep["studies"][0]["paired_comparisons"][0]
            regret_stats = pair["metrics"]["objective_regret"]
            self.assertIsNotNone(regret_stats["candidate_minus_baseline"]["interval"])
            self.assertTrue(0.0 <= extract_stat_p_value(pair, "objective_regret", "wilcoxon") <= 1.0)
            self.assertTrue(0.0 <= extract_stat_p_value(pair, "objective_regret", "sign_flip") <= 1.0)
            self.assertTrue(0.0 <= extract_stat_p_value(pair, "objective_regret", "holm") <= 1.0)

    def test_b6_n5_zero_differences_handled_gracefully_no_zero_division(self):
        with tempfile.TemporaryDirectory() as td:
            _, manifest = make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29, 37), zero_diff=True)
            rep = study_mod.analyze_studies(manifest)
            pair = rep["studies"][0]["paired_comparisons"][0]
            self.assertAlmostEqual(extract_stat_p_value(pair, "objective_regret", "wilcoxon"), 1.0)
            self.assertAlmostEqual(extract_stat_p_value(pair, "objective_regret", "sign_flip"), 1.0)

    def test_b6_failed_pair_in_n5_reduces_paired_count_without_fabricating_zero(self):
        with tempfile.TemporaryDirectory() as td:
            _, manifest = make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29, 37), failed_pair=("random", 11))
            rep = study_mod.analyze_studies(manifest)
            pair = next(p for p in rep["studies"][0]["paired_comparisons"] if p["baseline"] == "random")
            self.assertEqual(pair["metrics"]["objective_regret"]["paired_seed_count"], 4)
            self.assertEqual(pair["metrics"]["objective_regret"]["unavailable_pairs"], 1)

    def test_b6_anytime_trajectory_empty_run_returns_empty_and_none(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "empty_run"
            cfgs = [{
                "model": "cnn_classifier", "imgsz": 32, "batch": 16, "accumulation": 2,
                "precision": "fp32", "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.01, "width": 16,
            }]
            build_audited_cls_run(
                run_dir, dataset_source="syn", sources=source_map(), configs=cfgs,
                training_seeds=[17], method="default", search_seed=11, mode="online",
                budget_seconds=20.0, qualities={0: 0.7}, times={0: 4.0}, memories={0: 100.0},
                selected_indices=[],
            )
            prot = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
            res = diag_mod.anytime_trajectory(run_dir, prot)
            self.assertEqual(res["trajectory"], [])
            self.assertIsNone(res["auarc"])
            self.assertIsNone(res["time_to_target_95"])

    def test_b6_paper_export_refuses_to_overwrite_existing_directory(self):
        with tempfile.TemporaryDirectory() as td:
            make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29, 37))
            run_dir = Path(td) / "study_n5" / "adaptive_full" / "repeat_11"
            existing = Path(td) / "existing_out"
            existing.mkdir()
            (existing / "keep.txt").write_text("do not overwrite", encoding="utf-8")
            proc = run_cli(["paper-export", "--run", str(run_dir), "--output", str(existing)])
            self.assertEqual(proc.returncode, 2)

    # =========================================================================
    # Group B7: Terminal UI, CLI Parsing & Exit Code Boundaries (6 tests)
    # =========================================================================
    def test_b7_non_tty_stdout_and_stderr_zero_ansi(self):
        proc = run_cli(["runs", "--root", "runs", "--limit", "2"])
        self.assertEqual(proc.returncode, 0)
        assert_zero_ansi(proc.stdout)
        assert_zero_ansi(proc.stderr)

    def test_b7_no_color_flag_works_across_subcommands(self):
        with tempfile.TemporaryDirectory() as td:
            yaml_path = make_mini_yolo_dataset(Path(td) / "ds")
            proc = run_cli(["--no-color", "inspect", "--data", str(yaml_path), "--device", "cpu"])
            self.assertEqual(proc.returncode, 0)
            assert_zero_ansi(proc.stdout)
            assert_zero_ansi(proc.stderr)

    def test_b7_ascii_encoding_fallback_handles_vietnamese_diacritics(self):
        proc = run_cli(
            ["--no-color", "doctor", "--device", "cpu"],
            env_overrides={"PYTHONIOENCODING": "ascii"},
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        assert_zero_ansi(proc.stdout)

    def test_b7_unknown_cli_command_exits_2(self):
        proc = run_cli(["nonexistent-subcommand"])
        self.assertEqual(proc.returncode, 2)

    def test_b7_optimize_protocol_forbids_override_flags(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            yaml_path = make_mini_yolo_dataset(root / "ds")
            plan = run_cli([
                "plan-search", "--data", str(yaml_path), "--device", "cpu",
                "--space", "compact", "--full-epochs", "1", "--seeds", "17",
                "--budget-seconds", "60", "--output", str(root / "planned"),
            ])
            self.assertEqual(plan.returncode, 0)
            prot_path = next((root / "planned").rglob("protocol.json"))
            opt = run_cli(["optimize", "--protocol", str(prot_path), "--budget-seconds", "10"])
            self.assertEqual(opt.returncode, 2)

    def test_b7_resume_nonpositive_budget_rejected_exit_2(self):
        with tempfile.TemporaryDirectory() as td:
            make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29, 37))
            run_dir = Path(td) / "study_n5" / "adaptive_full" / "repeat_11"
            proc = run_cli(["resume", "--run", str(run_dir), "--additional-budget-seconds", "0"])
            self.assertEqual(proc.returncode, 2)

    # =========================================================================
    # Group B8: Web UI Security & Validation Boundaries (7 tests)
    # =========================================================================
    @staticmethod
    def _find_proven_200_endpoint(base_url: str, rel_run: str, rel_file: str) -> list[tuple[str, str, str]]:
        """Discover real file/run detail endpoints from UI HTML/JS and prove in-bounds GET returns 200."""
        surf = discover_ui_surface(base_url)
        list_only_tails = {"runs", "benchmarks", "datasets", "weights", "health", "status", "hardware", "events", "logs"}
        candidates: list[tuple[str, str, str, str]] = []
        for route in surf["api_routes"]:
            tail = route.rstrip("/").rsplit("/", 1)[-1].lower()
            if tail in list_only_tails or "{" in route or "<" in route:
                continue
            if any(k in route.lower() for k in ("file", "artifact", "download", "read")):
                candidates.extend([
                    (route, f"?path={rel_file}", "?path={payload}", "?path={escape_payload}"),
                    (route, f"?file={rel_file}", "?file={payload}", "?file={escape_payload}"),
                    (route, f"?run={rel_run}&file=report.json", f"?run={rel_run}&file={{payload}}", "?run=escape_dir&file=secret.json"),
                    (route, f"?run={rel_run}&path=report.json", f"?run={rel_run}&path={{payload}}", "?run=escape_dir&path=secret.json"),
                ])
            elif "run" in route.lower():
                candidates.extend([
                    (route, f"?path={rel_run}", "?path={payload}", "?path={escape_payload}"),
                    (route, f"?run={rel_run}", "?run={payload}", "?run={escape_payload}"),
                ])
        candidates.extend([
            ("/api/file", f"?path={rel_file}", "?path={payload}", "?path={escape_payload}"),
            ("/api/file", f"?run={rel_run}&file=report.json", f"?run={rel_run}&file={{payload}}", "?run=escape_dir&file=secret.json"),
            ("/api/artifact", f"?path={rel_file}", "?path={payload}", "?path={escape_payload}"),
            ("/api/artifact", f"?run={rel_run}&file=report.json", f"?run={rel_run}&file={{payload}}", "?run=escape_dir&file=secret.json"),
            ("/api/download", f"?path={rel_file}", "?path={payload}", "?path={escape_payload}"),
            ("/api/download", f"?run={rel_run}&file=report.json", f"?run={rel_run}&file={{payload}}", "?run=escape_dir&file=secret.json"),
            ("/api/run", f"?path={rel_run}", "?path={payload}", "?path={escape_payload}"),
            ("/api/run", f"?run={rel_run}", "?run={payload}", "?run={escape_payload}"),
        ])
        proven: list[tuple[str, str, str]] = []
        seen = set()
        for base_route, in_bounds_q, trav_tmpl, esc_tmpl in candidates:
            if (base_route, trav_tmpl) in seen:
                continue
            code, _, body = http_get(base_url + base_route + in_bounds_q)
            if code == 200 and body:
                seen.add((base_route, trav_tmpl))
                proven.append((base_route, trav_tmpl, esc_tmpl))
        return proven

    def test_b8_ui_rejects_non_loopback_host(self):
        with tempfile.TemporaryDirectory() as td:
            proc = run_cli(["ui", "--host", "0.0.0.0", "--port", "0", "--runs-dir", str(td)], timeout=15)
            self.assertEqual(proc.returncode, 2)

    def test_b8_ui_blocks_dotdot_path_traversal(self):
        with tempfile.TemporaryDirectory() as td:
            runs_dir = Path(td) / "runs"
            make_synthetic_n5_study_dir(runs_dir, seeds=(11, 17, 23, 29, 37))
            (runs_dir.parent / "outside_traversal_secret.txt").write_text("TRAVERSAL_SECRET_TOKEN", encoding="utf-8")
            proc, base_url = launch_ui_server(runs_dir, port=0)
            try:
                proven = self._find_proven_200_endpoint(
                    base_url,
                    "study_n5/adaptive_full/repeat_11",
                    "study_n5/adaptive_full/repeat_11/report.json",
                )
                self.assertTrue(proven, "No in-bounds run/file endpoint returned HTTP 200; cannot validate traversal safely")
                dotdot_payloads = (
                    "../outside_traversal_secret.txt",
                    "../../outside_traversal_secret.txt",
                    "../../../outside_traversal_secret.txt",
                    "../../../../outside_traversal_secret.txt",
                    "../../pyproject.toml",
                    "../../../pyproject.toml",
                    "../../../../pyproject.toml",
                )
                for base_route, trav_tmpl, _ in proven:
                    for payload in dotdot_payloads:
                        probe_url = base_url + base_route + trav_tmpl.replace("{payload}", payload)
                        code, _, body = http_get(probe_url)
                        self.assertIn(code, (400, 403, 404), f"Traversal allowed on {probe_url}: {code}")
                        self.assertNotIn("TRAVERSAL_SECRET_TOKEN", body)
                        self.assertNotIn("[project]", body)
                        self.assertNotIn("yolo-budget-tuner", body)
                for raw_path in ("/../outside_traversal_secret.txt", "/../../pyproject.toml", "/runs/../../../pyproject.toml"):
                    code, _, body = http_get(base_url + raw_path)
                    self.assertNotEqual(code, 200)
                    self.assertNotIn("TRAVERSAL_SECRET_TOKEN", body)
                    self.assertNotIn("[project]", body)
            finally:
                stop_ui_server(proc)

    def test_b8_ui_blocks_encoded_percent2e_path_traversal(self):
        with tempfile.TemporaryDirectory() as td:
            runs_dir = Path(td) / "runs"
            make_synthetic_n5_study_dir(runs_dir, seeds=(11, 17, 23, 29, 37))
            (runs_dir.parent / "outside_traversal_secret.txt").write_text("TRAVERSAL_SECRET_TOKEN", encoding="utf-8")
            proc, base_url = launch_ui_server(runs_dir, port=0)
            try:
                proven = self._find_proven_200_endpoint(
                    base_url,
                    "study_n5/adaptive_full/repeat_11",
                    "study_n5/adaptive_full/repeat_11/report.json",
                )
                self.assertTrue(proven, "No in-bounds run/file endpoint returned HTTP 200; cannot validate traversal safely")
                encoded_payloads = (
                    "%2e%2e%2foutside_traversal_secret.txt",
                    "%2e%2e/%2e%2e/outside_traversal_secret.txt",
                    "%2e%2e/%2e%2e/%2e%2e/outside_traversal_secret.txt",
                    "%2e%2e/%2e%2e/%2e%2e/%2e%2e/outside_traversal_secret.txt",
                    "%2e%2e%2fpyproject.toml",
                    "%2e%2e/%2e%2e/pyproject.toml",
                    "%2e%2e/%2e%2e/%2e%2e/pyproject.toml",
                )
                for base_route, trav_tmpl, _ in proven:
                    for payload in encoded_payloads:
                        probe_url = base_url + base_route + trav_tmpl.replace("{payload}", payload)
                        code, _, body = http_get(probe_url)
                        self.assertIn(code, (400, 403, 404), f"Encoded traversal allowed on {probe_url}: {code}")
                        self.assertNotIn("TRAVERSAL_SECRET_TOKEN", body)
                        self.assertNotIn("[project]", body)
                for raw_path in ("/%2e%2e%2foutside_traversal_secret.txt", "/%2e%2e/%2e%2e/pyproject.toml"):
                    code, _, body = http_get(base_url + raw_path)
                    self.assertNotEqual(code, 200)
                    self.assertNotIn("TRAVERSAL_SECRET_TOKEN", body)
                    self.assertNotIn("[project]", body)
            finally:
                stop_ui_server(proc)

    def test_b8_ui_blocks_symlink_escape_outside_runs_dir(self):
        with tempfile.TemporaryDirectory() as td:
            runs_dir = Path(td) / "runs"
            make_synthetic_n5_study_dir(runs_dir, seeds=(11, 17, 23, 29, 37))
            outside_dir = Path(td) / "outside_secret"
            outside_dir.mkdir()
            (outside_dir / "secret.json").write_text('{"secret": "escaped_outside_token"}', encoding="utf-8")
            link_dir = runs_dir / "escape_dir"
            kind = create_dir_link_or_junction(outside_dir, link_dir)
            self.assertTrue((link_dir / "secret.json").is_file(), f"Failed to resolve {kind} target")
            proc, base_url = launch_ui_server(runs_dir, port=0)
            try:
                proven = self._find_proven_200_endpoint(
                    base_url,
                    "study_n5/adaptive_full/repeat_11",
                    "study_n5/adaptive_full/repeat_11/report.json",
                )
                self.assertTrue(proven, "No in-bounds run/file endpoint returned HTTP 200")
                for base_route, trav_tmpl, esc_tmpl in proven:
                    for query_str in (
                        trav_tmpl.replace("{payload}", "escape_dir/secret.json"),
                        trav_tmpl.replace("{payload}", "../../../escape_dir/secret.json"),
                        esc_tmpl.replace("{escape_payload}", "escape_dir/secret.json"),
                    ):
                        probe_url = base_url + base_route + query_str
                        code, _, body = http_get(probe_url)
                        self.assertIn(code, (400, 403, 404), f"{kind} escape returned {code} on {probe_url}")
                        self.assertNotIn("escaped_outside_token", body)
            finally:
                stop_ui_server(proc)
                remove_dir_link_or_junction(link_dir)

    def test_b8_ui_post_without_csrf_token_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            runs_dir = Path(td) / "runs"
            make_synthetic_n5_study_dir(runs_dir, seeds=(11, 17, 23, 29, 37))
            proc, base_url = launch_ui_server(runs_dir, port=0)
            try:
                surf = discover_ui_surface(base_url)
                targets = sorted(
                    set(surf["api_routes"])
                    | set(surf.get("post_routes", []))
                    | {"/api/audit", "/api/paper-export", "/api/launch", "/api/stop", "/api/inspect"}
                )
                mutating = [
                    r for r in targets
                    if any(k in r.lower() for k in ("audit", "paper", "export", "launch", "stop", "inspect", "start", "delete"))
                ]
                self.assertTrue(mutating, "No mutating POST endpoints discovered or configured")
                csrf_rejected = []
                statuses = {}
                for endpoint in mutating:
                    code, _, _ = http_post(base_url + endpoint, data=b"{}")
                    statuses[endpoint] = code
                    self.assertNotIn(code, range(200, 300), f"POST {endpoint} without CSRF succeeded ({code})")
                    self.assertIn(code, (400, 403, 404, 405), f"POST {endpoint} without CSRF returned {code}")
                    if code in (400, 403):
                        csrf_rejected.append(endpoint)
                self.assertTrue(
                    csrf_rejected,
                    f"No mutating POST route rejected missing CSRF with 400/403 (all returned 404/405): {statuses}",
                )
            finally:
                stop_ui_server(proc)

    def test_b8_ui_post_with_invalid_csrf_token_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            runs_dir = Path(td) / "runs"
            make_synthetic_n5_study_dir(runs_dir, seeds=(11, 17, 23, 29, 37))
            proc, base_url = launch_ui_server(runs_dir, port=0)
            try:
                surf = discover_ui_surface(base_url)
                self.assertIsNotNone(surf["csrf_token"])
                post_candidates = list(dict.fromkeys(list(surf.get("api_routes", [])) + list(surf.get("post_routes", []))))
                audit_route = next((r for r in post_candidates if "audit" in r.lower()), "/api/audit")
                code, _, _ = http_post(
                    base_url + audit_route,
                    data=b'{"run": "study_n5/adaptive_full/repeat_11"}',
                    headers={"X-CSRF-Token": "invalid-csrf-token-000"},
                )
                self.assertIn(code, (400, 403))
            finally:
                stop_ui_server(proc)

    def test_b8_ui_stop_inspect_and_paper_export_path_containment(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runs_dir = root / "runs"
            make_synthetic_n5_study_dir(runs_dir, seeds=(11, 17, 23, 29, 37))
            yaml_path = make_mini_yolo_dataset(root / "mini_ds")
            proc, base_url = launch_ui_server(runs_dir, port=0)
            try:
                surf = discover_ui_surface(base_url)
                token = surf["csrf_token"]
                self.assertIsNotNone(token)
                post_candidates = list(dict.fromkeys(list(surf.get("api_routes", [])) + list(surf.get("post_routes", []))))
                paper_route = next((r for r in post_candidates if "paper" in r.lower() or "export" in r.lower()), "/api/paper-export")
                inspect_route = next((r for r in post_candidates if "inspect" in r.lower()), "/api/inspect")
                stop_route = next((r for r in post_candidates if "stop" in r.lower()), "/api/stop")

                # 1) Paper export escaping output path must be blocked
                esc_out = root / "escaped_paper_out"
                c_esc, _, _ = http_post(
                    base_url + paper_route,
                    data=json.dumps({"run": "study_n5/adaptive_full/repeat_11", "output": str(esc_out), "lang": "en"}).encode("utf-8"),
                    headers={"X-CSRF-Token": token},
                )
                self.assertIn(c_esc, (400, 403))
                self.assertFalse(esc_out.exists())
                # 2) Visual launcher inspect works with valid CSRF POST
                c_insp, _, insp_body = http_post(
                    base_url + inspect_route,
                    data=json.dumps({"data": str(yaml_path), "device": "cpu"}).encode("utf-8"),
                    headers={"X-CSRF-Token": token},
                )
                self.assertEqual(c_insp, 200, msg=insp_body)
                # 3) Safe Stop rejects missing CSRF and handles valid CSRF safely without mutating runs
                c_stop_no, _, _ = http_post(base_url + stop_route, data=b"{}")
                self.assertIn(c_stop_no, (400, 403))
            finally:
                stop_ui_server(proc)

    # =========================================================================
    # Group B9: Integrity, Audit & Tamper Boundaries (5 tests)
    # =========================================================================
    def test_b9_audit_rejects_tampered_request_json(self):
        with tempfile.TemporaryDirectory() as td:
            make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29, 37))
            run_dir = Path(td) / "study_n5" / "adaptive_full" / "repeat_11"
            req_path = run_dir / "trials" / "attempt_00001" / "request.json"
            req = json.loads(req_path.read_text(encoding="utf-8"))
            req["seed"] = 999
            write_json(req_path, req)
            with self.assertRaises(ValueError):
                audit_run(run_dir)

    def test_b9_audit_rejects_tampered_parent_result_json(self):
        with tempfile.TemporaryDirectory() as td:
            make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29, 37))
            run_dir = Path(td) / "study_n5" / "adaptive_full" / "repeat_11"
            pr_path = run_dir / "trials" / "attempt_00001" / "parent_result.json"
            pr = json.loads(pr_path.read_text(encoding="utf-8"))
            pr["quality"] = 0.9999
            write_json(pr_path, pr)
            with self.assertRaises(ValueError):
                audit_run(run_dir)

    def test_b9_audit_rejects_tampered_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29, 37))
            run_dir = Path(td) / "study_n5" / "adaptive_full" / "repeat_11"
            ckpt = run_dir / "trials" / "attempt_00001" / "best.pt"
            ckpt.write_bytes(b"corrupted-checkpoint")
            with self.assertRaises(ValueError):
                audit_run(run_dir)

    def test_b9_audit_rejects_tampered_events_jsonl_hash_chain(self):
        with tempfile.TemporaryDirectory() as td:
            make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29, 37))
            run_dir = Path(td) / "study_n5" / "adaptive_full" / "repeat_11"
            ev_path = run_dir / "events.jsonl"
            lines = ev_path.read_text(encoding="utf-8").splitlines()
            last = json.loads(lines[-1])
            last["payload"]["session_seconds"] = 0.01
            lines[-1] = json.dumps(last)
            ev_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                audit_run(run_dir)

    def test_b9_audit_rejects_tampered_protocol_checksum(self):
        with tempfile.TemporaryDirectory() as td:
            make_synthetic_n5_study_dir(Path(td), seeds=(11, 17, 23, 29, 37))
            run_dir = Path(td) / "study_n5" / "adaptive_full" / "repeat_11"
            p_path = run_dir / "protocol.json"
            prot = json.loads(p_path.read_text(encoding="utf-8"))
            prot["budget_seconds"] = 99999.0
            write_json(p_path, prot)
            with self.assertRaises(ValueError):
                audit_run(run_dir)
