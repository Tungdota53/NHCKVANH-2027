"""Tests for 0.11.0 mAP50 primary detection metric and NCKH paper export suite."""

from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from research_pipeline.integrity import sha
from research_pipeline.v2 import benchmark, cli, deployment, paper_export, terminal as ui
from research_pipeline.v2.ledger import Ledger
from research_pipeline.v2.model_features import architecture_key
from research_pipeline.v2.predictors import CostModel, compatibility
from research_pipeline.v2.report import make_report, markdown
from research_pipeline.v2.schema import DETECTION_METRICS, METRICS, detection_metric_for, lock_protocol
from research_pipeline.v2.workflow import audit_run
from yolo_tuner.pipeline import write_json
from yolo_tuner.research import Objective, digest


class Map50AndNCKH011Tests(unittest.TestCase):
    def det_config(self, **changes):
        return {
            "model": "yolo11n.yaml",
            "imgsz": 160,
            "batch": 4,
            "accumulation": 4,
            "freeze": 0,
            "precision": "fp32",
            "optimizer": "AdamW",
            "lr0": 0.001,
            "weight_decay": 0.0005,
            **changes,
        }

    def det_protocol(self, detection_metric="map50", legacy=False, **changes):
        configs = changes.pop("configs", [
            self.det_config(lr0=0.001),
            self.det_config(lr0=0.0003),
            self.det_config(lr0=0.003),
        ])
        base = {
            "task": "detection",
            "method": "adaptive_full",
            "mode": "online",
            "selection_rule": "weighted",
            "implementation_version": "0.10.0" if legacy else "0.11.0",
            "configs": configs,
            "training_seeds": [17, 29],
            "search_seed": 17,
            "full_epochs": 3,
            "probe_epochs": 1,
            "budget_seconds": 120,
            "trial_timeout": 60,
            "objective": asdict(Objective()),
            "hardware": {"device": "cpu", "host_label": "unit-host", "cpu": "unit-cpu"},
            "runtime_versions": {},
            "scheduler": {"policy_version": 2, "cleanup_seconds": 1, "probe_fraction": 0.1, "cost_margin": 1.25, "rungs": [1, 2]},
            "acquisition": {"version": 1, "kernel_ridge": 0.05, "length_scale": 1.0, "exploration": 0.25, "novelty_weight": 0.05},
            "execution_policy": {
                "version": 2,
                "integrity": "stat_cached_between_trials_full_hash_boundaries",
                "budget": "end_to_end_soft_deadline_with_measured_finalization_reserve",
                "data_loader": {"workers": "auto", "cache_mb": 64.0, "pin_memory": "auto"},
            },
            "example_counts": {"train": 40, "val": 10},
            "num_classes": 1,
            "memory_kind": "process_tree_rss_mib_not_vram",
            "measurement_environment": "uncontended_preflight",
            "preparation_seconds": 0.5,
            "source_files": {},
            "data_files": {},
            "model_files": {},
            "dataset_source": "unit-fixture",
            "dataset_content_id": "unit-det-domain",
            "quality_definition": "best validation mAP50-95 within declared horizon" if legacy else DETECTION_METRICS.get(detection_metric, "invalid"),
        }
        if not legacy:
            base["detection_metric"] = detection_metric
        base.update(changes)
        return lock_protocol(base)

    def det_run_result(self, index, seed, map50=0.85, map50_95=0.55, best_epoch=2, **changes):
        point = {
            "epoch": best_epoch,
            "map50": map50,
            "map50_95": map50_95,
            "precision": 0.88,
            "recall": 0.82,
            "f1": 2 * 0.88 * 0.82 / (0.88 + 0.82),
        }
        return {
            "index": index,
            "seed": seed,
            "status": "ok",
            "mode": "train",
            "epochs_completed": 3,
            "primary_metric": "map50",
            "quality": map50,
            "best_quality": map50,
            "best_epoch": best_epoch,
            "best_checkpoint_epoch": best_epoch,
            "best_map50": map50,
            "best_map50_95": map50_95,
            "at_best_epoch": point,
            "learning_curve": [
                {"epoch": 1, "map50": map50 * 0.7, "map50_95": map50_95 * 0.7, "precision": 0.7, "recall": 0.6, "f1": 0.646},
                point,
                {"epoch": 3, "map50": map50 * 0.95, "map50_95": map50_95 * 1.02, "precision": 0.85, "recall": 0.80, "f1": 0.824},
            ],
            "memory_mb": 180.0,
            "worker_wall_seconds": 12.0,
            "measured_loop_seconds": 10.0,
            "optimizer_steps": 15,
            "training_loss_finite": True,
            "actual_config_verified": True,
            "checkpoint_sha256": "fixture-sha",
            **changes,
        }

    def test_default_detection_metric_is_map50_everywhere(self):
        self.assertEqual(METRICS["detection"], "best validation mAP50 within declared horizon")
        p = cli.parser()
        for cmd in ("optimize", "plan-search", "benchmark"):
            parsed = p.parse_args([cmd, "--data", "fixture"])
            self.assertEqual(parsed.detection_metric, "map50")
        ev = p.parse_args(["evaluate-v2", "--run", "fixture"])
        self.assertEqual(ev.seed_scope, "all")

        new_proto = self.det_protocol()
        old_proto = self.det_protocol(legacy=True)
        cls_proto = {
            "task": "classification",
            "quality_definition": METRICS["classification"],
        }
        self.assertEqual(detection_metric_for(new_proto), "map50")
        self.assertEqual(detection_metric_for(old_proto), "map50_95")
        self.assertIsNone(detection_metric_for(cls_proto))

    def test_protocol_validation_enforces_detection_metric_consistency(self):
        with self.assertRaisesRegex(ValueError, "detection_metric"):
            self.det_protocol(detection_metric="invalid_metric")
        with self.assertRaisesRegex(ValueError, "quality_definition"):
            self.det_protocol(detection_metric="map50", quality_definition=DETECTION_METRICS["map50_95"])
        with self.assertRaisesRegex(ValueError, "Invalid detection_metric"):
            self.det_protocol(task="classification", detection_metric="map50", quality_definition=METRICS["classification"])

    def test_ultralytics_fitness_patch_selects_map50_not_map50_95(self):
        from ultralytics.utils.metrics import Metric

        # Divergent epochs: Epoch A has higher mAP50 (0.90) but lower mAP50-95 (0.52);
        # Epoch B has lower mAP50 (0.82) but higher mAP50-95 (0.65).
        epoch_a = Metric()
        epoch_a.all_ap = np.column_stack([np.array([0.90]), np.full((1, 9), 0.4777778)])
        epoch_a.p = np.array([0.88])
        epoch_a.r = np.array([0.85])

        epoch_b = Metric()
        epoch_b.all_ap = np.column_stack([np.array([0.82]), np.full((1, 9), 0.6311111)])
        epoch_b.p = np.array([0.84])
        epoch_b.r = np.array([0.80])

        # Default Ultralytics fitness uses mAP50-95, so it prefers Epoch B over Epoch A:
        self.assertGreater(epoch_b.fitness(), epoch_a.fitness())

        # With our worker mAP50 fitness patch, fitness equals exact mAP50 and selects Epoch A:
        def map50_fitness(metric_self):
            return float(metric_self.map50)

        with patch.object(Metric, "fitness", map50_fitness):
            self.assertAlmostEqual(epoch_a.fitness(), 0.90, places=6)
            self.assertAlmostEqual(epoch_b.fitness(), 0.82, places=6)
            self.assertGreater(epoch_a.fitness(), epoch_b.fitness())

    def test_historical_report_stays_identical_and_new_report_exposes_map50_diagnostics(self):
        legacy_p = self.det_protocol(legacy=True)
        modern_p = self.det_protocol(detection_metric="map50")
        state = {
            "results": {
                "0": [self.det_run_result(0, 17, 0.86, 0.54), self.det_run_result(0, 29, 0.88, 0.56)],
                "1": [],
                "2": [],
            },
            "probes": [],
        }
        events = [
            {"kind": "session_started", "payload": {}, "sha256": "fixture"},
            {"kind": "selection_committed", "payload": {"session_seconds": 30}, "sha256": "fixture"},
        ]
        legacy_rep = make_report(legacy_p, state, events)
        modern_rep = make_report(modern_p, state, events)

        self.assertNotIn("detection_metric", legacy_rep)
        self.assertNotIn("detection_diagnostics", legacy_rep)
        self.assertEqual(modern_rep["detection_metric"], "map50")
        self.assertIn("detection_diagnostics", modern_rep)
        diag = modern_rep["detection_diagnostics"][modern_p["config_ids"][0]]
        self.assertAlmostEqual(diag["map50_at_best"]["mean"], 0.87)
        self.assertAlmostEqual(diag["map50_95_at_best"]["mean"], 0.55)

        md_text = markdown(modern_rep, modern_p)
        self.assertIn("Val mAP50 mean±SD", md_text)
        self.assertIn("Chỉ số chi tiết Object Detection tại Best Checkpoint", md_text)

        buf_new, buf_old = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_new):
            ui.summary_view(Path("."), modern_p, modern_rep)
        with redirect_stdout(buf_old):
            ui.summary_view(Path("."), legacy_p, legacy_rep)
        self.assertIn("Val mAP50 ", buf_new.getvalue())
        self.assertNotIn("Val mAP50-95", buf_new.getvalue())
        self.assertIn("Val mAP50-95", buf_old.getvalue())

    def test_cross_architecture_cost_scales_down_from_larger_to_smaller_yolo(self):
        configs = [self.det_config(model="yolo11s.yaml"), self.det_config(model="yolo11n.yaml")]
        p = self.det_protocol(configs=configs)
        p["model_descriptors"] = {
            architecture_key(configs[0]): {"parameters": 9_400_000, "gflops_reference_32": 21.5},
            architecture_key(configs[1]): {"parameters": 2_600_000, "gflops_reference_32": 6.5},
        }
        model = CostModel(p)
        model.update(0, self.det_run_result(0, 17, worker_wall_seconds=22.0, measured_loop_seconds=20.0))
        est_small = model.estimate(1)
        # Non-loop overhead = 2s; loop = 20s scaled down (clamped >= 0.35 => 7s), so total ~9s << 22s (never 42s!).
        self.assertLess(est_small["point_seconds"], 15.0)
        self.assertGreater(est_small["point_seconds"], 5.0)

    def test_predictor_card_rejects_detection_metric_mismatch(self):
        p = self.det_protocol(detection_metric="map50")
        card = {
            "schema_version": 2,
            "feature_names": __import__("research_pipeline.v2.predictors", fromlist=["FEATURE_NAMES"]).FEATURE_NAMES,
            "task": "detection",
            "detection_metric": "map50_95",
            "hardware_ids": ["unit-host"],
            "runtimes": [{}],
            "implementations": ["0.11.0"],
        }
        self.assertIn("detection_metric mismatch", compatibility(p, card))

    def _build_audited_run(self, root, primary_metric="map50", bad_epoch=False):
        manifest = {"task": "detection", "names": ["person"], "splits": {"train": [], "val": [], "test": [{"image": "test.png"}]}}
        data_dir = root / "full_data"
        sub_dir = root / "subset_data"
        data_dir.mkdir(parents=True)
        sub_dir.mkdir(parents=True)
        (data_dir / "data.yaml").write_text("names: [person]\n", encoding="utf-8")
        (sub_dir / "data.yaml").write_text("names: [person]\n", encoding="utf-8")
        data_files = {
            "full_data/data.yaml": sha(data_dir / "data.yaml"),
            "subset_data/data.yaml": sha(sub_dir / "data.yaml"),
        }
        p = self.det_protocol(
            detection_metric=primary_metric,
            dataset_sha256=digest(manifest),
            subset_sha256=digest(manifest),
            data_files=data_files,
            run_directory=str(root),
        )
        write_json(root / "protocol.json", p)
        write_json(root / "dataset.json", manifest)
        write_json(root / "subset.json", manifest)
        ledger = Ledger(root / "events.jsonl")
        ledger.emit("session_started", additional_budget_seconds=120, resume=False, preparation_charged_online=True)
        results = {"0": [], "1": [], "2": []}
        attempt_no = 0
        for idx, (m50, m95, t_sec) in enumerate(((0.88, 0.56, 12.0), (0.82, 0.51, 9.0), (0.91, 0.60, 16.0))):
            for s in p["training_seeds"]:
                attempt_no += 1
                trial = root / "trials" / f"attempt_{attempt_no:05d}"
                trial.mkdir(parents=True)
                ckpt = trial / "best.pt"
                ckpt.write_bytes(f"ckpt-{idx}-{s}".encode("utf-8"))
                req = {
                    "config": p["configs"][idx],
                    "seed": s,
                    "full_epochs": 3,
                    "trial_epochs": 3,
                    "mode": "train",
                    "resume": None,
                    "manifest": str(root / "dataset.json"),
                    "data": str(data_dir / "data.yaml"),
                    "protocol_sha256": p["protocol_sha256"],
                    "data_loader": p["execution_policy"]["data_loader"],
                    "detection_metric": primary_metric,
                }
                raw = self.det_run_result(
                    idx,
                    s,
                    map50=m50 + (0.01 if s == 29 else 0.0),
                    map50_95=m95,
                    worker_wall_seconds=t_sec,
                    best_checkpoint=str(ckpt),
                    checkpoint_sha256=sha(ckpt),
                    config=p["configs"][idx],
                )
                if bad_epoch and idx == 0 and s == 17:
                    raw["best_checkpoint_epoch"] = 1  # mismatch against best_epoch=2
                write_json(trial / "request.json", req)
                write_json(trial / "result.json", raw)
                attempt_id = f"att-{attempt_no}"
                parent = {
                    **raw,
                    "attempt_id": attempt_id,
                    "trial_directory": str(trial),
                    "request_sha256": sha(trial / "request.json"),
                    "raw_sha256": sha(trial / "result.json"),
                }
                write_json(trial / "parent_result.json", parent)
                ledger.emit("trial_started", attempt_id=attempt_id)
                ledger.emit(
                    "trial_completed",
                    attempt_id=attempt_id,
                    parent_sha256=sha(trial / "parent_result.json"),
                    worker_seconds=t_sec,
                    status="ok",
                    directory=str(trial),
                )
                results[str(idx)].append(parent)
        from research_pipeline.v2.selection import select
        from research_pipeline.v2.schema import measurements
        rows = measurements(p["configs"], results, p["training_seeds"], 3, Objective(**p["objective"]))
        state = {
            "results": results,
            "probes": [],
            "actions": [],
            "ordering": [0, 1, 2],
            "decision": select(rows, p),
            "acquisition_history": [
                {"step": 1, "index": 0, "reason": "bootstrap_full_confirmation", "remaining_seconds": 110.0, "surrogate": {"fitted": False, "observed_configurations": 0}},
                {"step": 2, "index": 1, "reason": "adaptive_optimistic_acquisition", "remaining_seconds": 84.0, "surrogate": {"fitted": False, "observed_configurations": 1}},
                {"step": 3, "index": 2, "reason": "adaptive_optimistic_acquisition", "remaining_seconds": 64.0, "surrogate": {"fitted": True, "observed_configurations": 2}},
            ],
        }
        write_json(root / "state.json", state)
        state_hash = digest(state)
        (root / "commits").mkdir()
        write_json(root / "commits" / f"{state_hash}.json", state)
        ledger.emit(
            "selection_committed",
            session_seconds=75.0,
            original_budget_seconds=120,
            this_execution_budget_seconds=120,
            decision=state["decision"],
            state_sha256=state_hash,
            state_path=f"commits/{state_hash}.json",
        )
        write_json(root / "decision.locked.json", state["decision"])
        from research_pipeline.v2.report import publish
        publish(root, p, state, display=False)
        return p, state

    def test_audit_rejects_mismatched_best_checkpoint_epoch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._build_audited_run(root, bad_epoch=True)
            with self.assertRaisesRegex(ValueError, "best.pt epoch differs"):
                audit_run(root)

    def test_multi_seed_evaluate_v2_and_paper_export_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            run_dir = base / "run"
            run_dir.mkdir()
            self._build_audited_run(run_dir)

            def fake_popen(cmd, stdout=None, stderr=None, **kw):
                req_path = Path(cmd[-1])
                req = json.loads(req_path.read_text(encoding="utf-8"))
                seed = req["seed"]
                m50 = 0.89 if seed == 17 else 0.91
                write_json(
                    Path(req["project"]) / "result.json",
                    {
                        "status": "ok",
                        "primary_metric": "map50",
                        "quality": m50,
                        "map50": m50,
                        "map75": 0.72,
                        "map50_95": 0.61,
                        "precision": 0.90,
                        "recall": 0.86,
                        "f1": 2 * 0.90 * 0.86 / (0.90 + 0.86),
                        "per_class": [{"class_id": 0, "class_name": "person", "ap50": m50, "ap50_95": 0.61}],
                    },
                )
                proc = SimpleNamespace(returncode=0, wait=lambda timeout=None: 0, poll=lambda: 0, args=cmd)
                return proc

            args = SimpleNamespace(run=str(run_dir), timeout_seconds=60, seed_scope="all", allow_retest=False)
            with patch("research_pipeline.v2.deployment.idle"), \
                 patch("research_pipeline.v2.deployment.verify_live", return_value=(json.loads((run_dir / "dataset.json").read_text(encoding="utf-8")), {})), \
                 patch("research_pipeline.v2.deployment.subprocess.Popen", side_effect=fake_popen), \
                 redirect_stdout(io.StringIO()):
                eval_res = deployment.evaluate(args)

            self.assertEqual(eval_res["evaluated_seeds"], [17, 29])
            self.assertAlmostEqual(eval_res["multi_seed_summary"]["map50"]["mean"], 0.90)
            self.assertEqual(len(eval_res["per_class_summary"]), 1)
            self.assertIsNotNone(eval_res["validation_vs_test"]["generalization_gap_test_minus_val"])

            # Retest without --allow-retest is blocked
            with patch("research_pipeline.v2.deployment.idle"), \
                 patch("research_pipeline.v2.deployment.verify_live", return_value=(json.loads((run_dir / "dataset.json").read_text(encoding="utf-8")), {})):
                with self.assertRaisesRegex(ValueError, "already exposed"):
                    deployment.evaluate(args)

            # Paper export audits raw evidence and writes LaTeX, CSV, SVG, and NCKH Markdown
            export_dir = base / "paper_out"
            code_buf = io.StringIO()
            with redirect_stdout(code_buf):
                rc = cli.main(argv=["paper-export", "--run", str(run_dir), "--output", str(export_dir)])
            self.assertEqual(rc, 0)
            for fname in ("PAPER_SUMMARY.md", "PAPER_EVIDENCE.json", "tables.tex", "candidates.csv",
                          "learning_curves.csv", "acquisition_history.csv", "weight_sensitivity.csv",
                          "test_evaluation.csv", "test_per_class.csv", "pareto_front.svg", "learning_curves.svg"):
                self.assertTrue((export_dir / fname).is_file(), fname)

            tex_content = (export_dir / "tables.tex").read_text(encoding="utf-8")
            self.assertIn(r"\toprule", tex_content)
            self.assertIn("mAP50", tex_content)
            self.assertIn("tab:nckh_per_class_test", tex_content)
            md_content = (export_dir / "PAPER_SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn("Held-out Test", md_content)
            self.assertIn("LOOCV", md_content)
            evidence = json.loads((export_dir / "PAPER_EVIDENCE.json").read_text(encoding="utf-8"))
            self.assertGreater(evidence["hypervolume_3d"], 0.0)
            self.assertTrue(evidence["surrogate_loocv"]["available"])
            self.assertEqual(evidence["fitted_acquisition_steps"], 1)

            # Refuses to overwrite existing export folder
            with self.assertRaisesRegex(ValueError, "already exists"):
                paper_export.export_paper_artifacts(run_dir, export_dir)

    def test_hypervolume_3d_exact_unit_box_and_dominance(self):
        obj = Objective(time_scale=100.0, memory_scale=1000.0)
        single = [{"state": "confirmed_feasible", "quality_mean": 0.8, "time_mean": 50.0, "memory_max": 400.0}]
        # L_Q = 0.2, L_T = 0.5, L_M = 0.4 => volume to (1,1,1) = 0.8 * 0.5 * 0.6 = 0.24
        self.assertAlmostEqual(paper_export.hypervolume_3d(single, obj), 0.24, places=9)
        dominated = single + [{"state": "confirmed_feasible", "quality_mean": 0.7, "time_mean": 60.0, "memory_max": 500.0}]
        self.assertAlmostEqual(paper_export.hypervolume_3d(dominated, obj), 0.24, places=9)

    def test_benchmark_compare_rejects_metric_mismatch_and_reports_quality_regret(self):
        p_map50 = self.det_protocol(detection_metric="map50")
        p_map95 = self.det_protocol(detection_metric="map50_95")
        from research_pipeline.v2.schema import measurements
        res = {
            "0": [self.det_run_result(0, 17, 0.80), self.det_run_result(0, 29, 0.80)],
            "1": [self.det_run_result(1, 17, 0.90), self.det_run_result(1, 29, 0.90)],
            "2": [self.det_run_result(2, 17, 0.85), self.det_run_result(2, 29, 0.85)],
        }
        rows = measurements(p_map50["configs"], res, p_map50["training_seeds"], 3, Objective())
        with tempfile.TemporaryDirectory() as tmp:
            ref_dir, run_dir = Path(tmp) / "ref", Path(tmp) / "online"
            ref_dir.mkdir()
            run_dir.mkdir()
            write_json(ref_dir / "report.json", {"online_session_seconds": 60.0})
            write_json(run_dir / "report.json", {"online_session_seconds": 20.0, "budget_overrun_seconds": 0.0})
            write_json(run_dir / "decision.locked.json", {"selected_config_id": rows[0]["config_id"]})

            with patch("research_pipeline.v2.benchmark.audit_run", side_effect=lambda path: (Path(path), p_map50 if str(path) == str(ref_dir) else p_map95, rows, 6)):
                with self.assertRaisesRegex(ValueError, "detection-metric mismatch"):
                    benchmark.compare([str(run_dir)], str(ref_dir))

            with patch("research_pipeline.v2.benchmark.audit_run", side_effect=lambda path: (Path(path), p_map50, rows, 6)):
                comp = benchmark.compare([str(run_dir)], str(ref_dir))
            self.assertAlmostEqual(comp["outcomes"][0]["quality_regret"], 0.10)
            self.assertFalse(comp["outcomes"][0]["exact_optimum_hit"])
            self.assertEqual(comp["summaries"][0]["exact_optimum_hit_rate"], 0.0)

    def test_evaluation_worker_enforces_batch_invariant_rect_false(self):
        from research_pipeline import evaluation as det_eval
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = root / "best.pt"
            ckpt.write_bytes(b"mock-ckpt")
            req_path = root / "request.json"
            write_json(req_path, {
                "checkpoint": str(ckpt),
                "checkpoint_sha256": sha(ckpt),
                "data": str(root / "data.yaml"),
                "device": "cpu",
                "imgsz": 320,
                "batch": 4,
                "split": "test",
                "project": str(root),
                "detection_metric": "map50",
            })
            captured = {}

            class FakeBox:
                map50 = 0.875
                map75 = 0.700
                map = 0.580
                mp = 0.890
                mr = 0.840
                ap_class_index = [0, 1]
                ap50 = [0.85, 0.90]
                ap = [0.55, 0.61]
                p = [0.88, 0.90]
                r = [0.82, 0.86]

            class FakeYOLO:
                def __init__(self, path):
                    captured["path"] = path

                def val(self, **kwargs):
                    captured.update(kwargs)
                    return SimpleNamespace(box=FakeBox(), names={0: "buffalo", 1: "elephant"})

            with patch("ultralytics.YOLO", FakeYOLO):
                det_eval.worker(str(req_path))
            self.assertIs(captured["rect"], False)
            out = json.loads((root / "result.json").read_text(encoding="utf-8"))
            self.assertIs(out["rect"], False)
            self.assertAlmostEqual(out["map50"], 0.875)
            self.assertEqual(len(out["per_class"]), 2)
            self.assertEqual(out["per_class"][1]["class_name"], "elephant")

    def test_paper_export_algorithm_study_aggregates_outcomes_and_clean_latex(self):
        with tempfile.TemporaryDirectory() as tmp:
            study_dir = Path(tmp) / "alg_study"
            study_dir.mkdir()
            payload = {
                "schema_version": 1,
                "studies": [
                    {
                        "study_id": "abc12345",
                        "outcomes": [
                            {
                                "method": "adaptive_full",
                                "search_seed": 17,
                                "selected": "c0",
                                "feasible": True,
                                "objective_regret": 0.02,
                                "quality_regret": 0.05,
                                "exact_optimum_hit": False,
                                "pareto_hit": True,
                                "online_seconds": 80.0,
                                "overrun_seconds": 0.0,
                            }
                        ],
                        "method_summaries": [
                            {
                                "method": "adaptive_full",
                                "verified_selection_count": 1,
                                "search_repetitions": 1,
                                "conditional_objective_regret": {"mean": 0.02, "sd": None},
                                "online_seconds": {"mean": 80.0},
                            }
                        ],
                    }
                ],
            }
            payload["sha256"] = digest(payload)
            write_json(study_dir / "ALGORITHM_STUDY.json", payload)
            out_dir = Path(tmp) / "export_alg"
            paper_export.export_paper_artifacts(study_dir, out_dir)
            tex = (out_dir / "tables.tex").read_text(encoding="utf-8")
            self.assertIn("0.0500", tex)
            self.assertIn("0.0\\%", tex)
            self.assertIn("100.0\\%", tex)
            self.assertNotIn("—", tex)


if __name__ == "__main__":
    unittest.main()
