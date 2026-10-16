"""Regression/method fixtures only: mocked outcomes are not scientific evidence."""

from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from yolo_tuner.research import Objective, digest
from yolo_tuner.pipeline import write_json
from research_pipeline.v2 import cli, benchmark
from research_pipeline.v2.schema import measurements, complete, lock_protocol, validate_protocol
from research_pipeline.v2.selection import select, rule_scores, presets, RULES
from research_pipeline.v2.predictors import feature_row, FEATURE_NAMES, grouped_residuals, fit_card, point_prediction
from research_pipeline.v2.model_features import model_identity
from research_pipeline.v2.space import validate_config, build_space
from research_pipeline.v2.scheduler import run_search
from research_pipeline.v2.ledger import Ledger


class UpgradeTests(unittest.TestCase):
    def config(self, **kw):
        return {"model": "cnn_classifier", "imgsz": 32, "batch": 4, "accumulation": 2, "width": 16,
                "precision": "fp32", "optimizer": "AdamW", "lr0": .001, "weight_decay": .01, **kw}

    def protocol(self, **kw):
        return {"task": "classification", "method": "random", "mode": "online", "configs": [self.config(), self.config(width=32), self.config(width=64)],
                "training_seeds": [17, 29], "full_epochs": 4, "probe_epochs": 1, "search_seed": 17, "budget_seconds": 300, "trial_timeout": 100,
                "objective": asdict(Objective()), "example_counts": {"train": 100, "val": 20}, "num_classes": 2,
                "scheduler": {"cleanup_seconds": 1, "probe_fraction": .1, "cost_margin": 1.25, "rungs": [1, 2]},
                "hardware": {"device": "cpu", "host_label": "test-only"}, "runtime_versions": {}, "implementation_version": "0.9.0",
                "source_files": {}, "memory_kind": "process_tree_rss_mib_not_vram", "dataset_content_id": "test-only", **kw}

    def result(self, index, seed, epochs=4, **kw):
        return {"index": index, "seed": seed, "status": "ok", "mode": "train", "epochs_completed": epochs,
                "quality": .5 + index * .1, "memory_mb": 20 + index * 10, "worker_wall_seconds": 3 + index * 2,
                "optimizer_steps": 3, "training_loss_finite": True, "actual_config_verified": True, "checkpoint_sha256": "fixture-not-weights", **kw}

    def rows(self):
        p = self.protocol()
        results = {str(i): [self.result(i, s) for s in p["training_seeds"]] for i in range(3)}
        return p, measurements(p["configs"], results, p["training_seeds"], 4, Objective())

    def test_failed_training_returns_nonzero(self):
        with patch("research_pipeline.v2.deployment.train", return_value={"execution_verified": False}):
            self.assertEqual(cli.main(argv=["train-v2", "--selection", "fixture"]), 3)

    def test_verified_training_returns_zero(self):
        with patch("research_pipeline.v2.deployment.train", return_value={"execution_verified": True}):
            self.assertEqual(cli.main(argv=["train-v2", "--selection", "fixture"]), 0)

    def test_zero_epochs_not_silently_replaced_by_default(self):
        from research_pipeline.v2.deployment import train
        args = cli.parser().parse_args(["train-v2", "--selection", "unused", "--epochs", "0"])
        with self.assertRaisesRegex(ValueError, "epochs"):
            train(args)

    def test_train_only_dataset_descriptor_never_reads_test_records(self):
        from research_pipeline.v2.data_features import describe_dataset
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "train.png"
            Image.new("RGB", (16, 16), "red").save(image)
            manifest = {"names": ["one", "two"], "splits": {"train": [{"image": str(image), "class_id": 0}],
                        "test": [{"image": "NONEXISTENT_TEST_DO_NOT_READ", "class_id": 1}]}}
            result = describe_dataset(manifest)
        self.assertEqual(result["split"], "train_only")
        self.assertEqual(result["sample_count"], 1)
        self.assertEqual(result["class_entropy"], 0)

    def test_proxy_spearman_ties_and_constant_values(self):
        from research_pipeline.v2.diagnostics import spearman
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1]), -1)
        self.assertAlmostEqual(spearman([1, 1, 3], [1, 1, 3]), 1)
        self.assertIsNone(spearman([1, 1], [2, 3]))

    def test_benchmark_child_contract_requests_json(self):
        args = cli.parser().parse_args(["benchmark", "--data", "fixture"])
        with tempfile.TemporaryDirectory() as temp:
            child = Mock(return_value=json.dumps({"hardware": {"ml_runtime_ready": False}}))
            with patch.object(benchmark, "idle"), patch.object(benchmark, "new_session", return_value=Path(temp)), patch.object(benchmark.subprocess, "check_output", child):
                with self.assertRaisesRegex(ValueError, "runtime"):
                    benchmark.benchmark(args)
        self.assertIn("--json", child.call_args.args[0])

    def test_failed_runtime_benchmark_does_not_spawn_training(self):
        args = cli.parser().parse_args(["benchmark", "--data", "fixture"])
        with tempfile.TemporaryDirectory() as temp, patch.object(benchmark, "idle"), \
             patch.object(benchmark, "new_session", return_value=Path(temp)), \
             patch.object(benchmark.subprocess, "check_output", return_value='{"hardware":{"ml_runtime_ready":false}}'), \
             patch.object(benchmark.subprocess, "run") as train:
            with self.assertRaises(ValueError):
                benchmark.benchmark(args)
        train.assert_not_called()

    def test_yolo_n_and_s_features_differ(self):
        p = self.protocol(task="detection")
        n, s = self.config(model="yolo11n.yaml"), self.config(model="yolo11s.yaml")
        self.assertNotEqual(feature_row(p, n), feature_row(p, s))
        self.assertEqual(len(feature_row(p, n)), len(FEATURE_NAMES))

    def test_lr_freeze_and_precision_have_distinct_features(self):
        p, c = self.protocol(), self.config()
        for new in (self.config(lr0=.003), self.config(freeze=5), self.config(precision="bf16")):
            self.assertNotEqual(feature_row(p, c), feature_row(p, new))

    def test_old_card_does_not_silently_use_new_vector(self):
        from research_pipeline.v2.predictors import compatibility
        p = self.protocol()
        card = {"schema_version": 1, "task": p["task"], "hardware_ids": ["test-only"], "runtimes": [{}], "implementations": ["0.9.0"]}
        self.assertTrue(any("feature schema" in r for r in compatibility(p, card)))

    def test_freeze_requires_learned_checkpoint_and_valid_integer(self):
        c = self.config(model="weights.pt", imgsz=160, freeze=5, optimizer="AdamW", weight_decay=.0005)
        validate_config("detection", c, {"device": "cpu"})
        for value in (-1, True, 2.5):
            with self.assertRaises(ValueError):
                validate_config("detection", {**c, "freeze": value}, {"device": "cpu"})
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            validate_config("detection", {**c, "model": "yolo11n.yaml"}, {"device": "cpu"})

    def test_research_grid_has_interactions_not_just_one_factor(self):
        with tempfile.TemporaryDirectory() as temp:
            configs = build_space("classification", {"device": "cpu"}, Path(temp), preset="research")
        self.assertEqual(len(configs), 8)
        self.assertTrue(any(c["imgsz"] == 64 and c["batch"] == 8 and c["lr0"] == .0003 for c in configs))

    def test_all_rules_select_only_full_feasible_rows(self):
        p, rows = self.rows()
        rows[2].update(state="unresolved", complete=False, quality_mean=.999, utility=None)
        for rule in RULES:
            selected = select(rows, {**p, "selection_rule": rule})
            self.assertNotEqual(selected["selected_index"], 2)
            self.assertEqual(selected["verification_status"], "fully_trained_best_observed")

    def test_no_feasible_rule_or_preset_invents_selection(self):
        p, rows = self.rows()
        for row in rows:
            row.update(state="unresolved", complete=False, utility=None)
        for rule in RULES:
            self.assertIsNone(select(rows, {**p, "selection_rule": rule})["selected_index"])
        self.assertTrue(all(v is None for v in presets(rows, p).values()))

    def test_degenerate_knee_has_explicit_fallback(self):
        p, rows = self.rows()
        _, fallback = rule_scores(rows[:2], {**p, "selection_rule": "knee"})
        self.assertIn("fallback", fallback)

    def test_topsis_does_not_need_population_minmax_or_test(self):
        p, rows = self.rows()
        score, _ = rule_scores(rows, {**p, "selection_rule": "topsis"})
        other, _ = rule_scores(rows[:2], {**p, "selection_rule": "topsis"})
        self.assertEqual(score[rows[0]["config_id"]], other[rows[0]["config_id"]])
        self.assertTrue(all(0 <= x <= 1 for x in score.values()))

    def test_presets_reference_actual_full_data_rows(self):
        p, rows = self.rows()
        result = presets(rows, {**p, "selection_rule": "weighted"})
        self.assertEqual(result["Accuracy"]["config_id"], rows[2]["config_id"])
        self.assertEqual(result["Time"]["config_id"], rows[0]["config_id"])
        self.assertEqual(result["Memory"]["config_id"], rows[0]["config_id"])

    def test_historical_decision_payload_stays_identical(self):
        from research_pipeline.v2.schema import decision
        p, rows = self.rows()
        self.assertEqual(select(rows, p), decision(rows))

    def test_grouped_calibration_counts_domains_not_rows(self):
        rows = [{"domain": ("same", "hardware"), "target": i / 100} for i in range(100)]
        calibration = grouped_residuals(rows, lambda r: 0)
        self.assertEqual(calibration["domain_count"], 1)
        self.assertIsNone(calibration["one_sided_quantile"])
        self.assertFalse(calibration["coverage_guarantee"])

    def test_grouped_finite_sample_quantile_requires_enough_blocks(self):
        rows = [{"domain": (str(i), "hardware"), "target": i} for i in range(20)]
        calibration = grouped_residuals(rows, lambda r: 0)
        self.assertEqual(calibration["one_sided_quantile"], 19)
        self.assertFalse(calibration["coverage_guarantee"])

    def test_multirung_sh_restarts_same_horizon_then_all_seed_finalizes(self):
        p = self.protocol(method="successive_halving", selection_rule="weighted")
        calls = []
        def evaluate(i, seed, epoch, full, seconds):
            calls.append((i, seed, epoch, full))
            return self.result(i, seed, epoch, mode="train" if full else "search", estimated_full_seconds=10)
        with tempfile.TemporaryDirectory() as temp:
            state = run_search(p, evaluate, lambda: 300, Ledger(Path(temp) / "events.jsonl"))
        self.assertEqual(len([c for c in calls if c[2] == 1 and not c[3]]), 3)
        self.assertEqual(len([c for c in calls if c[2] == 2 and not c[3]]), 2)
        self.assertEqual([c[1] for c in calls if c[3]], [17, 29])
        self.assertEqual(state["decision"]["selected_index"], 2)
        self.assertFalse(state["decision"]["space_complete"])

    def test_incomplete_rung_does_not_promote_mixed_fidelity(self):
        p = self.protocol(method="successive_halving")
        count = [0]
        def evaluate(i, seed, epoch, full, seconds):
            count[0] += 1
            if not full and count[0] == 1:
                return self.result(i, seed, epoch, mode="search", status="timeout")
            return self.result(i, seed, epoch, mode="train" if full else "search")
        with tempfile.TemporaryDirectory() as temp:
            ledger = Ledger(Path(temp) / "events.jsonl")
            state = run_search(p, evaluate, lambda: 300, ledger)
        self.assertTrue(any(e["kind"] == "rung_incomplete" for e in ledger.events))
        self.assertEqual(sum(len(v) for v in state["results"].values()), 6)

    def test_card_fits_quality_without_calibration_claim_and_rejects_unseen_model(self):
        p = self.protocol(training_seeds=[17, 29, 43], configs=[self.config(width=w) for w in (16, 24, 32, 64)])
        results = {str(i): [self.result(i, seed) for seed in p["training_seeds"]] for i in range(4)}
        rows = measurements(p["configs"], results, p["training_seeds"], 4, Objective())
        with tempfile.TemporaryDirectory() as temp:
            manifest, output = Path(temp) / "partitions.json", Path(temp) / "card.json"
            write_json(manifest, {"fit": ["fixture"], "calibration": [], "evaluation": []})
            with patch("research_pipeline.v2.workflow.audit_run", return_value=(Path(temp), p, rows, 12)):
                card = fit_card(manifest, output)
        self.assertEqual(card["schema_version"], 2)
        self.assertEqual(len(card["quality_coefficients"]), len(FEATURE_NAMES))
        prediction = point_prediction(p, p["configs"][0], card)
        self.assertTrue(prediction["available"])
        self.assertTrue(0 <= prediction["quality"] <= 1)
        self.assertFalse(card["coverage_guarantee"])
        self.assertFalse(point_prediction(p, self.config(model="custom_other_model"), card)["available"])


if __name__ == "__main__":
    unittest.main()
