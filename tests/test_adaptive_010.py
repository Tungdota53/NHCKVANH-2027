"""Algorithm fixtures only, never evidence of ML quality/speed superiority."""

import copy
from dataclasses import asdict
import json
import math
import unittest

from yolo_tuner.research import Objective
from research_pipeline.v2.acquisition import acquisition_spec, configuration_distances, rank_candidates
from research_pipeline.v2.schema import measurements


class AdaptiveAcquisitionTests(unittest.TestCase):
    def config(self, **changes):
        return {"model": "cnn_classifier", "imgsz": 32, "batch": 4, "accumulation": 2,
                "width": 16, "precision": "fp32", "optimizer": "AdamW", "lr0": .001,
                "weight_decay": .01, **changes}

    def protocol(self, **changes):
        return {"configs": [self.config(width=w) for w in (16, 24, 32, 48, 64)],
                "objective": asdict(Objective()), "training_seeds": [17, 29, 43],
                "full_epochs": 4, "search_seed": 17,
                "acquisition": {"version": 1, "kernel_ridge": .05, "length_scale": 1,
                                "exploration": .25, "novelty_weight": .05}, **changes}

    def result(self, index, seed, quality, **changes):
        return {"index": index, "seed": seed, "status": "ok", "mode": "train",
                "epochs_completed": 4, "quality": quality, "memory_mb": 100,
                "worker_wall_seconds": 10, "optimizer_steps": 1,
                "training_loss_finite": True, "actual_config_verified": True,
                "checkpoint_sha256": "fixture-not-weights", **changes}

    def rows(self, protocol, values=None, partial=None):
        results = {str(i): [] for i in range(len(protocol["configs"]))}
        for index, quality in (values or {}).items():
            results[str(index)] = [self.result(index, seed, quality) for seed in protocol["training_seeds"]]
        for index, runs in (partial or {}).items():
            results[str(index)].extend(runs)
        return measurements(protocol["configs"], results, protocol["training_seeds"],
                            protocol["full_epochs"], Objective(**protocol["objective"]))

    def forecasts(self, protocol, **changes):
        return {str(i): {"point_seconds": 10, "planning_seconds": 12.5,
                         "predicted_memory_mb": 100, **changes} for i in range(len(protocol["configs"]))}

    def test_no_observations_uses_finite_deterministic_seeded_fallback(self):
        p = self.protocol()
        output = rank_candidates(p, self.rows(p), self.forecasts(p), order=[4, 2, 0, 1, 3])
        self.assertEqual(output["ranking"], [4, 2, 0, 1, 3])
        self.assertFalse(output["surrogate"]["fitted"])
        self.assertEqual(output["surrogate"]["observed_configurations"], 0)
        self.assertIn("uncalibrated", output["surrogate"]["calibration"])
        self.assertEqual(output, rank_candidates(p, self.rows(p), self.forecasts(p), order=[4, 2, 0, 1, 3]))
        json.dumps(output, allow_nan=False)

    def test_fit_uses_all_seed_full_data_rows_including_constraint_failures(self):
        p = self.protocol()
        p["objective"]["min_quality"] = .5
        output = rank_candidates(p, self.rows(p, {0: .2, 4: .8}), self.forecasts(p))
        self.assertTrue(output["surrogate"]["fitted"])
        self.assertEqual(output["surrogate"]["training_indices"], [0, 4])
        self.assertEqual(output["surrogate"]["training_quality_mean"], [.2, .8])
        self.assertNotIn(0, output["ranking"])
        self.assertNotIn(4, output["ranking"])
        self.assertIsNotNone(output["surrogate"]["incumbent_utility"])

    def test_incomplete_wrong_mode_wrong_horizon_never_quality_labels(self):
        p = self.protocol()
        partial = {1: [self.result(1, 17, .999)],
                   2: [self.result(2, seed, .999, mode="search") for seed in p["training_seeds"]],
                   3: [self.result(3, seed, .999, epochs_completed=2) for seed in p["training_seeds"]]}
        rows = self.rows(p, {0: .2, 4: .8}, partial)
        output = rank_candidates(p, rows, self.forecasts(p))
        self.assertEqual(output["surrogate"]["training_indices"], [0, 4])
        self.assertEqual(output["surrogate"]["training_quality_mean"], [.2, .8])

    def test_claimed_complete_without_complete_raw_workers_is_not_fit_evidence(self):
        p = self.protocol()
        rows = self.rows(p, {0: .2, 4: .8})
        rows[1].update(complete=True, state="confirmed_feasible", quality_mean=1,
                       completed_seeds=list(p["training_seeds"]), runs=[])
        output = rank_candidates(p, rows, self.forecasts(p), candidates=[2, 3])
        self.assertEqual(output["surrogate"]["training_indices"], [0, 4])

    def test_probe_learning_curves_and_quality_priors_are_invariant(self):
        p = self.protocol()
        rows = self.rows(p, {0: .3, 4: .7})
        forecasts = self.forecasts(p)
        before = rank_candidates(p, rows, forecasts)
        mutated = copy.deepcopy(rows)
        for row in mutated:
            row["learning_curve"] = [{"quality_score": .99999, "epoch": 999}]
            row["quality_mean"] = .99999
            for run in row["runs"]:
                run["learning_curve"] = [{"quality_score": .99999}]
            row["runs"].append(self.result(row["index"], 17, 1, mode="search"))
        for forecast in forecasts.values():
            forecast["predicted_quality"] = 1
            forecast["learning_curve"] = [{"quality_score": 1}]
        after = rank_candidates(p, mutated, forecasts)
        self.assertEqual(before, after)

    def test_quality_predictions_bonus_and_uncertainty_are_bounded(self):
        p = self.protocol()
        output = rank_candidates(p, self.rows(p, {0: 0, 4: 1}), self.forecasts(p))
        for score in output["scores"].values():
            self.assertLessEqual(score["quality_mean_prediction"], score["quality_optimistic"])
            self.assertTrue(0 <= score["quality_mean_prediction"] <= 1)
            self.assertTrue(0 <= score["quality_optimistic"] <= 1)
            self.assertTrue(0 <= score["uncertainty"] <= 1)
            self.assertFalse(score["quality_measured"])
            self.assertFalse(score["safety_guarantee"])
            self.assertTrue(math.isfinite(score["score"]))

    def test_no_cost_ablation_removes_only_cost_denominator(self):
        p = self.protocol()
        forecasts = self.forecasts(p)
        forecasts["0"]["point_seconds"] = 100
        forecasts["1"]["point_seconds"] = 1
        rows = self.rows(p)
        cost = rank_candidates(p, rows, forecasts, candidates=[0, 1], order=[0, 1])
        no_cost = rank_candidates(p, rows, forecasts, candidates=[0, 1], order=[0, 1], use_cost=False)
        self.assertEqual(cost["ranking"], [1, 0])
        self.assertEqual(no_cost["ranking"], [0, 1])
        self.assertEqual(cost["scores"]["1"]["quality_mean_prediction"], no_cost["scores"]["1"]["quality_mean_prediction"])

    def test_no_uncertainty_and_no_diversity_controls_are_explicit(self):
        p = self.protocol()
        rows = self.rows(p, {0: .3, 4: .7})
        forecasts = self.forecasts(p)
        normal = rank_candidates(p, rows, forecasts)
        no_uncertainty = rank_candidates(p, rows, forecasts, use_uncertainty=False)
        no_diversity = rank_candidates(p, rows, forecasts, use_diversity=False)
        for key in normal["scores"]:
            self.assertEqual(no_uncertainty["scores"][key]["quality_bonus"], 0)
            self.assertEqual(no_uncertainty["scores"][key]["quality_optimistic"], normal["scores"][key]["quality_mean_prediction"])
            self.assertEqual(no_diversity["scores"][key]["diversity_bonus"], 0)
            self.assertEqual(no_diversity["scores"][key]["uncertainty"], normal["scores"][key]["uncertainty"])

    def test_partial_worker_mean_cost_times_only_missing_seeds(self):
        p = self.protocol()
        rows = self.rows(p, partial={0: [self.result(0, 17, .999, worker_wall_seconds=6),
                                        self.result(0, 29, .999, worker_wall_seconds=10)]})
        output = rank_candidates(p, rows, self.forecasts(p), candidates=[{"index": 0, "missing_seeds": [43]}])
        score = output["scores"]["0"]
        self.assertEqual(score["worker_cost_seconds"], 8)
        self.assertEqual(score["remaining_all_seed_cost_seconds"], 8)
        self.assertEqual(output["surrogate"]["observed_configurations"], 0)
        self.assertIn("mean completed", score["cost_basis"])

    def test_unknown_bad_forecasts_are_finite_and_never_reject_candidates(self):
        p = self.protocol()
        bad = {"0": {"point_seconds": float("nan"), "predicted_memory_mb": float("inf")},
               "1": {"point_seconds": -20, "planning_seconds": float("inf"), "predicted_memory_mb": -1}}
        output = rank_candidates(p, self.rows(p), bad)
        self.assertEqual(set(output["ranking"]), set(range(5)))
        self.assertTrue(output["scores"]["0"]["unknown_cost"])
        self.assertTrue(output["scores"]["0"]["unknown_memory"])
        self.assertGreater(output["scores"]["0"]["remaining_all_seed_cost_seconds"], 0)
        json.dumps(output, allow_nan=False)

    def test_extreme_finite_forecasts_do_not_overflow_unknown_placeholders(self):
        p = self.protocol()
        forecasts = self.forecasts(p, point_seconds=1e308, predicted_memory_mb=1e308)
        forecasts["4"] = {}
        output = rank_candidates(p, self.rows(p), forecasts)
        self.assertTrue(math.isfinite(output["scores"]["4"]["worker_cost_seconds"]))
        self.assertTrue(math.isfinite(output["scores"]["4"]["memory_prediction_mb"]))
        json.dumps(output, allow_nan=False)

    def test_predicted_memory_violation_is_planning_penalty_not_hard_rejection(self):
        p = self.protocol()
        p["objective"]["max_memory"] = 200
        forecasts = self.forecasts(p)
        forecasts["0"]["predicted_memory_mb"] = 400
        output = rank_candidates(p, self.rows(p), forecasts, candidates=[0, 1])
        self.assertEqual(set(output["ranking"]), {0, 1})
        self.assertEqual(output["scores"]["0"]["planning_resource_penalty"], .5)
        self.assertGreater(output["scores"]["0"]["score"], 0)

    def test_mixed_features_distinguish_numeric_and_model_identity(self):
        p = self.protocol(configs=[self.config(model="yolo11n.yaml"), self.config(model="yolo11s.yaml"),
                                   self.config(model="yolo11n.yaml", imgsz=64)])
        distances, metadata = configuration_distances(p)
        self.assertEqual(distances[0, 0], 0)
        self.assertGreater(distances[0, 1], 0)
        self.assertGreater(distances[0, 2], 0)
        self.assertIn("model_content_identity", metadata["categorical_fields"])
        self.assertIn("imgsz", metadata["numeric_fields"])

    def test_identical_features_remain_numerically_stable(self):
        p = self.protocol(configs=[self.config() for _ in range(4)])
        output = rank_candidates(p, self.rows(p, {0: .1, 1: .9}), self.forecasts(p), order=[3, 2, 1, 0])
        self.assertTrue(output["surrogate"]["fitted"])
        self.assertEqual(output["ranking"], [3, 2])
        json.dumps(output, allow_nan=False)

    def test_numerically_singular_user_ridge_has_transparent_safe_fallback(self):
        p = self.protocol(configs=[self.config() for _ in range(4)])
        p["acquisition"]["kernel_ridge"] = 1e-300
        output = rank_candidates(p, self.rows(p, {0: .1, 1: .9}), self.forecasts(p))
        self.assertFalse(output["surrogate"]["fitted"])
        self.assertIn("kernel solve", output["surrogate"]["fit_warning"])
        json.dumps(output, allow_nan=False)

    def test_spec_validation_and_prelocked_parameter_values(self):
        p = self.protocol()
        self.assertEqual(acquisition_spec(p), p["acquisition"])
        for key, value in (("version", 2), ("kernel_ridge", 0), ("length_scale", -1),
                           ("exploration", float("nan")), ("novelty_weight", True)):
            changed = copy.deepcopy(p)
            changed["acquisition"][key] = value
            with self.assertRaises(ValueError):
                rank_candidates(changed, self.rows(changed), {})

    def test_candidate_missing_count_validation(self):
        p = self.protocol()
        for entry in ({"index": 0, "missing_seed_count": 0}, {"index": 0, "missing_seed_count": True}, 999):
            with self.assertRaises(ValueError):
                rank_candidates(p, self.rows(p), {}, candidates=[entry])


if __name__ == "__main__":
    unittest.main()
