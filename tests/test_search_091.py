"""Budget/selection regression fixtures, not measured ML-performance evidence."""

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from yolo_tuner.research import Objective
from research_pipeline.v2.ledger import Ledger
from research_pipeline.v2.model_features import architecture_key
from research_pipeline.v2.predictors import CostModel
from research_pipeline.v2.scheduler import run_search, small_first_key
from research_pipeline.v2.selection import presets, select


class Search091Tests(unittest.TestCase):
    def config(self, **changes):
        return {"model": "cnn_classifier", "imgsz": 32, "batch": 4,
                "accumulation": 2, "width": 16, "precision": "fp32",
                "optimizer": "AdamW", "lr0": .001, "weight_decay": .01, **changes}

    def protocol(self, **changes):
        return {"task": "classification", "method": "successive_halving", "mode": "online",
                "configs": [self.config(lr0=lr) for lr in (.001, .002, .003)],
                "training_seeds": [17, 29], "full_epochs": 4, "probe_epochs": 1,
                "search_seed": 17, "budget_seconds": 200, "trial_timeout": 200,
                "objective": asdict(Objective()), "selection_rule": "weighted",
                "example_counts": {"train": 100, "val": 20}, "num_classes": 2,
                "scheduler": {"policy_version": 2, "cleanup_seconds": 1,
                              "probe_fraction": .5, "cost_margin": 1.25, "rungs": [1, 2]},
                "hardware": {"device": "cpu", "host_label": "test-only"},
                "runtime_versions": {}, "implementation_version": "0.9.1", "source_files": {}, **changes}

    def outcome(self, index, seed, epochs=4, **changes):
        return {"index": index, "seed": seed, "status": "ok", "mode": "train",
                "epochs_completed": epochs, "quality": .5 + index * .1, "memory_mb": 20,
                "worker_wall_seconds": 10, "measured_loop_seconds": 9,
                "estimated_full_seconds": 10, "optimizer_steps": 3,
                "training_loss_finite": True, "actual_config_verified": True,
                "checkpoint_sha256": "fixture-not-weights", **changes}

    def run_fixture(self, protocol, full_cost=10, probe_cost=3, overhead=0, initial=None):
        left = [protocol["budget_seconds"]]
        calls = []
        def evaluate(index, seed, epochs, full, seconds):
            desired = full_cost if full else probe_cost * epochs
            actual = min(seconds, desired + overhead)
            left[0] -= actual
            worker = max(0, actual - overhead)
            calls.append({"index": index, "seed": seed, "epochs": epochs, "full": full,
                          "cap": seconds, "spent": actual, "worker": worker})
            return self.outcome(index, seed, epochs, mode="train" if full else "search",
                                status="ok" if actual >= desired + overhead else "timeout",
                                worker_wall_seconds=worker, measured_loop_seconds=max(0, worker - 1),
                                estimated_full_seconds=full_cost)
        with tempfile.TemporaryDirectory() as temp:
            ledger = Ledger(Path(temp) / "events.jsonl")
            state = run_search(protocol, evaluate, lambda: left[0], ledger, initial)
            events = list(ledger.events)
        return state, calls, left[0], events

    def test_ood_card_falls_back_to_observed_full_and_records_candidate_reason(self):
        p = self.protocol(configs=[self.config(), self.config(width=32)], predictor={"fixture": True})
        with patch("research_pipeline.v2.predictors.compatibility", return_value=[]), \
             patch("research_pipeline.v2.predictors.point_prediction", return_value={"available": False, "reasons": ["candidate feature range unseen"]}):
            model = CostModel(p)
            model.update(0, self.outcome(0, 17))
            result = model.estimate(1)
        self.assertEqual(result["point_seconds"], 37)
        self.assertEqual(result["planning_seconds"], 46.25)
        self.assertIn("candidate feature range unseen", result["ood_reasons"])
        self.assertIn("cross-config", result["basis"])
        self.assertFalse(result["upper_bound_guaranteed"])

    def test_ood_without_observations_is_transparently_unavailable(self):
        p = self.protocol(predictor={"fixture": True})
        with patch("research_pipeline.v2.predictors.compatibility", return_value=[]), \
             patch("research_pipeline.v2.predictors.point_prediction", return_value={"available": False, "reasons": ["model unseen"]}):
            result = CostModel(p).estimate(0)
        self.assertIsNone(result["planning_seconds"])
        self.assertEqual(result["ood_reasons"], ["model unseen"])
        self.assertIn("no usable prior", result["basis"])

    def test_cross_width_does_not_multiply_observed_non_loop_overhead(self):
        p = self.protocol(configs=[self.config(), self.config(width=32)])
        model = CostModel(p)
        model.update(0, self.outcome(0, 17, worker_wall_seconds=4, measured_loop_seconds=1))
        result = model.estimate(1)
        self.assertEqual(result["point_seconds"], 7)  # 3 overhead + 1 loop * 2**2
        self.assertEqual(result["cost_components"]["observed_non_loop_overhead_seconds"], 3)
        self.assertEqual(result["width_scaling"]["distinct_observed_widths"], 1)
        self.assertIn("fallback", result["width_scaling"]["basis"])
        self.assertFalse(result["upper_bound_guaranteed"])
        self.assertFalse(result["cost_components"]["future_non_loop_overhead_guaranteed"])

    def test_width_exponent_fits_medians_and_ignores_lr_but_not_batch(self):
        p = self.protocol(configs=[self.config(), self.config(width=32, lr0=.003),
                                   self.config(width=64), self.config(width=64, batch=8)])
        model = CostModel(p)
        for seed, loop in ((17, 1), (29, 1), (43, 100)):
            model.update(0, self.outcome(0, seed, worker_wall_seconds=loop + 3, measured_loop_seconds=loop))
        model.update(1, self.outcome(1, 17, worker_wall_seconds=5, measured_loop_seconds=2))
        model.update(3, self.outcome(3, 17, worker_wall_seconds=103, measured_loop_seconds=100))
        result = model.estimate(2)
        self.assertAlmostEqual(result["point_seconds"], 7)
        self.assertAlmostEqual(result["width_scaling"]["exponent"], 1)
        self.assertEqual(result["width_scaling"]["distinct_observed_widths"], 2)
        self.assertEqual(result["cost_components"]["reference_index"], 1)
        self.assertIn("median-loop", result["width_scaling"]["basis"])

    def test_width_exponent_is_clamped_and_not_a_future_cost_guarantee(self):
        p = self.protocol(configs=[self.config(width=w) for w in (16, 32, 64)])
        for second_loop, expected in ((.5, 0), (16, 2)):
            model = CostModel(p)
            model.update(0, self.outcome(0, 17, worker_wall_seconds=4, measured_loop_seconds=1))
            model.update(1, self.outcome(1, 17, worker_wall_seconds=second_loop + 3, measured_loop_seconds=second_loop))
            result = model.estimate(2)
            self.assertEqual(result["width_scaling"]["exponent"], expected)
            self.assertFalse(result["width_scaling"]["future_cost_guaranteed"])

    def test_missing_loop_telemetry_is_an_explicit_whole_worker_fallback(self):
        p = self.protocol(configs=[self.config(), self.config(width=32)])
        model = CostModel(p)
        row = self.outcome(0, 17)
        row.pop("measured_loop_seconds")
        model.update(0, row)
        result = model.estimate(1)
        self.assertEqual(result["point_seconds"], 40)
        self.assertIn("loop telemetry unavailable", result["basis"])
        self.assertFalse(result["upper_bound_guaranteed"])

    def test_legacy_cross_cost_still_scales_whole_worker(self):
        p = self.protocol(configs=[self.config(), self.config(model="other", width=32)])
        p["scheduler"].pop("policy_version")
        model = CostModel(p)
        model.update(0, self.outcome(0, 17))
        result = model.estimate(1)
        self.assertEqual(result["point_seconds"], 80)
        self.assertNotIn("cost_components", result)

    def test_probe_charge_uses_elapsed_not_dynamic_reserve_changes(self):
        p = self.protocol(method="reserve", configs=[self.config()], budget_seconds=100)
        left, clock = [100], [0]
        def evaluate(index, seed, epochs, full, seconds):
            duration = 10 if full else 2
            clock[0] += duration
            left[0] -= duration if full else 30  # reserve increased by 28, not elapsed time
            return self.outcome(index, seed, epochs, mode="train" if full else "search",
                                worker_wall_seconds=duration, measured_loop_seconds=duration - 1)
        with tempfile.TemporaryDirectory() as temp:
            state = run_search(p, evaluate, lambda: left[0], Ledger(Path(temp) / "events.jsonl"), elapsed=lambda: clock[0])
        self.assertEqual(state["probe_budget_spent_seconds"], 2)

    def test_thin_dynamic_budget_bootstraps_full_incumbent_not_all_grid_probes(self):
        p = self.protocol(budget_seconds=18)
        p["scheduler"]["probe_fraction"] = .1
        state, calls, left, events = self.run_fixture(p, full_cost=4, probe_cost=6)
        self.assertTrue(calls[0]["full"])
        self.assertIsNotNone(state["decision"]["selected_index"])
        self.assertGreaterEqual(left, 1)
        self.assertLessEqual(sum(c["spent"] for c in calls if not c["full"]), 1.8)
        self.assertTrue(any(e["kind"] == "rung_deferred" for e in events))
        self.assertTrue(any(e["kind"] == "candidate_cost_unavailable" for e in events))

    def test_protected_sh_completes_equal_fidelity_rungs_when_budget_fits(self):
        p = self.protocol()
        state, calls, left, events = self.run_fixture(p)
        self.assertEqual(sum(not c["full"] and c["epochs"] == 1 for c in calls), 3)
        self.assertEqual(sum(not c["full"] and c["epochs"] == 2 for c in calls), 2)
        self.assertEqual(state["decision"]["selected_index"], 2)
        self.assertEqual({r["seed"] for r in state["results"]["2"]}, {17, 29})
        self.assertLessEqual(state["probe_budget_spent_seconds"], 100)
        self.assertGreaterEqual(left, 1)
        self.assertEqual(sum(e["kind"] == "rung_completed" for e in events), 2)

    def test_probe_budget_charges_controller_time_and_survives_resume(self):
        p = self.protocol()
        state, calls, _, _ = self.run_fixture(p, overhead=2)
        probes = [c for c in calls if not c["full"]]
        self.assertTrue(probes)
        self.assertAlmostEqual(state["probe_budget_spent_seconds"], sum(c["spent"] for c in probes))
        self.assertGreater(state["probe_budget_spent_seconds"], sum(c["worker"] for c in probes))
        state["probe_budget_spent_seconds"] = 99
        count = len(state["probes"])
        resumed, calls, _, _ = self.run_fixture(p, initial=state)
        self.assertEqual(len(resumed["probes"]), count)
        self.assertEqual(resumed["probe_budget_spent_seconds"], 99)
        self.assertFalse(any(not c["full"] for c in calls))

    def test_zero_probe_fraction_does_not_start_probes(self):
        p = self.protocol()
        p["scheduler"]["probe_fraction"] = 0
        state, calls, _, _ = self.run_fixture(p)
        self.assertTrue(calls)
        self.assertTrue(all(c["full"] for c in calls))
        self.assertIsNotNone(state["decision"]["selected_index"])

    def test_legacy_sh_without_policy_version_retains_old_behavior(self):
        p = self.protocol(budget_seconds=18)
        p["scheduler"].pop("policy_version")
        state, calls, _, _ = self.run_fixture(p, full_cost=4, probe_cost=6)
        self.assertFalse(calls[0]["full"])
        self.assertIsNone(state["decision"]["selected_index"])
        self.assertNotIn("probe_budget_spent_seconds", state)

    def test_small_first_uses_architecture_parameters_not_yolo_default_width(self):
        p = self.protocol(configs=[self.config(model="yolo11s.yaml"), self.config(model="yolo11n.yaml")])
        p["model_descriptors"] = {architecture_key(p["configs"][0]): {"parameters": 9_000_000},
                                  architecture_key(p["configs"][1]): {"parameters": 2_000_000}}
        self.assertEqual(sorted([0, 1], key=lambda i: small_first_key(p, i)), [1, 0])

    def selection_rows(self):
        return [{"index": i, "config_id": key, "config": {}, "state": "confirmed_feasible",
                 "complete": True, "quality_mean": .9, "time_mean": time,
                 "memory_max": memory, "utility": .9, "completed_seeds": [17, 29]}
                for i, (key, time, memory) in enumerate((("aaa", 1, 10), ("zzz", 2, 20)))]

    def test_presets_and_zero_weight_balanced_ties_exclude_dominated_points(self):
        p, rows = self.protocol(), self.selection_rows()
        p["objective"]["weights"] = [1, 0, 0]
        self.assertEqual(select(rows, p)["selected_config_id"], "aaa")
        self.assertTrue(all(r["config_id"] == "aaa" for r in presets(rows, p).values()))

    def test_legacy_preset_and_zero_weight_tie_payload_stays_unchanged(self):
        p, rows = self.protocol(), self.selection_rows()
        p["scheduler"].pop("policy_version")
        self.assertEqual(select(rows, p)["selected_config_id"], "zzz")
        self.assertEqual(presets(rows, p)["Accuracy"]["config_id"], "zzz")


if __name__ == "__main__":
    unittest.main()
