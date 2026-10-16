"""Mocked scheduler safety fixtures, NOT empirical training/research results."""

from dataclasses import asdict
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from yolo_tuner.research import Objective
from research_pipeline.v2.ledger import Ledger
from research_pipeline.v2.scheduler import run_search
from research_pipeline.v2.schema import complete


METHODS = ("adaptive_full", "adaptive_no_cost", "adaptive_no_uncertainty", "adaptive_no_diversity")


class FixtureCost:
    """Known fixture costs isolate scheduling contracts from model fitting."""
    def __init__(self, protocol):
        self.protocol, self.fulls, self.probes = protocol, {}, {}

    def update(self, index, result):
        if result.get("status") == "ok" and result.get("mode") == "train":
            self.fulls.setdefault(index, []).append(result)

    def estimate(self, index):
        cost = self.protocol.get("fixture_costs", {}).get(index, 2.0)
        return {"point_seconds": cost, "planning_seconds": cost * 1.25 if cost is not None else None,
                "predicted_memory_mb": None, "predicted_quality": None, "basis": "mock fixture only",
                "calibration": "uncalibrated_scheduling", "ood_reasons": ["no learned prior"],
                "upper_bound_guaranteed": False}


class AdaptiveSafetyTests(unittest.TestCase):
    def protocol(self, method="adaptive_full", **updates):
        configs = [{"model": "cnn_classifier", "imgsz": 32, "width": 16,
                    "batch": 4, "accumulation": 2, "lr0": lr, "weight_decay": .01,
                    "precision": "fp32", "optimizer": "AdamW"} for lr in (.0003, .001, .003)]
        p = {"task": "classification", "method": method, "mode": "online", "selection_rule": "weighted",
             "configs": configs, "training_seeds": [17, 29, 43], "full_epochs": 6, "probe_epochs": 1,
             "search_seed": 19, "budget_seconds": 120, "trial_timeout": 60,
             "objective": asdict(Objective(weights=(1, 0, 0))),
             "example_counts": {"train": 100, "val": 20}, "num_classes": 2,
             "scheduler": {"cleanup_seconds": 1, "cost_margin": 1.25, "probe_fraction": .1,
                           "policy_version": 2, "rungs": [1, 2, 4]},
             "acquisition": {"version": 1, "kernel_ridge": .05, "length_scale": 1.,
                             "exploration": .25, "novelty_weight": .05},
             "hardware": {"device": "cpu", "host_label": "test-only"}, "runtime_versions": {},
             "implementation_version": "0.10.0", "source_files": {}, "memory_kind": "process_tree_rss_mib_not_vram",
             "dataset_content_id": "mock-not-real-training"}
        p.update(updates)
        return p

    def result(self, index, seed, **updates):
        result = {"index": index, "seed": seed, "status": "ok", "mode": "train", "epochs_completed": 6,
                  "quality": (.45, .55, .95)[index], "memory_mb": 20, "worker_wall_seconds": 2,
                  "optimizer_steps": 8, "training_loss_finite": True, "actual_config_verified": True,
                  "checkpoint_sha256": "mock-not-a-checkpoint", "measured_loop_seconds": 1,
                  "learning_curve": [{"epoch": 1, "quality": .001 if index == 2 else .999},
                                     {"epoch": 6, "quality": (.45, .55, .95)[index]}]}
        result.update(updates)
        return result

    def initial(self, probes=None):
        return {"results": {str(i): [] for i in range(3)}, "probes": list(probes or []),
                "actions": [], "ordering": [0, 1, 2]}

    def run_fixture(self, p, initial=None, failed=None, hostile_curves=True, cost_class=FixtureCost):
        now, calls = [0.0], []
        def evaluate(index, seed, epochs, full, seconds):
            self.assertTrue(full, "Adaptive full must never launch a probe")
            self.assertEqual(epochs, p["full_epochs"])
            self.assertGreater(seconds, 0)
            calls.append((index, seed, epochs, full))
            spent = min(2.0, seconds)
            now[0] += spent
            r = self.result(index, seed, worker_wall_seconds=spent)
            if not hostile_curves:
                r["learning_curve"] = [{"epoch": 1, "quality": r["quality"]}]
            if failed == (index, seed):
                r.update(status="oom", quality=.9999, epochs_completed=1, checkpoint_sha256=None)
            elif spent < 2:
                r.update(status="timeout", epochs_completed=1, checkpoint_sha256=None)
            return r
        with tempfile.TemporaryDirectory() as temp, patch("research_pipeline.v2.scheduler.CostModel", cost_class):
            ledger = Ledger(Path(temp) / "events.jsonl")
            state = run_search(p, evaluate, lambda: p["budget_seconds"] - now[0], ledger,
                               initial=initial, elapsed=lambda: now[0])
        return state, calls, now[0]

    def test_slow_starter_still_gets_full_horizon_and_all_seeds(self):
        for method in METHODS:
            with self.subTest(method=method):
                p = self.protocol(method)
                state, calls, _ = self.run_fixture(p)
                self.assertEqual(len(calls), 9)
                self.assertFalse(state["probes"])
                self.assertEqual(state["decision"]["selected_index"], 2)
                self.assertTrue(state["decision"]["space_complete"])
                for index in range(3):
                    self.assertEqual({r["seed"] for r in state["results"][str(index)] if complete(r, 6)}, set(p["training_seeds"]))

    def test_probe_quality_and_learning_curves_cannot_change_acquisition_order(self):
        p = self.protocol()
        ordinary, normal_calls, _ = self.run_fixture(p, initial=self.initial(), hostile_curves=False)
        probes = [self.result(0, 17, mode="search", epochs_completed=1, quality=.99999,
                              estimated_full_seconds=.00001, worker_wall_seconds=.00001),
                  self.result(2, 17, mode="search", epochs_completed=1, quality=.00001,
                              estimated_full_seconds=1000000)]
        hostile, hostile_calls, _ = self.run_fixture(p, initial=self.initial(probes))
        self.assertEqual(normal_calls, hostile_calls)
        self.assertEqual(ordinary["decision"], hostile["decision"])
        self.assertEqual(len(hostile["probes"]), 2, "Historical evidence must be retained, not silently erased")

    def test_resume_finishes_missing_seeds_before_new_candidate(self):
        p, state = self.protocol(), self.initial()
        state["results"]["2"] = [self.result(2, 17)]
        resumed, calls, _ = self.run_fixture(p, initial=state)
        self.assertEqual(calls[:2], [(2, 29, 6, True), (2, 43, 6, True)])
        self.assertNotIn((2, 17, 6, True), calls)
        self.assertEqual(resumed["decision"]["selected_index"], 2)

    def test_initial_probes_and_incomplete_fulls_are_not_replayed_to_cost_model(self):
        replayed = []
        class RecordingCost(FixtureCost):
            def update(self, index, result):
                replayed.append((result["mode"], result["epochs_completed"], result.get("fixture_tag")))
                super().update(index, result)
        state = self.initial([self.result(0, 17, mode="search", epochs_completed=1, fixture_tag="hostile probe")])
        state["results"]["1"] = [self.result(1, 17, epochs_completed=1, fixture_tag="partial full")]
        self.run_fixture(self.protocol(), initial=state, cost_class=RecordingCost)
        self.assertTrue(replayed)
        self.assertTrue(all(mode == "train" and epochs == 6 and tag is None for mode, epochs, tag in replayed))

    def test_failed_seed_with_high_quality_cannot_win(self):
        state, _, _ = self.run_fixture(self.protocol(), failed=(2, 29))
        self.assertNotEqual(state["decision"]["selected_index"], 2)
        self.assertFalse(state["decision"]["space_complete"])
        self.assertFalse(any(complete(r, 6) for r in state["results"]["2"] if r["seed"] == 29))

    def test_cost_affordability_does_not_claim_unmeasured_candidate_infeasible(self):
        p = self.protocol(budget_seconds=12, fixture_costs={0: 2, 1: 1000, 2: 1000})
        state, calls, spent = self.run_fixture(p)
        self.assertTrue(calls)
        self.assertTrue(all(index == 0 for index, _, _, _ in calls))
        self.assertLessEqual(spent, p["budget_seconds"])
        self.assertEqual(state["decision"]["selected_index"], 0)
        self.assertFalse(state["decision"]["space_complete"])
        self.assertEqual(state["results"]["1"], [])
        self.assertEqual(state["results"]["2"], [])

    def test_too_small_budget_returns_no_unverified_recommendation(self):
        p = self.protocol(budget_seconds=3)
        state, _, spent = self.run_fixture(p)
        self.assertIsNone(state["decision"]["selected_index"])
        self.assertLessEqual(spent, p["budget_seconds"])


if __name__ == "__main__":
    unittest.main()
