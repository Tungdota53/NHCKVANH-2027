"""Phase 1 unit & integration tests: physical VRAM/cost, OOM guard, heteroscedastic GP/LOOCV, acquisition v2, adaptive_ehvi, adaptive_staged_seeds & rule unlock."""

from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from research_pipeline.integrity import sha
from research_pipeline.v2 import benchmark, cli, wizard
from research_pipeline.v2.acquisition import (
    _exact_3d_ehvi,
    _gaussian_ei,
    _kernel_matrix,
    acquisition_spec,
    configuration_distances,
    full_observations,
    rank_candidates,
)
from research_pipeline.v2.ledger import Ledger
from research_pipeline.v2.model_features import architecture_key, describe_models, trainable_parameters_for
from research_pipeline.v2.predictors import CostModel, feature_row, observed_cross_cost, observed_cross_memory
from research_pipeline.v2.report import make_report, publish
from research_pipeline.v2.scheduler import ADAPTIVE_METHODS, METHODS, run_search
from research_pipeline.v2.schema import complete, lock_protocol, measurements, validate_protocol
from research_pipeline.v2.selection import RULES, presets, select
from research_pipeline.v2.workflow import audit_run
from yolo_tuner.pipeline import write_json
from yolo_tuner.research import Objective, digest


class Phase1SurrogateCostTests(unittest.TestCase):
    def det_config(self, **changes):
        return {
            "model": "yolo11n.pt",
            "imgsz": 320,
            "batch": 8,
            "accumulation": 2,
            "freeze": 0,
            "precision": "fp16",
            "optimizer": "AdamW",
            "lr0": 0.001,
            "weight_decay": 0.0005,
            **changes,
        }

    def det_protocol(self, configs=None, **changes):
        configs = configs or [
            self.det_config(batch=4, accumulation=4),
            self.det_config(batch=8, accumulation=2),
            self.det_config(batch=8, accumulation=2, freeze=5),
            self.det_config(model="yolo11s.pt", batch=8, accumulation=2),
            self.det_config(model="yolo26n.pt", batch=8, accumulation=2),
        ]
        descriptors = {
            architecture_key(self.det_config(model="yolo11n.pt")): {
                "parameters": 2_605_612,
                "trainable_parameters": 2_605_612,
                "layers": 24,
                "gflops_reference_32": 0.01653,
                "block_parameters": [10000] * 5 + [100000] * 19,
            },
            architecture_key(self.det_config(model="yolo11s.pt")): {
                "parameters": 9_429_340,
                "trainable_parameters": 9_429_340,
                "layers": 24,
                "gflops_reference_32": 0.05427,
                "block_parameters": [35000] * 5 + [380000] * 19,
            },
            architecture_key(self.det_config(model="yolo26n.pt")): {
                "parameters": 2_535_344,
                "trainable_parameters": 2_535_344,
                "layers": 24,
                "gflops_reference_32": 0.01529,
                "block_parameters": [9000] * 5 + [100000] * 19,
            },
        }
        base = {
            "task": "detection",
            "method": "adaptive_full",
            "mode": "online",
            "selection_rule": "weighted",
            "implementation_version": "1.0.0",
            "configs": configs,
            "model_descriptors": descriptors,
            "training_seeds": [17, 29],
            "search_seed": 17,
            "full_epochs": 4,
            "probe_epochs": 1,
            "budget_seconds": 180.0,
            "trial_timeout": 120.0,
            "objective": asdict(Objective(time_scale=60.0, memory_scale=512.0, max_memory=2500.0)),
            "hardware": {"device": "0", "host_label": "test-gpu", "vram_free_mb": 5000.0},
            "runtime_versions": {},
            "scheduler": {
                "policy_version": 3,
                "cleanup_seconds": 5.0,
                "probe_fraction": 0.1,
                "cost_margin": 1.25,
                "rungs": [1, 2, 4],
            },
            "acquisition": {
                "version": 2,
                "kernel_ridge": 0.05,
                "length_scale": 1.0,
                "exploration": 0.25,
                "novelty_weight": 0.05,
            },
            "execution_policy": {
                "version": 2,
                "integrity": "stat_cached_between_trials_full_hash_boundaries",
                "budget": "end_to_end_soft_deadline_with_measured_finalization_reserve",
                "data_loader": {"workers": "auto", "cache_mb": 64.0, "pin_memory": "auto"},
            },
            "example_counts": {"train": 100, "val": 30},
            "num_classes": 4,
            "memory_kind": "cuda_reserved_allocator_mib",
            "measurement_environment": "uncontended_preflight",
            "preparation_seconds": 1.0,
            "source_files": {},
            "data_files": {},
            "model_files": {},
            "dataset_source": "unit-fixture",
            "dataset_content_id": "unit-det-domain",
            "quality_definition": "best validation mAP50 within declared horizon",
            "detection_metric": "map50",
        }
        base.update(changes)
        return lock_protocol(base)

    def full_result(self, index, seed, quality=0.6, wall=36.0, loop=26.0, memory=400.0, **changes):
        return {
            "index": index,
            "seed": seed,
            "status": "ok",
            "mode": "train",
            "epochs_completed": 4,
            "quality": quality,
            "memory_mb": memory,
            "worker_wall_seconds": wall,
            "measured_loop_seconds": loop,
            "optimizer_steps": 20,
            "training_loss_finite": True,
            "actual_config_verified": True,
            "checkpoint_sha256": f"ckpt-{index}-{seed}",
            **changes,
        }

    # --- 1. model_features & observed_cross_memory ---

    def test_trainable_parameters_for_priorities_and_fallbacks(self):
        p = self.det_protocol()
        cfg = self.det_config(freeze=5)
        total_p, train_p, basis = trainable_parameters_for(
            p, cfg, [{"config": cfg, "observed_forward": {"trainable_parameters": 2_530_748, "frozen_parameters": 74_864, "freeze": 5}}]
        )
        self.assertEqual(basis, "observed_forward")
        self.assertEqual(train_p, 2_530_748)
        self.assertEqual(total_p, 2_605_612)

        total_b, train_b, basis_b = trainable_parameters_for(p, cfg)
        self.assertEqual(basis_b, "descriptor_block_parameters")
        self.assertEqual(total_b, 2_605_612)
        self.assertEqual(train_b, 2_605_612 - 50_000)

        p_no_blocks = self.det_protocol()
        for desc in p_no_blocks["model_descriptors"].values():
            desc.pop("block_parameters", None)
        total_f, train_f, basis_f = trainable_parameters_for(p_no_blocks, cfg)
        self.assertEqual(basis_f, "freeze_layer_ratio_fallback")
        self.assertLess(train_f, total_f)

        total_u, train_u, basis_u = trainable_parameters_for(p, self.det_config(freeze=0))
        self.assertEqual(basis_u, "descriptor_unfrozen")
        self.assertEqual(total_u, train_u)

    def test_observed_cross_memory_decomposition_invariance_and_scaling(self):
        p = self.det_protocol(configs=[
            self.det_config(batch=4, accumulation=1, imgsz=320, freeze=0, optimizer="AdamW", precision="fp16"),
            self.det_config(batch=4, accumulation=8, imgsz=320, freeze=0, optimizer="AdamW", precision="fp16"),
            self.det_config(batch=8, accumulation=2, imgsz=320, freeze=0, optimizer="AdamW", precision="fp16"),
            self.det_config(batch=4, accumulation=4, imgsz=640, freeze=0, optimizer="AdamW", precision="fp16"),
            self.det_config(batch=8, accumulation=2, imgsz=320, freeze=5, optimizer="AdamW", precision="fp16"),
            self.det_config(batch=8, accumulation=2, imgsz=320, freeze=0, optimizer="SGD", precision="fp16"),
            self.det_config(batch=8, accumulation=2, imgsz=320, freeze=0, optimizer="AdamW", precision="fp32"),
        ])
        self.assertIsNone(observed_cross_memory(p, {}, 1))
        self.assertIsNone(observed_cross_memory(p, {0: [{"memory_mb": float("nan")}]}, 1))

        fulls = {0: [self.full_result(0, 17, memory=320.0)]}
        m0 = observed_cross_memory(p, fulls, 0)
        m1 = observed_cross_memory(p, fulls, 1)  # accumulation 1 vs 8
        self.assertIsNotNone(m1)
        self.assertFalse(m1["upper_bound_guaranteed"])
        self.assertAlmostEqual(m0["activation_mb"], m1["activation_mb"], places=9)
        self.assertAlmostEqual(m0["point_mb"], m1["point_mb"], places=9)

        m2 = observed_cross_memory(p, fulls, 2)  # batch 8 vs 4 -> 2x activation
        self.assertAlmostEqual(m2["activation_mb"], 2.0 * m0["activation_mb"], places=6)
        self.assertAlmostEqual(m2["base_allocator_mb"], m0["base_allocator_mb"], places=9)
        self.assertAlmostEqual(m2["static_model_state_mb"], m0["static_model_state_mb"], places=9)

        m3 = observed_cross_memory(p, fulls, 3)  # imgsz 640 vs 320 -> 4x activation
        self.assertAlmostEqual(m3["activation_mb"], 4.0 * m0["activation_mb"], places=6)

        m4 = observed_cross_memory(p, fulls, 4)  # freeze=5 reduces static + activation
        self.assertLess(m4["static_model_state_mb"], m2["static_model_state_mb"])
        self.assertLess(m4["activation_mb"], m2["activation_mb"])

        m5 = observed_cross_memory(p, fulls, 5)  # SGD < AdamW static
        self.assertLess(m5["static_model_state_mb"], m2["static_model_state_mb"])

        m6 = observed_cross_memory(p, fulls, 6)  # fp32 > fp16 activation
        self.assertGreater(m6["activation_mb"], m2["activation_mb"])

    def test_observed_cross_memory_ols_two_point_and_non_detection_fallbacks(self):
        p = self.det_protocol(configs=[
            self.det_config(batch=4, accumulation=4),
            self.det_config(batch=8, accumulation=2),
            self.det_config(batch=16, accumulation=1),
        ])
        fulls = {
            0: [self.full_result(0, 17, memory=320.0)],
            1: [self.full_result(1, 17, memory=464.0)],
        }
        m2 = observed_cross_memory(p, fulls, 2)
        self.assertIn("2-param OLS", m2["basis"])
        self.assertAlmostEqual(m2["point_mb"], 752.0, delta=5.0)
        self.assertGreater(m2["base_allocator_mb"], 0.0)

        # Negative slope noisy observations clamp/fallback cleanly without NaN or negative values
        noisy_fulls = {
            0: [self.full_result(0, 17, memory=500.0)],
            1: [self.full_result(1, 17, memory=300.0)],
        }
        m_noisy = observed_cross_memory(p, noisy_fulls, 2)
        self.assertTrue(math.isfinite(m_noisy["point_mb"]) and m_noisy["point_mb"] > 0.0)
        self.assertGreaterEqual(m_noisy["base_allocator_mb"], 0.0)
        self.assertGreaterEqual(m_noisy["activation_mb"], 0.0)

        # Classification & captioning with missing descriptors
        for task, model in (("classification", "cnn_classifier"), ("captioning", "attn_gru_captioner")):
            cfg0 = {"model": model, "imgsz": 32, "batch": 4, "accumulation": 2, "width": 16, "precision": "fp32", "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.01}
            if task == "captioning":
                cfg0.update(hidden=128, feature_grid=4)
            cfg1 = {**cfg0, "batch": 8, "width": 32}
            p_other = lock_protocol({
                "task": task, "method": "adaptive_full", "mode": "online", "selection_rule": "weighted",
                "configs": [cfg0, cfg1], "training_seeds": [17], "search_seed": 17,
                "full_epochs": 4, "probe_epochs": 1, "budget_seconds": 100.0, "trial_timeout": 50.0,
                "objective": asdict(Objective()), "hardware": {"device": "cpu", "host_label": "cpu-host"},
                "runtime_versions": {}, "scheduler": {"policy_version": 3, "cleanup_seconds": 1.0, "probe_fraction": 0.1, "cost_margin": 1.25},
                "acquisition": {"version": 2, "kernel_ridge": 0.05, "length_scale": 1.0, "exploration": 0.25, "novelty_weight": 0.05},
                "example_counts": {"train": 50, "val": 10}, "memory_kind": "process_tree_rss_mib_not_vram",
                "measurement_environment": "uncontended_preflight", "preparation_seconds": 0.1,
                "source_files": {}, "data_files": {}, "model_files": {}, "dataset_source": "fixture", "dataset_content_id": "fixture",
                "quality_definition": "fixture",
            })
            mem_other = observed_cross_memory(p_other, {0: [self.full_result(0, 17, memory=120.0)]}, 1)
            self.assertIsNotNone(mem_other)
            self.assertGreater(mem_other["point_mb"], 120.0)
            self.assertTrue(math.isfinite(mem_other["point_mb"]))

    # --- 2. CostModel policy_version=3 & online error smoothing ---

    def test_cost_model_v3_separates_non_loop_overhead_and_tracks_online_errors(self):
        p = self.det_protocol()
        cm = CostModel(p)
        # First observation: wall=35s, non_loop=10s, loop=25s
        cm.update(4, self.full_result(4, 17, wall=35.0, loop=25.0, memory=500.0))
        est_same = cm.estimate(4)
        # No online errors yet -> effective_margin == 1.25, planning = 10 + 25 * 1.25 = 41.25 (NOT 35*1.25 = 43.75)
        self.assertAlmostEqual(est_same["non_loop_seconds"], 10.0)
        self.assertAlmostEqual(est_same["loop_seconds"], 25.0)
        self.assertAlmostEqual(est_same["effective_margin"], 1.25)
        self.assertAlmostEqual(est_same["planning_seconds"], 41.25)
        self.assertIsNone(est_same["online_error_ratio"])

        # Accurate subsequent predictions smoothly reduce effective_margin
        for idx in (0, 1, 2):
            pred = cm.estimate(idx)
            self.assertIsNotNone(pred["predicted_memory_mb"])
            self.assertFalse(pred["upper_bound_guaranteed"])
            cm.update(idx, self.full_result(idx, 17, wall=pred["non_loop_seconds"] + pred["loop_seconds"], loop=pred["loop_seconds"], memory=400.0))
        est_after_accurate = cm.estimate(3)
        self.assertIsNotNone(est_after_accurate["online_error_ratio"])
        self.assertLess(est_after_accurate["effective_margin"], 1.25)

        # Underpredictions raise effective_margin up toward 1.35
        cm_under = CostModel(p)
        cm_under.update(4, self.full_result(4, 17, wall=35.0, loop=25.0, memory=500.0))
        for idx in (0, 1, 2):
            pred = cm_under.estimate(idx)
            cm_under.update(idx, self.full_result(idx, 17, wall=pred["non_loop_seconds"] + pred["loop_seconds"] * 1.6, loop=pred["loop_seconds"] * 1.6, memory=400.0))
        self.assertGreater(cm_under.estimate(3)["effective_margin"], 1.25)

    def test_african_wildlife_011_replay_cost_vram_and_budget_utilization(self):
        bench_dir = Path("runs/african_wildlife_nckh_benchmark/20261118_173050_553132")
        if not bench_dir.is_dir():
            self.skipTest("Reference 0.11.0 benchmark artifact not present")
        r17_dir = next((bench_dir / "adaptive_full" / "repeat_17").iterdir())
        ref_dir = next((bench_dir / "reference").iterdir())
        p_011 = json.loads((r17_dir / "protocol.json").read_text(encoding="utf-8"))
        ref_state = json.loads((ref_dir / "state.json").read_text(encoding="utf-8"))
        r17_state = json.loads((r17_dir / "state.json").read_text(encoding="utf-8"))

        # Construct policy_version=3 protocol from exact 0.11.0 configs/descriptors
        p_v3 = {**p_011, "scheduler": {**p_011["scheduler"], "policy_version": 3}}
        cm_v2 = CostModel(p_011)
        cm_v3 = CostModel(p_v3)
        for r in r17_state["results"]["4"]:
            cm_v2.update(4, r)
            cm_v3.update(4, r)

        est_c02_v2 = cm_v2.estimate(2)
        est_c02_v3 = cm_v3.estimate(2)
        est_c03_v2 = cm_v2.estimate(3)
        est_c03_v3 = cm_v3.estimate(3)

        # 0.11.0 had huge c03/c02 planning inflation; v3 separates non-loop overhead and populates physical VRAM
        self.assertIsNotNone(est_c03_v3["predicted_memory_mb"])
        self.assertLess(abs(est_c03_v3["predicted_memory_mb"] - 966.0) / 966.0, 0.25)
        self.assertIsNotNone(est_c02_v3["predicted_memory_mb"])
        self.assertLess(abs(est_c02_v3["predicted_memory_mb"] - 376.0) / 376.0, 0.25)
        self.assertLess(est_c02_v3["planning_seconds"], est_c02_v2["planning_seconds"])
        self.assertLess(est_c03_v3["planning_seconds"], est_c03_v2["planning_seconds"])
        # 0.11.0 adaptive_full/repeat_17 planned 124.19s for c03 and 47.26s for c02
        self.assertLess(est_c03_v3["planning_seconds"], 124.19 * 0.65)
        self.assertLess(est_c02_v3["planning_seconds"], 47.26 * 0.90)

        # Simulate scheduler with exact measured reference worker outcomes & budget=150s
        # under adaptive_staged_seeds + policy_version=3 + acquisition.version=2
        p_sim = {
            **p_v3,
            "method": "adaptive_staged_seeds",
            "acquisition": {**p_v3["acquisition"], "version": 2},
            "budget_seconds": 150.0,
        }
        p_sim.pop("protocol_sha256", None)
        p_sim = lock_protocol(p_sim)
        now = [9.0]  # match 9.0s preflight/session start overhead of 0.11.0 run
        ref_by_idx_seed = {(i, r["seed"]): r for i_str, rs in ref_state["results"].items() for i in [int(i_str)] for r in rs}
        for r in r17_state["results"]["4"]:
            ref_by_idx_seed[(4, r["seed"])] = r

        def sim_eval(index, seed, epochs, full, seconds):
            row = ref_by_idx_seed[(index, seed)]
            wall = min(float(row["worker_wall_seconds"]), float(seconds))
            now[0] += wall
            return {**row, "worker_wall_seconds": wall, "status": "ok" if wall >= row["worker_wall_seconds"] else "timeout"}

        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp) / "events.jsonl", clock=lambda: now[0])
            state = run_search(p_sim, sim_eval, lambda: 150.0 - now[0], ledger, elapsed=lambda: now[0])
        self.assertLessEqual(now[0], 150.0)
        self.assertGreater(now[0], 110.0)  # Eliminates 81.25s/150s early exit waste
        self.assertIsNotNone(state["decision"]["selected_index"])

    # --- 3. Early OOM / Max-Memory Risk Guard ---

    def test_scheduler_oom_risk_guard_defers_hard_and_steers_soft(self):
        # Config 0: small (300 MB), Config 1: soft-risky (950 MB vs max_memory=1000), Config 2: hard-exceeds (1300 MB)
        p = self.det_protocol(
            configs=[
                self.det_config(batch=4, lr0=0.001),
                self.det_config(batch=8, lr0=0.001),
                self.det_config(batch=16, lr0=0.001),
            ],
            objective=asdict(Objective(time_scale=60.0, memory_scale=512.0, max_memory=1000.0)),
            budget_seconds=200.0,
        )
        now, calls = [0.0], []

        def evaluate(index, seed, epochs, full, seconds):
            calls.append((index, seed))
            now[0] += 5.0
            mem = {0: 300.0, 1: 950.0, 2: 1300.0}[index]
            qual = {0: 0.60, 1: 0.90, 2: 0.99}[index]
            return self.full_result(index, seed, quality=qual, wall=5.0, loop=4.0, memory=mem)

        class StubCost(CostModel):
            def estimate(self, index):
                base = super().estimate(index)
                if self.fulls:
                    base["predicted_memory_mb"] = {0: 300.0, 1: 950.0, 2: 1300.0}[index]
                    base["memory_basis"] = "stub physical memory"
                    if base["planning_seconds"] is None:
                        base["point_seconds"] = 5.0
                        base["planning_seconds"] = 6.0
                return base

        with tempfile.TemporaryDirectory() as tmp, patch("research_pipeline.v2.scheduler.CostModel", StubCost):
            ledger = Ledger(Path(tmp) / "events.jsonl", clock=lambda: now[0])
            state = run_search(p, evaluate, lambda: 200.0 - now[0], ledger, elapsed=lambda: now[0])
            events = list(ledger.events)

        # Config 2 (1300 MB > 1000 MB) must NEVER be launched
        self.assertNotIn(2, [idx for idx, _ in calls])
        self.assertTrue(any(e["kind"] == "candidate_memory_risk_deferred" and e["payload"]["index"] == 2 for e in events))
        self.assertEqual(len(state["memory_risk_deferred"]), 1)
        self.assertEqual(state["memory_risk_deferred"][0]["index"], 2)
        self.assertFalse(state["memory_risk_deferred"][0]["upper_bound_guaranteed"])
        # Config 2 remains unresolved (not claimed confirmed_constraint_failure)
        rows = measurements(p["configs"], state["results"], p["training_seeds"], p["full_epochs"], Objective(**p["objective"]))
        self.assertEqual(rows[2]["state"], "unresolved")
        report = make_report(p, state, events)
        self.assertIn("memory_risk_deferred", report)
        self.assertEqual(report["memory_risk_deferred"][0]["index"], 2)

        # Soft steering test: Config 1 (safe, 400 MB) must be acquired BEFORE Config 2 (soft near max_memory, 950 MB)
        p_soft = self.det_protocol(
            configs=[
                self.det_config(batch=4, lr0=0.001),
                self.det_config(batch=6, lr0=0.002),
                self.det_config(batch=8, lr0=0.003),
            ],
            objective=asdict(Objective(time_scale=60.0, memory_scale=512.0, max_memory=1000.0)),
            budget_seconds=200.0,
        )
        soft_calls = []
        now_s = [0.0]

        def eval_soft(index, seed, epochs, full, seconds):
            soft_calls.append(index)
            now_s[0] += 5.0
            mem = {0: 300.0, 1: 400.0, 2: 950.0}[index]
            return self.full_result(index, seed, quality=0.7, wall=5.0, loop=4.0, memory=mem)

        class SoftStubCost(CostModel):
            def estimate(self, index):
                base = super().estimate(index)
                if self.fulls:
                    base["predicted_memory_mb"] = {0: 300.0, 1: 400.0, 2: 950.0}[index]
                    base["memory_basis"] = "stub physical memory"
                    if base["planning_seconds"] is None:
                        base["point_seconds"] = 5.0
                        base["planning_seconds"] = 6.0
                return base

        with tempfile.TemporaryDirectory() as tmp, patch("research_pipeline.v2.scheduler.CostModel", SoftStubCost):
            ledger = Ledger(Path(tmp) / "events.jsonl", clock=lambda: now_s[0])
            run_search(p_soft, eval_soft, lambda: 200.0 - now_s[0], ledger, elapsed=lambda: now_s[0])
        first_1 = soft_calls.index(1)
        first_2 = soft_calls.index(2)
        self.assertLess(first_1, first_2)

    # --- 4. Heteroscedastic GP / Adaptive Kernel Ridge & Exact Analytic LOOCV ---

    def test_exact_analytic_loocv_matches_bruteforce_and_heteroscedastic_noise(self):
        configs = [self.det_config(lr0=lr) for lr in (0.0001, 0.0003, 0.001, 0.003, 0.01)]
        p = self.det_protocol(configs=configs, training_seeds=[17, 29, 43])
        distances, _ = configuration_distances(p)
        train_indices = [0, 2, 4]
        y = np.array([0.42, 0.78, 0.65], dtype=float)
        eta = np.array([0.0001, 0.02, 0.0004], dtype=float)

        for kernel_name in ("matern52", "rbf"):
            ls, ridge = 0.5, 0.05
            k_full = _kernel_matrix(distances, kernel_name, ls)
            k_tt = k_full[np.ix_(train_indices, train_indices)]
            prior = float(y.mean())
            d = ridge + eta
            gram = k_tt + np.diag(d)
            cf = cho_factor(gram, lower=True, check_finite=False)
            alpha = cho_solve(cf, y - prior, check_finite=False)
            a_inv = cho_solve(cf, np.eye(len(train_indices)), check_finite=False)
            y_loo_analytic = np.clip(y - alpha / np.diag(a_inv), 0.0, 1.0)

            y_loo_bf = []
            for hold in range(len(train_indices)):
                keep = [k for k in range(len(train_indices)) if k != hold]
                y_sub = y[keep]
                p_sub = float(y_sub.mean())
                # Note: analytic hat-matrix identity keeps fixed centered prior=mean(y)
                g_sub = k_tt[np.ix_(keep, keep)] + np.diag(d[keep])
                a_sub = np.linalg.solve(g_sub, y_sub - prior)
                pred_hold = float(np.clip(prior + k_tt[hold, keep] @ a_sub, 0.0, 1.0))
                y_loo_bf.append(pred_hold)
            self.assertLess(float(np.max(np.abs(y_loo_analytic - np.array(y_loo_bf)))), 1e-12)

        # Heteroscedastic check via rank_candidates: noisy config (high seed variance) gets higher noise variance
        results = {str(i): [] for i in range(5)}
        results["0"] = [self.full_result(0, s, quality=0.50) for s in (17, 29, 43)]
        results["2"] = [
            self.full_result(2, 17, quality=0.40),
            self.full_result(2, 29, quality=0.80),
            self.full_result(2, 43, quality=0.60),
        ]
        results["4"] = [self.full_result(4, s, quality=0.75) for s in (17, 29, 43)]
        rows = measurements(p["configs"], results, p["training_seeds"], p["full_epochs"], Objective(**p["objective"]))
        forecasts = {str(i): {"point_seconds": 30.0, "planning_seconds": 35.0, "predicted_memory_mb": 400.0} for i in range(5)}
        out = rank_candidates(p, rows, forecasts)
        diag = out["surrogate_diagnostics"]
        self.assertIn(diag["kernel"], {"matern52", "rbf"})
        self.assertTrue(diag["heteroscedastic"])
        self.assertGreater(diag["noise_variances"][1], diag["noise_variances"][0])
        self.assertGreater(diag["grid_size_evaluated"], 0)
        self.assertIsNotNone(diag["loocv_rmse"])
        self.assertIsNotNone(diag["loocv_mae"])
        self.assertEqual(set(diag["predictions"].keys()), {"0", "1", "2", "3", "4"})
        json.dumps(out, allow_nan=False)

        # Single-seed mode sets heteroscedastic=False and noise_variances=0
        p_single = self.det_protocol(configs=configs, training_seeds=[17])
        res_single = {k: [v[0]] if v else [] for k, v in results.items()}
        rows_single = measurements(p_single["configs"], res_single, p_single["training_seeds"], p_single["full_epochs"], Objective(**p_single["objective"]))
        out_single = rank_candidates(p_single, rows_single, forecasts)
        self.assertFalse(out_single["surrogate_diagnostics"]["heteroscedastic"])
        self.assertEqual(out_single["surrogate_diagnostics"]["noise_variances"], [0.0, 0.0, 0.0])

        # Singular ridge fallback in v2
        p_sing = {
            "task": "detection",
            "method": "adaptive_full",
            "mode": "online",
            "selection_rule": "weighted",
            "configs": [self.det_config() for _ in range(4)],
            "model_descriptors": p["model_descriptors"],
            "training_seeds": [17],
            "search_seed": 17,
            "full_epochs": 4,
            "objective": asdict(Objective()),
            "acquisition": {"version": 2, "kernel_ridge": 1e-300, "ridge_grid": [1e-300], "length_scale": 1.0, "exploration": 0.25, "novelty_weight": 0.05},
        }
        res_dup = {
            "0": [self.full_result(0, 17, quality=0.4)],
            "1": [self.full_result(1, 17, quality=0.8)],
            "2": [],
            "3": [],
        }
        rows_dup = measurements(p_sing["configs"], res_dup, [17], 4, Objective(**p_sing["objective"]))
        out_sing = rank_candidates(p_sing, rows_dup, forecasts)
        self.assertFalse(out_sing["surrogate"]["fitted"])
        self.assertIn("kernel solve", out_sing["surrogate"]["fit_warning"])
        json.dumps(out_sing, allow_nan=False)

    # --- 5. Exact 3D EHVI & Rule-Consistent Acquisition v2 + Rule Unlock ---

    def test_exact_3d_ehvi_matches_monte_carlo(self):
        front = [(0.7, 0.6, 0.7), (0.85, 0.4, 0.5)]
        mu_q, sigma_q = 0.80, 0.10
        t_s, m_s = 20.0, 200.0
        t_scale, m_scale = 100.0, 1000.0
        exact = _exact_3d_ehvi(mu_q, sigma_q, t_s, m_s, front, t_scale, m_scale)

        rng = np.random.default_rng(42)
        samples = np.clip(rng.normal(mu_q, sigma_q, size=20000), 0.0, 1.0)
        b_t = 1.0 - t_s / t_scale
        b_m = 1.0 - m_s / m_scale

        def hv3(pts):
            # Brute-force union volume in [0,1]^3 by sorting unique q coordinates
            qs = sorted({0.0, 1.0, *(p[0] for p in pts)})
            vol = 0.0
            for i in range(len(qs) - 1):
                q_lo, q_hi = qs[i], qs[i + 1]
                active = [(p[1], p[2]) for p in pts if p[0] >= q_hi - 1e-12]
                if not active:
                    continue
                active.sort(key=lambda x: x[0], reverse=True)
                area, max_m = 0.0, 0.0
                for at, am in active:
                    if am > max_m:
                        area += (am - max_m) * at
                        max_m = am
                vol += (q_hi - q_lo) * area
            return vol

        base_hv = hv3(front)
        mc_gains = [hv3(front + [(float(q), b_t, b_m)]) - base_hv for q in samples]
        mc_mean = float(np.mean(mc_gains))
        self.assertAlmostEqual(exact, mc_mean, delta=0.005)

    def test_v2_rule_consistent_acquisition_and_rule_unlock(self):
        configs = [self.det_config(lr0=lr, batch=b) for lr, b in ((0.0003, 4), (0.001, 4), (0.003, 8), (0.0005, 8), (0.002, 4))]
        forecasts = {str(i): {"point_seconds": 20.0 + i * 5.0, "planning_seconds": 25.0 + i * 5.0, "predicted_memory_mb": 300.0 + i * 50.0} for i in range(5)}
        for rule in RULES:
            for method in ("adaptive_full", "adaptive_ehvi", "adaptive_staged_seeds"):
                p = self.det_protocol(configs=configs, method=method, selection_rule=rule)
                validate_protocol(p)  # Must NOT reject non-weighted rules!
                results = {str(i): [] for i in range(5)}
                results["0"] = [self.full_result(0, s, quality=0.55, wall=20.0, memory=300.0) for s in (17, 29)]
                results["1"] = [self.full_result(1, s, quality=0.72, wall=25.0, memory=350.0) for s in (17, 29)]
                rows = measurements(p["configs"], results, p["training_seeds"], p["full_epochs"], Objective(**p["objective"]))
                out = rank_candidates(p, rows, forecasts)
                self.assertEqual(len(out["ranking"]), 3)
                self.assertIn("surrogate_diagnostics", out)
                for s_entry in out["scores"].values():
                    self.assertIn("ei", s_entry)
                    self.assertIn("ehvi", s_entry)
                    self.assertIn("ucb_improvement", s_entry)
                    self.assertEqual(s_entry["selection_rule"], rule)
                json.dumps(out, allow_nan=False)

        # CLI must accept adaptive_full / adaptive_ehvi / adaptive_staged_seeds with --selection-rule topsis|knee|tchebycheff
        with patch("research_pipeline.v2.workflow.prepare", return_value=(Path("fix"), {}, None)) as mock_prep, \
             patch("research_pipeline.v2.workflow.execute"):
            for rule in ("tchebycheff", "topsis", "knee"):
                rc = cli.main(argv=["optimize", "--data", "dummy", "--method", "adaptive_ehvi", "--selection-rule", rule, "--acquisition-version", "2"])
                self.assertEqual(rc, 0)
        self.assertEqual(mock_prep.call_args.args[0].acquisition_version, 2)

        # Benchmark forwards --acquisition-version
        args_b = cli.parser().parse_args(["benchmark", "--data", "fixture", "--methods", "adaptive_full", "adaptive_ehvi", "--selection-rule", "topsis", "--acquisition-version", "2"])
        self.assertEqual(args_b.acquisition_version, 2)
        self.assertEqual(args_b.selection_rule, "topsis")

        # Wizard advanced flow allows adaptive_full / adaptive_ehvi / adaptive_staged_seeds on non-weighted rules
        dispatch = unittest.mock.Mock()
        insp = ("real-path", "cpu", {"task": "classification", "validation_ready": True}, {"ml_runtime_ready": True})
        inputs = ["30", "8", "2", "1", "y", "topsis", "adaptive_ehvi", "0.6 0.25 0.15", "", "", "600", "1024", "runs/v2", "2", "y"]
        with patch.object(wizard, "inspect_dataset", return_value=insp), patch("builtins.input", side_effect=inputs), redirect_stdout(io.StringIO()):
            wizard.optimize_wizard(dispatch)
        parsed = cli.parser().parse_args(dispatch.call_args.args[0])
        self.assertEqual(parsed.method, "adaptive_ehvi")
        self.assertEqual(parsed.selection_rule, "topsis")

    # --- 6. adaptive_ehvi & adaptive_staged_seeds End-to-End + Audit ---

    def _run_end_to_end_method(self, method, seed_specs, budget=100.0):
        configs = [
            self.det_config(lr0=0.0003, batch=4),
            self.det_config(lr0=0.001, batch=4),
            self.det_config(lr0=0.002, batch=8),
            self.det_config(model="yolo11s.pt", lr0=0.005, batch=8),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"task": "detection", "names": ["animal"], "splits": {"train": [], "val": [], "test": [{"image": "t.png"}]}}
            (root / "full_data").mkdir()
            (root / "subset_data").mkdir()
            (root / "full_data" / "data.yaml").write_text("names: [animal]\n", encoding="utf-8")
            (root / "subset_data" / "data.yaml").write_text("names: [animal]\n", encoding="utf-8")
            data_files = {"full_data/data.yaml": sha(root / "full_data" / "data.yaml"), "subset_data/data.yaml": sha(root / "subset_data" / "data.yaml")}
            p = self.det_protocol(
                configs=configs, method=method, selection_rule="topsis",
                budget_seconds=budget, dataset_sha256=digest(manifest), subset_sha256=digest(manifest),
                data_files=data_files, run_directory=str(root),
            )
            write_json(root / "protocol.json", p)
            write_json(root / "dataset.json", manifest)
            write_json(root / "subset.json", manifest)
            write_json(root / "status.json", {"phase": "search"})
            (root / "source_snapshot").mkdir()
            now = [0.0]
            ledger = Ledger(root / "events.jsonl", clock=lambda: now[0])
            ledger.emit("session_started", additional_budget_seconds=budget, resume=False, preparation_charged_online=True)
            attempt_no = [0]

            def evaluate(index, seed, epochs, full, seconds):
                attempt_no[0] += 1
                spec = seed_specs[(index, seed)]
                wall = min(float(spec["wall"]), float(seconds))
                now[0] += wall
                trial = root / "trials" / f"attempt_{attempt_no[0]:05d}"
                trial.mkdir(parents=True)
                ckpt = trial / "best.pt"
                ckpt.write_bytes(f"ckpt-{index}-{seed}".encode("utf-8"))
                req = {
                    "config": p["configs"][index], "seed": seed, "full_epochs": p["full_epochs"],
                    "trial_epochs": epochs, "mode": "train" if full else "search", "resume": None,
                    "manifest": str(root / ("dataset.json" if full else "subset.json")),
                    "data": str(root / ("full_data" if full else "subset_data") / "data.yaml"),
                    "protocol_sha256": p["protocol_sha256"],
                    "data_loader": p["execution_policy"]["data_loader"],
                    "detection_metric": "map50",
                }
                raw = self.full_result(
                    index, seed, quality=spec["quality"], wall=wall, loop=spec["loop"],
                    memory=spec["memory"], best_epoch=2, best_checkpoint_epoch=2, primary_metric="map50",
                    best_checkpoint=str(ckpt), checkpoint_sha256=sha(ckpt), config=p["configs"][index],
                    status="ok" if wall >= spec["wall"] else "timeout",
                )
                write_json(trial / "request.json", req)
                write_json(trial / "result.json", raw)
                att_id = f"att-{attempt_no[0]}"
                parent = {**raw, "attempt_id": att_id, "trial_directory": str(trial),
                          "request_sha256": sha(trial / "request.json"), "raw_sha256": sha(trial / "result.json")}
                write_json(trial / "parent_result.json", parent)
                ledger.emit("trial_started", attempt_id=att_id)
                ledger.emit("trial_completed", attempt_id=att_id, parent_sha256=sha(trial / "parent_result.json"),
                            worker_seconds=wall, status=parent["status"], directory=str(trial))
                return parent

            state = run_search(p, evaluate, lambda: budget - now[0], ledger, elapsed=lambda: now[0])
            write_json(root / "state.json", state)
            state_hash = digest(state)
            (root / "commits").mkdir()
            write_json(root / "commits" / f"{state_hash}.json", state)
            ledger.emit("selection_committed", session_seconds=now[0], original_budget_seconds=budget,
                        this_execution_budget_seconds=budget, decision=state["decision"],
                        state_sha256=state_hash, state_path=f"commits/{state_hash}.json")
            write_json(root / "decision.locked.json", state["decision"])
            publish(root, p, state, display=False)
            _, _, rows, count = audit_run(root)
            events = list(ledger.events)
            rep = make_report(p, state, events)
            return p, state, rows, events, rep, count

    def test_adaptive_ehvi_and_staged_seeds_end_to_end_and_partial_isolation(self):
        # Config 0: moderate (0.65, 10s, 300MB)
        # Config 1: clearly weak (0.20, 10s, 300MB) -> staged_seeds defers seed 2 while unexplored exist
        # Config 2: high quality (0.92, 10s, 350MB) -> staged_seeds promotes immediately
        # Config 3: memory violation on seed 1 (2800MB > 2500MB) -> staged_seeds stops after seed 1
        seed_specs = {}
        for s in (17, 29):
            seed_specs[(0, s)] = {"quality": 0.65, "wall": 10.0, "loop": 8.0, "memory": 300.0}
            seed_specs[(1, s)] = {"quality": 0.20, "wall": 10.0, "loop": 8.0, "memory": 300.0}
            seed_specs[(2, s)] = {"quality": 0.92, "wall": 10.0, "loop": 8.0, "memory": 350.0}
            seed_specs[(3, s)] = {"quality": 0.99, "wall": 10.0, "loop": 8.0, "memory": 2800.0}

        # 1. adaptive_ehvi full cycle
        _, state_e, rows_e, events_e, rep_e, count_e = self._run_end_to_end_method("adaptive_ehvi", seed_specs, budget=120.0)
        self.assertGreater(count_e, 0)
        self.assertEqual(state_e["decision"]["selected_index"], 2)
        self.assertTrue(any("surrogate_diagnostics" in h for h in state_e["acquisition_history"]))
        self.assertIn("selection_rule", rep_e)

        # 2. adaptive_staged_seeds with tight budget (75s):
        # c0 completes 2 seeds (20s), c1 runs seed 1 (10s) and gets deferred_weak,
        # c2 runs seed 1 (10s) and completes seed 2 (10s) -> 50s,
        # c3 runs seed 1 (10s) -> 2800MB > max_memory=2500 -> stopped infeasible -> 60s!
        p_s, state_s, rows_s, events_s, rep_s, _ = self._run_end_to_end_method("adaptive_staged_seeds", seed_specs, budget=70.0)
        self.assertEqual(state_s["decision"]["selected_index"], 2)
        self.assertTrue(any(e["kind"] == "adaptive_staged_stopped_infeasible" and e["payload"]["index"] == 3 for e in events_s))
        # Config 3 only ran 1 seed (not 2!), saving budget and never winning
        self.assertEqual(len(state_s["results"]["3"]), 1)
        self.assertFalse(rows_s[3]["complete"])
        self.assertEqual(rows_s[3]["state"], "unresolved")
        self.assertIsNone(rows_s[3]["utility"])
        # Partial results never enter full_observations()
        full_obs = full_observations(p_s, rows_s, Objective(**p_s["objective"]))
        self.assertNotIn(3, [o["index"] for o in full_obs])
        # Presets & report only use confirmed feasible configs
        pre = presets(rows_s, p_s)
        self.assertNotEqual(pre["Accuracy"]["config_id"], rows_s[3]["config_id"])
        self.assertEqual(rep_s["decision"]["selected_index"], 2)

    def test_describe_models_records_block_and_trainable_parameters(self):
        cfg = self.det_config(model="yolo11n.yaml")
        desc = describe_models("detection", [cfg], 4)
        key = architecture_key(cfg)
        self.assertIn(key, desc)
        self.assertIsInstance(desc[key]["block_parameters"], list)
        self.assertGreater(len(desc[key]["block_parameters"]), 5)
        self.assertEqual(sum(desc[key]["block_parameters"]), desc[key]["parameters"])
        self.assertGreater(desc[key]["trainable_parameters"], 0)
        self.assertLessEqual(desc[key]["trainable_parameters"], desc[key]["parameters"])

    def test_oom_guard_vram_free_fallback_and_observed_seed_exceeds_max_memory(self):
        # 1. max_memory=None uses 0.98 * vram_free_mb
        p = self.det_protocol(
            configs=[self.det_config(batch=4, lr0=0.001), self.det_config(batch=8, lr0=0.002)],
            objective=asdict(Objective(time_scale=60.0, memory_scale=512.0, max_memory=None)),
            hardware={"device": "0", "host_label": "test-gpu", "vram_free_mb": 1000.0},
            budget_seconds=100.0,
        )
        now, calls = [0.0], []

        def eval_vf(index, seed, epochs, full, seconds):
            calls.append(index)
            now[0] += 5.0
            return self.full_result(index, seed, quality=0.7, wall=5.0, loop=4.0, memory=300.0)

        class VramStub(CostModel):
            def estimate(self, index):
                base = super().estimate(index)
                if self.fulls:
                    base["predicted_memory_mb"] = 990.0 if index == 1 else 300.0  # 990 > 0.98 * 1000
                    base["point_seconds"] = 5.0
                    base["planning_seconds"] = 6.0
                return base

        with tempfile.TemporaryDirectory() as tmp, patch("research_pipeline.v2.scheduler.CostModel", VramStub):
            ledger = Ledger(Path(tmp) / "events.jsonl", clock=lambda: now[0])
            state = run_search(p, eval_vf, lambda: 100.0 - now[0], ledger, elapsed=lambda: now[0])
        self.assertEqual(calls, [0, 0])
        self.assertEqual(state["memory_risk_deferred"][0]["risk_level"], "hard_exceeds_vram_free")

        # 2. Observed partial seed > max_memory defers remaining seeds without launching seed 2
        p_obs = self.det_protocol(
            configs=[self.det_config(batch=4, lr0=0.001)],
            objective=asdict(Objective(max_memory=500.0)),
            budget_seconds=100.0,
        )
        initial = {
            "results": {"0": [self.full_result(0, 17, memory=600.0)]},
            "probes": [], "actions": [], "ordering": [0],
        }
        calls_obs = []
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp) / "events.jsonl")
            state_obs = run_search(
                p_obs,
                lambda *a: calls_obs.append(a) or self.full_result(0, 29),
                lambda: 100.0,
                ledger,
                initial=initial,
            )
        self.assertEqual(calls_obs, [])
        self.assertEqual(state_obs["memory_risk_deferred"][0]["risk_level"], "observed_seed_exceeded_max_memory")
        # Historical state without memory_risk_deferred omits field in make_report()
        clean_hist = {"results": {"0": [self.full_result(0, 17, memory=600.0)]}, "probes": [], "actions": [], "ordering": [0]}
        self.assertNotIn("memory_risk_deferred", make_report(p_obs, clean_hist, []))

    def test_v2_small_n_and_degenerate_knee_and_v1_preservation(self):
        configs = [self.det_config(lr0=lr) for lr in (0.0003, 0.001, 0.003)]
        p_v1 = self.det_protocol(configs=configs, acquisition={"version": 1, "kernel_ridge": 0.05, "length_scale": 1.0, "exploration": 0.25, "novelty_weight": 0.05})
        p_v2 = self.det_protocol(configs=configs, selection_rule="knee")
        forecasts = {str(i): {"point_seconds": 10.0, "planning_seconds": 12.0, "predicted_memory_mb": 300.0} for i in range(3)}

        # n == 0
        rows0 = measurements(configs, {}, [17, 29], 4, Objective(**p_v2["objective"]))
        out0 = rank_candidates(p_v2, rows0, forecasts)
        self.assertFalse(out0["surrogate"]["fitted"])
        self.assertIsNone(out0["surrogate_diagnostics"]["loocv_rmse"])

        # n == 1 + degenerate knee (< 3 Pareto extremes -> tchebycheff fallback)
        rows1 = measurements(configs, {"0": [self.full_result(0, 17, quality=0.6), self.full_result(0, 29, quality=0.6)]}, [17, 29], 4, Objective(**p_v2["objective"]))
        out1_v2 = rank_candidates(p_v2, rows1, forecasts)
        self.assertFalse(out1_v2["surrogate"]["fitted"])
        self.assertTrue(any("tchebycheff fallback" in r for r in out1_v2["scores"]["1"]["reasons"]))

        # v1 omits surrogate_diagnostics
        out1_v1 = rank_candidates(p_v1, rows1, forecasts)
        self.assertNotIn("surrogate_diagnostics", out1_v1)

    def test_staged_seeds_drains_deferred_weak_when_budget_remains(self):
        # c0 completes (0.75), c1 is weak (0.30) -> deferred while c2 unexplored (0.85) -> c2 completes -> drain c1!
        seed_specs = {}
        for s in (17, 29):
            seed_specs[(0, s)] = {"quality": 0.75, "wall": 10.0, "loop": 8.0, "memory": 300.0}
            seed_specs[(1, s)] = {"quality": 0.30, "wall": 10.0, "loop": 8.0, "memory": 300.0}
            seed_specs[(2, s)] = {"quality": 0.85, "wall": 10.0, "loop": 8.0, "memory": 300.0}
            seed_specs[(3, s)] = {"quality": 0.70, "wall": 10.0, "loop": 8.0, "memory": 300.0}
        _, state, rows, events, _, _ = self._run_end_to_end_method("adaptive_staged_seeds", seed_specs, budget=150.0)
        self.assertTrue(any(e["kind"] == "adaptive_staged_deferred_weak" for e in events))
        self.assertTrue(any(h["reason"] == "staged_seed_drain_completion" for h in state["acquisition_history"]))
        self.assertTrue(all(r["complete"] for r in rows))

    def test_vram_free_between_seed_blocking_adaptive_full_and_staged_seeds(self):
        # Regression test for Challenger finding (scheduler.py:160,345):
        # When max_memory=None and vram_free_mb=1000.0 (effective limit = 980.0 MB),
        # observing 990.0 MB on seed 1 must block launching seed 2 in BOTH adaptive_full and adaptive_staged_seeds.
        configs = [
            self.det_config(batch=4, lr0=0.001),
            self.det_config(batch=8, lr0=0.002),
        ]

        # 1. adaptive_full: bootstrap c0 observes 990 MB on seed 1 -> confirm() must block seed 2
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"task": "detection", "names": ["animal"], "splits": {"train": [], "val": [], "test": [{"image": "t.png"}]}}
            (root / "full_data").mkdir()
            (root / "subset_data").mkdir()
            (root / "full_data" / "data.yaml").write_text("names: [animal]\n", encoding="utf-8")
            (root / "subset_data" / "data.yaml").write_text("names: [animal]\n", encoding="utf-8")
            data_files = {"full_data/data.yaml": sha(root / "full_data" / "data.yaml"), "subset_data/data.yaml": sha(root / "subset_data" / "data.yaml")}
            p_full = self.det_protocol(
                configs=configs, method="adaptive_full", selection_rule="weighted",
                objective=asdict(Objective(time_scale=60.0, memory_scale=512.0, max_memory=None)),
                hardware={"device": "0", "host_label": "test-gpu", "vram_free_mb": 1000.0},
                budget_seconds=120.0, dataset_sha256=digest(manifest), subset_sha256=digest(manifest),
                data_files=data_files, run_directory=str(root),
            )
            write_json(root / "protocol.json", p_full)
            write_json(root / "dataset.json", manifest)
            write_json(root / "subset.json", manifest)
            write_json(root / "status.json", {"phase": "search"})
            (root / "source_snapshot").mkdir()
            now, calls = [0.0], []
            ledger = Ledger(root / "events.jsonl", clock=lambda: now[0])
            ledger.emit("session_started", additional_budget_seconds=120.0, resume=False, preparation_charged_online=True)
            attempt_no = [0]

            def eval_full(index, seed, epochs, full, seconds):
                attempt_no[0] += 1
                calls.append((index, seed))
                now[0] += 5.0
                trial = root / "trials" / f"attempt_{attempt_no[0]:05d}"
                trial.mkdir(parents=True)
                ckpt = trial / "best.pt"
                ckpt.write_bytes(f"ckpt-{index}-{seed}".encode("utf-8"))
                req = {
                    "config": p_full["configs"][index], "seed": seed, "full_epochs": p_full["full_epochs"],
                    "trial_epochs": epochs, "mode": "train" if full else "search", "resume": None,
                    "manifest": str(root / "dataset.json"), "data": str(root / "full_data" / "data.yaml"),
                    "protocol_sha256": p_full["protocol_sha256"],
                    "data_loader": p_full["execution_policy"]["data_loader"], "detection_metric": "map50",
                }
                mem = 990.0 if index == 0 else 300.0
                raw = self.full_result(
                    index, seed, quality=0.75, wall=5.0, loop=4.0, memory=mem,
                    best_epoch=2, best_checkpoint_epoch=2, primary_metric="map50",
                    best_checkpoint=str(ckpt), checkpoint_sha256=sha(ckpt), config=p_full["configs"][index],
                )
                write_json(trial / "request.json", req)
                write_json(trial / "result.json", raw)
                att_id = f"att-{attempt_no[0]}"
                parent = {**raw, "attempt_id": att_id, "trial_directory": str(trial),
                          "request_sha256": sha(trial / "request.json"), "raw_sha256": sha(trial / "result.json")}
                write_json(trial / "parent_result.json", parent)
                ledger.emit("trial_started", attempt_id=att_id)
                ledger.emit("trial_completed", attempt_id=att_id, parent_sha256=sha(trial / "parent_result.json"),
                            worker_seconds=5.0, status=parent["status"], directory=str(trial))
                return parent

            state_full = run_search(p_full, eval_full, lambda: 120.0 - now[0], ledger, elapsed=lambda: now[0])
            write_json(root / "state.json", state_full)
            state_hash = digest(state_full)
            (root / "commits").mkdir()
            write_json(root / "commits" / f"{state_hash}.json", state_full)
            ledger.emit("selection_committed", session_seconds=now[0], original_budget_seconds=120.0,
                        this_execution_budget_seconds=120.0, decision=state_full["decision"],
                        state_sha256=state_hash, state_path=f"commits/{state_hash}.json")
            write_json(root / "decision.locked.json", state_full["decision"])
            publish(root, p_full, state_full, display=False)
            _, _, rows_full, _ = audit_run(root)
            events_full = list(ledger.events)

        self.assertIn((0, 17), calls)
        self.assertNotIn((0, 29), calls)
        self.assertFalse(rows_full[0]["complete"])
        self.assertEqual(rows_full[0]["state"], "unresolved")
        self.assertTrue(any(
            e["kind"] == "candidate_unresolved" and e["payload"]["index"] == 0
            and e["payload"].get("risk_level") == "observed_seed_exceeded_vram_free"
            for e in events_full
        ))

        # 2. adaptive_staged_seeds: c0 succeeds (300 MB), c1 predicted safe (900 MB) but observes 990 MB on seed 1 -> seed 2 blocked
        class StagedVramStub(CostModel):
            def estimate(self, index):
                base = super().estimate(index)
                if self.fulls and index == 1 and not any(r.get("memory_mb", 0) > 980.0 for r in self.fulls.get(1, [])):
                    base["predicted_memory_mb"] = 900.0
                    base["memory_basis"] = "stub underprediction before seed 1"
                return base

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"task": "detection", "names": ["animal"], "splits": {"train": [], "val": [], "test": [{"image": "t.png"}]}}
            (root / "full_data").mkdir()
            (root / "subset_data").mkdir()
            (root / "full_data" / "data.yaml").write_text("names: [animal]\n", encoding="utf-8")
            (root / "subset_data" / "data.yaml").write_text("names: [animal]\n", encoding="utf-8")
            data_files = {"full_data/data.yaml": sha(root / "full_data" / "data.yaml"), "subset_data/data.yaml": sha(root / "subset_data" / "data.yaml")}
            p_staged = self.det_protocol(
                configs=configs, method="adaptive_staged_seeds", selection_rule="weighted",
                objective=asdict(Objective(time_scale=60.0, memory_scale=512.0, max_memory=None)),
                hardware={"device": "0", "host_label": "test-gpu", "vram_free_mb": 1000.0},
                budget_seconds=120.0, dataset_sha256=digest(manifest), subset_sha256=digest(manifest),
                data_files=data_files, run_directory=str(root),
            )
            write_json(root / "protocol.json", p_staged)
            write_json(root / "dataset.json", manifest)
            write_json(root / "subset.json", manifest)
            write_json(root / "status.json", {"phase": "search"})
            (root / "source_snapshot").mkdir()
            now_s, calls_s = [0.0], []
            ledger_s = Ledger(root / "events.jsonl", clock=lambda: now_s[0])
            ledger_s.emit("session_started", additional_budget_seconds=120.0, resume=False, preparation_charged_online=True)
            attempt_s = [0]

            def eval_staged(index, seed, epochs, full, seconds):
                attempt_s[0] += 1
                calls_s.append((index, seed))
                now_s[0] += 5.0
                trial = root / "trials" / f"attempt_{attempt_s[0]:05d}"
                trial.mkdir(parents=True)
                ckpt = trial / "best.pt"
                ckpt.write_bytes(f"ckpt-{index}-{seed}".encode("utf-8"))
                req = {
                    "config": p_staged["configs"][index], "seed": seed, "full_epochs": p_staged["full_epochs"],
                    "trial_epochs": epochs, "mode": "train" if full else "search", "resume": None,
                    "manifest": str(root / "dataset.json"), "data": str(root / "full_data" / "data.yaml"),
                    "protocol_sha256": p_staged["protocol_sha256"],
                    "data_loader": p_staged["execution_policy"]["data_loader"], "detection_metric": "map50",
                }
                mem = 300.0 if index == 0 else 990.0
                qual = 0.65 if index == 0 else 0.95
                raw = self.full_result(
                    index, seed, quality=qual, wall=5.0, loop=4.0, memory=mem,
                    best_epoch=2, best_checkpoint_epoch=2, primary_metric="map50",
                    best_checkpoint=str(ckpt), checkpoint_sha256=sha(ckpt), config=p_staged["configs"][index],
                )
                write_json(trial / "request.json", req)
                write_json(trial / "result.json", raw)
                att_id = f"att-{attempt_s[0]}"
                parent = {**raw, "attempt_id": att_id, "trial_directory": str(trial),
                          "request_sha256": sha(trial / "request.json"), "raw_sha256": sha(trial / "result.json")}
                write_json(trial / "parent_result.json", parent)
                ledger_s.emit("trial_started", attempt_id=att_id)
                ledger_s.emit("trial_completed", attempt_id=att_id, parent_sha256=sha(trial / "parent_result.json"),
                            worker_seconds=5.0, status=parent["status"], directory=str(trial))
                return parent

            with patch("research_pipeline.v2.scheduler.CostModel", StagedVramStub):
                state_staged = run_search(p_staged, eval_staged, lambda: 120.0 - now_s[0], ledger_s, elapsed=lambda: now_s[0])
            write_json(root / "state.json", state_staged)
            state_hash = digest(state_staged)
            (root / "commits").mkdir()
            write_json(root / "commits" / f"{state_hash}.json", state_staged)
            ledger_s.emit("selection_committed", session_seconds=now_s[0], original_budget_seconds=120.0,
                        this_execution_budget_seconds=120.0, decision=state_staged["decision"],
                        state_sha256=state_hash, state_path=f"commits/{state_hash}.json")
            write_json(root / "decision.locked.json", state_staged["decision"])
            publish(root, p_staged, state_staged, display=False)
            _, _, rows_staged, _ = audit_run(root)
            events_staged = list(ledger_s.events)

        self.assertEqual(calls_s, [(0, 17), (0, 29), (1, 17)])
        self.assertEqual(state_staged["decision"]["selected_index"], 0)
        self.assertFalse(rows_staged[1]["complete"])
        self.assertEqual(rows_staged[1]["state"], "unresolved")
        self.assertTrue(any(
            e["kind"] == "adaptive_staged_stopped_infeasible" and e["payload"]["index"] == 1
            and e["payload"].get("memory_limit_kind") == "vram_free"
            for e in events_staged
        ))

        # 3. Resumed initial state with seed 1 > 0.98 * vram_free_mb (max_memory=None) defers via observed_seed_exceeded_vram_free
        p_res = self.det_protocol(
            configs=[self.det_config(batch=4, lr0=0.001)],
            objective=asdict(Objective(max_memory=None)),
            hardware={"device": "0", "host_label": "test-gpu", "vram_free_mb": 1000.0},
            budget_seconds=100.0,
        )
        init_res = {"results": {"0": [self.full_result(0, 17, memory=990.0)]}, "probes": [], "actions": [], "ordering": [0]}
        calls_res = []
        with tempfile.TemporaryDirectory() as tmp:
            ledger_r = Ledger(Path(tmp) / "events.jsonl")
            st_res = run_search(p_res, lambda *a: calls_res.append(a) or self.full_result(0, 29), lambda: 100.0, ledger_r, initial=init_res)
        self.assertEqual(calls_res, [])
        self.assertEqual(st_res["memory_risk_deferred"][0]["risk_level"], "observed_seed_exceeded_vram_free")

    def test_cross_dataset_sensitivity_zero_hardcoding_and_distinct_selections(self):
        # Same candidate config grid across two genuinely distinct datasets/domains:
        # Domain A: small fast-converging 4-class dataset favoring lightweight high-LR yolo26n (c5)
        # Domain B: dense 20-class small-object dataset favoring higher-capacity/resolution yolo11s (c4)
        configs = [
            self.det_config(model="yolo11n.pt", imgsz=320, batch=4, accumulation=4, lr0=0.0003, freeze=0),
            self.det_config(model="yolo11n.pt", imgsz=640, batch=8, accumulation=2, lr0=0.003, freeze=0),
            self.det_config(model="yolo11n.pt", imgsz=640, batch=8, accumulation=2, lr0=0.001, freeze=5),
            self.det_config(model="yolo11s.pt", imgsz=320, batch=8, accumulation=2, lr0=0.001, freeze=0),
            self.det_config(model="yolo11s.pt", imgsz=320, batch=8, accumulation=2, lr0=0.0003, freeze=0),
            self.det_config(model="yolo26n.pt", imgsz=320, batch=16, accumulation=1, lr0=0.002, freeze=0),
        ]

        domain_a = {
            "dataset_source": "domain_a_sparse_4cls",
            "dataset_content_id": "domain-a-sparse-4cls-v1",
            "example_counts": {"train": 120, "val": 30},
            "num_classes": 4,
            "dataset_descriptor": {
                "class_entropy": 0.94,
                "objects_per_image": 1.8,
                "small_box_fraction_area_lt_1pct_not_coco": 0.08,
                "pixel_mean": 0.48,
                "pixel_sd": 0.21,
                "edge_mean": 0.07,
            },
        }
        domain_b = {
            "dataset_source": "domain_b_dense_20cls",
            "dataset_content_id": "domain-b-dense-20cls-v1",
            "example_counts": {"train": 2400, "val": 600},
            "num_classes": 20,
            "dataset_descriptor": {
                "class_entropy": 0.65,
                "objects_per_image": 12.4,
                "small_box_fraction_area_lt_1pct_not_coco": 0.54,
                "pixel_mean": 0.30,
                "pixel_sd": 0.17,
                "edge_mean": 0.15,
            },
        }

        # Verify feature_row reflects dataset descriptors dynamically
        obj_cross = asdict(Objective(time_scale=120.0, memory_scale=2048.0, max_memory=3000.0, weights=(0.80, 0.10, 0.10)))
        p_a = self.det_protocol(configs=configs, objective=obj_cross, selection_rule="topsis", budget_seconds=500.0, **domain_a)
        p_b = self.det_protocol(configs=configs, objective=obj_cross, selection_rule="topsis", budget_seconds=500.0, **domain_b)
        self.assertNotEqual(feature_row(p_a, configs[0]), feature_row(p_b, configs[0]))

        # Live empirical outcomes per domain (distinct quality surface, seed variance, runtime, and VRAM profile)
        specs_a = {
            (0, 17): {"quality": 0.72, "wall": 10.0, "loop": 7.0, "memory": 310.0},
            (0, 29): {"quality": 0.74, "wall": 10.5, "loop": 7.4, "memory": 312.0},
            (1, 17): {"quality": 0.92, "wall": 18.0, "loop": 14.0, "memory": 620.0},
            (1, 29): {"quality": 0.94, "wall": 18.4, "loop": 14.3, "memory": 625.0},
            (2, 17): {"quality": 0.84, "wall": 18.0, "loop": 14.0, "memory": 640.0},
            (2, 29): {"quality": 0.86, "wall": 18.4, "loop": 14.3, "memory": 645.0},
            (3, 17): {"quality": 0.48, "wall": 15.0, "loop": 11.5, "memory": 520.0},
            (3, 29): {"quality": 0.50, "wall": 15.2, "loop": 11.7, "memory": 524.0},
            (4, 17): {"quality": 0.45, "wall": 16.0, "loop": 12.0, "memory": 530.0},
            (4, 29): {"quality": 0.47, "wall": 16.4, "loop": 12.3, "memory": 535.0},
            (5, 17): {"quality": 0.68, "wall": 12.0, "loop": 9.0, "memory": 430.0},
            (5, 29): {"quality": 0.70, "wall": 12.2, "loop": 9.1, "memory": 432.0},
        }
        specs_b = {
            (0, 17): {"quality": 0.42, "wall": 26.0, "loop": 20.0, "memory": 460.0},
            (0, 29): {"quality": 0.52, "wall": 27.0, "loop": 21.0, "memory": 468.0},
            (1, 17): {"quality": 0.44, "wall": 45.0, "loop": 37.0, "memory": 1080.0},
            (1, 29): {"quality": 0.46, "wall": 46.0, "loop": 38.0, "memory": 1090.0},
            (2, 17): {"quality": 0.52, "wall": 44.0, "loop": 36.0, "memory": 1060.0},
            (2, 29): {"quality": 0.56, "wall": 45.0, "loop": 37.0, "memory": 1070.0},
            (3, 17): {"quality": 0.80, "wall": 34.0, "loop": 28.0, "memory": 820.0},
            (3, 29): {"quality": 0.86, "wall": 34.5, "loop": 28.5, "memory": 825.0},
            (4, 17): {"quality": 0.93, "wall": 35.0, "loop": 29.0, "memory": 830.0},
            (4, 29): {"quality": 0.95, "wall": 35.5, "loop": 29.5, "memory": 835.0},
            (5, 17): {"quality": 0.42, "wall": 31.0, "loop": 24.0, "memory": 660.0},
            (5, 29): {"quality": 0.54, "wall": 32.0, "loop": 25.0, "memory": 668.0},
        }

        # 1. CostModel time & VRAM decomposition sensitivity on the same observed indices [0, 2]
        cm_a, cm_b = CostModel(p_a), CostModel(p_b)
        for idx in (0, 2):
            for s in (17, 29):
                sa, sb = specs_a[(idx, s)], specs_b[(idx, s)]
                cm_a.update(idx, self.full_result(idx, s, **sa))
                cm_b.update(idx, self.full_result(idx, s, **sb))
        for unseen in (1, 3, 4, 5):
            ea, eb = cm_a.estimate(unseen), cm_b.estimate(unseen)
            self.assertNotAlmostEqual(ea["point_seconds"], eb["point_seconds"], places=2)
            self.assertNotAlmostEqual(ea["predicted_memory_mb"], eb["predicted_memory_mb"], places=2)
            self.assertNotAlmostEqual(ea["memory_components"]["base_allocator_mb"], eb["memory_components"]["base_allocator_mb"], places=2)
            self.assertNotAlmostEqual(ea["memory_components"]["activation_mb"], eb["memory_components"]["activation_mb"], places=2)

        # 2. Surrogate diagnostics & rank_candidates() sensitivity on observed indices [0, 2, 3, 5]
        obs_indices = (0, 2, 3, 5)
        res_a = {str(i): ([self.full_result(i, s, **specs_a[(i, s)]) for s in (17, 29)] if i in obs_indices else []) for i in range(len(configs))}
        res_b = {str(i): ([self.full_result(i, s, **specs_b[(i, s)]) for s in (17, 29)] if i in obs_indices else []) for i in range(len(configs))}
        rows_a = measurements(configs, res_a, [17, 29], 4, Objective(**p_a["objective"]))
        rows_b = measurements(configs, res_b, [17, 29], 4, Objective(**p_b["objective"]))
        fc_a = {str(i): cm_a.estimate(i) for i in range(len(configs))}
        fc_b = {str(i): cm_b.estimate(i) for i in range(len(configs))}

        rank_a = rank_candidates(p_a, rows_a, fc_a)
        rank_b = rank_candidates(p_b, rows_b, fc_b)
        diag_a = rank_a["surrogate_diagnostics"]
        diag_b = rank_b["surrogate_diagnostics"]
        self.assertNotEqual(
            (diag_a["kernel"], diag_a["length_scale"], diag_a["ridge"]),
            (diag_b["kernel"], diag_b["length_scale"], diag_b["ridge"]),
        )
        self.assertNotAlmostEqual(diag_a["loocv_rmse"], diag_b["loocv_rmse"], places=4)
        self.assertNotEqual(diag_a["predictions"], diag_b["predictions"])
        self.assertEqual(rank_a["ranking"], [1, 4])
        self.assertEqual(rank_b["ranking"], [4, 1])

        # 3. Full run_search sensitivity across adaptive_full, adaptive_ehvi, and adaptive_staged_seeds
        for method in ("adaptive_full", "adaptive_ehvi", "adaptive_staged_seeds"):
            pa_m = self.det_protocol(configs=configs, objective=obj_cross, method=method, selection_rule="topsis", budget_seconds=500.0, **domain_a)
            pb_m = self.det_protocol(configs=configs, objective=obj_cross, method=method, selection_rule="topsis", budget_seconds=500.0, **domain_b)

            def run_domain(proto, specs):
                now = [0.0]
                def eval_fn(index, seed, epochs, full, seconds):
                    sp = specs[(index, seed)]
                    now[0] += sp["wall"]
                    return self.full_result(index, seed, **sp)
                with tempfile.TemporaryDirectory() as tmp:
                    ledger = Ledger(Path(tmp) / "events.jsonl", clock=lambda: now[0])
                    return run_search(proto, eval_fn, lambda: 500.0 - now[0], ledger, elapsed=lambda: now[0])

            st_a = run_domain(pa_m, specs_a)
            st_b = run_domain(pb_m, specs_b)
            self.assertNotEqual(st_a["decision"]["selected_index"], st_b["decision"]["selected_index"])
            self.assertEqual(st_a["decision"]["selected_index"], 1)
            self.assertEqual(st_b["decision"]["selected_index"], 4)
            # Last acquisition step's surrogate diagnostics & cost predictions differ across domains
            last_diag_a = next(h["surrogate_diagnostics"] for h in reversed(st_a["acquisition_history"]) if "surrogate_diagnostics" in h)
            last_diag_b = next(h["surrogate_diagnostics"] for h in reversed(st_b["acquisition_history"]) if "surrogate_diagnostics" in h)
            self.assertNotEqual(last_diag_a["predictions"], last_diag_b["predictions"])
            self.assertNotAlmostEqual(
                st_a["cost_predictions"]["4"]["predicted_memory_mb"],
                st_b["cost_predictions"]["4"]["predicted_memory_mb"],
                places=2,
            )


if __name__ == "__main__":
    unittest.main()
