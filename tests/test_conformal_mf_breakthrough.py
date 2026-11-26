"""Unit tests for Phase 1 breakthrough: Conformal VRAM, epsilon_constraint, conformal_mf, mf_no_conformal, tpe_bo."""

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_pipeline.v2.acquisition import rank_candidates
from research_pipeline.v2.conformal_vram import (
    calibrate_online_2point,
    evaluate_conformal_vram,
    fit_conformal_vram_card,
    predict_conformal_vram,
    verify_conformal_vram_card,
)
from research_pipeline.v2.ledger import Ledger
from research_pipeline.v2.scheduler import METHODS, run_search
from research_pipeline.v2.schema import lock_protocol, measurements
from research_pipeline.v2.selection import RULES, select
from yolo_tuner.research import Objective


def _make_record(batch, imgsz, params=2_600_000, flops=0.01653, precision="fp16", freeze=0, optimizer="AdamW", gpu_shift=0.0, gpu_scale=1.0, noise=0.0):
    cfg = {
        "model": f"model_{params}.pt",
        "imgsz": imgsz,
        "batch": batch,
        "accumulation": 1,
        "precision": precision,
        "optimizer": optimizer,
        "lr0": 0.001,
        "weight_decay": 0.0005,
        "freeze": freeze,
    }
    desc = {
        "model_identity": cfg["model"],
        "family_scale": [2, 1 if params < 5_000_000 else 2],
        "parameters": params,
        "layers": 24,
        "gflops_reference_32": flops,
        "trainable_parameters": int(params * (1.0 - min(freeze, 24) / 24.0)),
    }
    prec_f = 2.0 if precision == "fp32" else 1.0
    m_f = (flops / 0.01653) ** 0.75
    act = batch * ((imgsz / 320.0) ** 2) * prec_f * m_f * max(0.25, 1.0 - 1.5 * freeze / 24.0)
    static_mb = ((6.0 if precision == "fp16" else 4.0) * params + 12.0 * desc["trainable_parameters"]) / (1024.0 ** 2)
    obs_mem = static_mb + gpu_scale * (220.0 + 95.0 * act) + gpu_shift + noise
    return {"config": cfg, "descriptor": desc, "memory_mb": float(obs_mem)}


class ConformalVRAMAndMethodTests(unittest.TestCase):
    def test_split_conformal_empirical_coverage_and_online_2pt_transfer(self):
        rng = random.Random(2027)
        records = []
        for params, flops in ((2_600_000, 0.01653), (9_450_000, 0.05428), (5_500_000, 0.032)):
            for imgsz in (320, 480, 640):
                for batch in (4, 8, 16):
                    for prec in ("fp16", "fp32"):
                        noise = rng.gauss(0.0, 22.0)
                        records.append(_make_record(batch, imgsz, params=params, flops=flops, precision=prec, noise=noise))

        ev = evaluate_conformal_vram(records, alphas=(0.10, 0.05), n_splits=30, seed=42, limit_mb=1800.0)
        self.assertGreaterEqual(ev["splits_summary"]["0.05"]["empirical_coverage_mean"], 0.90)
        self.assertGreaterEqual(ev["splits_summary"]["0.1"]["empirical_coverage_mean"], 0.85)
        self.assertIsNotNone(ev["splits_summary"]["0.05"]["oom_prevention_rate"])
        self.assertTrue(ev["leave_one_model_out"])

        # Cross-GPU shift + scale transfer: 2-point online calibration should cut error sharply.
        half = len(records) // 2
        card = fit_conformal_vram_card(records[:half], records[half:], alphas=(0.05,), task="detection")
        self.assertTrue(verify_conformal_vram_card(card))

        target_gpu_recs = [
            _make_record(b, s, params=9_450_000, flops=0.05428, gpu_shift=180.0, gpu_scale=1.35)
            for s in (320, 640) for b in (2, 4, 8, 16)
        ]
        cal_2pt = calibrate_online_2point(card, target_gpu_recs[:2], task="detection")
        self.assertEqual(cal_2pt["mode"], "2-point-affine")
        err_before = [abs(r["memory_mb"] - predict_conformal_vram(card, r["config"], r["descriptor"], alpha=0.05)["point_mb"]) for r in target_gpu_recs[2:]]
        err_after = [abs(r["memory_mb"] - predict_conformal_vram(card, r["config"], r["descriptor"], alpha=0.05, online_cal=cal_2pt)["point_mb"]) for r in target_gpu_recs[2:]]
        self.assertLess(sum(err_after), sum(err_before) * 0.35)

    def test_epsilon_constraint_selection_and_acquisition_parity(self):
        self.assertIn("epsilon_constraint", RULES)
        cfgs = [
            {"model": f"m{i}.pt", "imgsz": 320 + i * 64, "batch": 4, "accumulation": 1, "precision": "fp16", "optimizer": "AdamW", "lr0": 1e-3, "weight_decay": 5e-4, "freeze": 0}
            for i in range(4)
        ]
        # c0: fast & cheap, loses > 0.03 quality; c1: slightly lower q (-0.015 <= 0.03) & 4x faster; c2: highest q=0.80, very slow; c3: dominated
        rows = [
            {"index": 0, "config_id": "c0", "config": cfgs[0], "complete": True, "completed_seeds": [17], "required_seeds": [17],
             "quality_mean": 0.72, "quality_min": 0.72, "time_mean": 10.0, "time_max": 10.0, "memory_max": 300.0, "quality_sd": 0.0, "runs": [], "utility": 0.0, "state": "confirmed_feasible"},
            {"index": 1, "config_id": "c1", "config": cfgs[1], "complete": True, "completed_seeds": [17], "required_seeds": [17],
             "quality_mean": 0.785, "quality_min": 0.785, "time_mean": 25.0, "time_max": 25.0, "memory_max": 450.0, "quality_sd": 0.0, "runs": [], "utility": 0.0, "state": "confirmed_feasible"},
            {"index": 2, "config_id": "c2", "config": cfgs[2], "complete": True, "completed_seeds": [17], "required_seeds": [17],
             "quality_mean": 0.80, "quality_min": 0.80, "time_mean": 120.0, "time_max": 120.0, "memory_max": 900.0, "quality_sd": 0.0, "runs": [], "utility": 0.0, "state": "confirmed_feasible"},
        ]
        obj = Objective("balanced", 0.0, None, 2000.0, 100.0, 1000.0, (0.6, 0.25, 0.15))
        for r in rows:
            r["utility"] = obj.value(r)
        prot = {
            "task": "detection",
            "training_seeds": [17],
            "search_seed": 17,
            "configs": cfgs,
            "model_descriptors": {c["model"]: {"model_identity": c["model"], "parameters": 2_600_000, "layers": 24, "gflops_reference_32": 0.0165} for c in cfgs},
            "example_counts": {"train": 100, "val": 20, "test": 20},
            "full_epochs": 5,
            "objective": {"name": "balanced", "min_quality": 0.0, "max_time": None, "max_memory": 2000.0, "time_scale": 100.0, "memory_scale": 1000.0, "weights": [0.6, 0.25, 0.15]},
            "selection_rule": "epsilon_constraint",
            "quality_tolerance": 0.03,
            "scheduler": {"policy_version": 3, "probe_fraction": 0.1, "cost_margin": 1.25, "cleanup_seconds": 1},
            "acquisition": {"version": 2, "kernel_ridge": 0.05, "length_scale": 1.0, "exploration": 0.25, "novelty_weight": 0.05},
        }
        sel = select(rows, prot)
        self.assertEqual(sel["selected_config_id"], "c1")
        acq = rank_candidates(prot, rows[:2], {3: {"point_seconds": 30.0, "predicted_memory_mb": 400.0}}, candidates=[{"index": 3, "missing_seeds": [17]}])
        self.assertEqual(acq["ranking"], [3])
        self.assertEqual(acq["scores"]["3"]["selection_rule"], "epsilon_constraint")

    def test_conformal_mf_prevents_oom_and_tpe_bo_runs_deterministically(self):
        self.assertIn("conformal_mf", METHODS)
        self.assertIn("mf_no_conformal", METHODS)
        self.assertIn("tpe_bo", METHODS)

        # Build 6 configs: c0,c1,c2 safe; c3,c4,c5 genuinely OOM (> 1000 MiB ceiling).
        fit_recs = [_make_record(b, s, params=2_600_000, flops=0.01653) for s in (320, 480) for b in (2, 4, 8)]
        cal_recs = [_make_record(b, s, params=2_600_000, flops=0.01653) for s in (320, 640) for b in (4, 8)]
        card = fit_conformal_vram_card(fit_recs, cal_recs, alphas=(0.05,), task="detection")

        cfgs = [
            {"model": "m.pt", "imgsz": 320, "batch": 2, "accumulation": 2, "precision": "fp16", "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.0005, "freeze": 0},  # safe, ok q
            {"model": "m.pt", "imgsz": 320, "batch": 4, "accumulation": 1, "precision": "fp16", "optimizer": "AdamW", "lr0": 0.002, "weight_decay": 0.0005, "freeze": 0},  # safe, best q
            {"model": "m.pt", "imgsz": 480, "batch": 4, "accumulation": 1, "precision": "fp16", "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.0005, "freeze": 0},  # safe, slower
            {"model": "m.pt", "imgsz": 640, "batch": 16, "accumulation": 1, "precision": "fp32", "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.0005, "freeze": 0}, # OOM!
            {"model": "m.pt", "imgsz": 800, "batch": 16, "accumulation": 1, "precision": "fp32", "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.0005, "freeze": 0}, # OOM!
            {"model": "m.pt", "imgsz": 640, "batch": 24, "accumulation": 1, "precision": "fp32", "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.0005, "freeze": 0}, # OOM!
        ]
        descriptors = {"m.pt": {"model_identity": "m.pt", "parameters": 2_600_000, "layers": 24, "gflops_reference_32": 0.01653, "trainable_parameters": 2_600_000}}

        def make_prot(method):
            return lock_protocol({
                "implementation_version": "1.0.0",
                "mode": "online",
                "method": method,
                "selection_rule": "epsilon_constraint",
                "quality_tolerance": 0.03,
                "conformal_vram_card": card if method == "conformal_mf" else None,
                "model_descriptors": descriptors,
                "dataset_descriptor": {"class_entropy": 1.0, "objects_per_image": 2.0, "small_box_fraction_area_lt_1pct_not_coco": 0.1, "pixel_mean": 0.5, "pixel_sd": 0.2, "edge_mean": 0.05},
                "run_directory": "tmp",
                "task": "detection",
                "detection_metric": "map50",
                "quality_definition": "best validation mAP50 within declared horizon",
                "dataset_source": "synthetic",
                "dataset_sha256": "a" * 64,
                "dataset_content_id": "b" * 64,
                "subset_sha256": "c" * 64,
                "hardware": {"device": "0", "gpu": "MockGPU", "vram_free_mb": 1800.0},
                "runtime_versions": {"python": "3.12"},
                "configs": cfgs,
                "training_seeds": [17, 29],
                "split_seed": 101,
                "search_seed": 17,
                "full_epochs": 4,
                "probe_epochs": 1,
                "trial_timeout": 100.0,
                "budget_seconds": 36.0,
                "objective": {"name": "balanced", "min_quality": 0.0, "max_time": None, "max_memory": 1800.0, "time_scale": 50.0, "memory_scale": 1000.0, "weights": [0.6, 0.25, 0.15]},
                "example_counts": {"train": 160, "val": 40, "test": 40},
                "num_classes": 4,
                "scheduler": {"policy_version": 3, "probe_fraction": 0.42, "cost_margin": 1.25, "cleanup_seconds": 1.0, "rungs": [1, 2, 4]},
                "acquisition": {"version": 2, "kernel_ridge": 0.05, "length_scale": 1.0, "exploration": 0.25, "novelty_weight": 0.05},
                "source_files": {},
                "model_files": {},
                "data_files": {},
            })

        def run_sim(method):
            prot = make_prot(method)
            clock = [0.0]
            q_table = {0: 0.72, 1: 0.88, 2: 0.89, 3: 0.90, 4: 0.90, 5: 0.90}
            mem_table = {0: 450.0, 1: 520.0, 2: 750.0, 3: 3200.0, 4: 4800.0, 5: 5500.0}
            cost_table = {0: 2.0, 1: 2.5, 2: 4.5, 3: 8.0, 4: 10.0, 5: 12.0}
            with tempfile.TemporaryDirectory() as td:
                ledger = Ledger(Path(td) / "events.jsonl")
                def evaluate(idx, seed, epochs, full, avail):
                    dt = (cost_table[idx] * (epochs / 4.0)) if idx < 3 else 4.0
                    clock[0] += dt
                    if idx >= 3:
                        return {
                            "status": "oom",
                            "epochs_completed": 0,
                            "worker_wall_seconds": dt,
                            "memory_mb": mem_table[idx],
                            "optimizer_steps": 0,
                            "training_loss_finite": False,
                            "actual_config_verified": False,
                            "checkpoint_sha256": None,
                            "quality": None,
                        }
                    q = q_table[idx] - (0.08 if not full else 0.0)
                    return {
                        "status": "ok",
                        "epochs_completed": epochs,
                        "worker_wall_seconds": dt,
                        "measured_loop_seconds": dt * 0.8,
                        "memory_mb": mem_table[idx],
                        "optimizer_steps": epochs * 10,
                        "training_loss_finite": True,
                        "actual_config_verified": True,
                        "checkpoint_sha256": "d" * 64,
                        "quality": q,
                    }
                st = run_search(prot, evaluate, lambda: max(0.0, 36.0 - clock[0]), ledger)
                return st

        st_cmf = run_sim("conformal_mf")
        st_no = run_sim("mf_no_conformal")
        st_tpe = run_search_st = run_sim("tpe_bo")

        oom_cmf = sum(1 for r in ([x for xs in st_cmf["results"].values() for x in xs] + st_cmf["probes"]) if r["status"] == "oom")
        oom_no = sum(1 for r in ([x for xs in st_no["results"].values() for x in xs] + st_no["probes"]) if r["status"] == "oom")
        self.assertEqual(oom_cmf, 0)
        self.assertGreaterEqual(oom_no, 1)
        self.assertEqual(len(st_cmf.get("conformal_pruned") or []), 3)
        self.assertIsNotNone(st_cmf["decision"]["selected_config_id"])
        self.assertIsNotNone(st_tpe["decision"]["selected_config_id"])
        # epsilon_constraint must choose c1 (q=0.88 vs max 0.89 within 0.03, and ~2x faster / lower VRAM than c2)
        rows_cmf = measurements(cfgs, st_cmf["results"], [17, 29], 4, Objective(**make_prot("conformal_mf")["objective"]))
        self.assertEqual(select(rows_cmf, make_prot("conformal_mf"))["selected_index"], 1)

    def test_cli_flags_and_protocol_validation_for_breakthrough_features(self):
        from research_pipeline.v2.cli import parser
        from research_pipeline.v2.schema import lock_protocol

        p = parser()
        args = p.parse_args([
            "optimize",
            "--task", "detection",
            "--data", "african_wildlife_benchmark",
            "--method", "conformal_mf",
            "--selection-rule", "epsilon_constraint",
            "--quality-tolerance", "0.025",
            "--conformal-vram-card", "configs/conformal_vram_card_rtx3050.json",
        ])
        self.assertEqual(args.method, "conformal_mf")
        self.assertEqual(args.selection_rule, "epsilon_constraint")
        self.assertAlmostEqual(args.quality_tolerance, 0.025)
        self.assertEqual(args.conformal_vram_card, "configs/conformal_vram_card_rtx3050.json")

        fit_recs = [_make_record(b, s, params=2_600_000, flops=0.01653) for s in (320, 480) for b in (2, 4)]
        cal_recs = [_make_record(b, s, params=2_600_000, flops=0.01653) for s in (320, 640) for b in (2, 4)]
        card = fit_conformal_vram_card(fit_recs, cal_recs, alphas=(0.05,), task="detection")

        prot = {
            "implementation_version": "1.0.0",
            "mode": "online",
            "method": "conformal_mf",
            "selection_rule": "epsilon_constraint",
            "quality_tolerance": 0.025,
            "conformal_vram_card": card,
            "model_descriptors": {"m.pt": {"model_identity": "m.pt", "parameters": 2_600_000, "layers": 24, "gflops_reference_32": 0.01653}},
            "dataset_descriptor": {"class_entropy": 1.0},
            "run_directory": "tmp",
            "task": "detection",
            "detection_metric": "map50",
            "quality_definition": "best validation mAP50 within declared horizon",
            "dataset_source": "synthetic",
            "dataset_sha256": "a" * 64,
            "dataset_content_id": "b" * 64,
            "subset_sha256": "c" * 64,
            "hardware": {"device": "cpu"},
            "runtime_versions": {"python": "3.12"},
            "configs": [{"model": "m.pt", "imgsz": 320, "batch": 4, "accumulation": 1, "precision": "fp32", "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.0005, "freeze": 0}],
            "training_seeds": [17],
            "split_seed": 101,
            "search_seed": 17,
            "full_epochs": 2,
            "probe_epochs": 1,
            "trial_timeout": 60.0,
            "budget_seconds": 30.0,
            "objective": {"name": "balanced", "min_quality": 0.0, "max_time": None, "max_memory": 1800.0, "time_scale": 50.0, "memory_scale": 1000.0, "weights": [0.6, 0.25, 0.15]},
            "example_counts": {"train": 10, "val": 5, "test": 5},
            "num_classes": 2,
            "scheduler": {"policy_version": 3, "probe_fraction": 0.42, "cost_margin": 1.25, "cleanup_seconds": 1.0},
            "acquisition": {"version": 2, "kernel_ridge": 0.05, "length_scale": 1.0, "exploration": 0.25, "novelty_weight": 0.05},
            "source_files": {},
            "model_files": {},
            "data_files": {},
        }
        locked = lock_protocol(prot)
        self.assertEqual(locked["selection_rule"], "epsilon_constraint")
        self.assertAlmostEqual(locked["quality_tolerance"], 0.025)

    def test_risk_gated_conformal_mf_handoff_to_tpe_and_webui_resources(self):
        from unittest.mock import patch
        from research_pipeline.v2.web_ui.data_access import launch_job_action, list_workspace_resources

        # Verify risk-gated handoff triggers on unconstrained memory when no card is loaded
        cfgs = [
            {"model": "cnn_classifier", "imgsz": 32, "batch": 16, "accumulation": 1, "precision": "fp32", "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.01, "width": 16},
            {"model": "cnn_classifier", "imgsz": 32, "batch": 16, "accumulation": 1, "precision": "fp32", "optimizer": "AdamW", "lr0": 0.0003, "weight_decay": 0.01, "width": 32},
            {"model": "cnn_classifier", "imgsz": 32, "batch": 16, "accumulation": 1, "precision": "fp32", "optimizer": "AdamW", "lr0": 0.003, "weight_decay": 0.01, "width": 64},
            {"model": "cnn_classifier", "imgsz": 48, "batch": 16, "accumulation": 1, "precision": "fp32", "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.01, "width": 16},
        ]
        descriptors = {"cnn_classifier": {"model_identity": "cnn_classifier", "parameters": 50_000, "layers": 6, "gflops_reference_32": 0.01, "trainable_parameters": 50_000}}
        prot = lock_protocol({
            "implementation_version": "1.0.0",
            "mode": "online",
            "method": "conformal_mf",
            "selection_rule": "epsilon_constraint",
            "quality_tolerance": 0.03,
            "conformal_vram_card": None,
            "model_descriptors": descriptors,
            "dataset_descriptor": {"class_entropy": 2.0},
            "run_directory": "tmp",
            "task": "classification",
            "quality_definition": "final epoch validation accuracy",
            "dataset_source": "synthetic",
            "dataset_sha256": "a" * 64,
            "dataset_content_id": "b" * 64,
            "subset_sha256": "c" * 64,
            "hardware": {"device": "cpu", "ram_available_mb": 8000.0},
            "runtime_versions": {"python": "3.12"},
            "configs": cfgs,
            "training_seeds": [17],
            "split_seed": 101,
            "search_seed": 17,
            "full_epochs": 5,
            "probe_epochs": 1,
            "trial_timeout": 100.0,
            "budget_seconds": 30.0,
            "objective": {"name": "balanced", "min_quality": 0.0, "max_time": None, "max_memory": 4000.0, "time_scale": 100.0, "memory_scale": 1024.0, "weights": [0.6, 0.25, 0.15]},
            "example_counts": {"train": 200, "val": 50, "test": 50},
            "num_classes": 10,
            "scheduler": {"policy_version": 3, "probe_fraction": 0.42, "cost_margin": 1.25, "cleanup_seconds": 1.0},
            "acquisition": {"version": 2, "kernel_ridge": 0.05, "length_scale": 1.0, "exploration": 0.25, "novelty_weight": 0.05},
            "source_files": {},
            "model_files": {},
            "data_files": {},
        })
        clock = [0.0]
        with tempfile.TemporaryDirectory() as td:
            ledger = Ledger(Path(td) / "events.jsonl")
            def eval_cpu(idx, seed, epochs, full, avail):
                dt = 3.0 if full else 1.5
                clock[0] += dt
                return {
                    "status": "ok",
                    "epochs_completed": epochs,
                    "worker_wall_seconds": dt,
                    "measured_loop_seconds": dt * 0.8,
                    "memory_mb": 420.0 + idx * 10.0,  # << 0.50 * 4000 MiB
                    "optimizer_steps": epochs * 10,
                    "training_loss_finite": True,
                    "actual_config_verified": True,
                    "checkpoint_sha256": "e" * 64,
                    "quality": 0.20 + 0.04 * idx if full else 0.08,
                }
            st = run_search(prot, eval_cpu, lambda: max(0.0, 30.0 - clock[0]), ledger)
            # Only 2 probes needed before risk_gated_tpe_handoff triggers cleanly
            self.assertEqual(len(st["probes"]), 2)
            self.assertIsNotNone(st["decision"]["selected_config_id"])
            ev_text = (Path(td) / "events.jsonl").read_text(encoding="utf-8")
            self.assertIn("risk_gated_tpe_handoff", ev_text)

        # Verify Web UI resources and CLI builder support breakthrough flags
        res = list_workspace_resources()
        self.assertIn("conformal_cards", res)
        self.assertTrue(any(c["name"] == "conformal_vram_card_rtx3050.json" for c in res["conformal_cards"]))
        with tempfile.TemporaryDirectory() as td:
            class _DummyProc:
                pid = 99999
                stdout = None
                def poll(self):
                    return 0
            with patch("research_pipeline.v2.web_ui.data_access.subprocess.Popen", return_value=_DummyProc()) as mp:
                out = launch_job_action(Path(td), {
                    "confirmed": True,
                    "mode": "optimize",
                    "data": "runs/datasets/african_wildlife_benchmark",
                    "method": "conformal_mf",
                    "selection_rule": "epsilon_constraint",
                    "quality_tolerance": 0.03,
                    "max_memory": 1800,
                    "conformal_vram_card": "configs/conformal_vram_card_rtx3050.json",
                    "output": "ui_test_out",
                })
                cmd_args = mp.call_args[0][0]
                self.assertIn("--conformal-vram-card", cmd_args)
                self.assertIn("--quality-tolerance", cmd_args)
                self.assertIn("--max-memory", cmd_args)
                self.assertTrue(out["launched"])


if __name__ == "__main__":
    unittest.main()
