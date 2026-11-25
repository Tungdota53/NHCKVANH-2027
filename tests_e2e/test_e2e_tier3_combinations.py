"""Tier 3 Cross-Feature Combinations: 19 pairwise & workflow integration tests across R1-R5."""

from __future__ import annotations

import json
import math
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
from research_pipeline.v2.ledger import Ledger, read_events
from research_pipeline.v2.model_features import describe_models, trainable_parameters_for
from research_pipeline.v2.report import make_report, publish
from research_pipeline.v2.workflow import audit_run, source_map
from yolo_tuner.pipeline import runtime_info, write_json
from yolo_tuner.research import Objective

from tests_e2e._e2e_fixtures import (
    REPO_ROOT,
    YOLO_TUNER_SHA256,
    assert_zero_ansi,
    audit_count,
    discover_ui_surface,
    extract_csrf_token,
    find_audit_dict,
    http_get,
    http_post,
    launch_ui_server,
    make_det_protocol,
    make_dirty_yolo_dataset,
    make_mini_cls_dataset,
    make_mini_yolo_dataset,
    make_synthetic_n5_study_dir,
    run_cli,
    sha256_file,
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


def _outcome(aid, idx, seed, epochs, q, wall, loop, mem):
    return {
        "attempt_id": aid,
        "index": idx,
        "seed": seed,
        "mode": "train",
        "status": "ok",
        "epochs_completed": epochs,
        "optimizer_steps": 6,
        "training_loss_finite": True,
        "actual_config_verified": True,
        "checkpoint_sha256": f"ckpt-{aid}",
        "quality": q,
        "memory_mb": mem,
        "worker_wall_seconds": wall,
        "measured_loop_seconds": loop,
    }


class TestTier3Combinations(unittest.TestCase):
    def test_c01_inspect_dirty_dedup_dataset_reinspect_plan_search(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dirty_yaml = make_dirty_yolo_dataset(root / "dirty", include_bad_boxes=False)
            r1 = run_cli(["inspect", "--data", str(dirty_yaml), "--device", "cpu", "--json"])
            self.assertEqual(r1.returncode, 0, msg=r1.stderr)
            a1 = find_audit_dict(json.loads(r1.stdout))
            self.assertGreater(audit_count(a1, "cross_split_leakage_count", "cross_split_leakage", "leakage.cross_split_count", "leakage.cross_split"), 0)

            clean_dir = root / "clean_copy"
            dd = run_cli(["dedup-dataset", "--data", str(dirty_yaml), "--output", str(clean_dir)])
            self.assertEqual(dd.returncode, 0, msg=dd.stderr + dd.stdout)

            clean_yaml = clean_dir / "data.yaml"
            r2 = run_cli(["inspect", "--data", str(clean_yaml), "--device", "cpu", "--json"])
            self.assertEqual(r2.returncode, 0, msg=r2.stderr)
            a2 = find_audit_dict(json.loads(r2.stdout))
            self.assertEqual(audit_count(a2, "intra_split_duplicates_count", "intra_split_duplicates", "duplicates.intra_split_count", "duplicates.intra_split"), 0)
            self.assertEqual(audit_count(a2, "cross_split_leakage_count", "cross_split_leakage", "leakage.cross_split_count", "leakage.cross_split"), 0)

            ps = run_cli([
                "plan-search", "--data", str(clean_yaml), "--device", "cpu",
                "--space", "compact", "--full-epochs", "1", "--seeds", "17",
                "--budget-seconds", "60", "--output", str(root / "planned"),
            ])
            self.assertEqual(ps.returncode, 0, msg=ps.stderr + ps.stdout)

    def test_c02_adaptive_full_v2_tchebycheff_with_cross_memory_and_surrogate_diag(self):
        cfgs = [_det_cfg(imgsz=s, lr0=lr) for s, lr in ((160, 0.001), (320, 0.0003), (160, 0.003), (320, 0.005))]
        descs = describe_models("detection", cfgs, 2)
        prot = make_det_protocol(
            cfgs, method="adaptive_full", selection_rule="tchebycheff",
            acquisition_version=2, training_seeds=[17], budget_seconds=80.0,
            max_memory=1800.0, model_descriptors=descs,
        )
        with tempfile.TemporaryDirectory() as td:
            ledger = Ledger(Path(td) / "events.jsonl")
            clock = [0.0]

            def eval_fn(idx, seed, epochs, full, seconds):
                clock[0] += 4.0
                return _outcome(f"a{idx}", idx, seed, epochs, 0.55 + 0.08 * idx, 4.0, 2.5, 450.0 + 100 * idx)

            state = sched_mod.run_search(prot, eval_fn, lambda: 80.0 - clock[0], ledger, elapsed=lambda: clock[0])
            rows = schema_mod.measurements(cfgs, state["results"], [17], 2, Objective(**prot["objective"]))
            sel = sel_mod.select(rows, prot)
            self.assertEqual(sel["selection_rule"], "tchebycheff")
            self.assertIsNotNone(sel["selected_index"])
            self.assertTrue(any("surrogate_diagnostics" in s for s in state["acquisition_history"]))
            cross_preds = [
                r.get("scheduling_prediction")
                for rows_list in state["results"].values()
                for r in rows_list
                if isinstance(r.get("scheduling_prediction"), dict)
                and r["scheduling_prediction"].get("predicted_memory_mb") is not None
            ]
            self.assertTrue(cross_preds, "Expected at least one trial with predicted_memory_mb in scheduling_prediction")
            for sp in cross_preds:
                mem_mb = sp["predicted_memory_mb"]
                self.assertIsInstance(mem_mb, (int, float))
                self.assertNotIsInstance(mem_mb, bool)
                self.assertTrue(math.isfinite(float(mem_mb)) and float(mem_mb) > 0.0)

    def test_c03_adaptive_full_v2_topsis_with_observed_cross_cost(self):
        cfgs = [_det_cfg(imgsz=s, lr0=lr) for s, lr in ((160, 0.001), (320, 0.0003), (160, 0.003), (320, 0.005))]
        prot = make_det_protocol(cfgs, method="adaptive_full", selection_rule="topsis", acquisition_version=2, training_seeds=[17], budget_seconds=60.0)
        with tempfile.TemporaryDirectory() as td:
            ledger = Ledger(Path(td) / "events.jsonl")
            clock = [0.0]

            def eval_fn(idx, seed, epochs, full, seconds):
                clock[0] += 5.0
                return _outcome(f"a{idx}", idx, seed, epochs, 0.60 + 0.07 * idx, 5.0, 2.0, 500.0)

            state = sched_mod.run_search(prot, eval_fn, lambda: 60.0 - clock[0], ledger, elapsed=lambda: clock[0])
            rows = schema_mod.measurements(cfgs, state["results"], [17], 2, Objective(**prot["objective"]))
            sel = sel_mod.select(rows, prot)
            self.assertEqual(sel["selection_rule"], "topsis")
            self.assertIsNotNone(sel["selected_index"])

    def test_c04_adaptive_full_v2_knee_with_make_report(self):
        cfgs = [_det_cfg(imgsz=s, lr0=lr) for s, lr in ((160, 0.001), (320, 0.0003), (160, 0.003), (320, 0.005))]
        prot = make_det_protocol(cfgs, method="adaptive_full", selection_rule="knee", acquisition_version=2, training_seeds=[17], budget_seconds=60.0)
        with tempfile.TemporaryDirectory() as td:
            ledger = Ledger(Path(td) / "events.jsonl")
            ledger.emit("session_started", method="adaptive_full", search_seed=17)
            clock = [0.0]

            def eval_fn(idx, seed, epochs, full, seconds):
                clock[0] += 4.0
                return _outcome(f"a{idx}", idx, seed, epochs, 0.50 + 0.1 * idx, 4.0, 2.5, 400.0 + 120 * idx)

            state = sched_mod.run_search(prot, eval_fn, lambda: 60.0 - clock[0], ledger, elapsed=lambda: clock[0])
            ledger.emit("selection_committed", session_seconds=clock[0], decision={})
            rep = make_report(prot, state, read_events(Path(td) / "events.jsonl"))
            self.assertEqual(rep["selection_rule"], "knee")
            self.assertIsNotNone(rep["recommendation"])

    def test_c05_adaptive_ehvi_with_oom_guard_and_3d_pareto(self):
        cfgs = [
            _det_cfg(imgsz=160, batch=2, lr0=0.001),
            _det_cfg(imgsz=640, batch=16, lr0=0.001),
            _det_cfg(imgsz=160, batch=2, lr0=0.0003),
            _det_cfg(imgsz=160, batch=2, lr0=0.003),
        ]
        descs = describe_models("detection", cfgs, 2)
        prot = make_det_protocol(
            cfgs, method="adaptive_ehvi", acquisition_version=2,
            training_seeds=[17], budget_seconds=60.0, max_memory=950.0, model_descriptors=descs,
        )
        with tempfile.TemporaryDirectory() as td:
            ledger = Ledger(Path(td) / "events.jsonl")
            clock = [0.0]
            ran = []

            def eval_fn(idx, seed, epochs, full, seconds):
                ran.append(idx)
                clock[0] += 4.0
                mem = 4000.0 if idx == 1 else 250.0 + 40 * idx
                return _outcome(f"a{idx}", idx, seed, epochs, 0.6 + 0.05 * idx, 4.0, 2.5, mem)

            state = sched_mod.run_search(prot, eval_fn, lambda: 60.0 - clock[0], ledger, elapsed=lambda: clock[0])
            self.assertNotIn(1, ran)
            rows = schema_mod.measurements(cfgs, state["results"], [17], 2, Objective(**prot["objective"]))
            hv = paper_mod.hypervolume_3d(rows, Objective(**prot["objective"]))
            self.assertGreater(hv, 0.0)

    def test_c06_adaptive_staged_seeds_partial_excluded_from_selection_and_report(self):
        cfgs = [_det_cfg(lr0=lr) for lr in (0.001, 0.0003, 0.003)]
        prot = make_det_protocol(cfgs, method="adaptive_staged_seeds", acquisition_version=2, training_seeds=[17, 29], budget_seconds=22.0)
        with tempfile.TemporaryDirectory() as td:
            ledger = Ledger(Path(td) / "events.jsonl")
            ledger.emit("session_started", method="adaptive_staged_seeds", search_seed=17)
            clock = [0.0]

            def eval_fn(idx, seed, epochs, full, seconds):
                clock[0] += 5.0
                return _outcome(f"a{idx}_{seed}", idx, seed, epochs, 0.9 if idx == 2 else 0.7, 5.0, 3.5, 450.0)

            state = sched_mod.run_search(prot, eval_fn, lambda: 22.0 - clock[0], ledger, elapsed=lambda: clock[0])
            ledger.emit("selection_committed", session_seconds=clock[0], decision={})
            rep = make_report(prot, state, read_events(Path(td) / "events.jsonl"))
            for cand in rep["candidates"]:
                if len(cand["completed_seeds"]) < 2:
                    self.assertFalse(cand["complete"])
                    self.assertNotEqual(rep["decision"]["selected_config_id"], cand["config_id"])

    def test_c07_adaptive_full_v2_heteroscedastic_gp_multi_seed_loocv(self):
        cfgs = [_det_cfg(lr0=lr) for lr in (0.001, 0.0003, 0.003, 0.0001)]
        prot = make_det_protocol(cfgs, method="adaptive_full", acquisition_version=2, training_seeds=[17, 29], budget_seconds=120.0)
        with tempfile.TemporaryDirectory() as td:
            ledger = Ledger(Path(td) / "events.jsonl")
            clock = [0.0]

            def eval_fn(idx, seed, epochs, full, seconds):
                clock[0] += 3.0
                delta = 0.08 if seed == 29 else 0.0
                return _outcome(f"a{idx}_{seed}", idx, seed, epochs, 0.6 + 0.05 * idx + delta, 3.0, 2.0, 450.0)

            state = sched_mod.run_search(prot, eval_fn, lambda: 120.0 - clock[0], ledger, elapsed=lambda: clock[0])
            hetero_steps = [
                s["surrogate_diagnostics"] for s in state["acquisition_history"]
                if isinstance(s.get("surrogate_diagnostics"), dict) and s["surrogate_diagnostics"].get("heteroscedastic")
            ]
            self.assertTrue(hetero_steps)

    def test_c08_controlled_space_with_pt_freeze5_and_trainable_params(self):
        hw = runtime_info("cpu")
        pt = str(REPO_ROOT / "weights" / "yolo11n.pt")
        with tempfile.TemporaryDirectory() as td:
            grid = Path(td) / "grid.json"
            write_json(grid, [
                {"model": pt, "imgsz": 64, "batch": 2, "accumulation": 1, "freeze": 0, "cos_lr": False, "warmup_epochs": 0.0, "mosaic": 0.0, "close_mosaic": 0},
                {"model": pt, "imgsz": 64, "batch": 2, "accumulation": 1, "freeze": 5, "cos_lr": True, "warmup_epochs": 0.5, "mosaic": 0.5, "close_mosaic": 1},
            ])
            cfgs = space_mod.build_space("detection", hw, Path(td) / "work", grid=str(grid), precision="fp32")
            descs = describe_models("detection", cfgs, 2)
            prot = {"task": "detection", "model_descriptors": descs}
            t_params = sorted(trainable_parameters_for(prot, cfg)[1] for cfg in cfgs)
            self.assertEqual(len(t_params), 2)
            self.assertLess(t_params[0], t_params[1])

    def test_c09_controlled_space_evaluate_v2_and_paper_export_vi(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            yaml_path = make_mini_yolo_dataset(root / "ds")
            grid = root / "grid.json"
            write_json(grid, [{
                "imgsz": 64, "batch": 2, "accumulation": 1, "lr0": 0.001, "freeze": 0,
                "cos_lr": True, "warmup_epochs": 0.5, "mosaic": 0.5, "close_mosaic": 0,
            }])
            opt = run_cli([
                "optimize", "--data", str(yaml_path), "--device", "cpu",
                "--models", str(REPO_ROOT / "weights" / "yolo11n.pt"),
                "--grid", str(grid), "--full-epochs", "1", "--seeds", "17",
                "--budget-seconds", "90", "--output", str(root / "runs"),
            ], timeout=180)
            self.assertEqual(opt.returncode, 0, msg=opt.stderr + opt.stdout)
            run_dir = [p for p in (root / "runs").iterdir() if p.is_dir() and (p / "protocol.json").is_file()][0]
            # Verify controlled hparams reached worker request & Ultralytics args.yaml
            req_files = sorted((run_dir / "trials").rglob("request.json"))
            self.assertTrue(req_files)
            req_cfg = json.loads(req_files[0].read_text(encoding="utf-8"))["config"]
            self.assertIs(req_cfg["cos_lr"], True)
            self.assertEqual(req_cfg["warmup_epochs"], 0.5)
            self.assertEqual(req_cfg["mosaic"], 0.5)
            self.assertEqual(req_cfg["close_mosaic"], 0)
            args_yamls = sorted((run_dir / "trials").rglob("args.yaml"))
            self.assertTrue(args_yamls, "Expected Ultralytics args.yaml inside trial outputs")
            args_txt = args_yamls[0].read_text(encoding="utf-8")
            self.assertIn("cos_lr: true", args_txt.lower())
            self.assertIn("warmup_epochs: 0.5", args_txt)
            self.assertIn("mosaic: 0.5", args_txt)
            ev = run_cli(["evaluate-v2", "--run", str(run_dir), "--seed-scope", "all"], timeout=180)
            self.assertEqual(ev.returncode, 0, msg=ev.stderr + ev.stdout)
            pe = run_cli(["paper-export", "--run", str(run_dir), "--output", str(root / "paper_vi"), "--lang", "vi"])
            self.assertEqual(pe.returncode, 0, msg=pe.stderr + pe.stdout)
            self.assertTrue((root / "paper_vi" / "pr_curve.svg").is_file())
            self.assertTrue((root / "paper_vi" / "confusion_matrix.svg").is_file())
            self.assertTrue((root / "paper_vi" / "PROVENANCE.json").is_file())

    def test_c10_controlled_space_evaluate_v2_and_paper_export_en(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            yaml_path = make_mini_yolo_dataset(root / "ds")
            grid = root / "grid.json"
            write_json(grid, [{"imgsz": 64, "batch": 2, "accumulation": 1, "lr0": 0.001, "freeze": 0}])
            opt = run_cli([
                "optimize", "--data", str(yaml_path), "--device", "cpu",
                "--models", str(REPO_ROOT / "weights" / "yolo11n.pt"),
                "--grid", str(grid), "--full-epochs", "1", "--seeds", "17",
                "--budget-seconds", "90", "--output", str(root / "runs"),
            ], timeout=180)
            self.assertEqual(opt.returncode, 0, msg=opt.stderr)
            run_dir = [p for p in (root / "runs").iterdir() if p.is_dir() and (p / "protocol.json").is_file()][0]
            self.assertEqual(run_cli(["evaluate-v2", "--run", str(run_dir), "--seed-scope", "all"], timeout=180).returncode, 0)
            pe = run_cli(["paper-export", "--run", str(run_dir), "--output", str(root / "paper_en"), "--lang", "en"])
            self.assertEqual(pe.returncode, 0, msg=pe.stderr + pe.stdout)
            tex = (root / "paper_en" / "tables.tex").read_text(encoding="utf-8")
            self.assertIn(r"\toprule", tex)
            self.assertTrue((root / "paper_en" / "PROVENANCE.json").is_file())

    def test_c11_benchmark_cli_to_anytime_trajectory_and_auarc(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cls_dir = make_mini_cls_dataset(root / "cls")
            out = root / "bench"
            bm = run_cli([
                "benchmark", "--data", str(cls_dir), "--task", "classification",
                "--device", "cpu", "--space", "compact", "--full-epochs", "1",
                "--seeds", "17", "--search-seeds", "17",
                "--methods", "adaptive_full", "adaptive_ehvi", "random", "default",
                "--budget-seconds", "30", "--reference-budget-seconds", "60", "--output", str(out),
            ], timeout=240)
            self.assertEqual(bm.returncode, 0, msg=bm.stderr + bm.stdout)
            study_dir = [p for p in out.iterdir() if p.is_dir() and (p / "benchmark_protocol.json").is_file()][0]
            bm_json = json.loads((study_dir / "BENCHMARK.json").read_text(encoding="utf-8"))
            self.assertTrue(bm_json.get("outcomes"))
            ref_dir = bm_json.get("reference")
            feasible_checked = 0
            for outc in bm_json["outcomes"]:
                run_p = Path(outc["run"])
                prot = json.loads((run_p / "protocol.json").read_text(encoding="utf-8"))
                traj = diag_mod.anytime_trajectory(run_p, prot, reference=ref_dir)
                self.assertIn("trajectory", traj)
                self.assertIn("auarc", traj)
                self.assertIn("time_to_target_95", traj)
                if outc.get("feasible"):
                    self.assertGreaterEqual(len(traj["trajectory"]), 1)
                    self.assertIsNotNone(traj["auarc"])
                    self.assertTrue(math.isfinite(float(traj["auarc"])))
                    feasible_checked += 1
            self.assertGreaterEqual(feasible_checked, 1)

    def test_c12_algorithm_report_n5_to_paper_export_svgs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, manifest = make_synthetic_n5_study_dir(root, seeds=(11, 17, 23, 29, 37))
            alg_out = root / "alg_report"
            self.assertEqual(run_cli(["algorithm-report", "--manifest", str(manifest), "--output", str(alg_out)]).returncode, 0)
            pe = run_cli(["paper-export", "--run", str(alg_out), "--output", str(root / "alg_paper"), "--lang", "en"])
            self.assertEqual(pe.returncode, 0, msg=pe.stderr + pe.stdout)
            self.assertTrue((root / "alg_paper" / "benchmark_regret.svg").is_file())
            self.assertTrue((root / "alg_paper" / "PROVENANCE.json").is_file())

    def test_c13_optimize_resume_audit_and_paper_export(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cls_dir = make_mini_cls_dataset(root / "cls")
            opt = run_cli([
                "optimize", "--data", str(cls_dir), "--task", "classification",
                "--device", "cpu", "--space", "compact", "--full-epochs", "1",
                "--seeds", "17", "--budget-seconds", "25", "--output", str(root / "runs"),
            ], timeout=120)
            self.assertEqual(opt.returncode, 0, msg=opt.stderr)
            run_dir = [p for p in (root / "runs").iterdir() if p.is_dir() and (p / "protocol.json").is_file()][0]
            self.assertEqual(run_cli(["resume", "--run", str(run_dir), "--additional-budget-seconds", "15"], timeout=120).returncode, 0)
            self.assertEqual(run_cli(["audit", "--run", str(run_dir)]).returncode, 0)
            self.assertEqual(run_cli(["paper-export", "--run", str(run_dir), "--output", str(root / "paper"), "--lang", "en"]).returncode, 0)

    def test_c14_optimize_evaluate_v2_and_web_ui_parity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cls_dir = make_mini_cls_dataset(root / "cls")
            runs_root = root / "runs"
            opt = run_cli([
                "optimize", "--data", str(cls_dir), "--task", "classification",
                "--device", "cpu", "--space", "compact", "--full-epochs", "1",
                "--seeds", "17", "--budget-seconds", "60", "--output", str(runs_root),
            ], timeout=120)
            self.assertEqual(opt.returncode, 0, msg=opt.stderr)
            run_dir = [p for p in runs_root.iterdir() if p.is_dir() and (p / "protocol.json").is_file()][0]
            self.assertEqual(run_cli(["evaluate-v2", "--run", str(run_dir), "--seed-scope", "all"], timeout=120).returncode, 0)
            raw_rep = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            proc, base_url = launch_ui_server(runs_root, port=0)
            try:
                surf = discover_ui_surface(base_url)
                found = False
                for route in surf["api_routes"] + ["/api/runs"]:
                    if "{" in route or "<" in route:
                        continue
                    code, _, body = http_get(base_url + route)
                    if code == 200 and raw_rep["protocol_sha256"] in body:
                        found = True
                        break
                self.assertTrue(found, f"protocol_sha256 not found in discovered API routes: {surf['api_routes']}")
            finally:
                stop_ui_server(proc)

    def test_c15_web_ui_csrf_post_audit_and_paper_export_keep_evidence_immutable(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = Path(td) / "runs"
            make_synthetic_n5_study_dir(runs_root, seeds=(11, 17, 23, 29, 37))
            run_dir = runs_root / "study_n5" / "adaptive_full" / "repeat_11"
            before = {p.relative_to(run_dir).as_posix(): sha256_file(p) for p in sorted(run_dir.rglob("*")) if p.is_file()}
            proc, base_url = launch_ui_server(runs_root, port=0)
            try:
                surf = discover_ui_surface(base_url)
                token = surf["csrf_token"]
                self.assertIsNotNone(token)
                audit_route = next((r for r in surf["api_routes"] if "audit" in r.lower()), "/api/audit")
                code, _, body = http_post(
                    base_url + audit_route,
                    data=json.dumps({"run": str(run_dir.relative_to(runs_root))}).encode("utf-8"),
                    headers={"X-CSRF-Token": token},
                )
                self.assertEqual(code, 200, msg=body)
            finally:
                stop_ui_server(proc)
            after = {p.relative_to(run_dir).as_posix(): sha256_file(p) for p in sorted(run_dir.rglob("*")) if p.is_file()}
            self.assertEqual(before, after)

    def test_c16_training_signature_exclusion_preserves_predictor_card_compat(self):
        sources = source_map()
        prot = {
            "source_files": dict(sources),
            "execution_policy": {"version": 2, "data_loader": {"workers": 0, "cache_mb": 0.0, "pin_memory": "off"}},
        }
        sig_before = pred_mod.training_signature(prot)
        prot_ui_edited = dict(prot)
        prot_ui_edited["source_files"] = {
            **sources,
            "research_pipeline/v2/web_ui/server.py": "f" * 64,
            "research_pipeline/v2/web_ui/routes.py": "f" * 64,
            "research_pipeline/v2/paper_export.py": "f" * 64,
            "research_pipeline/v2/wizard.py": "f" * 64,
        }
        self.assertEqual(sig_before, pred_mod.training_signature(prot_ui_edited))

    def test_c17_inspect_audit_json_plan_search_topsis_and_status_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            yaml_path = make_mini_yolo_dataset(root / "ds")
            audit_json = root / "audit.json"
            self.assertEqual(run_cli(["inspect", "--data", str(yaml_path), "--device", "cpu", "--audit-json", str(audit_json)]).returncode, 0)
            self.assertTrue(audit_json.is_file())
            ps = run_cli([
                "plan-search", "--data", str(yaml_path), "--device", "cpu",
                "--space", "compact", "--method", "adaptive_full", "--selection-rule", "topsis",
                "--full-epochs", "1", "--seeds", "17", "--budget-seconds", "30",
                "--output", str(root / "planned"),
            ])
            self.assertEqual(ps.returncode, 0, msg=ps.stderr)
            run_dir = [p for p in (root / "planned").iterdir() if p.is_dir() and (p / "protocol.json").is_file()][0]
            st = run_cli(["status", "--run", str(run_dir), "--json"])
            self.assertEqual(st.returncode, 0)
            self.assertEqual(json.loads(st.stdout)["saved_status"]["phase"], "prepared")

    def test_c18_no_color_cli_chain_and_yolo_tuner_invariance(self):
        for args in (["--no-color", "doctor", "--device", "cpu"], ["--no-color", "runs", "--root", "runs", "--limit", "3"]):
            proc = run_cli(args)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            assert_zero_ansi(proc.stdout)
            assert_zero_ansi(proc.stderr)
        for rel, expected in YOLO_TUNER_SHA256.items():
            self.assertEqual(sha256_file(REPO_ROOT / rel), expected)

    def test_c19_cross_dataset_and_outcome_sensitivity_no_hardcoded_winner(self):
        """Changing dataset descriptors & empirical outcomes must change predictor, surrogate, search, and selection."""
        cfgs = [
            _det_cfg(imgsz=64, batch=2, lr0=0.0003),
            _det_cfg(imgsz=160, batch=4, lr0=0.001),
            _det_cfg(imgsz=320, batch=8, lr0=0.003),
            _det_cfg(imgsz=160, batch=4, lr0=0.005),
        ]
        descs = describe_models("detection", cfgs, 2)
        desc_a = {
            "class_entropy": 0.98,
            "objects_per_image": 1.2,
            "small_box_fraction_area_lt_1pct_not_coco": 0.05,
            "pixel_mean": 0.48,
            "pixel_sd": 0.22,
            "edge_mean": 0.14,
            "sample_count": 16,
            "split": "train_only",
        }
        desc_b = {
            "class_entropy": 0.41,
            "objects_per_image": 6.4,
            "small_box_fraction_area_lt_1pct_not_coco": 0.75,
            "pixel_mean": 0.21,
            "pixel_sd": 0.09,
            "edge_mean": 0.39,
            "sample_count": 16,
            "split": "train_only",
        }
        p_a = make_det_protocol(
            cfgs, method="adaptive_full", acquisition_version=2, training_seeds=[17],
            model_descriptors=descs, dataset_descriptor=desc_a,
        )
        p_b = make_det_protocol(
            cfgs, method="adaptive_full", acquisition_version=2, training_seeds=[17],
            model_descriptors=descs, dataset_descriptor=desc_b,
        )
        self.assertNotEqual(p_a["dataset_descriptor"], p_b["dataset_descriptor"])
        self.assertNotEqual(pred_mod.feature_row(p_a, cfgs[0]), pred_mod.feature_row(p_b, cfgs[0]))

        # Case 1: Dataset/outcomes favor cfg 0 (small imgsz, high quality, fast)
        res_a = {
            "0": [_outcome("a0", 0, 17, 2, 0.91, 4.0, 2.0, 350.0)],
            "1": [_outcome("a1", 1, 17, 2, 0.62, 8.0, 5.0, 500.0)],
        }
        # Case 2: Distinct dataset/outcomes favor cfg 1 (larger imgsz, highest quality)
        res_b = {
            "0": [_outcome("b0", 0, 17, 2, 0.48, 4.0, 2.0, 350.0)],
            "1": [_outcome("b1", 1, 17, 2, 0.89, 8.0, 5.0, 500.0)],
        }
        rows_a = schema_mod.measurements(cfgs, res_a, [17], 2, Objective(**p_a["objective"]))
        rows_b = schema_mod.measurements(cfgs, res_b, [17], 2, Objective(**p_b["objective"]))
        self.assertNotEqual(sel_mod.select(rows_a, p_a)["selected_index"], sel_mod.select(rows_b, p_b)["selected_index"])

        cm_a = pred_mod.CostModel(p_a)
        cm_a.update(0, res_a["0"][0])
        cm_b = pred_mod.CostModel(p_b)
        cm_b.update(1, res_b["1"][0])
        fc_a = {i: cm_a.estimate(i) for i in (2, 3)}
        fc_b = {i: cm_b.estimate(i) for i in (2, 3)}
        self.assertNotEqual(fc_a[2]["predicted_memory_mb"], fc_b[2]["predicted_memory_mb"])
        self.assertNotEqual(fc_a[2]["point_seconds"], fc_b[2]["point_seconds"])

        out_a = acq_mod.rank_candidates(p_a, rows_a, fc_a)
        out_b = acq_mod.rank_candidates(p_b, rows_b, fc_b)
        self.assertNotEqual(
            out_a["surrogate_diagnostics"]["predictions"]["2"]["mean"],
            out_b["surrogate_diagnostics"]["predictions"]["2"]["mean"],
        )
        self.assertNotEqual(out_a["scores"], out_b["scores"])

        # Live run_search sensitivity across distinct dataset descriptors & outcome regimes
        with tempfile.TemporaryDirectory() as td:
            p_run_a = make_det_protocol(
                cfgs, method="adaptive_full", acquisition_version=2, training_seeds=[17],
                budget_seconds=300.0, max_memory=32768.0, model_descriptors=descs, dataset_descriptor=desc_a,
            )
            p_run_b = make_det_protocol(
                cfgs, method="adaptive_full", acquisition_version=2, training_seeds=[17],
                budget_seconds=300.0, max_memory=32768.0, model_descriptors=descs, dataset_descriptor=desc_b,
            )
            q_a = {0: 0.93, 1: 0.72, 2: 0.61, 3: 0.55}
            t_a = {0: 2.0, 1: 4.0, 2: 8.0, 3: 5.0}
            m_a = {0: 300.0, 1: 420.0, 2: 600.0, 3: 450.0}
            q_b = {0: 0.42, 1: 0.65, 2: 0.94, 3: 0.58}
            t_b = {0: 2.0, 1: 4.0, 2: 4.5, 3: 6.0}
            m_b = {0: 410.0, 1: 520.0, 2: 340.0, 3: 610.0}

            clk_a, clk_b = [0.0], [0.0]

            def eval_a(idx, seed, epochs, full, seconds):
                clk_a[0] += t_a[idx]
                return _outcome(f"ra_{idx}", idx, seed, epochs, q_a[idx], t_a[idx], t_a[idx] * 0.6, m_a[idx])

            def eval_b(idx, seed, epochs, full, seconds):
                clk_b[0] += t_b[idx]
                return _outcome(f"rb_{idx}", idx, seed, epochs, q_b[idx], t_b[idx], t_b[idx] * 0.6, m_b[idx])

            st_a = sched_mod.run_search(p_run_a, eval_a, lambda: 300.0 - clk_a[0], Ledger(Path(td) / "ev_a.jsonl"), elapsed=lambda: clk_a[0])
            st_b = sched_mod.run_search(p_run_b, eval_b, lambda: 300.0 - clk_b[0], Ledger(Path(td) / "ev_b.jsonl"), elapsed=lambda: clk_b[0])
            sel_a = sel_mod.select(schema_mod.measurements(cfgs, st_a["results"], [17], 2, Objective(**p_run_a["objective"])), p_run_a)
            sel_b = sel_mod.select(schema_mod.measurements(cfgs, st_b["results"], [17], 2, Objective(**p_run_b["objective"])), p_run_b)
            self.assertNotEqual(sel_a["selected_index"], sel_b["selected_index"])
            preds_a = [s["surrogate_diagnostics"]["predictions"] for s in st_a["acquisition_history"] if "surrogate_diagnostics" in s]
            preds_b = [s["surrogate_diagnostics"]["predictions"] for s in st_b["acquisition_history"] if "surrogate_diagnostics" in s]
            self.assertTrue(preds_a and preds_b)
            self.assertNotEqual(preds_a[-1], preds_b[-1])

        # Verify distinct real datasets produce distinct dataset_content_id & descriptors via plan-search
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ds1 = make_mini_yolo_dataset(root / "ds_small", n_train=4, n_val=2, n_test=2, size=64)
            ds2 = make_mini_yolo_dataset(root / "ds_large", n_train=8, n_val=4, n_test=4, size=96)
            for tag, ds_yaml in (("s", ds1), ("l", ds2)):
                ps = run_cli([
                    "plan-search", "--data", str(ds_yaml), "--device", "cpu",
                    "--space", "compact", "--full-epochs", "1", "--seeds", "17",
                    "--budget-seconds", "30", "--output", str(root / f"out_{tag}"),
                ])
                self.assertEqual(ps.returncode, 0, msg=ps.stderr)
            prot1 = json.loads(next((root / "out_s").rglob("protocol.json")).read_text(encoding="utf-8"))
            prot2 = json.loads(next((root / "out_l").rglob("protocol.json")).read_text(encoding="utf-8"))
            self.assertNotEqual(prot1["dataset_content_id"], prot2["dataset_content_id"])
            self.assertNotEqual(prot1["example_counts"], prot2["example_counts"])
            self.assertNotEqual(prot1["dataset_descriptor"], prot2["dataset_descriptor"])
