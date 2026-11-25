"""Tier 4 Real-World Application Scenarios: 11 complete CPU-safe E2E workflows."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import unittest

from PIL import Image, ImageDraw

from research_pipeline.v2 import algorithm_study as study_mod
from research_pipeline.v2 import diagnostics as diag_mod
from research_pipeline.v2.model_features import trainable_parameters_for
from research_pipeline.v2.report import make_report
from research_pipeline.v2.workflow import audit_run
from yolo_tuner.pipeline import write_json

from tests_e2e._e2e_fixtures import (
    ALGORITHM_010_EXPECTED_SHA256,
    REPO_ROOT,
    YOLO_TUNER_SHA256,
    assert_zero_ansi,
    audit_count,
    discover_ui_surface,
    extract_csrf_token,
    extract_external_urls,
    extract_nav_links,
    extract_stat_p_value,
    find_audit_dict,
    http_get,
    http_post,
    launch_ui_server,
    make_dirty_yolo_dataset,
    make_mini_cls_dataset,
    make_mini_yolo_dataset,
    make_synthetic_n5_study_dir,
    offline_yolo_val,
    run_cli,
    sha256_file,
    stop_ui_server,
)


class TestTier4RealWorldScenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls._root = Path(cls._tmp.name)
        cls._yolo_run_dir: Path | None = None
        cls._yolo_eval_json: dict | None = None
        cls._cls_bench_dir: Path | None = None

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @classmethod
    def _ensure_shared_yolo_run(cls) -> tuple[Path, dict]:
        if cls._yolo_run_dir is not None and cls._yolo_eval_json is not None:
            return cls._yolo_run_dir, cls._yolo_eval_json
        ws = cls._root / "shared_yolo"
        yaml_path = make_mini_yolo_dataset(ws / "ds", n_train=4, n_val=2, n_test=2, size=64)
        insp = run_cli(["inspect", "--data", str(yaml_path), "--device", "cpu", "--json"])
        if insp.returncode != 0:
            raise AssertionError(insp.stderr + insp.stdout)
        grid = ws / "grid.json"
        write_json(grid, [{"imgsz": 64, "batch": 2, "accumulation": 1, "lr0": 0.001, "freeze": 0}])
        runs_root = ws / "runs"
        opt = run_cli([
            "optimize",
            "--data", str(yaml_path),
            "--device", "cpu",
            "--models", str(REPO_ROOT / "weights" / "yolo11n.pt"),
            "--grid", str(grid),
            "--full-epochs", "1",
            "--seeds", "17",
            "--budget-seconds", "90",
            "--output", str(runs_root),
        ], timeout=180)
        if opt.returncode != 0:
            raise AssertionError(opt.stderr + opt.stdout)
        run_dir = [p for p in runs_root.iterdir() if p.is_dir() and (p / "protocol.json").is_file()][0]
        ev = run_cli(["evaluate-v2", "--run", str(run_dir), "--seed-scope", "all"], timeout=180)
        if ev.returncode != 0:
            raise AssertionError(ev.stderr + ev.stdout)
        eval_dirs = sorted(p for p in (run_dir / "evaluation").iterdir() if p.is_dir())
        eval_json = json.loads((eval_dirs[-1] / "evaluation.json").read_text(encoding="utf-8"))
        cls._yolo_run_dir = run_dir
        cls._yolo_eval_json = eval_json
        return run_dir, eval_json

    @classmethod
    def _ensure_shared_cls_benchmark(cls) -> Path:
        if cls._cls_bench_dir is not None:
            return cls._cls_bench_dir
        ws = cls._root / "shared_cls_bench"
        cls_dir = make_mini_cls_dataset(ws / "cls", n_train=8, n_val=4, n_test=4, size=16)
        bm = run_cli([
            "benchmark",
            "--data", str(cls_dir),
            "--task", "classification",
            "--device", "cpu",
            "--space", "compact",
            "--full-epochs", "1",
            "--seeds", "17",
            "--search-seeds", "17",
            "--methods", "adaptive_full", "adaptive_ehvi", "random", "default",
            "--budget-seconds", "30",
            "--reference-budget-seconds", "60",
            "--output", str(ws / "bench_out"),
        ], timeout=240)
        if bm.returncode != 0:
            raise AssertionError(bm.stderr + bm.stdout)
        study_dir = [p for p in (ws / "bench_out").iterdir() if p.is_dir() and (p / "benchmark_protocol.json").is_file()][0]
        cls._cls_bench_dir = study_dir
        return study_dir

    # S1: Complete Real YOLO Detection Lifecycle + <1e-12 rect=False parity
    def test_s01_complete_real_yolo_detection_lifecycle_and_val_parity(self):
        run_dir, ev_json = self._ensure_shared_yolo_run()
        st = run_cli(["status", "--run", str(run_dir), "--json"])
        self.assertEqual(st.returncode, 0)
        self.assertEqual(json.loads(st.stdout)["saved_status"]["phase"], "complete")

        aud = run_cli(["audit", "--run", str(run_dir)])
        self.assertEqual(aud.returncode, 0, msg=aud.stderr + aud.stdout)

        for key in ("latency", "pr_curve", "f1_curve", "confusion_matrix"):
            self.assertIn(key, ev_json)
        cm = ev_json["confusion_matrix"]
        self.assertIs(cm.get("available"), True)
        raw_mat = cm["sum_raw"] if "sum_raw" in cm else cm["raw"]
        self.assertGreater(sum(float(v) for row in raw_mat for v in row), 0.0)
        self.assertGreater(sum(float(v) for row in cm["normalized"] for v in row), 0.0)
        fps_raw = ev_json["latency"]["fps"]
        fps_val = fps_raw["mean"] if isinstance(fps_raw, dict) else fps_raw
        self.assertGreater(fps_val, 0.0)

        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        ckpt = state["results"]["0"][0]["best_checkpoint"]
        direct_map50, direct_map50_95 = offline_yolo_val(
            ckpt, run_dir / "full_data" / "data.yaml", imgsz=64, batch=2
        )
        self.assertLess(abs(ev_json["map50"] - direct_map50), 1e-12)
        self.assertLess(abs(ev_json["map50_95"] - direct_map50_95), 1e-12)

    # S2: Bilingual Scientific Publication Pipeline (en + vi)
    def test_s02_bilingual_scientific_publication_pipeline(self):
        run_dir, _ = self._ensure_shared_yolo_run()
        before = {p.relative_to(run_dir).as_posix(): sha256_file(p) for p in sorted(run_dir.rglob("*")) if p.is_file()}
        for lang in ("en", "vi"):
            out_dir = self._root / f"s02_paper_{lang}"
            proc = run_cli(["paper-export", "--run", str(run_dir), "--output", str(out_dir), "--lang", lang])
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
            for fname in (
                "tables.tex",
                "PAPER_SUMMARY.md",
                "PAPER_EVIDENCE.json",
                "PROVENANCE.json",
                "pareto_front.svg",
                "learning_curves.svg",
                "anytime_convergence.svg",
                "per_class_ap50.svg",
                "pr_curve.svg",
                "confusion_matrix.svg",
            ):
                self.assertTrue((out_dir / fname).is_file(), f"Missing {fname} for lang={lang}")
            self.assertIn(r"\toprule", (out_dir / "tables.tex").read_text(encoding="utf-8"))
            cm_svg = (out_dir / "confusion_matrix.svg").read_text(encoding="utf-8")
            cm_vals = [float(m) for m in re.findall(r"<text[^>]*>\s*([0-9]+\.[0-9]{2})\s*</text>", cm_svg)]
            self.assertTrue(any(v > 0.0 for v in cm_vals), f"confusion_matrix.svg ({lang}) has all-zero cells: {cm_vals}")
        after = {p.relative_to(run_dir).as_posix(): sha256_file(p) for p in sorted(run_dir.rglob("*")) if p.is_file()}
        self.assertEqual(before, after)

    # S3: Dirty Dataset Preflight Repair & Protocol-Locked Training
    def test_s03_dirty_dataset_preflight_repair_and_training(self):
        ws = self._root / "s03_repair"
        dirty_yaml = make_dirty_yolo_dataset(ws / "dirty")
        audit_json = ws / "data_audit.json"
        insp = run_cli(["inspect", "--data", str(dirty_yaml), "--device", "cpu", "--audit-json", str(audit_json), "--json"])
        self.assertEqual(insp.returncode, 0, msg=insp.stderr + insp.stdout)
        self.assertTrue(audit_json.is_file())
        saved_audit = find_audit_dict(json.loads(audit_json.read_text(encoding="utf-8")))
        self.assertGreater(audit_count(saved_audit, "cross_split_leakage_count", "cross_split_leakage", "leakage.cross_split_count", "leakage.cross_split"), 0)
        self.assertGreater(audit_count(saved_audit, "out_of_bounds_boxes_count", "out_of_bounds_boxes"), 0)

        dirty_clean_dir = ws / "dirty_dedup"
        dd_dirty = run_cli(["dedup-dataset", "--data", str(dirty_yaml), "--output", str(dirty_clean_dir)])
        self.assertEqual(dd_dirty.returncode, 0, msg=dd_dirty.stderr + dd_dirty.stdout)
        blocked_plan = run_cli([
            "plan-search", "--data", str(dirty_clean_dir / "data.yaml"), "--device", "cpu",
            "--models", str(REPO_ROOT / "weights" / "yolo11n.pt"),
            "--space", "compact", "--full-epochs", "1", "--seeds", "17",
            "--budget-seconds", "60", "--output", str(ws / "blocked_planned"),
        ])
        self.assertEqual(blocked_plan.returncode, 2)

        leaked_yaml = make_dirty_yolo_dataset(ws / "leaked", include_bad_boxes=False)
        clean_dir = ws / "clean"
        dd = run_cli(["dedup-dataset", "--data", str(leaked_yaml), "--output", str(clean_dir)])
        self.assertEqual(dd.returncode, 0, msg=dd.stderr + dd.stdout)

        clean_yaml = clean_dir / "data.yaml"
        plan = run_cli([
            "plan-search", "--data", str(clean_yaml), "--device", "cpu",
            "--models", str(REPO_ROOT / "weights" / "yolo11n.pt"),
            "--space", "compact", "--full-epochs", "1", "--seeds", "17",
            "--budget-seconds", "60", "--output", str(ws / "planned"),
        ])
        self.assertEqual(plan.returncode, 0, msg=plan.stderr + plan.stdout)

    # S4: Controlled Research Space & .pt Freeze Transfer (freeze=0 vs freeze=5)
    def test_s04_controlled_research_space_and_pt_freeze_transfer(self):
        ws = self._root / "s04_controlled"
        yaml_path = make_mini_yolo_dataset(ws / "ds", n_train=4, n_val=2, n_test=2, size=64)
        grid = ws / "grid.json"
        write_json(grid, [
            {
                "imgsz": 64, "batch": 2, "accumulation": 1, "lr0": 0.001, "freeze": 0,
                "cos_lr": False, "warmup_epochs": 0.0, "mosaic": 0.0, "close_mosaic": 0,
            },
            {
                "imgsz": 64, "batch": 2, "accumulation": 1, "lr0": 0.001, "freeze": 5,
                "cos_lr": True, "warmup_epochs": 0.5, "mosaic": 0.5, "close_mosaic": 0,
            },
        ])
        opt = run_cli([
            "optimize", "--data", str(yaml_path), "--device", "cpu",
            "--models", str(REPO_ROOT / "weights" / "yolo11n.pt"),
            "--grid", str(grid), "--full-epochs", "1", "--seeds", "17",
            "--budget-seconds", "120", "--output", str(ws / "runs"),
        ], timeout=240)
        self.assertEqual(opt.returncode, 0, msg=opt.stderr + opt.stdout)
        run_dir = [p for p in (ws / "runs").iterdir() if p.is_dir() and (p / "protocol.json").is_file()][0]
        _, prot, _, _ = audit_run(run_dir)
        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        obs_runs = [r for runs in (state.get("results") or {}).values() for r in runs if isinstance(r, dict)]
        t_params = [trainable_parameters_for(prot, cfg, observed_runs=obs_runs)[1] for cfg in prot["configs"]]
        self.assertTrue(all(isinstance(x, int) and x > 0 for x in t_params))
        self.assertEqual(len(set(t_params)), 2)
        self.assertLess(min(t_params), max(t_params))

    # S5: Deployment Re-Training (train-v2) & Budget Resume Integrity
    def test_s05_deployment_train_v2_and_budget_resume_workflow(self):
        run_dir, _ = self._ensure_shared_yolo_run()
        deploy_out = self._root / "s05_deploy"
        tr = run_cli([
            "train-v2",
            "--selection", str(run_dir / "selected.json"),
            "--device", "cpu",
            "--epochs", "1",
            "--budget-seconds", "90",
            "--output", str(deploy_out),
            "--preset", "Balanced",
        ], timeout=180)
        self.assertEqual(tr.returncode, 0, msg=tr.stderr + tr.stdout)
        train_json = next(deploy_out.rglob("TRAINING.json"))
        payload = json.loads(train_json.read_text(encoding="utf-8"))
        self.assertTrue(payload["execution_verified"])
        self.assertEqual(payload["status"], "execution_verified_optimality_not_reestablished")

        cls_dir = make_mini_cls_dataset(self._root / "s05_cls")
        cls_out = self._root / "s05_cls_runs"
        opt = run_cli([
            "optimize", "--data", str(cls_dir), "--task", "classification",
            "--device", "cpu", "--space", "compact", "--full-epochs", "1",
            "--seeds", "17", "--budget-seconds", "25", "--output", str(cls_out),
        ], timeout=120)
        self.assertEqual(opt.returncode, 0, msg=opt.stderr)
        cls_run = [p for p in cls_out.iterdir() if p.is_dir() and (p / "protocol.json").is_file()][0]
        res = run_cli(["resume", "--run", str(cls_run), "--additional-budget-seconds", "15"], timeout=120)
        self.assertEqual(res.returncode, 0, msg=res.stderr + res.stdout)
        audit_run(cls_run)

    # S6: Multi-Method Budget Benchmark & Anytime Convergence
    def test_s06_multi_method_benchmark_and_anytime_convergence(self):
        study_dir = self._ensure_shared_cls_benchmark()
        rep = run_cli(["benchmark-report", "--run", str(study_dir)], timeout=120)
        self.assertEqual(rep.returncode, 0, msg=rep.stderr + rep.stdout)
        bm = json.loads((study_dir / "BENCHMARK.json").read_text(encoding="utf-8"))
        methods = {o["method"] for o in bm["outcomes"]}
        self.assertEqual(methods, {"adaptive_full", "adaptive_ehvi", "random", "default"})
        feasible_count = 0
        for outc in bm["outcomes"]:
            run_p = Path(outc["run"])
            prot = json.loads((run_p / "protocol.json").read_text(encoding="utf-8"))
            traj = diag_mod.anytime_trajectory(run_p, prot, reference=bm["reference"])
            self.assertIn("trajectory", traj)
            self.assertIn("auarc", traj)
            self.assertIn("time_to_target_95", traj)
            if outc["feasible"]:
                self.assertGreaterEqual(len(traj["trajectory"]), 1)
                self.assertIsNotNone(traj["auarc"])
                self.assertTrue(math.isfinite(float(traj["auarc"])))
                feasible_count += 1
            else:
                self.assertIsNone(outc["objective_regret"])
        self.assertGreaterEqual(feasible_count, 1)

    # S7: Conditional Paired Non-Parametric Statistics (n >= 5) & 0.10.0 Parity
    def test_s07_conditional_paired_statistics_n5_and_legacy_010_parity(self):
        rep_010 = study_mod.analyze_studies(REPO_ROOT / "configs" / "algorithm_010_studies.json")
        self.assertEqual(rep_010["sha256"], ALGORITHM_010_EXPECTED_SHA256)

        _, manifest = make_synthetic_n5_study_dir(self._root / "s07_n5", seeds=(11, 17, 23, 29, 37))
        alg_out = self._root / "s07_alg_out"
        proc = run_cli(["algorithm-report", "--manifest", str(manifest), "--output", str(alg_out)])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
        data = json.loads((alg_out / "ALGORITHM_STUDY.json").read_text(encoding="utf-8"))
        self.assertFalse(data["statistical_superiority_established"])
        self.assertIsNone(data["summary"]["cross_domain_confidence_interval"])
        self.assertIn("not a confidence interval", data["summary"]["independence_warning"].lower())
        pair = data["studies"][0]["paired_comparisons"][0]
        self.assertTrue(0.0 <= extract_stat_p_value(pair, "objective_regret", "wilcoxon") <= 1.0)
        self.assertTrue(0.0 <= extract_stat_p_value(pair, "objective_regret", "sign_flip") <= 1.0)
        self.assertTrue(0.0 <= extract_stat_p_value(pair, "objective_regret", "holm") <= 1.0)

    # S8: Live Local Web Dashboard Studio (6 Views, Offline, Dark/Light, Parity)
    def test_s08_live_web_ui_6_views_offline_and_numeric_parity(self):
        run_dir, eval_json = self._ensure_shared_yolo_run()
        runs_root = run_dir.parent
        raw_rep = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        expected_u = float(raw_rep["recommendation"]["utility"])
        expected_map50 = float(eval_json["map50"])
        expected_map50_95 = float(eval_json["map50_95"])
        fps_raw = eval_json["latency"]["fps"]
        total_ms_raw = eval_json["latency"]["total_ms_per_image"]
        expected_fps = float(fps_raw["mean"] if isinstance(fps_raw, dict) else fps_raw)
        expected_total_ms = float(total_ms_raw["mean"] if isinstance(total_ms_raw, dict) else total_ms_raw)
        expected_per_class = eval_json.get("per_class") or []
        self.assertTrue(expected_per_class, "evaluation.json missing per_class entries")

        proc, base_url = launch_ui_server(runs_root, port=0, entry="script")
        try:
            surf = discover_ui_surface(base_url)
            self.assertEqual(extract_external_urls(surf["root_html"]), [])
            distinct_views = {"/"} | set(surf["nav_links"])
            self.assertGreaterEqual(len(distinct_views), 6, f"Expected >= 6 views, found {sorted(distinct_views)}")
            corpus_parts = [surf["corpus"]]
            for link in sorted(distinct_views):
                c, _, page_html = http_get(base_url + link)
                self.assertEqual(c, 200, f"{link} -> {c}")
                self.assertEqual(extract_external_urls(page_html), [])
                corpus_parts.append(page_html)
            full_corpus = "\n".join(corpus_parts)

            # Interactive Dark/Light theme toggle & mandatory post-hoc sensitivity disclaimer
            has_toggle_el = bool(re.search(
                r"<(?:button|input|select)\b[^>]*(?:theme|dark|light)[^>]*>|<button\b[^>]*>[^<]*(?:theme|dark|light|sáng|tối)[^<]*</button>",
                full_corpus,
                flags=re.IGNORECASE,
            ))
            has_theme_js = bool(re.search(
                r"(?:data-theme|dataset\.theme|classList\.(?:toggle|add|remove)|localStorage\.(?:getItem|setItem))\s*(?:\([^)]*(?:theme|dark|light)|=)",
                full_corpus,
                flags=re.IGNORECASE,
            ))
            has_theme_css = bool(re.search(
                r"(?::root\s*\{[^}]*--|\[\s*data-theme\s*=|\.(?:dark|light)(?:-theme|-mode)?\s*\{)",
                full_corpus,
                flags=re.IGNORECASE,
            ))
            self.assertTrue(
                has_toggle_el and has_theme_js and has_theme_css,
                f"Missing interactive Dark/Light theme toggle (control={has_toggle_el}, js={has_theme_js}, css={has_theme_css})",
            )
            low = full_corpus.lower()
            self.assertIn("phân tích độ nhạy hậu nghiệm", low)
            self.assertIn("không làm thay đổi quyết định nghiên cứu đã khóa", low)

            # Live HTTP JSON numeric parity (< 1e-12) against report.json & evaluation.json (no HTML rounding fallback)
            ui_json_blobs = []
            run_candidates = [
                f"/api/run?path={run_dir.name}",
                f"/api/run?run={run_dir.name}",
                f"/api/runs/{run_dir.name}",
                f"/api/evaluation?path={run_dir.name}",
                f"/api/evaluation?run={run_dir.name}",
            ]
            for route in surf["api_routes"]:
                if "{" in route or "<" in route:
                    sub = re.sub(r"\{[^}]+\}|<[^>]+>", run_dir.name, route)
                    run_candidates.append(sub)
                else:
                    run_candidates.append(route)
                    if any(k in route.lower() for k in ("run", "eval", "detail", "report")):
                        run_candidates.extend([f"{route}?path={run_dir.name}", f"{route}?run={run_dir.name}"])
            for cand in dict.fromkeys(run_candidates):
                c_api, _, api_body = http_get(base_url + cand)
                if c_api == 200 and api_body.strip().startswith(("{", "[")):
                    try:
                        ui_json_blobs.append(json.loads(api_body))
                    except ValueError:
                        pass
            self.assertTrue(ui_json_blobs, "No live JSON API responses returned HTTP 200")

            def _walk_dicts(obj):
                if isinstance(obj, dict):
                    yield obj
                    for v in obj.values():
                        yield from _walk_dicts(v)
                elif isinstance(obj, list):
                    for item in obj:
                        yield from _walk_dicts(item)

            all_dicts = list(_walk_dicts(ui_json_blobs))

            def _is_num(val) -> bool:
                return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))

            # 1) Selected candidate utility parity
            rec_utils = [float(d["recommendation"]["utility"]) for d in all_dicts if isinstance(d.get("recommendation"), dict) and _is_num(d["recommendation"].get("utility"))]
            for u in rec_utils:
                self.assertLess(abs(u - expected_u), 1e-12, f"UI recommendation.utility={u} != report.json utility={expected_u}")
            keyed_utils = [float(d["utility"]) for d in all_dicts if _is_num(d.get("utility"))]
            self.assertTrue(
                any(abs(u - expected_u) < 1e-12 for u in keyed_utils),
                f"Live UI JSON did not match report.json utility={expected_u}; found utility={keyed_utils}",
            )

            # 2) Held-out evaluation map50 & map50_95 parity
            eval_dicts = [
                d["evaluation"] for d in all_dicts if isinstance(d.get("evaluation"), dict)
            ] + [
                d for d in all_dicts
                if _is_num(d.get("map50")) and (isinstance(d.get("latency"), dict) or "per_class" in d or "test_used_for_selection" in d)
            ]
            for ed in eval_dicts:
                if _is_num(ed.get("map50")):
                    self.assertLess(abs(float(ed["map50"]) - expected_map50), 1e-12, f"UI eval map50={ed['map50']} != {expected_map50}")
                if _is_num(ed.get("map50_95")):
                    self.assertLess(abs(float(ed["map50_95"]) - expected_map50_95), 1e-12, f"UI eval map50_95={ed['map50_95']} != {expected_map50_95}")
            keyed_map50 = [float(d[k]) for d in all_dicts for k in ("map50", "test_map50") if _is_num(d.get(k))]
            keyed_map50_95 = [float(d[k]) for d in all_dicts for k in ("map50_95", "test_map50_95") if _is_num(d.get(k))]
            self.assertTrue(
                any(abs(v - expected_map50) < 1e-12 for v in keyed_map50),
                f"Live UI JSON did not match evaluation.json map50={expected_map50}; found={keyed_map50}",
            )
            self.assertTrue(
                any(abs(v - expected_map50_95) < 1e-12 for v in keyed_map50_95),
                f"Live UI JSON did not match evaluation.json map50_95={expected_map50_95}; found={keyed_map50_95}",
            )

            # 3) Latency (fps & total_ms_per_image) parity
            lat_dicts = [d["latency"] for d in all_dicts if isinstance(d.get("latency"), dict)]
            for ld in lat_dicts:
                if _is_num(ld.get("fps")):
                    self.assertLess(abs(float(ld["fps"]) - expected_fps), 1e-12, f"UI latency.fps={ld['fps']} != {expected_fps}")
                if _is_num(ld.get("total_ms_per_image")):
                    self.assertLess(abs(float(ld["total_ms_per_image"]) - expected_total_ms), 1e-12, f"UI latency.total_ms_per_image={ld['total_ms_per_image']} != {expected_total_ms}")
            keyed_fps = [float(d["fps"]) for d in all_dicts if _is_num(d.get("fps"))]
            keyed_total_ms = [float(d["total_ms_per_image"]) for d in all_dicts if _is_num(d.get("total_ms_per_image"))]
            self.assertTrue(
                any(abs(v - expected_fps) < 1e-12 for v in keyed_fps),
                f"Live UI JSON did not match evaluation.json latency.fps={expected_fps}; found={keyed_fps}",
            )
            self.assertTrue(
                any(abs(v - expected_total_ms) < 1e-12 for v in keyed_total_ms),
                f"Live UI JSON did not match evaluation.json latency.total_ms_per_image={expected_total_ms}; found={keyed_total_ms}",
            )

            # 4) Per-class AP parity
            per_class_containers = [
                d[k] for d in all_dicts for k in ("per_class", "per_class_summary")
                if d.get(k) is not None and (isinstance(d[k], (list, dict)) and len(d[k]) > 0)
            ]
            self.assertTrue(per_class_containers, "Live UI JSON did not include per_class / per_class_summary data")
            matched_all_classes = False
            for container in per_class_containers:
                if isinstance(container, dict):
                    # Dict keyed by class_name or str(class_id)
                    ok = True
                    for exp_c in expected_per_class:
                        cid_str = str(exp_c["class_id"])
                        cname = str(exp_c["class_name"])
                        entry = container.get(cname) if cname in container else container.get(cid_str)
                        if isinstance(entry, dict):
                            ap50_val = entry.get("ap50", entry.get("ap50_mean"))
                            ap95_val = entry.get("ap50_95", entry.get("ap50_95_mean"))
                            if not (_is_num(ap50_val) and abs(float(ap50_val) - float(exp_c["ap50"])) < 1e-12):
                                self.fail(f"UI per_class ap50 mismatch for {cname}: {entry} != {exp_c['ap50']}")
                            if _is_num(ap95_val) and abs(float(ap95_val) - float(exp_c["ap50_95"])) >= 1e-12:
                                self.fail(f"UI per_class ap50_95 mismatch for {cname}: {entry} != {exp_c['ap50_95']}")
                        elif _is_num(entry):
                            if abs(float(entry) - float(exp_c["ap50"])) >= 1e-12:
                                self.fail(f"UI per_class value mismatch for {cname}: {entry} != {exp_c['ap50']}")
                        else:
                            ok = False
                    if ok:
                        matched_all_classes = True
                elif isinstance(container, list):
                    ok = True
                    for exp_c in expected_per_class:
                        cid = int(exp_c["class_id"])
                        cname = str(exp_c["class_name"])
                        match = next(
                            (
                                item for item in container
                                if isinstance(item, dict)
                                and (item.get("class_id") == cid or str(item.get("class_name")) == cname)
                            ),
                            None,
                        )
                        if match is None:
                            ok = False
                            break
                        ap50_val = match.get("ap50", match.get("ap50_mean"))
                        ap95_val = match.get("ap50_95", match.get("ap50_95_mean"))
                        if not _is_num(ap50_val):
                            ok = False
                            break
                        self.assertLess(abs(float(ap50_val) - float(exp_c["ap50"])), 1e-12, f"UI per_class ap50 mismatch for {cname}")
                        if _is_num(ap95_val):
                            self.assertLess(abs(float(ap95_val) - float(exp_c["ap50_95"])), 1e-12, f"UI per_class ap50_95 mismatch for {cname}")
                    if ok:
                        matched_all_classes = True
            self.assertTrue(matched_all_classes, f"Live UI JSON per_class did not match expected classes: {per_class_containers}")

            # 5) Live Web UI /api/evaluation confusion_matrix parity (< 1e-12) and non-zero mass
            c_ev, _, ev_api_body = http_get(base_url + f"/api/evaluation?path={run_dir.name}")
            self.assertEqual(c_ev, 200)
            ev_api_doc = json.loads(ev_api_body)
            ui_cm = (ev_api_doc.get("evaluation") or {}).get("confusion_matrix") or {}
            self.assertIs(ui_cm.get("available"), True)
            expected_norm = eval_json["confusion_matrix"]["normalized"]
            ui_norm = ui_cm.get("normalized") or []
            self.assertEqual(len(ui_norm), len(expected_norm))
            self.assertGreater(sum(float(v) for row in ui_norm for v in row), 0.0)
            for r_ui, r_exp in zip(ui_norm, expected_norm):
                self.assertEqual(len(r_ui), len(r_exp))
                for v_ui, v_exp in zip(r_ui, r_exp):
                    self.assertLess(abs(float(v_ui) - float(v_exp)), 1e-12)
        finally:
            stop_ui_server(proc)

    # S9: Web UI Security, Audit & Paper Export Actions
    def test_s09_web_ui_security_and_audit_paper_export_actions(self):
        run_dir, _ = self._ensure_shared_yolo_run()
        runs_root = run_dir.parent
        secret_file = runs_root.parent / "outside_traversal_secret.txt"
        secret_file.write_text("TRAVERSAL_SECRET_TOKEN", encoding="utf-8")
        before = {p.relative_to(run_dir).as_posix(): sha256_file(p) for p in sorted(run_dir.rglob("*")) if p.is_file()}
        proc, base_url = launch_ui_server(runs_root, port=0)
        try:
            surf = discover_ui_surface(base_url)
            token = surf["csrf_token"]
            self.assertIsNotNone(token)
            post_candidates = list(dict.fromkeys(list(surf.get("api_routes", [])) + list(surf.get("post_routes", []))))
            audit_route = next((r for r in post_candidates if "audit" in r.lower()), "/api/audit")
            paper_route = next((r for r in post_candidates if "paper" in r.lower() or "export" in r.lower()), "/api/paper-export")

            # Reject missing CSRF
            c_no_csrf, _, _ = http_post(base_url + audit_route, data=b"{}")
            self.assertIn(c_no_csrf, (400, 403))

            # Prove in-bounds file/run detail access returns 200 before testing traversal (zero /api/runs vacuous fallback)
            in_bounds_probes = [
                ("/api/file?path=", f"{run_dir.name}/report.json", "/api/file?path={payload}"),
                ("/api/file?run=", f"{run_dir.name}&file=report.json", f"/api/file?run={run_dir.name}&file={{payload}}"),
                ("/api/artifact?path=", f"{run_dir.name}/report.json", "/api/artifact?path={payload}"),
                ("/api/artifact?run=", f"{run_dir.name}&file=report.json", f"/api/artifact?run={run_dir.name}&file={{payload}}"),
                ("/api/run?path=", run_dir.name, "/api/run?path={payload}"),
                ("/api/run?run=", run_dir.name, "/api/run?run={payload}"),
            ]
            for route in surf["api_routes"]:
                tail = route.rstrip("/").rsplit("/", 1)[-1].lower()
                if tail in {"runs", "benchmarks", "datasets", "weights", "health", "status", "hardware", "events", "logs"} or "{" in route or "<" in route:
                    continue
                if any(k in route.lower() for k in ("file", "artifact", "download", "read")):
                    in_bounds_probes.extend([
                        (f"{route}?path=", f"{run_dir.name}/report.json", f"{route}?path={{payload}}"),
                        (f"{route}?run=", f"{run_dir.name}&file=report.json", f"{route}?run={run_dir.name}&file={{payload}}"),
                    ])
                elif "run" in route.lower():
                    in_bounds_probes.extend([
                        (f"{route}?path=", run_dir.name, f"{route}?path={{payload}}"),
                        (f"{route}?run=", run_dir.name, f"{route}?run={{payload}}"),
                    ])
            proven_templates = []
            seen_tmpl = set()
            for pfx, suffix, tmpl in in_bounds_probes:
                if tmpl in seen_tmpl:
                    continue
                if http_get(base_url + pfx + suffix)[0] == 200:
                    seen_tmpl.add(tmpl)
                    proven_templates.append(tmpl)
            self.assertTrue(
                proven_templates,
                f"No in-bounds file/run detail endpoint returned HTTP 200; discovered routes={surf['api_routes']}",
            )
            traversal_payloads = (
                "../outside_traversal_secret.txt",
                "../../outside_traversal_secret.txt",
                "../../../outside_traversal_secret.txt",
                "../../../../outside_traversal_secret.txt",
                "%2e%2e%2foutside_traversal_secret.txt",
                "%2e%2e/%2e%2e/outside_traversal_secret.txt",
                "%2e%2e/%2e%2e/%2e%2e/outside_traversal_secret.txt",
                "%2e%2e/%2e%2e/%2e%2e/%2e%2e/outside_traversal_secret.txt",
                "%2e%2e%2fpyproject.toml",
                "../../pyproject.toml",
            )
            for tmpl in proven_templates:
                for payload in traversal_payloads:
                    c_trav, _, trav_body = http_get(base_url + tmpl.replace("{payload}", payload))
                    self.assertIn(c_trav, (400, 403, 404))
                    self.assertNotIn("TRAVERSAL_SECRET_TOKEN", trav_body)
                    self.assertNotIn("[project]", trav_body)

            # Accept valid CSRF audit
            c_aud, _, aud_body = http_post(
                base_url + audit_route,
                data=json.dumps({"run": run_dir.name}).encode("utf-8"),
                headers={"X-CSRF-Token": token},
            )
            self.assertEqual(c_aud, 200, msg=aud_body)

            # Reject paper-export output escaping runs_root, accept contained output inside runs_root
            esc_out = self._root / "s09_escaped_outside"
            c_pe_esc, _, _ = http_post(
                base_url + paper_route,
                data=json.dumps({"run": run_dir.name, "output": str(esc_out), "lang": "en"}).encode("utf-8"),
                headers={"X-CSRF-Token": token},
            )
            self.assertIn(c_pe_esc, (400, 403))
            self.assertFalse(esc_out.exists())

            paper_out = runs_root / "paper_exports" / "s09_ui_paper"
            c_pe, _, pe_body = http_post(
                base_url + paper_route,
                data=json.dumps({"run": run_dir.name, "output": str(paper_out), "lang": "en"}).encode("utf-8"),
                headers={"X-CSRF-Token": token},
            )
            self.assertEqual(c_pe, 200, msg=pe_body)
            self.assertTrue((paper_out / "PROVENANCE.json").is_file())
        finally:
            stop_ui_server(proc)
            secret_file.unlink(missing_ok=True)
        after = {p.relative_to(run_dir).as_posix(): sha256_file(p) for p in sorted(run_dir.rglob("*")) if p.is_file()}
        self.assertEqual(before, after)

    # S10: Non-Interactive CLI, No-Color & Legacy Invariant Guardrails
    def test_s10_cli_non_interactive_no_color_and_legacy_guardrails(self):
        for cmd in (
            ["--version"],
            ["--no-color", "doctor", "--device", "cpu"],
            ["runs", "--root", "runs", "--limit", "5", "--json"],
            ["legacy-help"],
        ):
            proc = run_cli(cmd)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
            assert_zero_ansi(proc.stdout)
            assert_zero_ansi(proc.stderr)

        for rel, expected in YOLO_TUNER_SHA256.items():
            self.assertEqual(sha256_file(REPO_ROOT / rel), expected)

    # S11: Dual-Dataset Live Execution Sensitivity (Zero Template / Hardcoding Bias)
    def test_s11_dual_dataset_live_execution_produces_dataset_specific_evidence(self):
        ws = self._root / "s11_dual"
        # Dataset A: clean linearly separable 2-class high-contrast images (16x16)
        ds_a = ws / "ds_a"
        idx = 0
        for split, n in (("train", 12), ("val", 6), ("test", 4)):
            for i in range(n):
                cls_id = i % 2
                img_path = ds_a / split / f"cls_{cls_id}" / f"{split}_{i:03d}.png"
                img_path.parent.mkdir(parents=True, exist_ok=True)
                bg = (235, 25, 25) if cls_id == 0 else (25, 25, 235)
                fg = (245, 245, 245) if cls_id == 0 else (15, 15, 15)
                im = Image.new("RGB", (16, 16), bg)
                ImageDraw.Draw(im).rectangle([4, 4, 12, 12], fill=fg)
                im.putpixel((idx % 16, (idx * 7) % 16), ((idx * 19 + 5) % 255, (idx * 37 + 11) % 255, (idx * 71 + 13) % 255))
                im.save(img_path, format="PNG")
                idx += 1

        # Dataset B: harder 3-class noisy/confounded dataset (24x24)
        ds_b = ws / "ds_b"
        rng = random.Random(2027)
        for split, n in (("train", 18), ("val", 9), ("test", 6)):
            for i in range(n):
                cls_id = i % 3
                img_path = ds_b / split / f"cls_{cls_id}" / f"{split}_{i:03d}.png"
                img_path.parent.mkdir(parents=True, exist_ok=True)
                im = Image.new("RGB", (24, 24))
                px = im.load()
                for y in range(24):
                    for x in range(24):
                        base = (120 + (cls_id - 1) * 18, 120 + (1 - cls_id) * 12, 120) if split == "train" else (120 + (1 - cls_id) * 18, 120, 120)
                        px[x, y] = tuple(max(0, min(255, ch + rng.randint(-60, 60))) for ch in base)
                px[idx % 24, (idx * 7) % 24] = ((idx * 19 + 5) % 255, (idx * 37 + 11) % 255, (idx * 71 + 13) % 255)
                im.save(img_path, format="PNG")
                idx += 1

        runs = {}
        for tag, ds_path in (("a", ds_a), ("b", ds_b)):
            out = ws / f"runs_{tag}"
            opt = run_cli([
                "optimize", "--data", str(ds_path), "--task", "classification",
                "--device", "cpu", "--space", "compact", "--full-epochs", "2",
                "--seeds", "17", "--budget-seconds", "60", "--output", str(out),
            ], timeout=120)
            self.assertEqual(opt.returncode, 0, msg=opt.stderr + opt.stdout)
            rdir = [p for p in out.iterdir() if p.is_dir() and (p / "protocol.json").is_file()][0]
            prot = json.loads((rdir / "protocol.json").read_text(encoding="utf-8"))
            rep = json.loads((rdir / "report.json").read_text(encoding="utf-8"))
            runs[tag] = (prot, rep)

        self.assertNotEqual(runs["a"][0]["dataset_content_id"], runs["b"][0]["dataset_content_id"])
        self.assertNotEqual(runs["a"][0]["example_counts"], runs["b"][0]["example_counts"])
        self.assertNotEqual(runs["a"][0]["num_classes"], runs["b"][0]["num_classes"])
        self.assertNotEqual(runs["a"][0]["dataset_descriptor"], runs["b"][0]["dataset_descriptor"])

        cands_a = [c for c in runs["a"][1]["candidates"] if c.get("complete")]
        cands_b = [c for c in runs["b"][1]["candidates"] if c.get("complete")]
        self.assertGreaterEqual(len(cands_a), 2)
        self.assertGreaterEqual(len(cands_b), 2)

        times_a = [float(c["time_mean"]) for c in cands_a]
        times_b = [float(c["time_mean"]) for c in cands_b]
        qualities_a = [float(c["quality_mean"]) for c in cands_a]
        qualities_b = [float(c["quality_mean"]) for c in cands_b]
        utilities_a = [float(c["utility"]) for c in cands_a]
        utilities_b = [float(c["utility"]) for c in cands_b]

        self.assertNotEqual(times_a, times_b)
        self.assertNotEqual(qualities_a, qualities_b)
        self.assertNotEqual(utilities_a, utilities_b)
        self.assertGreater(max(qualities_a), max(qualities_b))
        # Candidate utilities within each run must genuinely vary across configs (no constant canned output)
        self.assertGreater(len({round(u, 8) for u in utilities_a}), 1)
        self.assertGreater(len({round(u, 8) for u in utilities_b}), 1)
        self.assertGreater(len({round(q, 6) for q in qualities_a + qualities_b}), 1)

        rec_a = runs["a"][1]["recommendation"]
        rec_b = runs["b"][1]["recommendation"]
        self.assertIsNotNone(rec_a)
        self.assertIsNotNone(rec_b)
        self.assertNotEqual(rec_a["validation_mean"], rec_b["validation_mean"])
        self.assertNotEqual(rec_a["utility"], rec_b["utility"])
