"""Tier 0 / R1 Invariants: must pass 100% immediately on 0.11.0 and remain green across 1.0.0."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from research_pipeline.v2.algorithm_study import analyze_studies
from research_pipeline.v2.ledger import read_events
from research_pipeline.v2.predictors import training_signature
from research_pipeline.v2.report import make_report
from research_pipeline.v2.schema import config_id, decision, measurements
from research_pipeline.v2.selection import select
from research_pipeline.v2.workflow import audit_run, source_map
from yolo_tuner.research import Objective, digest

from tests_e2e._e2e_fixtures import (
    ALGORITHM_010_EXPECTED_SHA256,
    ALGORITHM_010_MANIFEST_FILE_SHA256,
    REPO_ROOT,
    YOLO_TUNER_SHA256,
    assert_zero_ansi,
    make_mini_yolo_dataset,
    make_synthetic_n5_study_dir,
    run_cli,
    sha256_file,
)


class TestR1Invariants(unittest.TestCase):
    def test_r1_all_19_yolo_tuner_files_present_and_sha256_unchanged(self):
        actual_files = sorted(
            p.relative_to(REPO_ROOT).as_posix()
            for p in (REPO_ROOT / "yolo_tuner").glob("*.py")
        )
        self.assertEqual(actual_files, sorted(YOLO_TUNER_SHA256.keys()))
        self.assertEqual(len(actual_files), 19)
        for rel_path, expected_hash in YOLO_TUNER_SHA256.items():
            self.assertEqual(
                sha256_file(REPO_ROOT / rel_path),
                expected_hash,
                f"Immutable legacy file hash changed: {rel_path}",
            )

    def test_r1_algorithm_010_manifest_sha256_unchanged(self):
        manifest_path = REPO_ROOT / "configs" / "algorithm_010_studies.json"
        self.assertTrue(manifest_path.is_file())
        self.assertEqual(sha256_file(manifest_path), ALGORITHM_010_MANIFEST_FILE_SHA256)

    def test_r1_algorithm_010_analyze_studies_api_sha256_unchanged(self):
        report = analyze_studies(REPO_ROOT / "configs" / "algorithm_010_studies.json")
        self.assertEqual(report["sha256"], ALGORITHM_010_EXPECTED_SHA256)

    def test_r1_algorithm_010_cli_report_sha256_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "alg_010_report"
            proc = run_cli(
                [
                    "algorithm-report",
                    "--manifest",
                    "configs/algorithm_010_studies.json",
                    "--output",
                    str(out_dir),
                ],
                timeout=120,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
            assert_zero_ansi(proc.stdout)
            assert_zero_ansi(proc.stderr)
            self.assertTrue((out_dir / "ALGORITHM_STUDY.json").is_file())
            self.assertTrue((out_dir / "ALGORITHM_STUDY.md").is_file())
            parsed = json.loads((out_dir / "ALGORITHM_STUDY.json").read_text(encoding="utf-8"))
            self.assertEqual(parsed["sha256"], ALGORITHM_010_EXPECTED_SHA256)

    def test_r1_historical_runs_representative_versions_audit_and_report(self):
        sample_paths = [
            REPO_ROOT / "runs" / "v2_smoke_cpu",
            REPO_ROOT / "runs" / "verified_detection_071",
            REPO_ROOT / "runs" / "upgrade_091_detection",
            REPO_ROOT / "runs" / "algorithm_010_detection_integration",
            REPO_ROOT / "runs" / "african_wildlife_deploy_accuracy",
            REPO_ROOT / "runs" / "nckh_011_map50_verification",
        ]
        checked_any = 0
        for base in sample_paths:
            if not base.exists():
                continue
            protocols = sorted(base.rglob("protocol.json"))
            for p_file in protocols:
                run_dir = p_file.parent
                status_file = run_dir / "status.json"
                if not status_file.exists():
                    continue
                if json.loads(status_file.read_text(encoding="utf-8")).get("phase") != "complete":
                    continue
                folder, protocol, rows, count = audit_run(run_dir)
                state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
                events = read_events(folder / "events.jsonl")
                report = make_report(protocol, state, events)
                self.assertEqual(report["adapter"], "research_v2")
                self.assertGreaterEqual(count, 1)
                checked_any += 1
        self.assertGreaterEqual(checked_any, 5)

    def test_r1_historical_runs_all_complete_audit_and_report(self):
        complete_runs = []
        for p_file in sorted((REPO_ROOT / "runs").rglob("protocol.json")):
            run_dir = p_file.parent
            prot = json.loads(p_file.read_text(encoding="utf-8"))
            if prot.get("adapter") != "research_v2":
                continue
            status_file = run_dir / "status.json"
            if not status_file.exists():
                continue
            if json.loads(status_file.read_text(encoding="utf-8")).get("phase") == "complete":
                complete_runs.append(run_dir)
        self.assertGreaterEqual(len(complete_runs), 61)
        for run_dir in complete_runs:
            folder, protocol, _, _ = audit_run(run_dir)
            state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
            events = read_events(folder / "events.jsonl")
            rebuilt = make_report(protocol, state, events)
            self.assertEqual(rebuilt["protocol_sha256"], protocol["protocol_sha256"])

    def test_r1_cli_version_and_help_exit_zero_no_ansi(self):
        for entry in ("module", "script"):
            v_proc = run_cli(["--version"], entry=entry)
            self.assertEqual(v_proc.returncode, 0, msg=v_proc.stderr)
            self.assertIn("train-config", v_proc.stdout)
            assert_zero_ansi(v_proc.stdout)
            assert_zero_ansi(v_proc.stderr)

            h_proc = run_cli(["--help"], entry=entry)
            self.assertEqual(h_proc.returncode, 0, msg=h_proc.stderr)
            self.assertIn("optimize", h_proc.stdout)
            self.assertIn("inspect", h_proc.stdout)
            assert_zero_ansi(h_proc.stdout)
            assert_zero_ansi(h_proc.stderr)

    def test_r1_cli_unknown_command_exit_code_2(self):
        proc = run_cli(["definitely-not-a-valid-command"])
        self.assertEqual(proc.returncode, 2)
        assert_zero_ansi(proc.stderr)

    def test_r1_cli_doctor_json_offline_schema(self):
        proc = run_cli(["doctor", "--device", "cpu", "--json"])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
        assert_zero_ansi(proc.stdout)
        payload = json.loads(proc.stdout)
        self.assertIn("hardware", payload)
        hw = payload["hardware"]
        self.assertTrue(hw.get("ml_runtime_ready"))
        self.assertEqual(hw.get("device"), "cpu")
        self.assertGreater(hw.get("ram_available_mb", 0), 0)

    def test_r1_cli_runs_json_reads_historical_runs(self):
        proc = run_cli(["runs", "--root", "runs", "--limit", "5", "--json"])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
        assert_zero_ansi(proc.stdout)
        entries = json.loads(proc.stdout)
        self.assertIsInstance(entries, list)
        self.assertGreaterEqual(len(entries), 1)
        self.assertLessEqual(len(entries), 5)

    def test_r1_cli_inspect_mini_yolo_json_and_non_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            ds_root = Path(td) / "mini_yolo"
            yaml_path = make_mini_yolo_dataset(ds_root, n_train=4, n_val=2, n_test=2, size=64)
            before = {
                p.relative_to(ds_root).as_posix(): sha256_file(p)
                for p in sorted(ds_root.rglob("*"))
                if p.is_file()
            }
            proc = run_cli(["inspect", "--data", str(yaml_path), "--device", "cpu", "--json"])
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
            assert_zero_ansi(proc.stdout)
            info = json.loads(proc.stdout)
            ds = info["dataset"]
            self.assertEqual(ds["task"], "detection")
            self.assertTrue(ds["validation_ready"])
            self.assertEqual(ds["counts"], {"train": 4, "val": 2, "test": 2})
            after = {
                p.relative_to(ds_root).as_posix(): sha256_file(p)
                for p in sorted(ds_root.rglob("*"))
                if p.is_file()
            }
            self.assertEqual(before, after)

    def test_r1_cli_plan_search_in_temp_dir_no_training(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            yaml_path = make_mini_yolo_dataset(root / "ds", n_train=4, n_val=2, n_test=2, size=64)
            out_dir = root / "planned"
            proc = run_cli([
                "plan-search",
                "--data", str(yaml_path),
                "--device", "cpu",
                "--space", "compact",
                "--full-epochs", "1",
                "--seeds", "17",
                "--budget-seconds", "60",
                "--output", str(out_dir),
            ])
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
            assert_zero_ansi(proc.stdout)
            run_folders = [p for p in out_dir.iterdir() if p.is_dir() and (p / "protocol.json").is_file()]
            self.assertEqual(len(run_folders), 1)
            run_dir = run_folders[0]
            prot = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(prot["schema_version"], 4)
            self.assertEqual(prot["adapter"], "research_v2")
            self.assertEqual(status["phase"], "prepared")
            self.assertFalse((run_dir / "trials").exists())

    def test_r1_config_id_invariance_on_canonical_configs(self):
        cfg_a = {
            "model": "yolo11n.yaml",
            "imgsz": 64,
            "batch": 2,
            "accumulation": 1,
            "precision": "fp32",
            "optimizer": "AdamW",
            "lr0": 0.001,
            "weight_decay": 0.0005,
            "freeze": 0,
        }
        cfg_b = dict(reversed(list(cfg_a.items())))
        self.assertEqual(config_id(cfg_a), config_id(cfg_b))
        self.assertEqual(len(config_id(cfg_a)), 12)
        self.assertEqual(config_id(cfg_a), digest(cfg_a)[:12])

    def test_r1_selection_and_measurements_exclude_partial_seeds(self):
        cfg = {
            "model": "cnn_classifier",
            "imgsz": 32,
            "batch": 16,
            "accumulation": 2,
            "precision": "fp32",
            "optimizer": "AdamW",
            "lr0": 0.001,
            "weight_decay": 0.01,
            "width": 16,
        }
        obj = Objective("balanced", 0.0, None, 2048.0, 60.0, 512.0, (0.6, 0.25, 0.15))
        partial_only = {
            "0": [
                {
                    "seed": 17,
                    "mode": "train",
                    "status": "ok",
                    "epochs_completed": 2,
                    "optimizer_steps": 4,
                    "training_loss_finite": True,
                    "actual_config_verified": True,
                    "checkpoint_sha256": "abc",
                    "quality": 0.99,
                    "memory_mb": 100.0,
                    "worker_wall_seconds": 1.0,
                }
            ]
        }
        rows = measurements([cfg], partial_only, [17, 29], 2, obj)
        self.assertFalse(rows[0]["complete"])
        self.assertEqual(rows[0]["state"], "unresolved")
        self.assertIsNone(rows[0]["utility"])
        dec = decision(rows)
        self.assertIsNone(dec["selected_index"])
        sel = select(rows, {"objective": obj.__dict__, "selection_rule": "weighted"})
        self.assertIsNone(sel["selected_index"])

    def test_r1_failed_run_has_null_regret_in_algorithm_study(self):
        with tempfile.TemporaryDirectory() as td:
            _, manifest = make_synthetic_n5_study_dir(Path(td), failed_pair=("random", 11))
            rep = analyze_studies(manifest)
            random_outcomes = [r for r in rep["studies"][0]["outcomes"] if r["method"] == "random"]
            failed = [r for r in random_outcomes if r["search_seed"] == 11][0]
            self.assertFalse(failed["feasible"])
            self.assertIsNone(failed["objective_regret"])
            random_sum = next(m for m in rep["studies"][0]["method_summaries"] if m["method"] == "random")
            self.assertEqual(random_sum["no_verified_selection_count"], 1)

    def test_r1_training_signature_present_and_sensitive_to_worker(self):
        sources = source_map()
        prot = {"source_files": dict(sources)}
        sig1 = training_signature(prot)
        sig2 = training_signature({"source_files": dict(sources)})
        self.assertEqual(sig1, sig2)
        mutated = dict(sources)
        mutated["research_pipeline/worker.py"] = "0" * 64
        self.assertNotEqual(sig1, training_signature({"source_files": mutated}))

    def test_r1_optimize_never_reads_or_leaks_test_split_into_subset_or_selection(self):
        from tests_e2e._e2e_fixtures import make_mini_cls_dataset
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cls_dir = make_mini_cls_dataset(root / "cls", n_train=8, n_val=4, n_test=4, size=16)
            out_dir = root / "runs"
            opt = run_cli([
                "optimize", "--data", str(cls_dir), "--task", "classification",
                "--device", "cpu", "--space", "compact", "--full-epochs", "1",
                "--seeds", "17", "--budget-seconds", "45", "--output", str(out_dir),
            ], timeout=120)
            self.assertEqual(opt.returncode, 0, msg=opt.stderr + opt.stdout)
            run_dir = [p for p in out_dir.iterdir() if p.is_dir() and (p / "protocol.json").is_file()][0]
            subset_txt = (run_dir / "subset.json").read_text(encoding="utf-8")
            state_txt = (run_dir / "state.json").read_text(encoding="utf-8")
            report_txt = (run_dir / "report.json").read_text(encoding="utf-8")
            subset_obj = json.loads(subset_txt)
            self.assertEqual(subset_obj.get("splits", {}).get("test", []), [])
            for idx in range(4):
                token = f"test_{idx:03d}.png"
                self.assertNotIn(token, subset_txt)
                self.assertNotIn(token, state_txt)
                self.assertNotIn(token, report_txt)
            for req_file in sorted((run_dir / "trials").rglob("request.json")):
                req_txt = req_file.read_text(encoding="utf-8")
                self.assertNotIn("test_000.png", req_txt)
            self.assertFalse((run_dir / "test_history.json").exists())
