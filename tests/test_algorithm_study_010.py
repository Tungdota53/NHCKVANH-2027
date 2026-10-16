"""Analysis fixtures only; mocked runs/intervals are not research evidence."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_pipeline.integrity import sha
from research_pipeline.v2 import algorithm_study as study
from yolo_tuner.pipeline import write_json
from yolo_tuner.research import digest


class AlgorithmStudyTests(unittest.TestCase):
    def fixture(self, root, name="one", seeds=(1, 2, 3), dataset="dataset-A", hardware=None, failed=None):
        folder = root / name
        folder.mkdir()
        methods = ["random", "reserve", "default", "successive_halving"]
        sources = {"fixture_worker.py": "not-real-evidence"}
        manifest = {"methods": methods, "search_seeds": list(seeds), "sources": sources, "test_used": False,
                    "arguments": {"budget_seconds": 20}, "phase": "locked_before_search"}
        manifest["sha256"] = digest(manifest)
        write_json(folder / "benchmark_protocol.json", manifest)
        hardware = hardware or {"device": "cpu", "host_label": "fixture-host", "cpu": "fixture-cpu", "torch_version": "fixture"}
        common = {"task": "classification", "dataset_content_id": dataset, "config_ids": ["c0", "c1"],
                  "training_seeds": [17, 29, 43], "full_epochs": 4, "objective": {"weights": [.6, .25, .15]},
                  "selection_rule": "weighted", "runtime_versions": {"fixture": "only"}, "memory_kind": "fixture-memory-kind",
                  "execution_policy": {"version": "fixture"}, "subset_sha256": "fixture-subset", "split_seed": 101,
                  "probe_epochs": 1, "trial_timeout": 20, "scheduler": {"cleanup_seconds": 1, "probe_fraction": .1},
                  "measurement_environment": "uncontended_preflight", "source_files": sources, "hardware": hardware,
                  "budget_seconds": 20}
        audits, outcomes, paths, decisions = {}, [], [], {}
        for method in methods:
            for seed in seeds:
                run = folder / method / str(seed)
                run.mkdir(parents=True)
                failed_selection = (method, seed) == failed
                decision = {"selected_config_id": None if failed_selection else "c0"}
                write_json(run / "decision.locked.json", decision)
                write_json(run / "report.json", {"online_session_seconds": 18, "total_granted_budget_seconds": 20,
                                                  "additional_resume_budget_seconds": 0})
                protocol = {**copy.deepcopy(common), "mode": "online", "method": method, "search_seed": seed}
                audits[str(run)] = (run, protocol, [{"complete": False}], 1)
                paths.append(str(run))
                decisions[str(run)] = digest(decision)
                regret = {"random": .2, "reserve": .1, "default": .3, "successive_halving": .15}[method]
                outcomes.append({"run": str(run), "method": method, "search_seed": seed,
                                 "selected": decision["selected_config_id"], "feasible": not failed_selection,
                                 "reference_feasibility_disagreement": False, "objective_regret": None if failed_selection else regret,
                                 "selection_rule_regret": None if failed_selection else regret,
                                 "online_seconds": 18, "budget_seconds": 20, "overrun_seconds": 0,
                                 "time_to_first_full_config_seconds": None if failed_selection else 12})
        boundary = folder / "decisions.locked.json"
        write_json(boundary, {"benchmark_sha256": manifest["sha256"], "runs": paths, "decision_hashes": decisions})
        reference = folder / "reference" / "exhaustive"
        reference.mkdir(parents=True)
        ref_protocol = {**copy.deepcopy(common), "mode": "offline_reference", "method": "random", "search_seed": 123456,
                        "reference_boundary": str(boundary), "reference_boundary_sha256": sha(boundary)}
        write_json(reference / "protocol.json", ref_protocol)
        write_json(reference / "report.json", {"online_session_seconds": 120})
        audits[str(reference)] = (reference, ref_protocol, [{"complete": True, "utility": .5}, {"complete": True, "utility": .4}], 2)
        return {"path": folder, "reference": reference, "audits": audits,
                "comparison": {"reference_complete": True, "outcomes": outcomes}, "protocol": manifest}

    def analyze(self, root, fixtures, *, output=None, entries=None, **options):
        manifest_path = root / "analysis_manifest.json"
        write_json(manifest_path, {"schema_version": 1, "studies": entries or [{"path": str(item["path"])} for item in fixtures],
                                  "bootstrap_samples": 100, **options})
        audits = {path: value for item in fixtures for path, value in item["audits"].items()}
        comparisons = {str(item["reference"]): item["comparison"] for item in fixtures}
        with patch.object(study.benchmark, "_finished_run", side_effect=lambda path: audits[str(Path(path).resolve())]), \
                patch.object(study.benchmark, "compare", side_effect=lambda paths, ref: comparisons[str(ref)]), \
                patch.object(study.benchmark, "child_command") as training:
            report = study.analyze_studies(manifest_path, output)
        training.assert_not_called()
        return report

    def test_failed_selection_is_separate_and_regret_is_not_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            fixture = self.fixture(root, failed=("reserve", 3))
            report = self.analyze(root, [fixture])
        reserve = next(row for row in report["studies"][0]["method_summaries"] if row["method"] == "reserve")
        self.assertEqual(reserve["no_verified_selection_count"], 1)
        self.assertEqual(reserve["conditional_objective_regret"]["n"], 2)
        self.assertAlmostEqual(reserve["conditional_objective_regret"]["mean"], .1)
        pair = next(row for row in report["studies"][0]["paired_comparisons"] if (row["candidate"], row["baseline"]) == ("reserve", "random"))
        self.assertEqual(pair["metrics"]["objective_regret"]["paired_seed_count"], 2)
        self.assertEqual(pair["metrics"]["objective_regret"]["unavailable_pairs"], 1)
        self.assertEqual(pair["metrics"]["feasible"]["paired_seed_count"], 3)

    def test_training_seeds_do_not_inflate_bootstrap_n(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            report = self.analyze(root, [self.fixture(root)])
        result = report["studies"][0]["paired_comparisons"][0]["metrics"]["objective_regret"]["candidate_minus_baseline"]
        self.assertEqual(result["n"], 3)
        self.assertIsNone(result["interval"])
        self.assertEqual(result["status"], "insufficient_search_repeats")

    def test_five_paired_search_repeats_provide_only_conditional_ci(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            report = self.analyze(root, [self.fixture(root, seeds=(1, 2, 3, 4, 5))])
        result = report["studies"][0]["paired_comparisons"][0]["metrics"]["objective_regret"]["candidate_minus_baseline"]
        self.assertEqual(result["n"], 5)
        self.assertIsNotNone(result["interval"])
        self.assertFalse(result["coverage_guaranteed"])
        self.assertFalse(report["statistical_superiority_established"])

    def test_seeded_bootstrap_is_reproducible(self):
        a = study.bootstrap_mean([0, 1, 2, 3, 4], seed=123, samples=100)
        b = study.bootstrap_mean([0, 1, 2, 3, 4], seed=123, samples=100)
        self.assertEqual(a, b)
        self.assertLessEqual(a["interval"][0], a["mean"])
        self.assertGreaterEqual(a["interval"][1], a["mean"])

    def test_same_dataset_repeated_studies_are_not_two_datasets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            report = self.analyze(root, [self.fixture(root, "one"), self.fixture(root, "two")])
        self.assertEqual(report["summary"]["study_count"], 2)
        self.assertEqual(report["summary"]["unique_dataset_count"], 1)
        self.assertEqual(report["summary"]["unique_observed_domain_count"], 1)
        self.assertEqual(report["summary"]["cpu_host_count"], 1)
        self.assertIsNone(report["summary"]["cross_domain_confidence_interval"])

    def test_cpu_and_cuda_same_host_are_not_two_physical_gpus(self):
        gpu = {"device": "0", "host_label": "fixture-host", "cpu": "fixture-cpu", "gpu_uuid": "GPU-fixture",
               "gpu_identity_verified": True, "gpu": "Fixture GPU", "torch_version": "fixture"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            report = self.analyze(root, [self.fixture(root, "cpu"), self.fixture(root, "cuda", hardware=gpu)])
        self.assertEqual(report["summary"]["unique_dataset_count"], 1)
        self.assertEqual(report["summary"]["physical_gpu_count"], 1)
        self.assertEqual(report["summary"]["cpu_host_count"], 1)

    def test_unverified_gpu_uuid_rejected(self):
        gpu = {"device": "0", "host_label": "fixture", "cpu": "fixture-cpu", "gpu_uuid": None, "gpu": "GPU name"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            with self.assertRaisesRegex(ValueError, "physical GPU UUID"):
                self.analyze(root, [self.fixture(root, hardware=gpu)])

    def test_incomplete_reference_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            fixture = self.fixture(root)
            fixture["audits"][str(fixture["reference"])][2][0]["complete"] = False
            with self.assertRaisesRegex(ValueError, "reference is incomplete"):
                self.analyze(root, [fixture])

    def test_missing_reference_rejected_by_existing_raw_auditor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            manifest = root / "analysis.json"
            fixture = self.fixture(root)
            write_json(manifest, {"schema_version": 1, "studies": [{"path": str(fixture["path"])}]})
            with patch.object(study.benchmark, "_finished_run", side_effect=FileNotFoundError("missing reference")):
                with self.assertRaises(FileNotFoundError):
                    study.analyze_studies(manifest)

    def test_missing_or_mismatched_method_rejected(self):
        for mismatch in ("missing", "wrong_method"):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve()
                fixture = self.fixture(root)
                first = next(value for value in fixture["audits"].values() if value[1]["mode"] == "online")
                first[1]["method"] = "reserve" if mismatch == "missing" else "not-locked"
                with self.assertRaisesRegex(ValueError, "Method/repeat"):
                    self.analyze(root, [fixture])

    def test_objective_grid_data_horizon_and_seeds_must_match(self):
        for field, value in (("objective", {"weights": [1, 0, 0]}), ("config_ids", ["different"]),
                             ("dataset_content_id", "other"), ("full_epochs", 8), ("training_seeds", [17]), ("acquisition", {"version": "different"})):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve()
                fixture = self.fixture(root)
                first = next(item for item in fixture["audits"].values() if item[1]["mode"] == "online")
                first[1][field] = value
                with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                    self.analyze(root, [fixture])

    def test_equal_declared_budget_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            fixture = self.fixture(root)
            first = next(item for item in fixture["audits"].values() if item[1]["mode"] == "online")
            first[1]["budget_seconds"] = 21
            write_json(first[0] / "report.json", {"online_session_seconds": 18, "total_granted_budget_seconds": 21})
            with self.assertRaisesRegex(ValueError, "different search budgets"):
                self.analyze(root, [fixture])

    def test_budget_must_be_smaller_than_measured_exhaustive_cost(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            fixture = self.fixture(root)
            write_json(fixture["reference"] / "report.json", {"online_session_seconds": 20})
            with self.assertRaisesRegex(ValueError, "smaller than measured exhaustive"):
                self.analyze(root, [fixture])

    def test_additional_resume_grants_are_not_equal_budget_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            fixture = self.fixture(root)
            first = next(item for item in fixture["audits"].values() if item[1]["mode"] == "online")
            write_json(first[0] / "report.json", {"online_session_seconds": 25, "total_granted_budget_seconds": 30,
                                                 "additional_resume_budget_seconds": 10})
            with self.assertRaisesRegex(ValueError, "Additional resume grants"):
                self.analyze(root, [fixture])

    def test_no_selection_with_numeric_zero_regret_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            fixture = self.fixture(root, failed=("reserve", 3))
            row = next(row for row in fixture["comparison"]["outcomes"] if not row["feasible"])
            row["objective_regret"] = 0
            with self.assertRaisesRegex(ValueError, "null regret"):
                self.analyze(root, [fixture])

    def test_duplicate_study_cannot_inflate_sample_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            fixture = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "Duplicate benchmark study"):
                self.analyze(root, [fixture], entries=[{"path": str(fixture["path"])}, {"path": str(fixture["path"])}])

    def test_output_is_new_local_evidence_and_existing_output_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            output = root / "new-evidence"
            fixture = self.fixture(root)
            report = self.analyze(root, [fixture], output=output)
            saved = json.loads((output / "ALGORITHM_STUDY.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["sha256"], digest({key: value for key, value in saved.items() if key != "sha256"}))
            self.assertTrue((output / "ALGORITHM_STUDY.md").is_file())
            self.assertTrue(report["analysis_only"])
            self.assertFalse(report["training_started"])
            self.assertFalse(report["predictor_fit_or_objective_tuning_performed"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                self.analyze(root, [fixture], output=output)

    def test_required_baselines_are_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            fixture = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "missing required method"):
                self.analyze(root, [fixture], required_methods=["unavailable_method"])

    def test_changed_independent_reference_boundary_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            fixture = self.fixture(root)
            fixture["audits"][str(fixture["reference"])][1]["reference_boundary_sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "reference boundary/source mismatch"):
                self.analyze(root, [fixture])

    def test_changed_cpu_or_gpu_device_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            fixture = self.fixture(root)
            first = next(item for item in fixture["audits"].values() if item[1]["mode"] == "online")
            first[1]["hardware"]["host_label"] = "different-host"
            with self.assertRaisesRegex(ValueError, "device mismatch"):
                self.analyze(root, [fixture])

    def test_nonfinite_cost_or_bootstrap_settings_rejected(self):
        with self.assertRaises(ValueError):
            study.bootstrap_mean([1, 2, float("nan"), 4, 5])
        for settings in ({"samples": 0}, {"samples": 20001}, {"confidence": float("inf")}, {"seed": True}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                study.bootstrap_mean([1, 2, 3, 4, 5], **settings)

    def test_actual_overrun_is_disclosed_not_relabelled_as_hard_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            fixture = self.fixture(root)
            fixture["comparison"]["outcomes"][0].update(online_seconds=22, overrun_seconds=2)
            report = self.analyze(root, [fixture])
        self.assertTrue(report["studies"][0]["any_actual_budget_overrun"])
        random_summary = next(row for row in report["studies"][0]["method_summaries"] if row["method"] == "random")
        self.assertEqual(random_summary["budget_overrun_count"], 1)
        self.assertFalse(report["statistical_superiority_established"])


if __name__ == "__main__":
    unittest.main()
