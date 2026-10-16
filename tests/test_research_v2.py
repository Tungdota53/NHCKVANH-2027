"""Logic/integrity fixtures only. These mocked outcomes are NOT benchmark data."""

from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from yolo_tuner.pipeline import write_json
from yolo_tuner.research import Objective, digest
from research_pipeline.integrity import sha
from research_pipeline.v2.schema import lock_protocol, validate_protocol, complete, measurements, decision, config_id
from research_pipeline.v2.ledger import Ledger, read_events, worker_cost
from research_pipeline.v2.space import validate_config, build_space, canonical
from research_pipeline.v2.scheduler import run_search, exploration_fits
from research_pipeline.v2.predictors import compatibility, CostModel, fit_card
from research_pipeline.v2.workflow import audit_run, reconcile_state


class V2Tests(unittest.TestCase):
    def config(self, **changes):
        return {"model": "cnn_classifier", "imgsz": 32, "batch": 4, "accumulation": 2, "width": 16,
                "optimizer": "AdamW", "precision": "fp32", "lr0": 0.001, "weight_decay": 0.01, **changes}

    def protocol(self, **changes):
        return lock_protocol({"task": "classification", "method": "reserve", "mode": "online",
                              "implementation_version": "0.7.0", "configs": [self.config(), self.config(lr0=0.003)],
                              "training_seeds": [17, 29, 43], "search_seed": 17, "full_epochs": 4, "probe_epochs": 1,
                              "budget_seconds": 30, "trial_timeout": 20, "objective": asdict(Objective()),
                              "hardware": {"device": "cpu", "host_label": "unit-host"}, "runtime_versions": {},
                              "scheduler": {"cleanup_seconds": 0.1, "probe_fraction": 0.1, "cost_margin": 1.25},
                              "example_counts": {"train": 100, "val": 30}, "memory_kind": "process_tree_rss_mib_not_vram",
                              "measurement_environment": "uncontended_preflight", "preparation_seconds": 0,
                              "source_files": {}, "data_files": {}, "model_files": {}, "dataset_source": "unit-mock-not-real-data",
                              "quality_definition": "mock fixture", **changes})

    def result(self, seed=17, **changes):
        return {"status": "ok", "seed": seed, "mode": "train", "epochs_completed": 4,
                "optimizer_steps": 10, "training_loss_finite": True, "actual_config_verified": True,
                "quality": 0.5, "memory_mb": 40, "worker_wall_seconds": 2,
                "checkpoint_sha256": "unit-placeholder-not-real-weights", **changes}

    def test_full_evidence_requires_all_conditions(self):
        self.assertTrue(complete(self.result(), 4))
        for bad in ({"mode": "search"}, {"epochs_completed": 3}, {"optimizer_steps": 0}, {"training_loss_finite": False},
                    {"actual_config_verified": False}, {"checkpoint_sha256": None}, {"quality": float("nan")},
                    {"quality": 1.1}, {"status": "timeout"}):
            self.assertFalse(complete(self.result(**bad), 4), bad)

    def test_partial_higher_quality_cannot_win(self):
        results = {"0": [self.result(s) for s in (17, 29, 43)], "1": [self.result(17, quality=0.99)]}
        rows = measurements(self.protocol()["configs"], results, [17, 29, 43], 4, Objective())
        self.assertEqual(decision(rows)["selected_index"], 0)
        self.assertEqual(decision(rows)["verification_status"], "fully_trained_best_observed")
        self.assertIsNone(rows[1]["utility"])

    def test_no_outcome_is_null_not_zero(self):
        rows = measurements([self.config()], {}, [17], 4, Objective())
        self.assertIsNone(rows[0]["quality_mean"])
        self.assertIsNone(decision(rows)["selected_index"])

    def test_worst_seed_constraints(self):
        rows = measurements([self.config()], {"0": [self.result(17), self.result(29, memory_mb=101)]}, [17, 29], 4, Objective(max_memory=100))
        self.assertEqual(rows[0]["state"], "confirmed_constraint_failure")

    def test_duplicate_seed_not_full_coverage(self):
        rows = measurements([self.config()], {"0": [self.result(17), self.result(17)]}, [17, 29], 4, Objective())
        self.assertFalse(rows[0]["complete"])

    def test_protocol_tampering_and_duplicate_seeds_rejected(self):
        p = self.protocol()
        p["budget_seconds"] = 999
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_protocol(p)
        with self.assertRaises(ValueError):
            self.protocol(training_seeds=[17, 17])

    def test_reserve_invariant_exact_bounds(self):
        for remaining in range(1, 50):
            for reserve in range(5):
                for action in range(10):
                    if exploration_fits(remaining, action, reserve, 1):
                        self.assertGreaterEqual(remaining - action, reserve + 1)
        self.assertFalse(exploration_fits(10, None, 1, 1))
        self.assertFalse(exploration_fits(10, float("nan"), 1, 1))

    def scheduler(self, method="reserve", budget=30, initial=None):
        now, calls = [0.0], []
        p = self.protocol(method=method, budget_seconds=budget)
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp) / "events.jsonl", clock=lambda: now[0])
            def evaluate(i, seed, epochs, full, seconds):
                cost = 2 if full else 1
                now[0] += min(cost, seconds)
                calls.append((i, seed, full))
                return self.result(seed, index=i, mode="train" if full else "search", epochs_completed=epochs,
                                   quality=0.5 + 0.1 * i, status="ok" if seconds >= cost else "timeout",
                                   worker_wall_seconds=min(cost, seconds), estimated_full_seconds=2,
                                   measured_loop_seconds=1, attempt_id=f"unit{len(calls)}")
            state = run_search(p, evaluate, lambda: budget - now[0], ledger, initial)
            return state, calls

    def test_confirm_three_seeds_before_opening_other_config(self):
        state, calls = self.scheduler()
        full_calls = [c for c in calls if c[2]]
        self.assertEqual(len({i for i, _, full in full_calls[:3]}), 1)
        self.assertEqual({s for _, s, _ in full_calls[:3]}, {17, 29, 43})
        self.assertIsNotNone(state["decision"]["selected_index"])

    def test_small_budget_does_not_promote_proxy_or_partial(self):
        state, calls = self.scheduler(budget=3)
        self.assertIsNone(state["decision"]["selected_index"])

    def test_no_reserve_ablation_opens_different_candidates(self):
        _, calls = self.scheduler("no_reserve")
        self.assertEqual(len({i for i, _, _ in calls[:2]}), 2)

    def test_profile_all_charges_probes_first(self):
        state, calls = self.scheduler("profile_all")
        self.assertFalse(calls[0][2])
        self.assertFalse(calls[1][2])
        self.assertEqual(len(state["probes"]), 2)

    def test_complete_resume_never_repeats_worker(self):
        initial = {"results": {str(i): [self.result(s, index=i) for s in (17, 29, 43)] for i in (0, 1)},
                   "probes": [], "actions": [], "ordering": [0, 1]}
        state, calls = self.scheduler(initial=initial)
        self.assertEqual(calls, [])
        self.assertTrue(state["decision"]["space_complete"])

    def test_failed_attempt_retained_after_retry(self):
        initial = {"results": {"0": [self.result(17, index=0, status="timeout", worker_wall_seconds=10)], "1": []},
                   "probes": [], "actions": [], "ordering": [0, 1]}
        state, _ = self.scheduler(initial=initial)
        self.assertEqual(state["results"]["0"][0]["worker_wall_seconds"], 10)
        self.assertEqual(state["results"]["0"][0]["status"], "timeout")

    def test_ledger_hash_chain_detects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "events.jsonl"
            clock = [0]
            ledger = Ledger(file, clock=lambda: clock[0])
            ledger.emit("start")
            clock[0] = 2
            ledger.emit("trial_completed", worker_seconds=1)
            self.assertEqual(worker_cost(read_events(file)), 1)
            value = file.read_text().replace('"worker_seconds": 1', '"worker_seconds": 0')
            file.write_text(value)
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_events(file)

    def test_ledger_interrupted_tail_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "events.jsonl"
            file.write_bytes(b'{"incomplete":')
            with self.assertRaisesRegex(ValueError, "interrupted"):
                read_events(file)

    def test_ledger_resume_excludes_downtime_and_retains_active_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "events.jsonl"
            clock = [0]
            ledger = Ledger(file, clock=lambda: clock[0])
            clock[0] = 10
            ledger.emit("end")
            clock[0] = 1000
            resumed = Ledger(file, clock=lambda: clock[0])
            clock[0] = 1002
            self.assertEqual(resumed.elapsed, 12)

    def test_content_id_stable_when_model_relocated(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.yaml", Path(tmp) / "b.yaml"
            a.write_bytes(b"unit model placeholder")
            b.write_bytes(a.read_bytes())
            self.assertEqual(config_id(self.config(model=str(a))), config_id(self.config(model=str(b))))

    def test_capabilities_reject_unsupported_precision_and_optimizer(self):
        for bad in (self.config(precision="fp16"), self.config(optimizer="SGD"), self.config(weight_decay=0.001), self.config(model="unknown")):
            with self.assertRaises(ValueError):
                validate_config("classification", bad, {"device": "cpu"})

    def test_core_has_twelve_controlled_configs_and_inactive_hidden_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            cs = build_space("classification", {"device": "cpu"}, folder, preset="core")
            self.assertEqual(len(cs), 12)
            self.assertTrue(all("hidden" not in c for c in cs))
            grid = folder / "grid.json"
            grid.write_text('[{"hidden":256}]')
            with self.assertRaisesRegex(ValueError, "supported"):
                build_space("classification", {"device": "cpu"}, folder, grid=grid)

    def test_card_rejects_unseen_physical_gpu(self):
        p = self.protocol(hardware={"device": "0", "gpu_uuid": "GPU-new"})
        card = {"task": "classification", "hardware_ids": ["GPU-old"], "runtimes": [{}], "implementations": ["0.7.0"]}
        self.assertIn("physical hardware unseen", compatibility(p, card))

    def test_cost_forecast_is_not_claimed_as_upper_bound(self):
        model = CostModel(self.protocol())
        model.update(0, self.result(index=0))
        prediction = model.estimate(0)
        self.assertFalse(prediction["upper_bound_guaranteed"])
        self.assertEqual(prediction["calibration"], "uncalibrated_scheduling")
        self.assertEqual(prediction["planning_seconds"], 2.5)

    def test_calibration_domain_leakage_is_rejected(self):
        p = self.protocol(dataset_content_id="same-domain")
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "partitions.json"
            file.write_text(json.dumps({"fit": ["fit"], "calibration": ["calibration"]}))
            with patch("research_pipeline.v2.workflow.audit_run", return_value=(Path(tmp), p, [], 0)):
                with self.assertRaisesRegex(ValueError, "leakage"):
                    fit_card(file, Path(tmp) / "card.json")

    def fixture(self, folder):
        manifest = {"task": "classification", "splits": {"train": [], "val": [], "test": []}}
        p = self.protocol(training_seeds=[17], configs=[self.config()], dataset_sha256=digest(manifest),
                          subset_sha256=digest(manifest), run_directory=str(folder))
        write_json(folder / "protocol.json", p)
        write_json(folder / "dataset.json", manifest)
        write_json(folder / "subset.json", manifest)
        write_json(folder / "status.json", {"phase": "interrupted"})
        trial = folder / "trials" / "attempt_00001"
        trial.mkdir(parents=True)
        checkpoint = trial / "last.pt"
        checkpoint.write_bytes(b"mock unit placeholder, not trained weights")
        req = {"config": p["configs"][0], "seed": 17, "full_epochs": 4, "trial_epochs": 4,
               "mode": "train", "resume": None, "manifest": str(folder / "dataset.json"),
               "protocol_sha256": p["protocol_sha256"]}
        raw = self.result(checkpoint_sha256=sha(checkpoint), best_checkpoint=str(checkpoint), config=req["config"])
        write_json(trial / "request.json", req)
        write_json(trial / "result.json", raw)
        result = {**raw, "index": 0, "attempt_id": "unit-original", "trial_directory": str(trial),
                  "request_sha256": sha(trial / "request.json"), "raw_sha256": sha(trial / "result.json")}
        write_json(trial / "parent_result.json", result)
        ledger = Ledger(folder / "events.jsonl")
        ledger.emit("trial_started", attempt_id="unit-original")
        ledger.emit("trial_completed", attempt_id="unit-original", parent_sha256=sha(trial / "parent_result.json"),
                    worker_seconds=2, status="ok", directory=str(trial))
        state = {"results": {"0": [result]}, "probes": [], "actions": [], "ordering": [0]}
        write_json(folder / "state.json", state)
        return trial, ledger, state

    def test_raw_audit_reads_actual_request_checkpoint_and_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            trial, _, _ = self.fixture(folder)
            self.assertEqual(audit_run(folder)[3], 1)
            (trial / "last.pt").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "Checkpoint"):
                audit_run(folder)

    def test_raw_changed_parent_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            trial, _, _ = self.fixture(folder)
            (trial / "parent_result.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "Parent"):
                audit_run(folder)

    def test_worker_omitted_from_state_is_not_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _, _, state = self.fixture(folder)
            state["results"]["0"] = []
            write_json(folder / "state.json", state)
            with self.assertRaisesRegex(ValueError, "omitted"):
                audit_run(folder)
            reconcile_state(folder)
            self.assertEqual(audit_run(folder)[3], 1)

    def test_unknown_cost_after_hard_crash_blocks_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _, ledger, _ = self.fixture(folder)
            ledger.emit("trial_started", attempt_id="unrecorded")
            with self.assertRaisesRegex(ValueError, "zero-cost"):
                reconcile_state(folder)

    def test_root_v2_help_and_unsupported_flags(self):
        from research_pipeline.v2.cli import main, parser
        stream = io.StringIO()
        with patch("sys.argv", ["train_config.py", "--help"]), patch("sys.stdout", stream), self.assertRaises(SystemExit) as exc:
            main()
        self.assertEqual(exc.exception.code, 0)
        self.assertIn("optimize", stream.getvalue())
        self.assertIn("benchmark", stream.getvalue())
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            parser().parse_args(["optimize", "--data", "mock", "--evaluate-test"])

    def test_adaptive_cells_match_forward_and_gradient(self):
        import torch
        import torch.nn.functional as F
        from research_pipeline.v2.pooling import deterministic_adaptive_pool2d
        for height, width in ((7, 11), (10, 10), (3, 2)):
            a = torch.randn(2, 3, height, width, dtype=torch.float64, requires_grad=True)
            expected = F.adaptive_avg_pool2d(a, (4, 4))
            actual = deterministic_adaptive_pool2d(a, (4, 4))
            self.assertTrue(torch.allclose(expected, actual, atol=1e-12))
            weights = torch.randn_like(expected)
            eg = torch.autograd.grad((expected * weights).sum(), a, retain_graph=True)[0]
            ag = torch.autograd.grad((actual * weights).sum(), a)[0]
            self.assertTrue(torch.allclose(eg, ag, atol=1e-12))

    def test_adaptive_cells_strict_cuda_backward_if_available(self):
        import torch
        from research_pipeline.v2.pooling import deterministic_adaptive_pool2d
        if not torch.cuda.is_available():
            self.skipTest("No CUDA runtime; CPU is not a CUDA determinism proof")
        prior = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            a = torch.randn(2, 3, 10, 10, device="cuda", requires_grad=True)
            deterministic_adaptive_pool2d(a, 4).sum().backward()
            self.assertTrue(torch.isfinite(a.grad).all().item())
        finally:
            torch.use_deterministic_algorithms(prior)

    def test_resume_total_cost_retains_prior_budget_and_attempts(self):
        from research_pipeline.v2.report import make_report
        p = self.protocol(budget_seconds=30)
        state = {"results": {"0": [], "1": []}, "probes": []}
        events = [
            {"kind": "session_started", "payload": {}, "sha256": "unit"},
            {"kind": "selection_committed", "payload": {"session_seconds": 25}, "sha256": "unit"},
            {"kind": "resume_requested", "payload": {"additional_budget_seconds": 20}, "sha256": "unit"},
            {"kind": "session_interrupted", "payload": {"session_seconds": 5}, "sha256": "unit"},
            {"kind": "session_started", "payload": {}, "sha256": "unit"},
            {"kind": "selection_committed", "payload": {"session_seconds": 10}, "sha256": "unit"},
        ]
        report = make_report(p, state, events)
        self.assertEqual(report["online_session_seconds"], 40)
        self.assertEqual(report["total_granted_budget_seconds"], 50)
        self.assertEqual(report["budget_overrun_seconds"], 0)

    def test_reference_regret_uses_reference_not_online_utility(self):
        from research_pipeline.v2.benchmark import compare
        p = self.protocol(dataset_content_id="unit-domain")
        results = {"0": [self.result(s, index=0) for s in (17, 29, 43)],
                   "1": [self.result(s, index=1, quality=0.6) for s in (17, 29, 43)]}
        rows = measurements(p["configs"], results, p["training_seeds"], 4, Objective())
        with tempfile.TemporaryDirectory() as tmp:
            reference, online = Path(tmp) / "reference", Path(tmp) / "online"
            reference.mkdir(); online.mkdir()
            write_json(reference / "report.json", {"online_session_seconds": 20})
            write_json(online / "report.json", {"online_session_seconds": 3, "budget_overrun_seconds": 0})
            write_json(online / "decision.locked.json", {"selected_config_id": rows[0]["config_id"]})
            # An implausible online utility must NOT affect independent regret.
            online_rows = [{**r, "utility": 999} for r in rows]
            def audit(path):
                return (Path(path), p, rows if str(path) == str(reference) else online_rows, 6)
            with patch("research_pipeline.v2.benchmark.audit_run", side_effect=audit):
                report = compare([str(online)], str(reference))
            self.assertAlmostEqual(report["outcomes"][0]["objective_regret"], 0.06)

    def test_prepared_protocol_option_is_explicit(self):
        from research_pipeline.v2.cli import parser
        args = parser().parse_args(["optimize", "--protocol", "saved.json"])
        self.assertIsNone(args.data)
        self.assertEqual(args.protocol, "saved.json")

    def test_post_commit_export_failure_not_double_charged(self):
        from research_pipeline.v2.report import make_report
        events = [{"kind": "session_started", "payload": {}, "sha256": "unit"},
                  {"kind": "selection_committed", "payload": {"session_seconds": 25}, "sha256": "unit"},
                  {"kind": "session_interrupted", "payload": {"session_seconds": 26}, "sha256": "unit"}]
        report = make_report(self.protocol(), {"results": {"0": [], "1": []}, "probes": []}, events)
        self.assertEqual(report["online_session_seconds"], 25)

    def test_reference_tuple_and_json_list_have_same_identity(self):
        objective = asdict(Objective())
        persisted = json.loads(json.dumps(objective))
        self.assertNotEqual(objective, persisted)
        self.assertEqual(digest(objective), digest(persisted))

    def test_external_yolo_yaml_without_local_images_is_supported(self):
        from research_pipeline.v2.data import discover
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            external = root / "external"
            for split, color in (("train", "red"), ("val", "blue")):
                (external / "images" / split).mkdir(parents=True)
                (external / "labels" / split).mkdir(parents=True)
                Image.new("RGB", (16, 16), color).save(external / "images" / split / "a.png")
                (external / "labels" / split / "a.txt").write_text("0 0.5 0.5 0.5 0.5")
            metadata = root / "metadata"
            metadata.mkdir()
            yaml = metadata / "data.yaml"
            yaml.write_text(f'path: "{external.as_posix()}"\ntrain: images/train\nval: images/val\nnames: [person]\n')
            data = discover(yaml, task="detection")
            self.assertEqual(data["counts"]["train"], 1)
            self.assertEqual(data["counts"]["val"], 1)

    def test_group_leakage_manifest_rejected(self):
        from research_pipeline.v2.data import discover
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, color in (("a", "red"), ("b", "blue")):
                Image.new("RGB", (16, 16), color).save(root / f"{name}.png")
            manifest = root / "manifest.json"
            write_json(manifest, {"task": "classification", "names": ["a", "b"], "splits": {
                "train": [{"image": "a.png", "class_id": 0, "group_id": "same-scene"}],
                "val": [{"image": "b.png", "class_id": 1, "group_id": "same-scene"}]}})
            with self.assertRaisesRegex(ValueError, "group crosses"):
                discover(manifest)


if __name__ == "__main__":
    unittest.main()
