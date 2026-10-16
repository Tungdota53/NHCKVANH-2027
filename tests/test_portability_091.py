"""Portable entrypoint and report-recovery fixtures; not training evidence."""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from research_pipeline.integrity import sha
from research_pipeline.v2 import benchmark, cli, terminal
from research_pipeline.v2.ledger import Ledger
from yolo_tuner.pipeline import write_json
from yolo_tuner.research import digest


class PortabilityTests(unittest.TestCase):
    def test_module_entrypoint_works_outside_repository_cwd(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([sys.executable, "-m", "research_pipeline.v2", "--version"],
                                    cwd=temp, capture_output=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("train-config", result.stdout)

    def test_benchmark_children_use_package_not_root_script(self):
        command = benchmark.child_command("doctor", "--json")
        self.assertEqual(command, [sys.executable, "-m", "research_pipeline.v2", "doctor", "--json"])
        self.assertFalse(any("train_config.py" in part for part in command))

    def test_copyable_commands_do_not_depend_on_repository_cwd(self):
        command = terminal.command(["status", "--run", "run with spaces"])
        self.assertIn("research_pipeline.v2", command)
        self.assertNotIn("train_config.py", command)

    def test_benchmark_forwards_locked_loader_policy(self):
        args = cli.parser().parse_args(["benchmark", "--data", "fixture", "--methods", "random", "--search-seeds", "17",
                                       "--workers", "2", "--image-cache-mb", "96", "--pin-memory", "on", "--max-memory", "512"])
        report = {"scope": "fixtures", "summaries": [], "reference_complete": True, "reference_cost_seconds": 2}
        commands = []
        def child(command, **kwargs):
            commands.append(command)
            output = Path(command[command.index("--output") + 1]) / "fixture-child"
            output.mkdir(parents=True)
            write_json(output / "protocol.json", {})
            write_json(output / "decision.locked.json", {"selected_config_id": None})
        with tempfile.TemporaryDirectory() as temp, patch.object(benchmark, "idle"), \
                patch.object(benchmark, "new_session", return_value=Path(temp)), patch.object(benchmark, "source_map", return_value={}), \
                patch.object(benchmark, "audit_run"), patch.object(benchmark, "compare", return_value=report), \
                patch.object(benchmark.subprocess, "run", side_effect=child), redirect_stdout(io.StringIO()):
            benchmark.benchmark(args)
        self.assertEqual(len(commands), 2)
        for command in commands:
            self.assertEqual(command[:3], [sys.executable, "-m", "research_pipeline.v2"])
            for flag, value in (("--workers", "2"), ("--image-cache-mb", "96.0"), ("--pin-memory", "on")):
                self.assertEqual(command[command.index(flag) + 1], value)

    @unittest.skipUnless(os.name == "nt" and shutil.which("powershell"), "Windows launcher check")
    def test_launcher_explicit_interpreter_runs_package_without_training(self):
        script = Path(__file__).resolve().parents[1] / "cli.ps1"
        result = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
                                 "-Python", sys.executable], capture_output=True, encoding="utf-8", errors="replace", timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(sys.executable, result.stdout)
        self.assertIn("benchmark-report", result.stdout)

    def test_loader_flags_are_explicit_and_validated(self):
        defaults = cli.parser().parse_args(["optimize", "--data", "fixture"])
        self.assertEqual((defaults.workers, defaults.image_cache_mb, defaults.pin_memory), ("auto", 64.0, "auto"))
        values = cli.parser().parse_args(["benchmark", "--data", "fixture", "--workers", "0",
                                         "--image-cache-mb", "0", "--pin-memory", "off"])
        self.assertEqual((values.workers, values.image_cache_mb, values.pin_memory), (0, 0.0, "off"))
        for flag, value in (("--workers", "-1"), ("--workers", "2.5"), ("--image-cache-mb", "nan"),
                            ("--image-cache-mb", "inf"), ("--image-cache-mb", "-1")):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.parser().parse_args(["optimize", "--data", "fixture", flag, value])

    def test_benchmark_report_cli_never_starts_training(self):
        with patch.object(benchmark, "finalize_benchmark", return_value={}) as finalize, \
                patch.object(benchmark.subprocess, "run") as train:
            self.assertEqual(cli.main(argv=["benchmark-report", "--run", "fixture"]), 0)
        finalize.assert_called_once_with("fixture", None)
        train.assert_not_called()

    def test_unjournaled_crash_is_not_finalized(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            ledger = Ledger(folder / "events.jsonl")
            ledger.emit("trial_started", attempt_id="unknown-cost")
            write_json(folder / "status.json", {"phase": "search"})
            with patch.object(benchmark, "audit_run") as audit:
                with self.assertRaisesRegex(ValueError, "unknown crash cost"):
                    benchmark._finished_run(folder)
            audit.assert_not_called()

    def fixture(self, folder, complete=True):
        online = folder / "random" / "repeat_17" / "online"
        reference = folder / "reference" / "reference"
        online.mkdir(parents=True)
        reference.mkdir(parents=True)
        sources = {"fixture.py": "mocked-source-not-research"}
        manifest = {"sources": sources, "methods": ["random"], "search_seeds": [17],
                    "external_shared_preparation_seconds": 1, "phase": "locked_before_search"}
        manifest["sha256"] = digest(manifest)
        write_json(folder / "benchmark_protocol.json", manifest)
        decision = {"selected_config_id": "fixture-config"}
        write_json(online / "decision.locked.json", decision)
        write_json(reference / "protocol.json", {})
        boundary = folder / "decisions.locked.json"
        write_json(boundary, {"benchmark_sha256": manifest["sha256"], "runs": [str(online)],
                              "decision_hashes": {str(online): digest(decision)}})
        online_protocol = {"mode": "online", "method": "random", "search_seed": 17, "source_files": sources}
        ref_protocol = {"mode": "offline_reference", "reference_boundary": str(boundary),
                        "reference_boundary_sha256": sha(boundary), "source_files": sources}
        rows = [{"complete": complete}]
        def finished(path):
            path = Path(path)
            return path, online_protocol if path == online else ref_protocol, rows, 1
        report = {"scope": "fixtures, not evidence", "summaries": [], "reference_complete": complete,
                  "reference_cost_seconds": 2, "outcomes": []}
        return online, reference, manifest, finished, report

    def test_report_rebuild_audits_boundary_and_never_trains(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp).resolve()
            online, reference, _, finished, report = self.fixture(folder)
            with patch.object(benchmark, "_finished_run", side_effect=finished) as audits, \
                    patch.object(benchmark, "compare", return_value=report) as compare, \
                    patch.object(benchmark.subprocess, "run") as train, redirect_stdout(io.StringIO()):
                result = benchmark.finalize_benchmark(folder)
            self.assertEqual(audits.call_count, 2)
            compare.assert_called_once_with([str(online)], str(reference))
            train.assert_not_called()
            self.assertTrue(result["report_generation"]["analysis_only"])
            self.assertFalse(result["report_generation"]["training_started"])
            self.assertTrue((folder / "BENCHMARK.json").is_file())

    def test_incomplete_reference_does_not_publish_result(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp).resolve()
            _, _, _, finished, _ = self.fixture(folder, complete=False)
            with patch.object(benchmark, "_finished_run", side_effect=finished), \
                    patch.object(benchmark, "compare") as compare:
                with self.assertRaisesRegex(ValueError, "not completed every"):
                    benchmark.finalize_benchmark(folder)
            compare.assert_not_called()
            self.assertFalse((folder / "BENCHMARK.json").exists())

    def test_changed_online_decision_is_rejected_before_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp).resolve()
            online, _, _, finished, _ = self.fixture(folder)
            write_json(online / "decision.locked.json", {"selected_config_id": "changed"})
            with patch.object(benchmark, "_finished_run", side_effect=finished):
                with self.assertRaisesRegex(ValueError, "decision changed"):
                    benchmark.finalize_benchmark(folder)
            self.assertFalse((folder / "BENCHMARK.json").exists())

    def test_changed_study_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp).resolve()
            _, _, manifest, _, _ = self.fixture(folder)
            write_json(folder / "benchmark_protocol.json", {**manifest, "search_seeds": [99]})
            with self.assertRaisesRegex(ValueError, "checksum"):
                benchmark.finalize_benchmark(folder)

    def test_ambiguous_reference_directory_not_arbitrarily_selected(self):
        with tempfile.TemporaryDirectory() as temp:
            for name in ("one", "two"):
                folder = Path(temp) / name
                folder.mkdir()
                write_json(folder / "protocol.json", {})
            with self.assertRaisesRegex(ValueError, "exactly one"):
                benchmark._single_run(temp)


if __name__ == "__main__":
    unittest.main()
