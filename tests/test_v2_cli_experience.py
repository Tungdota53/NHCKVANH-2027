"""UX/control-flow tests. Fixtures are not research benchmark evidence."""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from research_pipeline.v2 import cli, terminal as ui, wizard


class CLIExperienceTests(unittest.TestCase):
    def capture(self, func, *args, **kwargs):
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            value = func(*args, **kwargs)
        return value, output.getvalue()

    def test_no_arguments_noninteractive_help_never_training(self):
        with patch.object(ui, "interactive", return_value=False), patch.object(cli, "legacy_main") as legacy:
            code, text = self.capture(cli.main, argv=[])
        self.assertEqual(code, 0)
        self.assertIn("wizard", text)
        self.assertIn("optimize", text)
        legacy.assert_not_called()

    def test_no_arguments_tty_opens_menu(self):
        with patch.object(ui, "interactive", return_value=True), patch.object(wizard, "menu", return_value=0) as menu:
            self.assertEqual(cli.main(argv=[]), 0)
        menu.assert_called_once()

    def test_explicit_menu_refuses_redirected_input(self):
        with patch.object(ui, "interactive", return_value=False), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                cli.main(argv=["menu"])
        self.assertEqual(raised.exception.code, 2)

    def test_typo_is_not_misread_as_dataset(self):
        with patch.object(cli, "legacy_main") as legacy, redirect_stderr(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as raised:
                cli.main(argv=["optimze"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("optimize", output.getvalue())
        legacy.assert_not_called()

    def test_legacy_routing_restores_argv_even_if_legacy_mutates(self):
        before = sys.argv[:]
        def old_main():
            self.assertEqual(sys.argv[1], "show-training")
            sys.argv.append("legacy-mutation")
            return 0
        with patch("research_pipeline.cli.main", side_effect=old_main):
            self.assertEqual(cli.main(argv=["show-training", "--run", "old"]), 0)
        self.assertEqual(sys.argv, before)

    def test_status_json_read_only_and_explicitly_unaudited(self):
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "status.json").write_text('{"phase":"prepared"}', encoding="utf-8")
            with patch.object(cli, "status") as human:
                _, text = self.capture(cli.main, argv=["status", "--run", temp, "--json"])
            result = json.loads(text)
        self.assertFalse(result["evidence_audited"])
        self.assertFalse(result["live_process_checked"])
        human.assert_not_called()

    def test_help_keeps_protocol_and_advanced_flags(self):
        p = cli.parser()
        parsed = p.parse_args(["optimize", "--protocol", "plan.json"])
        self.assertIsNone(parsed.data)
        self.assertEqual(parsed.seeds, [17, 29, 43])
        self.assertEqual(p.parse_args(["optimize", "--data", "a", "--weights", ".5", ".3", ".2"]).weights, [.5, .3, .2])

    def test_user_input_time_not_charged_to_search_budget(self):
        def fake_menu(dispatch, *args, **kwargs):
            return dispatch(["optimize", "--data", "fixture"])
        with patch.object(ui, "interactive", return_value=True), patch.object(wizard, "menu", side_effect=fake_menu), \
             patch.object(cli.time, "monotonic", return_value=1000), \
             patch("research_pipeline.v2.workflow.prepare", return_value=(Path("fixture"), {}, Mock())) as prepare, \
             patch("research_pipeline.v2.workflow.execute"):
            self.assertEqual(cli.main(origin=0, argv=["menu"]), 0)
        self.assertEqual(prepare.call_args.args[1], 1000)

    def test_doctor_returns_failure_if_runtime_unavailable(self):
        with patch("yolo_tuner.pipeline.runtime_info", return_value={"device": "cpu", "ml_runtime_ready": False}):
            code, text = self.capture(cli.main, argv=["doctor"])
        self.assertEqual(code, 2)
        self.assertIn("CHƯA SẴN SÀNG", text)

    def test_debug_preserves_traceback(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                cli.main(argv=["status", "--run", temp, "--debug"])

    def test_source_change_warning_is_read_only(self):
        changes = ui.source_changes({"source_files": {"train_config.py": "not-real-sha"}})
        self.assertEqual(changes, ["train_config.py"])

    def test_cancel_menu_does_not_dispatch(self):
        dispatch = Mock()
        with patch("builtins.input", return_value="0"):
            code, _ = self.capture(wizard.menu, dispatch)
        self.assertEqual(code, 0)
        dispatch.assert_not_called()

    def test_menu_eof_no_job(self):
        dispatch = Mock()
        with patch("builtins.input", side_effect=EOFError):
            code, _ = self.capture(wizard.menu, dispatch)
        self.assertEqual(code, 0)
        dispatch.assert_not_called()

    def test_menu_keyboard_cancel_returns_130(self):
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            code, _ = self.capture(wizard.menu, Mock())
        self.assertEqual(code, 130)

    def test_back_to_menu(self):
        with patch("builtins.input", side_effect=["b", "0"]):
            code, text = self.capture(wizard.menu, Mock())
        self.assertEqual(code, 0)
        self.assertIn("quay lại", text)

    def test_numeric_rejects_invalid_values_accepts_decimal_comma(self):
        with patch("builtins.input", side_effect=["abc", "nan", "inf", "-1", "0", "1,5"]):
            result, _ = self.capture(wizard.numeric, "Budget", 60)
        self.assertEqual(result, 1.5)

    def test_optional_numeric_blank_is_none_not_zero(self):
        with patch("builtins.input", return_value=""):
            self.assertIsNone(wizard.numeric("Limit", None, optional=True))

    def test_quoted_path_and_retry(self):
        with tempfile.TemporaryDirectory(prefix="CLI space ") as temp:
            with patch("builtins.input", side_effect=["does-not-exist-cli-fixture", f'"{temp}"']):
                result, _ = self.capture(wizard.path_input, "Dataset", directory=True)
        self.assertEqual(result, str(Path(temp).resolve()))

    def test_confirm_defaults_to_no(self):
        with patch("builtins.input", return_value=""):
            value, _ = self.capture(wizard.confirm, ["optimize", "--data", "a"], note="Training")
        self.assertFalse(value)

    def inspection(self):
        return "real-path-fixture", "cpu", {"task": "classification", "validation_ready": True}, {"ml_runtime_ready": True}

    def test_wizard_cancel_before_prepare(self):
        dispatch = Mock()
        with patch.object(wizard, "inspect_dataset", return_value=self.inspection()), patch("builtins.input", side_effect=["30", "8", "2", "1", "n", "0"]):
            self.capture(wizard.optimize_wizard, dispatch)
        dispatch.assert_not_called()

    def test_wizard_requires_second_confirmation(self):
        dispatch = Mock()
        with patch.object(wizard, "inspect_dataset", return_value=self.inspection()), patch("builtins.input", side_effect=["30", "8", "2", "1", "n", "1", "n"]):
            self.capture(wizard.optimize_wizard, dispatch)
        dispatch.assert_not_called()

    def test_wizard_plan_uses_seconds_and_all_seeds(self):
        dispatch = Mock()
        with patch.object(wizard, "inspect_dataset", return_value=self.inspection()), patch("builtins.input", side_effect=["30", "8", "2", "1", "n", "2", "y"]):
            self.capture(wizard.optimize_wizard, dispatch)
        args = cli.parser().parse_args(dispatch.call_args.args[0])
        self.assertEqual(args.command, "plan-search")
        self.assertEqual(args.budget_seconds, 1800)
        self.assertEqual(args.full_epochs, 8)
        self.assertEqual(args.seeds, [17, 29, 43])
        self.assertEqual(args.space, "compact")

    def test_unlabeled_dataset_cannot_start(self):
        dispatch = Mock()
        data = ("path", "cpu", {"task": "reconstruction", "validation_ready": False}, {"ml_runtime_ready": True})
        with patch.object(wizard, "inspect_dataset", return_value=data):
            self.capture(wizard.optimize_wizard, dispatch)
        dispatch.assert_not_called()

    def test_inspection_has_no_fake_vram_on_cpu(self):
        _, text = self.capture(ui.inspection_view, {"hardware": {"device": "cpu", "ml_runtime_ready": True, "ram_available_mb": 100}})
        self.assertIn("Không áp dụng", text)
        self.assertNotIn("MiB", next(line for line in text.splitlines() if "VRAM" in line))

    def test_readable_width_and_control_sanitization(self):
        self.assertEqual(ui.width("a\u0301"), 1)
        self.assertEqual(ui.width("界"), 2)
        self.assertLessEqual(ui.width(ui.clip("cấu hình " * 10, 15)), 15)
        _, text = self.capture(ui.say, "bad\x1b[31m\rtext")
        self.assertNotIn("\x1b", text)
        self.assertNotIn("\r", text)

    def test_ascii_fallback_does_not_crash(self):
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="ascii")
        ui.say("Đường dẫn dữ liệu", file=stream)
        stream.flush()
        self.assertIn(b"Duong", raw.getvalue())
        stream.detach()

    def test_duration_none_not_zero(self):
        self.assertEqual(ui.duration(None), "chưa đo")
        self.assertIn("phút", ui.duration(120))
        self.assertIn("giờ", ui.duration(3600))

    def test_powershell_command_quotes_apostrophe_and_dollar(self):
        with patch.object(ui.os, "name", "nt"):
            result = ui.command(["inspect", "--data", "D:/user's/$dataset"])
        self.assertIn("'D:/user''s/$dataset'", result)
        self.assertTrue(result.startswith("& "))

    def test_inventory_distinguishes_deployment_and_skips_worker_protocols(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, mode in (("search", "online"), ("deployment", "deployment"), ("search/trials/hidden", "online")):
                folder = root / name
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "protocol.json").write_text(json.dumps({"adapter": "research_v2", "mode": mode, "task": "classification"}), encoding="utf-8")
            entries = ui.inventory(temp)
        self.assertEqual(len(entries), 2)
        self.assertEqual({r["kind"] for r in entries}, {"v2", "v2 training"})

    def test_wait_worker_updates_without_extending_deadline(self):
        process, progress = Mock(), Mock()
        process.wait.side_effect = [subprocess.TimeoutExpired("fixture", 1), 0]
        with patch.object(ui.time, "monotonic", side_effect=[10, 10.1, 11.2]):
            ui.wait_worker(process, 2, progress)
        self.assertEqual(progress.update.call_count, 2)
        self.assertAlmostEqual(process.wait.call_args.kwargs["timeout"], .8)

    def test_wait_worker_hard_deadline(self):
        process = Mock(args=["fixture"])
        with patch.object(ui.time, "monotonic", side_effect=[10, 13]):
            with self.assertRaises(subprocess.TimeoutExpired):
                ui.wait_worker(process, 2)
        process.wait.assert_not_called()

    def test_progress_handles_partial_json(self):
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "result.json").write_text('{"unfinished":', encoding="utf-8")
            with redirect_stdout(io.StringIO()) as output:
                progress = ui.TrialProgress(temp, 0, 17, 2, True, {"model": "cnn", "imgsz": 32, "batch": 4, "precision": "fp32"}, 60)
                progress.update()
        self.assertIn("epoch 0/2", output.getvalue())
        self.assertIn("FULL", output.getvalue())


if __name__ == "__main__":
    unittest.main()
