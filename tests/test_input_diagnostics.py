"""No model training: distinguish image annotations from ordinary tabular CSV."""

from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from research_pipeline.v2 import cli, wizard, terminal
from research_pipeline.v2.data import discover
from research_pipeline.v2.input_diagnostics import DatasetInputError, csv_header


class InputDiagnosticsTests(unittest.TestCase):
    def table(self, root, delimiter="\t"):
        path = Path(root) / "marketing_campaign.csv"
        path.write_text(delimiter.join(["ID", "Income", "Education", "Response"]) + "\nPRIVATE_ROW,DO_NOT_DISPLAY\n", encoding="utf-8")
        return path

    def test_csv_header_sniffs_tab_comma_semicolon(self):
        for delimiter in ("\t", ",", ";"):
            with self.subTest(delimiter=delimiter), tempfile.TemporaryDirectory() as temp:
                signature = csv_header(self.table(temp, delimiter))
                self.assertEqual(signature["column_count"], 4)
                self.assertEqual(signature["delimiter"], delimiter)
                self.assertNotIn("PRIVATE_ROW", str(signature))

    def test_table_directory_gives_actionable_error(self):
        with tempfile.TemporaryDirectory() as temp:
            self.table(temp)
            with self.assertRaises(DatasetInputError) as raised:
                discover(temp)
        error = raised.exception
        self.assertEqual(error.kind, "tabular_not_supported")
        self.assertIn("marketing_campaign.csv", str(error))
        self.assertIn("tabular", error.hint)
        self.assertIn("cột mục tiêu", error.hint)

    def test_explicit_table_never_falls_back_to_unrelated_images(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.table(temp)
            with patch("yolo_tuner.adapters.discover") as images:
                with self.assertRaises(DatasetInputError):
                    discover(path)
        images.assert_not_called()

    def test_caption_csv_without_images_not_called_tabular(self):
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "captions.csv").write_text("image,caption\nmissing.jpg,a cat\n", encoding="utf-8")
            with self.assertRaises(DatasetInputError) as raised:
                discover(temp)
        self.assertEqual(raised.exception.kind, "missing_referenced_images")

    def test_empty_folder_has_specific_image_guidance(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(DatasetInputError) as raised:
                discover(temp)
        self.assertEqual(raised.exception.kind, "no_local_images")
        self.assertIn("ảnh", raised.exception.hint)

    def test_archive_file_not_reinterpreted_as_images(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp, "dataset.zip")
            path.write_bytes(b"fixture-not-real-archive")
            with patch("yolo_tuner.adapters.discover") as images:
                with self.assertRaises(DatasetInputError) as raised:
                    discover(path)
        self.assertEqual(raised.exception.kind, "compressed_input")
        images.assert_not_called()

    def test_archive_directory_guidance_no_automatic_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "data.zip").write_bytes(b"fixture")
            with self.assertRaises(DatasetInputError) as raised:
                discover(temp)
            self.assertEqual([p.name for p in Path(temp).iterdir()], ["data.zip"])
        self.assertEqual(raised.exception.kind, "compressed_input")

    def test_bounded_and_empty_headers(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp, "huge.csv")
            path.write_text("a" * 70000 + ",Response\n", encoding="utf-8")
            self.assertIsNone(csv_header(path))
            path.write_text("", encoding="utf-8")
            self.assertIsNone(csv_header(path))

    def test_wizard_retry_stays_at_dataset_step_without_training(self):
        issue = DatasetInputError("Table", "Needs target", kind="tabular_not_supported")
        data, hw = {"task": "classification", "counts": {}, "validation_ready": True}, {"ml_runtime_ready": True, "device": "cpu"}
        with patch.object(wizard, "path_input", side_effect=["wrong", "right"]), \
             patch("builtins.input", side_effect=["auto", "1", ""]), \
             patch("research_pipeline.v2.data.discover", side_effect=[issue, data]) as read, \
             patch("yolo_tuner.pipeline.runtime_info", return_value=hw) as runtime, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = wizard.inspect_dataset()
        self.assertEqual(result[0], "right")
        self.assertEqual(read.call_count, 2)
        runtime.assert_called_once_with("auto")

    def test_bad_dataset_back_never_checks_gpu_or_dispatches(self):
        issue = DatasetInputError("Table", "Needs target", kind="tabular_not_supported")
        dispatch = Mock()
        with patch.object(wizard, "path_input", return_value="table"), \
             patch("builtins.input", side_effect=["auto", "0"]), \
             patch("research_pipeline.v2.data.discover", side_effect=issue), \
             patch("yolo_tuner.pipeline.runtime_info") as runtime, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(wizard.Back):
                wizard.optimize_wizard(dispatch)
        runtime.assert_not_called()
        dispatch.assert_not_called()

    def test_inspect_invalid_dataset_does_not_initialize_gpu(self):
        with tempfile.TemporaryDirectory() as temp:
            self.table(temp)
            with patch("yolo_tuner.pipeline.runtime_info") as runtime, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    cli.main(argv=["inspect", "--data", temp])
        self.assertEqual(raised.exception.code, 2)
        runtime.assert_not_called()

    def test_structured_hint_used_not_generic_help(self):
        error = DatasetInputError("Table", "Cần task và cột mục tiêu", kind="tabular_not_supported")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as output:
            terminal.error_view(error)
        self.assertIn("Cần task và cột mục tiêu", output.getvalue())
        self.assertNotIn("Dùng --help", output.getvalue())

    def test_activation_command_is_only_input_never_executed(self):
        with patch("builtins.input", side_effect=['(Set-ExecutionPolicy ...); (& "Activate.ps1")', "0"]), redirect_stdout(io.StringIO()) as output:
            value = wizard.choice("Menu", {"0", "1"}, "1")
        self.assertEqual(value, "0")
        self.assertIn("không phải lựa chọn menu", output.getvalue())


if __name__ == "__main__":
    unittest.main()
