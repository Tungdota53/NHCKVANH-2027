import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from yolo_tuner import cli_ui
from yolo_tuner.cli import announce, parser, show_report


class CLIUITests(unittest.TestCase):
    def test_duration_is_human_readable(self):
        self.assertEqual(cli_ui.duration(9), "9.0 giây")
        self.assertEqual(cli_ui.duration(90), "1.5 phút")
        self.assertEqual(cli_ui.duration(7200), "2.0 giờ")

    def test_plan_explains_budget_and_rungs(self):
        output = io.StringIO()
        args = SimpleNamespace(rungs=[1, 2, 4], budget=20, preset="Balanced")
        with redirect_stdout(output):
            cli_ui.search_plan(args, [1, 2])
        text = output.getvalue()
        self.assertIn("2 cấu hình", text)
        self.assertIn("1 → 2 → 4", text)
        self.assertIn("20.0 phút", text)

    def test_reused_baseline_is_not_printed_twice(self):
        output = io.StringIO()
        with redirect_stdout(output):
            announce({"reused_baseline": True})
        self.assertEqual(output.getvalue(), "")

    def test_empty_results_table_does_not_crash(self):
        with redirect_stdout(io.StringIO()):
            cli_ui.results({"presets": {}})

    def capture_pipeline(self, task="captioning", measured=True, metric=True):
        config = {"model": "cnn_gru_captioner", "imgsz": 160, "batch": 4, "width": 32, "lr0": 0.001}
        row = {"config": config, "estimated_full_seconds": 90, "memory_mb": 128}
        if metric:
            row.update(validation_loss=2.95, quality_score=0.75)
        report = {"hardware": {"device": "cpu"}, "presets": {"Balanced": row} if measured else {},
                  "history": [row] if measured else [], "search_wall_seconds": 30, "budget_seconds": 60}
        selection = {"config": config, "task": task, "device": "cpu", "preset": "Balanced",
                     "status": "measured_best_observed" if measured else "recommended_unverified"}
        output = io.StringIO()
        with redirect_stdout(output):
            cli_ui.pipeline_results(report, selection, Path("runs/test_ui"))
        return output.getvalue()

    def test_caption_report_displays_loss_not_map(self):
        text = self.capture_pipeline()
        self.assertIn("Val CE ↓", text)
        self.assertIn("2.9500", text)
        self.assertIn("Peak RAM", text)
        self.assertIn("CẤU HÌNH ĐÃ CHỌN", text)
        self.assertIn("--selection", text)
        self.assertNotIn("mAP", text)

    def test_classification_report_displays_accuracy(self):
        text = self.capture_pipeline(task="classification")
        self.assertIn("Val accuracy ↑", text)
        self.assertIn("75.00%", text)
        self.assertNotIn("2.9500", text)

    def test_unmeasured_is_not_printed_as_success(self):
        text = self.capture_pipeline(measured=False)
        self.assertIn("CHƯA KIỂM CHỨNG", text)
        self.assertNotIn("Đã chọn", text)
        self.assertNotIn("0.0000", text)

    def test_missing_metric_does_not_invent_zero(self):
        self.assertNotIn("0.0000", self.capture_pipeline(metric=False))

    def test_trial_includes_progress_and_batch_probe_throughput(self):
        output = io.StringIO()
        config = {"model": "captioner", "imgsz": 160, "batch": 8, "width": 32, "lr0": 0.001}
        with redirect_stdout(output):
            cli_ui.search_rung(2, 3)
            cli_ui.trial_start(config, 2, 1, 3, 90)
            cli_ui.trial({"config": config, "status": "ok", "mode": "search", "batch_verification": True,
                          "validation_loss": 2.9, "epochs_completed": 2, "throughput": 150})
        text = output.getvalue()
        self.assertIn("VÒNG 2 EPOCH", text)
        self.assertIn("[1/3]", text)
        self.assertIn("còn 1.5 phút", text)
        self.assertIn("Đo batch", text)
        self.assertIn("150.0 mẫu/giây", text)
        self.assertNotIn("mAP", text)

    def test_show_reads_saved_report_without_training(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            report = {"dataset": {"task": "captioning", "counts": {"train": 10, "val": 2, "test": 1},
                                  "validation_ready": True}, "hardware": {"device": "cpu"}}
            selection = {"schema_version": 2, "task": "captioning", "config": {"batch": 2},
                         "status": "recommended_unverified", "trainable": False}
            (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")
            (folder / "selected.json").write_text(json.dumps(selection), encoding="utf-8")
            for path in (folder, folder / "report.json"):
                args = parser().parse_args(["show", "--report", str(path)])
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(show_report(args), 0)
                text = output.getvalue()
                self.assertIn("không chạy lại", text)
                self.assertIn("captioning", text)
                self.assertIn("CHƯA KIỂM CHỨNG", text)
                self.assertNotIn("LỆNH FULL TRAINING", text)


if __name__ == "__main__":
    unittest.main()
