import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import psutil

from yolo_tuner.completion import complete


class CompletionTests(unittest.TestCase):
    def test_failed_training_does_not_evaluate_or_claim_quality(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "dataset.json").write_text("{}", encoding="utf-8")
            report = {"selection": {"full_epochs": 20}, "attempts": [{"status": "timeout", "epochs_completed": 1}]}
            (folder / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
            with patch("psutil.Process", side_effect=psutil.NoSuchProcess(1)), patch("yolo_tuner.captioning.evaluate") as evaluation:
                with self.assertRaisesRegex(ValueError, "did not complete"):
                    complete(folder, 1)
                evaluation.assert_not_called()
            status = json.loads((folder / "completion_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "failed")

    def test_evaluated_but_below_baseline_is_not_reported_as_passed(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "dataset.json").write_text("{}", encoding="utf-8")
            report = {"selection": {"full_epochs": 20}, "attempts": [{"status": "ok", "epochs_completed": 20,
                      "last_checkpoint": str(folder / "trials" / "trial_0001" / "last.pt")}]}
            (folder / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
            def evaluate(args):
                result = Path(args.output) / "done"
                result.mkdir(parents=True)
                (result / "evaluation.json").write_text("{}", encoding="utf-8")
                return 3
            with patch("psutil.Process", side_effect=psutil.NoSuchProcess(1)), patch("yolo_tuner.captioning.evaluate", side_effect=evaluate):
                self.assertEqual(complete(folder, 1), 3)
            status = json.loads((folder / "completion_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "evaluated")
            self.assertFalse(status["quality_gate_passed"])
            self.assertFalse(status["human_accuracy_verified"])


if __name__ == "__main__":
    unittest.main()
