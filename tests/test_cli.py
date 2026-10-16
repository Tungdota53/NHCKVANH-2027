from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from yolo_tuner.cli import parser, tune


class FakeRunner:
    """Deterministic measurements to exercise CLI artifact/constraint semantics."""
    def __init__(self, *args, **kwargs):
        pass

    def run(self, config, mode, seconds, epochs=1, resume=None):
        base = {"config": asdict(config), "status": "ok", "worker_wall_seconds": 0.01,
                "peak_reserved_mb": 0, "actual_batch": config.batch}
        if mode == "profile":
            return {**base, "images_per_second_probe": 100}
        return {**base, "epochs_completed": epochs, "map50_proxy": 0.5,
                "map50_95_proxy": 0.25, "estimated_full_seconds": 4000}


class CLITests(unittest.TestCase):
    def fixture(self, root):
        import yaml
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            for i in range(2):
                (root / "images" / split / f"{i}.jpg").write_bytes(b"fixture")
                (root / "labels" / split / f"{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        data = root / "data.yaml"
        data.write_text(yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val", "names": ["object"]}))
        return data

    def exercise(self, extra):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self.fixture(root)
            args = parser().parse_args(["tune", "--data", str(data), "--models", "fake.pt",
                                        "--device", "cpu", "--imgsz", "128", "--freeze", "0",
                                        "--max-batch", "1", "--full-epochs", "4", "--rungs", "1", "2",
                                        "--output", str(root / "runs")] + extra)
            with patch("yolo_tuner.cli.hardware", return_value={"fp16": False, "bf16": False, "gpu": None}), \
                 patch("yolo_tuner.cli.Runner", FakeRunner), patch("builtins.print"):
                code = tune(args)
            report_path = next((root / "runs").glob("*/report.json"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            return code, report, (report_path.parent / "selected.json").exists(), (report_path.parent / "train.yaml").exists()

    def test_exports_selection_with_all_presets(self):
        code, report, selection, training_yaml = self.exercise([])
        self.assertEqual(code, 0)
        self.assertTrue(selection and training_yaml)
        self.assertEqual(set(report["presets"]), {"Accuracy", "Time", "Memory", "Balanced"})
        self.assertEqual(report["presets"]["Balanced"]["epochs_completed"], 2)

    def test_final_accuracy_constraint_never_uses_early_map(self):
        code, report, selection, training_yaml = self.exercise(["--min-accuracy", "0.4"])
        self.assertEqual(code, 2)
        self.assertFalse(selection or training_yaml)
        self.assertEqual(report["accuracy_constraint_status"], "unverified_without_calibrated_final_accuracy_predictor")


if __name__ == "__main__":
    unittest.main()
