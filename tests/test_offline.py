import csv
from pathlib import Path
import tempfile
import unittest

from yolo_tuner.offline import CATEGORICAL, NUMERIC, TARGETS, fit, predict


class OfflineTests(unittest.TestCase):
    def test_group_splits_and_ood_rejection(self):
        rows = []
        for dataset in range(40):
            for batch in (1, 2, 4):
                rows.append({"dataset_id": f"dataset_{dataset}", "model": "yolo11n.pt",
                             "precision": "fp16", "optimizer": "AdamW", "gpu": "test-gpu",
                             "runtime_protocol": "test-protocol-v1",
                             "imgsz": 640, "batch": batch, "freeze": 0, "nbs": 64, "lr0": 0.001,
                             "train_images": 100 + dataset, "val_images": 20, "classes": 2,
                             "objects_per_image": 1.0, "small_object_ratio": 0.1, "full_epochs": 100,
                             "final_map50_95": 0.4 + dataset * 0.002,
                             "peak_reserved_mb": 1000 + batch * 100,
                             "full_training_seconds": 100 + dataset})
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, artifact_path = Path(tmp) / "measured.csv", Path(tmp) / "predictor.json"
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=("dataset_id",) + CATEGORICAL + NUMERIC + TARGETS)
                writer.writeheader()
                writer.writerows(rows)
            artifact = fit(csv_path, artifact_path)
            groups = artifact["split_groups"]
            self.assertFalse(set(groups["train"]) & set(groups["calibration"]))
            self.assertFalse(set(groups["train"]) & set(groups["test"]))
            self.assertFalse(set(groups["calibration"]) & set(groups["test"]))
            in_domain = {**rows[20], "train_images": 120}
            result = predict(artifact, in_domain)
            self.assertEqual(result["status"], "in_domain")
            self.assertGreaterEqual(result["predictions"]["final_map50_95"]["lower"], 0)
            self.assertLessEqual(result["predictions"]["final_map50_95"]["upper"], 1)
            self.assertEqual(predict(artifact, {**in_domain, "gpu": "unseen"})["status"], "out_of_domain")
            self.assertEqual(predict(artifact, {**in_domain, "batch": 16})["status"], "out_of_domain")
            self.assertEqual(predict(artifact, {**in_domain, "runtime_protocol": "new-version"})["status"], "out_of_domain")
            self.assertEqual(predict({**artifact, "schema_version": 1}, in_domain)["status"], "out_of_domain")


if __name__ == "__main__":
    unittest.main()
