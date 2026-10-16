from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from yolo_tuner.caption_features import FeatureStore, dataset_signature
from yolo_tuner.core import training_finished
from yolo_tuner.pipeline import candidate_configs, select_presets
from yolo_tuner.vision_models import build


class CaptionV2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def config(self, encoder="cnn"):
        return {"model": "attn_gru_captioner", "width": 16, "hidden": 32,
                "encoder": encoder, "feature_grid": 2, "dropout": 0.0, "imgsz": 32}

    def test_attention_and_image_encoder_receive_gradients(self):
        model = build("captioning", self.config(), 12)
        output = model(torch.rand(1, 3, 32, 32), torch.tensor([[2, 4, 5]]))
        loss = torch.nn.functional.cross_entropy(output.reshape(-1, 12), torch.tensor([4, 5, 3]))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        for module in (model.encoder, model.attention, model.decoder):
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters()))

    def test_generation_step_matches_forward_and_uses_image_features(self):
        model = build("captioning", self.config(), 12).eval()
        features = torch.rand(2, 4, 48)
        with torch.inference_mode():
            logits = model.forward_features(features, torch.full((2, 1), 2))[:, 0].clone()
            logits[:, 0] = logits[:, 2] = -torch.inf
            generated = model.generate_features(features, 2)
            self.assertEqual(generated[:, 0].tolist(), logits.argmax(-1).tolist())
            other = model.forward_features(torch.zeros_like(features), torch.full((2, 1), 2))
            self.assertFalse(torch.allclose(model.forward_features(features, torch.full((2, 1), 2)), other))

    def test_frozen_encoder_keeps_batchnorm_statistics_and_no_gradient(self):
        model = build("captioning", self.config("resnet18"), 12)
        model.train()
        before = {k: v.clone() for k, v in model.encoder.state_dict().items() if "running_mean" in k}
        output = model(torch.rand(1, 3, 32, 32), torch.tensor([[2, 4, 5]]))
        output.sum().backward()
        self.assertFalse(model.encoder.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.encoder.parameters()))
        after = model.encoder.state_dict()
        self.assertTrue(all(torch.equal(value, after[key]) for key, value in before.items()))

    def test_feature_store_checks_identity_and_uses_disk_mmap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            array = np.ones((1, 4, 512), dtype=np.float16)
            np.save(root / "features.npy", array)
            index = {"data": str(root / "features.npy"), "shape": list(array.shape),
                     "images": [{"image": "image.png", "sha256": "original"}]}
            path = root / "index.json"
            path.write_text(json.dumps(index), encoding="utf-8")
            with FeatureStore(path) as store:
                self.assertIsInstance(store.array, np.memmap)
                self.assertEqual(store.get(index["images"][0]).shape, (4, 512))
                with self.assertRaisesRegex(ValueError, "identity"):
                    store.get({"image": "image.png", "sha256": "changed"})

    def test_quality_failed_candidate_is_not_selected_as_accuracy_preset(self):
        bad = {"status": "ok", "resource_feasible": True, "quality_score": 0.9, "quality_eligible": False,
               "estimated_full_seconds": 10, "memory_mb": 100}
        good = {**bad, "quality_score": 0.5, "quality_eligible": True}
        presets, front = select_presets([bad, good])
        self.assertIs(presets["Accuracy"], good)
        self.assertEqual(front, [good])
        self.assertEqual(select_presets([bad]), ({}, []))

    def test_timeout_and_early_stopping_are_not_confused(self):
        row = {"status": "ok", "epochs_completed": 7, "training_completed": True, "early_stopped": True}
        self.assertTrue(training_finished(row, 20))
        self.assertFalse(training_finished({**row, "status": "timeout"}, 20))
        self.assertFalse(training_finished({**row, "early_stopped": False}, 20))
        self.assertTrue(training_finished({"status": "ok", "epochs_completed": 20}, 20))

    def test_portable_caption_candidates_are_v2_not_legacy(self):
        hw = {"device": "cpu", "ram_available_mb": 2000}
        configs = candidate_configs("captioning", hw, {"counts": {"train": 4}})
        self.assertTrue(all(c["model"] == "attn_gru_captioner" for c in configs))
        self.assertTrue(all(c["precision"] == "fp32" and c["batch"] <= 4 for c in configs))

    def test_resume_signature_ignores_relocation_and_test_but_checks_targets(self):
        def manifest(path, caption="một người"):
            return {"splits": {"train": [{"image": path, "sha256": "same image", "captions": [caption]}],
                    "val": [{"image": "val.png", "sha256": "val image", "captions": ["xin chào"]}],
                    "test": [{"image": path, "sha256": "anything", "captions": ["test"]}]}}
        self.assertEqual(dataset_signature(manifest("old.png")), dataset_signature(manifest("new.png")))
        self.assertNotEqual(dataset_signature(manifest("old.png")), dataset_signature(manifest("old.png", "khác")))


if __name__ == "__main__":
    unittest.main()
