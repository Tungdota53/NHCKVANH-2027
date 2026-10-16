import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from yolo_tuner.captioning import CaptionPredictor, decode_ids, evaluate, normalize, score_captions, validate_holdout, quality_assessment
from yolo_tuner.vision_models import Captioner


class GenerationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_generation_stops_at_eos_without_reference_tokens(self):
        model = Captioner(6, width=16, hidden=32).eval()
        with torch.no_grad():
            model.output.weight.zero_()
            model.output.bias.fill_(-100)
            model.output.bias[3] = 100
        generated = model.generate(torch.rand(2, 3, 32, 32), max_tokens=10)
        self.assertEqual(generated.tolist(), [[3], [3]])

    def test_generation_masks_bos_pad_and_respects_length_cap(self):
        model = Captioner(6, width=16, hidden=32).eval()
        with torch.no_grad():
            model.output.weight.zero_()
            model.output.bias.copy_(torch.tensor([1000., -100., 1000., -100., 10., -100.]))
        self.assertEqual(model.generate(torch.rand(1, 3, 32, 32), 5).tolist(), [[4] * 4])

    def test_decode_stops_at_eos_and_preserves_unknown(self):
        vocabulary = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "bóng": 4}
        self.assertEqual(decode_ids([2, 4, 1, 3, 4], vocabulary), "bóng <unk>")
        self.assertEqual(normalize("Một người, đá bóng!"), "một người , đá bóng !")

    def test_predict_loads_checkpoint_and_real_image_on_cpu(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image = folder / "image.png"
            Image.new("RGB", (20, 10), "red").save(image)
            model = Captioner(6, width=16, hidden=32)
            with torch.no_grad():
                model.output.weight.zero_()
                model.output.bias.fill_(-100)
                model.output.bias[4] = 100
            checkpoint = folder / "best.pt"
            torch.save({"task": "captioning", "model": model.state_dict(), "config": {"width": 16, "hidden": 32, "imgsz": 32},
                        "vocabulary": {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "bóng": 4, "đá": 5},
                        "max_tokens": 5, "epoch": 2}, checkpoint)
            predictor = CaptionPredictor(checkpoint, "cpu")
            self.assertEqual(predictor.predict([image]), ["bóng bóng bóng bóng"])
            self.assertEqual(predictor.predict([]), [])

    def test_overlapping_holdout_is_rejected(self):
        for record in ({"image": "train.png", "sha256": "different"},
                       {"image": "copy.png", "sha256": "same"}):
            manifest = {"splits": {"train": [{"image": "train.png", "sha256": "same"}], "val": [], "test": [record]}}
            with self.assertRaisesRegex(ValueError, "overlaps"):
                validate_holdout(manifest, "test")

    def test_quality_gate_does_not_accept_collapse_or_below_baseline(self):
        baseline = {"bleu4": 20, "chrf": 30}
        low = {"bleu4": 10, "chrf": 40}
        high = {"bleu4": 25, "chrf": 40}
        self.assertEqual(quality_assessment(low, baseline, ["a", "b"])["status"], "quality_not_established")
        self.assertTrue(quality_assessment(high, baseline, ["a", "a"])["single_caption_collapse"])
        good = quality_assessment(high, baseline, ["a", "b"])
        self.assertEqual(good["status"], "exceeds_text_only_baseline_not_human_verified")
        self.assertFalse(good["human_semantic_accuracy_verified"])


@unittest.skipUnless(importlib.util.find_spec("sacrebleu"), "Install the evaluation extra to test scoring")
class CaptionMetricTests(unittest.TestCase):
    def test_end_to_end_evaluation_saves_predictions_and_quality_status(self):
        from PIL import Image
        from yolo_tuner.captioning import predict
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image = folder / "test.png"
            Image.new("RGB", (16, 16), "red").save(image)
            config = {"width": 16, "hidden": 32, "imgsz": 32}
            model = Captioner(5, width=16, hidden=32)
            with torch.no_grad():
                model.output.weight.zero_()
                model.output.bias.fill_(-100)
                model.output.bias[4] = 100
            checkpoint = folder / "best.pt"
            torch.save({"task": "captioning", "model": model.state_dict(), "config": config,
                        "vocabulary": {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "bóng": 4},
                        "max_tokens": 5, "epoch": 2}, checkpoint)
            manifest = {"task": "captioning", "splits": {"train": [{"image": str(folder / "train.png"),
                        "captions": ["bóng bóng bóng bóng"], "source": "train"}], "val": [],
                        "test": [{"image": str(image), "captions": ["bóng bóng bóng bóng"], "source": "test"}]}}
            manifest_path = folder / "dataset.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            args = SimpleNamespace(batch=1, manifest=manifest_path, split="test", checkpoint=checkpoint,
                                   device="cpu", output=folder / "evaluation", image=[image])
            with redirect_stdout(io.StringIO()):
                self.assertEqual(predict(args), 0)
                self.assertEqual(evaluate(args), 3)  # Exactly ties the text-only baseline.
            result_path = next((folder / "evaluation").glob("*/evaluation.json"))
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["images"], 1)
            self.assertAlmostEqual(result["metrics"]["bleu4"], 100)
            self.assertEqual(result["quality_assessment"]["status"], "quality_not_established")
            self.assertFalse(result["checkpoint_training_provenance_verified"])
            self.assertTrue((result_path.parent / "predictions.json").exists())
            self.assertTrue((result_path.parent / "EVALUATION.md").exists())

    def test_exact_multi_reference_score_and_variable_reference_counts(self):
        predictions = ["một người đang đá bóng", "một con chó đang chạy"]
        refs = [["một người đang đá bóng", "người này đang đá bóng"], ["một con chó đang chạy"]]
        score = score_captions(predictions, refs)
        self.assertAlmostEqual(score["bleu4"], 100)
        self.assertAlmostEqual(score["chrf"], 100)
        self.assertEqual(score["exact_reference_match_rate"], 1)
        self.assertIn("nrefs:var", score["bleu4_signature"])

    def test_empty_predictions_are_zero_not_fake_accuracy(self):
        score = score_captions([""], [["một người đang đá bóng"]])
        self.assertEqual(score["bleu4"], 0)
        self.assertEqual(score["empty_caption_rate"], 1)

    def test_invalid_reference_alignment_rejected(self):
        for hypotheses, refs in (([], []), (["a"], []), (["a"], [[]])):
            with self.assertRaises(ValueError):
                score_captions(hypotheses, refs)

    def test_missing_test_is_not_replaced_with_train(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "dataset.json"
            path.write_text(json.dumps({"task": "captioning", "splits": {"test": []}}), encoding="utf-8")
            args = SimpleNamespace(batch=1, manifest=path, split="test", checkpoint="missing.pt")
            with self.assertRaisesRegex(ValueError, "No images"):
                evaluate(args)


if __name__ == "__main__":
    unittest.main()
