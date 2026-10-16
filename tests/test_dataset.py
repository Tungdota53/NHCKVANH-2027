from pathlib import Path
import tempfile
import unittest

from yolo_tuner.dataset import annotations, label_path, prepare, sample


class DatasetTests(unittest.TestCase):
    def test_multilabel_small_and_negative_coverage(self):
        images = [Path(str(i)) for i in range(6)]
        features = {images[0]: ({0, 1}, 2, 0), images[1]: ({0}, 1, 1),
                    images[2]: (set(), 0, 0), **{p: ({0}, 1, 0) for p in images[3:]}}
        subset = sample(images, features, 3, 17)
        self.assertEqual(set(subset), set(images[:3]))
        self.assertEqual(subset, sample(images, features, 3, 17))

    def test_invalid_label_and_split_overlap_rejected(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            image = root / "images" / "a.jpg"
            image.write_bytes(b"test")
            label_path(image).write_text("0 0.5 0.5 0.1 0.1", encoding="utf-8")
            self.assertEqual(annotations(image, 1), ({0}, 1, 0))
            spec = root / "data.yaml"
            spec.write_text(yaml.safe_dump({"path": str(root), "train": "images", "val": "images", "names": ["a"]}))
            with self.assertRaisesRegex(ValueError, "overlap"):
                prepare(spec, root / "output")
            label_path(image).write_text("1 0.5 0.5 0.1 0.1")
            with self.assertRaisesRegex(ValueError, "class ID"):
                annotations(image, 1)


if __name__ == "__main__":
    unittest.main()
