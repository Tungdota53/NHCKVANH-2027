"""No-training checks for bounded decoding, exact samples and spawn loading."""

import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from PIL import Image

from research_pipeline.v2.loaders import (
    DecodedImageCache, ImageRecords, detection_loader_builder, make_loader,
    observed_loaders, resolve_policy,
)


class PolicyTests(unittest.TestCase):
    def test_conservative_auto_and_cuda_pinning(self):
        for device in ("cpu", "mps", "0", "cuda:0"):
            p = resolve_policy({}, device, 128, 16)
            self.assertEqual(p["workers"], 0)
            self.assertEqual(p["pin_memory"], device not in {"cpu", "mps"})
            self.assertEqual(p["cache_bytes_per_instance"], 32 * 2**20)

    def test_large_cuda_auto_is_capped(self):
        p = resolve_policy({}, "0", 10000, 64)
        self.assertEqual(p["workers"], 2)
        self.assertLessEqual(p["cache_bytes_per_instance"] * 6, 64 * 2**20)
        self.assertEqual(resolve_policy({}, "cpu", 10000, 64)["workers"], 0)

    def test_explicit_policy_and_cache_zero(self):
        p = resolve_policy({"workers": 1, "cache_mb": 0, "pin_memory": "off"}, "0", 20, 8)
        self.assertEqual(p["workers"], 1)
        self.assertFalse(p["pin_memory"])
        self.assertEqual(p["cache_bytes_per_instance"], 0)

    def test_cache_cap_preserves_requested_budget(self):
        p = resolve_policy({"cache_mb": 64, "workers": 2}, "0", 10000, 16, available_ram_mb=200)
        self.assertEqual(p["cache_requested_mb"], 64)
        self.assertEqual(p["cache_mb"], 20)
        self.assertLessEqual(p["cache_bytes_per_instance"] * 6, 20 * 2**20)

    def test_invalid_settings(self):
        for key, value in (("workers", True), ("workers", -1), ("workers", 9),
                           ("workers", "2"), ("cache_mb", -1), ("cache_mb", float("nan")),
                           ("cache_mb", True), ("pin_memory", 1), ("pin_memory", "yes")):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                resolve_policy({key: value}, "cpu", 100)

    def test_detection_caps_val_workers_and_pinning(self):
        policy = resolve_policy({"workers": 2, "pin_memory": "off"}, "0", 100)
        original = Mock(return_value="loader")
        builder = detection_loader_builder(original, policy)
        self.assertEqual(builder("dataset", batch=8, workers=4, shuffle=False), "loader")
        original.assert_called_once_with("dataset", batch=8, workers=2, shuffle=False, pin_memory=False)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = {"imgsz": 16, "encoder": "tiny"}
        self.records = []
        for index in range(6):
            path = Path(self.temp.name) / f"{index}.png"
            Image.new("RGB", (24, 20), (index * 30, 10, 20)).save(path)
            self.records.append({"image": str(path), "class_id": index, "source": "fixture",
                                 "captions": ["một ảnh", "ảnh khác"]})

    def test_caption_records_share_decoded_image(self):
        ds = ImageRecords(self.records[:1], self.config, "captioning", {"ảnh": 4}, 8, 10000)
        first, ids1 = ds[0]
        second, ids2 = ds[1]
        self.assertIs(first, second)
        self.assertEqual(ds.cache.hits, 1)
        self.assertEqual(ds.cache.misses, 1)
        self.assertEqual(ids1.shape, (8,))
        self.assertEqual(ids2.shape, (8,))
        self.assertEqual(ids1[0], 2)
        self.assertEqual(ds.image(self.records[0]).data_ptr(), first.data_ptr())

    def test_cache_lru_bound_and_disabled(self):
        bytes_per_image = 3 * 16 * 16 * 4
        cache = DecodedImageCache(bytes_per_image * 2)
        for row in self.records:
            cache.get(row["image"], self.config)
            self.assertLessEqual(cache.bytes, cache.limit_bytes)
        self.assertEqual(len(cache.items), 2)
        self.assertTrue(all(key[0] != self.records[0]["image"] for key in cache.items))
        disabled = DecodedImageCache(0)
        disabled.get(self.records[0]["image"], self.config)
        self.assertEqual(disabled.bytes, 0)
        self.assertFalse(disabled.items)

    def test_spawn_serialization_drops_parent_cache(self):
        ds = ImageRecords(self.records, self.config, "classification", cache_bytes=10000)
        ds[0]
        restored = pickle.loads(pickle.dumps(ds))
        self.assertFalse(restored.cache.items)
        self.assertEqual(restored.cache.limit_bytes, ds.cache.limit_bytes)
        self.assertTrue(torch.equal(ds[1][0], restored[1][0]))

    def test_cache_identity_includes_preprocessing(self):
        cache = DecodedImageCache(100000)
        first = cache.get(self.records[0]["image"], self.config)
        second = cache.get(self.records[0]["image"], {**self.config, "imgsz": 32})
        self.assertEqual(first.shape, (3, 16, 16))
        self.assertEqual(second.shape, (3, 32, 32))
        self.assertEqual(cache.misses, 2)

    def test_same_preprocessing_as_legacy_and_exact_partial_batch(self):
        from yolo_tuner.caption_features import image_tensor
        ds = ImageRecords(self.records[:5], self.config, "classification", cache_bytes=10000)
        self.assertTrue(torch.equal(ds[0][0], image_tensor(self.records[0]["image"], self.config)))
        policy = resolve_policy({}, "cpu", 5)
        loader = make_loader(ds, 2, policy)
        batches = list(loader)
        self.assertEqual([len(y) for _, y in batches], [2, 2, 1])
        self.assertEqual(torch.cat([y for _, y in batches]).tolist(), list(range(5)))
        observed = observed_loaders(policy, (loader,), (ds,))
        self.assertTrue(observed["verified"])
        self.assertEqual(observed["observed_workers"], [0])

    def test_windows_spawn_worker_reads_all_exact_samples(self):
        ds = ImageRecords(self.records, self.config, "classification", cache_bytes=10000)
        policy = resolve_policy({"workers": 1, "pin_memory": "off"}, "cpu", 6)
        loader = make_loader(ds, 4, policy)
        iterator = iter(loader)
        try:
            batches = list(iterator)
            self.assertEqual([len(y) for _, y in batches], [4, 2])
            self.assertEqual(torch.cat([y for _, y in batches]).tolist(), list(range(6)))
        finally:
            iterator._shutdown_workers()

    def test_cache_is_not_encoder_feature_cache(self):
        ds = ImageRecords(self.records[:1], self.config, "captioning", maximum=8, cache_bytes=10000)
        with patch("yolo_tuner.caption_features.image_tensor", return_value=torch.ones(3, 16, 16)) as decode:
            first, _ = ds[0]
            second, _ = ds[1]
        self.assertEqual(first.shape, (3, 16, 16))
        self.assertIs(first, second)
        decode.assert_called_once()


if __name__ == "__main__":
    unittest.main()
