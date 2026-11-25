"""Phase 2 tests: preflight data/anti-leakage audit, controlled search space, local .pt, and deep evaluation."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from research_pipeline import evaluation as det_eval
from research_pipeline import runner as det_runner
from research_pipeline import worker as det_worker
from research_pipeline.integrity import sha
from research_pipeline.v2 import acquisition, cli, data as v2_data, deployment, model_features, predictors, space
from research_pipeline.v2.data_features import describe_dataset
from research_pipeline.v2.input_diagnostics import DatasetInputError
from research_pipeline.v2.schema import config_id, lock_protocol
from yolo_tuner import adapters as legacy_adapters
from yolo_tuner.pipeline import write_json
from yolo_tuner.research import Objective, digest


def _write_img(path, color, size=(128, 128)):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def _snapshot_tree(root):
    root = Path(root)
    return {
        p.relative_to(root).as_posix(): (p.stat().st_mtime_ns, sha(p))
        for p in sorted(root.rglob("*")) if p.is_file()
    }


class Phase2DataAuditTests(unittest.TestCase):
    def _make_dirty_yolo_dataset(self, root):
        root = Path(root)
        # Train images: a.png (valid), a_dup.png (exact SHA-256 dup of a.png),
        # leak_tv.png (shared with val), leak_tt.png (shared with test),
        # missing_lbl.png, empty_lbl.png, malformed.png, bad_boxes.png
        _write_img(root / "images" / "train" / "a.png", (10, 20, 30), (200, 200))
        shutil.copyfile(root / "images" / "train" / "a.png", root / "images" / "train" / "a_dup.png")
        _write_img(root / "images" / "train" / "leak_tv.png", (40, 50, 60), (200, 200))
        _write_img(root / "images" / "train" / "leak_tt.png", (70, 80, 90), (200, 200))
        _write_img(root / "images" / "train" / "missing_lbl.png", (11, 22, 33), (200, 200))
        _write_img(root / "images" / "train" / "empty_lbl.png", (44, 55, 66), (200, 200))
        _write_img(root / "images" / "train" / "malformed.png", (77, 88, 99), (200, 200))
        _write_img(root / "images" / "train" / "bad_boxes.png", (100, 110, 120), (200, 200))

        # Val images: v1.png (valid), leak_from_train.png (exact dup of train leak_tv.png), leak_vt.png (shared with test)
        _write_img(root / "images" / "val" / "v1.png", (130, 140, 150), (200, 200))
        shutil.copyfile(root / "images" / "train" / "leak_tv.png", root / "images" / "val" / "leak_from_train.png")
        _write_img(root / "images" / "val" / "leak_vt.png", (160, 170, 180), (200, 200))

        # Test images: t1.png (valid), t1_dup.png (intra-test dup), leak_from_train.png, leak_from_val.png
        _write_img(root / "images" / "test" / "t1.png", (190, 200, 210), (200, 200))
        shutil.copyfile(root / "images" / "test" / "t1.png", root / "images" / "test" / "t1_dup.png")
        shutil.copyfile(root / "images" / "train" / "leak_tt.png", root / "images" / "test" / "leak_from_train.png")
        shutil.copyfile(root / "images" / "val" / "leak_vt.png", root / "images" / "test" / "leak_from_val.png")

        lbl_dir = root / "labels"
        for sp in ("train", "val", "test"):
            (lbl_dir / sp).mkdir(parents=True, exist_ok=True)

        # Class 0: small COCO (< 32^2 = 1024 px; at 200x200, w=0.08,h=0.08 -> 16x16=256 px, norm=0.0064 < 0.01)
        # Class 1: medium COCO ([32^2, 96^2); at 200x200, w=0.25,h=0.25 -> 50x50=2500 px, norm=0.0625 in [0.01,0.09))
        # Class 1: large COCO (>= 96^2 = 9216 px; at 200x200, w=0.6,h=0.6 -> 120x120=14400 px, norm=0.36 >= 0.09)
        # Class 2: zero-count class!
        (lbl_dir / "train" / "a.txt").write_text("0 0.5 0.5 0.08 0.08\n1 0.5 0.5 0.25 0.25\n", encoding="utf-8")
        (lbl_dir / "train" / "a_dup.txt").write_text("0 0.5 0.5 0.08 0.08\n", encoding="utf-8")
        (lbl_dir / "train" / "leak_tv.txt").write_text("1 0.5 0.5 0.6 0.6\n", encoding="utf-8")
        (lbl_dir / "train" / "leak_tt.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (lbl_dir / "train" / "empty_lbl.txt").write_text("   \n\n", encoding="utf-8")
        (lbl_dir / "train" / "malformed.txt").write_text("0 0.5 0.5 0.2\nnot_a_num 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (lbl_dir / "train" / "bad_boxes.txt").write_text(
            "3 0.5 0.5 0.2 0.2\n"      # invalid class >= nc (nc=3)
            "-1 0.5 0.5 0.2 0.2\n"     # invalid class < 0
            "0.5 0.5 0.5 0.2 0.2\n"    # non-int class
            "0 1.2 0.5 0.2 0.2\n"      # OOB center x > 1
            "0 0.05 0.5 0.4 0.2\n"     # OOB corner x - w/2 < 0
            "0 0.5 0.5 0.0 0.2\n"      # degenerate w <= 0
            "0 0.5 0.5 0.2 -0.1\n",    # degenerate h <= 0 (also OOB)
            encoding="utf-8",
        )
        (lbl_dir / "val" / "v1.txt").write_text("1 0.5 0.5 0.6 0.6\n", encoding="utf-8")
        (lbl_dir / "val" / "leak_from_train.txt").write_text("1 0.5 0.5 0.6 0.6\n", encoding="utf-8")
        (lbl_dir / "val" / "leak_vt.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        (lbl_dir / "test" / "t1.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        (lbl_dir / "test" / "t1_dup.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        (lbl_dir / "test" / "leak_from_train.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (lbl_dir / "test" / "leak_from_val.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

        yaml_path = root / "data.yaml"
        yaml_path.write_text(
            f"path: {root.as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "names: [cat, dog, bird]\n",
            encoding="utf-8",
        )
        return yaml_path

    def test_audit_detects_intra_dups_cross_split_leakage_and_od_anomalies_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            yaml_path = self._make_dirty_yolo_dataset(raw_dir)
            before_tree = _snapshot_tree(raw_dir)

            audit = v2_data.audit_dataset(yaml_path, task="detection")
            self.assertTrue(audit["read_only"])
            self.assertEqual(audit["counts"], {"train": 8, "val": 3, "test": 4, "unsplit": 0})
            self.assertEqual(audit["intra_split_duplicates"]["total_duplicate_files"], 2)
            self.assertEqual(audit["intra_split_duplicates"]["by_split"]["train"]["duplicate_count"], 1)
            self.assertEqual(audit["intra_split_duplicates"]["by_split"]["test"]["duplicate_count"], 1)
            self.assertTrue(audit["cross_split_leakage"]["has_leakage"])
            self.assertEqual(audit["cross_split_leakage"]["sha256_leak_count"], 3)
            self.assertEqual(set(audit["cross_split_leakage"]["split_pairs"]), {"train<->val", "train<->test", "val<->test"})

            # OD anomalies
            self.assertEqual(audit["missing_labels"]["count"], 1)
            self.assertEqual(audit["empty_labels"]["count"], 1)
            self.assertEqual(audit["malformed_labels"]["count"], 2)
            self.assertEqual(audit["invalid_class_ids"]["count"], 3)
            self.assertEqual(audit["degenerate_boxes"]["count"], 2)
            self.assertEqual(audit["out_of_bounds_boxes"]["count"], 3)
            self.assertEqual(audit["zero_count_classes"], ["bird"])
            self.assertIsNone(audit["imbalance_ratio"])
            self.assertIsNotNone(audit["nonzero_imbalance_ratio"])
            self.assertGreater(audit["object_size_distribution"]["coco_pixel_area"]["small"], 0)
            self.assertGreater(audit["object_size_distribution"]["coco_pixel_area"]["medium"], 0)
            self.assertGreater(audit["object_size_distribution"]["coco_pixel_area"]["large"], 0)
            self.assertGreater(audit["object_size_distribution"]["normalized_area"]["small"], 0)
            self.assertGreater(audit["object_size_distribution"]["normalized_area"]["medium"], 0)
            self.assertGreater(audit["object_size_distribution"]["normalized_area"]["large"], 0)

            # Immutability check: raw dataset untouched
            self.assertEqual(_snapshot_tree(raw_dir), before_tree)

            # Strict discover() must reject dirty dataset
            with self.assertRaises(ValueError):
                v2_data.discover(yaml_path, task="detection")

            # CLI inspect --data --json --audit-json works on dirty dataset without crashing & writes valid JSON
            audit_out = Path(tmp) / "exports" / "data_audit.json"
            buf = io.StringIO()
            with patch("yolo_tuner.pipeline.runtime_info", return_value={"device": "cpu", "ml_runtime_ready": True}), \
                 redirect_stdout(buf):
                rc = cli.main(argv=["inspect", "--data", str(yaml_path), "--json", "--audit-json", str(audit_out)])
            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertFalse(payload["dataset"]["validation_ready"])
            self.assertTrue(payload["dataset"]["audit"]["cross_split_leakage"]["has_leakage"])
            self.assertTrue(audit_out.is_file())
            self.assertEqual(json.loads(audit_out.read_text(encoding="utf-8"))["intra_split_duplicates"]["total_duplicate_files"], 2)
            self.assertEqual(_snapshot_tree(raw_dir), before_tree)

    def test_discover_rejects_cross_split_sha256_and_group_leakage_across_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 1. Clean YOLO yaml with train/val/test SHA-256 leak only (valid labels)
            _write_img(root / "yolo" / "images" / "train" / "a.png", (1, 2, 3), (64, 64))
            (root / "yolo" / "images" / "val").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / "yolo" / "images" / "train" / "a.png", root / "yolo" / "images" / "val" / "b.png")
            for sp, fn in (("train", "a.txt"), ("val", "b.txt")):
                (root / "yolo" / "labels" / sp).mkdir(parents=True, exist_ok=True)
                (root / "yolo" / "labels" / sp / fn).write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            y = root / "yolo" / "data.yaml"
            y.write_text(f"path: {(root / 'yolo').as_posix()}\ntrain: images/train\nval: images/val\nnames: [obj]\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Exact image content crosses split boundary"):
                v2_data.discover(y)

            # 2. ImageFolder dir with train/val SHA-256 leak
            _write_img(root / "imgfolder" / "train" / "c0" / "a.png", (10, 20, 30), (32, 32))
            _write_img(root / "imgfolder" / "train" / "c1" / "b.png", (40, 50, 60), (32, 32))
            (_write_img(root / "imgfolder" / "val" / "c0" / "dup.png", (1, 1, 1), (32, 32)))
            shutil.copyfile(root / "imgfolder" / "train" / "c0" / "a.png", root / "imgfolder" / "val" / "c0" / "dup.png")
            _write_img(root / "imgfolder" / "val" / "c1" / "v.png", (70, 80, 90), (32, 32))
            with self.assertRaisesRegex(ValueError, "Exact image content crosses split boundary"):
                v2_data.discover(root / "imgfolder")

            # 3. Manifest JSON group_id leakage between train and test
            _write_img(root / "m" / "1.png", (1, 1, 1), (32, 32))
            _write_img(root / "m" / "2.png", (2, 2, 2), (32, 32))
            _write_img(root / "m" / "3.png", (3, 3, 3), (32, 32))
            manifest = root / "m" / "manifest.json"
            write_json(manifest, {
                "task": "classification",
                "names": ["a", "b"],
                "splits": {
                    "train": [{"image": "1.png", "class_id": 0, "group_id": "scene_X"}],
                    "val": [{"image": "2.png", "class_id": 1, "group_id": "scene_Y"}],
                    "test": [{"image": "3.png", "class_id": 0, "group_id": "scene_X"}],
                },
            })
            manifest_audit = v2_data.audit_dataset(manifest)
            self.assertTrue(manifest_audit["cross_split_leakage"]["has_leakage"])
            self.assertEqual(manifest_audit["cross_split_leakage"]["group_id_leak_count"], 1)
            with self.assertRaisesRegex(ValueError, "group crosses"):
                v2_data.discover(manifest)

    def test_imbalance_ratio_edge_cases_and_real_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Single-class dataset -> imbalance_ratio == 1.0
            _write_img(root / "single" / "images" / "train" / "1.png", (1, 2, 3), (100, 100))
            _write_img(root / "single" / "images" / "val" / "2.png", (4, 5, 6), (100, 100))
            for sp, fn in (("train", "1.txt"), ("val", "2.txt")):
                (root / "single" / "labels" / sp).mkdir(parents=True, exist_ok=True)
                (root / "single" / "labels" / sp / fn).write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            y1 = root / "single" / "data.yaml"
            y1.write_text(f"path: {(root / 'single').as_posix()}\ntrain: images/train\nval: images/val\nnames: [pedestrian]\n", encoding="utf-8")
            a1 = v2_data.audit_dataset(y1)
            self.assertEqual(a1["imbalance_ratio"], 1.0)
            self.assertEqual(a1["nonzero_imbalance_ratio"], 1.0)

            # Normal multi-class -> 3/1 = 3.0
            (root / "single" / "labels" / "train" / "1.txt").write_text("0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            (root / "single" / "labels" / "val" / "2.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            y1.write_text(f"path: {(root / 'single').as_posix()}\ntrain: images/train\nval: images/val\nnames: [c0, c1]\n", encoding="utf-8")
            a2 = v2_data.audit_dataset(y1)
            self.assertAlmostEqual(a2["imbalance_ratio"], 3.0)

        # Real datasets in workspace
        aw = v2_data.audit_dataset("runs/datasets/african_wildlife_benchmark/data.yaml")
        self.assertFalse(aw["cross_split_leakage"]["has_leakage"])
        self.assertEqual(aw["intra_split_duplicates"]["total_duplicate_files"], 0)
        self.assertGreater(aw["imbalance_ratio"], 1.0)

        pf = v2_data.audit_dataset("runs/datasets/penn_fudan_research/data.yaml")
        self.assertFalse(pf["cross_split_leakage"]["has_leakage"])
        self.assertEqual(pf["imbalance_ratio"], 1.0)

    def test_dedup_dataset_clean_copy_and_inplace_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            _write_img(raw_dir / "images" / "train" / "a.png", (10, 20, 30), (64, 64))
            shutil.copyfile(raw_dir / "images" / "train" / "a.png", raw_dir / "images" / "train" / "a_dup.png")
            _write_img(raw_dir / "images" / "train" / "b.png", (40, 50, 60), (64, 64))
            _write_img(raw_dir / "images" / "val" / "v.png", (70, 80, 90), (64, 64))
            shutil.copyfile(raw_dir / "images" / "train" / "b.png", raw_dir / "images" / "val" / "b_leak.png")
            _write_img(raw_dir / "images" / "test" / "t.png", (100, 110, 120), (64, 64))
            shutil.copyfile(raw_dir / "images" / "val" / "v.png", raw_dir / "images" / "test" / "v_leak.png")
            for sp, fns in (("train", ("a.txt", "a_dup.txt", "b.txt")), ("val", ("v.txt", "b_leak.txt")), ("test", ("t.txt", "v_leak.txt"))):
                (raw_dir / "labels" / sp).mkdir(parents=True, exist_ok=True)
                for fn in fns:
                    (raw_dir / "labels" / sp / fn).write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            y = raw_dir / "data.yaml"
            y.write_text(f"path: {raw_dir.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [obj]\n", encoding="utf-8")
            before_tree = _snapshot_tree(raw_dir)

            # Refuses in-place or inside source or non-empty existing dir
            with self.assertRaisesRegex(ValueError, "in-place"):
                v2_data.dedup_dataset(y, raw_dir)
            with self.assertRaisesRegex(ValueError, "in-place"):
                v2_data.dedup_dataset(y, raw_dir / "clean_inside")
            nonempty = Path(tmp) / "nonempty"
            nonempty.mkdir()
            (nonempty / "keep.txt").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-empty"):
                v2_data.dedup_dataset(y, nonempty)

            clean_dir = Path(tmp) / "clean_copy"
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli.main(argv=["dedup-dataset", "--data", str(y), "--output", str(clean_dir), "--json"])
            self.assertEqual(rc, 0)
            summary = json.loads(buf.getvalue())
            self.assertEqual(summary["kept_count"], 4)
            self.assertEqual(summary["removed_count"], 3)
            self.assertEqual(summary["after_audit"]["intra_split_duplicates"]["total_duplicate_files"], 0)
            self.assertFalse(summary["after_audit"]["cross_split_leakage"]["has_leakage"])
            self.assertTrue((clean_dir / "dedup_audit.json").is_file())

            # Clean copy succeeds with strict discover()
            manifest = v2_data.discover(clean_dir / "data.yaml", task="detection")
            self.assertEqual(manifest["counts"], {"train": 2, "val": 1, "test": 1})
            self.assertEqual(_snapshot_tree(raw_dir), before_tree)

            # Zero test leakage into describe_dataset / subset
            desc = describe_dataset({**manifest, "splits": {**manifest["splits"], "test": [{"image": "does_not_exist.png", "boxes": []}]}})
            self.assertEqual(desc["split"], "train_only")
            sub = legacy_adapters.subset(manifest, 2, 1)
            self.assertEqual(sub["counts"]["test"], 0)

            # v2_manifest dedup-dataset preserves manifest.json & removes intra/cross-split dups
            m_root = Path(tmp) / "manifest_raw"
            _write_img(m_root / "t1.png", (1, 2, 3), (32, 32))
            _write_img(m_root / "t2.png", (4, 5, 6), (32, 32))
            _write_img(m_root / "v1.png", (7, 8, 9), (32, 32))
            shutil.copyfile(m_root / "t1.png", m_root / "v_leak.png")
            m_path = m_root / "manifest.json"
            write_json(m_path, {
                "task": "classification",
                "names": ["a", "b"],
                "splits": {
                    "train": [{"image": "t1.png", "class_id": 0}, {"image": "t2.png", "class_id": 1}],
                    "val": [{"image": "v1.png", "class_id": 0}, {"image": "v_leak.png", "class_id": 0}],
                },
            })
            m_clean = Path(tmp) / "manifest_clean"
            m_sum = v2_data.dedup_dataset(m_path, m_clean)
            self.assertTrue((m_clean / "manifest.json").is_file())
            self.assertEqual(m_sum["removed_count"], 1)
            self.assertFalse(m_sum["after_audit"]["cross_split_leakage"]["has_leakage"])
            m_disc = v2_data.discover(m_clean / "manifest.json")
            self.assertEqual(m_disc["counts"], {"train": 2, "val": 1, "test": 0})

    def test_inspect_invalid_input_does_not_init_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "table.csv"
            csv_path.write_text("id,age,income\n1,20,5000\n", encoding="utf-8")
            with patch("yolo_tuner.pipeline.runtime_info") as rt, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    cli.main(argv=["inspect", "--data", str(csv_path)])
            self.assertEqual(exc.exception.code, 2)
            rt.assert_not_called()

    def test_dedup_dataset_resolves_cross_split_group_id_leakage_and_preserves_same_split_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "manifest_group_raw"
            for idx, name in enumerate(("tr_a1.png", "tr_a2.png", "tr_c.png", "va_a.png", "va_b.png", "te_b.png", "te_d.png")):
                _write_img(raw_dir / name, (10 + idx * 25, 30, 50), (32, 32))
            manifest = raw_dir / "manifest.json"
            write_json(manifest, {
                "task": "classification",
                "names": ["c0", "c1"],
                "splits": {
                    "train": [
                        {"image": "tr_a1.png", "class_id": 0, "group_id": "scene_A"},
                        {"image": "tr_a2.png", "class_id": 1, "group_id": "scene_A"},
                        {"image": "tr_c.png", "class_id": 0, "group_id": "scene_C"},
                    ],
                    "val": [
                        {"image": "va_a.png", "class_id": 0, "group_id": "scene_A"},
                        {"image": "va_b.png", "class_id": 1, "group_id": "scene_B"},
                    ],
                    "test": [
                        {"image": "te_b.png", "class_id": 1, "group_id": "scene_B"},
                        {"image": "te_d.png", "class_id": 0, "group_id": "scene_D"},
                    ],
                },
            })
            clean_dir = root / "manifest_group_clean"
            res = v2_data.dedup_dataset(manifest, clean_dir)
            self.assertEqual(res["kept_count"], 5)
            self.assertEqual(res["removed_count"], 2)
            self.assertEqual({r["reason"] for r in res["removed_files"]}, {"cross_split_group_leakage"})
            self.assertFalse(res["after_audit"]["cross_split_leakage"]["has_leakage"])
            self.assertEqual(res["after_audit"]["cross_split_leakage"]["group_id_leak_count"], 0)
            discovered = v2_data.discover(res["entrypoint"])
            self.assertEqual(discovered["counts"], {"train": 3, "val": 1, "test": 1})
            train_groups = [r["group_id"] for r in discovered["splits"]["train"]]
            self.assertEqual(train_groups.count("scene_A"), 2)

            # Dropping all val records due to cross-split group_id leakage must raise cleanly
            all_leak_dir = root / "all_leak_raw"
            _write_img(all_leak_dir / "tr.png", (1, 2, 3), (32, 32))
            _write_img(all_leak_dir / "va.png", (4, 5, 6), (32, 32))
            leak_manifest = all_leak_dir / "manifest.json"
            write_json(leak_manifest, {
                "task": "classification",
                "names": ["c0", "c1"],
                "splits": {
                    "train": [{"image": "tr.png", "class_id": 0, "group_id": "shared"}],
                    "val": [{"image": "va.png", "class_id": 1, "group_id": "shared"}],
                },
            })
            with self.assertRaisesRegex(ValueError, "train or val split empty"):
                v2_data.dedup_dataset(leak_manifest, root / "all_leak_clean")

    def test_dedup_dataset_refuses_external_image_label_parents_and_junction_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 1. External image/label directories referenced via relative path in data.yaml
            ext_imgs = root / "external_images"
            ext_lbls = root / "external_labels"
            _write_img(ext_imgs / "train" / "t1.png", (10, 20, 30), (64, 64))
            _write_img(ext_imgs / "val" / "v1.png", (40, 50, 60), (64, 64))
            (ext_lbls / "train").mkdir(parents=True, exist_ok=True)
            (ext_lbls / "val").mkdir(parents=True, exist_ok=True)
            (ext_lbls / "train" / "t1.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            (ext_lbls / "val" / "v1.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            ds_yaml = root / "yaml_only" / "data.yaml"
            ds_yaml.parent.mkdir(parents=True, exist_ok=True)
            ds_yaml.write_text(
                f"path: {ds_yaml.parent.as_posix()}\n"
                "train: ../external_images/train\n"
                "val: ../external_images/val\n"
                "names: [obj]\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "in-place"):
                v2_data.dedup_dataset(ds_yaml, ext_imgs / "train" / "nested_out")

            # 2. Directory junction / symlink split target pointing to external directory
            ext_target = root / "ext_junction_target"
            _write_img(ext_target / "sub" / "j1.png", (70, 80, 90), (64, 64))
            j_ds = root / "junction_ds"
            _write_img(j_ds / "images" / "val" / "v1.png", (100, 110, 120), (64, 64))
            (j_ds / "labels" / "val").mkdir(parents=True, exist_ok=True)
            (j_ds / "labels" / "val" / "v1.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            link_path = j_ds / "images" / "train_link"
            linked = False
            if os.name == "nt":
                cp = subprocess.run(["cmd", "/c", "mklink", "/J", str(link_path), str(ext_target)], capture_output=True, text=True)
                linked = (cp.returncode == 0 and link_path.exists())
            else:
                try:
                    os.symlink(ext_target, link_path, target_is_directory=True)
                    linked = link_path.exists()
                except OSError:
                    linked = False
            if linked:
                try:
                    j_yaml = j_ds / "data.yaml"
                    j_yaml.write_text(
                        f"path: {j_ds.as_posix()}\n"
                        "train: images/train_link\n"
                        "val: images/val\n"
                        "names: [obj]\n",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, "in-place"):
                        v2_data.dedup_dataset(j_yaml, link_path / "nested_inside_raw")
                    with self.assertRaisesRegex(ValueError, "in-place"):
                        v2_data.dedup_dataset(j_yaml, ext_target / "nested_inside_target")
                finally:
                    if os.name == "nt":
                        subprocess.run(["cmd", "/c", "rmdir", str(link_path)], capture_output=True)
                    elif link_path.is_symlink():
                        link_path.unlink()

            # 3. Manifest referencing external images
            ext_m = root / "ext_manifest_imgs"
            _write_img(ext_m / "a.png", (1, 2, 3), (32, 32))
            _write_img(ext_m / "b.png", (4, 5, 6), (32, 32))
            m_dir = root / "m_dir"
            m_dir.mkdir()
            m_path = m_dir / "manifest.json"
            write_json(m_path, {
                "task": "classification",
                "names": ["c0", "c1"],
                "splits": {
                    "train": [{"image": "../ext_manifest_imgs/a.png", "class_id": 0}],
                    "val": [{"image": "../ext_manifest_imgs/b.png", "class_id": 1}],
                },
            })
            with self.assertRaisesRegex(ValueError, "in-place"):
                v2_data.dedup_dataset(m_path, ext_m / "nested_inside")

    def test_dedup_dataset_unsplit_only_yolo_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_unsplit = root / "yolo_unsplit"
            _write_img(raw_unsplit / "images" / "u1.png", (10, 20, 30), (64, 64))
            shutil.copyfile(raw_unsplit / "images" / "u1.png", raw_unsplit / "images" / "u1_dup.png")
            _write_img(raw_unsplit / "images" / "u2.png", (40, 50, 60), (64, 64))
            (raw_unsplit / "labels").mkdir(parents=True, exist_ok=True)
            (raw_unsplit / "labels" / "u1.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            (raw_unsplit / "labels" / "u1_dup.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            (raw_unsplit / "labels" / "u2.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")

            clean_unsplit = root / "yolo_unsplit_clean"
            res = v2_data.dedup_dataset(raw_unsplit, clean_unsplit, task="detection")
            self.assertEqual(res["kept_count"], 2)
            self.assertEqual(res["removed_count"], 1)
            self.assertEqual(res["after_audit"]["intra_split_duplicates"]["total_duplicate_files"], 0)
            self.assertFalse(res["after_audit"]["cross_split_leakage"]["has_leakage"])
            self.assertEqual(res["after_audit"]["counts"]["unsplit"], 2)
            disc = v2_data.discover(res["entrypoint"], task="detection")
            self.assertTrue(disc["validation_ready"])
            self.assertEqual(disc["counts"]["train"] + disc["counts"]["val"], 2)

    def test_empty_whitespace_and_falsy_zero_group_id_consistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 1. Unsplit manifest with "" and "   " group_id -> dedup_dataset() + discover() succeed
            u_dir = root / "unsplit_empty_gid"
            _write_img(u_dir / "u1.png", (10, 20, 30), (32, 32))
            _write_img(u_dir / "u2.png", (40, 50, 60), (32, 32))
            u_manifest = u_dir / "manifest.json"
            write_json(u_manifest, {
                "task": "classification",
                "names": ["c0", "c1"],
                "splits": {
                    "unsplit": [
                        {"image": "u1.png", "class_id": 0, "group_id": ""},
                        {"image": "u2.png", "class_id": 1, "group_id": "   "},
                    ],
                },
            })
            u_res = v2_data.dedup_dataset(u_manifest, root / "unsplit_empty_gid_clean")
            self.assertFalse(u_res["after_audit"]["cross_split_leakage"]["has_leakage"])
            self.assertEqual(u_res["kept_count"], 2)
            u_disc = v2_data.discover(u_res["entrypoint"])
            self.assertTrue(u_disc["validation_ready"])
            self.assertEqual(u_disc["counts"]["train"] + u_disc["counts"]["val"], 2)
            # Direct discover on raw unsplit manifest with "" / "   " also succeeds
            self.assertTrue(v2_data.discover(u_manifest)["validation_ready"])

            # 2. Split manifest with "" and "   " in train & val -> treated as missing (not leaked), val preserved
            s_dir = root / "split_empty_gid"
            _write_img(s_dir / "t1.png", (11, 22, 33), (32, 32))
            _write_img(s_dir / "t2.png", (44, 55, 66), (32, 32))
            _write_img(s_dir / "v1.png", (77, 88, 99), (32, 32))
            s_manifest = s_dir / "manifest.json"
            write_json(s_manifest, {
                "task": "classification",
                "names": ["c0", "c1"],
                "splits": {
                    "train": [
                        {"image": "t1.png", "class_id": 0, "group_id": ""},
                        {"image": "t2.png", "class_id": 1, "group_id": "  "},
                    ],
                    "val": [
                        {"image": "v1.png", "class_id": 0, "group_id": ""},
                    ],
                },
            })
            s_audit = v2_data.audit_dataset(s_manifest)
            self.assertFalse(s_audit["cross_split_leakage"]["has_leakage"])
            self.assertEqual(s_audit["cross_split_leakage"]["group_id_leak_count"], 0)
            s_res = v2_data.dedup_dataset(s_manifest, root / "split_empty_gid_clean")
            self.assertEqual(s_res["kept_count"], 3)
            self.assertEqual(s_res["removed_count"], 0)
            s_disc = v2_data.discover(s_res["entrypoint"])
            self.assertEqual(s_disc["counts"], {"train": 2, "val": 1, "test": 0})

            # 3. Falsy int group_id=0 is treated as a real group (not missing!)
            z_dir = root / "zero_gid"
            for idx, fn in enumerate(("t0a.png", "t0b.png", "v0_leak.png", "v1_ok.png")):
                _write_img(z_dir / fn, (10 + idx * 30, 20, 40), (32, 32))
            z_manifest = z_dir / "manifest.json"
            write_json(z_manifest, {
                "task": "classification",
                "names": ["c0", "c1"],
                "splits": {
                    "train": [
                        {"image": "t0a.png", "class_id": 0, "group_id": 0},
                        {"image": "t0b.png", "class_id": 1, "group_id": 0},
                    ],
                    "val": [
                        {"image": "v0_leak.png", "class_id": 0, "group_id": 0},
                        {"image": "v1_ok.png", "class_id": 1, "group_id": 1},
                    ],
                },
            })
            z_audit = v2_data.audit_dataset(z_manifest)
            self.assertTrue(z_audit["cross_split_leakage"]["has_leakage"])
            self.assertEqual(z_audit["cross_split_leakage"]["group_id_leak_count"], 1)
            z_res = v2_data.dedup_dataset(z_manifest, root / "zero_gid_clean")
            self.assertEqual(z_res["kept_count"], 3)
            self.assertEqual(z_res["removed_count"], 1)
            self.assertFalse(z_res["after_audit"]["cross_split_leakage"]["has_leakage"])
            z_disc = v2_data.discover(z_res["entrypoint"])
            self.assertEqual(z_disc["counts"], {"train": 2, "val": 1, "test": 0})
            self.assertEqual([r["group_id"] for r in z_disc["splits"]["train"]], ["0", "0"])

            # Unsplit group_id=0 stays together as a single group via finalize()
            zu_dir = root / "zero_gid_unsplit"
            for idx, fn in enumerate(("a0.png", "a1.png", "b0.png")):
                _write_img(zu_dir / fn, (50 + idx * 40, 10, 20), (32, 32))
            zu_manifest = zu_dir / "manifest.json"
            write_json(zu_manifest, {
                "task": "classification",
                "names": ["c0", "c1"],
                "splits": {
                    "unsplit": [
                        {"image": "a0.png", "class_id": 0, "group_id": 0},
                        {"image": "a1.png", "class_id": 1, "group_id": 0},
                        {"image": "b0.png", "class_id": 0, "group_id": 1},
                    ],
                },
            })
            zu_disc = v2_data.discover(zu_manifest)
            tr_gids = {r["group_id"] for r in zu_disc["splits"]["train"]}
            va_gids = {r["group_id"] for r in zu_disc["splits"]["val"]}
            self.assertFalse(tr_gids & va_gids)
            self.assertIn("0", tr_gids | va_gids)

    def test_dedup_dataset_refuses_nested_sub_junction_inside_split_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ext_nested = root / "ext_nested"
            _write_img(ext_nested / "subA" / "t2.png", (70, 80, 90), (64, 64))
            (ext_nested / "subA" / "t2.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

            ds_dir = root / "nested_j_ds"
            _write_img(ds_dir / "images" / "train_norm" / "t1.png", (10, 20, 30), (64, 64))
            _write_img(ds_dir / "images" / "val" / "v1.png", (40, 50, 60), (64, 64))
            (ds_dir / "labels" / "train_norm").mkdir(parents=True, exist_ok=True)
            (ds_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
            (ds_dir / "labels" / "train_norm" / "t1.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            (ds_dir / "labels" / "val" / "v1.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

            sub_link = ds_dir / "images" / "train_norm" / "sub_link"
            linked = False
            if os.name == "nt":
                cp = subprocess.run(["cmd", "/c", "mklink", "/J", str(sub_link), str(ext_nested)], capture_output=True, text=True)
                linked = (cp.returncode == 0 and sub_link.exists())
            else:
                try:
                    os.symlink(ext_nested, sub_link, target_is_directory=True)
                    linked = sub_link.exists()
                except OSError:
                    linked = False
            if not linked:
                self.skipTest("Filesystem does not permit directory junction/symlink creation")
            try:
                yp = ds_dir / "data.yaml"
                yp.write_text(
                    f"path: {ds_dir.as_posix()}\n"
                    "train: images/train_norm\n"
                    "val: images/val\n"
                    "names: [obj]\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "in-place"):
                    v2_data.dedup_dataset(yp, ext_nested / "sibling_in_ext_nested")
                with self.assertRaisesRegex(ValueError, "in-place"):
                    v2_data.dedup_dataset(yp, sub_link / "sibling")

                clean_ok = root / "clean_nested_ok"
                res = v2_data.dedup_dataset(yp, clean_ok)
                self.assertEqual(res["kept_count"], 3)
                self.assertFalse(res["after_audit"]["cross_split_leakage"]["has_leakage"])
                disc = v2_data.discover(res["entrypoint"], task="detection")
                self.assertEqual(disc["counts"], {"train": 2, "val": 1, "test": 0})
            finally:
                if os.name == "nt":
                    subprocess.run(["cmd", "/c", "rmdir", str(sub_link)], capture_output=True)
                elif sub_link.is_symlink():
                    sub_link.unlink()

    def _make_dir_link(self, link_path, target_path):
        link_path = Path(link_path)
        target_path = Path(target_path)
        link_path.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            cp = subprocess.run(["cmd", "/c", "mklink", "/J", str(link_path), str(target_path)], capture_output=True, text=True)
            return cp.returncode == 0 and link_path.exists()
        try:
            os.symlink(target_path, link_path, target_is_directory=True)
            return link_path.exists()
        except OSError:
            return False

    def _remove_dir_link(self, link_path):
        link_path = Path(link_path)
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "rmdir", str(link_path)], capture_output=True)
        elif link_path.is_symlink():
            link_path.unlink()

    def test_dedup_dataset_refuses_relative_escape_junctions_in_yaml_and_txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 1. data.yaml with train: ../ext_rel_wrap/sub_link/subD (sub_link -> ext_rel_target)
            ext_rel_target = root / "ext_rel_target"
            _write_img(ext_rel_target / "subD" / "r1.png", (10, 20, 30), (64, 64))
            (ext_rel_target / "subD" / "r1.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            rel_link = root / "ext_rel_wrap" / "sub_link"
            if not self._make_dir_link(rel_link, ext_rel_target):
                self.skipTest("Filesystem does not permit directory junction/symlink creation")
            try:
                ds_rel = root / "ds_rel_escape"
                _write_img(ds_rel / "images" / "val" / "v1.png", (40, 50, 60), (64, 64))
                (ds_rel / "labels" / "val").mkdir(parents=True, exist_ok=True)
                (ds_rel / "labels" / "val" / "v1.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
                yaml_rel = ds_rel / "data.yaml"
                yaml_rel.write_text(
                    f"path: {ds_rel.as_posix()}\n"
                    "train: ../ext_rel_wrap/sub_link/subD\n"
                    "val: images/val\n"
                    "names: [obj]\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "in-place"):
                    v2_data.dedup_dataset(yaml_rel, ext_rel_target / "sibling_in_target")
                with self.assertRaisesRegex(ValueError, "in-place"):
                    v2_data.dedup_dataset(yaml_rel, rel_link / "sibling_unresolved")
                res_rel = v2_data.dedup_dataset(yaml_rel, root / "clean_rel_ok")
                self.assertEqual(res_rel["kept_count"], 2)
                self.assertEqual(v2_data.discover(res_rel["entrypoint"], task="detection")["counts"], {"train": 1, "val": 1, "test": 0})
            finally:
                self._remove_dir_link(rel_link)

            # 2. data.yaml with train: train.txt listing ../ext_txt_wrap/sub_link/subE/t1.png
            ext_txt_target = root / "ext_txt_target"
            _write_img(ext_txt_target / "subE" / "t1.png", (70, 80, 90), (64, 64))
            (ext_txt_target / "subE" / "t1.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            txt_link = root / "ext_txt_wrap" / "sub_link"
            if not self._make_dir_link(txt_link, ext_txt_target):
                self.skipTest("Filesystem does not permit directory junction/symlink creation")
            try:
                ds_txt = root / "ds_txt_escape"
                _write_img(ds_txt / "images" / "val" / "v1.png", (100, 110, 120), (64, 64))
                (ds_txt / "labels" / "val").mkdir(parents=True, exist_ok=True)
                (ds_txt / "labels" / "val" / "v1.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
                (ds_txt / "train.txt").write_text("../ext_txt_wrap/sub_link/subE/t1.png\n", encoding="utf-8")
                yaml_txt = ds_txt / "data.yaml"
                yaml_txt.write_text(
                    f"path: {ds_txt.as_posix()}\n"
                    "train: train.txt\n"
                    "val: images/val\n"
                    "names: [obj]\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "in-place"):
                    v2_data.dedup_dataset(yaml_txt, ext_txt_target / "sibling_in_target")
                with self.assertRaisesRegex(ValueError, "in-place"):
                    v2_data.dedup_dataset(yaml_txt, txt_link / "sibling_unresolved")
                res_txt = v2_data.dedup_dataset(yaml_txt, root / "clean_txt_ok")
                self.assertEqual(res_txt["kept_count"], 2)
                self.assertEqual(v2_data.discover(res_txt["entrypoint"], task="detection")["counts"], {"train": 1, "val": 1, "test": 0})
            finally:
                self._remove_dir_link(txt_link)

    def test_dedup_dataset_yolo_dir_and_image_folder_with_junctions_preserve_split_and_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 1. yolo_dir (no data.yaml) with images/train/sub_link -> ext_yd
            ext_yd = root / "ext_yd"
            _write_img(ext_yd / "subA" / "t1.png", (10, 20, 30), (64, 64))
            (ext_yd / "subA" / "t1.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            ds_yd = root / "ds_yd"
            _write_img(ds_yd / "images" / "val" / "v1.png", (40, 50, 60), (64, 64))
            (ds_yd / "labels" / "val").mkdir(parents=True, exist_ok=True)
            (ds_yd / "labels" / "val" / "v1.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            yd_link = ds_yd / "images" / "train" / "sub_link"
            if not self._make_dir_link(yd_link, ext_yd):
                self.skipTest("Filesystem does not permit directory junction/symlink creation")
            try:
                audit_yd = v2_data.audit_dataset(ds_yd, task="detection")
                self.assertEqual(audit_yd["format"], "yolo_dir")
                self.assertEqual(audit_yd["counts"], {"train": 1, "val": 1, "test": 0, "unsplit": 0})
                with self.assertRaisesRegex(ValueError, "in-place"):
                    v2_data.dedup_dataset(ds_yd, ext_yd / "sibling")
                with self.assertRaisesRegex(ValueError, "in-place"):
                    v2_data.dedup_dataset(ds_yd, yd_link / "sibling")
                res_yd = v2_data.dedup_dataset(ds_yd, root / "clean_yd_ok")
                self.assertEqual(res_yd["kept_count"], 2)
                self.assertEqual(res_yd["after_audit"]["counts"], {"train": 1, "val": 1, "test": 0, "unsplit": 0})
                disc_yd = v2_data.discover(res_yd["entrypoint"], task="detection")
                self.assertEqual(disc_yd["counts"], {"train": 1, "val": 1, "test": 0})
            finally:
                self._remove_dir_link(yd_link)

            # 2. image_folder with train/cat/sub_link -> ext_cat
            ext_cat = root / "ext_cat"
            _write_img(ext_cat / "subC" / "c1.png", (70, 80, 90), (32, 32))
            ds_cls = root / "ds_cls"
            _write_img(ds_cls / "train" / "dog" / "d1.png", (100, 110, 120), (32, 32))
            _write_img(ds_cls / "val" / "cat" / "vc1.png", (130, 140, 150), (32, 32))
            _write_img(ds_cls / "val" / "dog" / "vd1.png", (160, 170, 180), (32, 32))
            cls_link = ds_cls / "train" / "cat" / "sub_link"
            if not self._make_dir_link(cls_link, ext_cat):
                self.skipTest("Filesystem does not permit directory junction/symlink creation")
            try:
                audit_cls = v2_data.audit_dataset(ds_cls, task="classification")
                self.assertEqual(audit_cls["format"], "image_folder")
                self.assertEqual(audit_cls["counts"], {"train": 2, "val": 2, "test": 0, "unsplit": 0})
                with self.assertRaisesRegex(ValueError, "in-place"):
                    v2_data.dedup_dataset(ds_cls, ext_cat / "sibling")
                res_cls = v2_data.dedup_dataset(ds_cls, root / "clean_cls_ok")
                self.assertEqual(res_cls["kept_count"], 4)
                disc_cls = v2_data.discover(res_cls["entrypoint"], task="classification")
                self.assertEqual(disc_cls["counts"], {"train": 2, "val": 2, "test": 0})
            finally:
                self._remove_dir_link(cls_link)

    def test_audit_dataset_handles_junction_cycle_without_inflation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = root / "ds_cycle"
            _write_img(ds / "images" / "train" / "a" / "t1.png", (10, 20, 30), (64, 64))
            (ds / "labels" / "train" / "a").mkdir(parents=True, exist_ok=True)
            (ds / "labels" / "train" / "a" / "t1.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            cycle_link = ds / "images" / "train" / "a" / "b"
            if not self._make_dir_link(cycle_link, ds / "images" / "train" / "a"):
                self.skipTest("Filesystem does not permit directory junction/symlink creation")
            try:
                audit = v2_data.audit_dataset(ds, task="detection")
                self.assertEqual(audit["format"], "yolo_dir")
                self.assertEqual(audit["counts"], {"train": 1, "val": 0, "test": 0, "unsplit": 0})
                self.assertEqual(audit["intra_split_duplicates"]["total_duplicate_files"], 0)
            finally:
                self._remove_dir_link(cycle_link)

    def test_dedup_dataset_refuses_txt_behind_junction_multihop_relative_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ext_imgs_deep = root / "ext_imgs_deep"
            _write_img(ext_imgs_deep / "t1.png", (15, 25, 35), (64, 64))
            (ext_imgs_deep / "t1.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

            ext_imgs_mid = root / "ext_imgs_mid"
            hop2 = ext_imgs_mid / "hop2"
            ext_holder = root / "ext_holder"
            txt_dir = ext_holder / "txt_dir"
            txt_dir.mkdir(parents=True, exist_ok=True)
            (txt_dir / "train.txt").write_text("../img_link/hop2/t1.png\n", encoding="utf-8")

            ds = root / "ds_txt_junc2"
            _write_img(ds / "images" / "val" / "v1.png", (45, 55, 65), (64, 64))
            (ds / "labels" / "val").mkdir(parents=True, exist_ok=True)
            (ds / "labels" / "val" / "v1.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

            splits_link = ds / "splits_link"
            img_link = ext_holder / "img_link"
            links = []
            try:
                for lp, tp in ((hop2, ext_imgs_deep), (img_link, ext_imgs_mid), (splits_link, txt_dir)):
                    if not self._make_dir_link(lp, tp):
                        self.skipTest("Filesystem does not permit directory junction/symlink creation")
                    links.append(lp)

                yaml_p = ds / "data.yaml"
                yaml_p.write_text(
                    f"path: {ds.as_posix()}\n"
                    "train: splits_link/train.txt\n"
                    "val: images/val\n"
                    "names: [obj]\n",
                    encoding="utf-8",
                )
                for bad_dest in (
                    ext_imgs_mid / "sibling",
                    ext_imgs_deep / "sibling",
                    txt_dir / "sibling",
                    img_link / "sibling",
                ):
                    with self.assertRaisesRegex(ValueError, "in-place"):
                        v2_data.dedup_dataset(yaml_p, bad_dest)

                clean_ok = root / "clean_deep12"
                res = v2_data.dedup_dataset(yaml_p, clean_ok)
                self.assertEqual(res["kept_count"], 2)
                self.assertEqual(v2_data.discover(res["entrypoint"], task="detection")["counts"], {"train": 1, "val": 1, "test": 0})
            finally:
                for lp in reversed(links):
                    self._remove_dir_link(lp)

    def test_audit_and_dedup_detect_dag_shared_junction_cross_split_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            links = []
            try:
                # 1. yolo_dir DAG-shared junction across train & val
                ext_yd = root / "ext_yd_shared"
                _write_img(ext_yd / "shared1.png", (10, 20, 30), (64, 64))
                (ext_yd / "shared1.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
                ds_yd = root / "ds_yd_shared"
                _write_img(ds_yd / "images" / "train" / "t_only.png", (40, 50, 60), (64, 64))
                _write_img(ds_yd / "images" / "val" / "v_only.png", (70, 80, 90), (64, 64))
                (ds_yd / "labels" / "train").mkdir(parents=True, exist_ok=True)
                (ds_yd / "labels" / "val").mkdir(parents=True, exist_ok=True)
                (ds_yd / "labels" / "train" / "t_only.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
                (ds_yd / "labels" / "val" / "v_only.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
                for lp in (ds_yd / "images" / "train" / "j_train", ds_yd / "images" / "val" / "j_val"):
                    if not self._make_dir_link(lp, ext_yd):
                        self.skipTest("Filesystem does not permit directory junction/symlink creation")
                    links.append(lp)

                audit_yd = v2_data.audit_dataset(ds_yd, task="detection")
                self.assertEqual(audit_yd["counts"], {"train": 2, "val": 2, "test": 0, "unsplit": 0})
                self.assertTrue(audit_yd["cross_split_leakage"]["has_leakage"])
                self.assertIn("train<->val", audit_yd["cross_split_leakage"]["split_pairs"])
                res_yd = v2_data.dedup_dataset(ds_yd, root / "clean_yd_shared")
                self.assertTrue(res_yd["before_audit"]["cross_split_leakage"]["has_leakage"])
                self.assertEqual(res_yd["removed_count"], 1)
                self.assertFalse(res_yd["after_audit"]["cross_split_leakage"]["has_leakage"])
                self.assertEqual(res_yd["after_audit"]["counts"], {"train": 2, "val": 1, "test": 0, "unsplit": 0})

                # 2. raw_images DAG-shared junction across train & val
                ext_raw = root / "ext_raw_shared"
                _write_img(ext_raw / "s1.png", (100, 110, 120), (32, 32))
                ds_raw = root / "ds_raw_shared"
                _write_img(ds_raw / "val" / "v_only.png", (130, 140, 150), (32, 32))
                for lp in (ds_raw / "train" / "j_train", ds_raw / "val" / "j_val"):
                    if not self._make_dir_link(lp, ext_raw):
                        self.skipTest("Filesystem does not permit directory junction/symlink creation")
                    links.append(lp)

                audit_raw = v2_data.audit_dataset(ds_raw)
                self.assertEqual(audit_raw["format"], "raw_images")
                self.assertEqual(audit_raw["counts"], {"train": 1, "val": 2, "test": 0, "unsplit": 0})
                self.assertTrue(audit_raw["cross_split_leakage"]["has_leakage"])

                # 3. image_folder DAG-shared junction across train & val
                ext_if = root / "ext_if_shared"
                _write_img(ext_if / "c_shared.png", (160, 170, 180), (32, 32))
                ds_if = root / "ds_if_shared"
                _write_img(ds_if / "train" / "dog" / "d1.png", (190, 200, 210), (32, 32))
                _write_img(ds_if / "val" / "dog" / "vd1.png", (210, 220, 230), (32, 32))
                _write_img(ds_if / "val" / "cat" / "vc_only.png", (220, 230, 240), (32, 32))
                for lp in (ds_if / "train" / "cat" / "j_train", ds_if / "val" / "cat" / "j_val"):
                    if not self._make_dir_link(lp, ext_if):
                        self.skipTest("Filesystem does not permit directory junction/symlink creation")
                    links.append(lp)

                audit_if = v2_data.audit_dataset(ds_if)
                self.assertEqual(audit_if["format"], "image_folder")
                self.assertEqual(audit_if["counts"], {"train": 2, "val": 3, "test": 0, "unsplit": 0})
                self.assertTrue(audit_if["cross_split_leakage"]["has_leakage"])
                res_if = v2_data.dedup_dataset(ds_if, root / "clean_if_shared")
                self.assertEqual(res_if["removed_count"], 1)
                self.assertFalse(res_if["after_audit"]["cross_split_leakage"]["has_leakage"])
                self.assertEqual(res_if["after_audit"]["counts"], {"train": 2, "val": 2, "test": 0, "unsplit": 0})
            finally:
                for lp in reversed(links):
                    self._remove_dir_link(lp)

    def test_dedup_dataset_refuses_yaml_behind_junction_and_path_junction_multihop_relative_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            links = []
            try:
                # 5A: data.yaml behind junction with train: ../img_link/hop2
                ext_imgs_deep = root / "ext_imgs_deep_5a"
                _write_img(ext_imgs_deep / "t1.png", (15, 25, 35), (64, 64))
                (ext_imgs_deep / "t1.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

                ext_imgs_mid = root / "ext_imgs_mid_5a"
                hop2 = ext_imgs_mid / "hop2"
                ext_holder = root / "ext_holder_5a"
                yaml_dir = ext_holder / "yaml_dir"
                _write_img(yaml_dir / "images" / "val" / "v1.png", (45, 55, 65), (64, 64))
                (yaml_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
                (yaml_dir / "labels" / "val" / "v1.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
                (yaml_dir / "data.yaml").write_text(
                    "train: ../img_link/hop2\n"
                    "val: images/val\n"
                    "names: [obj]\n",
                    encoding="utf-8",
                )

                ds_5a = root / "ds_5a"
                yaml_link = ds_5a / "yaml_link"
                img_link = ext_holder / "img_link"
                for lp, tp in ((hop2, ext_imgs_deep), (img_link, ext_imgs_mid), (yaml_link, yaml_dir)):
                    if not self._make_dir_link(lp, tp):
                        self.skipTest("Filesystem does not permit directory junction/symlink creation")
                    links.append(lp)

                yaml_5a = yaml_link / "data.yaml"
                audit_5a = v2_data.audit_dataset(yaml_5a, task="detection")
                self.assertEqual(audit_5a["counts"], {"train": 1, "val": 1, "test": 0, "unsplit": 0})
                self.assertEqual(audit_5a["unreadable_images"], [])
                for bad_dest in (
                    ext_imgs_mid / "sibling",
                    ext_imgs_deep / "sibling",
                    img_link / "sibling",
                    yaml_dir / "sibling",
                    yaml_link / "sibling",
                ):
                    with self.assertRaisesRegex(ValueError, "in-place"):
                        v2_data.dedup_dataset(yaml_5a, bad_dest)

                res_5a = v2_data.dedup_dataset(yaml_5a, root / "clean_5a")
                self.assertEqual(res_5a["kept_count"], 2)
                self.assertEqual(v2_data.discover(res_5a["entrypoint"], task="detection")["counts"], {"train": 1, "val": 1, "test": 0})

                # 5A2: data.yaml NOT behind junction, using path: path_link and train: ../img_link/hop2
                ext_imgs_deep2 = root / "ext_imgs_deep_5a2"
                _write_img(ext_imgs_deep2 / "t1.png", (75, 85, 95), (64, 64))
                (ext_imgs_deep2 / "t1.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

                ext_imgs_mid2 = root / "ext_imgs_mid_5a2"
                hop2_2 = ext_imgs_mid2 / "hop2"
                ext_holder2 = root / "ext_holder_5a2"
                base_dir2 = ext_holder2 / "base_dir"
                _write_img(base_dir2 / "images" / "val" / "v1.png", (105, 115, 125), (64, 64))
                (base_dir2 / "labels" / "val").mkdir(parents=True, exist_ok=True)
                (base_dir2 / "labels" / "val" / "v1.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

                ds_5a2 = root / "ds_5a2"
                path_link2 = ds_5a2 / "path_link"
                img_link2 = ext_holder2 / "img_link"
                for lp, tp in ((hop2_2, ext_imgs_deep2), (img_link2, ext_imgs_mid2), (path_link2, base_dir2)):
                    if not self._make_dir_link(lp, tp):
                        self.skipTest("Filesystem does not permit directory junction/symlink creation")
                    links.append(lp)

                yaml_5a2 = ds_5a2 / "data.yaml"
                yaml_5a2.write_text(
                    "path: path_link\n"
                    "train: ../img_link/hop2\n"
                    "val: images/val\n"
                    "names: [obj]\n",
                    encoding="utf-8",
                )
                audit_5a2 = v2_data.audit_dataset(yaml_5a2, task="detection")
                self.assertEqual(audit_5a2["counts"], {"train": 1, "val": 1, "test": 0, "unsplit": 0})
                self.assertEqual(audit_5a2["unreadable_images"], [])
                for bad_dest in (
                    ext_imgs_mid2 / "sibling",
                    ext_imgs_deep2 / "sibling",
                    img_link2 / "sibling",
                    base_dir2 / "sibling",
                    path_link2 / "sibling",
                ):
                    with self.assertRaisesRegex(ValueError, "in-place"):
                        v2_data.dedup_dataset(yaml_5a2, bad_dest)

                res_5a2 = v2_data.dedup_dataset(yaml_5a2, root / "clean_5a2")
                self.assertEqual(res_5a2["kept_count"], 2)
                self.assertEqual(v2_data.discover(res_5a2["entrypoint"], task="detection")["counts"], {"train": 1, "val": 1, "test": 0})
            finally:
                for lp in reversed(links):
                    self._remove_dir_link(lp)

    def test_audit_and_dedup_manifest_and_coco_behind_junction_multihop_relative_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            links = []
            try:
                # 5D: v2_manifest behind junction with relative ../img_link/hop2/m1.png
                ext_imgs_deep = root / "ext_imgs_deep_5d"
                _write_img(ext_imgs_deep / "m1.png", (10, 20, 30), (32, 32))

                ext_imgs_mid = root / "ext_imgs_mid_5d"
                hop2 = ext_imgs_mid / "hop2"
                ext_holder = root / "ext_holder_5d"
                man_dir = ext_holder / "man_dir"
                _write_img(man_dir / "v1.png", (40, 50, 60), (32, 32))
                write_json(man_dir / "manifest.json", {
                    "schema_version": 1,
                    "task": "classification",
                    "names": ["c0", "c1"],
                    "splits": {
                        "train": [{"image": "../img_link/hop2/m1.png", "class_id": 0}],
                        "val": [{"image": "v1.png", "class_id": 1}],
                    },
                })

                ds_5d = root / "ds_5d"
                man_link = ds_5d / "man_link"
                img_link = ext_holder / "img_link"
                for lp, tp in ((hop2, ext_imgs_deep), (img_link, ext_imgs_mid), (man_link, man_dir)):
                    if not self._make_dir_link(lp, tp):
                        self.skipTest("Filesystem does not permit directory junction/symlink creation")
                    links.append(lp)

                man_file = man_link / "manifest.json"
                self.assertEqual(v2_data.discover(man_file)["counts"], {"train": 1, "val": 1, "test": 0})
                audit_5d = v2_data.audit_dataset(man_file)
                self.assertEqual(audit_5d["format"], "v2_manifest")
                self.assertEqual(audit_5d["counts"], {"train": 1, "val": 1, "test": 0, "unsplit": 0})
                self.assertEqual(audit_5d["unreadable_images"], [])
                for bad_dest in (
                    ext_imgs_mid / "sibling",
                    ext_imgs_deep / "sibling",
                    img_link / "sibling",
                    man_dir / "sibling",
                    man_link / "sibling",
                ):
                    with self.assertRaisesRegex(ValueError, "in-place"):
                        v2_data.dedup_dataset(man_file, bad_dest)

                res_5d = v2_data.dedup_dataset(man_file, root / "clean_5d")
                self.assertEqual(res_5d["kept_count"], 2)
                self.assertEqual(v2_data.discover(res_5d["entrypoint"])["counts"], {"train": 1, "val": 1, "test": 0})

                # COCO JSON behind junction with relative ../img_link/hop2/c1.png
                ext_imgs_deep_c = root / "ext_imgs_deep_coco"
                _write_img(ext_imgs_deep_c / "c1.png", (70, 80, 90), (64, 64))

                ext_imgs_mid_c = root / "ext_imgs_mid_coco"
                hop2_c = ext_imgs_mid_c / "hop2"
                ext_holder_c = root / "ext_holder_coco"
                coco_dir = ext_holder_c / "coco_dir"
                _write_img(coco_dir / "v1.png", (100, 110, 120), (64, 64))
                write_json(coco_dir / "coco.json", {
                    "images": [
                        {"id": 1, "file_name": "../img_link/hop2/c1.png", "width": 64, "height": 64, "split": "train"},
                        {"id": 2, "file_name": "v1.png", "width": 64, "height": 64, "split": "val"},
                    ],
                    "annotations": [
                        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [8, 8, 24, 24], "iscrowd": 0},
                        {"id": 2, "image_id": 2, "category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 0},
                    ],
                    "categories": [{"id": 1, "name": "obj"}],
                })

                ds_coco = root / "ds_coco_junc"
                coco_link = ds_coco / "coco_link"
                img_link_c = ext_holder_c / "img_link"
                for lp, tp in ((hop2_c, ext_imgs_deep_c), (img_link_c, ext_imgs_mid_c), (coco_link, coco_dir)):
                    if not self._make_dir_link(lp, tp):
                        self.skipTest("Filesystem does not permit directory junction/symlink creation")
                    links.append(lp)

                coco_file = coco_link / "coco.json"
                self.assertEqual(v2_data.discover(coco_file)["counts"], {"train": 1, "val": 1, "test": 0})
                audit_coco = v2_data.audit_dataset(coco_file)
                self.assertEqual(audit_coco["format"], "coco_detection")
                self.assertEqual(audit_coco["counts"], {"train": 1, "val": 1, "test": 0, "unsplit": 0})
                self.assertEqual(audit_coco["unreadable_images"], [])
                for bad_dest in (
                    ext_imgs_mid_c / "sibling",
                    ext_imgs_deep_c / "sibling",
                    img_link_c / "sibling",
                    coco_dir / "sibling",
                    coco_link / "sibling",
                ):
                    with self.assertRaisesRegex(ValueError, "in-place"):
                        v2_data.dedup_dataset(coco_file, bad_dest)

                res_coco = v2_data.dedup_dataset(coco_file, root / "clean_coco")
                self.assertEqual(res_coco["kept_count"], 2)
                self.assertEqual(v2_data.discover(res_coco["entrypoint"])["counts"], {"train": 1, "val": 1, "test": 0})
            finally:
                for lp in reversed(links):
                    self._remove_dir_link(lp)


class Phase2ControlledSpaceAndLocalPtTests(unittest.TestCase):
    def test_optional_detection_keys_validation_and_non_detection_rejection(self):
        hw_gpu = {"device": "0", "fp16": True, "bf16": False}
        base_det = {"model": "yolo11n.yaml", "imgsz": 320, "batch": 4, "accumulation": 4, "precision": "fp32", "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.0005, "freeze": 0}
        space.validate_config("detection", {**base_det, "cos_lr": True, "warmup_epochs": 2.5, "mosaic": 0.5, "close_mosaic": 5}, hw_gpu)

        for bad in ({"cos_lr": 1}, {"warmup_epochs": -1.0}, {"warmup_epochs": True}, {"warmup_epochs": float("nan")},
                    {"mosaic": 1.2}, {"mosaic": -0.1}, {"mosaic": False}, {"close_mosaic": -1}, {"close_mosaic": 1.5}, {"close_mosaic": True}):
            with self.assertRaises(ValueError):
                space.validate_config("detection", {**base_det, **bad}, hw_gpu)

        base_cls = {"model": "cnn_classifier", "imgsz": 32, "batch": 16, "accumulation": 2, "width": 16, "precision": "fp32", "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.01}
        for key, val in (("cos_lr", True), ("warmup_epochs", 1.0), ("mosaic", 0.5), ("close_mosaic", 2)):
            with self.assertRaises(ValueError):
                space.validate_config("classification", {**base_cls, key: val}, {"device": "cpu"})

    def test_core_and_research_presets_with_local_pt_and_legacy_config_id_parity(self):
        hw = {"device": "cpu", "fp16": False}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Legacy grids must keep identical config_id(), canonical(), and configuration_distances()
            for grid_path in ("configs/african_wildlife_nckh_grid.json", "configs/detection_v2_controlled.json"):
                if not Path(grid_path).is_file():
                    continue
                cfgs = space.build_space("detection", hw, root / Path(grid_path).stem, grid=grid_path, precision="fp32")
                for c in cfgs:
                    for opt in space.OPTIONAL_DETECTION_KEYS:
                        self.assertNotIn(opt, c)
                    self.assertEqual(config_id(c), digest(space.canonical(c))[:12])

            # YAML-only core stays 12 configs with freeze=0
            yaml_core = space.build_space("detection", hw, root / "yaml_core", preset="core", precision="fp32")
            self.assertEqual(len(yaml_core), 12)
            self.assertTrue(all(c["freeze"] == 0 for c in yaml_core))

            # Single local .pt in core yields 12 unique configs including freeze=5
            pt1 = Path("weights/yolo11n.pt")
            pt2 = Path("weights/yolo11s.pt")
            pt_core_1 = space.build_space("detection", hw, root / "pt_core_1", preset="core", models=[str(pt1)], precision="fp32")
            self.assertEqual(len(pt_core_1), 12)
            self.assertEqual(sum(1 for c in pt_core_1 if c["freeze"] == 5), 1)

            # Two local .pt in core yields 13 unique configs including freeze=5
            pt_core_2 = space.build_space("detection", hw, root / "pt_core_2", preset="core", models=[str(pt1), str(pt2)], precision="fp32")
            self.assertEqual(len(pt_core_2), 13)
            self.assertEqual(sum(1 for c in pt_core_2 if c["freeze"] == 5), 1)

            # Mixed yaml + pt never applies freeze=5 to yaml
            mixed_core = space.build_space("detection", hw, root / "mixed_core", preset="core", models=["yolo11n.yaml", str(pt1)], precision="fp32")
            for c in mixed_core:
                if not str(c["model"]).lower().endswith(".pt"):
                    self.assertEqual(c["freeze"], 0)
            self.assertEqual(sum(1 for c in mixed_core if c["freeze"] == 5), 1)

            # Research preset with .pt includes freeze=5 and controlled optional contrasts on choices[0]
            pt_research = space.build_space("detection", hw, root / "pt_research", preset="research", models=[str(pt1)], precision="fp32")
            self.assertTrue(any(c.get("freeze") == 5 for c in pt_research))
            self.assertTrue(any(c.get("cos_lr") is True for c in pt_research))
            self.assertTrue(any(c.get("warmup_epochs") == 3.0 for c in pt_research))
            self.assertTrue(any(c.get("mosaic") == 0.5 and c.get("close_mosaic") == 5 for c in pt_research))
            self.assertTrue(any(c.get("mosaic") == 0.0 and c.get("close_mosaic") == 0 for c in pt_research))

    def test_local_pt_descriptors_and_offline_missing_pt_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Create local yolov8n.pt offline from installed yolov8n.yaml
            from ultralytics import YOLO
            v8_pt = Path(tmp) / "yolov8n.pt"
            YOLO("yolov8n.yaml").save(str(v8_pt))

            configs = [{"model": str(p)} for p in ("weights/yolo11n.pt", "weights/yolo11s.pt", "weights/yolo26n.pt", str(v8_pt))]
            descs = model_features.describe_models("detection", configs, 4)
            self.assertEqual(len(descs), 4)
            for d in descs.values():
                self.assertTrue(d["pretrained_checkpoint"])
                self.assertGreater(d["parameters"], 0)
                self.assertGreater(d["trainable_parameters"], 0)
                self.assertGreater(d["layers"], 0)
                self.assertGreater(d["gflops_reference_32"], 0.0)

            # Offline rejection for missing .pt without calling YOLO download
            missing = Path(tmp) / "missing_weights.pt"
            with patch("ultralytics.YOLO") as mock_yolo:
                with self.assertRaisesRegex(ValueError, "network downloads disabled"):
                    model_features.describe_models("detection", [{"model": str(missing)}], 4)
                mock_yolo.assert_not_called()

        # Verify predictor card schema_version=2 & 32 FEATURE_NAMES backward compatibility
        self.assertEqual(len(predictors.FEATURE_NAMES), 32)
        card_path = Path("runs/predictor_091_cpu_fashion_fit_only.json")
        if card_path.is_file():
            card = json.loads(card_path.read_text(encoding="utf-8"))
            self.assertEqual(card["feature_names"], predictors.FEATURE_NAMES)

    def test_worker_forwards_optional_keys_and_actual_matches(self):
        base = {"model": "yolo11n.yaml", "imgsz": 64, "batch": 2, "accumulation": 2, "precision": "fp32", "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.0005, "freeze": 0}
        # Legacy config without optional keys still matches legacy actual
        actual_legacy = {k: v for k, v in base.items() if k not in {"model", "freeze"}}
        self.assertTrue(det_runner.actual_matches(base, actual_legacy))

        # Config with optional keys requires matching actual
        ext = {**base, "cos_lr": True, "warmup_epochs": 2.0, "mosaic": 0.5, "close_mosaic": 3}
        self.assertFalse(det_runner.actual_matches(ext, actual_legacy))
        actual_ext = {**actual_legacy, "cos_lr": True, "warmup_epochs": 2.0, "mosaic": 0.5, "close_mosaic": 3}
        self.assertTrue(det_runner.actual_matches(ext, actual_ext))
        self.assertFalse(det_runner.actual_matches(ext, {**actual_ext, "cos_lr": False}))
        self.assertFalse(det_runner.actual_matches(ext, {**actual_ext, "close_mosaic": 4}))

        # Verify research_pipeline/worker.py forwards kwargs to model.train(**kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captured_kw = {}

            class FakeYOLO:
                task = "detect"
                def __init__(self, path):
                    self.callbacks = {}
                    self.trainer = None
                def add_callback(self, event, cb):
                    self.callbacks[event] = cb
                def train(self, **kw):
                    captured_kw.update(kw)
                    raise RuntimeError("stop-after-kwargs-check")

            req = {
                "config": ext, "mode": "train", "data": str(root / "data.yaml"),
                "device": "cpu", "full_epochs": 2, "trial_epochs": 2, "seed": 17,
                "project": str(root), "counts": {"train": 4, "val": 2},
                "full_counts": {"train": 4, "val": 2}, "detection_metric": "map50",
            }
            out = root / "result.json"
            with patch("ultralytics.YOLO", FakeYOLO), redirect_stderr(io.StringIO()):
                det_worker.execute(req, out)
            self.assertIs(captured_kw["cos_lr"], True)
            self.assertEqual(captured_kw["warmup_epochs"], 2.0)
            self.assertEqual(captured_kw["mosaic"], 0.5)
            self.assertEqual(captured_kw["close_mosaic"], 3)

    def test_acquisition_distances_and_rank_candidates_sensitivity(self):
        base = {"model": "yolo11n.yaml", "imgsz": 320, "batch": 4, "accumulation": 4, "precision": "fp32", "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.0005, "freeze": 0}
        p_legacy = lock_protocol({
            "task": "detection", "method": "adaptive_full", "mode": "online", "implementation_version": "0.11.0",
            "configs": [base, {**base, "lr0": 0.0003}, {**base, "lr0": 0.003}],
            "training_seeds": [17], "search_seed": 17, "full_epochs": 2, "probe_epochs": 1,
            "budget_seconds": 30, "trial_timeout": 20, "objective": asdict(Objective()),
            "hardware": {"device": "cpu", "host_label": "h"}, "runtime_versions": {},
            "scheduler": {"cleanup_seconds": 0.1, "probe_fraction": 0.1, "cost_margin": 1.25},
            "example_counts": {"train": 10, "val": 4}, "memory_kind": "process_tree_rss_mib_not_vram",
            "measurement_environment": "uncontended_preflight", "preparation_seconds": 0,
            "source_files": {}, "data_files": {}, "model_files": {}, "dataset_source": "unit",
            "quality_definition": "val map50",
            "acquisition": {"version": 2, "kernel_ridge": 0.05, "length_scale": 1.0, "exploration": 0.25, "novelty_weight": 0.05},
        })
        dist_leg, meta_leg = acquisition.configuration_distances(p_legacy)
        for k in ("cos_lr", "warmup_epochs", "mosaic", "close_mosaic"):
            self.assertNotIn(k, meta_leg["numeric_fields"])
            self.assertNotIn(k, meta_leg["categorical_fields"])

        # Varying new hyperparameters changes distances & rank_candidates
        c_new = [
            base,
            {**base, "cos_lr": False, "warmup_epochs": 0.0, "mosaic": 1.0, "close_mosaic": 0},
            {**base, "cos_lr": True, "warmup_epochs": 3.0, "mosaic": 0.2, "close_mosaic": 5},
        ]
        # Make c0 and c1 differ only via new keys vs c2
        c_varied = [
            {**base, "cos_lr": False, "warmup_epochs": 0.0, "mosaic": 1.0, "close_mosaic": 0},
            {**base, "cos_lr": False, "warmup_epochs": 0.1, "mosaic": 0.95, "close_mosaic": 0},
            {**base, "cos_lr": True, "warmup_epochs": 5.0, "mosaic": 0.0, "close_mosaic": 10},
        ]
        p_var = lock_protocol({**{k: v for k, v in p_legacy.items() if k not in {"configs", "config_ids", "protocol_sha256"}}, "configs": c_varied})
        dist_var, meta_var = acquisition.configuration_distances(p_var)
        self.assertIn("warmup_epochs", meta_var["numeric_fields"])
        self.assertIn("mosaic", meta_var["numeric_fields"])
        self.assertIn("close_mosaic", meta_var["numeric_fields"])
        self.assertIn("cos_lr", meta_var["categorical_fields"])
        self.assertGreater(dist_var[0, 2], dist_var[0, 1])

        rows = [
            {"index": 0, "config": c_varied[0], "complete": True, "state": "confirmed_feasible",
             "runs": [{"seed": 17, "mode": "train", "status": "ok", "epochs_completed": 2, "optimizer_steps": 1,
                       "training_loss_finite": True, "actual_config_verified": True, "checkpoint_sha256": "x",
                       "quality": 0.4, "worker_wall_seconds": 5.0, "memory_mb": 100.0}]},
            {"index": 2, "config": c_varied[2], "complete": True, "state": "confirmed_feasible",
             "runs": [{"seed": 17, "mode": "train", "status": "ok", "epochs_completed": 2, "optimizer_steps": 1,
                       "training_loss_finite": True, "actual_config_verified": True, "checkpoint_sha256": "y",
                       "quality": 0.9, "worker_wall_seconds": 5.0, "memory_mb": 100.0}]},
        ]
        ranked_a = acquisition.rank_candidates(p_var, rows, {1: {"point_seconds": 5.0, "predicted_memory_mb": 100.0}})
        # Flip qualities between c0 and c2 -> c1 (closer to c0) prediction shifts accordingly
        rows_flipped = [{**rows[0], "runs": [{**rows[0]["runs"][0], "quality": 0.9}]},
                        {**rows[1], "runs": [{**rows[1]["runs"][0], "quality": 0.4}]}]
        ranked_b = acquisition.rank_candidates(p_var, rows_flipped, {1: {"point_seconds": 5.0, "predicted_memory_mb": 100.0}})
        self.assertNotAlmostEqual(ranked_a["scores"]["1"]["quality_mean_prediction"], ranked_b["scores"]["1"]["quality_mean_prediction"])


class Phase2DeepEvaluationTests(unittest.TestCase):
    def test_real_yolo11n_val_parity_and_deep_outputs(self):
        from ultralytics import YOLO

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = Path("weights/yolo11n.pt").resolve()
            model_obj = YOLO(str(ckpt))
            coco_names = [model_obj.names[i] for i in range(len(model_obj.names))]
            aw_root = Path("runs/datasets/african_wildlife_benchmark")
            elephants = ["test_0010_9a7ed68c608e", "test_0011_004ebfc6a797"]
            (root / "images" / "test").mkdir(parents=True, exist_ok=True)
            (root / "labels" / "test").mkdir(parents=True, exist_ok=True)
            if all((aw_root / "images" / "test" / f"{s}.jpg").is_file() for s in elephants):
                for s in elephants:
                    shutil.copyfile(aw_root / "images" / "test" / f"{s}.jpg", root / "images" / "test" / f"{s}.jpg")
                    raw_lines = (aw_root / "labels" / "test" / f"{s}.txt").read_text(encoding="utf-8").splitlines()
                    # Remap african_wildlife class 1 (elephant) -> COCO class 20 (elephant)
                    mapped = ["20 " + " ".join(l.split()[1:]) for l in raw_lines if l.strip()]
                    (root / "labels" / "test" / f"{s}.txt").write_text("\n".join(mapped) + "\n", encoding="utf-8")
            else:
                for i in range(2):
                    _write_img(root / "images" / "test" / f"img_{i}.png", (40 + i * 30, 80, 120), (64, 64))
                    (root / "labels" / "test" / f"img_{i}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            import yaml
            data_yaml = root / "data.yaml"
            data_yaml.write_text(yaml.safe_dump({"path": root.as_posix(), "train": "images/test", "val": "images/test", "test": "images/test", "names": coco_names}, sort_keys=False), encoding="utf-8")

            # Direct reference calls (plots=False for scalar metrics; plots=True offline for confusion matrix)
            ref = YOLO(str(ckpt)).val(
                data=str(data_yaml), split="test", imgsz=160, batch=2, rect=False, quantize=None,
                device="cpu", workers=0, plots=False, project=str(root / "ref"), name="val", verbose=False,
            )
            with patch("ultralytics.data.utils.check_font", return_value=None), \
                 patch("ultralytics.utils.checks.check_font", return_value=None), \
                 patch("ultralytics.utils.plotting.check_font", return_value=None):
                ref_cm = YOLO(str(ckpt)).val(
                    data=str(data_yaml), split="test", imgsz=160, batch=2, rect=False, quantize=None,
                    device="cpu", workers=0, plots=True, project=str(root / "ref_cm"), name="val", verbose=False,
                )

            proj = root / "eval_proj"
            proj.mkdir()
            req_path = proj / "request.json"
            write_json(req_path, {
                "checkpoint": str(ckpt),
                "checkpoint_sha256": sha(ckpt),
                "data": str(data_yaml),
                "device": "cpu",
                "imgsz": 160,
                "batch": 2,
                "split": "test",
                "project": str(proj),
                "detection_metric": "map50",
                "inference_precision": "fp32",
                "warmup_batches": 2,
            })
            det_eval.worker(str(req_path))
            res = json.loads((proj / "result.json").read_text(encoding="utf-8"))

            self.assertLess(abs(res["map50"] - float(ref.box.map50)), 1e-12)
            self.assertLess(abs(res["map75"] - float(ref.box.map75)), 1e-12)
            self.assertLess(abs(res["map50_95"] - float(ref.box.map)), 1e-12)
            self.assertLess(abs(res["precision"] - float(ref.box.mp)), 1e-12)
            self.assertLess(abs(res["recall"] - float(ref.box.mr)), 1e-12)
            ref_f1 = (2.0 * float(ref.box.mp) * float(ref.box.mr) / (float(ref.box.mp) + float(ref.box.mr))) if (float(ref.box.mp) + float(ref.box.mr)) > 0 else 0.0
            self.assertLess(abs(res["f1"] - ref_f1), 1e-12)
            self.assertEqual(len(res["per_class"]), len(ref.box.ap_class_index))

            # Latency
            lat = res["latency"]
            self.assertTrue(lat["available"])
            self.assertGreater(lat["total_ms_per_image"], 0.0)
            self.assertGreater(lat["fps"], 0.0)
            self.assertEqual(lat["batch_size"], 2)
            self.assertEqual(lat["precision"], "fp32")
            self.assertEqual(lat["warmup_batches"], 2)
            self.assertIs(lat["rect"], False)

            # Curves & confusion matrix
            self.assertTrue(res["pr_curve"]["available"])
            self.assertGreater(len(res["pr_curve"]["recall_grid"]), 0)
            self.assertEqual(len(res["pr_curve"]["mean_precision"]), len(res["pr_curve"]["recall_grid"]))
            self.assertTrue(res["f1_curve"]["available"])
            self.assertIsNotNone(res["f1_curve"]["best_confidence"])
            self.assertTrue(res["confusion_matrix"]["available"])
            cm = res["confusion_matrix"]
            self.assertEqual(cm["labels"][-1], "background")
            self.assertEqual(len(cm["raw"]), cm["nc"] + 1)
            raw_arr = np.asarray(cm["raw"], dtype=float)
            norm_arr = np.asarray(cm["normalized"], dtype=float)
            self.assertGreater(float(raw_arr.sum()), 0.0)
            self.assertGreater(float(norm_arr.sum()), 0.0)
            self.assertTrue(any(raw_arr[:, col].sum() > 0 for col in range(raw_arr.shape[1])))
            for col in range(raw_arr.shape[1]):
                if raw_arr[:, col].sum() > 0:
                    self.assertAlmostEqual(float(norm_arr[:, col].sum()), 1.0, places=9)
            ref_cm_arr = np.asarray(ref_cm.confusion_matrix.matrix, dtype=float)
            np.testing.assert_allclose(raw_arr, ref_cm_arr, atol=1e-12)
            val_dir = proj / "validation"
            leftover_plots = list(val_dir.glob("*.png")) + list(val_dir.glob("*.jpg")) if val_dir.exists() else []
            self.assertEqual(leftover_plots, [])

    def test_zero_target_zero_detection_nan_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = root / "best.pt"
            ckpt.write_bytes(b"mock-zero-target-ckpt")
            req_path = root / "request.json"
            write_json(req_path, {
                "checkpoint": str(ckpt), "checkpoint_sha256": sha(ckpt),
                "data": str(root / "data.yaml"), "device": "cpu", "imgsz": 64, "batch": 2,
                "split": "test", "project": str(root), "detection_metric": "map50",
                "inference_precision": "fp32", "warmup_batches": 0,
            })

            class ZeroBox:
                map50 = 0.0
                map75 = 0.0
                map = 0.0
                mp = 0.0
                mr = 0.0
                ap_class_index = []
                ap50 = []
                ap = []
                p = []
                r = []

            class ZeroCM:
                matrix = np.zeros((3, 3), dtype=float)
                nc = 2
                names = {0: "cat", 1: "dog"}

            px = np.linspace(0.0, 1.0, 10)

            class FakeZeroYOLO:
                def __init__(self, path):
                    pass

                def val(self, **kwargs):
                    return SimpleNamespace(
                        box=ZeroBox(),
                        names={0: "cat", 1: "dog"},
                        speed={"preprocess": 0.5, "inference": float("nan"), "postprocess": 0.2},
                        curves_results=[
                            [px, np.full((1, 10), float("nan")), "Recall", "Precision"],
                            [px, np.zeros((0, 10)), "Confidence", "F1"],
                        ],
                        confusion_matrix=ZeroCM(),
                    )

            with patch("ultralytics.YOLO", FakeZeroYOLO):
                det_eval.worker(str(req_path))
            out = json.loads((root / "result.json").read_text(encoding="utf-8"))
            serialized = json.dumps(out, allow_nan=False)
            self.assertIn("confusion_matrix", serialized)
            self.assertEqual(out["f1_curve"]["mean_f1"], [0.0] * 10)
            self.assertEqual(out["pr_curve"]["mean_precision"], [0.0] * 10)
            self.assertEqual(out["confusion_matrix"]["normalized"], [[0.0] * 3 for _ in range(3)])

    def test_multi_seed_evaluate_v2_writes_test_summary_json_and_aggregates(self):
        from tests.test_map50_and_nckh_011 import Map50AndNCKH011Tests

        helper = Map50AndNCKH011Tests()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            helper._build_audited_run(run_dir)
            px = [0.0, 0.5, 1.0]

            def fake_popen(cmd, stdout=None, stderr=None, **kw):
                req_path = Path(cmd[-1])
                req = json.loads(req_path.read_text(encoding="utf-8"))
                seed = req["seed"]
                m50 = 0.88 if seed == 17 else 0.92
                write_json(Path(req["project"]) / "result.json", {
                    "status": "ok",
                    "primary_metric": "map50",
                    "rect": False,
                    "quality": m50,
                    "map50": m50,
                    "map75": 0.72,
                    "map50_95": 0.61,
                    "precision": 0.90,
                    "recall": 0.86,
                    "f1": 0.8795454545,
                    "per_class": [{"class_id": 0, "class_name": "person", "ap50": m50, "ap50_95": 0.61}],
                    "latency": {
                        "available": True, "preprocess_ms": 1.0, "inference_ms": 4.0 if seed == 17 else 6.0,
                        "postprocess_ms": 1.0, "total_ms_per_image": 6.0 if seed == 17 else 8.0,
                        "fps": 1000.0 / (6.0 if seed == 17 else 8.0),
                        "batch_size": req["batch"], "precision": req["inference_precision"],
                        "warmup_batches": req["warmup_batches"], "device": req["device"], "rect": False,
                    },
                    "pr_curve": {
                        "available": True, "recall_grid": px, "mean_precision": [1.0, m50, 0.5],
                        "per_class": [{"class_id": 0, "class_name": "person", "precision": [1.0, m50, 0.5]}],
                    },
                    "f1_curve": {
                        "available": True, "confidence_grid": px, "mean_f1": [0.5, m50, 0.3],
                        "per_class": [{"class_id": 0, "class_name": "person", "f1": [0.5, m50, 0.3]}],
                        "best_confidence": 0.5, "max_mean_f1": m50,
                    },
                    "confusion_matrix": {
                        "available": True, "labels": ["person", "background"], "nc": 1,
                        "axes": "rows=predicted, cols=true", "normalization": "column_true_sum",
                        "raw": [[8.0, 1.0], [2.0, 0.0]], "normalized": [[0.8, 1.0], [0.2, 0.0]],
                    },
                })
                return SimpleNamespace(returncode=0, wait=lambda timeout=None: 0, poll=lambda: 0, args=cmd)

            args = SimpleNamespace(
                run=str(run_dir), timeout_seconds=60, seed_scope="all",
                inference_precision="fp32", warmup_batches=2, allow_retest=False,
            )
            with patch("research_pipeline.v2.deployment.idle"), \
                 patch("research_pipeline.v2.deployment.verify_live", return_value=(json.loads((run_dir / "dataset.json").read_text(encoding="utf-8")), {})), \
                 patch("research_pipeline.v2.deployment.subprocess.Popen", side_effect=fake_popen), \
                 redirect_stdout(io.StringIO()):
                eval_res = deployment.evaluate(args)

            self.assertTrue((run_dir / "evaluation" / "test_summary.json").is_file())
            summary_doc = json.loads((run_dir / "evaluation" / "test_summary.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(summary_doc["latency"]["total_ms_per_image"]["mean"], 7.0)
            self.assertEqual(summary_doc["latency"]["warmup_batches"], 2)
            self.assertAlmostEqual(summary_doc["pr_curve"]["mean_precision"][1], 0.90)
            self.assertAlmostEqual(summary_doc["f1_curve"]["best_confidence"], 0.5)
            self.assertEqual(summary_doc["confusion_matrix"]["sum_raw"], [[16.0, 2.0], [4.0, 0.0]])
            self.assertAlmostEqual(summary_doc["confusion_matrix"]["normalized"][0][0], 0.8)
            self.assertIn("latency", eval_res)

    def test_cuda_fp16_or_cpu_downgrade_precision_recorded_honestly(self):
        import torch
        from ultralytics import YOLO

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = Path("weights/yolo11n.pt").resolve()
            model_obj = YOLO(str(ckpt))
            coco_names = [model_obj.names[i] for i in range(len(model_obj.names))]
            (root / "images" / "test").mkdir(parents=True, exist_ok=True)
            (root / "labels" / "test").mkdir(parents=True, exist_ok=True)
            _write_img(root / "images" / "test" / "0.png", (50, 60, 70), (64, 64))
            (root / "labels" / "test" / "0.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            import yaml
            data_yaml = root / "data.yaml"
            data_yaml.write_text(yaml.safe_dump({"path": root.as_posix(), "train": "images/test", "val": "images/test", "test": "images/test", "names": coco_names}, sort_keys=False), encoding="utf-8")

            device = "0" if torch.cuda.is_available() else "cpu"
            expected_prec = "fp16" if torch.cuda.is_available() else "fp32"
            proj = root / "eval_fp16"
            proj.mkdir()
            req_path = proj / "request.json"
            write_json(req_path, {
                "checkpoint": str(ckpt), "checkpoint_sha256": sha(ckpt),
                "data": str(data_yaml), "device": device, "imgsz": 64, "batch": 1,
                "split": "test", "project": str(proj), "detection_metric": "map50",
                "inference_precision": "fp16", "warmup_batches": 1,
            })
            det_eval.worker(str(req_path))
            res = json.loads((proj / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(res["inference_precision"], expected_prec)
            self.assertEqual(res["latency"]["precision"], expected_prec)
            self.assertEqual(res["latency"]["warmup_batches"], 1)
            self.assertIs(res["confusion_matrix"]["available"], True)
            self.assertGreater(float(np.asarray(res["confusion_matrix"]["raw"], dtype=float).sum()), 0.0)

    def test_cli_evaluate_v2_end_to_end_multi_seed_orchestration(self):
        import yaml
        from ultralytics import YOLO
        from research_pipeline.integrity import runtime
        from research_pipeline.v2.ledger import Ledger
        from research_pipeline.v2.report import publish
        from research_pipeline.v2.schema import DETECTION_METRICS, measurements
        from research_pipeline.v2.selection import select

        ckpt_src = Path("weights/yolo11n.pt").resolve()
        coco_names = [YOLO(str(ckpt_src)).names[i] for i in range(len(YOLO(str(ckpt_src)).names))]

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "opt_run"
            data_dir = run_dir / "full_data"
            sub_dir = run_dir / "subset_data"
            for sp in ("train", "val", "test"):
                (data_dir / "images" / sp).mkdir(parents=True, exist_ok=True)
                (data_dir / "labels" / sp).mkdir(parents=True, exist_ok=True)
            sub_dir.mkdir(parents=True, exist_ok=True)

            split_imgs = {}
            for sp, color in (("train", (20, 40, 60)), ("val", (80, 100, 120)), ("test", (140, 160, 180))):
                img_p = _write_img(data_dir / "images" / sp / f"{sp}_0.png", color, (64, 64))
                (data_dir / "labels" / sp / f"{sp}_0.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
                split_imgs[sp] = img_p

            data_yaml_text = yaml.safe_dump({
                "path": data_dir.as_posix(),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": coco_names,
            }, sort_keys=False)
            (data_dir / "data.yaml").write_text(data_yaml_text, encoding="utf-8")
            (sub_dir / "data.yaml").write_text(data_yaml_text, encoding="utf-8")

            manifest = {
                "schema_version": 1,
                "task": "detection",
                "names": coco_names,
                "splits": {
                    sp: [{"image": str(p), "sha256": sha(p), "boxes": [[0, 0.5, 0.5, 0.5, 0.5]]} for sp, p in split_imgs.items()][i:i+1]
                    for i, sp in enumerate(("train", "val", "test"))
                },
            }
            data_files = {
                "full_data/data.yaml": sha(data_dir / "data.yaml"),
                "subset_data/data.yaml": sha(sub_dir / "data.yaml"),
            }
            cfg = {
                "model": "yolo11n.yaml", "imgsz": 64, "batch": 1, "accumulation": 1,
                "freeze": 0, "precision": "fp32", "optimizer": "AdamW",
                "lr0": 0.001, "weight_decay": 0.0005,
            }
            p = lock_protocol({
                "task": "detection",
                "method": "adaptive_full",
                "mode": "online",
                "selection_rule": "weighted",
                "implementation_version": "0.11.0",
                "configs": [cfg],
                "training_seeds": [17, 29],
                "search_seed": 17,
                "full_epochs": 2,
                "probe_epochs": 1,
                "budget_seconds": 120,
                "trial_timeout": 60,
                "objective": asdict(Objective()),
                "hardware": {"device": "cpu", "host_label": "unit-host", "cpu": "unit-cpu"},
                "runtime_versions": runtime(),
                "scheduler": {"policy_version": 3, "cleanup_seconds": 1, "probe_fraction": 0.1, "cost_margin": 1.25, "rungs": [1, 2]},
                "acquisition": {"version": 2, "kernel_ridge": 0.05, "length_scale": 1.0, "exploration": 0.25, "novelty_weight": 0.05},
                "execution_policy": {
                    "version": 2,
                    "integrity": "stat_cached_between_trials_full_hash_boundaries",
                    "budget": "end_to_end_soft_deadline_with_measured_finalization_reserve",
                    "data_loader": {"workers": "auto", "cache_mb": 64.0, "pin_memory": "auto"},
                },
                "example_counts": {"train": 1, "val": 1},
                "num_classes": len(coco_names),
                "memory_kind": "process_tree_rss_mib_not_vram",
                "measurement_environment": "uncontended_preflight",
                "preparation_seconds": 0.5,
                "source_files": {},
                "data_files": data_files,
                "model_files": {},
                "dataset_source": str(data_dir / "data.yaml"),
                "dataset_content_id": "unit-eval-cli-domain",
                "dataset_sha256": digest(manifest),
                "subset_sha256": digest(manifest),
                "detection_metric": "map50",
                "quality_definition": DETECTION_METRICS["map50"],
                "run_directory": str(run_dir),
            })
            write_json(run_dir / "protocol.json", p)
            write_json(run_dir / "dataset.json", manifest)
            write_json(run_dir / "subset.json", manifest)
            ledger = Ledger(run_dir / "events.jsonl")
            ledger.emit("session_started", additional_budget_seconds=120, resume=False, preparation_charged_online=True)

            results = {"0": []}
            for att_no, s in enumerate((17, 29), start=1):
                trial = run_dir / "trials" / f"attempt_{att_no:05d}"
                trial.mkdir(parents=True)
                ckpt = trial / "best.pt"
                shutil.copyfile(ckpt_src, ckpt)
                req = {
                    "config": cfg, "seed": s, "full_epochs": 2, "trial_epochs": 2,
                    "mode": "train", "resume": None,
                    "manifest": str(run_dir / "dataset.json"),
                    "data": str(data_dir / "data.yaml"),
                    "protocol_sha256": p["protocol_sha256"],
                    "data_loader": p["execution_policy"]["data_loader"],
                    "detection_metric": "map50",
                }
                raw = {
                    "index": 0, "seed": s, "status": "ok", "mode": "train",
                    "epochs_completed": 2, "primary_metric": "map50",
                    "quality": 0.85, "best_quality": 0.85,
                    "best_epoch": 2, "best_checkpoint_epoch": 2,
                    "best_map50": 0.85, "best_map50_95": 0.55,
                    "at_best_epoch": {"epoch": 2, "map50": 0.85, "map50_95": 0.55, "precision": 0.9, "recall": 0.8, "f1": 0.847},
                    "learning_curve": [
                        {"epoch": 1, "map50": 0.70, "map50_95": 0.40, "precision": 0.7, "recall": 0.7, "f1": 0.7},
                        {"epoch": 2, "map50": 0.85, "map50_95": 0.55, "precision": 0.9, "recall": 0.8, "f1": 0.847},
                    ],
                    "memory_mb": 150.0, "worker_wall_seconds": 5.0, "measured_loop_seconds": 4.0,
                    "optimizer_steps": 4, "training_loss_finite": True,
                    "actual_config_verified": True, "best_checkpoint": str(ckpt),
                    "checkpoint_sha256": sha(ckpt), "config": cfg,
                }
                write_json(trial / "request.json", req)
                write_json(trial / "result.json", raw)
                attempt_id = f"att-{att_no}"
                parent = {
                    **raw, "attempt_id": attempt_id, "trial_directory": str(trial),
                    "request_sha256": sha(trial / "request.json"), "raw_sha256": sha(trial / "result.json"),
                }
                write_json(trial / "parent_result.json", parent)
                ledger.emit("trial_started", attempt_id=attempt_id)
                ledger.emit("trial_completed", attempt_id=attempt_id, parent_sha256=sha(trial / "parent_result.json"),
                            worker_seconds=5.0, status="ok", directory=str(trial))
                results["0"].append(parent)

            rows = measurements(p["configs"], results, p["training_seeds"], 2, Objective(**p["objective"]))
            state = {
                "results": results, "probes": [], "actions": [], "ordering": [0],
                "decision": select(rows, p), "acquisition_history": [],
            }
            write_json(run_dir / "state.json", state)
            state_hash = digest(state)
            (run_dir / "commits").mkdir()
            write_json(run_dir / "commits" / f"{state_hash}.json", state)
            ledger.emit("selection_committed", session_seconds=12.0, original_budget_seconds=120,
                        this_execution_budget_seconds=120, decision=state["decision"],
                        state_sha256=state_hash, state_path=f"commits/{state_hash}.json")
            write_json(run_dir / "decision.locked.json", state["decision"])
            write_json(run_dir / "status.json", {"phase": "complete"})
            publish(run_dir, p, state, display=False)

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli.main(argv=[
                    "evaluate-v2", "--run", str(run_dir),
                    "--seed-scope", "all", "--warmup-batches", "1", "--timeout-seconds", "120",
                ])
            self.assertEqual(rc, 0)

            history = json.loads((run_dir / "test_history.json").read_text(encoding="utf-8"))
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["status"], "evaluated")
            self.assertEqual(history[0]["seed_scope"], "all")
            self.assertEqual(history[0]["evaluated_seeds"], [17, 29])

            eval_out = Path(history[0]["output"])
            for fname in ("result.json", "evaluation.json", "test_summary.json", "EVALUATION.md"):
                self.assertTrue((eval_out / fname).is_file(), fname)
            self.assertTrue((run_dir / "evaluation" / "test_summary.json").is_file())
            self.assertTrue((eval_out / "seed_17" / "result.json").is_file())
            self.assertTrue((eval_out / "seed_29" / "result.json").is_file())

            ev_doc = json.loads((eval_out / "evaluation.json").read_text(encoding="utf-8"))
            res_doc = json.loads((eval_out / "result.json").read_text(encoding="utf-8"))
            sum_doc = json.loads((eval_out / "test_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(ev_doc, res_doc)
            json.dumps(sum_doc, allow_nan=False)
            self.assertEqual(sum_doc["evaluated_seeds"], [17, 29])
            self.assertTrue(sum_doc["latency"]["available"])
            self.assertEqual(sum_doc["latency"]["n_seeds"], 2)
            self.assertIs(sum_doc["latency"]["rect"], False)
            self.assertTrue(sum_doc["pr_curve"]["available"])
            self.assertTrue(sum_doc["f1_curve"]["available"])
            self.assertTrue(sum_doc["confusion_matrix"]["available"])
            self.assertGreater(float(np.asarray(sum_doc["confusion_matrix"]["sum_raw"], dtype=float).sum()), 0.0)
            self.assertGreater(float(np.asarray(sum_doc["confusion_matrix"]["normalized"], dtype=float).sum()), 0.0)
            self.assertIsNotNone(sum_doc["validation_vs_test"]["test_mean"])
            self.assertIn("Tốc độ suy luận", (eval_out / "EVALUATION.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
