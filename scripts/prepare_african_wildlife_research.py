"""Prepare genuine 4-class African Wildlife YOLO detection dataset with SHA256 deduplication.

Source archive: https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip
Classes: [buffalo, elephant, rhino, zebra]
"""

import argparse
from collections import defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import random
import zipfile

from PIL import Image
import yaml

NAMES = ["buffalo", "elephant", "rhino", "zebra"]


def prepare(archive, output_full, output_benchmark=None, seed=101, per_class_train=40, per_class_val=10, per_class_test=10):
    archive = Path(archive).resolve()
    output_full = Path(output_full).resolve()
    if output_full.exists():
        raise ValueError(f"Output directory already exists: {output_full}")
    if output_benchmark is not None and Path(output_benchmark).resolve().exists():
        raise ValueError(f"Benchmark directory already exists: {output_benchmark}")

    seen_hashes = {}
    duplicates_excluded = []
    clean_by_split = {"train": [], "val": [], "test": []}

    with zipfile.ZipFile(archive) as zf:
        entries = set(zf.namelist())
        images = sorted(
            n for n in entries
            if n.startswith("images/") and not n.endswith("/") and Path(n).suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        # Process train -> val -> test deterministically so any cross-split duplicate stays in train only
        order_rank = {"train": 0, "val": 1, "test": 2}
        images.sort(key=lambda n: (order_rank.get(n.split("/")[1], 99), n))

        for img_entry in images:
            parts = img_entry.split("/")
            split = parts[1]
            if split not in clean_by_split:
                continue
            raw_bytes = zf.read(img_entry)
            digest = hashlib.sha256(raw_bytes).hexdigest()
            if digest in seen_hashes:
                duplicates_excluded.append({"entry": img_entry, "split": split, "kept_in": seen_hashes[digest], "sha256": digest})
                continue
            with Image.open(io.BytesIO(raw_bytes)) as im:
                im.load()
                width, height = im.size
            lbl_entry = "labels/" + "/".join(parts[1:]).rsplit(".", 1)[0] + ".txt"
            lbl_text = zf.read(lbl_entry).decode("utf-8") if lbl_entry in entries else ""
            boxes = []
            for line in lbl_text.splitlines():
                if not line.strip():
                    continue
                vals = [float(x) for x in line.split()]
                if len(vals) != 5 or any(not math.isfinite(v) for v in vals):
                    raise ValueError(f"Malformed label in {lbl_entry}")
                cls_id = int(vals[0])
                if cls_id != vals[0] or not 0 <= cls_id < len(NAMES):
                    raise ValueError(f"Invalid class id in {lbl_entry}: {vals[0]}")
                _, xc, yc, bw, bh = vals
                x1 = max(0.0, xc - bw / 2.0)
                y1 = max(0.0, yc - bh / 2.0)
                x2 = min(1.0, xc + bw / 2.0)
                y2 = min(1.0, yc + bh / 2.0)
                if x2 - x1 <= 1e-6 or y2 - y1 <= 1e-6:
                    continue
                boxes.append([cls_id, (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1])
            if not boxes:
                continue
            seen_hashes[digest] = split
            clean_by_split[split].append({
                "entry": img_entry,
                "ext": Path(img_entry).suffix.lower(),
                "sha256": digest,
                "width": width,
                "height": height,
                "boxes": boxes,
                "primary_class": boxes[0][0],
                "raw_bytes": raw_bytes,
            })

    def write_dataset(target_dir, split_records, description):
        target_dir = Path(target_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=False)
        manifest_records = []
        for split in ("train", "val", "test"):
            for idx, rec in enumerate(split_records[split]):
                safe_stem = f"{split}_{idx:04d}_{rec['sha256'][:12]}"
                img_out = target_dir / "images" / split / f"{safe_stem}{rec['ext']}"
                lbl_out = target_dir / "labels" / split / f"{safe_stem}.txt"
                img_out.parent.mkdir(parents=True, exist_ok=True)
                lbl_out.parent.mkdir(parents=True, exist_ok=True)
                img_out.write_bytes(rec["raw_bytes"])
                lbl_out.write_text(
                    "\n".join(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in rec["boxes"]) + "\n",
                    encoding="utf-8",
                )
                manifest_records.append({
                    "source_entry": rec["entry"],
                    "split": split,
                    "sha256": rec["sha256"],
                    "size": [rec["width"], rec["height"]],
                    "box_count": len(rec["boxes"]),
                    "classes": sorted({b[0] for b in rec["boxes"]}),
                })
        spec = {
            "path": target_dir.as_posix(),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": NAMES,
        }
        (target_dir / "data.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        provenance = {
            "source": "Ultralytics African Wildlife (4 classes: buffalo, elephant, rhino, zebra)",
            "description": description,
            "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip",
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "seed": seed,
            "names": NAMES,
            "counts": {s: len(split_records[s]) for s in ("train", "val", "test")},
            "upstream_duplicates_excluded_count": len(duplicates_excluded),
            "upstream_duplicates_excluded": duplicates_excluded,
            "test_usage": "strictly untouched during preparation and configuration search",
            "records": manifest_records,
        }
        (target_dir / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        print(f"Prepared {target_dir}: counts={provenance['counts']}, excluded_duplicates={len(duplicates_excluded)}")

    write_dataset(output_full, clean_by_split, "Full deduplicated 4-class African Wildlife dataset")

    if output_benchmark is not None:
        rng = random.Random(seed)
        quotas = {"train": per_class_train, "val": per_class_val, "test": per_class_test}
        sub_splits = {"train": [], "val": [], "test": []}
        for split, per_cls in quotas.items():
            by_cls = defaultdict(list)
            for rec in clean_by_split[split]:
                by_cls[rec["primary_class"]].append(rec)
            for cid in range(len(NAMES)):
                pool = list(by_cls[cid])
                rng.shuffle(pool)
                if len(pool) < per_cls:
                    raise ValueError(f"Insufficient class {cid} examples in {split}: {len(pool)} < {per_cls}")
                sub_splits[split].extend(pool[:per_cls])
        write_dataset(
            output_benchmark,
            sub_splits,
            f"Stratified {per_class_train}/{per_class_val}/{per_class_test}-per-class subset preserving official split boundaries",
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--archive", default="runs/datasets/raw/african-wildlife.zip")
    p.add_argument("--output-full", default="runs/datasets/african_wildlife_full")
    p.add_argument("--output-benchmark", default="runs/datasets/african_wildlife_benchmark")
    p.add_argument("--seed", type=int, default=101)
    args = p.parse_args()
    prepare(args.archive, args.output_full, args.output_benchmark, args.seed)
