"""Real-caption development subset; NEVER a full ViECAP or blind-test claim."""

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolo_tuner.adapters import discover
from yolo_tuner.research import digest
from yolo_tuner.pipeline import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train", type=int, default=64)
    parser.add_argument("--val", type=int, default=16)
    parser.add_argument("--test", type=int, default=16)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    manifest = discover(args.source, args.seed, "captioning")
    folder = Path(args.output).resolve()
    if folder.exists():
        raise ValueError("Output already exists; no overwriting dataset artifacts")
    if min(args.train, args.val, args.test) < 1:
        raise ValueError("Subset limits must be positive")
    folder.mkdir(parents=True)
    rng = random.Random(args.seed)
    provenance = {"source": manifest["source"], "source_manifest_sha256": digest(manifest),
                  "purpose": "development subset with genuine original captions, not full ViECAP or new blind test", "splits": {}}
    for split in ("train", "val", "test"):
        records = list(manifest["splits"][split])
        rng.shuffle(records)
        records = records[:getattr(args, split)]
        doc = {"images": [], "annotations": []}
        for i, row in enumerate(records):
            src = Path(row["image"])
            image = folder / "images" / split / (row["sha256"] + src.suffix.lower())
            image.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, image)
            except OSError:
                shutil.copy2(src, image)
            doc["images"].append({"id": i, "file_name": str(image)})
            doc["annotations"].extend({"image_id": i, "caption": caption} for caption in row["captions"])
        write_json(folder / f"captions_{split}.json", doc)
        provenance["splits"][split] = [{"image": r["image"], "sha256": r["sha256"], "captions": r["captions"]} for r in records]
    write_json(folder / "PROVENANCE.json", provenance)
    print(str(folder))


if __name__ == "__main__":
    main()
