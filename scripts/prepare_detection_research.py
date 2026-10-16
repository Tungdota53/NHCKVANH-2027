"""Convert genuine Penn-Fudan instance masks to YOLO boxes; no synthetic labels.

Original annotation sources: https://www.cis.upenn.edu/~jshi/ped_html/
Archive is supplied locally, allowing explicit download/checksum provenance.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import zipfile

import numpy as np
from PIL import Image
import yaml


def prepare(archive, output, seed=101):
    archive, output = Path(archive).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Output already exists; preserve previous dataset and choose another directory")
    output.mkdir(parents=True)
    records = []
    with zipfile.ZipFile(archive) as source:
        images = sorted(n for n in source.namelist() if "/PNGImages/" in n and n.endswith(".png"))
        if len(images) != 170:
            raise ValueError("Expected the official archive's 170 images")
        # Adjacent image indices can show overlapping views. Keep consecutive
        # blocks together; these are conservative filename groups, NOT verified
        # ground-truth scene IDs (the archive supplies no scene metadata).
        groups = {}
        for name in images:
            stem = Path(name).stem
            campus = "Penn" if stem.startswith("Penn") else "Fudan"
            index = int(stem.split("Ped")[-1])
            groups.setdefault((campus, (index - 1) // 10), []).append(name)
        assignment = {}
        rng = random.Random(seed)
        for campus in ("Penn", "Fudan"):
            keys = sorted(k for k in groups if k[0] == campus)
            rng.shuffle(keys)
            for i, key in enumerate(keys):
                split = "test" if i == 0 else "val" if i == 1 else "train"
                for name in groups[key]:
                    assignment[name] = (split, key)
        seen = set()
        for name in images:
            raw = source.read(name)
            checksum = hashlib.sha256(raw).hexdigest()
            if checksum in seen:
                raise ValueError("Duplicate content in official source; revise split before proceeding")
            seen.add(checksum)
            stem = Path(name).stem
            mask_name = name.replace("/PNGImages/", "/PedMasks/").replace(".png", "_mask.png")
            mask_raw = source.read(mask_name)
            mask = np.asarray(Image.open(io.BytesIO(mask_raw)))
            width, height = Image.open(io.BytesIO(raw)).size
            boxes = []
            for instance in np.unique(mask):
                if instance == 0:
                    continue
                ys, xs = np.where(mask == instance)
                x1, x2, y1, y2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
                boxes.append([0, (x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height])
            if not boxes:
                raise ValueError("Expected genuine pedestrian annotations")
            split, group = assignment[name]
            image_path = output / "images" / split / Path(name).name
            label_path = output / "labels" / split / (stem + ".txt")
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(raw)
            label_path.write_text("\n".join(" ".join(str(v) for v in box) for box in boxes)+"\n", encoding="utf-8")
            records.append({"source_image": name, "split": split, "filename_group": list(group),
                            "sha256": checksum, "mask_sha256": hashlib.sha256(mask_raw).hexdigest(), "boxes": boxes})
    spec = {"path": output.as_posix(), "train": "images/train", "val": "images/val", "test": "images/test", "names": ["pedestrian"]}
    (output / "data.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    provenance = {"source": "Penn-Fudan, genuine instance mask-derived boxes", "url": "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip",
                  "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "seed": seed,
                  "grouping_policy": "campus x blocks of10 filename indices; not verified scene identity",
                  "counts": {s: sum(r["split"] == s for r in records) for s in ("train", "val", "test")},
                  "test_usage": "not evaluated or used for config selection during preparation", "records": records}
    (output / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Prepared: {output} | {provenance['counts']} | archive SHA256 {provenance['archive_sha256']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", default="runs/datasets/penn_fudan_research")
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    prepare(args.archive, args.output, args.seed)
