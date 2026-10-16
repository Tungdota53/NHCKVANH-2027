"""Strict normalized manifests and external-path YAML/COCO without fake labels."""

from collections import defaultdict
import json
import math
from pathlib import Path
import random

from yolo_tuner import adapters as legacy
from .input_diagnostics import check_explicit_file, no_images_error


def check_manifest(data):
    task, names = data["task"], data["names"]
    if task not in {"detection", "classification", "captioning"}:
        raise ValueError("Only genuine supervised detection/classification/captioning are supported")
    if task != "captioning" and (not names or len(set(names)) != len(names)):
        raise ValueError("Names must be a nonempty unique taxonomy")
    if task == "classification" and len(names) < 2:
        raise ValueError("Single-class classification accuracy is degenerate; need at least two classes")
    hashes, groups = {}, {}
    for split, records in data["splits"].items():
        if split not in {"train", "val", "test"}:
            raise ValueError("Unknown split")
        for r in records:
            if r["sha256"] in hashes and hashes[r["sha256"]] != split:
                raise ValueError("Exact image content crosses split boundary")
            hashes[r["sha256"]] = split
            group = r.get("group_id")
            if group is not None:
                if group in groups and groups[group] != split:
                    raise ValueError("Declared scene/video/subject group crosses splits")
                groups[group] = split
            if task == "classification" and (type(r.get("class_id")) is not int or not 0 <= r["class_id"] < len(names)):
                raise ValueError("Invalid class_id/taxonomy")
            if task == "captioning" and (not r.get("captions") or any(not isinstance(c, str) or not c.strip() for c in r["captions"])):
                raise ValueError("Caption labels must be nonempty strings")
            if task == "detection":
                for box in r.get("boxes", []):
                    if (len(box) != 5 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)
                            or box[0] != int(box[0]) or not 0 <= box[0] < len(names)):
                        raise ValueError("Detection boxes must be normalized [class,xc,yc,w,h]")
                    _, x, y, w, h = box
                    if not 0 < w <= 1 or not 0 < h <= 1 or min(x - w / 2, y - h / 2) < -1e-6 or max(x + w / 2, y + h / 2) > 1 + 1e-6:
                        raise ValueError("Normalized box outside image bounds")
    if not data["splits"].get("train") or not data["splits"].get("val"):
        raise ValueError("Independent nonempty train/validation required")
    if task == "detection" and any(not any(r.get("boxes") for r in data["splits"][s]) for s in ("train", "val")):
        raise ValueError("Detection mAP requires real positive boxes in train and validation; no reconstruction fallback")
    data["grouping_status"] = "declared_groups_verified_disjoint" if groups else "true_groups_not_supplied; near-duplicate scene leakage not excluded"
    data["validation_ready"] = True
    return data


def finalize(source, task, records, names, seed):
    known = {r.get("group_id") for r in records if r.get("group_id") is not None}
    if not known:
        return check_manifest(legacy.finalize(source, task, records, names, [], seed))
    from PIL import Image
    # Explicit groups are never split by the legacy image-level splitter.
    for r in records:
        with Image.open(r["image"]) as im:
            im.load()
        r["sha256"] = legacy.file_hash(Path(r["image"]))
    splits = {s: [r for r in records if r["split"] == s] for s in ("train", "val", "test")}
    unsplit = [r for r in records if r["split"] == "unsplit"]
    if not splits["val"]:
        unsplit += splits["train"]
        splits["train"] = []
    if unsplit:
        grouped = defaultdict(list)
        for r in unsplit:
            grouped[r.get("group_id") or r["sha256"]].append(r)
        keys = sorted(grouped)
        random.Random(seed).shuffle(keys)
        nval = max(1, round(len(keys) * 0.2))
        if len(keys) < 2:
            raise ValueError("At least two independent groups required to derive validation")
        for i, key in enumerate(keys):
            split = "val" if i < nval else "train"
            for r in grouped[key]:
                r["split"] = split
                splits[split].append(r)
    data = {"schema_version": 1, "task": task, "source": str(source), "seed": seed, "names": names,
            "splits": splits, "warnings": [], "sampled_image_sizes": [],
            "counts": {s: len(rs) for s, rs in splits.items()},
            "captions": {s: sum(len(r.get("captions", [])) for r in rs) for s, rs in splits.items()}}
    return check_manifest(data)


def discover(source, seed=101, task="auto"):
    source = Path(source).expanduser().resolve()
    check_explicit_file(source)
    root = source.parent
    if source.is_file() and source.suffix.lower() in {".yaml", ".yml"} and task in {"auto", "detection"}:
        parsed = legacy.yolo_yaml(source)
        if parsed:
            records, names = parsed
            return finalize(source, "detection", records, names, seed)
    if source.is_file() and source.suffix.lower() == ".json":
        doc = json.loads(source.read_text(encoding="utf-8-sig"))
        if doc.get("task") in {"detection", "classification", "captioning"} and isinstance(doc.get("splits"), dict):
            actual_task = doc["task"]
            if task not in {"auto", actual_task}:
                raise ValueError("Manifest task differs from requested task")
            records = []
            for split, rows in doc["splits"].items():
                if split not in {"train", "val", "test", "unsplit"}:
                    raise ValueError("Unsupported manifest split")
                for row in rows:
                    path = Path(row["image"])
                    path = path if path.is_absolute() else root / path
                    r = {**row, "image": str(path.resolve()), "source": str(root), "split": split}
                    if r.get("group_id") is not None:
                        r["group_id"] = str(r["group_id"])
                    records.append(r)
            names = doc.get("names", [])
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names, key=lambda k: int(k))]
            return finalize(source, actual_task, records, list(map(str, names)), seed)
        if isinstance(doc.get("images"), list) and isinstance(doc.get("annotations"), list):
            from collections import defaultdict
            index = defaultdict(list)
            cap = any("caption" in a for a in doc["annotations"])
            actual_task = "captioning" if cap and task in {"auto", "captioning"} else "detection"
            if task not in {"auto", actual_task}:
                raise ValueError("COCO task differs")
            parsed = legacy.caption_records(source, root, index) if actual_task == "captioning" else legacy.coco_records(source, root, index)
            if parsed:
                records, names = (parsed, []) if actual_task == "captioning" else parsed
                by_path = {str(legacy.resolve_image(i["file_name"], source, root, index)): i for i in doc["images"]}
                for r in records:
                    metadata = by_path[r["image"]]
                    if metadata.get("split") in {"train", "val", "test"}:
                        r["split"] = metadata["split"]
                    for key in ("group_id", "scene_id", "video_id", "subject_id"):
                        if metadata.get(key) is not None:
                            r["group_id"] = str(metadata[key])
                            break
                return finalize(source, actual_task, records, names, seed)
    try:
        return check_manifest(legacy.discover(source, seed, task))
    except ValueError as exc:
        if "No local image files found" in str(exc):
            raise no_images_error(source) from exc
        raise
