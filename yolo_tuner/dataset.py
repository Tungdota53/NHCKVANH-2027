"""Local YOLO detection datasets, deterministic multilabel/small-object sampling."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def label_path(image):
    parts = list(image.parts)
    indices = [i for i, p in enumerate(parts) if p.lower() == "images"]
    if indices:
        parts[indices[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def images_from_entry(entry, root):
    result = []
    for raw in entry if isinstance(entry, list) else [entry]:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if path.is_dir():
            result.extend(p.resolve() for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.suffix.lower() == ".txt" and path.is_file():
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                item = Path(line.strip())
                # Ultralytics resolves ./ paths relative to the image-list file.
                item = path.parent / item if not item.is_absolute() else item
                result.append(item.resolve())
        elif path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            result.append(path)
        else:
            raise ValueError(f"Dataset split path does not exist: {path}")
    result = sorted(set(result), key=str)
    for path in result:
        if not path.is_file():
            raise ValueError(f"Image does not exist: {path}")
    if not result:
        raise ValueError("Dataset split has no images")
    return result


def annotations(image, nc):
    label = label_path(image)
    if not label.exists():
        return set(), 0, 0  # Standard YOLO negative image.
    classes, count, small = set(), 0, 0
    for line in label.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Detection requires cls x y w h (5 columns): {label}")
        cls_value, x, y, w, h = map(float, values)
        cls = int(cls_value)
        if cls != cls_value or not 0 <= cls < nc:
            raise ValueError(f"Invalid class ID in {label}")
        if not all(0 <= v <= 1 for v in (x, y, w, h)) or w <= 0 or h <= 0:
            raise ValueError(f"Invalid normalized bounding box in {label}")
        classes.add(cls)
        count += 1
        small += w * h < 0.01  # Relative area, NOT COCO's pixel-area definition.
    return classes, count, small


def sample(images, features, size, seed):
    """Greedy rare-label coverage, then inverse-frequency weighted sampling.

    Images can have multiple classes. Small-object and negative-image strata are
    explicit labels. This is a deterministic heuristic, not exact stratification.
    """
    size = min(size, len(images))
    if size < 1:
        raise ValueError("Subset sizes must be positive")
    rng = random.Random(seed)
    shuffled = list(images)
    rng.shuffle(shuffled)
    labels = {}
    for p in shuffled:
        classes, count, small = features[p]
        labels[p] = set(classes) | ({"small"} if small else set()) | ({"negative"} if not count else set())
    freq = Counter(label for tags in labels.values() for label in tags)
    uncovered, selected = set(freq), []
    available = set(shuffled)
    while uncovered and len(selected) < size:
        best = max((p for p in shuffled if p in available),
                   key=lambda p: sum(1 / freq[k] for k in labels[p] & uncovered))
        selected.append(best)
        available.remove(best)
        uncovered -= labels[best]
    priorities = []
    for p in shuffled:
        if p in available:
            weight = sum(1 / freq[k] for k in labels[p]) or 1 / len(images)
            priorities.append((-math.log(max(rng.random(), 1e-15)) / weight, p))
    selected.extend(p for _, p in sorted(priorities, key=lambda item: item[0])[:size-len(selected)])
    return sorted(selected, key=str)


def prepare(data_yaml, output, train_size=400, val_size=150, seed=17):
    import yaml

    source = Path(data_yaml).resolve()
    spec = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
    if not isinstance(spec, dict) or "train" not in spec or "val" not in spec:
        raise ValueError("Dataset YAML requires train and val splits")
    root = Path(spec.get("path") or source.parent).expanduser()
    root = root.resolve() if root.is_absolute() else (source.parent / root).resolve()
    names = spec.get("names")
    if isinstance(names, dict):
        if {str(k) for k in names} != {str(i) for i in range(len(names))}:
            raise ValueError("Dataset names IDs must be contiguous from zero")
        names = [names[i] if i in names else names[str(i)] for i in range(len(names))]
    if not isinstance(names, list) or not names:
        raise ValueError("Dataset YAML requires non-empty names")
    train, val = [images_from_entry(spec[k], root) for k in ("train", "val")]
    if set(train) & set(val):
        raise ValueError("Train/val image overlap detected. Create disjoint splits before tuning.")
    features = {p: annotations(p, len(names)) for p in train + val}
    if not any(features[p][1] for p in train):
        raise ValueError("No detection boxes in training data. Use the auto pipeline for captioning or unlabeled image pretraining.")
    train_sub = sample(train, features, train_size, seed)
    val_sub = sample(val, features, val_size, seed + 1)
    folder = Path(output).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    for key, paths in (("train", train_sub), ("val", val_sub)):
        (folder / f"{key}.txt").write_text("\n".join(p.as_posix() for p in paths) + "\n", encoding="utf-8")
    subset = folder / "subset.yaml"
    subset.write_text(yaml.safe_dump({"path": folder.as_posix(), "train": "train.txt", "val": "val.txt", "names": names},
                                     allow_unicode=True), encoding="utf-8")
    digest = hashlib.sha256()
    for p in train_sub + val_sub:
        digest.update(str(p).encode())
        digest.update(str(p.stat().st_size).encode())
        label = label_path(p)
        digest.update(label.read_bytes() if label.exists() else b"negative")
    stats = {
        "source": str(source), "subset_yaml": str(subset), "seed": seed,
        "train_images": len(train), "val_images": len(val),
        "train_subset_images": len(train_sub), "val_subset_images": len(val_sub),
        "classes": len(names), "subset_fingerprint": digest.hexdigest(),
        "fingerprint_definition": "absolute paths, image file sizes and label bytes; not image content hash",
        "objects": sum(features[p][1] for p in train),
        "small_object_ratio": sum(features[p][2] for p in train) / max(1, sum(features[p][1] for p in train)),
        "sampling": "multilabel rare-class coverage + small-object/negative strata",
        "missing_classes_train_subset": sorted(set(range(len(names))) - set().union(*(features[p][0] for p in train_sub))),
        "missing_classes_val_subset": sorted(set(range(len(names))) - set().union(*(features[p][0] for p in val_sub))),
    }
    (folder / "dataset.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats
