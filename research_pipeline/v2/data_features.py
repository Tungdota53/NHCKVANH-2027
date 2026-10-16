"""Small train-only descriptors for point priors; never fit on val/test."""

from collections import Counter
import math


def describe_dataset(manifest):
    import numpy as np
    from PIL import Image
    records = manifest["splits"]["train"]
    labels = Counter(r["class_id"] for r in records if "class_id" in r)
    boxes = [b for r in records for b in r.get("boxes", [])]
    labels.update(int(b[0]) for b in boxes)
    total = sum(labels.values())
    entropy = -sum((n / total) * math.log(n / total) for n in labels.values()) / math.log(max(2, len(manifest["names"]))) if total else 0.0
    stats = []
    for row in sorted(records, key=lambda r: r["image"])[:16]:
        with Image.open(row["image"]) as image:
            pixels = np.asarray(image.convert("RGB").resize((32, 32)), dtype=np.float64) / 255
        edge = (np.abs(np.diff(pixels, axis=0)).mean() + np.abs(np.diff(pixels, axis=1)).mean()) / 2
        stats.append((float(pixels.mean()), float(pixels.std()), float(edge)))
    return {"class_entropy": entropy, "objects_per_image": len(boxes) / max(1, len(records)),
            "small_box_fraction_area_lt_1pct_not_coco": sum(b[3] * b[4] < 0.01 for b in boxes) / max(1, len(boxes)),
            "pixel_mean": sum(s[0] for s in stats) / max(1, len(stats)),
            "pixel_sd": sum(s[1] for s in stats) / max(1, len(stats)),
            "edge_mean": sum(s[2] for s in stats) / max(1, len(stats)), "sample_count": len(stats), "split": "train_only"}
