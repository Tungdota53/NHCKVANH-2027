"""Genuine MNIST domain for heldout predictor diagnostics, not pseudo labels."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolo_tuner.pipeline import write_json


def main():
    from torchvision.datasets import MNIST
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/datasets/mnist_v2")
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if root.exists():
        raise ValueError("Output already exists; use a new output, do not overwrite dataset")
    root.mkdir(parents=True)
    train = MNIST(str(root / "raw"), train=True, download=True)
    test = MNIST(str(root / "raw"), train=False, download=True)
    rng, seen, indices = random.Random(args.seed), set(), {}
    for label in range(10):
        train_ids = [i for i, y in enumerate(train.targets.tolist()) if y == label]
        test_ids = [i for i, y in enumerate(test.targets.tolist()) if y == label]
        rng.shuffle(train_ids); rng.shuffle(test_ids)
        cursor = iter(train_ids)
        for split, dataset, candidates, limit in (("train", train, cursor, 100), ("val", train, cursor, 30), ("test", test, iter(test_ids), 50)):
            folder = root / split / str(label)
            folder.mkdir(parents=True)
            chosen = []
            for i in candidates:
                image, original_label = dataset[i]
                fingerprint = hashlib.sha256(image.tobytes()).hexdigest()
                if fingerprint in seen:
                    continue
                if original_label != label:
                    raise ValueError("Original label mismatch")
                seen.add(fingerprint)
                image.save(folder / f"{i:05d}.png")
                chosen.append(i)
                if len(chosen) == limit:
                    break
            if len(chosen) != limit:
                raise ValueError("Not enough content-disjoint examples")
            indices[f"{split}/{label}"] = chosen
    archives = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "raw").rglob("*.gz")}
    write_json(root / "PROVENANCE.json", {"source": "Torchvision MNIST verified official resource MD5s",
               "documentation": "https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html",
               "seed": args.seed, "indices": indices, "archive_sha256": archives,
               "labels": "genuine original digit labels", "counts": {"train": 1000, "val": 300, "test": 500},
               "purpose": "heldout dataset for T/M predictor diagnostics; not used to retune prior/policy",
               "split": "content-disjoint train/val from official train, test from official test"})
    print(str(root))


if __name__ == "__main__":
    main()
