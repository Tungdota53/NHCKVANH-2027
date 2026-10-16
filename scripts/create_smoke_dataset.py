"""Synthetic geometry dataset for integration testing ONLY, not research results."""

from pathlib import Path
import random


def main():
    from PIL import Image, ImageDraw
    import yaml
    root = Path("runs/smoke_data").resolve()
    rng = random.Random(17)
    for split, size in (("train", 24), ("val", 8)):
        images, labels = root / "images" / split, root / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        for i in range(size):
            image = Image.new("RGB", (160, 160), (30, 30, 30))
            draw = ImageDraw.Draw(image)
            x, y = rng.randint(10, 90), rng.randint(10, 90)
            w, h = rng.randint(15, 50), rng.randint(15, 50)
            draw.rectangle((x, y, x + w, y + h), fill=(230, 70, 30))
            image.save(images / f"{i:03d}.jpg")
            (labels / f"{i:03d}.txt").write_text(f"0 {(x+w/2)/160} {(y+h/2)/160} {w/160} {h/160}\n")
    spec = root / "data.yaml"
    spec.write_text(yaml.safe_dump({"path": root.as_posix(), "train": "images/train", "val": "images/val", "names": ["rectangle"]}))
    print(spec)


if __name__ == "__main__":
    main()
