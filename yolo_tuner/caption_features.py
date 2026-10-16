"""Verified pretrained encoder and bounded disk-backed frozen image features."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


WEIGHT_NAME = "resnet18-f37072fd.pth"


def dataset_signature(manifest):
    """Relocation-safe identity of train/val images and targets; excludes test."""
    splits = {}
    for split in ("train", "val"):
        rows = [{"sha256": r.get("sha256") or r["image"], "captions": sorted(r["captions"])}
                for r in manifest["splits"][split]]
        splits[split] = sorted(rows, key=lambda r: json.dumps(r, sort_keys=True, ensure_ascii=False))
    return hashlib.sha256(json.dumps(splits, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def cache_matches(index_path, manifest, config):
    try:
        metadata = json.loads(Path(index_path).read_text(encoding="utf-8"))
        recipe = metadata["recipe"]
        images = [{"image": r["image"], "sha256": r.get("sha256")}
                  for split in ("train", "val") for r in manifest["splits"][split]]
        return (Path(metadata["data"]).is_file() and metadata["images"] == images
                and recipe["imgsz"] == config["imgsz"] and recipe["grid"] == config["feature_grid"]
                and recipe["weight_sha256"] == config["encoder_sha256"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def ensure_weights(offline=False):
    import torch
    from torchvision.models import ResNet18_Weights
    path = Path(torch.hub.get_dir()) / "checkpoints" / WEIGHT_NAME
    if not path.is_file():
        if offline:
            raise FileNotFoundError("Offline: pretrained ResNet18 weights are not cached")
        ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not digest.startswith("f37072fd"):
        raise ValueError("Cached ResNet18 checksum differs from official weights; replace that file explicitly")
    return path, digest


def image_tensor(path, config):
    import numpy as np
    import torch
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert("RGB")
        if config.get("encoder") == "resnet18":
            from torchvision.models import ResNet18_Weights
            transform = ResNet18_Weights.IMAGENET1K_V1.transforms(
                crop_size=config["imgsz"], resize_size=round(config["imgsz"] * 256 / 224))
            return transform(image)
        image = image.resize((config["imgsz"], config["imgsz"]))
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255


def prepare(manifest, config, folder, device="cpu", batch=8):
    import numpy as np
    import torch
    from .vision_models import spatial_encoder
    if batch < 1:
        raise ValueError("Feature batch must be positive")
    _, weight_hash = ensure_weights(offline=True)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    records = [r for split in ("train", "val") for r in manifest["splits"][split]]
    if not records:
        raise ValueError("Cannot build empty train/val feature cache")
    recipe = {"encoder": "resnet18", "weight_sha256": weight_hash, "imgsz": config["imgsz"],
              "grid": config.get("feature_grid", 7), "preprocessing": "torchvision_imagenet_center_crop_v1"}
    key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:16]
    cache = folder / key
    cache.mkdir(exist_ok=True)
    # Cache identity includes content hashes, paths/order and preprocessing.
    images = [{"image": r["image"], "sha256": r.get("sha256")} for r in records]
    identity = hashlib.sha256(json.dumps(images, sort_keys=True).encode()).hexdigest()[:16]
    index_path = cache / f"{identity}.json"
    data_path = cache / f"{identity}.npy"
    if index_path.exists() and data_path.exists():
        return index_path
    target = torch.device(device if device in {"cpu", "mps"} or device.startswith("cuda:") else f"cuda:{device}")
    encoder, _ = spatial_encoder(config, load_pretrained=True)
    encoder.to(target).eval()
    grid = recipe["grid"]
    temporary = cache / f"{identity}.tmp.npy"
    import shutil
    required = len(records) * grid * grid * 512 * 2 + 1024
    free = shutil.disk_usage(cache).free
    if required > free * 0.9:
        raise OSError(f"Frozen feature cache needs {required / 2**20:.0f} MiB; available disk {free / 2**20:.0f} MiB")
    array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16,
                                     shape=(len(records), grid * grid, 512))
    with torch.inference_mode():
        for start in range(0, len(records), batch):
            group = records[start:start + batch]
            tensors = torch.stack([image_tensor(r["image"], config) for r in group]).to(target)
            features = torch.nn.functional.adaptive_avg_pool2d(encoder(tensors), (grid, grid)).flatten(2).transpose(1, 2)
            array[start:start + len(group)] = features.cpu().numpy().astype(np.float16)
            if start == 0 or start + batch >= len(records) or start // batch % 20 == 0:
                print(f"Feature cache: {min(start + batch, len(records))}/{len(records)} images (train/val only)", flush=True)
    array.flush()
    del array
    temporary.replace(data_path)
    metadata = {"recipe": recipe, "data": str(data_path.resolve()), "images": images,
                "shape": [len(records), grid * grid, 512], "test_included": False}
    tmp_index = index_path.with_suffix(".tmp")
    tmp_index.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    tmp_index.replace(index_path)
    return index_path


class FeatureStore:
    def __init__(self, index_path):
        import numpy as np
        self.metadata = json.loads(Path(index_path).read_text(encoding="utf-8"))
        self.array = np.load(self.metadata["data"], mmap_mode="r", allow_pickle=False)
        if list(self.array.shape) != self.metadata["shape"]:
            raise ValueError("Feature cache shape differs from metadata")
        self.indices = {r["image"]: i for i, r in enumerate(self.metadata["images"])}
        self.hashes = {r["image"]: r.get("sha256") for r in self.metadata["images"]}

    def get(self, record):
        import numpy as np
        import torch
        if record.get("sha256") != self.hashes.get(record["image"]):
            raise ValueError("Image content identity differs from feature cache")
        return torch.from_numpy(np.array(self.array[self.indices[record["image"]]], dtype=np.float32, copy=True))

    def close(self):
        if self.array is not None:
            self.array._mmap.close()
            self.array = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--manifest")
    parser.add_argument("--folder")
    parser.add_argument("--result")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    path, digest = ensure_weights(args.offline)
    result = {"path": str(path), "sha256": digest}
    if args.manifest:
        import torch
        torch.set_num_threads(4)
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        index = prepare(manifest, {"encoder": "resnet18", "imgsz": 224, "feature_grid": 7},
                        args.folder, args.device, args.batch)
        result["feature_cache"] = str(index.resolve())
    if args.result:
        Path(args.result).write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
