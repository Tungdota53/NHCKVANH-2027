"""Locked, bounded CPU image decoding and Windows-spawn-safe data loaders.

The cache contains decoded CPU images, never encoder activations: a trainable
encoder still executes for every sample. Preprocessing matches the legacy worker.
"""

from collections import OrderedDict, defaultdict
import math
import random
import time


def resolve_policy(policy, device, train_images, logical_cpus=1, available_ram_mb=None):
    requested = dict(policy or {})
    workers = requested.get("workers", "auto")
    cuda = str(device) not in {"cpu", "mps"}
    if workers == "auto":
        # Spawn/import is expensive on Windows; small probes stay synchronous.
        workers = min(2, max(0, int(logical_cpus) // 4)) if cuda and train_images >= 8192 else 0
    if type(workers) is not int or not 0 <= workers <= 8:
        raise ValueError("data loader workers must be auto or an integer in 0..8")
    cache = requested.get("cache_mb", requested.get("image_cache_mb", 64))
    if isinstance(cache, bool) or not isinstance(cache, (int, float)) or not math.isfinite(cache) or cache < 0:
        raise ValueError("image cache must be finite and nonnegative MiB")
    pin = requested.get("pin_memory", "auto")
    if not (type(pin) is bool or isinstance(pin, str) and pin in ("auto", "on", "off")):
        raise ValueError("pin_memory must be auto/on/off or bool")
    pin = cuda if pin == "auto" else pin in ("on", True)
    # Pinning is useful only for CUDA; do not reserve pinned pages on CPU/MPS.
    pin = pin and cuda
    if available_ram_mb is None:
        import psutil
        available_ram_mb = psutil.virtual_memory().available / 2**20
    if not math.isfinite(available_ram_mb) or available_ram_mb < 0:
        raise ValueError("available RAM must be finite and nonnegative")
    # This is only a decoded-cache cap, not a promise that model/queues fit RAM.
    effective_cache = min(cache, available_ram_mb * 0.10)
    total_bytes = int(effective_cache * 2**20)
    # Train and val each have worker-local caches. Generation also uses the
    # parent val cache. Keep all these copies within the declared total cap.
    instances = 2 * (workers + 1) if workers else 2
    return {"requested": requested, "workers": workers, "pin_memory": pin,
            "non_blocking": pin and cuda, "cache_mb": float(effective_cache),
            "cache_requested_mb": float(cache), "cache_ram_cap_fraction": 0.10,
            "cache_available_ram_mb_at_resolution": float(available_ram_mb),
            "cache_bytes_per_instance": total_bytes // instances,
            "cache_budget_scope": "sum of decoded-image caches across train/val parent and worker processes; excludes loader queues/tensors",
            "prefetch_factor": 2 if workers else None, "persistent_workers": workers > 0,
            "auto_workers_threshold_images": 8192, "worker_start_method": "spawn" if workers else None}


class DecodedImageCache:
    def __init__(self, limit_bytes=0):
        self.limit_bytes = int(limit_bytes)
        if self.limit_bytes < 0:
            raise ValueError("cache limit must be nonnegative")
        self.items = OrderedDict()
        self.bytes = self.hits = self.misses = 0

    def get(self, path, config):
        key = (str(path), config["imgsz"], config.get("encoder", "tiny"))
        value = self.items.pop(key, None)
        if value is not None:
            self.items[key] = value
            self.hits += 1
            return value
        from yolo_tuner.caption_features import image_tensor
        self.misses += 1
        value = image_tensor(path, config)
        size = value.numel() * value.element_size()
        if size <= self.limit_bytes:
            while self.items and self.bytes + size > self.limit_bytes:
                _, old = self.items.popitem(last=False)
                self.bytes -= old.numel() * old.element_size()
            self.items[key] = value
            self.bytes += size
        return value

    def summary(self):
        return {"hits": self.hits, "misses": self.misses, "resident_bytes": self.bytes,
                "limit_bytes": self.limit_bytes, "resident_images": len(self.items)}


class ImageRecords:
    """Map-style dataset. Module-level class is picklable under Windows spawn."""
    def __init__(self, records, config, task, vocabulary=None, maximum=48, cache_bytes=0):
        self.config, self.task = dict(config), task
        self.vocabulary, self.maximum = dict(vocabulary or {}), maximum
        self.items = [(record, caption) for record in records for caption in record["captions"]] if task == "captioning" else [(record, None) for record in records]
        self.cache = DecodedImageCache(cache_bytes)
        self.io_stats = defaultdict(lambda: [0.0, 0])

    def __getstate__(self):
        state = dict(self.__dict__)
        state["cache"] = DecodedImageCache(self.cache.limit_bytes)
        # defaultdict(lambda) is not picklable. Child statistics are intentionally
        # not silently presented as parent totals.
        state["io_stats"] = {}
        return state

    def __len__(self):
        return len(self.items)

    def image(self, record):
        return self.cache.get(record["image"], self.config)

    def __getitem__(self, index):
        import torch
        from yolo_tuner.vision_worker import tokens
        start = time.monotonic()
        record, caption = self.items[index]
        image = self.image(record)
        observed = self.io_stats.setdefault(record.get("source", "unknown"), [0.0, 0])
        observed[0] += time.monotonic() - start
        observed[1] += 1
        if self.task == "captioning":
            ids = [2] + [self.vocabulary.get(t, 1) for t in tokens(caption)[:self.maximum - 2]] + [3]
            ids += [0] * (self.maximum - len(ids))
            return image, torch.tensor(ids, dtype=torch.long)
        if self.task == "classification":
            return image, torch.tensor(record["class_id"], dtype=torch.long)
        raise ValueError("Unsupported v2 image task")


def seed_worker(worker_id):
    import numpy as np
    import torch
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


def make_loader(dataset, batch, policy, shuffle=False, generator=None):
    from torch.utils.data import DataLoader
    options = {"batch_size": batch, "shuffle": shuffle, "num_workers": policy["workers"],
               "pin_memory": policy["pin_memory"], "generator": generator,
               "worker_init_fn": seed_worker, "drop_last": False}
    if policy["workers"]:
        options.update(persistent_workers=True, prefetch_factor=2, multiprocessing_context="spawn")
    return DataLoader(dataset, **options)


def observed_loaders(policy, loaders, datasets):
    matches = all(loader.num_workers == policy["workers"] and loader.pin_memory == policy["pin_memory"] and not loader.drop_last for loader in loaders)
    return {**policy, "verified": matches, "observed_workers": [loader.num_workers for loader in loaders],
            "observed_pin_memory": [loader.pin_memory for loader in loaders],
            "parent_cache": [dataset.cache.summary() for dataset in datasets],
            "cache_stats_scope": "parent process only; worker-local cache hits are not aggregated"}


def detection_loader_builder(original, policy):
    """Cap both YOLO train and val pools; no unbounded Ultralytics RAM cache."""
    def build(dataset, batch, workers, **kwargs):
        kwargs["pin_memory"] = policy["pin_memory"]
        return original(dataset, batch=batch, workers=min(workers, policy["workers"]), **kwargs)
    return build
