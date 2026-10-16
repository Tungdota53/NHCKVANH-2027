"""Full-run measurements on one local dataset, usable on Windows/Linux/Colab.

This is an explicit long-running experiment, outside the online search budget.
Repeat with the same stable dataset ID for each source and on multiple GPUs.
"""

import argparse
import csv
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import random
import sys

# Allow direct execution from a source checkout before editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolo_tuner.cli import announce, save_json, session
from yolo_tuner.core import Config, Deadline
from yolo_tuner.dataset import prepare
from yolo_tuner.offline import CATEGORICAL, NUMERIC, TARGETS, input_row
from yolo_tuner.runner import Runner, hardware


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--models", nargs="+", default=["yolo11n.pt", "yolo11s.pt"])
    p.add_argument("--imgsz", nargs="+", type=int, default=[480, 640])
    p.add_argument("--batch", nargs="+", type=int, default=[2, 4, 8])
    p.add_argument("--freeze", nargs="+", type=int, default=[0, 10])
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-runs", type=int, default=12)
    p.add_argument("--hours", type=float, default=6)
    p.add_argument("--run-timeout-hours", type=float, default=2)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--csv", default="measurements.csv")
    p.add_argument("--output", default="runs/offline")
    args = p.parse_args()
    if not args.dataset_id.strip() or args.epochs < 1 or args.max_runs < 1:
        p.error("dataset-id must be nonempty; epochs and max-runs must be positive")
    if any(s < 32 or s % 32 for s in args.imgsz) or min(args.batch) < 1 or min(args.freeze) < 0:
        p.error("imgsz must be positive multiples of 32; batch >=1; freeze >=0")
    if args.device == "cpu" and args.precision != "fp32":
        p.error("CPU collection requires --precision fp32")
    deadline = Deadline(args.hours * 3600)
    folder = session(args.output)
    hw = hardware(args.device)
    stats = prepare(args.data, folder / "subset", seed=args.seed)
    full_stats = {**stats, "train_subset_images": stats["train_images"], "val_subset_images": stats["val_images"]}
    runner = Runner(folder / "trials", str(Path(args.data).resolve()), full_stats, args.device, args.epochs, args.seed)
    configs = [Config(model=m, imgsz=s, batch=b, precision=args.precision, freeze=f)
               for m, s, b, f in itertools.product(args.models, args.imgsz, args.batch, args.freeze)]
    random.Random(args.seed).shuffle(configs)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("dataset_id",) + CATEGORICAL + NUMERIC + TARGETS
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    if exists:
        with csv_path.open(encoding="utf-8-sig", newline="") as file:
            if next(csv.reader(file)) != list(columns):
                p.error("Existing CSV header differs from expected offline schema")
    manifest = {"dataset_id": args.dataset_id, "dataset": stats, "hardware": hw,
                "design": "seeded random order of a finite factorial configuration grid",
                "full_epochs": args.epochs, "seed": args.seed, "attempts": []}
    for config in configs[:args.max_runs]:
        if deadline.remaining < 1:
            break
        result = runner.run(config, "train", min(deadline.remaining, args.run_timeout_hours * 3600))
        announce(result)
        manifest["attempts"].append(result)
        # A trainer may silently halve batch. Such a trajectory is not a clean
        # measurement of the requested static configuration and is not fit data.
        clean = (result["status"] == "ok" and result.get("epochs_completed") == args.epochs
                 and result["config"]["batch"] == config.batch)
        result["exported_to_fit_csv"] = clean
        if clean:
            row = input_row(result["config"], stats, hw, args.epochs)
            row.update(dataset_id=args.dataset_id,
                       final_map50_95=result.get("final_best_map50_95", result["map50_95_proxy"]),
                       peak_reserved_mb=result["peak_reserved_mb"],
                       full_training_seconds=result["worker_wall_seconds"])
            with csv_path.open("a", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                if not exists:
                    writer.writeheader()
                    exists = True
                writer.writerow({k: row[k] for k in columns})
        manifest["elapsed_seconds"] = deadline.elapsed
        save_json(folder / "collection.json", manifest)
    print(f"Collection log: {folder / 'collection.json'}; fit CSV: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
