"""Recover a locked pilot after controller-only metadata-format bug fixes.

Search decisions are NOT replayed or edited. Source differences are disclosed.
Any worker/runner/model/metric source difference refuses the comparison.
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolo_tuner.pipeline import write_json
from research_pipeline.v2.cli import parser
from research_pipeline.v2.workflow import prepare, execute
from research_pipeline.v2.benchmark import compare


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--benchmark", required=True)
    argparser.add_argument("--reference-run", help="Analyze an already fully measured reference without starting training")
    args = argparser.parse_args()
    root = Path(args.benchmark).resolve()
    manifest = json.loads((root / "benchmark_protocol.json").read_text(encoding="utf-8"))
    locked = json.loads((root / "decisions.locked.json").read_text(encoding="utf-8"))
    original = manifest["arguments"]
    argv = ["optimize", "--data", original["data"]]
    for name in ("task", "device", "space", "full_epochs", "probe_epochs", "train_subset", "val_subset", "split_seed", "trial_timeout",
                 "probe_fraction", "cost_margin", "cleanup_seconds", "time_scale", "memory_scale", "min_quality", "max_memory", "max_time", "grid", "precision", "predictor"):
        if original.get(name) is not None:
            argv += ["--" + name.replace("_", "-"), str(original[name])]
    for name in ("seeds", "weights", "models"):
        if original.get(name):
            argv += ["--" + name.replace("_", "-"), *map(str, original[name])]
    argv += ["--method", "random", "--search-seed", "123456", "--mode", "offline_reference",
             "--reference-boundary", str(root / "decisions.locked.json"),
             "--budget-seconds", str(original["reference_budget_seconds"]), "--output", str(root / "reference_after_controller_fix")]
    if args.reference_run:
        folder = Path(args.reference_run).resolve()
    else:
        origin = time.monotonic()
        ref_args = parser().parse_args(argv)
        folder, protocol, ledger = prepare(ref_args, origin)
        execute(folder, protocol, ledger)
    report = compare(locked["runs"], str(folder), allow_controller_patch=True)
    report["reference_recovery_reason"] = "Tuple/list JSON objective-format comparison fix after search lock; no search replay, no training-source change"
    report["reference_parent_status"] = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    if report["reference_parent_status"]["phase"] != "complete":
        report["reference_publication_note"] = "All required full outcomes audited; standalone helper console encoding failed after locked selection/report files were committed. Original error preserved."
    report["external_shared_preparation_seconds"] = manifest["external_shared_preparation_seconds"]
    write_json(root / "BENCHMARK.json", report)
    lines = ["# Benchmark v2 — reference khôi phục minh bạch", "", report["scope"], "",
             report["source_comparison"], report["reference_recovery_reason"], "",
             "| Method | Feasible | Conditional regret mean | Actual online seconds mean |", "|---|---:|---:|---:|"]
    for r in report["summaries"]:
        lines.append(f"| {r['method']} | {r['feasible_count']}/{r['n']} | {r['conditional_regret_mean']} (available {r['regret_available_count']}) | {r['online_seconds_mean']:.2f} |")
    lines += ["", f"Reference complete: {report['reference_complete']}; cost {report['reference_cost_seconds']}s.",
              "Failed selection regret is null, not zero. This one-domain pilot does not establish statistical or unseen-GPU superiority."]
    (root / "BENCHMARK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(root / "BENCHMARK.md"))


if __name__ == "__main__":
    main()
