"""Finish evaluation for a full-training process already started without that flag.

This is a local dependent compute job, not a scheduler or notification service.
New runs should use `train --evaluate-test` instead.
"""

import argparse
import json
from pathlib import Path


def complete(run, pid, timeout_hours=24):
    import psutil
    from .captioning import evaluate
    from .cli import parser
    from .pipeline import write_json
    from .core import training_finished
    folder = Path(run).resolve()
    if not (folder / "dataset.json").is_file():
        raise FileNotFoundError(f"No training dataset manifest under {folder}")
    status_path = folder / "completion_status.json"
    try:
        try:
            process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            process = None
        if process is not None:
            command = process.cmdline()
            if "train" not in command or not any("train_config.py" in value or "yolo_tuner" in value for value in command):
                raise ValueError("PID is not a training command; refusing to follow an unrelated process")
            write_json(status_path, {"status": "waiting_for_full_training", "pid": pid, "training_run": str(folder)})
            print(f"Waiting for full training process {pid}: {folder}", flush=True)
            process.wait(timeout=timeout_hours * 3600)
        report = json.loads((folder / "training_report.json").read_text(encoding="utf-8"))
        last = report["attempts"][-1]
        if not training_finished(last, report["selection"]["full_epochs"]):
            raise ValueError("Full training did not complete; no final test-quality claim is made")
        checkpoint = Path(last["last_checkpoint"]).parent / "best.pt"
        args = parser().parse_args(["evaluate-caption", "--checkpoint", str(checkpoint),
                                   "--manifest", str(folder / "dataset.json"), "--training-run", str(folder),
                                   "--split", "test", "--output", str(folder / "evaluation")])
        write_json(status_path, {"status": "evaluating_test", "training_run": str(folder)})
        code = evaluate(args)
        evaluations = sorted(folder.glob("evaluation/*/evaluation.json"))
        write_json(status_path, {"status": "evaluated", "evaluation_exit_code": code,
                                "quality_gate_passed": code == 0, "human_accuracy_verified": False,
                                "evaluation": str(evaluations[-1]), "training_run": str(folder)})
        return code
    except Exception as exc:
        write_json(status_path, {"status": "failed", "error": str(exc), "training_run": str(folder)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--pid", required=True, type=int)
    args = parser.parse_args()
    raise SystemExit(complete(args.run, args.pid))


if __name__ == "__main__":
    main()
