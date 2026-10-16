"""Unified root CLI with legacy-compatible compact tasks and detection extension."""

import json
from pathlib import Path
import sys

from yolo_tuner import cli as legacy
from yolo_tuner.adapters import discover
from yolo_tuner.research import Objective, aggregate, score_decisions
from yolo_tuner.pipeline import write_json
from .integrity import verify_raw, verify_live, sha, hardware_changes
from .methods import METHODS
from .report import markdown, sensitivity, SCENARIOS


def parser():
    p = legacy.parser()
    commands = p._subparsers._group_actions[0]
    for name in ("research", "verify-config"):
        child = commands.choices[name]
        if name == "research":
            child.set_defaults(methods=["default", "random", "successive_halving", "adaptive_verified"])
        for action in child._actions:
            if action.dest == "task":
                action.choices = [*action.choices, "detection"]
            elif action.dest == "methods" and name == "research":
                action.choices = list(METHODS)
            elif action.dest == "grid":
                action.help = "JSON configuration overrides; detection also supports model/imgsz/batch/precision/accumulation/optimizer/lr0/weight_decay"
        child.add_argument("--models", nargs="+", help="Detection: installed architecture YAML or local checkpoint; no implicit weight downloads")
        child.add_argument("--prepare-only", action="store_true", help="Detection: freeze protocol/export data without training; start with research-start")
        child.add_argument("--allow-contention", action="store_true", help="Detection smoke only: permit competing ML jobs; never certify measurements/selection")
    start = commands.add_parser("research-start", help="Start a never-executed frozen detection protocol once hardware is idle")
    start.add_argument("--run", required=True)
    start.add_argument("--timeout-hours", type=float, default=3)
    analysis = commands.add_parser("research-analysis", help="Audit study and report weight sensitivity without training or changing primary selection")
    analysis.add_argument("--run", required=True)
    evaluation = commands.add_parser("evaluate-detection", help="Explicit test evaluation after config selection; never used by search")
    evaluation.add_argument("--run", required=True)
    evaluation.add_argument("--seed", type=int)
    evaluation.add_argument("--timeout-minutes", type=float, default=30)
    evaluation.add_argument("--allow-retest", action="store_true")
    return p


def is_detection_run(path):
    file = Path(path) / "protocol.json"
    return file.exists() and json.loads(file.read_text(encoding="utf-8")).get("adapter") == "yolo_research"


def read_run(path):
    folder = Path(path).resolve()
    report = json.loads((folder / "study.json").read_text(encoding="utf-8"))
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    decisions = json.loads((folder / "decisions.locked.json").read_text(encoding="utf-8"))
    if report["protocol"] != protocol or report["decisions"] != decisions:
        raise ValueError("Published study differs from frozen protocol/decisions")
    checked = verify_raw(folder, protocol, decisions, report["reference"])
    return folder, report, checked


def audit(path):
    from .report import summarize
    folder, report, checked = read_run(path)
    results = {row["config_id"]: row["runs"] for row in report["reference"]}
    again = summarize(report["protocol"], report["decisions"], results, report["elapsed_seconds"], report.get("previous_active_seconds"))
    # JSON object keys normalize None to "null". Compare persisted canonical
    # representation, not Python-only dictionary-key types after recomputation.
    again = json.loads(json.dumps(again, allow_nan=False))
    for key in ("reference", "empirical_best", "reference_complete", "verification_status", "pareto", "recommendation", "scores", "summaries", "weight_sensitivity"):
        if again[key] != report[key]:
            raise ValueError(f"Study summary differs from raw measurements: {key}")
    audited = {**report, "audit": {"checked_unique_worker_requests": checked, "new_training_performed": False}}
    write_json(folder / "study.audited.json", audited)
    (folder / "RESEARCH_AUDITED.md").write_text(markdown(audited), encoding="utf-8")
    print(f"Audit passed: {checked} original requests/results | {folder / 'RESEARCH_AUDITED.md'}")
    return 0


def analysis(path):
    if is_detection_run(path):
        folder, report, checked = read_run(path)
        # Recompute aggregates so editing summary fields cannot change sensitivity.
        rows = [aggregate(r["config_id"], r["runs"], report["protocol"]["training_seeds"]) for r in report["reference"]]
        scenarios = report["protocol"]["weight_scenarios"]
    else:
        from yolo_tuner.verification import evidence_row
        evidence_row(path)  # audit original worker requests before reading metrics
        folder = Path(path).resolve()
        report = json.loads((folder / "study.json").read_text(encoding="utf-8"))
        rows = [aggregate(r["config_id"], r["runs"], report["protocol"]["training_seeds"]) for r in report["reference"]]
        scenarios, checked = SCENARIOS, None
    output = {"protocol_sha256": report["protocol"]["protocol_sha256"], "reference_complete": all(r["complete"] for r in rows),
              "weight_sensitivity": sensitivity(rows, Objective(**report["protocol"]["objective"]), scenarios),
              "checked_requests": checked, "primary_selection_changed": False,
              "scenario_registration": "frozen in detection protocol" if checked is not None else "post-hoc exploratory for legacy run, not preregistered",
              "superiority_established": False}
    write_json(folder / "analysis.json", output)
    lines = ["# Phân tích độ nhạy", "", f"Primary selection unchanged. {output['scenario_registration']}.",
             "Không dùng sensitivity để chọn lại weights sau khi xem test/reference.", "",
             "| Scenario | Weights A/T/M | Selected | U |", "|---|---|---|---:|"]
    for row in output["weight_sensitivity"]:
        lines.append(f"| {row['scenario']} | {row['weights']} | {row['selected']} | {row['utility']} |")
    (folder / "ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Analysis: {folder / 'ANALYSIS.md'}")
    return 0


def show(path):
    folder = Path(path).resolve()
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    print(f"Detection | phase={status['phase']} | {len(protocol['configs'])} configs x {len(protocol['training_seeds'])} seeds x {protocol['full_epochs']} epochs")
    print(f"Objective: {protocol['objective']} | environment={protocol['measurement_environment']}")
    if (folder / "study.json").exists():
        report = json.loads((folder / "study.json").read_text(encoding="utf-8"))
        print(f"Verification: {report['verification_status']} | selected={report['empirical_best']} | Pareto={report['pareto']}")
        print(f"Report: {folder / 'RESEARCH.md'}")
    else:
        for process_file in folder.rglob("process.json"):
            process = json.loads(process_file.read_text(encoding="utf-8"))
            if process["status"] == "running":
                import psutil
                try:
                    active = psutil.Process(process["pid"])
                    if abs(active.create_time() - process["create_time"]) < 0.01:
                        result_file = process_file.parent / "result.json"
                        row = json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else {}
                        print(f"Worker PID{process['pid']} confirmed alive: epoch{row.get('epochs_completed',0)} | {process_file.parent / 'worker.log'}")
                except psutil.NoSuchProcess:
                    print("Saved running process no longer alive; phase alone does not prove completion")
    return 0


def transfer(args):
    from yolo_tuner.verification import evidence_row, content_identity, hardware_identity, summarize_evidence
    from yolo_tuner.research import digest
    from yolo_tuner.pipeline import new_session
    rows = []
    for path in args.runs:
        if not is_detection_run(path):
            rows.append(evidence_row(path))
            continue
        folder, report, _ = read_run(path)
        audit(path)
        protocol = report["protocol"]
        manifest = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
        portable = [{**c, "model": protocol["model_files"][c["model"]]} for c in protocol["configs"]]
        rows.append({"run": str(folder), "task": "detection", "dataset_content_id": content_identity(manifest),
                     "hardware": hardware_identity(protocol["hardware"]), "runtime": protocol["runtime_versions"],
                     "space_id": digest(portable), "full_epochs": protocol["full_epochs"], "seeds": protocol["training_seeds"],
                     "objective": protocol["objective"], "quality_definition": protocol["quality_definition"],
                     "protocol_sha256": protocol["protocol_sha256"],
                     "reference_complete": report["reference_complete"] and protocol["measurement_environment"] == "uncontended_preflight",
                     "checked_full_runs": sum(r.get("full_completed", False) for row in report["reference"] for r in row["runs"]),
                     "selected": report["empirical_best"], "mode": report["mode"], "method_summaries": report["summaries"],
                     "generalization_certified": False})
    report = summarize_evidence(rows)
    folder = new_session(args.output)
    write_json(folder / "transfer.json", report)
    lines = ["# Kiểm chứng nhiều task/dataset/backend", "", f"Complete runs: {report['complete_runs']}; dataset splits: {report['distinct_dataset_splits']}; physical CUDA GPUs: {report['distinct_physical_cuda_gpus']}.",
             f"Unseen GPU: {report['unseen_gpu_validation']}. Không gộp mAP/accuracy/BLEU hoặc RSS/VRAM.", "",
             "| Run | Task | Complete clean | Full rows checked | Selected |", "|---|---|---|---:|---|"]
    for row in rows:
        lines.append(f"| {row['run']} | {row['task']} | {row['reference_complete']} | {row['checked_full_runs']} | {row['selected']} |")
    lines += ["", "## Giới hạn", ""] + [f"- {s}" for s in report["limitations"]]
    (folder / "TRANSFER.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Transfer: {folder / 'TRANSFER.md'} | unseen GPU={report['unseen_gpu_validation']}")
    return 0


def train(args, selection):
    from .study import require_idle
    from .runner import DetectionRunner
    from yolo_tuner.pipeline import runtime_info, new_session
    require_idle()
    folder, report, _ = read_run(Path(selection["research_provenance"]).parent)
    if not selection.get("trainable") or report["empirical_best"] is None:
        raise ValueError("No clean verified selection to train")
    expected = report["protocol"]["configs"][int(report["empirical_best"][1:])]
    if selection["config"] != expected or selection["protocol_sha256"] != report["protocol"]["protocol_sha256"]:
        raise ValueError("Selection config differs from verified study")
    if args.resume:
        raise ValueError("Detection fixed-horizon research deployment does not support checkpoint continuation; start a new full run")
    manifest = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    hw = runtime_info(args.device)
    import platform
    hw["host_label"] = platform.node()
    config = dict(selection["config"])
    epochs = args.epochs or selection["full_epochs"]
    if epochs < 1:
        raise ValueError("epochs must be positive")
    changes = []
    if args.data:
        manifest = discover(args.data, report["protocol"]["split_seed"], "detection")
        changes.append("dataset override")
    if epochs != report["protocol"]["full_epochs"]:
        changes.append("horizon")
    if hw["device"] in {"cpu", "mps"} and config["precision"] != "fp32":
        config["precision"] = "fp32"
        changes.append("precision")
    changes.extend(f"hardware/runtime:{key}" for key in hardware_changes(report["protocol"]["hardware"], hw))
    try:
        verify_live(report["protocol"], json.loads((folder / "dataset.json").read_text(encoding="utf-8")), folder=folder)
    except ValueError as exc:
        changes.append(str(exc))
    output = new_session(args.output)
    write_json(output / "dataset.json", manifest)
    from yolo_tuner.adapters import export_yolo
    data = export_yolo(manifest, output / "data")
    runner = DetectionRunner(output, output / "dataset.json", data, hw, epochs, selection["seed"], manifest["counts"], manifest["counts"])
    print(f"Full training: {epochs} epochs | optimality scope changes: {changes}", flush=True)
    result = runner.run(config, epochs, args.timeout_hours * 3600, "train")
    write_json(output / "training_report.json", {"result": result, "selection": selection, "scope_changes": changes,
                                                "status": "modified_protocol_optimality_not_verified" if changes else report["verification_status"]})
    print(f"Training: {result['status']} | {output / 'training_report.json'}", flush=True)
    return 0 if result["status"] == "ok" else 3


def main():
    argv = sys.argv[1:]
    if argv and argv[0] in {"-h", "--help"}:
        parser().parse_args(argv)
    commands = {"research", "verify-config", "research-start", "research-analysis", "research-show", "research-audit", "research-resume", "train", "evaluate-detection", "transfer-report"}
    if not argv or argv[0] not in commands:
        return legacy.main()
    args = parser().parse_args(argv)
    try:
        if args.command in {"research", "verify-config"}:
            manifest = discover(args.data, args.split_seed, args.task)
            if manifest["task"] == "detection":
                from .study import study
                code = study(args, manifest)
            else:
                if args.prepare_only or args.allow_contention or args.models:
                    raise ValueError("Detection-only arguments are not supported by compact task protocol")
                if any(m not in {"default", "random", "random_proxy", "successive_halving", "pareto_racing", "adaptive_verified"} for m in args.methods):
                    raise ValueError("New scheduler ablations currently supported by detection adapter only")
                from yolo_tuner.research import study
                code = study(args)
        elif args.command == "research-analysis":
            code = analysis(args.run)
        elif args.command == "evaluate-detection":
            from .evaluation import evaluate
            code = evaluate(args)
        elif args.command == "transfer-report":
            code = transfer(args)
        elif args.command == "research-start":
            from .study import resume
            code = resume(args, start=True)
        elif args.command in {"research-show", "research-audit", "research-resume"} and is_detection_run(args.run):
            if args.command == "research-show":
                code = show(args.run)
            elif args.command == "research-audit":
                code = audit(args.run)
            else:
                from .study import resume
                code = resume(args)
        elif args.command == "train":
            selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
            if selection.get("backend") != "yolo_research":
                return legacy.main()
            code = train(args, selection)
        else:
            return legacy.main()
    except (ValueError, OSError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        code = 2
    raise SystemExit(code)
