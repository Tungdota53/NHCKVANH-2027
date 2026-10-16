"""Command-line interface: inspect, tune, and explicitly launch full training."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
import math
from pathlib import Path

from .core import Config, Deadline, choose_presets, memory_batch_estimate, rank, successive_halving, training_finished
from .runner import Runner, hardware, resolve_device
from . import cli_ui


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def session(output):
    folder = Path(output).resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    return folder


def announce(result):
    if result.get("reused_baseline"):
        return
    cli_ui.trial(result)


def tune(args):
    from .dataset import prepare

    if Path(args.data).is_dir() or Path(args.data).suffix.lower() in {".json",".csv"}:
        from .pipeline import optimize
        args.task="auto"; args.plan_only=False
        return optimize(args)
    args.device=resolve_device(args.device)

    cli_ui.heading("YOLO BUDGET TUNER", "Đo thực tế • tìm cấu hình trong ngân sách • lưu đầy đủ bằng chứng")
    if any(s <= 0 or s % 32 for s in args.imgsz):
        raise ValueError("Image sizes must be positive multiples of 32")
    if args.full_epochs < max(args.rungs) or any(e < 1 for e in args.rungs) or args.rungs != sorted(set(args.rungs)):
        raise ValueError("Rungs must be strictly increasing positive epochs <= full-epochs")
    if min(args.train_subset, args.val_subset, args.max_batch, args.max_candidates) < 1 or min(args.freeze) < 0:
        raise ValueError("Subset/batch/candidate sizes must be positive; freeze must be >= 0")
    if not 0 < args.reserve_fraction < 1:
        raise ValueError("Reserve fraction must be between 0 and 1")
    if args.min_accuracy is not None and not 0 <= args.min_accuracy <= 1:
        raise ValueError("min-accuracy is a fraction between 0 and 1")
    if args.max_time is not None and args.max_time <= 0:
        raise ValueError("max-time must be positive")
    folder = session(args.output)
    deadline = Deadline(args.budget * 60)
    hw = hardware(args.device)
    cli_ui.hardware_summary(hw)
    stats = prepare(args.data, folder / "subset", args.train_subset, args.val_subset, args.seed)
    cli_ui.dataset_summary(stats)
    precision = args.precision
    if precision == "auto":
        precision = "fp16" if hw["fp16"] else "fp32"
    if args.device == "cpu" and precision != "fp32":
        raise ValueError("CPU search requires fp32")
    if precision == "bf16" and not hw["bf16"]:
        raise ValueError("BF16 unsupported on selected GPU")
    configs = [Config(m, s, 1, precision, f) for m in args.models for s in args.imgsz for f in args.freeze]
    # Round-robin across model families so a cap does not remove the later models.
    configs = sorted(configs, key=lambda c: (c.imgsz, c.freeze, args.models.index(c.model)))
    if len(configs) > args.max_candidates:
        cli_ui.candidate_cap(args.max_candidates, len(configs))
    configs = configs[:args.max_candidates]
    cli_ui.search_plan(args, configs)
    runner = Runner(folder / "trials", stats["subset_yaml"], stats, args.device, args.full_epochs, args.seed)
    profiles, history, verified = [], [], []
    profile_end = deadline.clock() + min(deadline.remaining * 0.35, 300)
    baseline = None

    def evaluate(c, epochs, remaining):
        attempts = []
        for retry in range(3):
            r = runner.run(c, "search", min(deadline.remaining, remaining, args.trial_timeout), epochs)
            attempts.append(r)
            if r["status"] != "oom" or r["config"]["batch"] <= 1 or deadline.remaining < 1:
                break
            c = replace(c, batch=max(1, r["config"]["batch"] // 2))
        result = dict(attempts[-1])
        if len(attempts) > 1:
            result["oom_attempts"] = attempts
            result["worker_wall_seconds"] = sum(r["worker_wall_seconds"] for r in attempts)
        return result

    def profile(c):
        points, successful = [], []
        max_batch = min(args.max_batch, stats["train_subset_images"])
        batches = [1, 2] if max_batch >= 2 else [1]
        for b in batches:
            remaining = min(deadline.remaining, profile_end - deadline.clock())
            if remaining < 1:
                break
            r = runner.run(replace(c, batch=b), "profile", min(remaining, args.trial_timeout))
            profiles.append(r)
            announce(r)
            if r["status"] != "ok":
                break
            successful.append(r)
            points.append((r["config"]["batch"], r["peak_reserved_mb"]))
        if hw.get("vram_free_mb") is not None and len(points) >= 2:
            proposal = memory_batch_estimate(points, hw["vram_free_mb"], max_batch, args.reserve_fraction)
            if proposal > points[-1][0] and profile_end - deadline.clock() > 1:
                r = runner.run(replace(c, batch=proposal), "profile",
                               min(deadline.remaining, profile_end-deadline.clock(), args.trial_timeout))
                profiles.append(r)
                announce(r)
                if r["status"] == "ok":
                    successful.append(r)
        elif successful and hw.get("ram_available_mb"):
            # CPU/MPS: test throughput at a larger batch instead of pretending
            # CUDA memory extrapolation applies to system/shared memory.
            candidate=min(max_batch,successful[-1]["config"]["batch"]*2)
            remaining=min(deadline.remaining,profile_end-deadline.clock())
            if candidate>successful[-1]["config"]["batch"] and remaining>1:
                r=runner.run(replace(c,batch=candidate),"profile",min(remaining,args.trial_timeout))
                profiles.append(r); announce(r)
                if r["status"]=="ok":
                    successful.append(r)
        if hw.get("vram_free_mb") is not None:
            # Reject unsafe measured peaks as well as extrapolations.
            limit = hw["vram_free_mb"] * (1 - args.reserve_fraction)
            successful = [r for r in successful if r["peak_reserved_mb"] <= limit]
        elif hw.get("ram_available_mb"):
            successful=[r for r in successful if r.get("peak_rss_mb",0)<=hw["ram_available_mb"]*(1-args.reserve_fraction)]
        if not successful:
            return None
        fastest = max(r["images_per_second_probe"] for r in successful)
        near_fastest = [r for r in successful if r["images_per_second_probe"] >= 0.95 * fastest]
        return Config(**min(near_fastest, key=lambda r: r["config"]["batch"])["config"])

    for c in configs:
        if deadline.remaining < 1 or deadline.clock() >= profile_end:
            break
        measured = profile(c)
        if measured is not None:
            verified.append(measured)
            if baseline is None:
                baseline = evaluate(measured, args.rungs[0], deadline.remaining)
                history.append(baseline)
                announce(baseline)
                if (baseline["status"] == "ok" and baseline.get("epochs_completed") == args.rungs[0]
                        and baseline["estimated_full_seconds"] < 3600 and not args.force_search):
                    break

    skipped = bool(baseline and baseline["status"] == "ok" and baseline.get("estimated_full_seconds", math.inf) < 3600
                   and not args.force_search)
    if skipped:
        cohort, complete = [baseline], True
    else:
        cached = dict(baseline) if baseline and baseline["status"] == "ok" else None

        def cached_evaluate(c, epochs, remaining):
            nonlocal cached
            if cached and c == Config(**cached["config"]) and epochs == cached["epochs_completed"]:
                result, cached = {**cached, "reused_baseline": True}, None
                return result
            return evaluate(c, epochs, remaining)

        cohort, trials, complete = successive_halving(verified, args.rungs, cached_evaluate, deadline, announce)
        history.extend(r for r in trials if not r.get("reused_baseline"))

    # Optional three-point LR refinement, only if all three can plausibly fit.
    if args.tune_lr and cohort:
        best = rank(cohort)[0]
        if deadline.remaining > 3 * (best["worker_wall_seconds"] + 5):
            refined = []
            for multiplier in (0.5, 1.0, 2.0):
                c = replace(Config(**best["config"]), lr0=best["config"]["lr0"] * multiplier)
                r = evaluate(c, best["epochs_completed"], deadline.remaining)
                history.append(r)
                announce(r)
                if r["status"] != "ok" or r.get("epochs_completed") != best["epochs_completed"]:
                    break
                refined.append(r)
            if len(refined) == 3:
                cohort = cohort + refined

    predictor = None
    accuracy_constraint_status = "not_requested"
    if args.surrogate:
        from .offline import input_row, predict
        predictor = json.loads(Path(args.surrogate).read_text(encoding="utf-8"))
        for row in cohort:
            row["offline_prediction"] = predict(predictor, input_row(row["config"], stats, hw, args.full_epochs))
        if cohort and all(r["offline_prediction"]["status"] == "in_domain" for r in cohort):
            for row in cohort:
                row["accuracy_score"] = row["offline_prediction"]["predictions"]["final_map50_95"]["lower"]
        if args.min_accuracy is not None:
            cohort = [r for r in cohort if "accuracy_score" in r and r["accuracy_score"] >= args.min_accuracy]
            accuracy_constraint_status = "predicted_feasible_not_full_training_verified" if cohort else "no_calibrated_feasible_candidate"
    elif args.min_accuracy is not None:
        # A hard final-accuracy requirement cannot be certified by early mAP50.
        cohort = []
        accuracy_constraint_status = "unverified_without_calibrated_final_accuracy_predictor"
    presets, front = choose_presets(cohort, args.max_time)
    report = {
        "schema_version": 1, "mode": "offline_assisted" if predictor else "measured_cold_start", "hardware": hw, "dataset": stats,
        "search_wall_seconds": deadline.elapsed, "budget_seconds": args.budget * 60,
        "halving_completed": complete, "search_skipped_under_one_hour": skipped,
        "constraints": {"max_full_training_seconds": args.max_time, "min_final_accuracy": args.min_accuracy},
        "accuracy_constraint_status": accuracy_constraint_status,
        "accuracy_metric": "mAP50 on fixed validation subset after reported epochs; proxy only",
        "accuracy_ranking_metric": "offline final mAP50-95 lower bound" if any("accuracy_score" in r for r in cohort) else "early mAP50 proxy",
        "time_constraint_status": "estimated_point_only_not_guaranteed" if args.max_time else "not_requested",
        "offline_artifact": str(Path(args.surrogate).resolve()) if args.surrogate else None,
        "balanced_method": "normalized augmented Tchebycheff, equal weights",
        "pareto_scope": "successful equal-fidelity candidates from last complete rung, or completed first-rung subset",
        "presets": presets, "pareto": front, "profiles": profiles, "history": history,
        "warnings": ["Final accuracy and accuracy constraints require full held-out validation.",
                     "Time extrapolation and two-point VRAM proposals have no calibrated coverage guarantee.",
                     "Train/val paths are disjoint; near-duplicates or source/video groups need separate auditing.",
                     "The scheduler includes worker startup/validation; it may return a partial cohort at timeout."]
    }
    save_json(folder / "report.json", report)
    if presets:
        chosen = presets[args.preset]
        config = Config(**chosen["config"])
        selection = {"config": asdict(config), "data": str(Path(args.data).resolve()), "device": args.device,
                     "full_epochs": args.full_epochs, "seed": args.seed, "provenance": str(folder / "report.json"),
                     "accuracy_final_prediction": chosen.get("offline_prediction", {}).get("predictions", {}).get("final_map50_95"),
                     "final_accuracy_verified": False}
        save_json(folder / "selected.json", selection)
        import yaml
        training = {**asdict(config), "amp": config.precision != "fp32", "data": selection["data"],
                    "device": args.device, "epochs": args.full_epochs, "seed": args.seed,
                    "warmup_epochs": 0.5, "warmup_bias_lr": 0.0, "close_mosaic": 0}
        training.pop("precision")
        if config.precision == "bf16":
            training["amp"] = "bf16"
        (folder / "train.yaml").write_text(yaml.safe_dump(training, sort_keys=False), encoding="utf-8")
        cli_ui.results(report, args.preset)
    else:
        cli_ui.warning("Không có cấu hình đã kiểm chứng thỏa điều kiện. Xem report và log từng trial.")
    cli_ui.warnings(report["warnings"])
    cli_ui.artifact("Báo cáo", folder / "report.json")
    if presets:
        cli_ui.artifact("Cấu hình đã chọn", folder / "selected.json")
    return 0 if presets else 2


def train(args):
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    if getattr(args,"epochs",None) is not None:
        if args.epochs<1:
            raise ValueError("epochs must be positive")
        selection["full_epochs"]=args.epochs
        if "epochs" in selection["config"]:
            selection["config"]["epochs"]=args.epochs
    if selection.get("backend")=="compact_vision":
        from .pipeline import train_selected
        return train_selected(args,selection)
    selection["device"]=resolve_device(args.device)
    if selection["device"] in {"cpu","mps"}:
        selection["config"]["precision"]="fp32"
    if args.data:
        from .adapters import discover,export_yolo
        manifest=discover(args.data,selection["seed"],"detection")
        data_folder=session(Path(args.output)/"adapted_dataset")
        selection["data"]=str(export_yolo(manifest,data_folder))
    config = Config(**selection["config"])
    training_hardware = hardware(selection["device"])
    cli_ui.heading("YOLO FULL TRAINING", f"{config.model} • ảnh {config.imgsz} • batch {config.batch} • {config.precision}")
    cli_ui.hardware_summary(training_hardware)
    # Full training is an explicit command, outside the search's 15-20 min budget.
    folder = session(args.output)
    stats = {"train_images": 1, "val_images": 1, "train_subset_images": 1, "val_subset_images": 1}
    runner = Runner(folder, selection["data"], stats, selection["device"], selection["full_epochs"], selection["seed"])
    deadline = Deadline(args.timeout_hours * 3600)
    checkpoint, attempts = args.resume, []
    for retry in range(args.oom_retries + 1):
        r = runner.run(config, "train", deadline.remaining, resume=checkpoint)
        attempts.append(r)
        save_json(folder / "training_report.json", {"attempts": attempts, "selection": selection,
                                                     "hardware": training_hardware})
        announce(r)
        if r["status"] != "oom" or config.batch <= 1 or retry == args.oom_retries or deadline.remaining < 1:
            break
        checkpoint = r.get("last_checkpoint")
        config = replace(config, batch=max(1, r["config"]["batch"] // 2))
        cli_ui.warning(f"Hết VRAM: thử lại batch={config.batch}, nbs={config.nbs}, checkpoint={checkpoint or 'khởi động lại từ pretrained'}")
    completed = attempts[-1]["status"] == "ok" and attempts[-1].get("epochs_completed") == selection["full_epochs"]
    if completed:
        cli_ui.success(f"Hoàn tất {selection['full_epochs']} epoch.")
    cli_ui.artifact("Báo cáo huấn luyện", folder / "training_report.json")
    return 0 if completed else 2


def export_offline(args):
    import csv
    from .offline import CATEGORICAL, NUMERIC, TARGETS, input_row
    search = json.loads(Path(args.report).read_text(encoding="utf-8"))
    training = json.loads(Path(args.training_report).read_text(encoding="utf-8"))
    selection = training["selection"]
    if Path(selection["provenance"]).resolve() != Path(args.report).resolve():
        raise ValueError("Training selection does not originate from this search report")
    last = training["attempts"][-1]
    if last["status"] != "ok" or last.get("epochs_completed") != selection["full_epochs"]:
        raise ValueError("Offline export requires a completed full training run")
    measured_hardware = {**training.get("hardware", search["hardware"]),
                         "torch_version": last["torch_version"],
                         "ultralytics_version": last["ultralytics_version"]}
    row = input_row(last["config"], search["dataset"], measured_hardware, selection["full_epochs"])
    row.update(dataset_id=args.dataset_id, final_map50_95=last.get("final_best_map50_95", last["map50_95_proxy"]),
               peak_reserved_mb=max(r.get("peak_reserved_mb", 0) for r in training["attempts"]),
               full_training_seconds=sum(r["worker_wall_seconds"] for r in training["attempts"]))
    # Configuration target measurements should be free of restarts/changed batches.
    if len(training["attempts"]) != 1:
        raise ValueError("Recovery runs are logged but excluded from clean offline fitting; rerun the final reduced batch")
    if last["config"]["batch"] != selection["config"]["batch"]:
        raise ValueError("Trainer changed batch internally; rerun that reduced batch for a clean offline measurement")
    columns = ("dataset_id",) + CATEGORICAL + NUMERIC + TARGETS
    path = Path(args.csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(encoding="utf-8-sig", newline="") as f:
            if next(csv.reader(f)) != list(columns):
                raise ValueError("Existing offline CSV header differs from the export schema")
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row[k] for k in columns})
    print(f"Appended measured full run to {path.resolve()}")
    return 0


def show_report(args):
    path = Path(args.report).resolve()
    if path.is_dir():
        path = path / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    selection = json.loads((path.parent / "selected.json").read_text(encoding="utf-8"))
    cli_ui.heading("BÁO CÁO CẤU HÌNH ĐÃ LƯU", "Chỉ xem kết quả • không chạy lại tìm kiếm hoặc training")
    cli_ui.hardware_summary(report.get("hardware", {}))
    cli_ui.dataset_summary(report.get("dataset", {}))
    if selection.get("schema_version") == 2:
        cli_ui.pipeline_results(report, selection, path.parent)
    else:
        cli_ui.results(report)
        cli_ui.warnings(report.get("warnings", []))
        cli_ui.artifact("Báo cáo", path)
        cli_ui.artifact("Cấu hình", path.parent / "selected.json")
    return 0


def show_training(args):
    folder = Path(args.run).resolve()
    requests = sorted(folder.glob("trials/trial_*/request.json"))
    if not requests:
        raise FileNotFoundError(f"No training request under {folder}")
    request_path = requests[-1]
    request = json.loads(request_path.read_text(encoding="utf-8"))
    result_path = request_path.parent / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    cli_ui.heading("TIẾN ĐỘ FULL TRAINING", "Đọc trạng thái đã lưu; không khởi động training mới")
    cli_ui._table("CẤU HÌNH THỰC THI", ["Tham số", "Giá trị"], list(request["config"].items()))
    if request.get("manifest"):
        cli_ui.dataset_summary(json.loads(Path(request["manifest"]).read_text(encoding="utf-8")))
    else:
        cli_ui.dataset_summary({"source":request.get("data", "—")})
    if result.get("epochs_completed"):
        if "map50_proxy" in result:
            cli_ui.trial(result)
        else:
            cli_ui.training_epoch(result, request["full_epochs"])
    else:
        print("  Chưa có epoch hoàn tất; xem worker.log nếu cần kiểm tra lỗi khởi động.")
    completed = False
    training_report = folder / "training_report.json"
    if training_report.exists():
        report = json.loads(training_report.read_text(encoding="utf-8"))
        last = report["attempts"][-1]
        completed = training_finished(last, request["full_epochs"])
        if not completed:
            cli_ui.warning(f"Lần chạy gần nhất: {last.get('status')}; epoch {last.get('epochs_completed', 0)}/{request['full_epochs']}.")
    process_path = request_path.parent / "process.json"
    active = None
    if process_path.exists():
        import psutil
        recorded = json.loads(process_path.read_text(encoding="utf-8"))
        try:
            process = psutil.Process(recorded["pid"])
            active = (recorded.get("status") == "running" and recorded.get("create_time") is not None
                      and abs(process.create_time() - recorded["create_time"]) < 0.001
                      and process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
        except psutil.NoSuchProcess:
            active = False
    print("  Trạng thái: HOÀN TẤT FULL TRAINING" if completed else
          "  Trạng thái: ĐANG TRAIN (tiến trình đã xác nhận)" if active else
          "  Trạng thái: CHƯA HOÀN TẤT, tiến trình đã dừng; kiểm tra log/resume" if active is False else
          "  Trạng thái: CHƯA HOÀN TẤT; run cũ chưa lưu định danh tiến trình.")
    if result.get("early_stopped"):
        print(f"  Early stopping: epoch {result['epochs_completed']} • checkpoint tốt nhất epoch {result.get('best_epoch', '—')}.")
    if result.get("last_checkpoint"):
        cli_ui.artifact("Checkpoint gần nhất", result["last_checkpoint"])
        best = Path(result.get("best_checkpoint", Path(result["last_checkpoint"]).parent / "best.pt"))
        if best.is_file():
            cli_ui.artifact("Checkpoint tốt nhất để sinh caption", best)
    evaluations = sorted(folder.glob("evaluation/*/evaluation.json"))
    if evaluations:
        evaluation = json.loads(evaluations[-1].read_text(encoding="utf-8"))
        metrics = evaluation["metrics"]
        print(f"  Test BLEU-4: {metrics['bleu4']:.4f} • chrF: {metrics['chrf']:.4f}")
        status = evaluation.get('quality_assessment', {}).get('status', 'chưa kết luận')
        label = {"quality_not_established": "CHƯA ĐẠT",
                 "exceeds_text_only_baseline_not_human_verified": "ĐÃ VƯỢT ĐỐI CHỨNG; chưa xác nhận độ đúng bằng con người"}.get(status, status)
        print(f"  Chất lượng đối chứng: {label}")
        cli_ui.artifact("Đánh giá test", evaluations[-1].parent / "EVALUATION.md")
    else:
        cli_ui.warning("Chưa có đánh giá test sau full training; chưa kết luận chất lượng cuối.")
    completion_path = folder / "completion_status.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        print(f"  Job đánh giá nối tiếp: {completion.get('status', '—')}")
        if completion.get("error"):
            cli_ui.warning(completion["error"])
    cli_ui.artifact("Worker log", request_path.parent / "worker.log")
    return 0


def parser():
    p = argparse.ArgumentParser(description="Portable, task-aware image training configuration pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    research = sub.add_parser("research",help="Compare budgeted HPO against a full finite-space reference")
    research.add_argument("--data",required=True)
    research.add_argument("--task",choices=["auto","captioning","classification","reconstruction"],default="auto")
    research.add_argument("--device",default="auto")
    research.add_argument("--seeds",nargs="+",type=int,default=[17,29,43])
    research.add_argument("--methods",nargs="+",choices=["default","random","random_proxy","successive_halving","pareto_racing","adaptive_verified"],
                          default=["default","random","successive_halving","pareto_racing"],help="Include random_proxy for matched subset/fidelity control")
    research.add_argument("--split-seed",type=int,default=17)
    research.add_argument("--full-epochs",type=int,default=8)
    research.add_argument("--rungs",nargs="+",type=int,default=[1,2,4])
    research.add_argument("--search-budget",type=float,default=2,help="Per-method per-repeat minutes; offline reference is separate")
    research.add_argument("--trial-timeout",type=float,default=900)
    research.add_argument("--timeout-hours",type=float,default=3)
    research.add_argument("--train-subset",type=int,default=256)
    research.add_argument("--val-subset",type=int,default=128)
    research.add_argument("--objective",choices=["quality","time","memory","balanced"],default="balanced")
    research.add_argument("--min-quality",type=float,default=0.0)
    research.add_argument("--max-time",type=float)
    research.add_argument("--max-memory",type=float,help="Measured peak allocator GPU / RSS CPU MiB limit")
    research.add_argument("--time-scale",type=float,default=600)
    research.add_argument("--memory-scale",type=float,default=1024)
    research.add_argument("--weights",nargs=3,type=float,default=[0.6,0.25,0.15])
    research.add_argument("--grid",help="JSON list of config overrides; same task/model family and caption encoder protocol")
    research.add_argument("--space",choices=["compact","expanded"],default="compact")
    research.add_argument("--precision",choices=["fp32","fp16","bf16"],help="Explicitly lock precision for paired backend experiments")
    research.add_argument("--caption-encoder",choices=["auto","resnet18","cnn"],default="auto")
    research.add_argument("--offline",action="store_true")
    research.add_argument("--output",default="runs/research")
    import copy
    verified=sub.add_parser("verify-config",parents=[copy.deepcopy(research)],add_help=False,help="Measure configs anew on this hardware/data; full-data confirmation, no offline oracle")
    verified.set_defaults(methods=["adaptive_verified"],space="expanded",search_budget=10,output="runs/verified")
    for action in verified._actions:
        if action.dest=="search_budget":
            action.help="Online minutes per training seed, including profiles and full confirmations; no offline oracle"
        elif action.dest=="methods":
            action.choices=["adaptive_verified"]
            action.help="Production full-data verification policy; independent baseline comparison uses research"
    transfer=sub.add_parser("transfer-report",help="Audit evidence from runs on multiple datasets/backends; never fabricate unseen-GPU coverage")
    transfer.add_argument("--runs",nargs="+",required=True)
    transfer.add_argument("--output",default="runs/transfer")
    research_show=sub.add_parser("research-show",help="Read a research run's live phase/reference/results")
    research_show.add_argument("--run",required=True)
    research_audit=sub.add_parser("research-audit",help="Verify completed protocol/reference artifacts and recompute proxy errors; no training")
    research_audit.add_argument("--run",required=True)
    research_resume=sub.add_parser("research-resume",help="Resume missing full-reference jobs after search decisions are locked; same protocol/runtime")
    research_resume.add_argument("--run",required=True)
    research_resume.add_argument("--timeout-hours",type=float,default=3)
    research_resume.add_argument("--trial-timeout",type=float,default=900)
    inspect = sub.add_parser("inspect", help="Read hardware; no training")
    inspect.add_argument("--device", default="auto")
    show = sub.add_parser("show", help="Display a saved report in the terminal; no training")
    show.add_argument("--report", required=True, help="report.json or its run folder")
    progress = sub.add_parser("show-training", help="Read progress/checkpoints/evaluation from a full-training run")
    progress.add_argument("--run", required=True)
    prediction = sub.add_parser("predict", help="Generate captions from a trained compact checkpoint")
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--image", nargs="+", required=True)
    prediction.add_argument("--device", default="auto")
    evaluation = sub.add_parser("evaluate-caption", help="Score freely generated captions on an independent split")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--manifest", required=True, help="Full dataset.json saved by training/search")
    evaluation.add_argument("--training-run", help="Verify checkpoint/manifest against this pipeline's completed training metadata")
    evaluation.add_argument("--split", choices=["val", "test"], default="test")
    evaluation.add_argument("--device", default="auto")
    evaluation.add_argument("--batch", type=int, default=8)
    evaluation.add_argument("--output", default="runs/evaluation")
    t = sub.add_parser("tune", help="Profile + successive halving on a fixed local dataset subset")
    t.add_argument("--data", required=True, help="Local YOLO detection dataset YAML")
    t.add_argument("--models", nargs="+", default=["yolo11n.pt", "yolo11s.pt"])
    t.add_argument("--imgsz", nargs="+", type=int, default=[480, 640])
    t.add_argument("--freeze", nargs="+", type=int, default=[0, 10])
    t.add_argument("--device", default="auto", help="auto, CUDA index, mps, or cpu")
    t.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    t.add_argument("--budget", type=float, default=20, help="Search wall-clock minutes")
    t.add_argument("--trial-timeout", type=float, default=120, help="Per-worker seconds, including startup")
    t.add_argument("--full-epochs", type=int, default=100)
    t.add_argument("--rungs", nargs="+", type=int, default=[1, 2, 4])
    t.add_argument("--train-subset", type=int, default=400)
    t.add_argument("--val-subset", type=int, default=150)
    t.add_argument("--max-batch", type=int, default=32)
    t.add_argument("--max-candidates", type=int, default=8)
    t.add_argument("--reserve-fraction", type=float, default=0.20)
    t.add_argument("--min-accuracy", type=float, help="Requested final mAP50-95; reported UNVERIFIED in cold-start mode")
    t.add_argument("--surrogate", help="Calibrated offline JSON from fit-offline")
    t.add_argument("--max-time", type=float, help="Estimated full training time limit, seconds")
    t.add_argument("--preset", choices=["Accuracy", "Time", "Memory", "Balanced"], default="Balanced")
    t.add_argument("--force-search", action="store_true", help="Search even when estimated full training <1 hour")
    t.add_argument("--tune-lr", action="store_true", help="Three-point LR check if enough search time remains")
    t.add_argument("--seed", type=int, default=17)
    t.add_argument("--output", default="runs")
    tr = sub.add_parser("train", help="Explicitly launch full training from a selected.json")
    tr.add_argument("--selection", required=True)
    tr.add_argument("--epochs",type=int,help="Override maximum full epochs, including when explicitly extending a resumed run")
    tr.add_argument("--device", default="auto", help="Re-detect hardware on the training machine")
    tr.add_argument("--resume",help="Resume a compact-vision last.pt checkpoint")
    tr.add_argument("--data",help="Optional dataset path override when moving the configuration to another machine")
    tr.add_argument("--timeout-hours", type=float, default=24)
    tr.add_argument("--oom-retries", type=int, choices=range(0, 4), default=2)
    tr.add_argument("--output", default="runs/full")
    tr.add_argument("--evaluate-test", action="store_true", help="After compact caption training, evaluate best.pt on held-out test (requires evaluation extra)")
    tr.add_argument("--offline", action="store_true", help="Do not download encoder weights")
    offline = sub.add_parser("fit-offline", help="Fit ridge + group-conformal intervals from measured full runs CSV")
    offline.add_argument("--csv", required=True)
    offline.add_argument("--output", required=True)
    offline.add_argument("--alpha", type=float, default=0.1)
    offline.add_argument("--seed", type=int, default=17)
    export = sub.add_parser("export-offline", help="Append a completed full training measurement to offline CSV")
    export.add_argument("--report", required=True, help="Original search report.json")
    export.add_argument("--training-report", required=True)
    export.add_argument("--dataset-id", required=True, help="Stable source-dataset ID; same ID across configurations/seeds")
    export.add_argument("--csv", required=True)
    auto=sub.add_parser("auto", aliases=["run"],help="Detect dataset task and hardware; measure and save a training configuration")
    auto.add_argument("--data",required=True,help="Dataset folder, YOLO YAML, COCO JSON or caption CSV")
    auto.add_argument("--task",choices=["auto","detection","captioning","classification","reconstruction"],default="auto")
    auto.add_argument("--device",default="auto")
    auto.add_argument("--budget",type=float,default=20,help="Total wall-clock search minutes")
    auto.add_argument("--trial-timeout",type=float,default=180)
    auto.add_argument("--full-epochs",type=int,default=20)
    auto.add_argument("--rungs",nargs="+",type=int,default=[1,2,4])
    auto.add_argument("--train-subset",type=int,default=256)
    auto.add_argument("--val-subset",type=int,default=96)
    auto.add_argument("--max-candidates",type=int,choices=range(1,7),default=6)
    auto.add_argument("--max-batch",type=int,default=64,help="Maximum measured batch; caption probes also respect the effective-batch target")
    auto.add_argument("--models",nargs="+",help="Optional local/pretrained YOLO models for the detection branch")
    auto.add_argument("--preset",choices=["Accuracy","Time","Memory","Balanced"],default="Balanced")
    auto.add_argument("--plan-only",action="store_true",help="Save a conservative plan without trial training")
    auto.add_argument("--caption-encoder",choices=["auto","resnet18","cnn"],default="auto")
    auto.add_argument("--offline",action="store_true",help="Use cached encoder weights only; auto falls back transparently")
    auto.add_argument("--train",action="store_true",help="After search, run full training and caption test evaluation")
    auto.add_argument("--timeout-hours",type=float,default=24,help="Full-training timeout when --train is used")
    auto.add_argument("--seed",type=int,default=17)
    auto.add_argument("--output",default="runs/auto")
    return p


def main():
    # Windows terminals may default to cp1252/cp1258; preserve Vietnamese output.
    import sys
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,"reconfigure"):
            stream.reconfigure(encoding="utf-8",errors="replace")
    commands={"auto","run","tune","train","inspect","show","show-training","predict","evaluate-caption","research","research-show","research-audit","research-resume","verify-config","transfer-report","fit-offline","export-offline"}
    if len(sys.argv)>1 and sys.argv[1] not in commands:
        if "--data" in sys.argv[1:]:
            sys.argv.insert(1,"auto")
        elif not sys.argv[1].startswith("-"):
            sys.argv[1:1]=["auto","--data"]
    args = parser().parse_args()
    try:
        if args.command in {"research","verify-config"}:
            from .research import study
            code=study(args)
        elif args.command == "transfer-report":
            from .verification import transfer_report
            code=transfer_report(args)
        elif args.command == "research-show":
            from .research import show
            code=show(args)
        elif args.command == "research-audit":
            from .research import audit
            code=audit(args)
        elif args.command == "research-resume":
            from .research import resume
            code=resume(args)
        elif args.command == "inspect":
            info = hardware(args.device, require_cuda=False)
            cli_ui.heading("YOLO BUDGET TUNER", "Kiểm tra môi trường trước khi tìm cấu hình")
            cli_ui.hardware_summary(info)
            code = 0
        elif args.command == "show":
            code = show_report(args)
        elif args.command == "show-training":
            code = show_training(args)
        elif args.command == "predict":
            from .captioning import predict
            code = predict(args)
        elif args.command == "evaluate-caption":
            from .captioning import evaluate
            code = evaluate(args)
        elif args.command in {"auto","run"}:
            from .pipeline import optimize
            code=optimize(args)
        elif args.command == "tune":
            if not math.isfinite(args.trial_timeout) or args.trial_timeout <= 0:
                raise ValueError("trial-timeout must be finite and positive")
            code = tune(args)
        elif args.command == "fit-offline":
            from .offline import fit
            artifact = fit(args.csv, args.output, args.alpha, args.seed)
            print(json.dumps({"file": str(Path(args.output).resolve()), "row_counts": artifact["row_counts"],
                              "diagnostics": {k: {n: v for n, v in p.items() if n != "coefficients"}
                                              for k, p in artifact["predictors"].items()}}, indent=2))
            code = 0
        elif args.command == "export-offline":
            code = export_offline(args)
        else:
            code = train(args)
    except (ValueError, OSError, ImportError) as exc:
        cli_ui.error(exc)
        code = 2
    raise SystemExit(code)
