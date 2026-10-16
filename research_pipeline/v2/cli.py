"""Unified CLI: v2 actions explicit; historical commands retain legacy routing."""

import argparse
import difflib
import json
import math
from pathlib import Path
import sys
import time
import subprocess

from .scheduler import METHODS
from .schema import positive
from . import __version__
from . import terminal as ui
from .selection import RULES

COMMANDS = {"menu", "wizard", "runs", "inspect", "doctor", "plan-search", "optimize", "status", "audit", "resume", "benchmark", "benchmark-report", "algorithm-report", "calibrate", "train-v2", "evaluate-v2", "legacy-help"}
LEGACY_COMMANDS = {"auto", "run", "tune", "train", "show", "show-training", "predict", "evaluate-caption", "research", "research-show", "research-audit", "research-resume", "verify-config", "transfer-report", "fit-offline", "export-offline", "research-start", "research-analysis", "evaluate-detection"}


class CLIParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, f"\nTham số chưa hợp lệ: {message}\nDùng --help để xem ví dụ, hoặc chạy không tham số để mở menu.\n")


def workers_value(value):
    if str(value).lower() == "auto":
        return "auto"
    try:
        result = int(value)
        if not 0 <= result <= 8 or str(result) != str(value):
            raise ValueError
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("workers cần auto hoặc số nguyên 0..8") from None
    return result


def cache_value(value):
    try:
        result = float(value)
        if not math.isfinite(result) or result < 0:
            raise ValueError
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("image-cache-mb cần số hữu hạn >= 0") from None
    return result


def add_common(p, data_required=True):
    basic = p.add_argument_group("Thiết lập chính")
    basic.add_argument("--data", required=data_required, help="Dataset ảnh / YAML / JSON / CSV caption, không phải bảng CSV; đặt nháy nếu có khoảng trắng")
    basic.add_argument("--task", choices=["auto", "detection", "classification", "captioning"], default="auto", help="Tác vụ; mặc định tự nhận dạng")
    basic.add_argument("--device", default="auto", help="auto, cpu, mps hoặc chỉ số CUDA (mặc định auto)")
    basic.add_argument("--output", default="runs/v2", help="Thư mục cha, mỗi lượt tạo thư mục mới (runs/v2)")
    basic.add_argument("--space", choices=["compact", "core", "research"], default="core", help="Core: 12 ứng viên; compact: 3 LR; research: tổ hợp có kiểm soát")
    basic.add_argument("--full-epochs", type=int, default=20, help="Epoch cho mỗi lượt full-data (20)")
    basic.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43], help="Seeds bắt buộc, mặc định 17 29 43")
    basic.add_argument("--budget-seconds", type=float, default=1200, help="Tổng budget search dùng chung, giây (1200); menu nhập phút")
    advanced = p.add_argument_group("Nâng cao — khóa trước search")
    advanced.add_argument("--grid", help="File JSON override thay cho space mặc định")
    advanced.add_argument("--models", nargs="+", help="Detection: YAML hoặc checkpoint local")
    advanced.add_argument("--precision", choices=["fp32", "fp16", "bf16"], help="Mặc định theo khả năng backend; không ép FP16 trên CPU")
    advanced.add_argument("--probe-epochs", type=int, default=1, help="Epoch probe subset (1), không là full confirmation")
    advanced.add_argument("--split-seed", type=int, default=101, help="Seed chia/lấy dữ liệu (101)")
    advanced.add_argument("--search-seed", type=int, default=17, help="Seed thứ tự search (17)")
    advanced.add_argument("--trial-timeout", type=float, default=1800, help="Trần mỗi worker, giây (1800), vẫn bị giới hạn bởi budget chung")
    advanced.add_argument("--probe-fraction", type=float, default=0.1, help="Tỷ phần budget tối đa cho probe (0.1)")
    advanced.add_argument("--cost-margin", type=float, default=1.25, help="Biên heuristic dự báo chi phí (1.25), không là guarantee")
    advanced.add_argument("--cleanup-seconds", type=float, default=5, help="Dự phòng dọn worker, giây (5)")
    advanced.add_argument("--train-subset", type=int, default=64, help="Số mẫu train của probe (64)")
    advanced.add_argument("--val-subset", type=int, default=32, help="Số mẫu val của probe (32)")
    advanced.add_argument("--method", choices=METHODS, default="adaptive_full", help="adaptive_full: chọn thích nghi từ full outcomes; reserve/random/SH và adaptive_no_* là đối chứng/ablation")
    advanced.add_argument("--acquisition-ridge", type=float, default=0.05, help="Kernel ridge đã khóa trước search (0.05)")
    advanced.add_argument("--acquisition-length-scale", type=float, default=1.0, help="Kernel length scale của mixed features (1.0)")
    advanced.add_argument("--acquisition-exploration", type=float, default=0.25, help="Hệ số optimistic uncertainty heuristic (0.25), không phải confidence guarantee")
    advanced.add_argument("--acquisition-novelty", type=float, default=0.05, help="Trọng số diversity (0.05); không phải tham số tối ưu phổ quát")
    advanced.add_argument("--weights", type=float, nargs=3, default=[0.6, 0.25, 0.15], metavar=("A", "T", "M"), help="Trọng số chất lượng / thời gian / bộ nhớ (0.6 0.25 0.15)")
    advanced.add_argument("--time-scale", type=float, default=600, help="Thang chuẩn hóa thời gian, giây (600)")
    advanced.add_argument("--memory-scale", type=float, default=1024, help="Thang chuẩn hóa bộ nhớ, MiB (1024)")
    advanced.add_argument("--max-memory", type=float, help="Trần bộ nhớ MiB; mặc định 80%% bộ nhớ còn trống")
    advanced.add_argument("--max-time", type=float, help="Trần thời gian mỗi full lượt, giây; không phải budget search")
    advanced.add_argument("--min-quality", type=float, default=0, help="Chất lượng validation tối thiểu [0,1] (0)")
    advanced.add_argument("--predictor", help="Card T/M/quality point priors; không thay full confirmation hoặc là safety guarantee")
    advanced.add_argument("--selection-rule", choices=RULES, default="weighted", help="Quy tắc khóa trước search; chỉ chọn từ full confirmations")
    advanced.add_argument("--rungs", nargs="+", type=int, help="Epoch subset cho multi-rung SH; mỗi rung restart cùng horizon scheduler")
    advanced.add_argument("--workers", type=workers_value, default="auto", help="DataLoader workers: auto hoặc số nguyên 0..8; khóa trong protocol")
    advanced.add_argument("--image-cache-mb", type=cache_value, default=64.0, help="Tổng trần cache ảnh giải mã train/val và workers, MiB (64); 0=tắt; detection không cache RAM")
    advanced.add_argument("--pin-memory", choices=["auto", "on", "off"], default="auto", help="Pinned host memory: tự chọn theo backend, bật hoặc tắt")


def parser():
    p = CLIParser(prog="train_config.py", formatter_class=argparse.RawDescriptionHelpFormatter,
                  description=f"TRAIN CONFIG {__version__} — cân bằng chất lượng, bộ nhớ và thời gian bằng training thật.",
                  epilog='''Bắt đầu dễ nhất:
  python train_config.py                   Menu tương tác (terminal)
  python train_config.py wizard            Hướng dẫn tìm cấu hình mới
  python train_config.py inspect --data "D:/datasets/my dataset"
  python train_config.py optimize --data "D:/datasets/my dataset" --budget-seconds 3600
  python train_config.py runs              Tìm các lượt chạy đã lưu
  python train_config.py status --run "runs/v2/<luot>"

Thêm --help sau từng lệnh để xem tham số; inspect/doctor/status/runs có --json.
Lệnh cũ --data, show-training, research... vẫn giữ nguyên; xem legacy-help.
Không hứa tối ưu mọi máy/dataset. Test không được dùng để chọn cấu hình.''')
    p.add_argument("--version", action="version", version=f"train-config {__version__}")
    p.add_argument("--debug", action="store_true", help="Hiện traceback khi chẩn đoán lỗi")
    sub = p.add_subparsers(dest="command", required=True, metavar="LỆNH", title="Chức năng")
    sub.add_parser("menu", help="Menu tương tác: tạo / kiểm tra / quản lý lượt chạy").add_argument("--root", default="runs", help="Thư mục quản lý lượt chạy (runs)")
    sub.add_parser("wizard", help="Hướng dẫn từng bước, xác nhận trước training")
    sub.add_parser("legacy-help", help="Trợ giúp toàn bộ lệnh tương thích cũ")
    child = sub.add_parser("runs", help="Danh sách lượt chạy đã lưu; không training")
    child.add_argument("--root", default="runs")
    child.add_argument("--limit", type=int, default=20)
    child.add_argument("--json", action="store_true")
    child = sub.add_parser("benchmark-report", help="Audit và xuất lại báo cáo benchmark đã hoàn tất; không training/resume")
    child.add_argument("--run", required=True, help="Thư mục study chứa benchmark_protocol.json và decisions.locked.json")
    child.add_argument("--reference", help="Lượt reference đã hoàn tất; mặc định tìm trong <study>/reference")
    child = sub.add_parser("algorithm-report", help="Audit và tổng hợp studies đối chứng budget-limited; không training")
    child.add_argument("--manifest", required=True, help="JSON liệt kê các benchmark study đã hoàn tất")
    child.add_argument("--output", required=True, help="Thư mục mới cho báo cáo thuật toán, không ghi đè")
    helps = {"plan-search": "Lưu protocol và phạm vi; chưa training", "optimize": "Tìm cấu hình bằng full training, budget chung", "benchmark": "Đối chứng nhiều thuật toán và reference độc lập (tốn nhiều lượt train)"}
    for name in ("plan-search", "optimize", "benchmark"):
        child = sub.add_parser(name, help=helps[name], description=helps[name])
        add_common(child, data_required=name != "optimize")
        if name == "optimize":
            child.add_argument("--protocol", help="Chạy protocol đã chuẩn bị và chưa bắt đầu; không override tham số")
        if name == "benchmark":
            child.add_argument("--methods", nargs="+", choices=METHODS, default=["adaptive_full", "random", "reserve", "successive_halving"])
            child.add_argument("--search-seeds", type=int, nargs="+", default=[17, 29, 43])
            child.add_argument("--reference-budget-seconds", type=float, default=3600)
        else:
            child.add_argument("--mode", choices=["online", "offline_reference"], default="online")
            child.add_argument("--reference-boundary")
    child = sub.add_parser("inspect", help="Đọc dataset và phần cứng; không training")
    child.add_argument("--data", required=True)
    child.add_argument("--task", default="auto", choices=["auto", "detection", "classification", "captioning"])
    child.add_argument("--device", default="auto")
    child.add_argument("--json", action="store_true", help="Xuất JSON cho script")
    child = sub.add_parser("doctor", help="Kiểm runtime/backend/dependencies")
    child.add_argument("--device", default="auto")
    child.add_argument("--json", action="store_true", help="Xuất JSON cho script")
    descriptions = {"status": "Xem trạng thái, kết quả và test đã lưu; không training", "audit": "Kiểm tra chứng cứ gốc và tính lại lựa chọn; không training", "resume": "Tiếp tục search với ngân sách bổ sung, giữ protocol"}
    for name in ("status", "audit", "resume"):
        child = sub.add_parser(name, help=descriptions[name])
        child.add_argument("--run", required=True)
        if name == "resume":
            child.add_argument("--additional-budget-seconds", type=float, required=True)
        if name == "status":
            child.add_argument("--json", action="store_true", help="Đọc JSON đã lưu; không xác nhận PID hay audit")
    child = sub.add_parser("calibrate", help="Fit card dự báo T/M từ phân hoạch dữ liệu độc lập")
    child.add_argument("--manifest", required=True, help="Explicit fit/calibration/evaluation run partitions; no domain leakage")
    child.add_argument("--output", required=True)
    child = sub.add_parser("train-v2", help="Huấn luyện thật từ selection; đổi scope không giữ certificate")
    child.add_argument("--selection", required=True)
    child.add_argument("--data")
    child.add_argument("--device", default="auto")
    child.add_argument("--epochs", type=int)
    child.add_argument("--budget-seconds", type=float, default=7200)
    child.add_argument("--output", default="runs/deploy_v2")
    child.add_argument("--preset", choices=["Balanced", "Accuracy", "Time", "Memory"], default="Balanced", help="Chọn preset đã full-confirmed để train riêng; không đổi selection/test của nghiên cứu")
    child = sub.add_parser("evaluate-v2", help="Mở test sau khi đã khóa selection, ghi exposure trước worker")
    child.add_argument("--run", required=True)
    child.add_argument("--timeout-seconds", type=float, default=1800)
    child.add_argument("--allow-retest", action="store_true")
    return p


def status(path):
    return ui.status_view(path)


def legacy_main(argv):
    from research_pipeline.cli import main as legacy
    previous = sys.argv[:]
    try:
        sys.argv = [previous[0], *argv]
        return legacy()
    finally:
        sys.argv = previous


def main(origin=None, argv=None):
    origin = time.monotonic() if origin is None else origin
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    argv = list(sys.argv[1:] if argv is None else argv)
    debug = "--debug" in argv
    if debug:
        argv.remove("--debug")
    if not argv:
        if ui.interactive():
            argv = ["menu"]
        else:
            parser().print_help()
            return 0
    if argv[0] not in COMMANDS | {"-h", "--help", "--version"}:
        if argv[0] in LEGACY_COMMANDS or argv[0].startswith("-") or Path(argv[0]).exists():
            if argv[0] in {"auto", "run", "tune"} or argv[0] == "--data" or Path(argv[0]).exists():
                ui.say("  Nhánh LEGACY: không phải optimize v2 với full-confirmation nhiều seed. Dùng optimize --data ... cho protocol nghiên cứu mới.")
            return legacy_main(argv)
        hint = difflib.get_close_matches(argv[0], sorted(COMMANDS | LEGACY_COMMANDS), n=1)
        parser().error(f"Lệnh không tồn tại: {argv[0]}" + (f". Có phải bạn muốn '{hint[0]}'?" if hint else ""))
    args = parser().parse_args(argv)
    try:
        if args.command in {"menu", "wizard"}:
            if not ui.interactive():
                raise ValueError("Menu cần terminal tương tác. Dùng optimize --data ... hoặc --help khi chạy script/redirect.")
            from .wizard import menu
            def dispatch(chosen):
                # User think/input time is not research preparation cost.
                try:
                    return main(origin=time.monotonic(), argv=chosen + (["--debug"] if debug else []))
                except SystemExit as exc:
                    if exc.code == 130:
                        raise KeyboardInterrupt from exc
                    return exc.code
            return menu(dispatch, getattr(args, "root", "runs"), start=args.command == "wizard")
        elif args.command == "legacy-help":
            return legacy_main(["--help"])
        elif args.command == "runs":
            positive(args.limit, "limit")
            entries = ui.inventory(args.root, args.limit)
            if args.json:
                print(json.dumps(entries, ensure_ascii=False, indent=2))
            else:
                ui.runs_view(entries)
                ui.say("\n  Dùng menu > 4 để chọn lượt, hoặc status --run <đường dẫn>.")
        elif args.command == "benchmark-report":
            from .benchmark import finalize_benchmark
            finalize_benchmark(args.run, args.reference)
        elif args.command == "algorithm-report":
            from .algorithm_study import analyze_studies
            analyze_studies(args.manifest, args.output)
        elif args.command in {"plan-search", "optimize", "benchmark"}:
            from .scheduler import ADAPTIVE_METHODS
            chosen_methods = args.methods if args.command == "benchmark" else [args.method]
            if any(method in ADAPTIVE_METHODS for method in chosen_methods) and args.selection_rule != "weighted":
                raise ValueError("adaptive_full hiện dùng weighted objective; chọn reserve/random/SH nếu dùng TOPSIS/knee/Tchebycheff")
            for name in ("budget_seconds", "trial_timeout", "full_epochs", "train_subset", "val_subset"):
                positive(getattr(args, name), name)
            if args.command == "benchmark":
                if len(set(args.methods)) != len(args.methods) or len(set(args.search_seeds)) != len(args.search_seeds):
                    raise ValueError("Duplicate method/repeat seed")
                positive(args.reference_budget_seconds, "reference_budget_seconds")
                from .benchmark import benchmark
                benchmark(args)
            else:
                from .workflow import prepare, execute
                if args.command == "optimize" and args.protocol:
                    if args.data or {v.split("=")[0] for v in argv[1:] if v.startswith("--")} - {"--protocol"}:
                        raise ValueError("--protocol forbids config/device/budget overrides; create a new plan for changes")
                    folder = Path(args.protocol).resolve().parent
                    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
                    if json.loads((folder / "status.json").read_text(encoding="utf-8"))["phase"] != "prepared":
                        raise ValueError("Prepared protocol already started; use resume with an explicit additional budget")
                    from .ledger import Ledger
                    execute(folder, protocol, Ledger(folder / "events.jsonl", origin=origin), charge_preparation=False)
                else:
                    if not args.data:
                        raise ValueError("Specify --data or --protocol")
                    folder, protocol, ledger = prepare(args, origin)
                    if args.command == "optimize":
                        execute(folder, protocol, ledger)
                    else:
                        ui.say("\n  Đã lưu protocol. Chưa training, chưa có chứng nhận tối ưu.")
                        ui.say("  Khi muốn bắt đầu: " + ui.command(["optimize", "--protocol", str(folder / "protocol.json")]))
        elif args.command in {"inspect", "doctor"}:
            if not args.json:
                ui.say("  Đang kiểm tra môi trường" + (" và dataset" if args.command == "inspect" else "") + "... (không training)")
            from yolo_tuner.pipeline import runtime_info
            info = {}
            if args.command == "inspect":
                from .data import discover
                data = discover(args.data, 101, args.task)
                info["dataset"] = {k: data.get(k) for k in ("task", "source", "counts", "captions", "validation_ready", "warnings")}
            info["hardware"] = runtime_info(args.device)
            if args.json:
                print(json.dumps(info, ensure_ascii=False, indent=2))
            else:
                ui.inspection_view(info)
            if args.command == "doctor" and not info["hardware"].get("ml_runtime_ready"):
                return 2
        elif args.command == "status":
            if args.json:
                folder = Path(args.run).resolve()
                print(json.dumps({"run": str(folder), "saved_status": ui.read_json(folder / "status.json"),
                                  "saved_report": ui.read_json(folder / "report.json", optional=True),
                                  "live_process_checked": False, "evidence_audited": False}, ensure_ascii=False, indent=2))
            else:
                status(args.run)
        elif args.command == "audit":
            from .workflow import audit_run
            folder, protocol, rows, count = audit_run(args.run)
            ui.title("AUDIT THÀNH CÔNG", "Đã kiểm tra chứng cứ gốc; không chạy training mới.")
            ui.say(f"  {count} lượt: request/result gốc, source snapshots, checkpoint, event ledger và lựa chọn tính lại.")
            ui.say(f"  Thư mục: {folder}")
        elif args.command == "resume":
            positive(args.additional_budget_seconds, "additional budget")
            from .workflow import resume_run
            resume_run(args.run, args.additional_budget_seconds, origin)
        elif args.command == "calibrate":
            from .predictors import fit_card
            card = fit_card(args.manifest, args.output)
            print(f"Predictor: {args.output} | {card['calibration_status']} | unseen GPU validated=False")
        elif args.command in {"train-v2", "evaluate-v2"}:
            from .deployment import train, evaluate
            report = (train if args.command == "train-v2" else evaluate)(args)
            if args.command == "train-v2" and not report.get("execution_verified"):
                return 3
    except KeyboardInterrupt:
        ui.say("\n  Đã dừng an toàn. Checkpoint/log được giữ lại; dùng status --run để kiểm tra.")
        raise SystemExit(130)
    except (ValueError, OSError, ImportError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        if debug:
            raise
        ui.error_view(exc)
        raise SystemExit(2)
    return 0
