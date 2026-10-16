"""Explicit, cancellable interactive workflows over the existing v2 commands."""

import math
import os
from pathlib import Path

from . import __version__
from .terminal import (command, duration, inspection_view, inventory, pairs,
                       read_json, runs_view, say, title)


class Back(Exception):
    """Return to the menu without starting a job."""


def ask(label, default=None):
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"  {label}{suffix}: ").strip()
    if value.lower() in {"b", "back", "quay lại"}:
        raise Back
    return value if value else str(default) if default is not None else ""


def choice(label, options, default=None):
    while True:
        value = ask(label, default)
        if value in options:
            return value
        if "Activate.ps1" in value or "Set-ExecutionPolicy" in value:
            say("  Đây là lệnh PowerShell kích hoạt môi trường, không phải lựa chọn menu.")
            say("  Không thực thi lệnh này trong CLI. Kích hoạt môi trường trước khi mở menu; hoặc gọi trực tiếp .venv\\Scripts\\python.exe train_config.py (không cần activate).")
        say("  Hãy nhập một trong: " + ", ".join(options) + "; b để quay lại.")


def numeric(label, default, *, integer=False, minimum=0, maximum=None, optional=False):
    while True:
        value = ask(label, default)
        if optional and not value:
            return None
        try:
            result = int(value) if integer else float(value.replace(",", "."))
            if not math.isfinite(result) or result <= minimum or maximum is not None and result > maximum:
                raise ValueError
            return result
        except ValueError:
            say(f"  Cần {'số nguyên' if integer else 'số'} hữu hạn > {minimum}" + (f" và <= {maximum}." if maximum is not None else "."))


def path_input(label, default=None, *, directory=False):
    while True:
        value = ask(label, default).strip().strip('"').strip("'")
        path = Path(os.path.expandvars(os.path.expanduser(value))).resolve() if value else None
        if path and (path.is_dir() if directory else path.exists()):
            return str(path)
        say("  Đường dẫn chưa tồn tại" + (" hoặc không phải thư mục." if directory else ".") + " Dán đường dẫn đầy đủ; không cần dấu nháy.")


def confirm(argv, *, note):
    say("\n  " + note)
    say("  Lệnh tương đương (có thể copy chạy lại):")
    say("  " + command(argv))
    return choice("Thực hiện? y = có, n = không", {"y", "n"}, "n") == "y"


def inspect_dataset():
    from .data import discover
    from .input_diagnostics import DatasetInputError
    from yolo_tuner.pipeline import runtime_info
    title("BƯỚC 1/3 — CHỌN VÀ KIỂM TRA DATASET", "Nhập b ở bất kỳ câu hỏi nào để quay lại. Không tự tạo nhãn.")
    say("  Hiện hỗ trợ dữ liệu ẢNH: detection / classification ảnh / captioning.")
    say("  CSV phải tham chiếu ảnh/caption; chưa hỗ trợ bảng đặc trưng CSV/Excel.")
    device = "auto"
    while True:
        source = path_input("Dataset ảnh (thư mục / YAML / JSON / CSV caption)")
        device = ask("Thiết bị: auto / cpu / chỉ số GPU", device)
        say("\n  Đang đọc nhãn và kiểm tra split; chưa training...")
        try:
            data = discover(source, 101, "auto")
            break
        except DatasetInputError as exc:
            from .terminal import error_view
            error_view(exc)
            if choice("1 = nhập dataset khác, 0 = quay lại menu", {"0", "1"}, "0") == "0":
                raise Back from exc
    say("  Dataset hợp lệ; đang kiểm tra phần cứng...")
    hw = runtime_info(device)
    inspection_view({"hardware": hw, "dataset": data})
    return source, device, data, hw


def optimize_wizard(dispatch):
    source, device, data, hw = inspect_dataset()
    if data["task"] not in {"detection", "classification", "captioning"} or not data.get("validation_ready"):
        say("\n  Chưa thể tìm cấu hình v2: cần task được hỗ trợ, nhãn thật và validation độc lập.")
        return
    if not hw.get("ml_runtime_ready"):
        say("\n  Runtime chưa sẵn sàng. Chọn 'Kiểm tra môi trường' trong menu để chẩn đoán.")
        return
    title("BƯỚC 2/3 — ĐẶT PHẠM VI VÀ NGÂN SÁCH")
    say("  Budget là tổng thời gian search, dùng chung giữa cấu hình và seeds.")
    say("  Hết budget có thể chỉ trả 'tốt nhất đã đo', hoặc chưa có cấu hình đủ điều kiện.")
    minutes = numeric("Budget (phút)", 20)
    epochs = numeric("Số epoch cho MỖI lượt full training", 20, integer=True)
    say("\n  [1] Core: 12 cấu hình, thay đổi có kiểm soát (khuyên dùng cho nghiên cứu)")
    say("  [2] Compact: 3 cấu hình LR, phạm vi hẹp hơn, ít lượt train hơn")
    say("  [3] Grid JSON của bạn: khóa tập ứng viên trước khi search")
    say("  [4] Research: tổ hợp model/image/batch/LR; Freeze khi dùng checkpoint local")
    scope = choice("Chọn tập cấu hình", {"1", "2", "3", "4"}, "1")
    grid = path_input("File grid JSON") if scope == "3" else None
    say("\n  [1] 3 seeds: 17, 29, 43 (khuyên dùng để đo biến thiên)")
    say("  [2] 1 seed: 17 (ít chi phí; không ước lượng biến thiên giữa seeds)")
    seeds = [17, 29, 43] if choice("Số seeds", {"1", "2"}, "1") == "1" else [17]
    argv = ["optimize", "--data", source, "--device", device, "--task", data["task"],
            "--budget-seconds", str(minutes * 60), "--full-epochs", str(epochs),
            "--space", "compact" if scope == "2" else "research" if scope == "4" else "core", "--seeds", *map(str, seeds)]
    if scope == "4" and data["task"] == "detection":
        say("  Không tự tải weights. Enter = YOLO11n/s YAML, không freeze backbone chưa học.")
        value = ask("Checkpoint local .pt, nhiều file cách nhau bằng dấu ; (Enter = mặc định)")
        if value:
            models = [v.strip().strip('"').strip("'") for v in value.split(";")]
            if any(not Path(v).is_file() or Path(v).suffix.lower() != ".pt" for v in models):
                raise ValueError("Research pretrained models must be existing local .pt files")
            argv += ["--models", *models]
    if grid:
        argv += ["--grid", grid]
    weights = [0.6, 0.25, 0.15]
    rule, method = "weighted", "adaptive_full"
    advanced = choice("Đặt mục tiêu/giới hạn nâng cao? y/n", {"y", "n"}, "n")
    if advanced == "y":
        from .selection import RULES
        rule = choice("Quy tắc: weighted / tchebycheff / topsis / knee", set(RULES), "weighted")
        argv += ["--selection-rule", rule]
        methods = {"reserve", "successive_halving", "prior_full", "random"}
        if rule == "weighted":
            methods.add("adaptive_full")
        method = choice("Thuật toán: " + " / ".join(sorted(methods)), methods, "adaptive_full" if rule == "weighted" else "reserve")
        argv += ["--method", method]
        say("  Các trọng số phải >= 0 và cộng lại > 0; được khóa trước search.")
        while True:
            value = ask("Trọng số accuracy time memory (cách nhau bằng khoảng trắng)", "0.6 0.25 0.15")
            try:
                weights = [float(v) for v in value.split()]
                if len(weights) != 3 or any(not math.isfinite(v) or v < 0 for v in weights) or sum(weights) <= 0:
                    raise ValueError
                break
            except ValueError:
                say("  Ví dụ hợp lệ: 0.6 0.25 0.15")
        argv += ["--weights", *map(str, weights)]
        memory = numeric("Trần bộ nhớ MiB (Enter = tự đặt 80% còn trống)", None, optional=True)
        train_time = numeric("Trần thời gian mỗi full lượt, phút (Enter = không đặt)", None, optional=True)
        time_scale = numeric("Thang chuẩn hóa thời gian, giây", 600)
        memory_scale = numeric("Thang chuẩn hóa bộ nhớ, MiB", 1024)
        if memory is not None:
            argv += ["--max-memory", str(memory)]
        if train_time is not None:
            argv += ["--max-time", str(train_time * 60)]
        argv += ["--time-scale", str(time_scale), "--memory-scale", str(memory_scale)]
        output = ask("Thư mục lưu lượt mới", "runs/v2").strip('"').strip("'")
        argv += ["--output", output]
    title("BƯỚC 3/3 — XEM LẠI TRƯỚC KHI CHẠY")
    pairs([("Dataset / task", f"{source} / {data['task']}"), ("Thiết bị", device),
           ("Ngân sách chung", duration(minutes * 60)), ("Epoch / full lượt", epochs),
           ("Seeds", seeds), ("Tập cấu hình", grid or ("compact (3)" if scope == "2" else "research (tổ hợp; số chính xác sau validate)" if scope == "4" else "core (12)")),
           ("Weights A / T / M", weights), ("Quy tắc lựa chọn", rule), ("Thuật toán", method + " + full confirmation; không dùng test")])
    if data["task"] == "detection" and not grid:
        say("  Detection mặc định dùng kiến trúc YOLO11 YAML, không pretrained. Dùng grid/model local để đổi.")
    say("  Các probe dùng subset; cấu hình được chọn phải full-data/full-epoch/đủ seeds.")
    say("  Budget này chưa được xác minh đủ cho dataset/máy của bạn; không cam kết đo hết grid.")
    action = choice("1 = chạy thật, 2 = chỉ lưu protocol, 0 = hủy", {"1", "2", "0"}, "0")
    if action == "0":
        say("  Đã hủy; chưa tạo lượt chạy hoặc khởi động training.")
        return
    if action == "2":
        argv[0] = "plan-search"
    if confirm(argv, note="Chạy training thật, ghi chứng cứ và dùng tài nguyên máy." if action == "1" else "Chỉ tạo thư mục/protocol mới; chưa training. Model local/YAML sẽ được chuẩn bị."):
        dispatch(argv)
    else:
        say("  Đã hủy; chưa tạo lượt chạy.")


def manage_run(dispatch, root):
    entries = inventory(root)
    runs_view(entries)
    say("\n  Nhập số để chọn; p = nhập đường dẫn lượt khác; 0 = quay lại.")
    selected = choice("Chọn lượt", {"0", "p", *map(str, range(1, len(entries) + 1))}, "0")
    if selected == "0":
        return
    if selected == "p":
        folder = Path(path_input("Thư mục lượt chạy", directory=True))
        protocol = read_json(folder / "protocol.json", optional=True) or {}
        kind = "v2 training" if protocol.get("adapter") == "research_v2" and protocol.get("mode") == "deployment" else "v2" if protocol.get("adapter") == "research_v2" else "research" if protocol.get("adapter") == "yolo_research" else "training cũ"
    else:
        entry = entries[int(selected) - 1]
        folder, kind = Path(entry["path"]), entry["kind"]
    if kind != "v2":
        dispatch(["status" if kind == "v2 training" else "research-show" if kind == "research" else "show-training", "--run", str(folder)])
        return
    while True:
        title("QUẢN LÝ LƯỢT CHẠY")
        say("  " + str(folder))
        saved = read_json(folder / "status.json", optional=True) or {}
        selection = read_json(folder / "selected.json", optional=True)
        say("\n  [1] Xem trạng thái và kết quả (chỉ đọc)")
        say("  [2] Audit chứng cứ gốc (không training)")
        say("  [3] " + ("Bắt đầu protocol chưa chạy" if saved.get("phase") == "prepared" else "Tiếp tục search với budget bổ sung"))
        if selection:
            say("  [4] Train lại cấu hình được chọn (lượt riêng)")
            say("  [5] Đánh giá test sau lựa chọn (mở test có ghi lịch sử)")
        say("  [0] Quay lại menu")
        action = choice("Chọn thao tác", {"0", "1", "2", "3", *( ["4", "5"] if selection else [])}, "1")
        if action == "0":
            return
        if action in {"1", "2"}:
            dispatch(["status" if action == "1" else "audit", "--run", str(folder)])
            continue
        if action == "3":
            if saved.get("phase") == "prepared":
                argv = ["optimize", "--protocol", str(folder / "protocol.json")]
                note = "Thực thi nguyên protocol đã khóa; không thay tham số."
            else:
                minutes = numeric("Budget BỔ SUNG (phút)", 60)
                argv = ["resume", "--run", str(folder), "--additional-budget-seconds", str(minutes * 60)]
                note = "Audit và kiểm tra tương thích trước resume; không bỏ qua hash/source/runtime."
        elif action == "4":
            preset = choice("Preset để train: Balanced / Accuracy / Time / Memory", {"Balanced", "Accuracy", "Time", "Memory"}, "Balanced")
            epochs = numeric("Số epoch", selection["full_epochs"], integer=True)
            minutes = numeric("Trần thời gian training (phút)", 120)
            device = ask("Thiết bị", "auto")
            argv = ["train-v2", "--selection", str(folder / "selected.json"), "--epochs", str(epochs),
                    "--budget-seconds", str(minutes * 60), "--device", device, "--preset", preset]
            note = "Training thật trong lượt riêng. Đổi GPU/epoch không kế thừa chứng nhận tối ưu."
        else:
            argv = ["evaluate-v2", "--run", str(folder)]
            history = read_json(folder / "test_history.json", optional=True) or []
            if history:
                say("  Test đã được mở; nên xem kết quả cũ bằng thao tác 1.")
                if choice("Chủ động cho phép đánh giá test LẠI? y/n", {"y", "n"}, "n") != "y":
                    continue
                argv.append("--allow-retest")
            note = "Mở test sau lựa chọn đã khóa; không dùng kết quả test để chọn lại cấu hình."
        if confirm(argv, note=note):
            dispatch(argv)
        else:
            say("  Đã hủy thao tác.")


def menu(dispatch, root="runs", *, start=False):
    try:
        if start:
            optimize_wizard(dispatch)
            return 0
        while True:
            title(f"TRAIN CONFIG {__version__} — TÌM CẤU HÌNH HUẤN LUYỆN",
                  "Cân bằng chất lượng, bộ nhớ và thời gian bằng số đo thực nghiệm.")
            say("\n  [1] Tìm cấu hình mới — hướng dẫn từng bước")
            say("  [2] Kiểm tra dataset và phần cứng — không training")
            say("  [3] Kiểm tra môi trường (doctor)")
            say("  [4] Xem / quản lý các lượt chạy đã lưu")
            say("  [5] Trợ giúp lệnh nâng cao")
            say("  [0] Thoát")
            say("\n  Enter chọn mục mặc định; b quay lại; Ctrl+C hủy an toàn.")
            try:
                selected = choice("Chọn chức năng", {"0", "1", "2", "3", "4", "5"}, "1")
                if selected == "0":
                    say("  Đã thoát. Không khởi động training mới.")
                    return 0
                if selected == "1":
                    optimize_wizard(dispatch)
                elif selected == "2":
                    inspect_dataset()
                elif selected == "3":
                    dispatch(["doctor"])
                elif selected == "4":
                    manage_run(dispatch, root)
                else:
                    dispatch(["--help"])
            except Back:
                say("  Đã quay lại menu; không khởi động thao tác đang nhập.")
            except (ValueError, OSError, ImportError, KeyError, TypeError) as exc:
                from .terminal import error_view
                error_view(exc)
    except EOFError:
        say("\n  Đã đóng đầu vào; không khởi động thao tác đang nhập.")
        return 0
    except Back:
        say("  Đã hủy hướng dẫn; không khởi động training.")
        return 0
    except KeyboardInterrupt:
        say("\n  Đã hủy/dừng an toàn. Giữ nguyên thư mục lượt chạy và log để xem trạng thái.")
        return 130
