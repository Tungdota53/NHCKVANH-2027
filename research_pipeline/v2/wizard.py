"""Explicit, cancellable interactive workflows over the existing v2 commands."""

import math
import os
from pathlib import Path

from . import __version__
from .terminal import (command, duration, inspection_view, inventory, pairs,
                       read_json, runs_view, say, style_text, _say_styled, title)

_WEIGHT_INFO_CACHE = {}


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


def path_input(label, default=None, *, directory=False, shortcuts=None):
    while True:
        value = ask(label, default).strip().strip('"').strip("'")
        if shortcuts and value in shortcuts:
            value = str(shortcuts[value])
        path = Path(os.path.expandvars(os.path.expanduser(value))).resolve() if value else None
        if path and (path.is_dir() if directory else path.exists()):
            return str(path)
        say("  Đường dẫn chưa tồn tại" + (" hoặc không phải thư mục." if directory else ".") + " Dán đường dẫn đầy đủ; không cần dấu nháy.")


def scan_datasets(base_dir="runs/datasets"):
    root = Path(base_dir)
    if not root.is_dir():
        return []
    found = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        try:
            if entry.is_dir():
                if (entry / "data.yaml").is_file():
                    found.append({"name": entry.name, "path": str((entry / "data.yaml").resolve()), "kind": "detection/yaml"})
                elif (entry / "dataset.json").is_file():
                    found.append({"name": entry.name, "path": str(entry.resolve()), "kind": "manifest"})
                elif (entry / "train").is_dir() and (entry / "val").is_dir():
                    found.append({"name": entry.name, "path": str(entry.resolve()), "kind": "folder"})
            elif entry.is_file() and entry.suffix.lower() in {".yaml", ".yml", ".json", ".csv"}:
                found.append({"name": entry.name, "path": str(entry.resolve()), "kind": entry.suffix.lower().lstrip(".")})
        except OSError:
            continue
    return found


def scan_weights(base_dir="weights"):
    root = Path(base_dir)
    if not root.is_dir():
        return []
    items = []
    for pt in sorted(root.glob("*.pt"), key=lambda p: p.name.lower()):
        if not pt.is_file():
            continue
        try:
            st = pt.stat()
            resolved = str(pt.resolve())
            cache_key = (resolved, st.st_mtime_ns, st.st_size)
            cached = _WEIGHT_INFO_CACHE.get(cache_key)
            if cached is None:
                params = None
                gflops_32 = None
                try:
                    from .model_features import describe_models
                    desc = describe_models("detection", [{"model": resolved}], 1)
                    first = next(iter(desc.values()), {})
                    params = first.get("parameters")
                    gflops_32 = first.get("gflops_reference_32")
                except Exception:
                    pass
                gflops_640 = (float(gflops_32) * (640.0 / 32.0) ** 2) if isinstance(gflops_32, (int, float)) and math.isfinite(gflops_32) else None
                cached = {
                    "path": str(pt),
                    "resolved": resolved,
                    "name": pt.name,
                    "size_mb": round(st.st_size / (1024.0 * 1024.0), 2),
                    "parameters": params,
                    "gflops_reference_32": gflops_32,
                    "gflops_640": gflops_640,
                }
                _WEIGHT_INFO_CACHE[cache_key] = cached
            items.append(dict(cached))
        except OSError:
            continue
    return items


def suggest_scales(hw, data=None, epochs=20):
    hw = hw or {}
    device = str(hw.get("device") or "cpu").lower()
    is_gpu = device not in {"cpu", "mps"}
    train_n = max(1, int(((data or {}).get("counts") or {}).get("train") or 64))
    epochs_val = max(1, int(epochs or 20))
    raw_t = (0.04 if is_gpu else 0.18) * train_n * epochs_val + (25.0 if is_gpu else 10.0)
    time_scale = int(max(60, min(3600, round(raw_t / 10.0) * 10)))
    base_mem = hw.get("vram_total_mb") or hw.get("vram_free_mb") or hw.get("ram_available_mb") or 4096.0
    raw_m = float(base_mem) * (0.35 if is_gpu else 0.25)
    memory_scale = int(max(256, min(8192, round(raw_m / 64.0) * 64)))
    return time_scale, memory_scale


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
    discovered = scan_datasets()
    shortcuts = {}
    if discovered:
        say("\n  Dataset phát hiện trong runs/datasets/:")
        for idx, item in enumerate(discovered, 1):
            shortcuts[str(idx)] = item["path"]
            say(f"    [{idx}] {item['name']} ({item['kind']}) -> {item['path']}")
        say("  Nhập số [1..N] để chọn nhanh hoặc dán đường dẫn đầy đủ.")
    device = "auto"
    while True:
        source = path_input("Dataset ảnh (thư mục / YAML / JSON / CSV caption)", shortcuts=shortcuts or None)
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
    det_metric = "map50"
    chosen_models = None
    if data["task"] == "detection" and not grid:
        scanned_weights = scan_weights()
        local_pts = [w["path"] for w in scanned_weights]
        default_pts = "; ".join(p for p in local_pts if Path(p).name in {"yolo11n.pt", "yolo11s.pt"}) or ("; ".join(local_pts[:2]) if local_pts else None)
        say("\n  Không tự tải weights qua mạng trong lúc chạy.")
        if scanned_weights:
            say("  Chi tiết local pretrained weights (weights/*.pt):")
            for w in scanned_weights:
                p_str = f"{w['parameters']:,} params" if w.get("parameters") is not None else f"{w['size_mb']} MB"
                g_str = f"{w['gflops_640']:.2f} GFLOPs@640 ({w['gflops_reference_32']:.4f}@32)" if w.get("gflops_640") is not None else "GFLOPs N/A"
                say(f"    - {w['name']}: {p_str} | {g_str}")
        if default_pts:
            say(f"  Phát hiện local pretrained weights: {default_pts}")
            say("  [1] Dùng local pretrained .pt (khuyên dùng cho thực nghiệm NCKH thực tế)")
            say("  [2] Dùng kiến trúc YAML khởi tạo từ đầu (yolo11n.yaml / yolo11s.yaml)")
            if choice("Nguồn mô hình detection", {"1", "2"}, "1") == "1":
                value = ask("Checkpoint local .pt, nhiều file cách nhau bằng dấu ;", default_pts)
                chosen_models = [v.strip().strip('"').strip("'") for v in value.split(";") if v.strip()]
        else:
            say("  Enter = YOLO11n/s YAML (không pretrained), hoặc nhập đường dẫn file .pt local cách nhau bằng dấu ;")
            value = ask("Checkpoint local .pt (Enter = mặc định YAML)")
            if value:
                chosen_models = [v.strip().strip('"').strip("'") for v in value.split(";") if v.strip()]
        if chosen_models:
            if any(not Path(v).is_file() or Path(v).suffix.lower() != ".pt" for v in chosen_models):
                raise ValueError("Pretrained detection models must be existing local .pt files")
            argv += ["--models", *chosen_models]
    if grid:
        argv += ["--grid", grid]
    weights = [0.6, 0.25, 0.15]
    rule, method = "weighted", "adaptive_full"
    advanced = choice("Đặt mục tiêu/giới hạn nâng cao? y/n", {"y", "n"}, "n")
    if advanced == "y":
        if data["task"] == "detection":
            det_metric = choice("Metric chính cho detection: map50 / map50_95", {"map50", "map50_95"}, "map50")
            argv += ["--detection-metric", det_metric]
        from .selection import RULES
        rule = choice("Quy tắc: weighted / tchebycheff / topsis / knee", set(RULES), "weighted")
        argv += ["--selection-rule", rule]
        methods = {"adaptive_full", "adaptive_ehvi", "adaptive_staged_seeds", "reserve", "successive_halving", "prior_full", "random"}
        method = choice("Thuật toán: " + " / ".join(sorted(methods)), methods, "adaptive_full")
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
        suggested_t, suggested_m = suggest_scales(hw, data, epochs)
        say(f"  Gợi ý chuẩn hóa theo phần cứng ({hw.get('device', 'cpu')}): time_scale={suggested_t}s, memory_scale={suggested_m} MiB")
        time_scale = numeric("Thang chuẩn hóa thời gian, giây", suggested_t)
        memory_scale = numeric("Thang chuẩn hóa bộ nhớ, MiB", suggested_m)
        if memory is not None:
            argv += ["--max-memory", str(memory)]
        if train_time is not None:
            argv += ["--max-time", str(train_time * 60)]
        argv += ["--time-scale", str(time_scale), "--memory-scale", str(memory_scale)]
        output = ask("Thư mục lưu lượt mới", "runs/v2").strip('"').strip("'")
        argv += ["--output", output]
    title("BƯỚC 3/3 — XEM LẠI TRƯỚC KHI CHẠY")
    review_rows = [("Dataset / task", f"{source} / {data['task']}"), ("Thiết bị", device)]
    if data["task"] == "detection":
        review_rows.append(("Metric chính (khóa)", "mAP50 (IoU=0.50)" if det_metric == "map50" else "mAP50-95 (IoU=0.50:0.95)"))
    review_rows += [("Ngân sách chung", duration(minutes * 60)), ("Epoch / full lượt", epochs),
                    ("Seeds", seeds), ("Tập cấu hình", grid or ("compact (3)" if scope == "2" else "research (tổ hợp; số chính xác sau validate)" if scope == "4" else "core (12)")),
                    ("Weights A / T / M", weights), ("Quy tắc lựa chọn", rule), ("Thuật toán", method + " + full confirmation; không dùng test")]
    pairs(review_rows)
    if data["task"] == "detection" and not grid and not chosen_models:
        say("  Đang dùng kiến trúc YOLO11 YAML (không pretrained). Dùng local .pt hoặc grid JSON nếu cần pretrained.")
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
        can_export = saved.get("phase") == "complete"
        say("\n  [1] Xem trạng thái và kết quả (chỉ đọc)")
        say("  [2] Audit chứng cứ gốc (không training)")
        say("  [3] " + ("Bắt đầu protocol chưa chạy" if saved.get("phase") == "prepared" else "Tiếp tục search với budget bổ sung"))
        if selection:
            say("  [4] Train lại cấu hình được chọn (lượt riêng)")
            say("  [5] Đánh giá test sau lựa chọn (mở test có ghi lịch sử)")
        if can_export:
            say("  [6] Xuất bảng LaTeX / CSV / biểu đồ SVG cho bài NCKH (paper-export)")
        say("  [0] Quay lại menu")
        options = {"0", "1", "2", "3", *(["4", "5"] if selection else []), *(["6"] if can_export else [])}
        action = choice("Chọn thao tác", options, "1")
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
        elif action == "5":
            argv = ["evaluate-v2", "--run", str(folder)]
            history = read_json(folder / "test_history.json", optional=True) or []
            if history:
                say("  Test đã được mở; nên xem kết quả cũ bằng thao tác 1.")
                if choice("Chủ động cho phép đánh giá test LẠI? y/n", {"y", "n"}, "n") != "y":
                    continue
                argv.append("--allow-retest")
            note = "Mở test trên toàn bộ seeds đã khóa của cấu hình đã chọn; không dùng test để chọn lại cấu hình."
        else:
            default_out = str(Path("runs") / "paper_exports" / folder.name)
            out_dir = ask("Thư mục mới để xuất tài liệu NCKH", default_out).strip('"').strip("'")
            argv = ["paper-export", "--run", str(folder), "--output", out_dir]
            note = "Chỉ đọc & audit chứng cứ gốc, xuất LaTeX/CSV/SVG sang thư mục mới; không sửa lượt chạy."
        if confirm(argv, note=note):
            dispatch(argv)
        else:
            say("  Đã hủy thao tác.")


def benchmark_wizard(dispatch):
    source, device, data, hw = inspect_dataset()
    if data["task"] not in {"detection", "classification", "captioning"} or not data.get("validation_ready"):
        say("\n  Chưa thể chạy benchmark v2: cần task được hỗ trợ, nhãn thật và validation độc lập.")
        return
    if not hw.get("ml_runtime_ready"):
        say("\n  Runtime chưa sẵn sàng. Chọn 'Kiểm tra môi trường' trong menu để chẩn đoán.")
        return
    title("BENCHMARK ĐỐI CHỨNG THUẬT TOÁN", "Chạy lặp nhiều search seeds và đối chiếu reference độc lập.")
    minutes = numeric("Budget cho MỖI lần chạy thuật toán (phút)", 15)
    epochs = numeric("Số epoch cho MỖI lượt full training", 10, integer=True)
    say("\n  [1] Core  [2] Compact  [3] Grid JSON  [4] Research")
    scope = choice("Chọn tập cấu hình", {"1", "2", "3", "4"}, "1")
    grid = path_input("File grid JSON") if scope == "3" else None
    seeds_str = ask("Training seeds (cách nhau bằng khoảng trắng)", "17 29 43")
    seeds = [int(s) for s in seeds_str.split()]
    methods_str = ask(
        "Thuật toán đối chứng (cách nhau bằng khoảng trắng)",
        "adaptive_full adaptive_ehvi random successive_halving",
    )
    methods = methods_str.split()
    search_seeds_str = ask("Search seeds (lặp benchmark)", "17 29 43")
    search_seeds = [int(s) for s in search_seeds_str.split()]
    ref_minutes = numeric("Budget reference độc lập (phút)", 60)
    from .selection import RULES
    rule = choice("Quy tắc lựa chọn: " + " / ".join(sorted(RULES)), set(RULES), "weighted")
    suggested_t, suggested_m = suggest_scales(hw, data, epochs)
    say(f"  Gợi ý chuẩn hóa theo phần cứng ({hw.get('device', 'cpu')}): time_scale={suggested_t}s, memory_scale={suggested_m} MiB")
    time_scale = numeric("Thang chuẩn hóa thời gian, giây", suggested_t)
    memory_scale = numeric("Thang chuẩn hóa bộ nhớ, MiB", suggested_m)
    output = ask("Thư mục lưu benchmark study", "runs/benchmark_v2").strip('"').strip("'")
    argv = [
        "benchmark", "--data", source, "--device", device, "--task", data["task"],
        "--budget-seconds", str(minutes * 60), "--full-epochs", str(epochs),
        "--space", "compact" if scope == "2" else "research" if scope == "4" else "core",
        "--seeds", *map(str, seeds),
        "--methods", *methods,
        "--search-seeds", *map(str, search_seeds),
        "--reference-budget-seconds", str(ref_minutes * 60),
        "--selection-rule", rule,
        "--time-scale", str(time_scale), "--memory-scale", str(memory_scale),
        "--output", output,
    ]
    if grid:
        argv += ["--grid", grid]
    if data["task"] == "detection" and not grid:
        scanned_weights = scan_weights()
        if scanned_weights:
            say("  Local pretrained weights (weights/*.pt):")
            for w in scanned_weights:
                p_str = f"{w['parameters']:,} params" if w.get("parameters") is not None else f"{w['size_mb']} MB"
                g_str = f"{w['gflops_640']:.2f} GFLOPs@640" if w.get("gflops_640") is not None else "GFLOPs N/A"
                say(f"    - {w['name']}: {p_str} | {g_str}")
            default_pts = "; ".join(w["path"] for w in scanned_weights[:2])
            val = ask("Checkpoint local .pt (cách nhau bằng ; hoặc Enter bỏ qua)", default_pts)
            pts = [v.strip().strip('"').strip("'") for v in val.split(";") if v.strip()]
            if pts:
                argv += ["--models", *pts]
    if confirm(argv, note="Khởi động benchmark nhiều thuật toán + reference độc lập."):
        dispatch(argv)
    else:
        say("  Đã hủy benchmark.")


def ui_wizard(dispatch, root="runs"):
    title("WEB DASHBOARD STUDIO OFFLINE (TIER 2)", "100% offline trên loopback 127.0.0.1; không gửi dữ liệu ra ngoài.")
    host = ask("Host loopback (127.0.0.1 / localhost / ::1)", "127.0.0.1")
    port = numeric("Cổng HTTP (0 = tự chọn)", 8765, integer=True, minimum=-1, maximum=65535)
    runs_dir = ask("Thư mục runs hiển thị", root).strip('"').strip("'") or root
    argv = ["ui", "--host", host, "--port", str(port), "--runs-dir", runs_dir]
    if confirm(argv, note="Khởi động Web Dashboard Studio local (Ctrl+C để dừng server)."):
        dispatch(argv)
    else:
        say("  Đã hủy mở Web Dashboard.")


def menu(dispatch, root="runs", *, start=False):
    try:
        if start:
            optimize_wizard(dispatch)
            return 0
        while True:
            title(f"TRAIN CONFIG {__version__} — TÌM CẤU HÌNH HUẤN LUYỆN",
                  "Cân bằng chất lượng, bộ nhớ và thời gian bằng số đo thực nghiệm.")
            _say_styled("\n  [1] Tìm cấu hình mới — hướng dẫn từng bước " + style_text("(optimize)", "info"))
            say("  [2] Kiểm tra dataset và phần cứng (inspect — không training)")
            say("  [3] Kiểm tra môi trường (doctor)")
            say("  [4] Xem / quản lý các lượt chạy đã lưu")
            _say_styled("  [5] Đối chứng nhiều thuật toán & reference " + style_text("(benchmark)", "info"))
            _say_styled("  [6] Mở Web Dashboard Studio offline " + style_text("(ui)", "success"))
            say("  [7] Trợ giúp lệnh nâng cao (--help)")
            say("  [0] Thoát")
            say("\n  Enter chọn mục mặc định; b quay lại; Ctrl+C hủy an toàn.")
            try:
                selected = choice("Chọn chức năng", {"0", "1", "2", "3", "4", "5", "6", "7"}, "1")
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
                elif selected == "5":
                    benchmark_wizard(dispatch)
                elif selected == "6":
                    ui_wizard(dispatch, root)
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
