"""Dependency-free terminal presentation for the command-line application."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import shutil
import textwrap

COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
CYAN, GREEN, YELLOW, RED, BOLD, DIM, RESET = ("\033[96m", "\033[92m", "\033[93m", "\033[91m", "\033[1m", "\033[2m", "\033[0m") if COLOR else ("",) * 7


def _paint(text, *styles):
    return "".join(styles) + str(text) + RESET if COLOR else str(text)


def _table(title, headers, rows):
    rows = [[str(value) for value in row] for row in rows]
    widths = [max([len(headers[i])] + [len(row[i]) for row in rows]) for i in range(len(headers))]
    available = max(20, shutil.get_terminal_size(fallback=(120, 24)).columns - 2 - 3 * (len(headers) - 1))
    minimum = [len(header) for header in headers]
    while sum(widths) > available:
        reducible = [i for i in range(len(widths)) if widths[i] > minimum[i]]
        if not reducible:
            break
        index = max(reducible, key=lambda i: widths[i])
        widths[index] -= 1
    print(f"\n{_paint(title, BOLD, CYAN)}")
    print("  " + " │ ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  " + "─┼─".join("─" * width for width in widths))
    for row in rows:
        cells = [textwrap.wrap(value, width=widths[i], break_on_hyphens=False,
                               replace_whitespace=False, drop_whitespace=False) or [""] for i, value in enumerate(row)]
        for line in range(max(map(len, cells))):
            print("  " + " │ ".join((cells[i][line] if line < len(cells[i]) else "").ljust(widths[i]) for i in range(len(headers))))


def duration(seconds):
    if seconds is None:
        return "—"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f} giây"
    if seconds < 3600:
        return f"{seconds / 60:.1f} phút"
    return f"{seconds / 3600:.1f} giờ"


def memory(megabytes):
    return "Không áp dụng" if megabytes in (None, 0) else f"{float(megabytes):,.0f} MB"


def heading(title, subtitle=None):
    line = "═" * 78
    print(f"\n{_paint(line, CYAN)}\n  {_paint(title, BOLD)}")
    if subtitle:
        print(f"  {_paint(subtitle, DIM)}")
    print(_paint(line, CYAN))


def hardware_summary(info):
    gpu = info.get("gpu") or "CPU"
    precision = ", ".join(name for name, enabled in (
        ("FP32", True), ("FP16", info.get("fp16")), ("BF16", info.get("bf16"))
    ) if enabled)
    vram = (f"{memory(info.get('vram_used_mb'))} / {memory(info.get('vram_total_mb'))}"
            if info.get("gpu") else "Không áp dụng")
    capability = ".".join(map(str, info.get("compute_capability", []))) or "—"
    ready = "MPS sẵn sàng" if info.get("device")=="mps" else "Sẵn sàng" if info.get("cuda_training_ready") else (
        "Chưa sẵn sàng" if info.get("gpu") else "CPU"
    )
    _table(
        "01  PHẦN CỨNG",
        ["Thiết bị", "CPU / GPU", "VRAM dùng/tổng", "Compute", "Precision", "CUDA train"],
        [[info.get("device", "—"), gpu, vram, capability, precision, ready]],
    )
    if info.get("gpu"):
        print(
            f"  Nguồn đo: {info.get('gpu_source', '—')}  •  Driver {info.get('driver_version', '—')}  •  "
            f"Nhiệt độ {info.get('temperature_c', '—')}°C  •  GPU load {info.get('gpu_utilization_percent', '—')}%"
        )
        print(
            f"  UUID: {info.get('gpu_uuid', '—')}  •  PCI: {info.get('pci_bus_id', '—')}  •  "
            f"PyTorch CUDA: {info.get('torch_cuda_version') or 'không có'}"
        )
        if not info.get("cuda_training_ready") and info.get("device")!="mps":
            warning("Đã nhận đúng GPU vật lý, nhưng PyTorch hiện là bản CPU; chưa thể tune/train bằng GPU.")


def dataset_summary(stats):
    if "counts" in stats:
        print(f"\n{_paint('02  DATASET', BOLD, CYAN)}")
        print(f"  Nguồn: {stats.get('source', '—')}")
        counts = stats["counts"]
        _table("  Phân chia dữ liệu", ["Task", "Train", "Val", "Test", "Validation độc lập"], [[
            stats.get("task", "—"), counts.get("train", 0), counts.get("val", 0),
            counts.get("test", 0), "Có" if stats.get("validation_ready") else "Không",
        ]])
        if stats.get("task") == "captioning":
            captions = stats.get("captions", {})
            print(f"  Caption train/val/test: {captions.get('train', '—')} / "
                  f"{captions.get('val', '—')} / {captions.get('test', '—')}")
        elif stats.get("names"):
            print(f"  Số lớp: {len(stats['names'])}")
        return
    _table("02  DATASET", ["Nguồn", "Train", "Val", "Subset train/val", "Số lớp"], [[
        stats.get("source", "—"), stats.get("train_images", "—"), stats.get("val_images", "—"),
        f"{stats.get('train_subset_images', '—')} / {stats.get('val_subset_images', '—')}",
        stats.get("classes", stats.get("num_classes", "—")),
    ]])


def search_plan(args, candidates):
    print(f"\n{_paint('03  KẾ HOẠCH TÌM KIẾM', BOLD, CYAN)}")
    print(f"  {len(candidates)} cấu hình  •  Rungs {' → '.join(map(str, args.rungs))} epoch  •  "
          f"Ngân sách {duration(args.budget * 60)}  •  Preset {_paint(args.preset, BOLD)}\n")


def candidate_cap(tested, total):
    warning(f"Giới hạn ứng viên: kiểm tra {tested} / {total} cấu hình.")


def trial(result):
    config = result["config"]
    status = result.get("status", "—")
    labels = {"ok": ("THÀNH CÔNG", GREEN), "oom": ("HẾT VRAM", YELLOW), "timeout": ("QUÁ GIỜ", YELLOW), "error": ("LỖI", RED)}
    label, color = labels.get(status, (str(status).upper(), ""))
    mode = "Đo batch" if result.get("batch_verification") else {
        "profile": "Đo batch", "search": "Tìm kiếm", "train": "Huấn luyện"
    }.get(result.get("mode"), "Thử nghiệm")
    text = (f"  {mode:<11} {_paint(label, BOLD, color)}  {config.get('model', '—')}  "
            f"ảnh {config.get('imgsz', '—')}  batch {config.get('batch', '—')}")
    if "width" in config:
        text += f"  width {config['width']}  LR {config.get('lr0', '—')}"
        if config.get("model")=="attn_gru_captioner":
            text += f"  hidden {config.get('hidden', '—')}  dropout {config.get('dropout', '—')}"
    elif "freeze" in config:
        text += f"  freeze {config['freeze']}"
    if "map50_proxy" in result:
        text += f"  │  mAP50 proxy {result['map50_proxy']:.4f}  epoch {result.get('epochs_completed', '—')}"
    elif result.get("validation_loss") is not None:
        text += f"  │  Val loss {result['validation_loss']:.4f}  epoch {result.get('epochs_completed', '—')}"
        if result.get("task") == "classification" and result.get("quality_score") is not None:
            text += f"  accuracy {result['quality_score']:.2%}"
    if result.get("caption_metrics"):
        metric = result["caption_metrics"]
        text += f"  │  BLEU4 {metric['bleu4']:.2f}  chrF {metric['chrf']:.2f}"
    if result.get("batch_verification") and result.get("throughput") is not None:
        text += f"  │  {result['throughput']:.1f} mẫu/giây"
    if result.get("worker_wall_seconds") is not None:
        text += f"  │  {duration(result['worker_wall_seconds'])}"
    print(text, flush=True)
    if result.get("error"):
        print(_paint(f"               {result['error']}", RED), flush=True)
    if result.get("resource_feasible") is False:
        warning("Trial hoàn tất nhưng vượt mức bộ nhớ an toàn; không dùng để chọn cấu hình.")
    if result.get("quality_eligible") is False:
        warning("Caption sinh chưa vượt đối chứng/đang lặp; không chứng nhận cấu hình đạt chất lượng.")


def trial_start(config, epochs, index=None, total=None, remaining=None, batch_probe=False):
    """Print before starting a subprocess, including on non-interactive terminals."""
    position = f"[{index}/{total}] " if index is not None else ""
    action = "Kiểm chứng batch" if batch_probe else "Đang chạy"
    print(f"\n  {_paint(position + action, BOLD, CYAN)}: {config.get('model', '—')} • "
          f"ảnh {config.get('imgsz', '—')} • batch {config.get('batch', '—')} • {epochs} epoch"
          + (f" • còn {duration(remaining)}" if remaining is not None else ""), flush=True)


def search_rung(epochs, candidates):
    print(f"\n{_paint(f'VÒNG {epochs} EPOCH — {candidates} cấu hình', BOLD, CYAN)}", flush=True)


def training_epoch(state, total):
    epoch = state.get("epochs_completed", 0)
    print(f"  Epoch {epoch}/{total} • Train loss {state.get('train_loss', float('nan')):.4f} • "
          f"Val loss {state.get('validation_loss', float('nan')):.4f} • "
          f"Đã chạy {duration(state.get('wall_seconds'))} • checkpoint đã lưu", flush=True)
    if state.get("caption_metrics"):
        m = state["caption_metrics"]
        print(f"    Caption val: BLEU4 {m['bleu4']:.2f} • chrF {m['chrf']:.2f} • "
              f"best epoch {state.get('best_epoch', '—')} • LR {state.get('lr', '—')}", flush=True)


def pipeline_results(report, selection, folder):
    """Task-aware results; do not present caption loss as detection mAP."""
    task = selection.get("task", report.get("task", "detection"))
    config = selection["config"]
    chosen = selection.get("preset")
    presets = report.get("presets", {})
    if chosen is None:
        matches = [name for name, row in presets.items() if row.get("config") == config]
        chosen = "Balanced" if "Balanced" in matches else next(iter(matches), None)
    metric = {"captioning": "Val CE ↓", "reconstruction": "Val MSE ↓",
              "classification": "Val accuracy ↑", "detection": "mAP50 proxy ↑"}.get(task, "Metric")
    generated = task == "captioning" and report.get("quality_metric") == "validation_generated_bleu4_chrf"
    if generated:
        metric = "BLEU4 ↑"
    mem_label = "Peak VRAM" if report.get("hardware", {}).get("device", "cpu") not in {"cpu", "mps"} else "Peak RAM"
    rows = []
    for name in ("Accuracy", "Time", "Memory", "Balanced"):
        row = presets.get(name)
        if not row:
            continue
        c = row["config"]
        if generated:
            value = row.get("caption_metrics", {}).get("bleu4")
            score = f"{value:.2f}" if value is not None else "—"
        elif task == "classification":
            value = row.get("quality_score")
            score = f"{value:.2%}" if value is not None else "—"
        else:
            value = row.get("map50_proxy" if task == "detection" else "validation_loss")
            score = f"{value:.4f}" if value is not None else "—"
        rows.append([("● " if name == chosen else "  ") + name, c.get("imgsz", "—"),
                     c.get("batch", "—"), c.get("hidden", "—") if generated else c.get("width", c.get("freeze", "—")), c.get("lr0", "—"),
                     score, duration(row.get("estimated_full_seconds")),
                     memory(row.get("memory_mb", row.get("peak_reserved_mb")))])
    if rows:
        _table("04  SO SÁNH CÁC PRESET", ["Preset", "Ảnh", "Batch", "Hidden" if generated else "Width" if task != "detection" else "Freeze",
               "LR", metric, "Ước tính train", mem_label], rows)
    _table("05  CẤU HÌNH ĐÃ CHỌN", ["Tham số", "Giá trị"], list(config.items()))
    elapsed = report.get("search_wall_seconds")
    print(f"\n  Tìm kiếm: {duration(elapsed)} / ngân sách {duration(report.get('budget_seconds'))}")
    print(f"  Trial đã ghi: {len(report.get('history', []))} • Task: {task} • Thiết bị: {selection.get('device', '—')}")
    status = selection.get("status", "recommended_unverified")
    if status == "measured_best_observed":
        success(f"Đã chọn {chosen or 'cấu hình'} từ phép đo thực tế ({status}).")
    elif status == "measured_quality_not_established":
        warning("Đã đo nhưng caption CHƯA ĐẠT ĐỐI CHỨNG; cấu hình lưu chỉ để tiếp tục huấn luyện, không phải chứng nhận chất lượng.")
    else:
        warning(f"Chỉ có cấu hình đề xuất CHƯA KIỂM CHỨNG ({status}); chưa có trial phù hợp hoàn tất.")
    if not selection.get("final_accuracy_verified"):
        warning("Chưa xác nhận chất lượng sau full training; không phải tối ưu toàn cục.")
    notes = list(dict.fromkeys(report.get("warnings", [])))
    warnings(notes[:5])
    if len(notes) > 5:
        print(f"  … còn {len(notes) - 5} lưu ý trong report.json.")
    folder = Path(folder).resolve()
    print()
    artifact("Báo cáo", folder / "REPORT.md")
    artifact("Chi tiết phép đo", folder / "report.json")
    artifact("Cấu hình", folder / "selected.json")
    if selection.get("trainable") is False:
        warning("Chưa đủ runtime hoặc validation để train; xem các lưu ý trước khi chạy.")
    else:
        print(f"\n{_paint('LỆNH FULL TRAINING (chạy riêng)', BOLD, CYAN)}")
        invocation = f'"{sys.executable}"'
        if os.name == "nt":
            invocation = "& " + invocation  # PowerShell requires & for a quoted executable path.
        print(f'  {invocation} -m yolo_tuner train --selection "{folder / "selected.json"}"', flush=True)


def results(report, chosen_name=None):
    rows = []
    for name in ("Accuracy", "Time", "Memory", "Balanced"):
        row = report.get("presets", {}).get(name)
        if not row:
            continue
        config = row["config"]
        rows.append([("● " if name == chosen_name else "  ") + name, config.get("model"), config.get("imgsz"),
                     config.get("batch"), config.get("precision"), f"{row.get('map50_proxy', 0):.4f}",
                     duration(row.get("estimated_full_seconds")), memory(row.get("peak_reserved_mb"))])
    _table("04  KẾT QUẢ ĐỀ XUẤT", ["Preset", "Model", "Ảnh", "Batch", "Precision", "mAP50 proxy", "Ước tính train", "Peak VRAM"], rows)
    if chosen_name:
        success(f"Đã chọn {chosen_name}. Dùng selected.json để chạy full training.")


def warnings(items):
    if items:
        translations = {
            "Short validation measurements do not certify final or globally optimal accuracy.":
                "Phép đo validation ngắn chưa xác nhận chất lượng cuối hoặc tối ưu toàn cục.",
            "Full-training time is a measured extrapolation, not a guaranteed runtime.":
                "Thời gian full training là ngoại suy từ phép đo, không phải thời gian được bảo đảm.",
            "No complete feasible trial: a conservative, unverified training recommendation was saved.":
                "Chưa có trial khả thi hoàn tất; đã lưu cấu hình bảo thủ chưa kiểm chứng.",
            "Final accuracy and accuracy constraints require full held-out validation.":
                "Độ chính xác cuối và ngưỡng accuracy chỉ được xác nhận sau full training trên tập validation độc lập.",
            "Time extrapolation and two-point VRAM proposals have no calibrated coverage guarantee.":
                "Thời gian và VRAM là ước tính từ phép đo ngắn, chưa có bảo đảm sai số đã hiệu chỉnh.",
            "Train/val paths are disjoint; near-duplicates or source/video groups need separate auditing.":
                "Train/val không trùng đường dẫn; vẫn cần tự kiểm tra ảnh gần trùng và nhóm cùng video/nguồn.",
            "The scheduler includes worker startup/validation; it may return a partial cohort at timeout.":
                "Ngân sách gồm cả khởi động và validation; hết giờ có thể chỉ trả nhóm ứng viên đã hoàn tất.",
        }
        print(f"\n{_paint('LƯU Ý KHI ĐỌC KẾT QUẢ', BOLD, YELLOW)}")
        for item in items:
            print(f"  • {translations.get(item, item)}")


def artifact(label, path):
    print(f"{_paint(label + ':', BOLD, CYAN)} {Path(path).resolve()}")


def success(message):
    print(f"\n{_paint('✓', BOLD, GREEN)} {message}")


def warning(message):
    print(f"{_paint('!', BOLD, YELLOW)} {message}")


def error(message):
    print(f"\n{_paint('LỖI', BOLD, RED)}\n  {message}", file=sys.stderr)
