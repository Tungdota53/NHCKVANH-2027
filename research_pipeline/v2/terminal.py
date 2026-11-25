"""Human-readable terminal views; never change evidence or select a configuration."""

import json
import hashlib
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import unicodedata

from . import __version__

_NO_COLOR = False
_VT_INITIALIZED = False
_ANSI_SGR_RE = re.compile(r"\x1b\[[0-9;]*m")

_SGR = {
    "plain": "",
    "bold": "\x1b[1m",
    "muted": "\x1b[2m",
    "success": "\x1b[1;32m",
    "info": "\x1b[1;36m",
    "warn": "\x1b[1;33m",
    "danger": "\x1b[1;31m",
    "accent": "\x1b[1;35m",
}

PHASES = {"preparing": "Đang chuẩn bị", "prepared": "Đã lập kế hoạch, chưa chạy",
          "search": "Đang tìm kiếm (trạng thái lưu)", "complete": "Đã kết thúc",
          "interrupted": "Bị gián đoạn", "selection_committed_report_error": "Đã khóa lựa chọn, lỗi xuất báo cáo"}
VERDICTS = {
    "empirical_optimal_in_declared_space": "TỐI ƯU THỰC NGHIỆM TRONG TẬP ĐÃ ĐO ĐỦ",
    "fully_trained_best_observed": "TỐT NHẤT ĐÃ ĐO — CHƯA ĐO ĐỦ TẬP CẤU HÌNH",
    "no_verified_feasible_selection": "CHƯA CÓ CẤU HÌNH ĐỦ ĐIỀU KIỆN",
    "contended_smoke_not_certified": "KHÔNG CHỨNG NHẬN: MÁY BỊ TRANH TÀI NGUYÊN",
}


def set_no_color(enabled: bool):
    global _NO_COLOR
    _NO_COLOR = bool(enabled)


def color_enabled(stream=None) -> bool:
    global _VT_INITIALIZED
    stream = stream or sys.stdout
    if _NO_COLOR or os.environ.get("NO_COLOR", "") != "" or os.environ.get("TERM", "").lower() == "dumb":
        return False
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty) or not isatty():
        return False
    if os.name == "nt" and not _VT_INITIALIZED:
        try:
            import colorama
            colorama.just_fix_windows_console()
        except Exception:
            pass
        _VT_INITIALIZED = True
    return True


def _supports_utf8(stream=None) -> bool:
    stream = stream or sys.stdout
    encoding = (getattr(stream, "encoding", None) or "utf-8").lower()
    return "utf" in encoding


def interactive():
    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def _is_safe_printable(c: str) -> bool:
    return c >= " " and not ("\x7f" <= c <= "\x9f")


def clean(value):
    # File names and worker errors are data, not terminal control instructions.
    return "".join(c if _is_safe_printable(c) else " " for c in str(value))


def width(value):
    visible = _ANSI_SGR_RE.sub("", str(value))
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in visible)


def clip(value, limit):
    value = clean(value)
    if width(value) <= limit:
        return value
    result = ""
    for c in value:
        if width(result + c) > max(0, limit - 3):
            break
        result += c
    return result + "." * min(3, limit)


def style_text(text, tone="plain", *, stream=None):
    sanitized = clean(text)
    prefix = _SGR.get(tone, "")
    if not prefix or not color_enabled(stream):
        return sanitized
    return f"{prefix}{sanitized}\x1b[0m"


def badge(label, tone="info", *, stream=None):
    tag = f"[{clean(label)}]"
    return style_text(tag, tone, stream=stream)


def _write_line(value="", *, file=None, allow_sgr=False):
    stream = file or sys.stdout
    raw = str(value)
    if allow_sgr and color_enabled(stream):
        # Preserve only explicit SGR sequences (\x1b[...m); strip any other control chars.
        parts = []
        pos = 0
        for m in _ANSI_SGR_RE.finditer(raw):
            seg = "".join(c for c in raw[pos:m.start()] if c in "\n\t" or _is_safe_printable(c))
            parts.append(seg)
            parts.append(m.group(0))
            pos = m.end()
        parts.append("".join(c for c in raw[pos:] if c in "\n\t" or _is_safe_printable(c)))
        value = "".join(parts)
    else:
        plain = _ANSI_SGR_RE.sub("", raw) if allow_sgr else raw
        value = "".join(c for c in plain if c in "\n\t" or _is_safe_printable(c))
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        value.encode(encoding)
    except (UnicodeError, LookupError):
        decomposed = unicodedata.normalize("NFKD", value.replace("đ", "d").replace("Đ", "D"))
        value = "".join(c for c in decomposed if not unicodedata.combining(c)).encode("ascii", "replace").decode()
    print(value, file=stream, flush=True)


def say(value="", *, file=None):
    _write_line(value, file=file, allow_sgr=False)


def _say_styled(value="", *, file=None):
    _write_line(value, file=file, allow_sgr=True)


def title(label, note=None):
    columns = max(20, min(88, shutil.get_terminal_size((88, 24)).columns))
    rule_char = "─" if _supports_utf8() else "="
    _say_styled("\n" + style_text(rule_char * columns, "muted"))
    _say_styled(style_text(clip(label, columns), "bold"))
    if note:
        say(note)
    _say_styled(style_text(rule_char * columns, "muted"))


def pairs(rows):
    size = min(24, max((width(k) for k, _ in rows), default=0))
    for key, value in rows:
        label = clean(key)
        _say_styled(f"  {style_text(label, 'muted')}{' ' * max(0, size - width(label))} : {clean(value)}")


def duration(seconds):
    if not isinstance(seconds, (float, int)) or not math.isfinite(seconds):
        return "chưa đo"
    seconds = max(0, seconds)
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} giờ"
    if seconds >= 60:
        return f"{seconds / 60:.1f} phút"
    return f"{seconds:.1f} giây"


def number(value, digits=4):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) and math.isfinite(value) else "chưa đo"


def command(argv):
    parts = [sys.executable, "-m", "research_pipeline.v2", *map(str, argv)]
    # PowerShell literal strings avoid expansion of $, backticks and whitespace.
    if os.name == "nt":
        return "& " + " ".join("'" + p.replace("'", "''") + "'" for p in parts)
    return shlex.join(parts)


def read_json(path, *, optional=False):
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        if optional:
            return None
        raise


def _verdict_badge(verdict):
    mapping = {
        "empirical_optimal_in_declared_space": ("EMPIRICAL OPTIMAL", "success"),
        "fully_trained_best_observed": ("BEST OBSERVED", "info"),
        "no_verified_feasible_selection": ("NO FEASIBLE", "warn"),
        "contended_smoke_not_certified": ("CONTENDED", "danger"),
    }
    label, tone = mapping.get(verdict, (str(verdict).upper(), "info"))
    return badge(label, tone)


def _candidate_status(row):
    runs = row.get("runs") or []
    has_oom = any(run.get("status") == "oom" for run in runs)
    has_timeout = any(run.get("status") == "timeout" for run in runs)
    if has_oom and has_timeout:
        return "OOM/TIMEOUT", "danger"
    if has_oom:
        return "OOM", "danger"
    if has_timeout:
        return "TIMEOUT", "danger"
    state = row.get("state")
    if state == "confirmed_feasible":
        return "OK", "success"
    if state in {"infeasible", "confirmed_constraint_failure", "observed_required_seed_oom"}:
        return ("OOM", "danger") if state == "observed_required_seed_oom" else ("FAIL", "danger")
    if row.get("completed_seeds"):
        return "PARTIAL", "warn"
    return "UNRES", "muted"


def _render_table(headers, rows, *, right_align=None):
    right_align = set(right_align or ())
    utf8 = _supports_utf8()
    widths = [max([width(h)] + [width(r[i]) for r in rows]) for i, h in enumerate(headers)]

    def pad(cell, col_idx):
        text = str(cell)
        gap = max(0, widths[col_idx] - width(text))
        return (" " * gap + text) if col_idx in right_align else (text + " " * gap)

    if utf8:
        top = "  ┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"
        mid = "  ├" + "┼".join("─" * (w + 2) for w in widths) + "┤"
        bot = "  └" + "┴".join("─" * (w + 2) for w in widths) + "┘"
        sep = " │ "
        left, right = "  │ ", " │"
    else:
        top = "  +" + "+".join("-" * (w + 2) for w in widths) + "+"
        mid = top
        bot = top
        sep = " | "
        left, right = "  | ", " |"

    _say_styled(style_text(top, "muted"))
    head_cells = [style_text(pad(h, i), "bold") for i, h in enumerate(headers)]
    _say_styled(left + sep.join(head_cells) + right)
    _say_styled(style_text(mid, "muted"))
    for row in rows:
        _say_styled(left + sep.join(pad(cell, i) for i, cell in enumerate(row)) + right)
    _say_styled(style_text(bot, "muted"))


def hardware_view(hw):
    ready_badge = badge("READY", "success") if hw.get("ml_runtime_ready") else badge("NOT READY", "danger")
    pairs([("Thiết bị", hw.get("gpu") or hw.get("device", "chưa rõ")),
           ("ML runtime", f"{'Sẵn sàng' if hw.get('ml_runtime_ready') else 'CHƯA SẴN SÀNG'}"),
           ("RAM còn trống", number(hw.get("ram_available_mb"), 0) + " MiB"),
           ("VRAM còn trống", number(hw.get("vram_free_mb"), 0) + " MiB" if hw.get("vram_free_mb") is not None else "Không áp dụng / chưa đọc được"),
           ("Precision hỗ trợ", "FP32" + (", FP16" if hw.get("fp16") else "") + (", BF16" if hw.get("bf16") else ""))])
    _say_styled("  Trạng thái runtime: " + ready_badge)
    if hw.get("reason"):
        _say_styled("  Lý do: " + style_text(hw["reason"], "warn"))


def inspection_view(info):
    title("KIỂM TRA MÁY VÀ DỮ LIỆU", "Chỉ đọc thông tin; không khởi động training.")
    hardware_view(info["hardware"])
    data = info.get("dataset")
    if data:
        counts = data.get("counts") or {}
        say("\nDataset")
        pairs([("Đường dẫn", data.get("source")), ("Tác vụ", data.get("task")),
               ("Ảnh train / val / test", " / ".join(str(counts.get(s, "?")) for s in ("train", "val", "test"))),
               ("Validation độc lập", "Có" if data.get("validation_ready") else "KHÔNG — cần sửa trước khi tối ưu")])
        if data.get("task") == "captioning" and data.get("captions"):
            say("  Caption train / val / test: " + " / ".join(str(data["captions"].get(s, "?")) for s in ("train", "val", "test")))
        audit = data.get("audit")
        if isinstance(audit, dict):
            intra = audit.get("intra_split_duplicates") or {}
            leak = audit.get("cross_split_leakage") or {}
            pairs([("Trùng lặp nội bộ (SHA-256)", intra.get("total_duplicate_files", 0)),
                   ("Rò rỉ giữa train/val/test", "CÓ — CẦN SỬA" if leak.get("has_leakage") else "Không (0)")])
            if data.get("task") == "detection":
                issues = sum((audit.get(k) or {}).get("count", 0) for k in ("missing_labels", "empty_labels", "malformed_labels", "out_of_bounds_boxes", "degenerate_boxes", "invalid_class_ids"))
                imb = audit.get("imbalance_ratio")
                imb_str = f"{imb:.2f}x" if isinstance(imb, (int, float)) else "có lớp = 0"
                pairs([("Vấn đề nhãn/bbox", issues), ("Tỷ lệ mất cân bằng lớp", imb_str)])
        for warning in data.get("warnings") or []:
            _say_styled("  Lưu ý: " + style_text(warning, "warn"))


def summary_view(folder, protocol, report):
    verdict = report["decision"]["verification_status"]
    title("KẾT QUẢ TÌM CẤU HÌNH", VERDICTS.get(verdict, verdict))
    _say_styled("  Trạng thái chứng nhận: " + _verdict_badge(verdict))
    rows = report.get("candidates", [])
    complete = sum(len(r.get("completed_seeds", [])) for r in rows)
    expected = len(protocol["configs"]) * len(protocol["training_seeds"])
    say(f"  Full training hoàn tất: {complete}/{expected} lượt config × seed; {protocol['full_epochs']} epoch/lượt.")
    say("  Chỉ tối ưu theo objective, tập cấu hình, dữ liệu và máy đã khóa; không bảo đảm mọi GPU/dataset.")
    from .schema import detection_metric_for
    det_metric = detection_metric_for(protocol)
    det_label = "Val mAP50" if det_metric == "map50" else "Val mAP50-95"
    metric = {"detection": det_label, "classification": "Val accuracy", "captioning": "Val (BLEU4+chrF)/200"}.get(protocol["task"], "Validation")

    if rows:
        selected_idx = report["decision"].get("selected_index")
        pareto_ids = set(report["decision"].get("pareto") or [])
        val_col = det_label if protocol.get("task") == "detection" else ("Val Acc" if protocol.get("task") == "classification" else "Val Qual")
        table_rows = []
        for row in rows:
            cfg = row.get("config") or {}
            cid = str(row.get("config_id") or f"c{row.get('index', 0):02d}")[:8]
            model_name = clip(Path(str(cfg.get("model", "?"))).name, 14)
            img_b = f"{cfg.get('imgsz', '?')}/{cfg.get('batch', '?')}"
            lr_val = cfg.get("lr0")
            lr_str = f"{lr_val:g}" if isinstance(lr_val, (int, float)) else "-"
            req_seeds = row.get("required_seeds") or protocol.get("training_seeds") or []
            comp_seeds = row.get("completed_seeds") or []
            seeds_str = f"{len(comp_seeds)}/{len(req_seeds)}"
            st_label, st_tone = _candidate_status(row)
            st_cell = style_text(st_label, st_tone)
            q_mean = row.get("quality_mean")
            q_str = f"{q_mean:.4f}" if isinstance(q_mean, (int, float)) and math.isfinite(q_mean) else "-"
            t_mean = row.get("time_mean")
            t_str = f"{t_mean:.1f}s" if isinstance(t_mean, (int, float)) and math.isfinite(t_mean) else "-"
            m_max = row.get("memory_max")
            m_str = f"{m_max:.0f}M" if isinstance(m_max, (int, float)) and math.isfinite(m_max) else "-"
            u_val = row.get("utility")
            u_str = f"{u_val:.4f}" if isinstance(u_val, (int, float)) and math.isfinite(u_val) else "-"
            is_sel = row.get("index") == selected_idx
            is_par = row.get("config_id") in pareto_ids
            if is_sel and is_par:
                tag = style_text("★ Pareto", "success")
            elif is_sel:
                tag = style_text("★", "success")
            elif is_par:
                tag = style_text("Pareto", "info")
            else:
                tag = ""
            table_rows.append([
                f"c{int(row.get('index', 0)):02d} {cid}",
                model_name,
                img_b,
                lr_str,
                seeds_str,
                st_cell,
                q_str,
                t_str,
                m_str,
                u_str,
                tag,
            ])
        say("\nBảng ứng viên:")
        _render_table(
            ["Cfg", "Model", "Img/B", "LR", "Seeds", "Status", val_col, "Time", "Mem", "U", "Tag"],
            table_rows,
            right_align={2, 3, 4, 6, 7, 8, 9},
        )

    r = report.get("recommendation")
    if r:
        c = r["config"]
        say("\nCấu hình được chọn")
        pairs([("Model", Path(c["model"]).name), ("Image / batch", f"{c['imgsz']} / {c['batch']}"),
               ("Precision", c["precision"].upper()), ("Tích lũy gradient", c["accumulation"]),
               ("Effective batch", c["batch"] * c["accumulation"]),
               ("Optimizer / LR / decay", f"{c['optimizer']} / {c['lr0']} / {c['weight_decay']}"),
               ("Epoch / seeds", f"{r['full_epochs']} / {r['completed_seeds']}")])
        if "width" in c:
            pairs([("Width / hidden", f"{c['width']} / {c.get('hidden', 'không áp dụng')}")])
        if "freeze" in c:
            pairs([("Freeze blocks", c["freeze"])])
        if "selection_rule" in report:
            pairs([("Quy tắc đã khóa", report["selection_rule"]), ("Điểm theo quy tắc", number(r.get("selection_score"), 6))])
            if report.get("selection_fallback"):
                say("  Fallback công bố: " + report["selection_fallback"])
        sd = f" ± SD {number(r.get('validation_sd_not_ci'))}" if len(r["completed_seeds"]) > 1 else " (1 seed; chưa ước lượng biến thiên)"
        memory_label = "VRAM reserved đo được" if r["memory_kind"] == "cuda_reserved_allocator_mib" else "RAM RSS cây tiến trình"
        pairs([(metric, number(r["validation_mean"]) + sd), ("Thời gian / full lượt", duration(r["full_worker_seconds_mean"])),
               (memory_label, number(r["memory_max_mb"], 1) + " MiB"), ("Điểm cân bằng U", number(r["utility"], 6))])
        chosen_id = report["decision"].get("selected_config_id")
        det_diag = (report.get("detection_diagnostics") or {}).get(chosen_id or "")
        if det_diag:
            pairs([("Tại best epoch (phụ)",
                    f"mAP50-95={number(det_diag['map50_95_at_best']['mean'])} | "
                    f"P={number(det_diag['precision_at_best']['mean'])} | "
                    f"R={number(det_diag['recall_at_best']['mean'])} | "
                    f"F1={number(det_diag['f1_at_best']['mean'])}")])
        say("  SD không phải khoảng tin cậy; thời gian trên không phải toàn bộ chi phí search.")
    else:
        say("  Không thay kết quả full training thiếu bằng số đo probe hoặc cấu hình chưa xác minh.")
    pairs([("Thời gian search đã ghi", duration(report.get("online_session_seconds"))),
           ("Ngân sách đã cấp", duration(report.get("total_granted_budget_seconds", report.get("budget_seconds")))),
           ("Báo cáo", Path(folder).resolve() / "REPORT.md")])
    if r:
        if report.get("presets"):
            say("\nPresets đã đo full (không phải accuracy/time/memory dự đoán)")
            for name, preset in report["presets"].items():
                say(f"  {name}: " + (f"{preset['config_id']} | Val={number(preset['validation_mean'])} | T={duration(preset['time_mean_seconds'])} | M={number(preset['memory_max_mib'], 1)} MiB" if preset else "chưa có cấu hình đủ điều kiện"))
        say("\nTrain lại cấu hình đã chọn (tách khỏi search; không chứng nhận tối ưu lại):")
        say("  " + command(["train-v2", "--selection", str(Path(folder).resolve() / "selected.json")]))


def status_view(path):
    import psutil
    folder = Path(path).resolve()
    saved = read_json(folder / "status.json")
    protocol = read_json(folder / "protocol.json", optional=True)
    if protocol and protocol.get("adapter") != "research_v2":
        raise ValueError("Not a v2 run; use research-show or show-training for historical runs")
    title("TRẠNG THÁI LƯỢT CHẠY", "Chỉ đọc trạng thái; không training, resume hay đánh giá test.")
    pairs([("Thư mục", folder), ("Trạng thái đã lưu", PHASES.get(saved.get("phase"), saved.get("phase", "chưa rõ")))])
    if not protocol:
        say("  Chưa có protocol khóa. Xem events.jsonl; chưa có chứng nhận training.")
        return
    pairs([("Tác vụ / thiết bị", f"{protocol['task']} / {protocol['hardware'].get('gpu') or protocol['hardware']['device']}"),
           ("Kế hoạch", f"{len(protocol['configs'])} cấu hình × {len(protocol['training_seeds'])} seeds × {protocol['full_epochs']} epoch"),
           ("Budget ban đầu", duration(protocol["budget_seconds"]))])
    changed = source_changes(protocol)
    if changed:
        say("  Mã hiện tại khác protocol cũ: " + ", ".join(changed[:3]))
        say("  Vẫn xem/audit được chứng cứ cũ. Resume/test cần source tương thích; không ép bỏ qua hash.")
    live, unknown = [], False
    for file in (folder / "trials").glob("*/process.json"):
        record = read_json(file, optional=True) or {}
        if record.get("status") != "running":
            continue
        try:
            process = psutil.Process(record["pid"])
            if record.get("create_time") is not None and abs(process.create_time() - record["create_time"]) < 0.01 and process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                progress = read_json(file.parent / "result.json", optional=True) or {}
                live.append(record["pid"])
                say(f"  Worker đang sống: PID {record['pid']} | epoch {progress.get('epochs_completed', 0)}")
                say(f"  Log: {file.parent / 'worker.log'}")
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            unknown = True
    say("  Tiến trình: " + (f"xác nhận {len(live)} worker đang sống" if live else "chưa xác định (thiếu quyền đọc PID)" if unknown else "không tìm thấy worker đang sống; không suy ra từ trạng thái lưu"))
    if protocol.get("mode") == "deployment":
        say("\n  Đây là lượt TRAIN LẠI, không phải tìm kiếm cấu hình mới.")
        pairs([("Thực thi xác minh", saved.get("execution_verified", "chưa xác nhận")),
               ("Chứng nhận tối ưu mới", "Không; train lại không xác lập tối ưu")])
        say(f"  Báo cáo training: {folder.parent.parent / 'TRAINING.json'}")
        return
    report = read_json(folder / "report.json", optional=True)
    if report and report.get("adapter") == "research_v2":
        say("\n  Đọc báo cáo đã lưu; lệnh status chưa audit tính toàn vẹn chứng cứ.")
        summary_view(folder, protocol, report)
        if saved.get("phase") != "complete":
            say("  Lưu ý: báo cáo trên là lựa chọn đã khóa trước đó; lần thực thi hiện tại chưa hoàn tất.")
    else:
        state = read_json(folder / "state.json", optional=True) or {}
        from .schema import complete
        count = sum(complete(r, protocol["full_epochs"]) for runs in state.get("results", {}).values() for r in runs)
        say(f"  Full lượt đã lưu: {count}; các probe không tính là full training.")
        say("  Ctrl+C trong cửa sổ đang chạy để dừng an toàn; không xóa thư mục/checkpoint.")
    history = read_json(folder / "test_history.json", optional=True) or []
    if history:
        last = history[-1]
        evaluation = read_json(Path(last["output"]) / "evaluation.json", optional=True) or {}
        say("\nTest sau lựa chọn (không dùng chọn cấu hình)")
        seeds_str = str(evaluation.get("evaluated_seeds") or last.get("seed"))
        pairs([("Trạng thái / seeds", f"{last.get('status')} / {seeds_str}"), ("Số lần mở test", len(history))])
        summary = evaluation.get("multi_seed_summary") or {}
        if summary:
            for metric in ("map50", "map75", "map50_95", "precision", "recall", "f1", "accuracy", "bleu4", "chrf"):
                if metric in summary and summary[metric].get("mean") is not None:
                    m = summary[metric]
                    sd_part = f" ± SD {number(m['sd'])}" if m.get("sd") is not None and m.get("n", 0) > 1 else ""
                    pairs([(f"Test {metric} (mean)", number(m["mean"]) + sd_part)])
        else:
            for metric in ("map50", "map75", "map50_95", "precision", "recall", "f1", "accuracy", "bleu4", "chrf"):
                if metric in evaluation:
                    pairs([(metric, number(evaluation[metric]))])
        lat = evaluation.get("latency") or {}
        if isinstance(lat, dict) and lat.get("available") and isinstance(lat.get("total_ms_per_image"), dict):
            ms = lat["total_ms_per_image"].get("mean")
            fps = (lat.get("fps") or {}).get("mean")
            pairs([("Inference Latency (rect=False)", f"{number(ms, 2)} ms/ảnh ({number(fps, 1)} FPS | {lat.get('precision', 'fp32')} | batch={lat.get('batch_size')})")])
        say(f"  Báo cáo: {Path(last['output']) / 'EVALUATION.md'}")
        say("  Kết quả test độc lập quá trình chọn cấu hình; lịch sử mở test đã được khóa.")
    else:
        say("\n  Test: chưa có lịch sử đánh giá sau lựa chọn; không tự động mở test.")
    say("\nKiểm tra chứng cứ gốc:")
    say("  " + command(["audit", "--run", str(folder)]))


def inventory(root="runs", limit=20):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Run directory not found: {root}")
    entries = []
    # Only read top-level artifacts; never inspect images/checkpoints or alter runs.
    files = set(root.rglob("protocol.json")) | set(root.rglob("training_report.json"))
    for file in files:
        if any(p in {"trials", "sources", "source_snapshot", "evaluation"} for p in file.relative_to(root).parts[:-1]):
            continue
        data = read_json(file, optional=True)
        if not isinstance(data, dict):
            continue
        adapter = data.get("adapter")
        if file.name == "protocol.json" and adapter not in {"research_v2", "yolo_research"}:
            continue
        saved = read_json(file.parent / "status.json", optional=True) or {}
        kind = "v2 training" if adapter == "research_v2" and data.get("mode") == "deployment" else "v2" if adapter == "research_v2" else "research" if adapter == "yolo_research" else "training cũ"
        entries.append({"path": str(file.parent), "kind": kind, "task": data.get("task") or data.get("dataset", {}).get("task", "?"),
                        "phase": saved.get("phase", "đọc báo cáo"), "mtime": file.stat().st_mtime})
    entries.sort(key=lambda r: (r["mtime"], r["path"]), reverse=True)
    return entries[:limit]


def source_changes(protocol):
    root = Path(__file__).resolve().parents[2]
    changed = []
    for name, expected in protocol.get("source_files", {}).items():
        path = root / name
        # A view must not follow arbitrary paths from an edited protocol.
        if not path.resolve().is_relative_to(root):
            changed.append(name)
            continue
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            actual = None
        if actual != expected:
            changed.append(name)
    return changed


def runs_view(entries):
    title("CÁC LƯỢT CHẠY ĐÃ LƯU", "Mới nhất trước. Đọc danh sách không chạy training.")
    if not entries:
        say("  Chưa tìm thấy protocol hoặc báo cáo training.")
    for i, row in enumerate(entries, 1):
        phase_text = PHASES.get(row["phase"], row["phase"])
        _say_styled(f"  [{i}] {style_text(row['task'], 'bold')} | {style_text(row['kind'], 'info')} | {phase_text}")
        say("      " + row["path"])


def error_view(exc):
    message = str(exc)
    title("KHÔNG THỂ HOÀN THÀNH LỆNH")
    _say_styled("  " + badge("ERROR", "danger", stream=sys.stderr) + " Lý do: " + clean(message), file=sys.stderr)
    lower = message.lower()
    if getattr(exc, "hint", None):
        hint = exc.hint
    elif "no local image files" in lower:
        hint = "Không có ảnh local. CSV hiện chỉ hỗ trợ tham chiếu ảnh/caption, không phải bảng đặc trưng. Chọn dataset ảnh có nhãn; dữ liệu bảng cần adapter riêng."
    elif "source changed" in lower or "new protocol required" in lower:
        hint = "Mã đã nâng cấp so với protocol cũ. Bạn vẫn xem/audit được lượt cũ; hãy tạo lượt optimize mới. Không sửa hash để ép resume."
    elif isinstance(exc, FileNotFoundError) or "not found" in lower or "does not exist" in lower:
        hint = "Kiểm tra đường dẫn có tồn tại. Với đường dẫn có khoảng trắng, đặt trong dấu nháy. Dùng menu để nhập đường dẫn."
    elif "validation" in lower or "labels" in lower or "unsupported task" in lower:
        hint = "Cần nhãn thật và tập validation độc lập. Dùng inspect --data để kiểm tra; v2 hỗ trợ detection, classification, captioning."
    elif "active" in lower or "locked" in lower and "process" in lower:
        hint = "Một job khác đang chạy. Chờ job đó xong rồi thử lại; chương trình không tự dừng job của bạn."
    elif "checksum" in lower or "differs" in lower or "corrupt" in lower or "ledger" in lower:
        hint = "Chứng cứ không khớp hoặc bị hỏng. Giữ nguyên files/log; chạy audit --run để chẩn đoán. Không bỏ qua kiểm tra tính toàn vẹn."
    elif isinstance(exc, ImportError) or "runtime" in lower or "precision" in lower:
        hint = "Chạy doctor để kiểm tra môi trường. Có thể chọn --device cpu cho lượt mới; CPU dùng FP32."
    elif "budget" in lower or isinstance(exc, TimeoutError):
        hint = "Ngân sách không đủ. Xem status; resume cần ngân sách bổ sung và protocol còn tương thích. Không coi timeout là training thành công."
    elif "test already" in lower:
        hint = "Test đã được mở. Xem kết quả đã lưu; chỉ dùng --allow-retest nếu chủ động chấp nhận mở test lại."
    else:
        hint = "Dùng --help để xem tham số, doctor để kiểm tra máy, hoặc status/audit --run cho lượt đã lưu."
    say("  Cách xử lý: " + hint, file=sys.stderr)


def _read_gpu_temp(device):
    if device is None:
        return None
    dev = str(device).strip().lower()
    if not dev or dev in {"cpu", "mps"}:
        return None
    try:
        import pynvml
        idx = 0
        if dev.startswith("cuda:"):
            idx = int(dev.split(":", 1)[1])
        elif dev.isdigit():
            idx = int(dev)
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        return float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    except Exception:
        return None


def _progress_bar(current, total, width_chars=10):
    total_val = max(1, int(total or 1))
    ratio = min(1.0, max(0.0, float(current or 0) / total_val))
    filled = int(round(ratio * width_chars))
    if _supports_utf8():
        return "[" + "█" * filled + "░" * (width_chars - filled) + "]"
    return "[" + "#" * filled + "-" * (width_chars - filled) + "]"


class TrialProgress:
    """Throttled append-only epoch updates, also readable in redirected logs."""

    def __init__(self, folder, index, seed, epochs, full, config, seconds, *, budget_remaining_fn=None, device=None, detection_metric=None):
        self.folder, self.index, self.seed, self.epochs = Path(folder), int(index), seed, int(epochs)
        self.full, self.seconds = bool(full), float(seconds)
        self.budget_remaining_fn = budget_remaining_fn
        self.device = device
        self.detection_metric = detection_metric
        self.started, self.last, self.epoch = time.monotonic(), -20.0, -1
        self._last_temp_at, self._cached_temp = -60.0, None
        self.mode_tag = "FULL" if full else "PROBE"
        self.label = f"c{self.index:02d} / seed {seed} / {self.mode_tag}"
        say(f"\n  Bắt đầu {self.label}: {Path(config['model']).name}, ảnh {config['imgsz']}, batch {config['batch']}, {config['precision']}")
        say(f"  {epochs} epoch | trần thời gian lượt này: {duration(seconds)} | Ctrl+C: dừng an toàn")
        say(f"  Log: {self.folder / 'worker.log'}")

    def update(self):
        now = time.monotonic() - self.started
        result = read_json(self.folder / "result.json", optional=True) or {}
        epoch = result.get("epochs_completed", 0)
        if epoch != self.epoch or now - self.last >= 20:
            self.epoch, self.last = epoch, now
            bar = _progress_bar(epoch, self.epochs)
            if callable(self.budget_remaining_fn):
                try:
                    rem_seconds = max(0.0, float(self.budget_remaining_fn()))
                except Exception:
                    rem_seconds = max(0.0, self.seconds - now)
            else:
                rem_seconds = max(0.0, self.seconds - now)
            metric_val = result.get("quality_score")
            if metric_val is None:
                metric_val = result.get("quality")
            if metric_val is None:
                metric_val = result.get("best_quality")
            primary = self.detection_metric or result.get("primary_metric")
            val_label = "Val mAP50" if primary == "map50" else "Val mAP50-95" if primary == "map50_95" else "Val"
            val_str = number(metric_val) if metric_val is not None else "chưa đo"
            mem_val = result.get("memory_mb")
            if mem_val is None:
                mem_val = result.get("peak_reserved_mb")
            if mem_val is None:
                mem_val = result.get("peak_rss_mb")
            mem_str = f"{number(mem_val, 1)} MiB" if mem_val is not None else "N/A"
            if now - self._last_temp_at >= 5.0 or self._cached_temp is None:
                self._cached_temp = _read_gpu_temp(self.device)
                self._last_temp_at = now
            temp_str = f"{self._cached_temp:.0f}°C" if self._cached_temp is not None else "N/A"
            mode_styled = style_text(self.mode_tag, "info" if self.full else "warn")
            line = (
                f"    [Config c{self.index:02d} | Seed {self.seed} | {mode_styled} | "
                f"epoch {epoch}/{self.epochs} | {bar} | Budget {duration(rem_seconds)} | "
                f"{val_label} {val_str} | VRAM {mem_str} | GPU Temp {temp_str}]"
            )
            _say_styled(line)


def wait_worker(process, seconds, progress=None):
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, seconds)
        try:
            process.wait(timeout=min(1.0, remaining))
            if progress:
                progress.update()
            return
        except subprocess.TimeoutExpired:
            if progress:
                progress.update()

