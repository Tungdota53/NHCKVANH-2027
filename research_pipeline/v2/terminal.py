"""Human-readable terminal views; never change evidence or select a configuration."""

import json
import hashlib
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import unicodedata

from . import __version__

PHASES = {"preparing": "Đang chuẩn bị", "prepared": "Đã lập kế hoạch, chưa chạy",
          "search": "Đang tìm kiếm (trạng thái lưu)", "complete": "Đã kết thúc",
          "interrupted": "Bị gián đoạn", "selection_committed_report_error": "Đã khóa lựa chọn, lỗi xuất báo cáo"}
VERDICTS = {
    "empirical_optimal_in_declared_space": "TỐI ƯU THỰC NGHIỆM TRONG TẬP ĐÃ ĐO ĐỦ",
    "fully_trained_best_observed": "TỐT NHẤT ĐÃ ĐO — CHƯA ĐO ĐỦ TẬP CẤU HÌNH",
    "no_verified_feasible_selection": "CHƯA CÓ CẤU HÌNH ĐỦ ĐIỀU KIỆN",
    "contended_smoke_not_certified": "KHÔNG CHỨNG NHẬN: MÁY BỊ TRANH TÀI NGUYÊN",
}


def interactive():
    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def clean(value):
    # File names and worker errors are data, not terminal control instructions.
    return "".join(c if c >= " " and c != "\x7f" else " " for c in str(value))


def width(value):
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(value))


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


def say(value="", *, file=None):
    stream = file or sys.stdout
    value = "".join(c for c in str(value) if c in "\n\t" or c >= " " and c != "\x7f")
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        value.encode(encoding)
    except (UnicodeError, LookupError):
        decomposed = unicodedata.normalize("NFKD", value.replace("đ", "d").replace("Đ", "D"))
        value = "".join(c for c in decomposed if not unicodedata.combining(c)).encode("ascii", "replace").decode()
    print(value, file=stream, flush=True)


def title(label, note=None):
    columns = max(20, min(88, shutil.get_terminal_size((88, 24)).columns))
    say("\n" + "=" * columns)
    say(clip(label, columns))
    if note:
        say(note)
    say("=" * columns)


def pairs(rows):
    size = min(24, max((width(k) for k, _ in rows), default=0))
    for key, value in rows:
        label = clean(key)
        say(f"  {label}{' ' * max(0, size - width(label))} : {clean(value)}")


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


def hardware_view(hw):
    pairs([("Thiết bị", hw.get("gpu") or hw.get("device", "chưa rõ")),
           ("ML runtime", "Sẵn sàng" if hw.get("ml_runtime_ready") else "CHƯA SẴN SÀNG"),
           ("RAM còn trống", number(hw.get("ram_available_mb"), 0) + " MiB"),
           ("VRAM còn trống", number(hw.get("vram_free_mb"), 0) + " MiB" if hw.get("vram_free_mb") is not None else "Không áp dụng / chưa đọc được"),
           ("Precision hỗ trợ", "FP32" + (", FP16" if hw.get("fp16") else "") + (", BF16" if hw.get("bf16") else ""))])
    if hw.get("reason"):
        say("  Lý do: " + clean(hw["reason"]))


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
        for warning in data.get("warnings") or []:
            say("  Lưu ý: " + clean(warning))


def summary_view(folder, protocol, report):
    verdict = report["decision"]["verification_status"]
    title("KẾT QUẢ TÌM CẤU HÌNH", VERDICTS.get(verdict, verdict))
    rows = report.get("candidates", [])
    complete = sum(len(r.get("completed_seeds", [])) for r in rows)
    expected = len(protocol["configs"]) * len(protocol["training_seeds"])
    say(f"  Full training hoàn tất: {complete}/{expected} lượt config × seed; {protocol['full_epochs']} epoch/lượt.")
    say("  Chỉ tối ưu theo objective, tập cấu hình, dữ liệu và máy đã khóa; không bảo đảm mọi GPU/dataset.")
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
        metric = {"detection": "Val mAP50-95", "classification": "Val accuracy", "captioning": "Val (BLEU4+chrF)/200"}.get(protocol["task"], "Validation")
        sd = f" ± SD {number(r.get('validation_sd_not_ci'))}" if len(r["completed_seeds"]) > 1 else " (1 seed; chưa ước lượng biến thiên)"
        memory_label = "VRAM reserved đo được" if r["memory_kind"] == "cuda_reserved_allocator_mib" else "RAM RSS cây tiến trình"
        pairs([(metric, number(r["validation_mean"]) + sd), ("Thời gian / full lượt", duration(r["full_worker_seconds_mean"])),
               (memory_label, number(r["memory_max_mb"], 1) + " MiB"), ("Điểm cân bằng U", number(r["utility"], 6))])
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
        pairs([("Trạng thái / seed", f"{last.get('status')} / {last.get('seed')}"), ("Số lần mở test", len(history))])
        for metric in ("map50_95", "map50", "accuracy", "bleu4", "chrf"):
            if metric in evaluation:
                pairs([(metric, number(evaluation[metric]))])
        say(f"  Báo cáo: {Path(last['output']) / 'EVALUATION.md'}")
        say("  Đây không phải test trung bình nhiều seed hoặc bằng chứng dataset hoàn toàn chưa từng được xem.")
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
        say(f"  [{i}] {row['task']} | {row['kind']} | {PHASES.get(row['phase'], row['phase'])}")
        say("      " + row["path"])


def error_view(exc):
    message = str(exc)
    title("KHÔNG THỂ HOÀN THÀNH LỆNH")
    say("  Lý do: " + clean(message), file=sys.stderr)
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


class TrialProgress:
    """Throttled append-only epoch updates, also readable in redirected logs."""

    def __init__(self, folder, index, seed, epochs, full, config, seconds):
        self.folder, self.epochs = Path(folder), epochs
        self.started, self.last, self.epoch = time.monotonic(), -20.0, -1
        self.label = f"c{index:02d} / seed {seed} / {'FULL' if full else 'PROBE'}"
        say(f"\n  Bắt đầu {self.label}: {Path(config['model']).name}, ảnh {config['imgsz']}, batch {config['batch']}, {config['precision']}")
        say(f"  {epochs} epoch | trần thời gian lượt này: {duration(seconds)} | Ctrl+C: dừng an toàn")
        say(f"  Log: {self.folder / 'worker.log'}")

    def update(self):
        now = time.monotonic() - self.started
        result = read_json(self.folder / "result.json", optional=True) or {}
        epoch = result.get("epochs_completed", 0)
        if epoch != self.epoch or now - self.last >= 20:
            self.epoch, self.last = epoch, now
            metric = result.get("quality")
            detail = " | Q=" + number(metric) if metric is not None else " | đang khởi tạo / xử lý batch" if not epoch else ""
            say(f"    {self.label} | epoch {epoch}/{self.epochs} | đã chạy {duration(now)}{detail}")


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

