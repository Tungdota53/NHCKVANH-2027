"""Read-only input diagnostics. Recognizing a table does not enable training it."""

import csv
from pathlib import Path


class DatasetInputError(ValueError):
    def __init__(self, message, hint, *, kind, details=None):
        super().__init__(message)
        self.hint, self.kind, self.details = hint, kind, details or {}


IMAGE_COLUMNS = {"image", "image_path", "file_name", "filename"}
ARCHIVES = {".zip", ".7z", ".rar", ".tar", ".gz", ".tgz"}


def csv_header(path):
    """Bounded header only: no customer records, inferred target or fake labels."""
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as file:
            header = file.readline(65537)
        if len(header) > 65536 or not header.strip():
            return None
        dialect = csv.Sniffer().sniff(header, delimiters="\t,;|")
        columns = next(csv.reader([header], dialect))
        if len(columns) < 2:
            return None
        return {"file": str(Path(path).resolve()), "columns": [c.strip() for c in columns],
                "column_count": len(columns), "delimiter": dialect.delimiter}
    except (OSError, UnicodeError, csv.Error, StopIteration):
        return None


def table_error(signature):
    name = Path(signature["file"]).name
    return DatasetInputError(
        f"Đã phát hiện dữ liệu dạng bảng: {name} ({signature['column_count']} cột), không phải CSV ảnh/caption.",
        "Pipeline hiện chỉ train dữ liệu ảnh: detection, classification ảnh và captioning. "
        "CSV được hỗ trợ là file có cột image/image_path/file_name/filename và caption/text, "
        "trỏ tới ảnh thật. Bảng đặc trưng cần nhánh tabular và xác định task/cột mục tiêu; "
        "chương trình không tự đoán cột cần dự đoán. Chọn dataset ảnh khác hoặc yêu cầu thêm tabular.",
        kind="tabular_not_supported", details=signature)


def check_explicit_file(source):
    """Prevent an explicit table/archive being reinterpreted as nearby images."""
    source = Path(source)
    if source.is_file() and source.suffix.lower() in ARCHIVES:
        raise DatasetInputError(f"Bạn chọn file nén: {source.name}; chưa phải dataset đã giải nén.",
                                "Giải nén vào thư mục riêng rồi chọn thư mục/manifest chứa ảnh và nhãn. Không tự giải nén hoặc tải dữ liệu.",
                                kind="compressed_input")
    if source.is_file() and source.suffix.lower() in {".csv", ".tsv"}:
        signature = csv_header(source)
        if signature and not IMAGE_COLUMNS.intersection(c.casefold() for c in signature["columns"]):
            raise table_error(signature)


def no_images_error(source):
    source = Path(source)
    files = [source] if source.is_file() else []
    if source.is_dir():
        from yolo_tuner.adapters import walk
        # Diagnose a few relevant files, not all customer rows or every archive.
        for path in walk(source):
            if path.suffix.lower() in {".csv", ".tsv", *ARCHIVES}:
                files.append(path)
                if len(files) >= 5:
                    break
    tables, captions = [], []
    for file in files:
        if file.suffix.lower() not in {".csv", ".tsv"}:
            continue
        signature = csv_header(file)
        if signature:
            (captions if IMAGE_COLUMNS.intersection(c.casefold() for c in signature["columns"]) else tables).append(signature)
    if captions:
        return DatasetInputError("Có CSV tham chiếu ảnh nhưng adapter chưa tìm thấy ảnh theo bố cục đang dùng.",
                                 "CSV chỉ chứa đường dẫn/nhãn, không chứa pixel ảnh. Cần có ảnh được tham chiếu trên máy, "
                                 "đường dẫn đúng theo thư mục CSV; CSV caption dùng dấu phẩy và cột caption/text. "
                                 "Với ảnh nằm ngoài bố cục thư mục hoặc annotation khác, dùng manifest JSON phù hợp.",
                                 kind="missing_referenced_images", details=captions[0])
    if tables:
        return table_error(tables[0])
    if any(p.suffix.lower() in ARCHIVES for p in files):
        return DatasetInputError("Thư mục có file nén nhưng chưa tìm thấy ảnh local.",
                                 "Giải nén dataset rồi chọn đúng thư mục ảnh/nhãn hoặc manifest. Chương trình chưa tự giải nén.",
                                 kind="compressed_input")
    return DatasetInputError(f"Không tìm thấy ảnh local tại: {source}",
                             "Chọn thư mục có ảnh thật và nhãn hoặc manifest trỏ tới ảnh trên máy. "
                             "CSV thông thường không tự trở thành dataset ảnh; dữ liệu bảng/audio/text cần adapter riêng.",
                             kind="no_local_images")
