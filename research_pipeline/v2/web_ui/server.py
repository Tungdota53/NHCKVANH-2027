"""Offline loopback ThreadingHTTPServer and Jinja2/JSON router for Tier 2 Web UI."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import threading
from urllib.parse import parse_qs, urlsplit
import webbrowser

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .. import __version__
from .data_access import (
    hardware_snapshot,
    launch_job_action,
    list_workspace_resources,
    load_benchmark_detail,
    load_evaluation,
    load_live_status,
    load_run_detail,
    run_audit_action,
    run_inspect_action,
    run_paper_export_action,
    scan_runs,
    stop_job_action,
)
from .security import (
    CSRFManager,
    is_raw_traversal,
    resolve_contained_path,
    validate_loopback_host,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_POST_BYTES = 2 * 1024 * 1024

_UI_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _UI_DIR / "templates"
_STATIC_DIR = _UI_DIR / "static"

_STATIC_FILES = {
    "style.css": "text/css; charset=utf-8",
    "charts.js": "application/javascript; charset=utf-8",
    "app.js": "application/javascript; charset=utf-8",
}

_PAGE_ROUTES = {
    "/": ("overview.html", "overview", "Tổng quan & Hệ thống"),
    "/monitor": ("monitor.html", "monitor", "Theo dõi trực tiếp"),
    "/launcher": ("launcher.html", "launcher", "Tạo & Chạy thực nghiệm"),
    "/analysis": ("analysis.html", "analysis", "Phân tích & Biên Pareto"),
    "/benchmark": ("benchmark.html", "benchmark", "So sánh thuật toán"),
    "/export": ("export.html", "export", "Kiểm toán & Xuất báo cáo"),
}

_POST_ROUTES = frozenset({
    "/api/inspect",
    "/api/audit",
    "/api/paper-export",
    "/api/launch",
    "/api/stop",
})

_FILE_CONTENT_TYPES = {
    ".json": "application/json; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".tex": "application/x-tex; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".svg": "image/svg+xml; charset=utf-8",
    ".log": "text/plain; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".yaml": "text/plain; charset=utf-8",
    ".yml": "text/plain; charset=utf-8",
}


class _DualStackHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class _IPv6HTTPServer(_DualStackHTTPServer):
    address_family = socket.AF_INET6


def _build_jinja_env():
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )


def _json_bytes(payload) -> bytes:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _make_handler(runs_root: Path, csrf: CSRFManager, jinja_env: Environment):
    class DashboardHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            return

        def _send_bytes(self, status: int, body: bytes, content_type: str, extra_headers=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            if extra_headers:
                for k, v in extra_headers.items():
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: dict):
            self._send_bytes(status, _json_bytes(payload), "application/json; charset=utf-8")

        def _send_error_json(self, status: int, message: str):
            self._send_json(status, {"error": str(message), "status": status})

        def _check_raw_path(self) -> bool:
            if is_raw_traversal(self.path):
                self._send_error_json(403, "Path traversal is forbidden")
                return False
            return True

        def _read_json_body(self):
            raw_len = self.headers.get("Content-Length")
            try:
                length = int(raw_len or "0")
            except ValueError:
                raise ValueError("Invalid Content-Length") from None
            if length < 0 or length > MAX_POST_BYTES:
                raise OverflowError("Request body exceeds maximum allowed size")
            raw = self.rfile.read(length) if length > 0 else b"{}"
            if not raw.strip():
                return {}
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object")
            return data

        def do_GET(self):
            if not self._check_raw_path():
                return
            parsed = urlsplit(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query, keep_blank_values=False)

            if path in _POST_ROUTES:
                self._send_bytes(
                    405,
                    _json_bytes({"error": "Method not allowed; use POST", "status": 405}),
                    "application/json; charset=utf-8",
                    {"Allow": "POST"},
                )
                return

            if path == "/favicon.ico":
                self.send_response(204)
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return

            if path in _PAGE_ROUTES:
                tpl_name, active_page, page_title = _PAGE_ROUTES[path]
                tpl = jinja_env.get_template(tpl_name)
                html = tpl.render(
                    csrf_token=csrf.token,
                    active_page=active_page,
                    page_title=page_title,
                    runs_dir=str(runs_root),
                    version=__version__,
                )
                self._send_bytes(200, html.encode("utf-8"), "text/html; charset=utf-8")
                return

            if path.startswith("/static/"):
                rel_name = path[len("/static/"):]
                ctype = _STATIC_FILES.get(rel_name)
                if not ctype:
                    self._send_error_json(404, "Static asset not found")
                    return
                asset_path = (_STATIC_DIR / rel_name).resolve()
                if not asset_path.is_relative_to(_STATIC_DIR.resolve()) or not asset_path.is_file():
                    self._send_error_json(404, "Static asset not found")
                    return
                self._send_bytes(200, asset_path.read_bytes(), ctype)
                return

            try:
                if path == "/api/hardware":
                    self._send_json(200, hardware_snapshot())
                    return
                if path == "/api/runs":
                    self._send_json(200, scan_runs(runs_root))
                    return
                if path == "/api/resources":
                    self._send_json(200, list_workspace_resources())
                    return
                if path == "/api/run" or path.startswith("/api/runs/"):
                    target = (
                        path[len("/api/runs/"):]
                        if path.startswith("/api/runs/")
                        else (qs.get("path") or qs.get("run") or [""])[0]
                    )
                    if not target:
                        self._send_error_json(400, "Missing path or run query parameter")
                        return
                    self._send_json(200, load_run_detail(runs_root, target))
                    return
                if path == "/api/evaluation":
                    target = (qs.get("path") or qs.get("run") or [""])[0]
                    if not target:
                        self._send_error_json(400, "Missing path or run query parameter")
                        return
                    self._send_json(200, load_evaluation(runs_root, target))
                    return
                if path == "/api/live":
                    target = (qs.get("run") or qs.get("path") or [None])[0]
                    self._send_json(200, load_live_status(runs_root, target))
                    return
                if path == "/api/benchmark":
                    target = (qs.get("path") or qs.get("run") or [""])[0]
                    if not target:
                        self._send_error_json(400, "Missing path or run query parameter")
                        return
                    self._send_json(200, load_benchmark_detail(runs_root, target))
                    return
                if path in {"/api/file", "/api/artifact", "/api/download"}:
                    run_q = (qs.get("run") or [None])[0]
                    file_q = (qs.get("file") or qs.get("path") or [None])[0]
                    if run_q and file_q:
                        run_dir = resolve_contained_path(runs_root, run_q, allow_absolute_inside=False, must_exist=True)
                        target_file = resolve_contained_path(run_dir, file_q, allow_absolute_inside=False, must_exist=True)
                        # Also re-check against runs_root
                        if not target_file.is_relative_to(runs_root.resolve()):
                            raise PermissionError("File escapes runs root")
                    elif file_q:
                        target_file = resolve_contained_path(runs_root, file_q, allow_absolute_inside=False, must_exist=True)
                    else:
                        self._send_error_json(400, "Missing path or file query parameter")
                        return
                    if not target_file.is_file():
                        self._send_error_json(403, "Target is not a regular file")
                        return
                    ext = target_file.suffix.lower()
                    ctype = _FILE_CONTENT_TYPES.get(ext)
                    if not ctype:
                        self._send_error_json(403, f"File type not permitted for preview/download: {ext}")
                        return
                    extra = {}
                    if (qs.get("download") or ["0"])[0] == "1":
                        extra["Content-Disposition"] = f'attachment; filename="{target_file.name}"'
                    self._send_bytes(200, target_file.read_bytes(), ctype, extra)
                    return
                self._send_error_json(404, f"Not found: {path}")
            except PermissionError as exc:
                self._send_error_json(403, str(exc))
            except FileNotFoundError as exc:
                self._send_error_json(404, str(exc))
            except (ValueError, KeyError, TypeError) as exc:
                self._send_error_json(400, str(exc))

        def do_POST(self):
            if not self._check_raw_path():
                return
            path = urlsplit(self.path).path
            if path not in _POST_ROUTES:
                self._send_error_json(404, f"Unknown POST endpoint: {path}")
                return
            try:
                body = self._read_json_body()
            except OverflowError as exc:
                self._send_error_json(413, str(exc))
                return
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_error_json(400, str(exc))
                return

            header_token = self.headers.get("X-CSRF-Token")
            token = header_token if header_token is not None else body.get("csrf_token")
            if not csrf.verify(token):
                self._send_error_json(403, "Missing or invalid X-CSRF-Token")
                return

            try:
                if path == "/api/inspect":
                    res = run_inspect_action(
                        body.get("data", ""),
                        task=body.get("task", "auto"),
                        device=body.get("device", "cpu"),
                    )
                    self._send_json(200, res)
                    return
                if path == "/api/audit":
                    res = run_audit_action(runs_root, body.get("run", ""))
                    self._send_json(200, res)
                    return
                if path == "/api/paper-export":
                    res = run_paper_export_action(
                        runs_root,
                        body.get("run", ""),
                        body.get("output", ""),
                        lang=body.get("lang", "vi"),
                    )
                    self._send_json(200, res)
                    return
                if path == "/api/launch":
                    res = launch_job_action(runs_root, body)
                    self._send_json(200, res)
                    return
                if path == "/api/stop":
                    res = stop_job_action(runs_root, body)
                    self._send_json(200, res)
                    return
            except PermissionError as exc:
                self._send_error_json(403, str(exc))
            except FileNotFoundError as exc:
                self._send_error_json(404, str(exc))
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self._send_error_json(400, str(exc))

        def _reject_method(self):
            path = urlsplit(self.path).path
            if path in _POST_ROUTES:
                self._send_bytes(
                    405,
                    _json_bytes({"error": "Method not allowed; use POST", "status": 405}),
                    "application/json; charset=utf-8",
                    {"Allow": "POST"},
                )
            else:
                self._send_error_json(405, "Method not allowed")

        do_PUT = _reject_method
        do_DELETE = _reject_method
        do_PATCH = _reject_method

    return DashboardHandler


def create_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, runs_dir: str | Path = "runs"):
    safe_host = validate_loopback_host(host)
    port_num = int(port)
    if not 0 <= port_num <= 65535:
        raise ValueError(f"Invalid port: {port}")
    runs_root = Path(runs_dir).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    csrf = CSRFManager()
    jinja_env = _build_jinja_env()
    handler = _make_handler(runs_root, csrf, jinja_env)
    server_cls = _IPv6HTTPServer if safe_host == "::1" else _DualStackHTTPServer
    server = server_cls((safe_host, port_num), handler)
    actual_port = server.server_address[1]
    display_host = "[::1]" if safe_host == "::1" else "127.0.0.1"
    url = f"http://{display_host}:{actual_port}"
    return server, url, csrf


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, runs_dir: str | Path = "runs", open_browser: bool = False):
    server, url, _ = create_server(host=host, port=port, runs_dir=runs_dir)
    print(f"Web UI running at {url} (100% offline loopback; runs_dir={Path(runs_dir).resolve()})", flush=True)
    if open_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


serve_ui = serve
