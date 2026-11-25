"""Phase 4 Two-Tier UI unit and integration tests (Rich Terminal, Wizard, Offline Web UI)."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from research_pipeline.v2 import cli, predictors, terminal as ui, wizard
from research_pipeline.v2.web_ui import create_server, data_access, security
from research_pipeline.v2.workflow import source_map


REPO_ROOT = Path(__file__).resolve().parents[1]
_ANSI_ESC_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_EXT_URL_RE = re.compile(r"https?://", re.IGNORECASE)


def _sha256_tree(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _find_sample_v2_run() -> Path | None:
    cand = REPO_ROOT / "runs" / "nckh_011_map50_verification"
    if cand.is_dir():
        prots = sorted(cand.rglob("protocol.json"))
        if prots:
            return prots[0].parent
    return None


def _make_synthetic_run(runs_dir: Path, name: str = "demo_run", *, with_eval: bool = True, eval_scalar_latency: bool = True) -> Path:
    run_dir = runs_dir / name
    (run_dir / "trials" / "attempt_00001").mkdir(parents=True, exist_ok=True)
    (run_dir / "evaluation" / "20261124_000000").mkdir(parents=True, exist_ok=True)
    cfg0 = {
        "model": "yolo11n.pt",
        "imgsz": 64,
        "batch": 4,
        "accumulation": 1,
        "precision": "fp32",
        "optimizer": "AdamW",
        "lr0": 0.001,
        "weight_decay": 0.0005,
        "freeze": 0,
    }
    cfg1 = {**cfg0, "imgsz": 128, "lr0": 0.002}
    prot = {
        "schema_version": 4,
        "adapter": "research_v2",
        "mode": "online",
        "protocol_sha256": "a" * 64,
        "method": "adaptive_full",
        "task": "detection",
        "detection_metric": "map50",
        "selection_rule": "weighted",
        "quality_definition": "mAP50",
        "memory_kind": "process_tree_rss_mib_not_vram",
        "dataset_source": "synthetic",
        "dataset_content_id": "ds123",
        "dataset_sha256": "b" * 64,
        "subset_sha256": "c" * 64,
        "full_epochs": 2,
        "probe_epochs": 1,
        "training_seeds": [17],
        "budget_seconds": 120.0,
        "configs": [cfg0, cfg1],
        "config_ids": ["cfg00001", "cfg00002"],
        "hardware": {"device": "cpu", "host_label": "test-host"},
        "objective": {
            "name": "balanced",
            "min_quality": 0.0,
            "max_time": 60.0,
            "max_memory": 2048.0,
            "time_scale": 60.0,
            "memory_scale": 1024.0,
            "weights": [0.6, 0.25, 0.15],
        },
        "source_files": {},
        "model_files": {},
        "data_files": {},
    }
    rep = {
        "schema_version": 4,
        "adapter": "research_v2",
        "protocol_sha256": "a" * 64,
        "method": "adaptive_full",
        "task": "detection",
        "detection_metric": "map50",
        "selection_rule": "weighted",
        "decision": {
            "selected_index": 0,
            "selected_config_id": "cfg00001",
            "verification_status": "empirical_optimal_in_declared_space",
            "pareto": ["cfg00001", "cfg00002"],
            "selection_score": 0.81234567890123,
        },
        "recommendation": {
            "config": cfg0,
            "validation_mean": 0.78123456789012,
            "validation_sd_not_ci": 0.0,
            "full_worker_seconds_mean": 5.25,
            "memory_max_mb": 320.5,
            "memory_kind": "process_tree_rss_mib_not_vram",
            "completed_seeds": [17],
            "full_epochs": 2,
            "utility": 0.81234567890123,
        },
        "candidates": [
            {
                "index": 0,
                "config_id": "cfg00001",
                "config": cfg0,
                "complete": True,
                "completed_seeds": [17],
                "required_seeds": [17],
                "state": "confirmed_feasible",
                "quality_mean": 0.78123456789012,
                "quality_sd": 0.0,
                "time_mean": 5.25,
                "memory_max": 320.5,
                "utility": 0.81234567890123,
                "runs": [{"seed": 17, "status": "ok", "mode": "train", "epochs_completed": 2}],
            },
            {
                "index": 1,
                "config_id": "cfg00002",
                "config": cfg1,
                "complete": False,
                "completed_seeds": [],
                "required_seeds": [17],
                "state": "unresolved",
                "quality_mean": None,
                "quality_sd": None,
                "time_mean": None,
                "memory_max": None,
                "utility": None,
                "runs": [{"seed": 17, "status": "oom", "mode": "train", "epochs_completed": 0}],
            },
        ],
        "online_session_seconds": 12.5,
        "budget_seconds": 120.0,
        "total_granted_budget_seconds": 120.0,
        "budget_overrun_seconds": 0.0,
        "all_attempt_worker_seconds": 5.25,
        "proxy_worker_seconds": 0.0,
        "preparation_seconds": 0.5,
        "acquisition_history": [
            {
                "step": 1,
                "index": 0,
                "reason": "initial",
                "remaining_seconds": 110.0,
                "surrogate_diagnostics": {
                    "kernel": "matern52",
                    "length_scale": 1.0,
                    "ridge": 0.05,
                    "loocv_mae": 0.012,
                    "loocv_rmse": 0.015,
                    "heteroscedastic": False,
                    "predictions": {"0": {"mean": 0.77, "std": 0.02}},
                },
            }
        ],
        "presets": {},
        "limitations": ["Synthetic test fixture"],
    }
    state = {
        "results": {
            "0": [
                {
                    "attempt_id": "att1",
                    "index": 0,
                    "seed": 17,
                    "mode": "train",
                    "status": "ok",
                    "epochs_completed": 2,
                    "quality": 0.78123456789012,
                    "worker_wall_seconds": 5.25,
                    "memory_mb": 320.5,
                    "learning_curve": [
                        {"epoch": 1, "map50": 0.65, "map50_95": 0.42},
                        {"epoch": 2, "map50": 0.78123456789012, "map50_95": 0.54},
                    ],
                }
            ],
            "1": [],
        },
        "probes": [],
        "acquisition_history": rep["acquisition_history"],
        "decision": rep["decision"],
    }
    (run_dir / "protocol.json").write_text(json.dumps(prot), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"phase": "complete", **rep["decision"]}), encoding="utf-8")
    (run_dir / "report.json").write_text(json.dumps(rep), encoding="utf-8")
    (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (run_dir / "trials" / "attempt_00001" / "worker.log").write_text("\x1b[32mEpoch 1/2 ok\x1b[0m\nEpoch 2/2 ok\n", encoding="utf-8")
    (run_dir / "trials" / "attempt_00001" / "result.json").write_text(
        json.dumps({"epochs_completed": 2, "quality": 0.78123456789012, "memory_mb": 320.5}),
        encoding="utf-8",
    )
    if with_eval:
        ev = {
            "map50": 0.76543210987654,
            "map75": 0.51234567890123,
            "map50_95": 0.53210987654321,
            "precision": 0.81,
            "recall": 0.74,
            "f1": 0.7734,
            "per_class": [
                {"class_id": 0, "class_name": "zebra", "ap50": 0.82111111111111, "ap50_95": 0.58222222222222},
                {"class_id": 1, "class_name": "rhino", "ap50": 0.70975310864197, "ap50_95": 0.48199753086420},
            ],
            "validation_vs_test": {
                "validation_mean": 0.78123456789012,
                "test_mean": 0.76543210987654,
                "generalization_gap_test_minus_val": -0.01580245801358,
            },
            "latency": {
                "available": True,
                "batch_size": 1,
                "precision": "fp32",
                "warmup_batches": 2,
                "rect": False,
                "total_ms_per_image": 14.1234567890123 if eval_scalar_latency else {"n": 2, "mean": 14.1234567890123, "sd": 0.2, "min": 13.9, "max": 14.3},
                "fps": 70.8041958041958 if eval_scalar_latency else {"n": 2, "mean": 70.8041958041958, "sd": 0.5, "min": 70.3, "max": 71.3},
                "mean_total_ms_per_image": 14.1234567890123,
                "mean_fps": 70.8041958041958,
            },
            "pr_curve": {
                "available": True,
                "recall_grid": [0.0, 0.5, 1.0],
                "mean_precision": [1.0, 0.85, 0.2],
                "per_class": [{"class_id": 0, "class_name": "zebra", "precision": [1.0, 0.9, 0.3]}],
            },
            "f1_curve": {
                "available": True,
                "confidence_grid": [0.0, 0.5, 1.0],
                "mean_f1": [0.4, 0.77, 0.0],
                "best_confidence": 0.5,
                "max_mean_f1": 0.77,
                "per_class": [],
            },
            "confusion_matrix": {
                "available": True,
                "labels": ["zebra", "rhino", "background"],
                "normalized": [[0.85, 0.05, 0.10], [0.10, 0.80, 0.10], [0.05, 0.15, 0.80]],
            },
        }
        ev_file = run_dir / "evaluation" / "20261124_000000" / "evaluation.json"
        ev_file.write_text(json.dumps(ev), encoding="utf-8")
        (run_dir / "evaluation" / "test_summary.json").write_text(json.dumps(ev), encoding="utf-8")
        (run_dir / "test_history.json").write_text(json.dumps([{"status": "ok", "output": str(ev_file.parent), "seed": 17}]), encoding="utf-8")
    return run_dir


class Phase4TrainingSignatureTests(unittest.TestCase):
    def test_pure_ui_modules_excluded_from_training_signature(self):
        sources = source_map()
        base = predictors.training_signature({"source_files": dict(sources)})
        for key in (
            "research_pipeline/v2/terminal.py",
            "research_pipeline/v2/wizard.py",
            "research_pipeline/v2/cli.py",
            "research_pipeline/v2/web_ui/server.py",
            "research_pipeline/v2/web_ui/security.py",
            "research_pipeline/v2/web_ui/data_access.py",
        ):
            mutated = dict(sources)
            mutated[key] = "f" * 64
            self.assertEqual(base, predictors.training_signature({"source_files": mutated}), f"{key} should be excluded")

        for core_key in ("research_pipeline/v2/worker.py", "research_pipeline/v2/runner.py"):
            mutated = dict(sources)
            mutated[core_key] = "1" * 64
            self.assertNotEqual(base, predictors.training_signature({"source_files": mutated}), f"{core_key} must invalidate")


class Phase4Tier1TerminalTests(unittest.TestCase):
    def test_color_enabled_respects_tty_no_color_and_term_dumb(self):
        class FakeTTY(io.StringIO):
            def isatty(self):
                return True

        tty_stream = FakeTTY()
        prev = ui._NO_COLOR
        try:
            ui.set_no_color(False)
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("NO_COLOR", None)
                os.environ["TERM"] = "xterm-256color"
                self.assertTrue(ui.color_enabled(tty_stream))
                self.assertIn("\x1b[", ui.badge("EMPIRICAL OPTIMAL", "success", stream=tty_stream))

            with patch.dict(os.environ, {"NO_COLOR": "1"}):
                self.assertFalse(ui.color_enabled(tty_stream))
                self.assertNotIn("\x1b[", ui.badge("EMPIRICAL OPTIMAL", "success", stream=tty_stream))

            with patch.dict(os.environ, {"TERM": "dumb"}):
                os.environ.pop("NO_COLOR", None)
                self.assertFalse(ui.color_enabled(tty_stream))

            ui.set_no_color(True)
            self.assertFalse(ui.color_enabled(tty_stream))
            self.assertFalse(ui.color_enabled(io.StringIO()))
        finally:
            ui.set_no_color(prev)

    def test_summary_view_renders_badges_star_pareto_and_map50_compat(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = _make_synthetic_run(Path(td), with_eval=False)
            prot = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
            rep = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            buf = io.StringIO()
            with redirect_stdout(buf):
                ui.summary_view(run_dir, prot, rep)
            out = buf.getvalue()
            self.assertNotRegex(out, _ANSI_ESC_RE)
            self.assertIn("EMPIRICAL OPTIMAL", out)
            self.assertIn("OOM", out)
            self.assertIn("★", out)
            self.assertIn("Pareto", out)
            self.assertIn("Val mAP50 ", out)
            self.assertNotIn("Val mAP50-95", out)

    def test_trial_progress_fields_and_nvml_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            (td_path / "result.json").write_text(
                json.dumps({"epochs_completed": 3, "quality_score": 0.745, "memory_mb": 512.4, "primary_metric": "map50"}),
                encoding="utf-8",
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                tp = ui.TrialProgress(
                    td_path, 2, 17, 5, True,
                    {"model": "yolo11n.pt", "imgsz": 64, "batch": 4, "precision": "fp32"},
                    60.0,
                    budget_remaining_fn=lambda: 42.5,
                    device="cpu",
                    detection_metric="map50",
                )
                tp.update()
            out = buf.getvalue()
            self.assertNotRegex(out, _ANSI_ESC_RE)
            self.assertIn("Config c02", out)
            self.assertIn("Seed 17", out)
            self.assertIn("FULL", out)
            self.assertIn("epoch 3/5", out)
            self.assertIn("Budget 42.5 giây", out)
            self.assertIn("Val mAP50 0.7450", out)
            self.assertIn("VRAM 512.4 MiB", out)
            self.assertIn("GPU Temp N/A", out)

    def test_c1_control_characters_stripped_including_8bit_csi(self):
        class FakeTTY(io.StringIO):
            def isatty(self):
                return True

        payload = "evil\x1b[31m\r\x9b[2J\x85\x9fhello Đúng"
        cleaned = ui.clean(payload)
        for code in range(0x7F, 0xA0):
            self.assertNotIn(chr(code), cleaned)
        self.assertIn("hello Đúng", cleaned)

        buf = io.StringIO()
        with redirect_stdout(buf):
            ui.say(payload)
        out_plain = buf.getvalue()
        self.assertNotIn("\x1b", out_plain)
        self.assertNotIn("\x9b", out_plain)
        self.assertNotIn("\x85", out_plain)
        self.assertIn("Đúng", out_plain)

        tty = FakeTTY()
        prev = ui._NO_COLOR
        try:
            ui.set_no_color(False)
            with patch.dict(os.environ, {"TERM": "xterm-256color"}, clear=False):
                os.environ.pop("NO_COLOR", None)
                ui._write_line("ok\x9b[2J" + ui.style_text("hi", "success", stream=tty), file=tty, allow_sgr=True)
            tty_out = tty.getvalue()
            self.assertNotIn("\x9b", tty_out)
            self.assertIn("\x1b[1;32m", tty_out)
        finally:
            ui.set_no_color(prev)


class Phase4WizardAndCLITests(unittest.TestCase):
    def test_scan_datasets_and_shortcut_resolution(self):
        with tempfile.TemporaryDirectory() as td:
            ds_root = Path(td)
            (ds_root / "ds_alpha").mkdir()
            (ds_root / "ds_alpha" / "data.yaml").write_text("path: .\n", encoding="utf-8")
            items = wizard.scan_datasets(ds_root)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["name"], "ds_alpha")
            with patch("builtins.input", return_value="1"):
                chosen = wizard.path_input("Dataset", shortcuts={"1": items[0]["path"]})
            self.assertEqual(chosen, items[0]["path"])

    def test_scan_weights_and_suggest_scales(self):
        weights_dir = REPO_ROOT / "weights"
        if weights_dir.is_dir() and list(weights_dir.glob("*.pt")):
            pts = wizard.scan_weights(weights_dir)
            self.assertTrue(pts)
            self.assertIsNotNone(pts[0]["parameters"])
            self.assertIsNotNone(pts[0]["gflops_reference_32"])
        t_cpu, m_cpu = wizard.suggest_scales({"device": "cpu", "ram_available_mb": 4096}, {"counts": {"train": 100}}, epochs=10)
        t_gpu, m_gpu = wizard.suggest_scales({"device": "0", "vram_total_mb": 6144}, {"counts": {"train": 100}}, epochs=10)
        self.assertGreater(t_cpu, t_gpu)
        self.assertGreater(m_gpu, m_cpu)

    def test_menu_benchmark_and_ui_flows_dispatch(self):
        dispatched = []
        scripted = "\n".join([
            "6",          # [6] Web Dashboard Studio
            "127.0.0.1",  # host
            "0",          # port
            "runs",       # runs_dir
            "y",          # confirm
            "0",          # exit
        ]) + "\n"
        old_stdin = sys.stdin
        buf = io.StringIO()
        try:
            sys.stdin = io.StringIO(scripted)
            with redirect_stdout(buf):
                rc = wizard.menu(lambda argv: dispatched.append(argv) or 0)
        finally:
            sys.stdin = old_stdin
        self.assertEqual(rc, 0)
        self.assertEqual(dispatched, [["ui", "--host", "127.0.0.1", "--port", "0", "--runs-dir", "runs"]])

    def test_cli_no_color_and_ui_rejects_non_loopback(self):
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            rc = cli.main(argv=["doctor", "--device", "cpu", "--no-color"])
        self.assertEqual(rc, 0)
        self.assertNotRegex(buf.getvalue(), _ANSI_ESC_RE)

        with tempfile.TemporaryDirectory() as td:
            buf2 = io.StringIO()
            with redirect_stdout(buf2), redirect_stderr(buf2):
                with self.assertRaises(SystemExit) as raised:
                    cli.main(argv=["ui", "--host", "0.0.0.0", "--port", "0", "--runs-dir", td])
            self.assertEqual(raised.exception.code, 2)


class Phase4WebUITests(unittest.TestCase):
    def _start_server(self, runs_root: Path):
        server, url, csrf = create_server(host="127.0.0.1", port=0, runs_dir=runs_root)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        return server, url, csrf

    def _req(self, url: str, method: str = "GET", path: str = "/", body: bytes | None = None, headers: dict | None = None):
        host_port = url.replace("http://", "")
        conn = http.client.HTTPConnection(host_port, timeout=10)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, dict(resp.getheaders()), raw
        finally:
            conn.close()

    def test_all_6_views_offline_theme_disclaimer_and_numeric_parity(self):
        with tempfile.TemporaryDirectory() as td:
            runs_dir = Path(td) / "runs"
            run_dir = _make_synthetic_run(runs_dir, "demo_run", with_eval=True, eval_scalar_latency=False)
            raw_rep = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            raw_ev = json.loads((run_dir / "evaluation" / "test_summary.json").read_text(encoding="utf-8"))

            server, url, csrf = self._start_server(runs_dir)
            try:
                for page in ("/", "/monitor", "/launcher", "/analysis", "/benchmark", "/export"):
                    code, _, html = self._req(url, "GET", page)
                    self.assertEqual(code, 200, f"{page} failed")
                    self.assertFalse(_EXT_URL_RE.search(html), f"External URL found in {page}")
                    self.assertIn(csrf.token, html)
                    self.assertIn("theme-toggle", html)
                    if page == "/analysis":
                        low = html.lower()
                        self.assertIn("phân tích độ nhạy hậu nghiệm", low)
                        self.assertIn("không làm thay đổi quyết định nghiên cứu đã khóa", low)

                for asset in ("/static/style.css", "/static/charts.js", "/static/app.js"):
                    code, _, txt = self._req(url, "GET", asset)
                    self.assertEqual(code, 200, f"{asset} failed")
                    self.assertFalse(_EXT_URL_RE.search(txt), f"External URL found in {asset}")

                code_r, _, body_r = self._req(url, "GET", "/api/run?path=demo_run")
                self.assertEqual(code_r, 200)
                run_json = json.loads(body_r)
                self.assertLess(abs(run_json["report"]["recommendation"]["utility"] - raw_rep["recommendation"]["utility"]), 1e-12)

                code_e, _, body_e = self._req(url, "GET", "/api/evaluation?path=demo_run")
                self.assertEqual(code_e, 200)
                ev_json = json.loads(body_e)["evaluation"]
                self.assertLess(abs(ev_json["map50"] - raw_ev["map50"]), 1e-12)
                self.assertLess(abs(ev_json["latency"]["fps"]["mean"] - raw_ev["latency"]["fps"]["mean"]), 1e-12)
            finally:
                server.shutdown()
                server.server_close()

    def test_hardware_api_cpu_fallback_and_gpu_mock(self):
        hw_cpu = data_access.hardware_snapshot()
        self.assertIn("cpu_percent", hw_cpu)
        self.assertIn("ram_available_mb", hw_cpu)
        self.assertIsInstance(hw_cpu["gpus"], list)

        with patch.object(data_access, "_torch_static_info", return_value={"torch_version": "2.14.1+cpu", "cuda_available": False}):
            with patch.dict(sys.modules, {"pynvml": None}):
                hw_fallback = data_access.hardware_snapshot()
                self.assertEqual(hw_fallback["mode"], "cpu")
                self.assertEqual(hw_fallback["gpus"], [])

    def test_security_traversal_csrf_and_non_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runs_dir = root / "runs"
            run_dir = _make_synthetic_run(runs_dir, "demo_run")
            (root / "outside_secret.txt").write_text("SECRET_SHOULD_NEVER_LEAK", encoding="utf-8")
            outside_dir = root / "outside_dir"
            outside_dir.mkdir()
            (outside_dir / "secret.json").write_text('{"secret":"LEAKED"}', encoding="utf-8")
            esc_link = runs_dir / "escape_dir"
            try:
                os.symlink(outside_dir, esc_link, target_is_directory=True)
            except (OSError, NotImplementedError):
                subprocess.run(["cmd", "/c", "mklink", "/J", str(esc_link), str(outside_dir)], capture_output=True, check=False)

            before_hashes = _sha256_tree(run_dir)
            server, url, csrf = self._start_server(runs_dir)
            try:
                for trav in (
                    "/api/file?path=../outside_secret.txt",
                    "/api/file?path=%2e%2e%2foutside_secret.txt",
                    "/api/file?path=%252e%252e%252foutside_secret.txt",
                    "/api/file?path=escape_dir/secret.json",
                    "/../outside_secret.txt",
                ):
                    code, _, body = self._req(url, "GET", trav)
                    self.assertEqual(code, 403, f"Expected 403 for {trav}, got {code}")
                    self.assertNotIn("SECRET_SHOULD_NEVER_LEAK", body)
                    self.assertNotIn("LEAKED", body)

                for post_ep in ("/api/inspect", "/api/audit", "/api/paper-export", "/api/launch", "/api/stop"):
                    code_get, _, _ = self._req(url, "GET", post_ep)
                    self.assertEqual(code_get, 405, f"GET {post_ep} must return 405")
                    code_nocsrf, _, _ = self._req(url, "POST", post_ep, b"{}", {"Content-Type": "application/json"})
                    self.assertEqual(code_nocsrf, 403, f"POST {post_ep} without CSRF must return 403")
                    code_badcsrf, _, _ = self._req(url, "POST", post_ep, b"{}", {"Content-Type": "application/json", "X-CSRF-Token": "wrong"})
                    self.assertEqual(code_badcsrf, 403, f"POST {post_ep} with bad CSRF must return 403")

                # Escaping paper-export rejected with 403 and not created
                esc_out = root / "escaped_export"
                code_pe_esc, _, _ = self._req(
                    url, "POST", "/api/paper-export",
                    json.dumps({"run": "demo_run", "output": str(esc_out), "lang": "en"}).encode("utf-8"),
                    {"Content-Type": "application/json", "X-CSRF-Token": csrf.token},
                )
                self.assertEqual(code_pe_esc, 403)
                self.assertFalse(esc_out.exists())

                # Launch validation: missing confirmed -> 400; escaping output -> 403
                code_l1, _, _ = self._req(
                    url, "POST", "/api/launch",
                    json.dumps({"data": str(run_dir), "confirmed": False}).encode("utf-8"),
                    {"Content-Type": "application/json", "X-CSRF-Token": csrf.token},
                )
                self.assertEqual(code_l1, 400)

                # Safe Stop without active process -> 200 no_active_process
                code_stop, _, body_stop = self._req(
                    url, "POST", "/api/stop",
                    json.dumps({"run": "demo_run"}).encode("utf-8"),
                    {"Content-Type": "application/json", "X-CSRF-Token": csrf.token},
                )
                self.assertEqual(code_stop, 200)
                self.assertEqual(json.loads(body_stop)["status"], "no_active_process")
            finally:
                server.shutdown()
                server.server_close()
                if esc_link.exists() or esc_link.is_symlink():
                    try:
                        esc_link.unlink()
                    except OSError:
                        os.rmdir(esc_link)
            self.assertEqual(before_hashes, _sha256_tree(run_dir))

    def test_audit_and_paper_export_on_real_v2_run_non_mutating(self):
        sample = _find_sample_v2_run()
        if sample is not None:
            before_sample = _sha256_tree(sample)
            server, url, csrf = self._start_server(sample.parent)
            try:
                code_a, _, body_a = self._req(
                    url, "POST", "/api/audit",
                    json.dumps({"run": sample.name}).encode("utf-8"),
                    {"Content-Type": "application/json", "X-CSRF-Token": csrf.token},
                )
                self.assertEqual(code_a, 200, msg=body_a)
                self.assertTrue(json.loads(body_a)["ok"])
            finally:
                server.shutdown()
                server.server_close()
            self.assertEqual(before_sample, _sha256_tree(sample))

        # Also run a tiny real classification optimize in tempdir to verify both /api/audit and /api/paper-export
        with tempfile.TemporaryDirectory() as td:
            from PIL import Image
            root = Path(td)
            ds = root / "mini_cls"
            for s_idx, split in enumerate(("train", "val")):
                for c_idx, cls in enumerate(("a", "b")):
                    d = ds / split / cls
                    d.mkdir(parents=True, exist_ok=True)
                    for i in range(4):
                        img = Image.new("RGB", (24, 24), (200 if cls == "a" else 30, 30 + s_idx * 40, 30 if cls == "a" else 200))
                        img.putpixel((i, s_idx * 3 + c_idx), (10 + i * 20, 250 - s_idx * 50, 120 + c_idx * 30))
                        img.save(d / f"{cls}_{i}.png")
            runs_dir = root / "runs"
            rc = cli.main(argv=[
                "optimize", "--data", str(ds), "--task", "classification", "--device", "cpu",
                "--space", "compact", "--full-epochs", "1", "--seeds", "17",
                "--budget-seconds", "60", "--output", str(runs_dir),
            ])
            self.assertEqual(rc, 0)
            run_dir = next(runs_dir.rglob("protocol.json")).parent
            before = _sha256_tree(run_dir)
            server, url, csrf = self._start_server(runs_dir)
            try:
                code_a, _, body_a = self._req(
                    url, "POST", "/api/audit",
                    json.dumps({"run": run_dir.name}).encode("utf-8"),
                    {"Content-Type": "application/json", "X-CSRF-Token": csrf.token},
                )
                self.assertEqual(code_a, 200, msg=body_a)
                self.assertTrue(json.loads(body_a)["ok"])

                out_dir = runs_dir / "paper_out"
                code_p, _, body_p = self._req(
                    url, "POST", "/api/paper-export",
                    json.dumps({"run": run_dir.name, "output": str(out_dir), "lang": "en"}).encode("utf-8"),
                    {"Content-Type": "application/json", "X-CSRF-Token": csrf.token},
                )
                self.assertEqual(code_p, 200, msg=body_p)
                self.assertTrue((out_dir / "PROVENANCE.json").is_file())
                self.assertTrue((out_dir / "tables.tex").is_file())
            finally:
                server.shutdown()
                server.server_close()
            self.assertEqual(before, _sha256_tree(run_dir))

    def test_load_test_eval_containment_blocks_external_and_falls_back(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runs_dir = root / "runs"
            outside_eval = root / "outside_eval_secret"
            outside_eval.mkdir(parents=True)
            (outside_eval / "evaluation.json").write_text(
                json.dumps({"secret_leak": "EXFILTRATED_OUTSIDE_RUNS_ROOT", "map50": 0.999}),
                encoding="utf-8",
            )

            # Case 1: external test_history output with no contained evaluation -> available=False
            evil_run = _make_synthetic_run(runs_dir, "evil_hist_run", with_eval=False)
            (evil_run / "test_history.json").write_text(
                json.dumps([{"status": "evaluated", "output": str(outside_eval)}]),
                encoding="utf-8",
            )

            # Case 2: copied run whose test_history points to external dir, but has contained evaluation/<ts>/evaluation.json
            copied_run = _make_synthetic_run(runs_dir, "copied_legacy_run", with_eval=True)
            (copied_run / "evaluation" / "test_summary.json").unlink()
            (copied_run / "test_history.json").write_text(
                json.dumps([{"status": "evaluated", "output": str(outside_eval)}]),
                encoding="utf-8",
            )

            server, url, _ = self._start_server(runs_dir)
            try:
                code1, _, body1 = self._req(url, "GET", "/api/evaluation?path=evil_hist_run")
                self.assertEqual(code1, 200)
                data1 = json.loads(body1)
                self.assertFalse(data1["available"])
                self.assertIsNone(data1["evaluation"])
                self.assertNotIn("EXFILTRATED_OUTSIDE_RUNS_ROOT", body1)

                code2, _, body2 = self._req(url, "GET", "/api/evaluation?path=copied_legacy_run")
                self.assertEqual(code2, 200)
                data2 = json.loads(body2)
                self.assertTrue(data2["available"])
                self.assertNotIn("EXFILTRATED_OUTSIDE_RUNS_ROOT", body2)
                self.assertAlmostEqual(data2["evaluation"]["map50"], 0.76543210987654, places=12)
            finally:
                server.shutdown()
                server.server_close()

    def test_benchmark_audit_containment_rejects_external_outcome_run(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runs_dir = root / "runs"
            outside_run = _make_synthetic_run(root / "external_runs", "outside_child", with_eval=False)
            bm_dir = runs_dir / "evil_bm"
            bm_dir.mkdir(parents=True)
            (bm_dir / "BENCHMARK.json").write_text(
                json.dumps({
                    "schema_version": 1,
                    "outcomes": [{"run": str(outside_run), "method": "adaptive_full", "search_seed": 17}],
                    "summaries": [],
                }),
                encoding="utf-8",
            )
            server, url, csrf = self._start_server(runs_dir)
            try:
                with patch("research_pipeline.v2.workflow.audit_run", side_effect=AssertionError("Must not audit outside runs_root")):
                    code, _, body = self._req(
                        url, "POST", "/api/audit",
                        json.dumps({"run": "evil_bm"}).encode("utf-8"),
                        {"Content-Type": "application/json", "X-CSRF-Token": csrf.token},
                    )
                self.assertEqual(code, 200)
                res = json.loads(body)
                self.assertFalse(res["ok"])
                containment_checks = [c for c in res["checks"] if c["name"] == "containment"]
                self.assertTrue(containment_checks)
                self.assertFalse(containment_checks[0]["passed"])
            finally:
                server.shutdown()
                server.server_close()

    def test_favicon_zero_404_and_app_js_contracts(self):
        from yolo_tuner.research import Objective
        with tempfile.TemporaryDirectory() as td:
            runs_dir = Path(td) / "runs"
            _make_synthetic_run(runs_dir, "demo_run", with_eval=True)
            server, url, _ = self._start_server(runs_dir)
            try:
                code_fav, _, _ = self._req(url, "GET", "/favicon.ico")
                self.assertIn(code_fav, (200, 204))

                code_home, _, html_home = self._req(url, "GET", "/")
                self.assertEqual(code_home, 200)
                self.assertIn('rel="icon" href="data:,"', html_home)

                code_js, _, js = self._req(url, "GET", "/static/app.js")
                self.assertEqual(code_js, 200)
                # 1. Multi-seed summary precedence over single-seed top-level keys
                self.assertIn("metricScalar(mss.map50) ?? metricScalar(ev.map50)", js)
                self.assertIn("metricScalar(mss.map50_95) ?? metricScalar(ev.map50_95)", js)
                self.assertIn("metricScalar(mss.precision) ?? metricScalar(ev.precision)", js)
                self.assertIn("ev.per_class_summary", js)
                self.assertLess(js.index("ev.per_class_summary"), js.index("ev.per_class || []"))

                # 2. Unclamped Objective.value() formula & slider initialization
                self.assertIn("nA * q - nT * tRatio - nM * mRatio", js)
                self.assertNotIn("Math.min(1, Number(c.time_mean", js)
                self.assertNotIn("Math.min(1, Number(c.memory_max", js)
                self.assertIn("prot.objective && prot.objective.weights", js)

                # Verify exact mathematical equivalence to Objective.value() when penalties > 1.0
                obj = Objective(name="balanced", time_scale=10.0, memory_scale=100.0, weights=(0.6, 0.25, 0.15))
                row = {"quality_mean": 0.8, "time_mean": 15.0, "memory_max": 50.0}
                expected_u = obj.value(row)
                js_equiv = 0.6 * row["quality_mean"] - 0.25 * (row["time_mean"] / 10.0) - 0.15 * (row["memory_max"] / 100.0)
                self.assertLess(abs(expected_u - js_equiv), 1e-12)

                # 3. Launcher CLI preview includes seeds, weights, and benchmark flags
                for flag in ("--seeds", "--weights", "--methods", "--search-seeds", "--reference-budget-seconds"):
                    self.assertIn(flag, js)

                # 4. Safe SVG preview via <img> (no raw innerHTML SVG injection)
                self.assertIn('<img src="${esc(url)}"', js)
                self.assertNotIn("previewEl.innerHTML = txt", js)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
