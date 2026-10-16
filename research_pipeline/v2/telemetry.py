"""Sampled resource telemetry; phase wall times are exclusive, not kernel times."""

from contextlib import contextmanager
import threading
import time

import psutil


class PhaseMonitor:
    def __init__(self, device, interval=0.2):
        self.device, self.interval = device, interval
        self.process = psutil.Process()
        self.lock, self.stop = threading.RLock(), threading.Event()
        self.phase, self.tick = "initialization", time.monotonic()
        self.phases, self.samples, self.rss_peak = {}, 0, 0.0
        self.reserved_peak, self.allocated_peak = 0.0, 0.0
        self.nvml = None
        self.diagnostics = {"device_usage_peak_mb": None, "process_vram_peak_mb": None,
                            "temperature_max_c": None, "power_max_watts": None}
        if device not in {"cpu", "mps"}:
            try:
                import torch
                import pynvml
                torch.cuda.set_device(int(str(device).removeprefix("cuda:")))
                raw = str(torch.cuda.get_device_properties().uuid)
                if not raw.startswith(("GPU-", "MIG-")):
                    import uuid
                    raw = "GPU-" + str(uuid.UUID(raw))
                pynvml.nvmlInit()
                self.nvml = (pynvml, pynvml.nvmlDeviceGetHandleByUUID(raw))
            except Exception:
                self.nvml = None
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def sample(self):
        with self.lock:
            rss = self.process.memory_info().rss
            for child in self.process.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self.rss_peak = max(self.rss_peak, rss / 2**20)
            allocated = reserved = 0.0
            if self.device not in {"cpu", "mps"}:
                import torch
                if torch.cuda.is_initialized():
                    index = int(str(self.device).removeprefix("cuda:"))
                    reserved = torch.cuda.memory_reserved(index) / 2**20
                    allocated = torch.cuda.memory_allocated(index) / 2**20
                    self.reserved_peak = max(self.reserved_peak, torch.cuda.max_memory_reserved(index) / 2**20)
                    self.allocated_peak = max(self.allocated_peak, torch.cuda.max_memory_allocated(index) / 2**20)
            point = self.phases.setdefault(self.phase, {"wall_seconds": 0.0, "sampled_rss_max_mb": 0.0,
                                                       "sampled_reserved_max_mb": 0.0, "sampled_allocated_max_mb": 0.0})
            point["sampled_rss_max_mb"] = max(point["sampled_rss_max_mb"], rss / 2**20)
            point["sampled_reserved_max_mb"] = max(point["sampled_reserved_max_mb"], reserved)
            point["sampled_allocated_max_mb"] = max(point["sampled_allocated_max_mb"], allocated)
            self.samples += 1
            if self.nvml and self.samples % 5 == 0:
                nv, handle = self.nvml
                for key, query in (("device_usage_peak_mb", lambda: nv.nvmlDeviceGetMemoryInfo(handle).used / 2**20),
                                   ("temperature_max_c", lambda: nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU)),
                                   ("power_max_watts", lambda: nv.nvmlDeviceGetPowerUsage(handle) / 1000)):
                    try:
                        v = query()
                        self.diagnostics[key] = max(self.diagnostics[key] or 0, v)
                    except Exception:
                        pass
                try:
                    usage = [p.usedGpuMemory for p in nv.nvmlDeviceGetComputeRunningProcesses(handle) if p.pid == self.process.pid]
                    if usage and 0 <= usage[0] < 2**60:  # WDDM often reports unavailable.
                        self.diagnostics["process_vram_peak_mb"] = max(self.diagnostics["process_vram_peak_mb"] or 0, usage[0] / 2**20)
                except Exception:
                    pass

    def loop(self):
        while not self.stop.wait(self.interval):
            try:
                self.sample()
            except (psutil.Error, RuntimeError):
                pass

    def set_phase(self, name):
        with self.lock:
            self.sample()
            now = time.monotonic()
            self.phases[self.phase]["wall_seconds"] += now - self.tick
            self.tick, self.phase = now, name

    @contextmanager
    def during(self, name):
        old = self.phase
        self.set_phase(name)
        try:
            yield
        finally:
            self.set_phase(old)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        self.set_phase("complete")
        if self.nvml:
            self.nvml[0].nvmlShutdown()
        return {"phases": self.phases, "sampling_interval_seconds": self.interval, "sample_count": self.samples,
                "peak_rss_mb": self.rss_peak, "peak_reserved_mb": self.reserved_peak,
                "peak_allocated_mb": self.allocated_peak, "diagnostics": self.diagnostics,
                "counter_policy": "primary CUDA reserved allocator; CPU sum process-tree RSS (shared pages may be counted more than once); NOT total device memory",
                "limitations": "sampled per-phase peaks may miss short peaks; phase times are wall time, not synchronized kernel attribution"}
