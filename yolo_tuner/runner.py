"""Subprocess timeout enforcement and hardware inspection."""

from dataclasses import asdict
import json
from importlib.metadata import version
from pathlib import Path
import shutil
import subprocess
import sys
import time


def stop_process_tree(process):
    """Terminate only this worker and its descendants, including Windows venv launchers."""
    import psutil
    if process.poll() is not None:
        return
    try:
        parent = psutil.Process(process.pid)
        descendants = parent.children(recursive=True)
        for child in reversed(descendants):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            parent.kill()
        except psutil.NoSuchProcess:
            pass
        psutil.wait_procs(descendants + [parent], timeout=5)
    except psutil.NoSuchProcess:
        pass
    process.wait(timeout=5)


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def resolve_device(device="auto"):
    """Pick a working compute backend, not merely a physically present adapter."""
    if device != "auto":
        return str(device).removeprefix("cuda:")
    try:
        import torch
        if torch.cuda.is_available():
            try:
                test = torch.ones(1,device="cuda:0") * 2
                torch.cuda.synchronize()
                del test
                return "0"
            except (RuntimeError,AssertionError):
                pass
        if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available():
            try:
                test = torch.ones(1,device="mps") * 2
                torch.mps.synchronize()
                del test
                return "mps"
            except (RuntimeError,AssertionError):
                pass
    except ImportError:
        pass
    return "cpu"


def _nvml_gpu(index, device_uuid=None):
    """Read one physical NVIDIA adapter directly from the driver through NVML."""
    import pynvml
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByUUID(device_uuid) if device_uuid else pynvml.nvmlDeviceGetHandleByIndex(index)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        pci = pynvml.nvmlDeviceGetPciInfo(handle)
        reported_uuid=_text(pynvml.nvmlDeviceGetUUID(handle))
        physical_uuid=reported_uuid
        if reported_uuid.startswith("MIG-"):
            parent=pynvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(handle)
            physical_uuid=_text(pynvml.nvmlDeviceGetUUID(parent))
        result = {
            "gpu": _text(pynvml.nvmlDeviceGetName(handle)),
            "gpu_uuid": physical_uuid,
            "gpu_instance_uuid": reported_uuid if reported_uuid.startswith("MIG-") else None,
            "gpu_identity_verified": True,
            "pci_bus_id": _text(pci.busId),
            "driver_version": _text(pynvml.nvmlSystemGetDriverVersion()),
            "vram_total_mb": memory.total / 2**20,
            "vram_used_mb": memory.used / 2**20,
            "vram_free_mb": memory.free / 2**20,
            "compute_capability": [major, minor],
            "temperature_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
            "gpu_source": "NVIDIA NVML driver",
        }
        try:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            result.update(gpu_utilization_percent=utilization.gpu,
                          memory_utilization_percent=utilization.memory)
        except pynvml.NVMLError:
            pass
        return result
    finally:
        pynvml.nvmlShutdown()


def hardware(device="0", require_cuda=True):
    import psutil
    import torch

    device = resolve_device(device)

    result = {"cpu": __import__("platform").processor(), "logical_cpus": psutil.cpu_count(),
              "ram_total_mb": psutil.virtual_memory().total / 2**20,
              "ram_available_mb": psutil.virtual_memory().available / 2**20,
              "disk_free_mb": shutil.disk_usage(Path.cwd()).free / 2**20,
              "device": device, "torch_version": torch.__version__,
              "ultralytics_version": version("ultralytics"), "cuda_available": torch.cuda.is_available(),
              "torch_cuda_version": torch.version.cuda, "cuda_training_ready": False}
    if device == "cpu":
        return {**result, "gpu": None, "gpu_source": "CPU selected", "fp16": False, "bf16": False,
                "vram_total_mb": None, "vram_used_mb": None, "vram_free_mb": None}
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS runtime unavailable; use --device auto or cpu")
        return {**result,"gpu":"Apple MPS","gpu_source":"PyTorch MPS runtime",
                "fp16":False,"bf16":False,"vram_total_mb":None,"vram_used_mb":None,
                "vram_free_mb":None,"backend":"mps","shared_memory":True}
    if not torch.cuda.is_available():
        if require_cuda:
            raise ValueError("CUDA unavailable. Install CUDA PyTorch, or pass --device cpu explicitly.")
        # Read physical NVIDIA hardware even in a CPU-only Python environment.
        try:
            physical = _nvml_gpu(int(device))
            major = physical["compute_capability"][0]
            return {**result, **physical, "fp16": major >= 6, "bf16": major >= 8,
                    "precision_check_basis": "physical GPU capability; PyTorch CUDA runtime unavailable"}
        except Exception as exc:
            raise ValueError(f"NVIDIA hardware inspection failed: {exc}") from exc
    index = int(device)
    torch.cuda.set_device(index)
    free, total = torch.cuda.mem_get_info(index)
    major, minor = torch.cuda.get_device_capability(index)
    raw_uuid=getattr(torch.cuda.get_device_properties(index),"uuid",None)
    cuda_uuid=None
    if raw_uuid is not None:
        value=str(raw_uuid)
        if value.startswith(("GPU-","MIG-")):
            cuda_uuid=value
        else:
            import uuid
            try:
                cuda_uuid="GPU-"+str(uuid.UUID(value))
            except (ValueError,AttributeError):
                pass
    try:
        # CUDA ordinal can be remapped by CUDA_VISIBLE_DEVICES/device ordering.
        # Never attach NVML physical index 0 to logical CUDA device 0 blindly.
        if not cuda_uuid:
            raise ValueError("CUDA device UUID unavailable; physical identity unverified")
        try:
            physical = _nvml_gpu(index,device_uuid=cuda_uuid)
        except Exception:
            if cuda_uuid.startswith("GPU-"):
                physical=_nvml_gpu(index,device_uuid="MIG-"+cuda_uuid[4:])
            else:
                raise
    except Exception:
        # Driver metadata is optional; actual CUDA measurements still use torch.
        physical = {"gpu": torch.cuda.get_device_name(index), "gpu_source": "PyTorch CUDA runtime; physical UUID unverified",
                    "gpu_uuid":None,"gpu_identity_verified":False}
    return {**result, **physical, "gpu": torch.cuda.get_device_name(index), "vram_total_mb": total / 2**20,
            "vram_free_mb": free / 2**20, "vram_used_mb": (total - free) / 2**20,
            "compute_capability": [major, minor], "fp16": major >= 6,
            "cuda_device_uuid":cuda_uuid,
            "bf16": torch.cuda.is_bf16_supported(), "cuda_training_ready": True,
            "precision_check_basis": "PyTorch CUDA runtime verified"}


class Runner:
    def __init__(self, folder, data, stats, device="0", full_epochs=100, seed=17):
        self.folder = Path(folder).resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.data, self.stats = data, stats
        self.device, self.full_epochs, self.seed = device, full_epochs, seed
        self.index = 0

    def run(self, config, mode, seconds, epochs=1, resume=None):
        self.index += 1
        folder = self.folder / f"trial_{self.index:04d}_{mode}"
        folder.mkdir()
        request = {"config": asdict(config), "mode": mode, "data": self.data,
                   "dataset_stats": self.stats, "device": self.device, "full_epochs": self.full_epochs,
                   "trial_epochs": epochs, "seed": self.seed, "project": str(folder), "resume": resume}
        request_path, result_path = folder / "request.json", folder / "result.json"
        request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
        started = time.monotonic()
        with (folder / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "yolo_tuner.worker", str(request_path), str(result_path)],
                                       stdout=log, stderr=subprocess.STDOUT)
            timed_out = False
            try:
                process.wait(timeout=max(0.01, seconds))
            except subprocess.TimeoutExpired:
                timed_out = True
                stop_process_tree(process)
            except BaseException:
                stop_process_tree(process)
                raise
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result = {"config": asdict(config), "status": "error", "error": f"Worker exited {process.returncode} without result"}
        if timed_out:
            result["status"] = "timeout"
            result["error"] = "Trial exceeded its process wall-clock deadline"
        elif process.returncode != 0:
            result["status"] = "error"
            result["error"] = f"Worker exited {process.returncode}; inspect worker.log"
        result["worker_wall_seconds"] = time.monotonic() - started
        result["log"] = str(folder / "worker.log")
        if result["status"] == "ok" and "estimated_full_seconds" in result:
            if mode == "search":
                setup = max(0, result["worker_wall_seconds"] - result.get("measured_loop_seconds", 0))
                final_validation = result["val_epoch_seconds"] * self.stats["val_images"] / self.stats["val_subset_images"]
                result["estimated_full_seconds"] += setup + final_validation
                result["estimated_setup_seconds"] = setup
                result["time_estimate_basis"] = (
                    "scaled measured train/val loops + measured worker setup + one final validation; "
                    "full checkpoint I/O and interval coverage not calibrated")
            elif mode == "train":
                result["estimated_full_seconds"] = result["worker_wall_seconds"]
                result["time_estimate_basis"] = "actual full-training worker wall time"
        # Trainer versions may auto-reduce batch or disable AMP. Use actual settings.
        result["config"] = {**asdict(config), "batch": result.get("actual_batch", config.batch),
                            "precision": result.get("actual_precision", config.precision)}
        return result
