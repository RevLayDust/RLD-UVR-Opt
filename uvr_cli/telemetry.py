"""Hardware telemetry and performance monitoring for UVR benchmarks.

Provides background resource sampling (CPU, GPU, VRAM, Power) and system
environment discovery, with graceful fallback if NVML or CUDA is unavailable.
"""

from __future__ import annotations

import os
import platform
import threading
import time
from typing import Any, Dict, List, Optional

import psutil

_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        try:
            import torch
            _torch = torch
        except Exception:
            _torch = False
    return _torch if _torch is not False else None


_ort_info = None


def _get_ort_info():
    global _ort_info
    if _ort_info is None:
        try:
            import onnxruntime as ort
            _ort_info = (True, ort.__version__)
        except Exception:
            _ort_info = (False, "N/A")
    return _ort_info


_pynvml = None
_nvml_init_attempted = False


def _get_pynvml():
    global _pynvml, _nvml_init_attempted
    if not _nvml_init_attempted:
        _nvml_init_attempted = True
        try:
            import pynvml
            pynvml.nvmlInit()
            _pynvml = pynvml
        except Exception:
            _pynvml = None
    return _pynvml


def __getattr__(name: str) -> Any:
    if name == "NVML_AVAILABLE":
        return _get_pynvml() is not None
    if name == "ORT_AVAILABLE":
        return _get_ort_info()[0]
    if name == "ORT_VERSION":
        return _get_ort_info()[1]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_cpu_name() -> str:
    """Detect and report the actual human-readable CPU model name."""
    system = platform.system()
    try:
        if system == "Windows":
            try:
                import winreg
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                )
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                winreg.CloseKey(key)
                if name and isinstance(name, str) and name.strip():
                    return " ".join(name.strip().split())
            except Exception:
                pass
        elif system == "Darwin":
            try:
                import subprocess
                cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
                output = subprocess.check_output(cmd, encoding="utf-8", stderr=subprocess.DEVNULL)
                if output and output.strip():
                    return " ".join(output.strip().split())
            except Exception:
                pass
        elif system == "Linux":
            try:
                with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if line.strip().startswith("model name"):
                            parts = line.split(":", 1)
                            if len(parts) > 1 and parts[1].strip():
                                return " ".join(parts[1].strip().split())
            except Exception:
                pass
    except Exception:
        pass

    fallback = platform.processor() or platform.machine() or "Unknown CPU"
    return " ".join(fallback.strip().split()) if isinstance(fallback, str) else "Unknown CPU"


def sync_cuda(device_index: int = 0) -> None:
    """Synchronize CUDA device if available."""
    torch_mod = _get_torch()
    if torch_mod is not None and torch_mod.cuda.is_available():
        try:
            torch_mod.cuda.synchronize(device_index)
        except Exception:
            pass


def collect_system_info(device_index: int = 0) -> Dict[str, Any]:
    """Collect hardware and software runtime environment details."""
    gpu_name = None
    driver_version = None
    total_gpu_vram_mb = 0.0

    torch_mod = _get_torch()
    nvml_mod = _get_pynvml()
    _, ort_version = _get_ort_info()

    if torch_mod is not None and torch_mod.cuda.is_available():
        try:
            gpu_name = torch_mod.cuda.get_device_name(device_index)
        except Exception:
            pass

    if nvml_mod is not None:
        try:
            handle = nvml_mod.nvmlDeviceGetHandleByIndex(device_index)
            if not gpu_name:
                gpu_name = nvml_mod.nvmlDeviceGetName(handle)
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode("utf-8")
            driver_version = nvml_mod.nvmlSystemGetDriverVersion()
            if isinstance(driver_version, bytes):
                driver_version = driver_version.decode("utf-8")
            mem_info = nvml_mod.nvmlDeviceGetMemoryInfo(handle)
            total_gpu_vram_mb = round(mem_info.total / (1024 * 1024), 2)
        except Exception:
            pass

    cpu_name = get_cpu_name()

    return {
        "os": platform.platform(),
        "cpu": cpu_name,
        "cpu_name": cpu_name,
        "cpu_count_logical": psutil.cpu_count(logical=True) or 0,
        "cpu_count_physical": psutil.cpu_count(logical=False) or 0,
        "system_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu_name": gpu_name or "N/A",
        "total_gpu_vram_mb": total_gpu_vram_mb,
        "driver_version": driver_version or "N/A",
        "cuda_available": torch_mod.cuda.is_available() if torch_mod is not None else False,
        "cuda_version": (torch_mod.version.cuda or "N/A") if torch_mod is not None else "N/A",
        "torch_version": torch_mod.__version__ if torch_mod is not None else "N/A",
        "onnxruntime_version": ort_version,
        "python_version": platform.python_version(),
    }


class TelemetrySampler:
    """Background sampling monitor measuring CPU, RAM, GPU utilization, VRAM, and power."""

    def __init__(self, device_index: int = 0, interval_sec: float = 0.1) -> None:
        self.device_index = device_index
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._process = psutil.Process(os.getpid())
        self._nvml_handle: Any = None
        nvml_mod = _get_pynvml()
        if nvml_mod is not None:
            try:
                self._nvml_handle = nvml_mod.nvmlDeviceGetHandleByIndex(self.device_index)
            except Exception:
                self._nvml_handle = None

        # Sample storage
        self.process_cpu_samples: List[float] = []
        self.system_cpu_samples: List[float] = []
        self.process_ram_samples: List[float] = []
        self.gpu_util_samples: List[float] = []
        self.device_vram_samples: List[float] = []
        self.power_samples: List[float] = []

        self.baseline_device_vram_mb: float = 0.0
        self.baseline_torch_allocated_mb: float = 0.0

    def start(self) -> None:
        """Start the background sampling thread and reset peak GPU stats."""
        # Initial CPU reading primes psutil
        try:
            self._process.cpu_percent()
            psutil.cpu_percent()
        except Exception:
            pass

        torch_mod = _get_torch()
        if torch_mod is not None and torch_mod.cuda.is_available():
            try:
                sync_cuda(self.device_index)
                torch_mod.cuda.reset_peak_memory_stats(self.device_index)
                self.baseline_torch_allocated_mb = round(
                    torch_mod.cuda.memory_allocated(self.device_index) / (1024 * 1024), 2
                )
            except Exception:
                pass

        if self._nvml_handle:
            nvml_mod = _get_pynvml()
            if nvml_mod is not None:
                try:
                    mem = nvml_mod.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    self.baseline_device_vram_mb = round(mem.used / (1024 * 1024), 2)
                except Exception:
                    pass

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True, name="TelemetrySampler")
        self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                # CPU & RAM
                proc_cpu = self._process.cpu_percent()
                sys_cpu = psutil.cpu_percent()
                proc_ram = self._process.memory_info().rss / (1024 * 1024)

                self.process_cpu_samples.append(proc_cpu)
                self.system_cpu_samples.append(sys_cpu)
                self.process_ram_samples.append(proc_ram)

                # NVML metrics
                if self._nvml_handle:
                    nvml_mod = _get_pynvml()
                    if nvml_mod is not None:
                        rates = nvml_mod.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                        self.gpu_util_samples.append(float(rates.gpu))

                        mem = nvml_mod.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                        self.device_vram_samples.append(mem.used / (1024 * 1024))

                        try:
                            power_w = nvml_mod.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0
                            self.power_samples.append(power_w)
                        except Exception:
                            pass
            except Exception:
                pass

            time.sleep(self.interval_sec)

    def stop(self) -> Dict[str, Any]:
        """Stop sampling and return aggregated hardware metrics."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        sync_cuda(self.device_index)

        torch_peak_alloc = 0.0
        torch_peak_reserved = 0.0
        torch_mod = _get_torch()
        if torch_mod is not None and torch_mod.cuda.is_available():
            try:
                torch_peak_alloc = round(
                    torch_mod.cuda.max_memory_allocated(self.device_index) / (1024 * 1024), 2
                )
                torch_peak_reserved = round(
                    torch_mod.cuda.max_memory_reserved(self.device_index) / (1024 * 1024), 2
                )
            except Exception:
                pass

        avg_proc_cpu = (
            round(sum(self.process_cpu_samples) / len(self.process_cpu_samples), 2)
            if self.process_cpu_samples
            else 0.0
        )
        peak_proc_cpu = round(max(self.process_cpu_samples), 2) if self.process_cpu_samples else 0.0

        avg_sys_cpu = (
            round(sum(self.system_cpu_samples) / len(self.system_cpu_samples), 2)
            if self.system_cpu_samples
            else 0.0
        )
        peak_sys_cpu = round(max(self.system_cpu_samples), 2) if self.system_cpu_samples else 0.0

        peak_proc_ram = (
            round(max(self.process_ram_samples), 2) if self.process_ram_samples else 0.0
        )

        avg_gpu_util = (
            round(sum(self.gpu_util_samples) / len(self.gpu_util_samples), 2)
            if self.gpu_util_samples
            else 0.0
        )
        peak_gpu_util = round(max(self.gpu_util_samples), 2) if self.gpu_util_samples else 0.0

        peak_device_vram = (
            round(max(self.device_vram_samples), 2)
            if self.device_vram_samples
            else (self.baseline_device_vram_mb or torch_peak_alloc)
        )

        avg_power = (
            round(sum(self.power_samples) / len(self.power_samples), 2)
            if self.power_samples
            else 0.0
        )
        peak_power = round(max(self.power_samples), 2) if self.power_samples else 0.0

        return {
            "samples_count": len(self.process_cpu_samples),
            "avg_process_cpu_percent": avg_proc_cpu,
            "peak_process_cpu_percent": peak_proc_cpu,
            "avg_system_cpu_percent": avg_sys_cpu,
            "peak_system_cpu_percent": peak_sys_cpu,
            "peak_process_ram_mb": peak_proc_ram,
            "avg_gpu_utilization_percent": avg_gpu_util,
            "peak_gpu_utilization_percent": peak_gpu_util,
            "baseline_device_vram_mb": self.baseline_device_vram_mb,
            "peak_device_vram_mb": peak_device_vram,
            "torch_peak_allocated_vram_mb": torch_peak_alloc,
            "torch_peak_reserved_vram_mb": torch_peak_reserved,
            "avg_power_watts": avg_power,
            "peak_power_watts": peak_power,
        }
