from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

FALSE_VALUES = {"0", "false", "no", "off", "disabled"}
SEPARATOR_BACKENDS = {
    "SeperateVR": "VR",
    "SeperateMDX": "MDX-Net ONNX",
    "SeperateMDXC": "MDX PyTorch",
    "SeperateDemucs": "Demucs",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "benchmark_cache"
RECORDER_VERSION = "1.0.0"

_WINDOWS_PATH = re.compile(r"(?i)(?<![A-Za-z0-9])[A-Z]:[\\/][^\r\n\"']+")
_UNC_PATH = re.compile(r"(?<![\\A-Za-z0-9])\\\\[^\\\r\n\"']+[\\/][^\r\n\"']+")
_ABSOLUTE_UNIX_PATH = re.compile(r"(?<![:/A-Za-z0-9])/(?!/)[^\r\n\"']+")
_SECRET_KEY = re.compile(r"(?i)(token|secret|password|passwd|api[_-]?key|credential)")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(token|secret|password|passwd|api[_-]?key|credential)\b\s*[:=]\s*[^\s,;]+"
)
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slugify(value: str, fallback: str = "unknown") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return slug or fallback


def sanitize_text(value: str) -> str:
    sanitized = _ANSI_ESCAPE.sub("", str(value))
    sanitized = _WINDOWS_PATH.sub("<redacted-path>", sanitized)
    sanitized = _UNC_PATH.sub("<redacted-path>", sanitized)
    sanitized = _ABSOLUTE_UNIX_PATH.sub("<redacted-path>", sanitized)
    return _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=<redacted>", sanitized)


def sanitize_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<redacted>" if _SECRET_KEY.search(str(key)) else sanitize_data(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_data(item) for item in value]
    if isinstance(value, str):
        return sanitize_text(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_data(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_environment(device_index: int = 0) -> dict[str, Any]:
    hardware: dict[str, Any] = {
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine() or "unknown",
        "system_ram_gb": _system_ram_gb(),
        "gpu_name": None,
        "gpu_vram_mb": None,
        "driver_version": None,
        "compute_capability": None,
        "form_factor": "unknown",
        "power_limit_w": None,
    }
    software: dict[str, Any] = {
        "python_version": platform.python_version(),
        "torch_version": None,
        "cuda_runtime_version": None,
        "cudnn_version": None,
        "torch_cuda_available": False,
        "onnxruntime_version": None,
        "onnxruntime_providers": [],
    }
    warnings: list[str] = []
    try:
        import torch

        software["torch_version"] = torch.__version__
        software["cuda_runtime_version"] = torch.version.cuda
        software["cudnn_version"] = torch.backends.cudnn.version()
        software["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available() and 0 <= device_index < torch.cuda.device_count():
            properties = torch.cuda.get_device_properties(device_index)
            hardware["gpu_name"] = properties.name
            hardware["gpu_vram_mb"] = properties.total_memory / (1024 * 1024)
            hardware["compute_capability"] = f"{properties.major}.{properties.minor}"
            hardware["form_factor"] = "laptop" if "laptop" in properties.name.lower() else "unknown"
    except Exception as error:
        warnings.append(f"PyTorch metadata unavailable: {type(error).__name__}")
    try:
        import onnxruntime as ort

        software["onnxruntime_version"] = ort.__version__
        software["onnxruntime_providers"] = list(ort.get_available_providers())
    except Exception as error:
        warnings.append(f"ONNX Runtime metadata unavailable: {type(error).__name__}")
    _add_nvml_metadata(hardware, device_index, warnings)
    return sanitize_data({
        "recorder_version": RECORDER_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "hardware": hardware,
        "software": software,
        "warnings": warnings,
    })


def _system_ram_gb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return None


def _add_nvml_metadata(hardware: dict[str, Any], device_index: int, warnings: list[str]) -> None:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle, mapped_by_pci = nvml_handle_for_torch_device(pynvml, device_index)
            if not mapped_by_pci:
                warnings.append("NVML metadata used device-index fallback")
            name = pynvml.nvmlDeviceGetName(handle)
            hardware["gpu_name"] = name.decode(errors="replace") if isinstance(name, bytes) else str(name)
            hardware["gpu_vram_mb"] = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 * 1024)
            driver = pynvml.nvmlSystemGetDriverVersion()
            hardware["driver_version"] = driver.decode() if isinstance(driver, bytes) else str(driver)
            try:
                hardware["power_limit_w"] = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            except Exception:
                pass
        finally:
            pynvml.nvmlShutdown()
    except Exception as error:
        warnings.append(f"NVML metadata unavailable: {type(error).__name__}")


def nvml_handle_for_torch_device(nvml: Any, device_index: int) -> tuple[Any, bool]:
    try:
        import torch

        properties = torch.cuda.get_device_properties(device_index)
        candidates = (
            f"{int(properties.pci_domain_id):08x}:{int(properties.pci_bus_id):02x}:{int(properties.pci_device_id):02x}.0",
            f"{int(properties.pci_domain_id):04x}:{int(properties.pci_bus_id):02x}:{int(properties.pci_device_id):02x}.0",
        )
        for candidate in candidates:
            for value in (candidate, candidate.encode("ascii")):
                try:
                    return nvml.nvmlDeviceGetHandleByPciBusId(value), True
                except Exception:
                    continue
    except Exception:
        pass
    return nvml.nvmlDeviceGetHandleByIndex(device_index), False


def install_gui_runtime_benchmark_hooks(
    namespace: dict[str, Any],
    recorder: "GUIRuntimeBenchmarkRecorder | None" = None,
) -> "GUIRuntimeBenchmarkRecorder | None":
    if os.environ.get("UVR_GUI_BENCHMARK", "1").strip().lower() in FALSE_VALUES:
        return None
    existing = namespace.get("_GUI_RUNTIME_BENCHMARK_RECORDER")
    runtime_recorder = recorder or existing or GUIRuntimeBenchmarkRecorder(namespace)
    namespace["_GUI_RUNTIME_BENCHMARK_RECORDER"] = runtime_recorder
    for class_name, backend in SEPARATOR_BACKENDS.items():
        separator_class = namespace.get(class_name)
        if separator_class is None:
            continue
        original = separator_class.seperate
        if getattr(original, "_uvr_gui_benchmark_wrapped", False):
            continue

        @functools.wraps(original)
        def wrapped(instance, *args, __original=original, __backend=backend, **kwargs):
            return runtime_recorder.run(instance, __backend, __original, args, kwargs)

        wrapped._uvr_gui_benchmark_wrapped = True
        wrapped._uvr_gui_benchmark_original = original
        separator_class.seperate = wrapped
    return runtime_recorder


class GUIRuntimeBenchmarkRecorder:
    def __init__(
        self,
        namespace: dict[str, Any],
        output_root: Path | None = None,
        sampler_factory: Callable[..., Any] | None = None,
        environment_collector: Callable[[int], dict[str, Any]] = collect_environment,
        progress_interval_seconds: float = 5.0,
    ):
        self.namespace = namespace
        configured_root = os.environ.get("UVR_GUI_BENCHMARK_DIR")
        selected_root = Path(configured_root).expanduser() if configured_root else (
            output_root or DEFAULT_CACHE_ROOT / "gui_runtime"
        )
        self.output_root = selected_root.resolve()
        self.sampler_factory = sampler_factory or RuntimeResourceSampler
        self.environment_collector = environment_collector
        self.progress_interval_seconds = max(float(progress_interval_seconds), 1.0)
        self._lock = threading.RLock()
        self._thread_state = threading.local()
        self._inference_index = 0
        self._last_model_signature: tuple[str, str] | None = None
        self._model_round = 0
        self._environment_cache: dict[int, dict[str, Any]] = {}
        self._model_hash_cache: dict[tuple[str, int, int], str] = {}
        self._session_directory: Path | None = None
        self._session_log: Path | None = None

    def run(
        self,
        instance: Any,
        backend: str,
        original: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        depth = getattr(self._thread_state, "depth", 0)
        if depth:
            return original(instance, *args, **kwargs)
        self._thread_state.depth = depth + 1
        try:
            return self._run_top_level(instance, backend, original, args, kwargs)
        finally:
            self._thread_state.depth = depth

    def _run_top_level(
        self,
        instance: Any,
        backend: str,
        original: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        started = time.perf_counter()
        record = None
        instrumentor = None
        sampler = None
        try:
            record = self._begin_record(instance, backend)
            instrumentor = RuntimeStageInstrumentor(
                self.namespace,
                instance,
                record["device_index"],
                lambda name, value: self._record_stage(record, name, value),
            )
            sampler = self.sampler_factory(
                device_index=record["device_index"],
                progress_interval_seconds=self.progress_interval_seconds,
                progress=lambda snapshot: self._record_progress(record, snapshot),
            )
            instrumentor.install()
            sampler.start()
        except Exception as error:
            setup_warning = f"Recorder setup failed: {type(error).__name__}"
            cleanup_warnings = []
            if instrumentor is not None:
                try:
                    instrumentor.restore()
                except Exception as cleanup_error:
                    cleanup_warnings.append(
                        f"Recorder hook cleanup failed: {type(cleanup_error).__name__}"
                    )
            if sampler is not None:
                try:
                    sampler.stop()
                except Exception as cleanup_error:
                    cleanup_warnings.append(
                        f"Recorder sampler cleanup failed: {type(cleanup_error).__name__}"
                    )
            warnings = [setup_warning, *cleanup_warnings]
            if record is not None:
                for warning in warnings:
                    self._record_warning(record, warning)
            self._notify_gui(instance, f"[Benchmark warning] {setup_warning}; processing will continue")
            failure: BaseException | None = None
            try:
                return original(instance, *args, **kwargs)
            except BaseException as backend_error:
                failure = backend_error
                raise
            finally:
                synchronize_cuda(selected_device_index(instance))
                if record is not None:
                    self._finish_record_safely(
                        record,
                        instance,
                        time.perf_counter() - started,
                        {"warnings": warnings},
                        failure,
                    )

        failure: BaseException | None = None
        synchronize_cuda(record["device_index"])
        try:
            return original(instance, *args, **kwargs)
        except BaseException as error:
            failure = error
            raise
        finally:
            synchronize_cuda(record["device_index"])
            total_seconds = time.perf_counter() - started
            resources: dict[str, Any] = {}
            recorder_warnings = []
            try:
                instrumentor.finish()
            except Exception as recorder_error:
                recorder_warnings.append(
                    f"Stage finalization failed: {type(recorder_error).__name__}"
                )
            try:
                resources = sampler.stop() or {}
            except Exception as recorder_error:
                recorder_warnings.append(
                    f"Resource sampler finalization failed: {type(recorder_error).__name__}"
                )
            try:
                instrumentor.restore()
            except Exception as recorder_error:
                recorder_warnings.append(
                    f"Recorder hook cleanup failed: {type(recorder_error).__name__}"
                )
            resources.setdefault("warnings", []).extend(recorder_warnings)
            self._finish_record_safely(
                record,
                instance,
                total_seconds,
                resources,
                failure,
            )

    def _begin_record(self, instance: Any, backend: str) -> dict[str, Any]:
        model_path = Path(str(getattr(instance, "model_path", "unknown-model")))
        model_name = model_path.name or str(getattr(instance, "model_basename", "unknown-model"))
        precision = str(getattr(getattr(instance, "model_data", None), "model_precision", "unknown"))
        device_index = selected_device_index(instance)
        normalized_model = os.path.normcase(os.path.abspath(str(model_path)))
        signature = (normalized_model, precision)
        with self._lock:
            self._inference_index += 1
            inference_index = self._inference_index
            if signature == self._last_model_signature:
                self._model_round += 1
            else:
                self._last_model_signature = signature
                self._model_round = 1
            model_round = self._model_round
            self._ensure_session()

        environment = self._environment(device_index)
        input_metadata = read_input_metadata(getattr(instance, "audio_file", None))
        model_hash = self._model_hash(model_path)
        cache_state = model_cache_state(self.namespace.get("MODEL_CACHE"), model_path, precision)
        file_slug = slugify(f"{model_path.stem}-{precision}")
        text_path = self._session_directory / f"inference_{inference_index:04d}_{file_slug}.log"
        json_path = self._session_directory / f"inference_{inference_index:04d}_{file_slug}.json"
        record = {
            "inference_index": inference_index,
            "inference_label": f"{ordinal(inference_index)} Inference",
            "model": model_name,
            "model_sha256": model_hash,
            "model_round": model_round,
            "backend": backend,
            "precision": precision,
            "device_index": device_index,
            "cache_state": cache_state,
            "os": environment.get("hardware", {}).get("os") or platform.platform(),
            "environment": environment,
            "input": input_metadata,
            "started_utc": utc_now(),
            "status": "running",
            "total_time_seconds": None,
            "stages": {},
            "resources": {},
            "runtime": {},
            "outputs": {},
            "warnings": [],
            "error": None,
            "text_path": text_path,
            "json_path": json_path,
        }
        header = [
            f"{record['inference_label']}:",
            f"Model: {model_name}",
            f"Model round: {model_round}",
            f"OS: {record['os']}",
            f"Backend: {backend}",
            f"Precision: {precision}",
            f"Model cache: {cache_state}",
            f"Model SHA-256: {model_hash or 'unavailable'}",
            f"Input: {input_metadata.get('name') or 'unavailable'}",
            f"Input duration: {format_seconds(input_metadata.get('duration_seconds'))}",
            f"Started UTC: {record['started_utc']}",
            "Status: RUNNING",
        ]
        text_path.write_text("\n".join(header) + "\n", encoding="utf-8")
        with self._session_log.open("a", encoding="utf-8") as stream:
            stream.write("\n".join(header) + "\n")
        self._checkpoint(record)
        relative_log = self._session_log.relative_to(self.output_root.parent)
        self._notify_gui(instance, f"[Benchmark] Recording {record['inference_label']} -> {relative_log}")
        return record

    def _finish_record(
        self,
        record: dict[str, Any],
        instance: Any,
        total_seconds: float,
        resources: dict[str, Any],
        failure: BaseException | None,
    ) -> None:
        record["status"] = "completed" if failure is None else "failed"
        record["total_time_seconds"] = total_seconds
        record["ended_utc"] = utc_now()
        record["resources"] = resources
        for warning in resources.get("warnings", []):
            self._record_warning(record, warning)
        try:
            record["runtime"] = effective_runtime(self.namespace, instance)
        except Exception as error:
            record["runtime"] = {}
            self._record_warning(record, f"Runtime metadata unavailable: {type(error).__name__}")
        try:
            record["outputs"] = output_metadata(instance)
        except Exception as error:
            record["outputs"] = {"count": 0, "validation": "unavailable"}
            self._record_warning(record, f"Output metadata unavailable: {type(error).__name__}")
        if failure is not None:
            record["error"] = f"{type(failure).__name__}: {sanitize_text(str(failure))}"
        final_lines = [
            f"Effective precision: {record['runtime'].get('effective_precision') or 'unavailable'}",
            f"Execution provider: {', '.join(record['runtime'].get('execution_provider') or []) or 'unavailable'}",
            f"Peak process RAM: {format_mb(resources.get('peak_process_ram_mb'))}",
            f"Peak device VRAM: {format_mb(resources.get('peak_device_vram_mb'))}",
            f"Peak Torch allocated VRAM: {format_mb(resources.get('torch_peak_allocated_vram_mb'))}",
            f"Peak Torch reserved VRAM: {format_mb(resources.get('torch_peak_reserved_vram_mb'))}",
            f"Average GPU utilization: {format_percent(resources.get('average_gpu_utilization_percent'))}",
            f"Peak GPU utilization: {format_percent(resources.get('peak_gpu_utilization_percent'))}",
            f"Average process CPU: {format_percent(resources.get('average_process_cpu_percent'))}",
            f"Peak process CPU: {format_percent(resources.get('peak_process_cpu_percent'))}",
            f"Output files: {record['outputs'].get('count', 0)}",
            f"Output validation: {record['outputs'].get('validation', 'unavailable')}",
            f"Status: {record['status'].upper()}",
        ]
        if record["error"]:
            final_lines.append(f"Error: {record['error']}")
        final_lines.extend([
            f"Total-time: {total_seconds:.6f} s",
            f"Ended UTC: {record['ended_utc']}",
            "=======",
        ])
        for line in final_lines:
            self._append(record, line, checkpoint=False)
        self._checkpoint(record)
        self._notify_gui(
            instance,
            f"[Benchmark] {record['inference_label']} {record['status']} in {total_seconds:.3f}s",
        )

    def _finish_record_safely(
        self,
        record: dict[str, Any],
        instance: Any,
        total_seconds: float,
        resources: dict[str, Any],
        failure: BaseException | None,
    ) -> None:
        try:
            self._finish_record(record, instance, total_seconds, resources, failure)
        except Exception as recorder_error:
            self._notify_gui(
                instance,
                f"[Benchmark warning] Recorder finalization failed: {type(recorder_error).__name__}",
            )

    def _record_stage(self, record: dict[str, Any], name: str, value: float) -> None:
        with self._lock:
            is_cumulative = name in record["stages"]
            record["stages"][name] = value
            label = f"Stage {name} (cumulative)" if is_cumulative else f"Stage {name}"
            self._append(record, f"{label}: {value:.6f} s")

    def _record_progress(self, record: dict[str, Any], snapshot: dict[str, Any]) -> None:
        with self._lock:
            record["resources"].update(snapshot)
            line = (
                f"Progress: elapsed={snapshot.get('elapsed_seconds', 0.0):.1f}s, "
                f"CPU={format_percent(snapshot.get('process_cpu_percent'))}, "
                f"GPU={format_percent(snapshot.get('gpu_utilization_percent'))}, "
                f"device VRAM={format_mb(snapshot.get('device_vram_mb'))}"
            )
            self._append(record, line)

    def _record_warning(self, record: dict[str, Any], warning: str) -> None:
        cleaned = sanitize_text(str(warning))
        with self._lock:
            if cleaned in record["warnings"]:
                return
            record["warnings"].append(cleaned)
            self._append(record, f"Warning: {cleaned}")

    def _append(self, record: dict[str, Any], line: str, checkpoint: bool = True) -> None:
        with self._lock:
            for path in (record["text_path"], self._session_log):
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
            if checkpoint:
                self._checkpoint(record)

    def _checkpoint(self, record: dict[str, Any]) -> None:
        write_json(record["json_path"], public_record(record))

    def _ensure_session(self) -> None:
        if self._session_directory is not None:
            return
        session_name = datetime.now(timezone.utc).strftime("session_%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
        self._session_directory = self.output_root / session_name
        self._session_directory.mkdir(parents=True, exist_ok=True)
        self._session_log = self._session_directory / "benchmark.log"
        uvr_path = Path(__file__).resolve().parents[1] / "UVR.py"
        uvr_hash = sha256_file(uvr_path) if uvr_path.is_file() else "unavailable"
        self._session_log.write_text(
            "UVR GUI Backend Benchmark\n"
            f"Session started UTC: {utc_now()}\n"
            f"UVR.py SHA-256: {uvr_hash}\n"
            "=======\n",
            encoding="utf-8",
        )
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "latest.txt").write_text(session_name + "\n", encoding="utf-8")

    def _environment(self, device_index: int) -> dict[str, Any]:
        if device_index not in self._environment_cache:
            try:
                self._environment_cache[device_index] = self.environment_collector(device_index)
            except Exception as error:
                self._environment_cache[device_index] = {
                    "hardware": {"os": platform.platform()},
                    "software": {},
                    "warnings": [f"Environment metadata unavailable: {type(error).__name__}"],
                }
        return self._environment_cache[device_index]

    def _model_hash(self, model_path: Path) -> str | None:
        try:
            stat = model_path.stat()
            key = (os.path.normcase(str(model_path.resolve())), stat.st_size, stat.st_mtime_ns)
            if key not in self._model_hash_cache:
                self._model_hash_cache[key] = sha256_file(model_path)
            return self._model_hash_cache[key]
        except Exception:
            return None

    @staticmethod
    def _notify_gui(instance: Any, message: str) -> None:
        writer = getattr(instance, "write_to_console", None)
        if not callable(writer):
            return
        try:
            writer(message, base_text="")
        except TypeError:
            try:
                writer(message)
            except Exception:
                pass
        except Exception:
            pass


class RuntimeStageInstrumentor:
    def __init__(
        self,
        namespace: dict[str, Any],
        instance: Any,
        device_index: int,
        stage_callback: Callable[[str, float], None],
    ):
        self.namespace = namespace
        self.instance = instance
        self.device_index = device_index
        self.stage_callback = stage_callback
        self.values: dict[str, float] = {}
        self._restores: list[Callable[[], None]] = []
        self._model_load_started: float | None = None

    def install(self) -> None:
        model_cache = self.namespace.get("MODEL_CACHE")
        if model_cache is not None:
            original_get = model_cache.get_model
            original_set = model_cache.set_model

            def get_model(*args, **kwargs):
                model = original_get(*args, **kwargs)
                if model is None and self._model_load_started is None:
                    synchronize_cuda(self.device_index)
                    self._model_load_started = time.perf_counter()
                return model

            def set_model(*args, **kwargs):
                result = original_set(*args, **kwargs)
                self._finish_model_loading()
                return result

            self._patch_attribute(model_cache, "get_model", get_model)
            self._patch_attribute(model_cache, "set_model", set_model)

        backend_name = type(self.instance).__name__
        if backend_name in {"SeperateMDX", "SeperateMDXC", "SeperateDemucs"}:
            self._patch_namespace("prepare_mix", self._timed(self.namespace["prepare_mix"], "preprocessing"))
        if backend_name in {"SeperateMDX", "SeperateMDXC"}:
            self._patch_attribute(self.instance, "demix", self._timed(self.instance.demix, "inference"))
        elif backend_name == "SeperateDemucs":
            self._patch_attribute(
                self.instance,
                "demix_demucs",
                self._timed(self.instance.demix_demucs, "inference"),
            )
        elif backend_name == "SeperateVR":
            self._patch_attribute(self.instance, "loading_mix", self._timed(self.instance.loading_mix, "preprocessing"))
            self._patch_attribute(self.instance, "inference_vr", self._timed(self.instance.inference_vr, "inference"))
        if hasattr(self.instance, "write_audio"):
            self._patch_attribute(self.instance, "write_audio", self._timed(self.instance.write_audio, "output_writing"))

    def finish(self) -> None:
        self._finish_model_loading()

    def restore(self) -> None:
        errors = []
        for restore in reversed(self._restores):
            try:
                restore()
            except Exception as error:
                errors.append(error)
        self._restores.clear()
        if errors:
            raise RuntimeError(f"failed to restore {len(errors)} benchmark hook(s)") from errors[0]

    def _timed(self, function: Callable[..., Any], stage_name: str) -> Callable[..., Any]:
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            synchronize_cuda(self.device_index)
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                synchronize_cuda(self.device_index)
                elapsed = time.perf_counter() - started
                self.values[stage_name] = self.values.get(stage_name, 0.0) + elapsed
                self.stage_callback(stage_name, self.values[stage_name])

        return wrapped

    def _finish_model_loading(self) -> None:
        if self._model_load_started is None:
            return
        synchronize_cuda(self.device_index)
        elapsed = time.perf_counter() - self._model_load_started
        self._model_load_started = None
        self.values["model_loading"] = self.values.get("model_loading", 0.0) + elapsed
        self.stage_callback("model_loading", self.values["model_loading"])

    def _patch_namespace(self, name: str, replacement: Any) -> None:
        original = self.namespace[name]
        self.namespace[name] = replacement
        self._restores.append(lambda: self.namespace.__setitem__(name, original))

    def _patch_attribute(self, owner: Any, name: str, replacement: Any) -> None:
        owner_dict = getattr(owner, "__dict__", {})
        had_local_value = name in owner_dict
        local_value = owner_dict.get(name)
        setattr(owner, name, replacement)

        def restore() -> None:
            if had_local_value:
                setattr(owner, name, local_value)
            else:
                try:
                    delattr(owner, name)
                except AttributeError:
                    pass

        self._restores.append(restore)


class RuntimeResourceSampler:
    def __init__(
        self,
        device_index: int = 0,
        interval_seconds: float = 0.25,
        progress_interval_seconds: float = 5.0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.device_index = device_index
        self.interval_seconds = max(float(interval_seconds), 0.05)
        self.progress_interval_seconds = max(float(progress_interval_seconds), self.interval_seconds)
        self.progress = progress or (lambda snapshot: None)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._last_progress = 0.0
        self._psutil: Any = None
        self._process: Any = None
        self._nvml: Any = None
        self._handle: Any = None
        self._sample_count = 0
        self._process_cpu_total = 0.0
        self._process_cpu_peak: float | None = None
        self._system_cpu_total = 0.0
        self._system_cpu_peak: float | None = None
        self._process_ram_peak: float | None = None
        self._gpu_util_total = 0.0
        self._gpu_util_peak: float | None = None
        self._device_vram_baseline: float | None = None
        self._device_vram_peak: float | None = None
        self._power_total = 0.0
        self._power_peak: float | None = None
        self._power_samples = 0
        self.warnings: list[str] = []

    def start(self) -> None:
        self._setup_psutil()
        self._setup_nvml()
        reset_torch_peak_memory(self.device_index, self.warnings)
        self._started = time.perf_counter()
        self._last_progress = self._started
        self._sample(emit_progress=False)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="uvr-gui-benchmark-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4))
        self._sample(emit_progress=False)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        torch_memory = torch_peak_memory(self.device_index)
        count = max(self._sample_count, 1)
        return {
            "sampling_interval_ms": self.interval_seconds * 1000.0,
            "average_process_cpu_percent": self._process_cpu_total / count if self._process_cpu_peak is not None else None,
            "peak_process_cpu_percent": self._process_cpu_peak,
            "average_system_cpu_percent": self._system_cpu_total / count if self._system_cpu_peak is not None else None,
            "peak_system_cpu_percent": self._system_cpu_peak,
            "peak_process_ram_mb": self._process_ram_peak,
            "average_gpu_utilization_percent": self._gpu_util_total / count if self._gpu_util_peak is not None else None,
            "peak_gpu_utilization_percent": self._gpu_util_peak,
            "baseline_device_vram_mb": self._device_vram_baseline,
            "peak_device_vram_mb": self._device_vram_peak,
            "average_power_w": self._power_total / self._power_samples if self._power_samples else None,
            "peak_power_w": self._power_peak,
            **torch_memory,
            "warnings": list(dict.fromkeys(self.warnings)),
        }

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample(emit_progress=True)

    def _sample(self, emit_progress: bool) -> None:
        snapshot: dict[str, Any] = {
            "elapsed_seconds": max(0.0, time.perf_counter() - self._started) if self._started else 0.0,
            "process_cpu_percent": None,
            "gpu_utilization_percent": None,
            "device_vram_mb": None,
        }
        if self._process is not None:
            try:
                process_cpu = float(self._process.cpu_percent(None))
                system_cpu = float(self._psutil.cpu_percent(None))
                process_ram = self._process.memory_info().rss / (1024 * 1024)
                self._process_cpu_total += process_cpu
                self._process_cpu_peak = max(self._process_cpu_peak or 0.0, process_cpu)
                self._system_cpu_total += system_cpu
                self._system_cpu_peak = max(self._system_cpu_peak or 0.0, system_cpu)
                self._process_ram_peak = max(self._process_ram_peak or 0.0, process_ram)
                snapshot["process_cpu_percent"] = process_cpu
            except Exception as error:
                self._warn_once(f"CPU/RAM sampling stopped: {type(error).__name__}")
                self._process = None
        if self._handle is not None:
            try:
                utilization = float(self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
                device_vram = self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / (1024 * 1024)
                self._gpu_util_total += utilization
                self._gpu_util_peak = max(self._gpu_util_peak or 0.0, utilization)
                if self._device_vram_baseline is None:
                    self._device_vram_baseline = device_vram
                self._device_vram_peak = max(self._device_vram_peak or 0.0, device_vram)
                snapshot["gpu_utilization_percent"] = utilization
                snapshot["device_vram_mb"] = device_vram
                try:
                    power = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                    self._power_total += power
                    self._power_samples += 1
                    self._power_peak = max(self._power_peak or 0.0, power)
                except Exception:
                    pass
            except Exception as error:
                self._warn_once(f"GPU sampling stopped: {type(error).__name__}")
                self._handle = None
        self._sample_count += 1
        now = time.perf_counter()
        if emit_progress and now - self._last_progress >= self.progress_interval_seconds:
            self._last_progress = now
            try:
                self.progress(snapshot)
            except Exception:
                pass

    def _setup_psutil(self) -> None:
        try:
            import psutil

            self._psutil = psutil
            self._process = psutil.Process(os.getpid())
            self._process.cpu_percent(None)
            psutil.cpu_percent(None)
        except Exception as error:
            self._warn_once(f"CPU/RAM sampling unavailable: {type(error).__name__}")

    def _setup_nvml(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle, mapped_by_pci = nvml_handle_for_torch_device(pynvml, self.device_index)
            if not mapped_by_pci:
                self._warn_once("NVML used device-index fallback")
        except Exception as error:
            self._warn_once(f"GPU sampling unavailable: {type(error).__name__}")
            self._handle = None

    def _warn_once(self, warning: str) -> None:
        if warning not in self.warnings:
            self.warnings.append(warning)


def public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"text_path", "json_path"}
    }


def selected_device_index(instance: Any) -> int:
    candidates = (
        getattr(getattr(instance, "model_data", None), "device_set", None),
        getattr(instance, "device", None),
    )
    for candidate in candidates:
        text = str(candidate or "")
        if text.isdigit():
            return int(text)
        if text.startswith("cuda:") and text.split(":", 1)[1].isdigit():
            return int(text.split(":", 1)[1])
    return 0


def read_input_metadata(value: Any) -> dict[str, Any]:
    if not value:
        return {"name": None, "duration_seconds": None, "sample_rate": None, "channels": None}
    path = Path(str(value))
    metadata = {"name": path.name, "duration_seconds": None, "sample_rate": None, "channels": None}
    try:
        import soundfile as sf

        info = sf.info(path)
        metadata.update({
            "duration_seconds": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
        })
    except Exception:
        pass
    return metadata


def model_cache_state(model_cache: Any, model_path: Path, precision: str) -> str:
    if model_cache is None:
        return "unknown"
    normalized = os.path.normpath(os.path.abspath(str(model_path)))
    entry = getattr(model_cache, "cache", {}).get(normalized)
    if not entry:
        return "cold"
    cached_precision = entry.get("precision_mode")
    return "warm" if cached_precision in (None, precision) else "cold-precision-change"


def effective_runtime(namespace: dict[str, Any], instance: Any) -> dict[str, Any]:
    model_path = Path(str(getattr(instance, "model_path", "")))
    model_cache = namespace.get("MODEL_CACHE")
    normalized = os.path.normpath(os.path.abspath(str(model_path)))
    entry = getattr(model_cache, "cache", {}).get(normalized, {}) if model_cache is not None else {}
    model = entry.get("model_obj")
    if model is None:
        model = getattr(instance, "model_run", None)
    if model is None:
        model = getattr(instance, "demucs", None)
    if hasattr(model, "get_inputs"):
        input_type = model.get_inputs()[0].type
        return {
            "effective_precision": "fp16" if "float16" in input_type else "fp32",
            "execution_provider": list(model.get_providers()),
            "onnx_input_type": input_type,
        }
    try:
        import torch

        if isinstance(model, torch.nn.Module):
            dtypes = {parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()}
            mapping = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}
            effective = {mapping[dtype] for dtype in dtypes if dtype in mapping}
            return {
                "effective_precision": effective.pop() if len(effective) == 1 else "unknown",
                "execution_provider": ["CUDA"] if str(getattr(instance, "device", "")).startswith("cuda") else ["CPU"],
            }
    except Exception:
        pass
    providers = []
    for provider in getattr(instance, "run_type", []) or []:
        providers.append(str(provider[0] if isinstance(provider, tuple) else provider))
    return {"effective_precision": "unknown", "execution_provider": providers}


def output_metadata(instance: Any) -> dict[str, Any]:
    export_path = Path(str(getattr(instance, "export_path", "")))
    base = str(getattr(instance, "audio_file_base", ""))
    if not export_path.is_dir() or not base:
        return {"count": 0, "total_size_mb": 0.0, "validation": "unavailable"}
    files = sorted(path for path in export_path.iterdir() if path.is_file() and path.name.startswith(base))
    readable_audio = 0
    invalid_audio = 0
    durations = []
    for path in files:
        if path.suffix.lower() not in {".wav", ".flac", ".mp3"}:
            continue
        try:
            import soundfile as sf

            info = sf.info(path)
            if info.frames > 0 and info.samplerate > 0 and info.channels > 0:
                readable_audio += 1
                durations.append(info.duration)
            else:
                invalid_audio += 1
        except Exception:
            invalid_audio += 1
    validation = "passed" if readable_audio and invalid_audio == 0 else "failed" if invalid_audio else "unavailable"
    return {
        "count": len(files),
        "readable_audio_count": readable_audio,
        "invalid_audio_count": invalid_audio,
        "duration_seconds": durations,
        "total_size_mb": sum(path.stat().st_size for path in files) / (1024 * 1024),
        "validation": validation,
    }


def reset_torch_peak_memory(device_index: int, warnings: list[str]) -> None:
    try:
        import torch

        if torch.cuda.is_available() and 0 <= device_index < torch.cuda.device_count():
            torch.cuda.reset_peak_memory_stats(device_index)
    except Exception as error:
        warnings.append(f"Torch peak reset unavailable: {type(error).__name__}")


def torch_peak_memory(device_index: int) -> dict[str, float | None]:
    try:
        import torch

        if torch.cuda.is_available() and 0 <= device_index < torch.cuda.device_count():
            return {
                "torch_peak_allocated_vram_mb": torch.cuda.max_memory_allocated(device_index) / (1024 * 1024),
                "torch_peak_reserved_vram_mb": torch.cuda.max_memory_reserved(device_index) / (1024 * 1024),
            }
    except Exception:
        pass
    return {"torch_peak_allocated_vram_mb": None, "torch_peak_reserved_vram_mb": None}


def synchronize_cuda(device_index: int) -> None:
    try:
        import torch

        if torch.cuda.is_available() and 0 <= device_index < torch.cuda.device_count():
            torch.cuda.synchronize(device_index)
    except Exception:
        pass


def ordinal(value: int) -> str:
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def format_seconds(value: Any) -> str:
    return "unavailable" if value is None else f"{float(value):.6f} s"


def format_mb(value: Any) -> str:
    return "unavailable" if value is None else f"{float(value):.2f} MiB"


def format_percent(value: Any) -> str:
    return "unavailable" if value is None else f"{float(value):.2f}%"
