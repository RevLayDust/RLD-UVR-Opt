"""Ultimate Vocal Remover - CLI-Only Inference & Real Performance Benchmarking Suite."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "2.0.0"

_LAZY_EXPORTS = {
    "BenchmarkEngine": (".engine", "BenchmarkEngine"),
    "HeadlessModelData": (".model_resolver", "HeadlessModelData"),
    "InferenceProgressBar": (".progress", "InferenceProgressBar"),
    "InferenceResult": (".runner", "InferenceResult"),
    "TelemetrySampler": (".telemetry", "TelemetrySampler"),
    "build_headless_model_data": (".model_resolver", "build_headless_model_data"),
    "collect_system_info": (".telemetry", "collect_system_info"),
    "compute_uvr_hash": (".model_resolver", "compute_uvr_hash"),
    "execute_inference": (".runner", "execute_inference"),
    "export_benchmark_csv": (".exporter", "export_benchmark_csv"),
    "export_benchmark_json": (".exporter", "export_benchmark_json"),
    "get_audio_metadata": (".runner", "get_audio_metadata"),
    "get_cpu_name": (".telemetry", "get_cpu_name"),
    "get_existing_outputs": (".runner", "get_existing_outputs"),
    "get_expected_stem_paths": (".runner", "get_expected_stem_paths"),
    "list_available_models": (".model_resolver", "list_available_models"),
    "resolve_model_file": (".model_resolver", "resolve_model_file"),
    "sanitize_path": (".exporter", "sanitize_path"),
    "sync_cuda": (".telemetry", "sync_cuda"),
}

__all__ = sorted(list(_LAZY_EXPORTS.keys()) + ["__version__"])

if TYPE_CHECKING:
    from .engine import BenchmarkEngine
    from .exporter import export_benchmark_csv, export_benchmark_json, sanitize_path
    from .model_resolver import (
        HeadlessModelData,
        build_headless_model_data,
        compute_uvr_hash,
        list_available_models,
        resolve_model_file,
    )
    from .progress import InferenceProgressBar
    from .runner import (
        InferenceResult,
        execute_inference,
        get_audio_metadata,
        get_existing_outputs,
        get_expected_stem_paths,
    )
    from .telemetry import TelemetrySampler, collect_system_info, get_cpu_name, sync_cuda


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_path, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_path, __package__)
        val = getattr(module, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
