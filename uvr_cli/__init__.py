"""Ultimate Vocal Remover - CLI-Only Inference & Real Performance Benchmarking Suite."""

from __future__ import annotations

from .engine import BenchmarkEngine
from .exporter import export_benchmark_csv, export_benchmark_json, sanitize_path
from .model_resolver import (
    HeadlessModelData,
    build_headless_model_data,
    compute_uvr_hash,
    list_available_models,
    resolve_model_file,
)
from .runner import (
    execute_inference,
    get_audio_metadata,
    get_existing_outputs,
    get_expected_stem_paths,
)
from .telemetry import TelemetrySampler, collect_system_info, sync_cuda

__version__ = "2.0.0"

__all__ = [
    "BenchmarkEngine",
    "HeadlessModelData",
    "TelemetrySampler",
    "build_headless_model_data",
    "collect_system_info",
    "compute_uvr_hash",
    "execute_inference",
    "export_benchmark_csv",
    "export_benchmark_json",
    "get_audio_metadata",
    "get_existing_outputs",
    "get_expected_stem_paths",
    "list_available_models",
    "resolve_model_file",
    "sanitize_path",
    "sync_cuda",
    "__version__",
]
