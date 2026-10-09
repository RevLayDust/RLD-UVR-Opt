"""TensorRT-RTX execution provider wrapper package for Ultimate Vocal Remover."""

from __future__ import annotations

from .environment import (
    PROVIDER_CPU,
    PROVIDER_CUDA,
    PROVIDER_TRT_PLUGIN,
    PROVIDER_TRT_RTX,
    get_supported_execution_providers,
    is_ort_gpu_available,
    is_trt_rtx_available,
    is_trt_rtx_installed,
    register_trt_rtx_libraries,
)
from .session import (
    DEFAULT_CUDA_OPTIONS,
    build_trt_options,
    build_trt_rtx_options,
    create_trt_rtx_session,
    get_default_cache_dir,
)

__all__ = [
    "PROVIDER_CPU",
    "PROVIDER_CUDA",
    "PROVIDER_TRT_PLUGIN",
    "PROVIDER_TRT_RTX",
    "DEFAULT_CUDA_OPTIONS",
    "is_trt_rtx_installed",
    "is_trt_rtx_available",
    "is_ort_gpu_available",
    "get_supported_execution_providers",
    "build_trt_options",
    "build_trt_rtx_options",
    "create_trt_rtx_session",
    "get_default_cache_dir",
    "register_trt_rtx_libraries",
]
