"""Environment detection and runtime provider discovery for TensorRT-RTX."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import torch

# Execution Provider names
PROVIDER_TRT_RTX = "NvTensorRTRTXExecutionProvider"
PROVIDER_TRT_PLUGIN = "nv_tensorrt_rtx"
PROVIDER_CUDA = "CUDAExecutionProvider"
PROVIDER_CPU = "CPUExecutionProvider"

_PLUGIN_REGISTERED = False


def register_trt_rtx_libraries() -> None:
    """Register TensorRT-RTX DLL directories and plugin library with ONNX Runtime."""
    global _PLUGIN_REGISTERED
    if _PLUGIN_REGISTERED:
        return

    # 1. On Windows, ensure bundled pip plugin library DLL directory is in search path
    if sys.platform == "win32":
        try:
            import site
            for site_dir in site.getsitepackages():
                rtx_dir = os.path.join(site_dir, "onnxruntime_ep_nv_tensorrt_rtx")
                if os.path.isdir(rtx_dir):
                    try:
                        os.add_dll_directory(rtx_dir)
                    except (AttributeError, OSError) as exc:
                        print(f"[BACKEND] Warning: Failed to add DLL directory '{rtx_dir}': {exc}")
                    os.environ["PATH"] = rtx_dir + os.pathsep + os.environ.get("PATH", "")
        except Exception as exc:
            print(f"[BACKEND] Warning: Error configuring Windows DLL search paths: {exc}")

    # 2. Register the standalone EP ABI plugin with ONNX Runtime
    try:
        import onnxruntime as ort
        if hasattr(ort, "register_execution_provider_library"):
            try:
                import onnxruntime_ep_nv_tensorrt_rtx as trt_ep
                ep_name = trt_ep.get_ep_name()
                ep_lib = trt_ep.get_library_path()
                print(f"[BACKEND] Registering TensorRT-RTX execution provider library '{ep_name}' ({ep_lib})...")
                ort.register_execution_provider_library(ep_name, ep_lib)
                print(f"[BACKEND] Successfully registered TensorRT-RTX execution provider '{ep_name}'.")
            except ImportError:
                # Standalone plugin package is not installed; normal for standard/CPU installs
                pass
            except Exception as exc:
                print(f"[BACKEND] Warning: Failed to register TensorRT-RTX library with ONNX Runtime: {exc}")
    except Exception as exc:
        print(f"[BACKEND] Warning: Error checking ONNX Runtime plugin support: {exc}")

    _PLUGIN_REGISTERED = True


def is_trt_rtx_installed() -> bool:
    """Check if TensorRT-RTX EP is genuinely registered and available in ONNX Runtime.

    Standard TensorrtExecutionProvider is deliberately NOT treated as TensorRT-RTX.
    """
    register_trt_rtx_libraries()
    try:
        import onnxruntime as ort
        all_provs = ort.get_all_providers() if hasattr(ort, "get_all_providers") else []
        avail_provs = ort.get_available_providers()

        # Check standalone EP ABI plugin via device enumeration
        if hasattr(ort, "get_ep_devices"):
            try:
                import onnxruntime_ep_nv_tensorrt_rtx as trt_ep
                plugin_name = trt_ep.get_ep_name()
                devices = [d for d in ort.get_ep_devices() if getattr(d, "ep_name", "") == plugin_name]
                if devices:
                    return True
            except ImportError:
                pass
            except Exception as exc:
                print(f"[BACKEND] Error inspecting TensorRT-RTX EP devices: {exc}")

        # Check built-in provider name (NvTensorRTRTXExecutionProvider)
        if PROVIDER_TRT_RTX in all_provs and PROVIDER_TRT_RTX in avail_provs:
            return True
    except Exception as exc:
        print(f"[BACKEND] Error checking TensorRT-RTX installation: {exc}")

    return False


def is_trt_rtx_available(device_id: int = 0) -> bool:
    """Check whether TensorRT-RTX can genuinely execute on the specified GPU device."""
    if not torch.cuda.is_available():
        return False
    if device_id < 0 or device_id >= torch.cuda.device_count():
        return False
    return is_trt_rtx_installed()


def is_ort_gpu_available() -> bool:
    """Check whether ONNX Runtime supports any GPU execution provider."""
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
        gpu_providers = (
            PROVIDER_CUDA,
            "ROCMExecutionProvider",
            "MIGraphXExecutionProvider",
            "DmlExecutionProvider",
        )
        if any(p in avail for p in gpu_providers):
            return True
        if is_trt_rtx_installed():
            return True
    except Exception as exc:
        print(f"[BACKEND] Warning: Failed to query ONNX Runtime GPU providers: {exc}")
    return False


def get_supported_execution_providers(has_gpu_selected: bool = False, device_id: int = 0) -> List[str]:
    """Return a list of execution providers supported by the current environment and selection.

    Follows GUI Rules:
    - If no GPU is selected (has_gpu_selected=False): returns [CPUExecutionProvider].
    - If GPU is selected: auto-detects providers supported by that GPU (TRT-RTX if available, CUDA, CPU).
    """
    providers: List[str] = []
    if not has_gpu_selected or not torch.cuda.is_available():
        return [PROVIDER_CPU]

    # GPU is selected: check available providers
    register_trt_rtx_libraries()
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()

        # Check TensorRT-RTX capability
        if is_trt_rtx_available(device_id):
            providers.append(PROVIDER_TRT_RTX)

        if PROVIDER_CUDA in avail:
            providers.append(PROVIDER_CUDA)
    except Exception as exc:
        print(f"[BACKEND] Error discovering supported execution providers: {exc}")

    if not providers:
        providers.append(PROVIDER_CPU)

    return providers
