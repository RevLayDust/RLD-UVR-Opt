"""Session creation and lifecycle management for TensorRT-RTX with graceful fallback."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .environment import (
    PROVIDER_CPU,
    PROVIDER_CUDA,
    PROVIDER_TRT_PLUGIN,
    PROVIDER_TRT_RTX,
    register_trt_rtx_libraries,
)

DEFAULT_CUDA_OPTIONS = {
    "arena_extend_strategy": "kSameAsRequested",
    "cudnn_conv_algo_search": "HEURISTIC",
    "do_copy_in_default_stream": "True",
}


def get_default_cache_dir() -> Path:
    """Return default directory for cached TensorRT engine plans."""
    project_root = Path(__file__).resolve().parents[2]
    cache_dir = project_root / "models" / "MDX_Net_Models" / "trt_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def build_trt_rtx_options(
    device_id: int = 0,
    cache_dir: Optional[Union[str, Path]] = None,
    enable_cuda_graph: bool = False,
) -> Dict[str, str]:
    """Construct configuration options dictionary for TensorRT-RTX Execution Provider."""
    target_cache = Path(cache_dir) if cache_dir else get_default_cache_dir()
    target_cache.mkdir(parents=True, exist_ok=True)

    opts = {
        "device_id": str(int(device_id)),
        "nv_runtime_cache_path": str(target_cache),
    }
    if enable_cuda_graph:
        opts["enable_cuda_graph"] = "1"
    return opts


# Backward-compatibility alias
build_trt_options = build_trt_rtx_options


def create_trt_rtx_session(
    model_input: Union[str, Path, bytes],
    device_id: int = 0,
    cuda_options: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    model_name: Optional[str] = None,
    enable_cuda_graph: bool = False,
) -> Any:
    """Create an ONNX Runtime InferenceSession preferring TensorRT-RTX with automatic fallback.

    Fallback chain:
        1. TensorRT-RTX EP (via EP plugin or built-in provider)
        2. CUDAExecutionProvider
        3. CPUExecutionProvider
    """
    import onnxruntime as ort

    register_trt_rtx_libraries()
    cuda_opts = cuda_options or DEFAULT_CUDA_OPTIONS
    trt_opts = build_trt_rtx_options(
        device_id=device_id,
        cache_dir=cache_dir,
        enable_cuda_graph=enable_cuda_graph,
    )

    if model_name:
        display_name = model_name
    elif isinstance(model_input, (str, Path)):
        display_name = os.path.basename(str(model_input))
    else:
        display_name = f"in-memory-model ({len(model_input) / (1024 ** 2):.2f} MiB)"

    # Strategy 1: Standalone EP ABI plugin (onnxruntime-ep-nv-tensorrt-rtx)
    try:
        if hasattr(ort, "get_ep_devices"):
            import onnxruntime_ep_nv_tensorrt_rtx as trt_ep
            plugin_ep_name = trt_ep.get_ep_name()
            ep_devices = [
                d for d in ort.get_ep_devices()
                if getattr(d, "ep_name", "") == plugin_ep_name
            ]
            if ep_devices:
                # Select the first matching TensorRT-RTX EP device
                target_dev = ep_devices[0]
                print(f"[BACKEND] Initializing TensorRT-RTX plugin session for {display_name}...")
                so = ort.SessionOptions()
                so.add_provider_for_devices([target_dev], trt_opts)
                session = ort.InferenceSession(model_input, sess_options=so)
                active = session.get_providers()[0] if hasattr(session, "get_providers") else PROVIDER_TRT_RTX
                print(f"[BACKEND] Active execution provider: {active}")
                return session
    except Exception as exc:
        print(f"[BACKEND] TensorRT-RTX plugin session creation failed ({type(exc).__name__}: {exc})")
        print("[BACKEND] Falling back to CUDAExecutionProvider...")

    # Strategy 2: Built-in NvTensorRTRTXExecutionProvider (if present)
    avail_provs = ort.get_available_providers()
    if PROVIDER_TRT_RTX in avail_provs:
        try:
            print(f"[BACKEND] Initializing built-in {PROVIDER_TRT_RTX} session for {display_name}...")
            providers: List[Union[str, Tuple[str, Dict[str, Any]]]] = [
                (PROVIDER_TRT_RTX, trt_opts),
                (PROVIDER_CUDA, cuda_opts),
                PROVIDER_CPU,
            ]
            session = ort.InferenceSession(model_input, providers=providers)
            active = session.get_providers()[0] if hasattr(session, "get_providers") else PROVIDER_TRT_RTX
            print(f"[BACKEND] Active execution provider: {active}")
            return session
        except Exception as exc:
            print(f"[BACKEND] {PROVIDER_TRT_RTX} session creation failed ({type(exc).__name__}: {exc})")
            print("[BACKEND] Falling back to CUDAExecutionProvider...")

    # Strategy 3: CUDAExecutionProvider fallback
    if PROVIDER_CUDA in avail_provs:
        try:
            print(f"[BACKEND] Creating CUDAExecutionProvider session for {display_name}...")
            session = ort.InferenceSession(
                model_input,
                providers=[(PROVIDER_CUDA, cuda_opts), PROVIDER_CPU],
            )
            active = session.get_providers()[0] if hasattr(session, "get_providers") else PROVIDER_CUDA
            print(f"[BACKEND] Active execution provider: {active}")
            return session
        except Exception as exc:
            print(f"[BACKEND] CUDAExecutionProvider failed ({type(exc).__name__}: {exc})")
            print("[BACKEND] Falling back to CPUExecutionProvider...")

    # Strategy 4: CPUExecutionProvider fallback
    print(f"[BACKEND] Creating CPUExecutionProvider session for {display_name}...")
    session = ort.InferenceSession(model_input, providers=[PROVIDER_CPU])
    print(f"[BACKEND] Active execution provider: {PROVIDER_CPU}")
    return session
