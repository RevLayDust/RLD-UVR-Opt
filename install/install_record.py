"""Persistent install record manager for UVR installation.

Tracks installer mode, detected dependencies, installed package versions,
and user choices to enable seamless switching between modes (CUDA, TensorRT-RTX, CPU).
"""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

INSTALL_DIR = Path(__file__).resolve().parent
ROOT_DIR = INSTALL_DIR.parent
DEFAULT_RECORD_PATH = INSTALL_DIR / "install_record.json"


def get_app_version() -> Dict[str, str]:
    """Extract app version and build ID from __version__.py."""
    version_file = ROOT_DIR / "__version__.py"
    version_info = {"version": "unknown", "build_id": "unknown"}
    if version_file.is_file():
        namespace: Dict[str, Any] = {}
        try:
            with open(version_file, "r", encoding="utf-8") as f:
                exec(f.read(), namespace)
            version_info["version"] = str(namespace.get("VERSION", "unknown"))
            version_info["build_id"] = str(namespace.get("BUILD_ID", "unknown"))
        except Exception:
            pass
    return version_info


def get_package_version(pkg_name: str) -> Optional[str]:
    """Retrieve installed version of a Python package safely."""
    try:
        return importlib.metadata.version(pkg_name)
    except Exception:
        return None


def read_install_record(record_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Read and validate the existing install record. Returns None if absent or invalid."""
    path = record_path or DEFAULT_RECORD_PATH
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "install_mode" in data:
            return data
    except Exception:
        pass
    return None


def inspect_installed_packages() -> Dict[str, Any]:
    """Inspect installed versions of key runtime dependencies."""
    return {
        "pytorch": {
            "torch": get_package_version("torch"),
            "torchvision": get_package_version("torchvision"),
            "torchaudio": get_package_version("torchaudio"),
        },
        "onnxruntime": {
            "onnxruntime": get_package_version("onnxruntime"),
            "onnxruntime_gpu": get_package_version("onnxruntime-gpu"),
        },
        "tensorrt_rtx": {
            "onnxruntime_ep_nv_tensorrt_rtx_cu12": get_package_version("onnxruntime-ep-nv-tensorrt-rtx-cu12"),
            "tensorrt_cu12_libs": get_package_version("tensorrt-cu12-libs"),
        },
    }


def inspect_installed_stack() -> Dict[str, str]:
    """Inspect current environment to determine Torch build type and ORT variant."""
    torch_type = "NONE"
    torch_ver = "NONE"
    torch_cuda = "NONE"
    try:
        import torch
        torch_ver = str(getattr(torch, "__version__", "NONE"))
        cuda_val = getattr(torch.version, "cuda", None)
        if cuda_val is not None:
            torch_type = "CUDA"
            torch_cuda = str(cuda_val)
        elif "+cu" in torch_ver:
            torch_type = "CUDA"
            torch_cuda = "UNKNOWN"
        else:
            torch_type = "CPU"
            torch_cuda = "NONE"
    except Exception:
        pass

    ort_type = "NONE"
    ort_ver = "NONE"
    try:
        import importlib.metadata
        installed = {d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()}
        has_trt_plugin = "onnxruntime-ep-nv-tensorrt-rtx-cu12" in installed
        has_ort_gpu = "onnxruntime-gpu" in installed
        has_ort_cpu = "onnxruntime" in installed

        if has_ort_gpu and has_trt_plugin:
            ort_type = "GPU_TRT"
            ort_ver = installed.get("onnxruntime-gpu", "NONE")
        elif has_ort_gpu:
            ort_type = "GPU_CUDA"
            ort_ver = installed.get("onnxruntime-gpu", "NONE")
        elif has_ort_cpu:
            ort_type = "CPU"
            ort_ver = installed.get("onnxruntime", "NONE")
    except Exception:
        pass

    return {
        "TORCH_TYPE": torch_type,
        "TORCH_VER": torch_ver,
        "TORCH_CUDA": torch_cuda,
        "ORT_TYPE": ort_type,
        "ORT_VER": ort_ver,
    }


def write_install_record(
    option: str,
    install_mode: str,
    reuse_venv: str = "Y",
    has_uv: bool = True,
    has_ffmpeg: bool = True,
    has_rubberband: bool = True,
    has_vc_redist: bool = True,
    record_path: Optional[Path] = None,
) -> Path:
    """Record a successful installation configuration to JSON."""
    path = record_path or DEFAULT_RECORD_PATH
    app_version = get_app_version()
    packages = inspect_installed_packages()

    record: Dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "app_version": app_version["version"],
        "app_build_id": app_version["build_id"],
        "selected_option": option.upper(),
        "install_mode": install_mode.upper(),
        "user_prompts": {
            "reuse_venv": reuse_venv.upper(),
        },
        "system_components": {
            "venv_path": str(ROOT_DIR / "venv"),
            "uv_detected": has_uv,
            "ffmpeg_present": has_ffmpeg,
            "rubberband_present": has_rubberband,
            "vc_redist_detected": has_vc_redist,
        },
        "stack_profile": (
            "tensorrt_rtx_ort124" if option.upper() == "B"
            else ("cpu_ort122" if option.upper() == "C" else "stable_cuda_ort122")
        ),
        "ort_stack_note": (
            "Option B uses onnxruntime-gpu>=1.24.4 for TensorRT-RTX EP ABI plugin compatibility (requires ORT>=1.24); CUDA EP remains available as stable fallback."
            if option.upper() == "B"
            else "Option A/C uses stable benchmarked onnxruntime==1.22.0 stack."
        ),
        "installed_packages": packages,
    }

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
    except Exception as exc:
        print(f"[WARN] Failed to write install record: {exc}", file=sys.stderr)

    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="UVR Install Record Utility")
    subparsers = parser.add_subparsers(dest="action")

    read_p = subparsers.add_parser("read", help="Read existing record")
    read_p.add_argument("--json", action="store_true", help="Output raw JSON")

    write_p = subparsers.add_parser("write", help="Write successful install record")
    write_p.add_argument("--option", required=True, help="Selected option letter (A, B, or C)")
    write_p.add_argument("--mode", required=True, help="Resulting install mode (NVIDIA, TENSORRT_RTX, CPU)")
    write_p.add_argument("--reuse-venv", default="Y", help="User answer for venv reuse (Y/N)")
    write_p.add_argument("--has-uv", default="1", help="Whether uv is present (1/0)")
    write_p.add_argument("--has-ffmpeg", default="1", help="Whether ffmpeg is present (1/0)")
    write_p.add_argument("--has-rubberband", default="1", help="Whether rubberband is present (1/0)")
    write_p.add_argument("--has-vc-redist", default="1", help="Whether VC redist is present (1/0)")

    inspect_p = subparsers.add_parser("inspect-stack", help="Inspect current venv stack")
    inspect_p.add_argument("--json", action="store_true", help="Output raw JSON")

    args = parser.parse_args()

    if args.action == "read":
        record = read_install_record()
        if not record:
            print("NO_RECORD")
            return 0
        if args.json:
            print(json.dumps(record, indent=2))
        else:
            print(f"MODE={record.get('install_mode', 'UNKNOWN')}")
            print(f"OPTION={record.get('selected_option', 'UNKNOWN')}")
            print(f"VERSION={record.get('app_version', 'UNKNOWN')}")
            print(f"TIMESTAMP={record.get('timestamp_utc', 'UNKNOWN')}")
        return 0

    elif args.action == "write":
        write_install_record(
            option=args.option,
            install_mode=args.mode,
            reuse_venv=args.reuse_venv,
            has_uv=args.has_uv == "1",
            has_ffmpeg=args.has_ffmpeg == "1",
            has_rubberband=args.has_rubberband == "1",
            has_vc_redist=args.has_vc_redist == "1",
        )
        print("RECORD_WRITTEN")
        return 0

    elif args.action == "inspect-stack":
        stack = inspect_installed_stack()
        if args.json:
            print(json.dumps(stack, indent=2))
        else:
            for k, v in stack.items():
                print(f"{k}={v}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
