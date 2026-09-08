"""Headless inference runner bridging directly to UVR's separate.py backends."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import soundfile as sf

from gui_data.constants import DEMUCS_ARCH_TYPE, MDX_ARCH_TYPE, VR_ARCH_TYPE
from .model_resolver import HeadlessModelData
from separate import SeperateDemucs, SeperateMDX, SeperateMDXC, SeperateVR


def get_audio_metadata(audio_path: str | Path) -> Dict[str, Any]:
    """Inspect audio file duration, sample rate, channels, and format."""
    path = Path(audio_path).resolve()
    try:
        info = sf.info(str(path))
        return {
            "file_name": path.name,
            "duration_sec": round(info.duration, 4),
            "samplerate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "subtype": info.subtype,
            "size_bytes": path.stat().st_size,
        }
    except Exception:
        # Fallback if soundfile info fails
        return {
            "file_name": path.name,
            "duration_sec": 0.0,
            "samplerate": 44100,
            "channels": 2,
            "format": "UNKNOWN",
            "subtype": "UNKNOWN",
            "size_bytes": path.stat().st_size if path.is_file() else 0,
        }


def execute_inference(
    model_data: HeadlessModelData,
    audio_path: str | Path,
    export_dir: str | Path,
    progress_callback: Optional[Callable[[float, float], None]] = None,
    console_callback: Optional[Callable[[str, str], None]] = None,
) -> Tuple[List[Path], bool]:
    """Execute UVR separation directly using core separate.py classes.

    Returns:
        (generated_stems, is_valid)
    """
    audio_path = Path(audio_path).resolve()
    export_path = Path(export_dir).resolve()
    export_path.mkdir(parents=True, exist_ok=True)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Input audio file not found: {audio_path}")

    # Snapshot existing files in export directory to detect new outputs
    existing_files = set(export_path.glob("*"))

    def default_progress(step: float, inference_iterations: float = 0.0) -> None:
        if progress_callback:
            progress_callback(step, inference_iterations)

    def default_console(text: str, base_text: str = "") -> None:
        if console_callback:
            console_callback(text, base_text)

    process_data: Dict[str, Any] = {
        "model_data": model_data,
        "export_path": str(export_path),
        "audio_file_base": audio_path.stem,
        "audio_file": str(audio_path),
        "set_progress_bar": default_progress,
        "write_to_console": default_console,
        "process_iteration": 0,
        "cached_source_callback": lambda process_method, model_name=None: (None, None),
        "cached_model_source_holder": lambda *args, **kwargs: None,
        "list_all_models": [],
        "is_ensemble_master": False,
        "is_4_stem_ensemble": False,
    }

    # Instantiate the corresponding UVR separator
    if model_data.process_method == VR_ARCH_TYPE:
        separator = SeperateVR(model_data, process_data)
    elif model_data.process_method == MDX_ARCH_TYPE:
        if model_data.is_mdx_c:
            separator = SeperateMDXC(model_data, process_data)
        else:
            separator = SeperateMDX(model_data, process_data)
    elif model_data.process_method == DEMUCS_ARCH_TYPE:
        separator = SeperateDemucs(model_data, process_data)
    else:
        raise ValueError(f"Unsupported architecture: {model_data.process_method}")

    # Run inference directly
    separator.seperate()

    # Detect generated stems
    all_current_files = set(export_path.glob("*"))
    new_files = [f for f in (all_current_files - existing_files) if f.is_file()]

    # Validate output files
    is_valid = True
    if not new_files:
        is_valid = False
    else:
        for stem_file in new_files:
            if stem_file.stat().st_size == 0:
                is_valid = False
                break

    return sorted(new_files), is_valid
