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


def get_expected_stem_paths(
    model_data: HeadlessModelData,
    audio_path: str | Path,
    export_dir: str | Path,
) -> List[Path]:
    """Calculate expected output stem file paths based on model settings and audio name."""
    audio_stem = Path(audio_path).stem
    export_path = Path(export_dir).resolve()

    stems = []
    # Demucs 4-stem or standard 2-stem
    if getattr(model_data, "process_method", "") == DEMUCS_ARCH_TYPE and getattr(model_data, "is_4_stem_ensemble", False):
        sources = getattr(model_data, "demucs_source_list", ["drums", "bass", "other", "vocals"])
        for src in sources:
            stems.append(src.capitalize())
    else:
        # Standard primary & secondary stems
        if not getattr(model_data, "is_secondary_stem_only", False) and getattr(model_data, "primary_stem", None):
            stems.append(model_data.primary_stem)
        if not getattr(model_data, "is_primary_stem_only", False) and getattr(model_data, "secondary_stem", None):
            stems.append(model_data.secondary_stem)

    ext = ".wav"
    save_format = getattr(model_data, "save_format", "WAV").lower()
    if save_format in ("mp3", "flac", "m4a"):
        ext = f".{save_format}"

    return [export_path / f"{audio_stem}_({stem}){ext}" for stem in stems]


def get_existing_outputs(
    model_data: HeadlessModelData,
    audio_path: str | Path,
    export_dir: str | Path,
) -> List[Path]:
    """Find any existing output files in export_dir matching this separation job."""
    export_path = Path(export_dir).resolve()
    if not export_path.is_dir():
        return []

    audio_stem = Path(audio_path).stem
    expected = set(get_expected_stem_paths(model_data, audio_path, export_dir))
    existing = [p for p in expected if p.is_file()]

    # Also detect existing files matching "{audio_stem}_(*).*"
    for p in export_path.glob(f"{audio_stem}_*.*"):
        if p.is_file() and p not in existing and "(" in p.name and ")" in p.name:
            existing.append(p)

    return sorted(existing)


def execute_inference(
    model_data: HeadlessModelData,
    audio_path: str | Path,
    export_dir: str | Path,
    progress_callback: Optional[Callable[[float, float], None]] = None,
    console_callback: Optional[Callable[[str, str], None]] = None,
    overwrite: bool = False,
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

    # Remove conflicting existing outputs if overwrite is enabled
    if overwrite:
        colliding = get_existing_outputs(model_data, audio_path, export_path)
        for f in colliding:
            try:
                f.unlink()
            except Exception:
                pass

    # Snapshot existing files and mtimes in export directory
    existing_files = set(export_path.glob("*"))
    existing_mtimes = {f: f.stat().st_mtime for f in existing_files if f.is_file()}

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

    # Detect generated stems:
    all_current_files = set(export_path.glob("*"))
    generated_set = set(
        f for f in (all_current_files - existing_files)
        if f.is_file() and f.name.startswith(f"{audio_path.stem}_")
    )

    # Expected output stems that were written or updated
    expected_stems = get_expected_stem_paths(model_data, audio_path, export_path)
    for p in expected_stems:
        if p.is_file():
            if p not in existing_mtimes or p.stat().st_mtime >= existing_mtimes.get(p, 0):
                generated_set.add(p)

    output_files = sorted([f for f in generated_set if f.name.startswith(f"{audio_path.stem}_")])

    # Validate output files
    is_valid = True
    if not output_files:
        is_valid = False
    else:
        for stem_file in output_files:
            if stem_file.stat().st_size == 0:
                is_valid = False
                break

    return output_files, is_valid
