"""Model resolution and headless configuration adapter for UVR models.

Discovers models, computes UVR MD5 hashes, parses model configurations,
and constructs HeadlessModelData objects suitable for direct ingestion by
separate.py separator classes (SeperateMDX, SeperateMDXC, SeperateVR, SeperateDemucs).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from ml_collections import ConfigDict

from gui_data.constants import (
    ALL_STEMS,
    CKPT,
    DEMUCS_2_SOURCE,
    DEMUCS_2_SOURCE_MAPPER,
    DEMUCS_4_SOURCE,
    DEMUCS_4_SOURCE_MAPPER,
    DEMUCS_ARCH_TYPE,
    DEMUCS_NEWER_TAGS,
    DEMUCS_UVR_MODEL,
    DEMUCS_V1,
    DEMUCS_V1_TAG,
    DEMUCS_V2,
    DEMUCS_V2_TAG,
    DEMUCS_V3,
    DEMUCS_V3_TAG,
    DEMUCS_V4,
    DEMUCS_V4_TAG,
    DEMUCS_VERSION_MAPPER,
    INST_STEM,
    MDX_ARCH_TYPE,
    MODEL_PRECISION_DEFAULT,
    MODEL_PRECISION_FP16,
    MODEL_PRECISION_FP32,
    MODEL_PRECISION_BF16,
    MODEL_PRECISION_OPTIONS,
    NO_MODEL,
    ONNX,
    PRIMARY_STEM,
    PTH,
    VOCAL_STEM,
    VR_ARCH_TYPE,
    YAML,
    normalize_model_precision,
    secondary_stem,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MDX_MODELS_DIR = MODELS_DIR / "MDX_Net_Models"
VR_MODELS_DIR = MODELS_DIR / "VR_Models"
DEMUCS_MODELS_DIR = MODELS_DIR / "Demucs_Models"
DEMUCS_NEWER_REPO_DIR = DEMUCS_MODELS_DIR / "v3_v4_repo"

MDX_HASH_DIR = MDX_MODELS_DIR / "model_data"
MDX_HASH_JSON = MDX_HASH_DIR / "model_data.json"
MDX_C_CONFIG_PATH = MDX_HASH_DIR / "mdx_c_configs"
MDX_NAME_MAPPER_PATH = MDX_HASH_DIR / "model_name_mapper.json"

VR_HASH_DIR = VR_MODELS_DIR / "model_data"
VR_HASH_JSON = VR_HASH_DIR / "model_data.json"
VR_PARAM_DIR = PROJECT_ROOT / "lib_v5" / "vr_network" / "modelparams"

DEMUCS_NAME_MAPPER_PATH = DEMUCS_MODELS_DIR / "model_data" / "model_name_mapper.json"


def compute_uvr_hash(model_path: str | Path) -> str:
    """Compute the model hash using UVR's exact algorithm.

    Reads the last 10,000 * 1024 bytes (or full file if smaller) and computes MD5.
    """
    path = Path(model_path).resolve()
    with open(path, "rb") as stream:
        try:
            stream.seek(-10000 * 1024, 2)
            return hashlib.md5(stream.read()).hexdigest()
        except (OSError, ValueError):
            stream.seek(0)
            return hashlib.md5(stream.read()).hexdigest()


class HeadlessModelData:
    """Lightweight configuration container compatible with SeperateAttributes."""

    def __init__(self, **kwargs: Any) -> None:
        # Default flags matching SeperateAttributes expectations
        self.process_method: str = MDX_ARCH_TYPE
        self.model_path: str = ""
        self.model_name: str = ""
        self.model_basename: str = ""
        self.model_hash: Optional[str] = None
        self.model_status: bool = True
        self.device_set: str = "0"
        self.is_gpu_conversion: int = 0
        self.is_use_opencl: bool = False
        self.model_precision: str = MODEL_PRECISION_DEFAULT
        self.save_format: str = "WAV"
        self.wav_type_set: str = "PCM_16"
        self.mp3_bit_set: str = "320k"
        self.is_normalization: bool = False
        self.is_primary_stem_only: bool = False
        self.is_secondary_stem_only: bool = False
        self.is_pitch_change: bool = False
        self.semitone_shift: float = 0.0
        self.is_match_frequency_pitch: bool = False
        self.is_invert_spec: bool = False
        self.is_deverb_vocals: bool = False
        self.is_mixer_mode: bool = False
        self.primary_stem: str = "Instrumental"
        self.secondary_stem: str = "Vocals"
        self.primary_stem_native: str = "Instrumental"
        self.model_samplerate: int = 44100
        self.model_capacity: Tuple[int, int] = (32, 128)
        self.is_vr_51_model: bool = False

        # Secondary / chain model placeholders
        self.is_secondary_model: bool = False
        self.is_secondary_model_activated: bool = False
        self.secondary_model: Any = None
        self.secondary_model_scale: Any = None
        self.primary_model_primary_stem: Any = None
        self.is_primary_model_primary_stem_only: bool = False
        self.is_primary_model_secondary_stem_only: bool = False
        self.is_pre_proc_model: bool = False
        self.pre_proc_model_activated: bool = False
        self.pre_proc_model: Any = None
        self.vocal_split_model: Any = None
        self.is_vocal_split_model: bool = False
        self.is_vocal_split_model_activated: bool = False
        self.is_save_inst_vocal_splitter: bool = False
        self.is_inst_only_voc_splitter: bool = False
        self.is_save_vocal_only: bool = False
        self.is_karaoke: bool = False
        self.is_bv_model: bool = False
        self.bv_model_rebalance: float = 0.0
        self.is_sec_bv_rebalance: bool = False
        self.deverb_vocal_opt: Any = None
        self.DENOISER_MODEL: Any = None
        self.DEVERBER_MODEL: Any = None

        # Ensemble flags
        self.is_ensemble_mode: bool = False
        self.is_4_stem_ensemble: bool = False
        self.is_multi_stem_ensemble: bool = False
        self.ensemble_primary_stem: Any = None
        self.ensemble_secondary_stem: Any = None

        # MDX specific
        self.is_mdx_ckpt: bool = False
        self.is_mdx_c: bool = False
        self.mdx_c_configs: Any = None
        self.mdx_dim_f_set: int = 2048
        self.mdx_dim_t_set: int = 8
        self.mdx_n_fft_scale_set: int = 6144
        self.compensate: Optional[float] = 1.035
        self.mdxnet_stem_select: str = "Vocals"
        self.mdx_batch_size: int = 1
        self.mdx_segment_size: int = 256
        self.overlap: float = 0.25
        self.overlap_mdx: Any = "Default"
        self.overlap_mdx23: int = 8
        self.chunks: int = 0
        self.margin: int = 44100
        self.is_adaptive_chunk: bool = False
        self.is_denoise: bool = False
        self.is_denoise_model: bool = False
        self.is_mdx_combine_stems: bool = False
        self.is_mdx_c_seg_def: bool = False
        self.mixer_path: str = str(MDX_MODELS_DIR / "mixer_val.ckpt")

        # VR specific
        self.vr_model_param: Any = None
        self.is_high_end_process: str = "mirroring"
        self.is_tta: bool = False
        self.is_post_process: bool = False
        self.window_size: int = 512
        self.batch_size: int = 1
        self.crop_size: int = 256
        self.post_process_threshold: float = 0.2
        self.aggression_setting: float = 0.05

        # Demucs specific
        self.demucs_version: str = DEMUCS_V4
        self.demucs_stems: str = ALL_STEMS
        self.demucs_source_list: List[str] = list(DEMUCS_4_SOURCE)
        self.demucs_source_map: Dict[str, Any] = dict(DEMUCS_4_SOURCE_MAPPER)
        self.demucs_stem_count: int = 4
        self.shifts: int = 2
        self.segment: Any = "Default"
        self.is_split_mode: bool = True
        self.is_chunk_demucs: bool = False
        self.is_demucs_combine_stems: bool = False
        self.secondary_model_4_stem: List[Any] = []
        self.secondary_model_4_stem_scale: List[Any] = []
        self.is_demucs_pre_proc_model_inst_mix: bool = False

        for k, v in kwargs.items():
            setattr(self, k, v)


def _load_json(path: Path) -> Dict[str, Any]:
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def list_available_models() -> List[Dict[str, Any]]:
    """Scan and list all installed ONNX models in MDX_Net_Models (non-recursive).

    Per MVP spec §3.1, only `.onnx` files in `models/MDX_Net_Models` are
    discovered.  Non-ONNX model formats are out of scope for this MVP.
    """
    results: List[Dict[str, Any]] = []

    if MDX_MODELS_DIR.is_dir():
        for file in MDX_MODELS_DIR.iterdir():
            if file.is_file() and file.suffix.lower() == ONNX:
                results.append({
                    "name": file.name,
                    "basename": file.stem,
                    "path": str(file),
                    "architecture": MDX_ARCH_TYPE,
                    "size_mb": round(file.stat().st_size / (1024 * 1024), 2),
                    "format": file.suffix.lower(),
                })

    return results


def resolve_model_file(model_identifier: str) -> Tuple[Path, str]:
    """Resolve an ONNX model name or file path to an existing file and its architecture.

    Per MVP spec §3.1, only `.onnx` files are accepted.  Non-ONNX model
    extensions are rejected with a clear error.
    """
    # 1. Direct path check
    direct_path = Path(model_identifier).resolve()
    if direct_path.is_file():
        ext = direct_path.suffix.lower()
        if ext != ONNX:
            raise ValueError(
                f"Non-ONNX model format '{ext}' is not supported in this MVP. "
                f"Only .onnx models are accepted. Got: {direct_path.name}"
            )
        return direct_path, MDX_ARCH_TYPE

    # 2. Search by filename or stem in MDX_Net_Models (.onnx only)
    for candidate in [
        MDX_MODELS_DIR / model_identifier,
        MDX_MODELS_DIR / f"{model_identifier}.onnx",
    ]:
        if candidate.is_file() and candidate.suffix.lower() == ONNX:
            return candidate, MDX_ARCH_TYPE

    # 3. Fallback search through discovered ONNX models
    for m in list_available_models():
        if model_identifier.lower() in [m["name"].lower(), m["basename"].lower()]:
            return Path(m["path"]), m["architecture"]

    raise FileNotFoundError(
        f"ONNX model '{model_identifier}' could not be located in "
        f"models/MDX_Net_Models/. Ensure the .onnx file is present."
    )


def build_headless_model_data(
    model_path: Path,
    architecture: str,
    precision: str = MODEL_PRECISION_DEFAULT,
    device: str = "cuda:0",
    segment_size: Optional[int] = None,
    overlap: Optional[float] = None,
    batch_size: int = 1,
) -> HeadlessModelData:
    """Build a complete HeadlessModelData object configured for the requested model."""
    model_path = model_path.resolve()
    model_name = model_path.name
    model_basename = model_path.stem
    model_hash = compute_uvr_hash(model_path)

    is_gpu = "cuda" in device.lower() or "directml" in device.lower()
    device_set = device.split(":")[-1] if ":" in device else "0"

    norm_precision = normalize_model_precision(precision)

    data = HeadlessModelData(
        process_method=architecture,
        model_path=str(model_path),
        model_name=model_name,
        model_basename=model_basename,
        model_hash=model_hash,
        model_precision=norm_precision,
        device_set=device_set,
        is_gpu_conversion=0 if is_gpu else -1,
    )

    if architecture == MDX_ARCH_TYPE:
        data.is_mdx_ckpt = model_name.endswith(CKPT)
        if segment_size is not None:
            data.mdx_segment_size = segment_size
        if overlap is not None:
            data.overlap_mdx = overlap
        data.mdx_batch_size = batch_size

        # Check for model settings in hash json, then model_data.json
        hash_file = MDX_HASH_DIR / f"{model_hash}.json"
        hash_data = _load_json(hash_file)
        if not hash_data:
            master_data = _load_json(MDX_HASH_JSON)
            hash_data = master_data.get(model_hash, {})

        if hash_data:
            if "config_yaml" in hash_data:
                data.is_mdx_c = True
                cfg_path = MDX_C_CONFIG_PATH / hash_data["config_yaml"]
                if cfg_path.is_file():
                    with cfg_path.open("r", encoding="utf-8") as f:
                        data.mdx_c_configs = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
                    target = getattr(data.mdx_c_configs.training, "target_instrument", None)
                    if target:
                        data.primary_stem = target
                    else:
                        stems = getattr(data.mdx_c_configs.training, "instruments", ["Vocals"])
                        data.primary_stem = stems[0]
            else:
                data.compensate = hash_data.get("compensate", 1.035)
                data.mdx_dim_f_set = hash_data.get("mdx_dim_f_set", 2048)
                data.mdx_dim_t_set = hash_data.get("mdx_dim_t_set", 8)
                data.mdx_n_fft_scale_set = hash_data.get("mdx_n_fft_scale_set", 6144)
                data.primary_stem = hash_data.get("primary_stem", "Instrumental")
                data.primary_stem_native = data.primary_stem
                data.is_karaoke = hash_data.get("is_karaoke", False)

        data.secondary_stem = secondary_stem(data.primary_stem)

    elif architecture == VR_ARCH_TYPE:
        hash_file = VR_HASH_DIR / f"{model_hash}.json"
        hash_data = _load_json(hash_file)
        if not hash_data:
            master_data = _load_json(VR_HASH_JSON)
            hash_data = master_data.get(model_hash, {})

        if hash_data:
            data.primary_stem = hash_data.get("primary_stem", "Vocals")
            data.secondary_stem = secondary_stem(data.primary_stem)
            vr_param_file = VR_PARAM_DIR / f"{hash_data.get('vr_model_param', '1_HP-UVR')}.json"
            if vr_param_file.is_file():
                from lib_v5.vr_network.model_param_init import ModelParameters
                data.vr_model_param = ModelParameters(str(vr_param_file))
                data.model_samplerate = data.vr_model_param.param.get("sr", 44100)

            if "nout" in hash_data and "nout_lstm" in hash_data:
                data.model_capacity = (hash_data["nout"], hash_data["nout_lstm"])
                data.is_vr_51_model = True

    elif architecture == DEMUCS_ARCH_TYPE:
        data.demucs_version = DEMUCS_V4
        for ver, tag in DEMUCS_VERSION_MAPPER.items():
            if tag in model_name:
                data.demucs_version = ver

        if DEMUCS_UVR_MODEL in model_name:
            data.demucs_source_list = list(DEMUCS_2_SOURCE)
            data.demucs_source_map = dict(DEMUCS_2_SOURCE_MAPPER)
            data.demucs_stem_count = 2
        else:
            data.demucs_source_list = list(DEMUCS_4_SOURCE)
            data.demucs_source_map = dict(DEMUCS_4_SOURCE_MAPPER)
            data.demucs_stem_count = 4

    return data
