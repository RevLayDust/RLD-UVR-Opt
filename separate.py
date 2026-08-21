from __future__ import annotations
from typing import TYPE_CHECKING
from demucs.apply import apply_model, demucs_segments
from demucs.hdemucs import HDemucs
from demucs.model_v2 import auto_load_demucs_model_v2
from demucs.pretrained import get_model as _gm
from demucs.utils import apply_model_v1
from demucs.utils import apply_model_v2
from lib_v5.tfc_tdf_v3 import TFC_TDF_net, STFT
from lib_v5 import spec_utils
from lib_v5.vr_network import nets
from lib_v5.vr_network import nets_new
from lib_v5.vr_network.model_param_init import ModelParameters
from pathlib import Path
from gui_data.constants import *
from gui_data.error_handling import *
from scipy import signal
import audioread
import gzip
import librosa
import math
import numpy as np
import torch
import onnxruntime as ort
import os
import warnings
import pydub
import soundfile as sf
import lib_v5.mdxnet as MdxnetSet
import math
#import random
from onnx import load
from onnx2pytorch import ConvertModel
import gc
import queue
import threading
import time

class ONNXExecutionWorker:
    def __init__(self):
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True, name="ONNX_Execution_Worker")
        self.thread.start()
        
    def _run(self):
        while True:
            func, args, kwargs, result_queue = self.queue.get()
            result = None
            try:
                result = func(*args, **kwargs)
                result_queue.put((result, None))
            except Exception as error:
                result_queue.put((None, error))
            finally:
                self.queue.task_done()
                func = None
                args = None
                kwargs = None
                result_queue = None
                result = None
            
    def execute(self, func, *args, **kwargs):
        res_queue = queue.Queue()
        self.queue.put((func, args, kwargs, res_queue))
        res, err = res_queue.get()
        if err is not None:
            raise err
        return res

ONNX_WORKER = ONNXExecutionWorker()

class ModelCacheManager:
    def __init__(self, max_cached_models=1, idle_timeout_seconds=300):
        self.cache = {}
        self.max_cached_models = 1
        self.idle_timeout_seconds = idle_timeout_seconds
        self._lock = threading.RLock()
        if max_cached_models != 1:
            print("[MODEL CACHE] Multi-model caching is disabled; only one model/precision session may remain active.")

    @staticmethod
    def _normalize_path(model_path):
        return os.path.normpath(os.path.abspath(model_path))

    @staticmethod
    def _same_precision(cached_precision, requested_precision):
        return requested_precision is None or cached_precision == requested_precision

    def get_model(self, model_path, precision_mode=None):
        normalized_path = self._normalize_path(model_path)
        with self._lock:
            self._evict_inactive(normalized_path, precision_mode)
            print(f"[DIAGNOSTIC] get_model lookup: {model_path} -> normalized: {normalized_path}")
            entry = self.cache.get(normalized_path)
            if entry is not None and self._same_precision(entry.get('precision_mode'), precision_mode):
                entry['last_used'] = time.time()
                print(f"Model Cache Hit: Reusing resident model {os.path.basename(model_path)}")
                return entry['model_obj']
            print(f"[DIAGNOSTIC] get_model Cache Miss for: {normalized_path}")
            return None

    def set_model(self, model_path, model_obj, size_bytes=0, precision_mode=None):
        normalized_path = self._normalize_path(model_path)
        with self._lock:
            self._evict_inactive(normalized_path, precision_mode)
            if normalized_path in self.cache:
                self.unload_model(normalized_path, reason="Session Replacement")
            self.cache[normalized_path] = {
                'model_obj': model_obj,
                'last_used': time.time(),
                'size_bytes': size_bytes,
                'precision_mode': precision_mode
            }
            print(f"Model Cache Store: Resident model cached {os.path.basename(model_path)} (precision: {precision_mode})")

    def _evict_inactive(self, requested_path, requested_precision):
        for cached_path, entry in list(self.cache.items()):
            same_path = cached_path == requested_path
            same_precision = self._same_precision(entry.get('precision_mode'), requested_precision)
            if same_path and same_precision:
                continue
            reason = "Precision Change" if same_path else "Model Change"
            self.unload_model(cached_path, reason=reason)

    def evict_if_necessary(self, incoming_size_bytes=0):
        with self._lock:
            now = time.time()
            for cached_path, entry in list(self.cache.items()):
                if now - entry['last_used'] > self.idle_timeout_seconds:
                    self.unload_model(cached_path, reason="Idle Timeout")

    def unload_model(self, model_path, reason="Explicit Release"):
        normalized_path = self._normalize_path(model_path)
        with self._lock:
            entry = self.cache.pop(normalized_path, None)
            if entry is None:
                return False
            print(f"Model Cache Eviction ({reason}): Unloading {os.path.basename(normalized_path)}")
            model_obj = entry.pop('model_obj', None)
            entry.clear()
            del model_obj
            del entry
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()
            return True

    def clear_all(self):
        with self._lock:
            for cached_path in list(self.cache):
                self.unload_model(cached_path, reason="Clear All")

MODEL_CACHE = ModelCacheManager()

class SmartPrecisionPolicy:
    """
    Smart Precision Policy abstraction that controls:
    - speed
    - VRAM
    - numerical stability
    - output quality
    
    Precision Preset Modes:
    1. Ultra Quality: FP32
    2. Balanced: BF16
    3. Performance: FP16
    """
    def __init__(self, mode_setting: str):
        self.raw = normalize_model_precision(mode_setting)
        self.mode = self._parse_mode(self.raw)

    @staticmethod
    def _parse_mode(raw: str) -> str:
        if raw == MODEL_PRECISION_FP16:
            return 'FP16'
        elif raw == MODEL_PRECISION_BF16:
            return 'BF16'
        return 'FP32'

    @property
    def is_fp32(self) -> bool:
        return self.mode == 'FP32'

    @property
    def is_bf16(self) -> bool:
        return self.mode == 'BF16'

    @property
    def is_fp16(self) -> bool:
        return self.mode == 'FP16'

    @property
    def is_half_or_lower(self) -> bool:
        return self.mode != 'FP32'

    @property
    def enabled_amp(self) -> bool:
        return self.mode != 'FP32'

    @property
    def amp_dtype(self):
        if self.mode == 'FP32':
            return torch.float32
        elif self.mode == 'BF16':
            return torch.bfloat16
        elif self.mode == 'FP16':
            return torch.float16
        return torch.float32

    def cast_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Casts PyTorch neural net weights based on policy."""
        if self.mode == 'FP32':
            return model
        elif self.mode == 'BF16':
            return model.bfloat16()
        elif self.mode == 'FP16':
            return model.half()
        return model

    def prepare_spectrogram(self, spek: torch.Tensor) -> torch.Tensor:
        """Applies input precision transforms based on policy."""
        if self.mode == 'FP32':
            return spek.to(torch.float32)
        elif self.mode == 'BF16':
            return spek.bfloat16()
        elif self.mode == 'FP16':
            return spek.half()
        return spek

    def effective_mode_for(self, model_type: str) -> str:
        """Returns what precision will actually be used for a given model type.
        
        Args:
            model_type: 'ONNX', 'PyTorch-CKPT', or 'PyTorch-ConvertModel'
            
        Returns:
            The actual precision mode that will be applied at runtime.
        """
        if model_type == 'ONNX':
            if self.mode == 'FP32':
                return 'FP32'
            elif self.mode == 'FP16':
                # ONNX only supports FP16 quantization via onnxconverter-common
                return 'FP16'
            elif self.mode == 'BF16':
                # ONNX Runtime has no native BF16 support — must stay FP32
                return 'FP32'
            return 'FP32'
        else:
            # PyTorch models support the available precision modes natively
            return self.mode

    def diagnose(self, model_run, model_type: str, execution_provider: str = "") -> dict:
        """Detect and report actual runtime precision state for full transparency.
        
        Args:
            model_run: The model object (torch.nn.Module or ort.InferenceSession)
            model_type: 'ONNX', 'PyTorch-CKPT', or 'PyTorch-ConvertModel'
            execution_provider: e.g. 'CUDAExecutionProvider'
            
        Returns:
            Dict with precision diagnostics.
        """
        effective = self.effective_mode_for(model_type)
        report = {
            'requested_mode': self.mode,
            'effective_mode': effective,
            'model_type': model_type,
            'execution_provider': execution_provider or 'N/A',
        }

        if isinstance(model_run, torch.nn.Module):
            first_param = next(model_run.parameters(), None)
            report['model_weight_dtype'] = str(first_param.dtype) if first_param is not None else 'N/A'
            report['autocast_enabled'] = self.enabled_amp
            report['autocast_dtype'] = str(self.amp_dtype)
            # Verify consistency
            if first_param is not None:
                actual_dtype_str = str(first_param.dtype)
                if self.mode == 'FP16' and 'float16' not in actual_dtype_str:
                    report['warning'] = f"Requested FP16 but model weights are {actual_dtype_str}"
                elif self.mode == 'BF16' and 'bfloat16' not in actual_dtype_str:
                    report['warning'] = f"Requested BF16 but model weights are {actual_dtype_str}"
        elif hasattr(model_run, 'get_inputs'):
            # ONNX InferenceSession
            input_type = model_run.get_inputs()[0].type
            report['onnx_input_type'] = input_type
            report['onnx_output_type'] = model_run.get_outputs()[0].type if model_run.get_outputs() else 'N/A'
            active_providers = model_run.get_providers() if hasattr(model_run, 'get_providers') else []
            report['active_providers'] = active_providers
            # Detect actual ONNX runtime precision
            if 'float16' in input_type:
                report['effective_mode'] = 'FP16'
            else:
                report['effective_mode'] = 'FP32'
            # Report limitations honestly
            if self.mode != report['effective_mode']:
                if self.mode == 'BF16':
                    report['fallback_reason'] = (
                        f"ONNX Runtime does not support {self.mode}. "
                        f"Model executes at {report['effective_mode']}. "
                        f"BF16 requires PyTorch (.ckpt) models."
                    )
                elif self.mode == 'FP16' and 'float16' not in input_type:
                    report['fallback_reason'] = (
                        f"FP16 quantization of ONNX model failed or was not applied. "
                        f"Model executes at FP32."
                    )
        else:
            # Lambda wrapper (ONNX via ONNX_WORKER)
            report['model_weight_dtype'] = 'opaque (lambda wrapper)'
            report['note'] = 'Model is accessed via lambda; precision is determined by its ONNX Runtime session.'

        # Print the report
        print(f"[PRECISION] === Runtime Precision Report ===")
        for k, v in report.items():
            print(f"[PRECISION]   {k}: {v}")
        if report.get('fallback_reason'):
            print(f"[PRECISION]   [WARN] LIMITATION: {report['fallback_reason']}")
        if self.mode != report['effective_mode']:
            print(f"[PRECISION]   [WARN] MISMATCH: Requested={self.mode}, Actual={report['effective_mode']}")
        else:
            print(f"[PRECISION]   [OK] Precision applied correctly: {report['effective_mode']}")
        print(f"[PRECISION] ==============================")
        
        return report

class AdaptiveChunkScheduler:
    def __init__(self, device, precision_mode, model_size_bytes, batch_size):
        self.device = device
        self.precision_mode = precision_mode
        self.policy = SmartPrecisionPolicy(precision_mode)
        self.model_size_bytes = model_size_bytes
        self.batch_size = batch_size
        
    def get_free_vram(self):
        if torch.cuda.is_available():
            try:
                free_mem, total_mem = torch.cuda.mem_get_info(self.device)
                return free_mem
            except Exception:
                pass
        return 8 * 1024 * 1024 * 1024 # Default fallback 8GB
        
    def determine_chunk_size(self):
        free_vram = self.get_free_vram()
        free_vram_gb = free_vram / (1024 ** 3)
        
        # Specification workflow
        if free_vram_gb > 14.0:
            chunk_size = 8192
        elif free_vram_gb > 10.0:
            chunk_size = 4096
        elif free_vram_gb > 6.0:
            chunk_size = 2048
        else:
            chunk_size = 1024
            
        # Refine based on precision, model size, and batch size
        if not self.policy.is_half_or_lower:
            # FP32 uses twice the memory of lower precision
            chunk_size = max(1024, chunk_size // 2)
            
        if self.batch_size > 4:
            # Larger batch size requires smaller chunks
            chunk_size = max(1024, chunk_size // 2)
            
        return chunk_size

    def adjust_overlap_ratio(self, chunk_size, user_overlap):
        # Larger chunk size can afford smaller overlap ratio to speed up processing
        # Smaller chunk size needs larger overlap ratio to maintain quality
        if chunk_size >= 4096:
            return min(user_overlap, 0.25) # Lower overlap
        elif chunk_size >= 2048:
            return min(user_overlap, 0.5)
        else:
            return user_overlap # Keep user selection for small chunks

def create_onnx_fp16_session(model_path, providers):
    import onnx
    from onnxconverter_common import float16

    model_name = os.path.basename(model_path)
    print(f"[PRECISION] Loading {model_name} into CPU RAM for in-memory FP16 conversion...")
    model_fp32 = onnx.load(model_path, load_external_data=True)
    try:
        model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=False)
    finally:
        del model_fp32
        gc.collect()

    try:
        serialized_fp16 = model_fp16.SerializeToString()
    finally:
        del model_fp16
        gc.collect()

    try:
        print(f"[PRECISION] Creating ONNX Runtime FP16 session from RAM ({len(serialized_fp16) / (1024 ** 2):.2f} MiB)...")
        return ort.InferenceSession(serialized_fp16, providers=providers)
    finally:
        del serialized_fp16
        gc.collect()


def create_onnx_session(model_path, providers, precision_policy):
    if not precision_policy.is_fp16:
        return ort.InferenceSession(model_path, providers=providers)

    try:
        return create_onnx_fp16_session(model_path, providers)
    except Exception as error:
        print(
            f"[PRECISION] In-memory FP16 conversion failed ({type(error).__name__}: {error}). "
            "Falling back to the original FP32 ONNX model without writing a temporary file."
        )
        return ort.InferenceSession(model_path, providers=providers)

 
if TYPE_CHECKING:
    from UVR import ModelData

# if not is_macos:
#     import torch_directml

mps_available = torch.backends.mps.is_available() if is_macos else False
cuda_available = torch.cuda.is_available()

# def get_gpu_info():
#     directml_device, directml_available = DIRECTML_DEVICE, False
    
#     if not is_macos:
#         directml_available = torch_directml.is_available()

#         if directml_available:
#             directml_device = str(torch_directml.device()).partition(":")[0]

#     return directml_device, directml_available

# DIRECTML_DEVICE, directml_available = get_gpu_info()

def clear_gpu_cache():
    gc.collect()
    if is_macos:
        torch.mps.empty_cache()
    else:
        torch.cuda.empty_cache()

warnings.filterwarnings("ignore")
cpu = torch.device('cpu')

class SeperateAttributes:
    def __init__(self, model_data: ModelData, 
                 process_data: dict, 
                 main_model_primary_stem_4_stem=None, 
                 main_process_method=None, 
                 is_return_dual=True, 
                 main_model_primary=None, 
                 vocal_stem_path=None, 
                 master_inst_source=None,
                 master_vocal_source=None):
        
        self.list_all_models: list
        self.model_data = model_data
        self.process_data = process_data
        self.progress_value = 0
        self.set_progress_bar = process_data['set_progress_bar']
        self.write_to_console = process_data['write_to_console']
        if vocal_stem_path:
            self.audio_file, self.audio_file_base = vocal_stem_path
            self.audio_file_base_voc_split = lambda stem, split:os.path.join(self.export_path, f'{self.audio_file_base.replace("_(Vocals)", "")}_({stem}_{split}).wav')
        else:
            self.audio_file = process_data['audio_file']
            self.audio_file_base = process_data['audio_file_base']
            self.audio_file_base_voc_split = None
        self.export_path = process_data['export_path']
        self.cached_source_callback = process_data['cached_source_callback']
        self.cached_model_source_holder = process_data['cached_model_source_holder']
        self.is_4_stem_ensemble = process_data['is_4_stem_ensemble']
        self.list_all_models = process_data['list_all_models']
        self.process_iteration = process_data['process_iteration']
        self.is_return_dual = is_return_dual
        self.is_pitch_change = model_data.is_pitch_change
        self.semitone_shift = model_data.semitone_shift
        self.is_match_frequency_pitch = model_data.is_match_frequency_pitch
        self.overlap = model_data.overlap
        self.overlap_mdx = model_data.overlap_mdx
        self.overlap_mdx23 = model_data.overlap_mdx23
        self.is_mdx_combine_stems = model_data.is_mdx_combine_stems
        self.is_mdx_c = model_data.is_mdx_c
        self.mdx_c_configs = model_data.mdx_c_configs
        self.mdxnet_stem_select = model_data.mdxnet_stem_select
        self.mixer_path = model_data.mixer_path
        self.model_samplerate = model_data.model_samplerate
        self.model_capacity = model_data.model_capacity
        self.is_vr_51_model = model_data.is_vr_51_model
        self.is_pre_proc_model = model_data.is_pre_proc_model
        self.is_secondary_model_activated = model_data.is_secondary_model_activated if not self.is_pre_proc_model else False
        self.is_secondary_model = model_data.is_secondary_model if not self.is_pre_proc_model else True
        self.process_method = model_data.process_method
        self.model_path = model_data.model_path
        self.model_name = model_data.model_name
        self.model_basename = model_data.model_basename
        self.wav_type_set = model_data.wav_type_set
        self.mp3_bit_set = model_data.mp3_bit_set
        self.save_format = model_data.save_format
        self.is_gpu_conversion = model_data.is_gpu_conversion
        self.is_normalization = model_data.is_normalization
        self.is_primary_stem_only = model_data.is_primary_stem_only if not self.is_secondary_model else model_data.is_primary_model_primary_stem_only
        self.is_secondary_stem_only = model_data.is_secondary_stem_only if not self.is_secondary_model else model_data.is_primary_model_secondary_stem_only      
        self.is_ensemble_mode = model_data.is_ensemble_mode
        self.secondary_model = model_data.secondary_model #
        self.primary_model_primary_stem = model_data.primary_model_primary_stem
        self.primary_stem_native = model_data.primary_stem_native
        self.primary_stem = model_data.primary_stem #
        self.secondary_stem = model_data.secondary_stem #
        self.is_invert_spec = model_data.is_invert_spec #
        self.is_deverb_vocals = model_data.is_deverb_vocals
        self.is_mixer_mode = model_data.is_mixer_mode #
        self.secondary_model_scale = model_data.secondary_model_scale #
        self.is_demucs_pre_proc_model_inst_mix = model_data.is_demucs_pre_proc_model_inst_mix #
        self.primary_source_map = {}
        self.secondary_source_map = {}
        self.primary_source = None
        self.secondary_source = None
        self.secondary_source_primary = None
        self.secondary_source_secondary = None
        self.main_model_primary_stem_4_stem = main_model_primary_stem_4_stem
        self.main_model_primary = main_model_primary
        self.ensemble_primary_stem = model_data.ensemble_primary_stem
        self.is_multi_stem_ensemble = model_data.is_multi_stem_ensemble
        self.is_other_gpu = False
        self.is_deverb = True
        self.DENOISER_MODEL = model_data.DENOISER_MODEL
        self.DEVERBER_MODEL = model_data.DEVERBER_MODEL
        self.is_source_swap = False
        self.vocal_split_model = model_data.vocal_split_model
        self.is_vocal_split_model = model_data.is_vocal_split_model
        self.master_vocal_path = None
        self.set_master_inst_source = None
        self.master_inst_source = master_inst_source
        self.master_vocal_source = master_vocal_source
        self.is_save_inst_vocal_splitter = isinstance(master_inst_source, np.ndarray) and model_data.is_save_inst_vocal_splitter
        self.is_inst_only_voc_splitter = model_data.is_inst_only_voc_splitter
        self.is_karaoke = model_data.is_karaoke
        self.is_bv_model = model_data.is_bv_model
        self.is_bv_model_rebalenced = model_data.bv_model_rebalance and self.is_vocal_split_model
        self.is_sec_bv_rebalance = model_data.is_sec_bv_rebalance
        self.stem_path_init = os.path.join(self.export_path, f'{self.audio_file_base}_({self.secondary_stem}).wav')
        self.deverb_vocal_opt = model_data.deverb_vocal_opt
        self.is_save_vocal_only = model_data.is_save_vocal_only
        self.device = cpu
        self.run_type = ['CPUExecutionProvider']
        self.is_opencl = False
        self.device_set = model_data.device_set
        self.is_use_opencl = model_data.is_use_opencl
        
        if self.is_inst_only_voc_splitter or self.is_sec_bv_rebalance:
            self.is_primary_stem_only = False
            self.is_secondary_stem_only = False
        
        if main_model_primary and self.is_multi_stem_ensemble:
            self.primary_stem, self.secondary_stem = main_model_primary, secondary_stem(main_model_primary)

        if self.is_gpu_conversion >= 0:
            if mps_available:
                self.device, self.is_other_gpu = 'mps', True
            else:
                device_prefix = None
                if self.device_set != DEFAULT:
                    device_prefix = CUDA_DEVICE#DIRECTML_DEVICE if self.is_use_opencl and directml_available else CUDA_DEVICE

                # if directml_available and self.is_use_opencl:
                #     self.device = torch_directml.device() if not device_prefix else f'{device_prefix}:{self.device_set}'
                #     self.is_other_gpu = True
                if cuda_available:# and not self.is_use_opencl:
                    self.device = CUDA_DEVICE if not device_prefix else f'{device_prefix}:{self.device_set}'
                    cuda_options = {
                        'arena_extend_strategy': 'kSameAsRequested',
                        'cudnn_conv_algo_search': 'HEURISTIC',
                        'do_copy_in_default_stream': 'True'
                    }
                    self.run_type = [('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider']

        if model_data.process_method == MDX_ARCH_TYPE:
            self.is_mdx_ckpt = model_data.is_mdx_ckpt
            self.primary_model_name, self.primary_sources = self.cached_source_callback(MDX_ARCH_TYPE, model_name=self.model_basename)
            self.is_denoise = model_data.is_denoise#
            self.is_denoise_model = model_data.is_denoise_model#
            self.is_mdx_c_seg_def = model_data.is_mdx_c_seg_def#
            self.mdx_batch_size = model_data.mdx_batch_size
            self.compensate = model_data.compensate
            self.mdx_segment_size = model_data.mdx_segment_size
            
            if self.is_mdx_c:
                if not self.is_4_stem_ensemble:
                    self.primary_stem = model_data.ensemble_primary_stem if process_data['is_ensemble_master'] else model_data.primary_stem
                    self.secondary_stem = model_data.ensemble_secondary_stem if process_data['is_ensemble_master'] else model_data.secondary_stem
            else:
                self.dim_f, self.dim_t = model_data.mdx_dim_f_set, 2**model_data.mdx_dim_t_set
                
            self.check_label_secondary_stem_runs()
            self.n_fft = model_data.mdx_n_fft_scale_set
            self.chunks = model_data.chunks
            self.margin = model_data.margin
            self.adjust = 1
            self.dim_c = 4
            self.hop = 1024

        if model_data.process_method == DEMUCS_ARCH_TYPE:
            self.demucs_stems = model_data.demucs_stems if not main_process_method in [MDX_ARCH_TYPE, VR_ARCH_TYPE] else None
            self.secondary_model_4_stem = model_data.secondary_model_4_stem
            self.secondary_model_4_stem_scale = model_data.secondary_model_4_stem_scale
            self.is_chunk_demucs = model_data.is_chunk_demucs
            self.segment = model_data.segment
            self.demucs_version = model_data.demucs_version
            self.demucs_source_list = model_data.demucs_source_list
            self.demucs_source_map = model_data.demucs_source_map
            self.is_demucs_combine_stems = model_data.is_demucs_combine_stems
            self.demucs_stem_count = model_data.demucs_stem_count
            self.pre_proc_model = model_data.pre_proc_model
            self.device = cpu if self.is_other_gpu and not self.demucs_version in [DEMUCS_V3, DEMUCS_V4] else self.device

            self.primary_stem = model_data.ensemble_primary_stem if process_data['is_ensemble_master'] else model_data.primary_stem
            self.secondary_stem = model_data.ensemble_secondary_stem if process_data['is_ensemble_master'] else model_data.secondary_stem

            if (self.is_multi_stem_ensemble or self.is_4_stem_ensemble) and not self.is_secondary_model:
                self.is_return_dual = False
            
            if self.is_multi_stem_ensemble and main_model_primary:
                self.is_4_stem_ensemble = False
                if main_model_primary in self.demucs_source_map.keys():
                    self.primary_stem = main_model_primary
                    self.secondary_stem = secondary_stem(main_model_primary)
                elif secondary_stem(main_model_primary) in self.demucs_source_map.keys():
                    self.primary_stem = secondary_stem(main_model_primary)
                    self.secondary_stem = main_model_primary

            if self.is_secondary_model and not process_data['is_ensemble_master']:
                if not self.demucs_stem_count == 2 and model_data.primary_model_primary_stem == INST_STEM:
                    self.primary_stem = VOCAL_STEM
                    self.secondary_stem = INST_STEM
                else:
                    self.primary_stem = model_data.primary_model_primary_stem
                    self.secondary_stem = secondary_stem(self.primary_stem)

            self.shifts = model_data.shifts
            self.is_split_mode = model_data.is_split_mode if not self.demucs_version == DEMUCS_V4 else True
            self.primary_model_name, self.primary_sources = self.cached_source_callback(DEMUCS_ARCH_TYPE, model_name=self.model_basename)

        if model_data.process_method == VR_ARCH_TYPE:
            self.check_label_secondary_stem_runs()
            self.primary_model_name, self.primary_sources = self.cached_source_callback(VR_ARCH_TYPE, model_name=self.model_basename)
            self.mp = model_data.vr_model_param
            self.high_end_process = model_data.is_high_end_process
            self.is_tta = model_data.is_tta
            self.is_post_process = model_data.is_post_process
            self.is_gpu_conversion = model_data.is_gpu_conversion
            self.batch_size = model_data.batch_size
            self.window_size = model_data.window_size
            self.input_high_end_h = None
            self.input_high_end = None
            self.post_process_threshold = model_data.post_process_threshold
            self.aggressiveness = {'value': model_data.aggression_setting, 
                                   'split_bin': self.mp.param['band'][1]['crop_stop'], 
                                   'aggr_correction': self.mp.param.get('aggr_correction')}
            
    def __del__(self):
        print(f"[DIAGNOSTIC] === SeperateAttributes Destructor Called for: {getattr(self, 'model_basename', 'Unknown Model')} ===")
            
    def check_label_secondary_stem_runs(self):

        # For ensemble master that's not a 4-stem ensemble, and not mdx_c
        if self.process_data['is_ensemble_master'] and not self.is_4_stem_ensemble and not self.is_mdx_c:
            if self.ensemble_primary_stem != self.primary_stem:
                self.is_primary_stem_only, self.is_secondary_stem_only = self.is_secondary_stem_only, self.is_primary_stem_only
            
        # For secondary models
        if self.is_pre_proc_model or self.is_secondary_model:
            self.is_primary_stem_only = False
            self.is_secondary_stem_only = False
            
    def start_inference_console_write(self):
        if self.is_secondary_model and not self.is_pre_proc_model and not self.is_vocal_split_model:
            self.write_to_console(INFERENCE_STEP_2_SEC(self.process_method, self.model_basename))
        
        if self.is_pre_proc_model:
            self.write_to_console(INFERENCE_STEP_2_PRE(self.process_method, self.model_basename))
            
        if self.is_vocal_split_model:
            self.write_to_console(INFERENCE_STEP_2_VOC_S(self.process_method, self.model_basename))
        
    def running_inference_console_write(self, is_no_write=False):
        self.write_to_console(DONE, base_text='') if not is_no_write else None
        self.set_progress_bar(0.05) if not is_no_write else None
        
        if self.is_secondary_model and not self.is_pre_proc_model and not self.is_vocal_split_model:
            self.write_to_console(INFERENCE_STEP_1_SEC)
        elif self.is_pre_proc_model:
            self.write_to_console(INFERENCE_STEP_1_PRE)
        elif self.is_vocal_split_model:
            self.write_to_console(INFERENCE_STEP_1_VOC_S)
        else:
            self.write_to_console(INFERENCE_STEP_1)
        
    def running_inference_progress_bar(self, length, is_match_mix=False):
        if not is_match_mix:
            self.progress_value += 1

            if (0.8/length*self.progress_value) >= 0.8:
                length = self.progress_value + 1
  
            self.set_progress_bar(0.1, (0.8/length*self.progress_value))
        
    def load_cached_sources(self):
        
        if self.is_secondary_model and not self.is_pre_proc_model:
            self.write_to_console(INFERENCE_STEP_2_SEC_CACHED_MODOEL(self.process_method, self.model_basename))
        elif self.is_pre_proc_model:
            self.write_to_console(INFERENCE_STEP_2_PRE_CACHED_MODOEL(self.process_method, self.model_basename))
        else:
            self.write_to_console(INFERENCE_STEP_2_PRIMARY_CACHED, "")
            
    def cache_source(self, secondary_sources):
        
        model_occurrences = self.list_all_models.count(self.model_basename)
        
        if not model_occurrences <= 1:
            if self.process_method == MDX_ARCH_TYPE:
                self.cached_model_source_holder(MDX_ARCH_TYPE, secondary_sources, self.model_basename)
                
            if self.process_method == VR_ARCH_TYPE:
                self.cached_model_source_holder(VR_ARCH_TYPE, secondary_sources, self.model_basename)

            if self.process_method == DEMUCS_ARCH_TYPE:
                self.cached_model_source_holder(DEMUCS_ARCH_TYPE, secondary_sources, self.model_basename)
           
    def process_vocal_split_chain(self, sources: dict):
        
        def is_valid_vocal_split_condition(master_vocal_source):
            """Checks if conditions for vocal split processing are met."""
            conditions = [
                isinstance(master_vocal_source, np.ndarray),
                self.vocal_split_model,
                not self.is_ensemble_mode,
                not self.is_karaoke,
                not self.is_bv_model
            ]
            return all(conditions)
        
        # Retrieve sources from the dictionary with default fallbacks
        master_inst_source = sources.get(INST_STEM, None)
        master_vocal_source = sources.get(VOCAL_STEM, None)

        # Process the vocal split chain if conditions are met
        if is_valid_vocal_split_condition(master_vocal_source):
            process_chain_model(
                self.vocal_split_model,
                self.process_data,
                vocal_stem_path=self.master_vocal_path,
                master_vocal_source=master_vocal_source,
                master_inst_source=master_inst_source
            )
  
    def process_secondary_stem(self, stem_source, secondary_model_source=None, model_scale=None):
        if not self.is_secondary_model:
            if self.is_secondary_model_activated and isinstance(secondary_model_source, np.ndarray):
                secondary_model_scale = model_scale if model_scale else self.secondary_model_scale
                stem_source = spec_utils.average_dual_sources(stem_source, secondary_model_source, secondary_model_scale)
  
        return stem_source
    
    def final_process(self, stem_path, source, secondary_source, stem_name, samplerate):
        source = self.process_secondary_stem(source, secondary_source)
        self.write_audio(stem_path, source, samplerate, stem_name=stem_name)
        
        return {stem_name: source}
    
    def write_audio(self, stem_path: str, stem_source, samplerate, stem_name=None):
        
        def save_audio_file(path, source):
            source = spec_utils.normalize(source, self.is_normalization)
            sf.write(path, source, samplerate, subtype=self.wav_type_set)

            if is_not_ensemble:
                save_format(path, self.save_format, self.mp3_bit_set)

        def save_voc_split_instrumental(stem_name, stem_source, is_inst_invert=False):
            inst_stem_name = "Instrumental (With Lead Vocals)" if stem_name == LEAD_VOCAL_STEM else "Instrumental (With Backing Vocals)"
            inst_stem_path_name = LEAD_VOCAL_STEM_I if stem_name == LEAD_VOCAL_STEM else BV_VOCAL_STEM_I
            inst_stem_path = self.audio_file_base_voc_split(INST_STEM, inst_stem_path_name)
            stem_source = -stem_source if is_inst_invert else stem_source
            inst_stem_source = spec_utils.combine_arrarys([self.master_inst_source, stem_source], is_swap=True)
            save_with_message(inst_stem_path, inst_stem_name, inst_stem_source)

        def save_voc_split_vocal(stem_name, stem_source):
            voc_split_stem_name = LEAD_VOCAL_STEM_LABEL if stem_name == LEAD_VOCAL_STEM else BV_VOCAL_STEM_LABEL
            voc_split_stem_path = self.audio_file_base_voc_split(VOCAL_STEM, stem_name)
            save_with_message(voc_split_stem_path, voc_split_stem_name, stem_source)

        def save_with_message(stem_path, stem_name, stem_source):
            is_deverb = self.is_deverb_vocals and (
                self.deverb_vocal_opt == stem_name or
                (self.deverb_vocal_opt == 'ALL' and 
                (stem_name == VOCAL_STEM or stem_name == LEAD_VOCAL_STEM_LABEL or stem_name == BV_VOCAL_STEM_LABEL)))

            self.write_to_console(f'{SAVING_STEM[0]}{stem_name}{SAVING_STEM[1]}')
            
            if is_deverb and is_not_ensemble:
                deverb_vocals(stem_path, stem_source)
            
            save_audio_file(stem_path, stem_source)
            self.write_to_console(DONE, base_text='')
            
        def deverb_vocals(stem_path:str, stem_source):
            self.write_to_console(INFERENCE_STEP_DEVERBING, base_text='')
            stem_source_deverbed, stem_source_2 = vr_denoiser(stem_source, self.device, is_deverber=True, model_path=self.DEVERBER_MODEL)
            save_audio_file(stem_path.replace(".wav", "_deverbed.wav"), stem_source_deverbed)
            save_audio_file(stem_path.replace(".wav", "_reverb_only.wav"), stem_source_2)
            
        is_bv_model_lead = (self.is_bv_model_rebalenced and self.is_vocal_split_model and stem_name == LEAD_VOCAL_STEM)
        is_bv_rebalance_lead = (self.is_bv_model_rebalenced and self.is_vocal_split_model and stem_name == BV_VOCAL_STEM)
        is_no_vocal_save = self.is_inst_only_voc_splitter and (stem_name == VOCAL_STEM or stem_name == BV_VOCAL_STEM or stem_name == LEAD_VOCAL_STEM) or is_bv_model_lead
        is_not_ensemble = (not self.is_ensemble_mode or self.is_vocal_split_model)
        is_do_not_save_inst = (self.is_save_vocal_only and self.is_sec_bv_rebalance and stem_name == INST_STEM)

        if is_bv_rebalance_lead:
            master_voc_source = spec_utils.match_array_shapes(self.master_vocal_source, stem_source, is_swap=True)
            bv_rebalance_lead_source = stem_source-master_voc_source
            
        if not is_bv_model_lead and not is_do_not_save_inst:
            if self.is_vocal_split_model or not self.is_secondary_model:
                if self.is_vocal_split_model and not self.is_inst_only_voc_splitter:
                    save_voc_split_vocal(stem_name, stem_source)
                    if is_bv_rebalance_lead:
                        save_voc_split_vocal(LEAD_VOCAL_STEM, bv_rebalance_lead_source)
                else:
                    if not is_no_vocal_save:
                        save_with_message(stem_path, stem_name, stem_source)
                    
                if self.is_save_inst_vocal_splitter and not self.is_save_vocal_only:
                    save_voc_split_instrumental(stem_name, stem_source)
                    if is_bv_rebalance_lead:
                        save_voc_split_instrumental(LEAD_VOCAL_STEM, bv_rebalance_lead_source, is_inst_invert=True)

                self.set_progress_bar(0.95)

        if stem_name == VOCAL_STEM:
            self.master_vocal_path = stem_path

    def pitch_fix(self, source, sr_pitched, org_mix):
        semitone_shift = self.semitone_shift
        source = spec_utils.change_pitch_semitones(source, sr_pitched, semitone_shift=semitone_shift)[0]
        source = spec_utils.match_array_shapes(source, org_mix)
        return source
    
    def match_frequency_pitch(self, mix):
        source = mix
        if self.is_match_frequency_pitch and self.is_pitch_change:
            source, sr_pitched = spec_utils.change_pitch_semitones(mix, 44100, semitone_shift=-self.semitone_shift)
            source = self.pitch_fix(source, sr_pitched, mix)

        return source

class SeperateMDX(SeperateAttributes):        

    def seperate(self):
        samplerate = 44100
    
        if self.primary_model_name == self.model_basename and isinstance(self.primary_sources, tuple):
            mix, source = self.primary_sources
            self.load_cached_sources()
        else:
            print(f"[DIAGNOSTIC] === SeperateMDX.seperate Start ===")
            if torch.cuda.is_available():
                print(f"[DIAGNOSTIC] Initial GPU Alloc: {torch.cuda.memory_allocated() / (1024**2):.2f} MB, Reserved: {torch.cuda.memory_reserved() / (1024**2):.2f} MB")
            print(f"[DIAGNOSTIC] Cache keys currently stored: {list(MODEL_CACHE.cache.keys())}")
            
            self.start_inference_console_write()

            precision = getattr(self.model_data, 'model_precision', MODEL_PRECISION_DEFAULT)
            policy = SmartPrecisionPolicy(precision)
            if not self.is_mdx_ckpt and hasattr(self, 'dim_t'):
                self.mdx_segment_size = min(self.mdx_segment_size, self.dim_t)

            cached_model = MODEL_CACHE.get_model(self.model_path, precision_mode=precision)
            if cached_model is not None:
                if self.is_mdx_ckpt:
                    self.model_run = cached_model
                    model_params = torch.load(self.model_path, map_location=lambda storage, loc: storage)['hyper_parameters']
                    self.dim_c, self.hop = model_params['dim_c'], model_params['hop_length']
                    policy.diagnose(self.model_run, 'PyTorch-CKPT', execution_provider=str(getattr(self, 'run_type', '')))
                else:
                    if self.mdx_segment_size <= self.dim_t and not self.is_other_gpu:
                        ort_ = cached_model
                        input_type = ort_.get_inputs()[0].type
                        if 'float16' in input_type:
                            self.model_run = lambda spek: ONNX_WORKER.execute(ort_.run, None, {'input': spek.cpu().numpy().astype(np.float16)})[0].astype(np.float32)
                        else:
                            self.model_run = lambda spek: ONNX_WORKER.execute(ort_.run, None, {'input': spek.cpu().numpy()})[0]
                        policy.diagnose(ort_, 'ONNX', execution_provider=str(getattr(self, 'run_type', '')))
                    else:
                        self.model_run = cached_model
                        policy.diagnose(self.model_run, 'PyTorch-ConvertModel', execution_provider=str(getattr(self, 'run_type', '')))
            else:
                if self.is_mdx_ckpt:
                    model_params = torch.load(self.model_path, map_location=lambda storage, loc: storage)['hyper_parameters']
                    self.dim_c, self.hop = model_params['dim_c'], model_params['hop_length']
                    separator = MdxnetSet.ConvTDFNet(**model_params)
                    self.model_run = separator.load_from_checkpoint(self.model_path)
                    self.model_run = policy.cast_model(self.model_run)
                    self.model_run = self.model_run.to(self.device).eval()
                    MODEL_CACHE.set_model(self.model_path, self.model_run, size_bytes=os.path.getsize(self.model_path), precision_mode=precision)
                    policy.diagnose(self.model_run, 'PyTorch-CKPT', execution_provider=str(getattr(self, 'run_type', '')))
                else:
                    if self.mdx_segment_size <= self.dim_t and not self.is_other_gpu:
                        ort_ = create_onnx_session(self.model_path, self.run_type, policy)
                        input_type = ort_.get_inputs()[0].type
                        if 'float16' in input_type:
                            self.model_run = lambda spek: ONNX_WORKER.execute(ort_.run, None, {'input': spek.cpu().numpy().astype(np.float16)})[0].astype(np.float32)
                        else:
                            self.model_run = lambda spek: ONNX_WORKER.execute(ort_.run, None, {'input': spek.cpu().numpy()})[0]
                        MODEL_CACHE.set_model(self.model_path, ort_, size_bytes=os.path.getsize(self.model_path), precision_mode=precision)
                        policy.diagnose(ort_, 'ONNX', execution_provider=str(getattr(self, 'run_type', '')))
                    else:
                        self.model_run = ConvertModel(load(self.model_path))
                        self.model_run = policy.cast_model(self.model_run)
                        self.model_run.to(self.device).eval()
                        MODEL_CACHE.set_model(self.model_path, self.model_run, size_bytes=os.path.getsize(self.model_path), precision_mode=precision)
                        policy.diagnose(self.model_run, 'PyTorch-ConvertModel', execution_provider=str(getattr(self, 'run_type', '')))

            self.running_inference_console_write()
            mix = prepare_mix(self.audio_file)
            
            source = self.demix(mix)
            
            if not self.is_vocal_split_model:
                self.cache_source((mix, source))
            self.write_to_console(DONE, base_text='')            

        mdx_net_cut = True if self.primary_stem in MDX_NET_FREQ_CUT and self.is_match_frequency_pitch else False

        if self.is_secondary_model_activated and self.secondary_model:
            self.secondary_source_primary, self.secondary_source_secondary = process_secondary_model(self.secondary_model, self.process_data, main_process_method=self.process_method, main_model_primary=self.primary_stem)
        
        if not self.is_primary_stem_only:
            secondary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.secondary_stem}).wav')
            if not isinstance(self.secondary_source, np.ndarray):
                raw_mix = self.demix(self.match_frequency_pitch(mix), is_match_mix=True) if mdx_net_cut else self.match_frequency_pitch(mix)
                self.secondary_source = spec_utils.invert_stem(raw_mix, source) if self.is_invert_spec else mix.T-source.T
            
            self.secondary_source_map = self.final_process(secondary_stem_path, self.secondary_source, self.secondary_source_secondary, self.secondary_stem, samplerate)
        
        if not self.is_secondary_stem_only:
            primary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.primary_stem}).wav')

            if not isinstance(self.primary_source, np.ndarray):
                self.primary_source = source.T
                
            self.primary_source_map = self.final_process(primary_stem_path, self.primary_source, self.secondary_source_primary, self.primary_stem, samplerate)
        
        if torch.cuda.is_available():
            print(f"[DIAGNOSTIC] SeperateMDX End before clear_gpu_cache: Allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB, Reserved: {torch.cuda.memory_reserved() / (1024**2):.2f} MB")
        clear_gpu_cache()
        if torch.cuda.is_available():
            print(f"[DIAGNOSTIC] SeperateMDX End after clear_gpu_cache: Allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB, Reserved: {torch.cuda.memory_reserved() / (1024**2):.2f} MB")

        secondary_sources = {**self.primary_source_map, **self.secondary_source_map}
        
        self.process_vocal_split_chain(secondary_sources)

        if self.is_secondary_model or self.is_pre_proc_model:
            return secondary_sources

    def initialize_model_settings(self):
        self.n_bins = self.n_fft//2+1
        self.trim = self.n_fft//2
        self.chunk_size = self.hop * (self.mdx_segment_size-1)
        self.gen_size = self.chunk_size-2*self.trim
        self.stft = STFT(self.n_fft, self.hop, self.dim_f, self.device)

    def demix(self, mix, is_match_mix=False):
        if getattr(self.model_data, 'is_adaptive_chunk', False) and not is_match_mix:
            precision = getattr(self.model_data, 'model_precision', MODEL_PRECISION_DEFAULT)
            scheduler = AdaptiveChunkScheduler(
                device=self.device,
                precision_mode=precision,
                model_size_bytes=os.stat(self.model_path).st_size,
                batch_size=self.mdx_batch_size
            )
            self.mdx_segment_size = scheduler.determine_chunk_size()
            print(f"Adaptive Chunk Scheduler: Selected segment size {self.mdx_segment_size}")

        while True:
            try:
                self.initialize_model_settings()
                
                org_mix = mix
                tar_waves_ = []

                if is_match_mix:
                    chunk_size = self.hop * (256-1)
                    overlap = 0.02
                else:
                    chunk_size = self.chunk_size
                    overlap = self.overlap_mdx
                    
                    if getattr(self.model_data, 'is_adaptive_chunk', False):
                        overlap = scheduler.adjust_overlap_ratio(self.mdx_segment_size, overlap)
                    
                    if self.is_pitch_change:
                        mix, sr_pitched = spec_utils.change_pitch_semitones(mix, 44100, semitone_shift=-self.semitone_shift)

                # Convert input to PyTorch tensor
                mix_tensor = torch.from_numpy(mix).to(torch.float32)

                gen_size = chunk_size-2*self.trim

                pad = gen_size + self.trim - ((mix_tensor.shape[-1]) % gen_size)
                
                # PyTorch-native padding
                trim_pad = torch.zeros(2, self.trim, dtype=torch.float32)
                pad_pad = torch.zeros(2, pad, dtype=torch.float32)
                mixture = torch.cat([trim_pad, mix_tensor, pad_pad], dim=1)

                step = self.chunk_size - self.n_fft if overlap == DEFAULT else int((1 - overlap) * chunk_size)
                
                # Setup result and divider buffers
                if is_match_mix or self.device == 'cpu':
                    result = torch.zeros((1, 2, mixture.shape[-1]), dtype=torch.float32)
                    divider = torch.zeros((1, 2, mixture.shape[-1]), dtype=torch.float32)
                else:
                    result = torch.zeros((1, 2, mixture.shape[-1]), dtype=torch.float32, device=self.device)
                    divider = torch.zeros((1, 2, mixture.shape[-1]), dtype=torch.float32, device=self.device)

                total = 0
                total_chunks = (mixture.shape[-1] + step - 1) // step

                # Generate slice ranges (highly memory efficient)
                slice_ranges = []
                for i in range(0, mixture.shape[-1], step):
                    start = i
                    end = min(i + chunk_size, mixture.shape[-1])
                    slice_ranges.append((start, end))

                def get_slice_tensor(idx):
                    if idx >= len(slice_ranges):
                        return None
                    start, end = slice_ranges[idx]
                    mix_part_ = mixture[:, start:end]
                    if end != start + chunk_size:
                        pad_size = (start + chunk_size) - end
                        pad_tensor = torch.zeros(2, pad_size, dtype=torch.float32)
                        mix_part_ = torch.cat([mix_part_, pad_tensor], dim=-1)
                    
                    part_tensor = mix_part_.unsqueeze(0)
                    if self.device != 'cpu':
                        part_tensor = part_tensor.pin_memory()
                    return part_tensor

                # Stream allocation
                if self.device != 'cpu' and torch.cuda.is_available() and not is_match_mix:
                    stream_transfer = torch.cuda.Stream(device=self.device)
                    stream_infer = torch.cuda.Stream(device=self.device)
                    stream_post = torch.cuda.Stream(device=self.device)
                else:
                    stream_transfer = None
                    stream_infer = None
                    stream_post = None

                # Keep track of windows and metadata
                part_metadata = []
                for idx, (start, end) in enumerate(slice_ranges):
                    chunk_size_actual = end - start
                    if overlap == 0:
                        window = None
                    else:
                        window = np.hanning(chunk_size_actual)
                        window = np.tile(window[None, None, :], (1, 2, 1))
                        if not is_match_mix and self.device != 'cpu':
                            window = torch.tensor(window, dtype=torch.float32, device=self.device)
                    part_metadata.append((start, end, chunk_size_actual, window))

                def transfer_mdx_part(part_tensor, stream):
                    if part_tensor is None:
                        return None
                    if stream is not None:
                        with torch.cuda.stream(stream):
                            return part_tensor.to(self.device, non_blocking=True)
                    else:
                        return part_tensor.to(self.device)

                num_parts = len(slice_ranges)
                total_stages = num_parts + 2

                part_gpu_transferring = None
                part_gpu_inferring = None
                tar_waves_inferring = None

                cnt_transferring = 0
                cnt_inferring = 0
                cnt_postprocessing = 0

                for stage in range(total_stages):
                    # --- Stage 3: Postprocessing ---
                    if stage >= 2 and cnt_postprocessing < num_parts:
                        start, end, chunk_size_actual, window = part_metadata[cnt_postprocessing]
                        if stream_post is not None:
                            stream_post.wait_stream(stream_infer)
                            with torch.cuda.stream(stream_post):
                                if window is not None:
                                    tar_waves_inferring[..., :chunk_size_actual] *= window 
                                    divider[..., start:end] += window
                                else:
                                    divider[..., start:end] += 1
                                result[..., start:end] += tar_waves_inferring[..., :end-start]
                        else:
                            if window is not None:
                                tar_waves_inferring[..., :chunk_size_actual] *= window 
                                divider[..., start:end] += window
                            else:
                                divider[..., start:end] += 1
                            result[..., start:end] += tar_waves_inferring[..., :end-start]
                        cnt_postprocessing += 1

                    # --- Stage 2: Inference ---
                    if stage >= 1 and cnt_inferring < num_parts:
                        self.running_inference_progress_bar(total_chunks, is_match_mix=is_match_mix)
                        if stream_infer is not None:
                            stream_infer.wait_stream(stream_transfer)
                            with torch.cuda.stream(stream_infer):
                                mix_waves = part_gpu_inferring.split(self.mdx_batch_size)
                                waves_list = []
                                for mix_wave in mix_waves:
                                    waves_list.append(self.run_model(mix_wave, is_match_mix=is_match_mix))
                                if not is_match_mix:
                                    tar_waves_inferring = torch.cat(waves_list, dim=0)
                                else:
                                    tar_waves_inferring = np.concatenate(waves_list, axis=0)
                        else:
                            mix_waves = part_gpu_inferring.split(self.mdx_batch_size)
                            waves_list = []
                            for mix_wave in mix_waves:
                                waves_list.append(self.run_model(mix_wave, is_match_mix=is_match_mix))
                            if not is_match_mix and isinstance(waves_list[0], torch.Tensor):
                                tar_waves_inferring = torch.cat(waves_list, dim=0)
                            else:
                                tar_waves_inferring = np.concatenate(waves_list, axis=0)
                        cnt_inferring += 1

                    # --- Stage 1: Transfer ---
                    if stage < num_parts:
                        part_tensor_transferring = get_slice_tensor(stage)
                        part_gpu_transferring = transfer_mdx_part(part_tensor_transferring, stream_transfer)
                    else:
                        part_gpu_transferring = None

                    # Shift buffer for next stage
                    part_gpu_inferring = part_gpu_transferring

                # Synchronize streams
                if stream_post is not None:
                    torch.cuda.synchronize(self.device)

                tar_waves = result / divider
                
                # Trim and crop in PyTorch
                tar_waves = tar_waves[:, :, self.trim:-self.trim]
                tar_waves = tar_waves[0, :, :mix.shape[-1]]
                
                source = tar_waves.cpu().detach().numpy()

                if self.is_pitch_change and not is_match_mix:
                    source = self.pitch_fix(source, sr_pitched, org_mix)

                source = source if is_match_mix else source*self.compensate

                if self.is_denoise_model and not is_match_mix:
                    if NO_STEM in self.primary_stem_native or self.primary_stem_native == INST_STEM:
                        if org_mix.shape[1] != source.shape[1]:
                            source = spec_utils.match_array_shapes(source, org_mix)
                        source = org_mix - vr_denoiser(org_mix-source, self.device, model_path=self.DENOISER_MODEL)
                    else:
                        source = vr_denoiser(source, self.device, model_path=self.DENOISER_MODEL)

                return source
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and self.mdx_segment_size > 256 and not is_match_mix:
                    print(f"CUDA Out Of Memory with segment size {self.mdx_segment_size}. Retrying with smaller segment size...")
                    self.mdx_segment_size = max(256, self.mdx_segment_size // 2)
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e

    def run_model(self, mix, is_match_mix=False):
        with torch.no_grad():
            spek = self.stft(mix.to(self.device))*self.adjust
            spek[:, :, :3, :] *= 0 

            precision = getattr(self.model_data, 'model_precision', MODEL_PRECISION_DEFAULT)
            policy = SmartPrecisionPolicy(precision)
            is_pytorch_model = isinstance(self.model_run, torch.nn.Module)

            # Pad spek to self.dim_t if it is smaller
            pad_t = 0
            if not is_match_mix and hasattr(self, 'dim_t') and spek.shape[-1] < self.dim_t:
                pad_t = self.dim_t - spek.shape[-1]
                spek = torch.nn.functional.pad(spek, (0, pad_t))

            if is_pytorch_model and not is_match_mix:
                spek = policy.prepare_spectrogram(spek)

            if is_match_mix:
                spec_pred = spek.to(torch.float32).cpu().numpy()
            else:
                with torch.cuda.amp.autocast(enabled=(policy.enabled_amp and is_pytorch_model), dtype=policy.amp_dtype):
                    spec_pred = -self.model_run(-spek)*0.5+self.model_run(spek)*0.5 if self.is_denoise else self.model_run(spek)
                if is_pytorch_model:
                    spec_pred = spec_pred.to(torch.float32)

            # Crop back to original length if padded
            if pad_t > 0:
                spec_pred = spec_pred[..., :-pad_t]

            if not isinstance(spec_pred, torch.Tensor):
                spec_pred = torch.tensor(spec_pred).to(self.device)

            if is_match_mix:
                return self.stft.inverse(spec_pred).cpu().detach().numpy()
            else:
                return self.stft.inverse(spec_pred)

class SeperateMDXC(SeperateAttributes):        

    def seperate(self):
        samplerate = 44100
        sources = None

        if self.primary_model_name == self.model_basename and isinstance(self.primary_sources, tuple):
            mix, sources = self.primary_sources
            self.load_cached_sources()
        else:
            self.start_inference_console_write()
            self.running_inference_console_write()
            mix = prepare_mix(self.audio_file)
            sources = self.demix(mix)
            if not self.is_vocal_split_model:
                self.cache_source((mix, sources))
            self.write_to_console(DONE, base_text='')

        stem_list = [self.mdx_c_configs.training.target_instrument] if self.mdx_c_configs.training.target_instrument else [i for i in self.mdx_c_configs.training.instruments]

        if self.is_secondary_model:
            if self.is_pre_proc_model:
                self.mdxnet_stem_select = stem_list[0]
            else:
                self.mdxnet_stem_select = self.main_model_primary_stem_4_stem if self.main_model_primary_stem_4_stem else self.primary_model_primary_stem
            self.primary_stem = self.mdxnet_stem_select
            self.secondary_stem = secondary_stem(self.mdxnet_stem_select)
            self.is_primary_stem_only, self.is_secondary_stem_only = False, False

        is_all_stems = self.mdxnet_stem_select == ALL_STEMS
        is_not_ensemble_master = not self.process_data['is_ensemble_master']
        is_not_single_stem = not len(stem_list) <= 2
        is_not_secondary_model = not self.is_secondary_model
        is_ensemble_4_stem = self.is_4_stem_ensemble and is_not_single_stem

        if (is_all_stems and is_not_ensemble_master and is_not_single_stem and is_not_secondary_model) or is_ensemble_4_stem and not self.is_pre_proc_model:
            for stem in stem_list:
                primary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({stem}).wav')
                self.primary_source = sources[stem].T
                self.write_audio(primary_stem_path, self.primary_source, samplerate, stem_name=stem)
                
                if stem == VOCAL_STEM and not self.is_sec_bv_rebalance:
                    self.process_vocal_split_chain({VOCAL_STEM:stem})
        else:
            if len(stem_list) == 1:
                source_primary = sources  
            else:
                source_primary = sources[stem_list[0]] if self.is_multi_stem_ensemble and len(stem_list) == 2 else sources[self.mdxnet_stem_select]
            if self.is_secondary_model_activated and self.secondary_model:
                self.secondary_source_primary, self.secondary_source_secondary = process_secondary_model(self.secondary_model, 
                                                                                                         self.process_data, 
                                                                                                         main_process_method=self.process_method, 
                                                                                                         main_model_primary=self.primary_stem)

            if not self.is_primary_stem_only:
                secondary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.secondary_stem}).wav')
                if not isinstance(self.secondary_source, np.ndarray):
                    
                    if self.is_mdx_combine_stems and len(stem_list) >= 2:
                        if len(stem_list) == 2:
                            secondary_source = sources[self.secondary_stem]
                        else:
                            sources.pop(self.primary_stem)
                            next_stem = next(iter(sources))
                            secondary_source = np.zeros_like(sources[next_stem])
                            for v in sources.values():
                                secondary_source += v
                                
                        self.secondary_source = secondary_source.T 
                    else:
                        self.secondary_source, raw_mix = source_primary, self.match_frequency_pitch(mix)
                        self.secondary_source = spec_utils.to_shape(self.secondary_source, raw_mix.shape)
                    
                        if self.is_invert_spec:
                            self.secondary_source = spec_utils.invert_stem(raw_mix, self.secondary_source)
                        else:
                            self.secondary_source = (-self.secondary_source.T+raw_mix.T)
                            
                self.secondary_source_map = self.final_process(secondary_stem_path, self.secondary_source, self.secondary_source_secondary, self.secondary_stem, samplerate)    

            if not self.is_secondary_stem_only:
                primary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.primary_stem}).wav')
                if not isinstance(self.primary_source, np.ndarray):
                    self.primary_source = source_primary.T

                self.primary_source_map = self.final_process(primary_stem_path, self.primary_source, self.secondary_source_primary, self.primary_stem, samplerate)

        clear_gpu_cache()
        
        secondary_sources = {**self.primary_source_map, **self.secondary_source_map}
        self.process_vocal_split_chain(secondary_sources)
        
        if self.is_secondary_model or self.is_pre_proc_model:
            return secondary_sources

    def demix(self, mix):
        sr_pitched = 441000
        org_mix = mix
        if self.is_pitch_change:
            mix, sr_pitched = spec_utils.change_pitch_semitones(mix, 44100, semitone_shift=-self.semitone_shift)

        precision = getattr(self.model_data, 'model_precision', MODEL_PRECISION_DEFAULT)
        policy = SmartPrecisionPolicy(precision)
        model = MODEL_CACHE.get_model(self.model_path, precision_mode=precision)
        if model is None:
            model = TFC_TDF_net(self.mdx_c_configs, device=self.device)
            model.load_state_dict(torch.load(self.model_path, map_location=cpu))
            model = policy.cast_model(model)
            model.to(self.device).eval()
            MODEL_CACHE.set_model(self.model_path, model, size_bytes=os.path.getsize(self.model_path), precision_mode=precision)

        policy.diagnose(model, 'PyTorch-MDXC', execution_provider=str(getattr(self, 'run_type', '')))

        try:
            S = model.num_target_instruments
        except Exception as e:
            S = model.module.num_target_instruments

        mdx_segment_size = self.mdx_c_configs.inference.dim_t if self.is_mdx_c_seg_def else self.mdx_segment_size
        if getattr(self.model_data, 'is_adaptive_chunk', False):
            scheduler = AdaptiveChunkScheduler(
                device=self.device,
                precision_mode=precision,
                model_size_bytes=os.stat(self.model_path).st_size,
                batch_size=self.mdx_batch_size
            )
            mdx_segment_size = scheduler.determine_chunk_size()
            print(f"Adaptive Chunk Scheduler: Selected segment size {mdx_segment_size}")

        while True:
            try:
                mix_tensor = torch.tensor(mix, dtype=torch.float32)
                batch_size = self.mdx_batch_size
                chunk_size = self.mdx_c_configs.audio.hop_length * (mdx_segment_size - 1)
                
                overlap = self.overlap_mdx23
                if getattr(self.model_data, 'is_adaptive_chunk', False):
                    overlap = int(scheduler.adjust_overlap_ratio(mdx_segment_size, overlap))
                    overlap = max(1, overlap)
                
                hop_size = chunk_size // overlap
                mix_shape = mix_tensor.shape[1]
                pad_size = hop_size - (mix_shape - chunk_size) % hop_size
                mix_tensor = torch.cat([torch.zeros(2, chunk_size - hop_size), mix_tensor, torch.zeros(2, pad_size + chunk_size - hop_size)], 1)

                chunks = mix_tensor.unfold(1, chunk_size, hop_size).transpose(0, 1)
                num_chunks = len(chunks)
                num_batches = (num_chunks + batch_size - 1) // batch_size
                
                def get_mdxc_batch(batch_idx):
                    if batch_idx >= num_batches:
                        return None
                    start = batch_idx * batch_size
                    end = min(start + batch_size, num_chunks)
                    batch = chunks[start:end]
                    if self.device != 'cpu':
                        batch = batch.pin_memory()
                    return batch

                X = torch.zeros(S, *mix_tensor.shape) if S > 1 else torch.zeros_like(mix_tensor)
                X = X.to(self.device)

                # Stream allocation
                if self.device != 'cpu' and torch.cuda.is_available():
                    stream_transfer = torch.cuda.Stream(device=self.device)
                    stream_infer = torch.cuda.Stream(device=self.device)
                    stream_post = torch.cuda.Stream(device=self.device)
                else:
                    stream_transfer = None
                    stream_infer = None
                    stream_post = None

                def transfer_mdxc_batch(batch_cpu, stream):
                    if batch_cpu is None:
                        return None
                    if stream is not None:
                        with torch.cuda.stream(stream):
                            batch_gpu = batch_cpu.to(self.device, non_blocking=True)
                            if policy.is_fp16:
                                batch_gpu = batch_gpu.half()
                            elif policy.is_bf16:
                                batch_gpu = batch_gpu.bfloat16()
                            return batch_gpu
                    else:
                        batch_gpu = batch_cpu.to(self.device)
                        if policy.is_fp16:
                            batch_gpu = batch_gpu.half()
                        elif policy.is_bf16:
                            batch_gpu = batch_gpu.bfloat16()
                        return batch_gpu

                total_stages = num_batches + 2

                batch_gpu_transferring = None
                batch_gpu_inferring = None
                x_inferring = None

                cnt_transferring = 0
                cnt_inferring = 0
                cnt_postprocessing = 0

                with torch.no_grad():
                    enabled_amp = policy.enabled_amp
                    amp_dtype = policy.amp_dtype
                    
                    for stage in range(total_stages):
                        # --- Stage 3: Postprocessing ---
                        if stage >= 2 and cnt_postprocessing < num_batches:
                            if stream_post is not None:
                                stream_post.wait_stream(stream_infer)
                                with torch.cuda.stream(stream_post):
                                    x_post = x_inferring.to(torch.float32)
                                    start_idx = cnt_postprocessing * batch_size
                                    for w_idx, w in enumerate(x_post):
                                        c_idx = start_idx + w_idx
                                        X[..., c_idx * hop_size : c_idx * hop_size + chunk_size] += w
                            else:
                                x_post = x_inferring.to(torch.float32)
                                start_idx = cnt_postprocessing * batch_size
                                for w_idx, w in enumerate(x_post):
                                    c_idx = start_idx + w_idx
                                    X[..., c_idx * hop_size : c_idx * hop_size + chunk_size] += w
                            cnt_postprocessing += 1

                        # --- Stage 2: Inference ---
                        if stage >= 1 and cnt_inferring < num_batches:
                            self.running_inference_progress_bar(num_batches)
                            if stream_infer is not None:
                                stream_infer.wait_stream(stream_transfer)
                                with torch.cuda.stream(stream_infer):
                                    with torch.cuda.amp.autocast(enabled=enabled_amp, dtype=amp_dtype):
                                        x_inferring = model(batch_gpu_inferring)
                            else:
                                with torch.cuda.amp.autocast(enabled=enabled_amp, dtype=amp_dtype):
                                    x_inferring = model(batch_gpu_inferring)
                            cnt_inferring += 1

                        # --- Stage 1: Transfer ---
                        if stage < num_batches:
                            batch_cpu_transferring = get_mdxc_batch(stage)
                            batch_gpu_transferring = transfer_mdxc_batch(batch_cpu_transferring, stream_transfer)
                        else:
                            batch_gpu_transferring = None

                        # Shift buffer for next stage
                        batch_gpu_inferring = batch_gpu_transferring

                if stream_post is not None:
                    torch.cuda.synchronize(self.device)

                estimated_sources = X[..., chunk_size - hop_size:-(pad_size + chunk_size - hop_size)] / overlap
                del X
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and mdx_segment_size > 256:
                    print(f"CUDA Out Of Memory with segment size {mdx_segment_size}. Retrying with smaller segment size...")
                    mdx_segment_size = max(256, mdx_segment_size // 2)
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e

        pitch_fix = lambda s:self.pitch_fix(s, sr_pitched, org_mix)

        if S > 1:
            sources = {k: pitch_fix(v) if self.is_pitch_change else v for k, v in zip(self.mdx_c_configs.training.instruments, estimated_sources.cpu().detach().numpy())}
            del estimated_sources
            if self.is_denoise_model:
                if VOCAL_STEM in sources.keys() and INST_STEM in sources.keys():
                    sources[VOCAL_STEM] = vr_denoiser(sources[VOCAL_STEM], self.device, model_path=self.DENOISER_MODEL)
                    if sources[VOCAL_STEM].shape[1] != org_mix.shape[1]:
                        sources[VOCAL_STEM] = spec_utils.match_array_shapes(sources[VOCAL_STEM], org_mix)
                    sources[INST_STEM] = org_mix - sources[VOCAL_STEM]
                            
            return sources
        else:
            est_s = estimated_sources.cpu().detach().numpy()
            del estimated_sources
            return pitch_fix(est_s) if self.is_pitch_change else est_s

class SeperateDemucs(SeperateAttributes):
    def seperate(self):
        samplerate = 44100
        source = None
        model_scale = None
        stem_source = None
        stem_source_secondary = None
        inst_mix = None
        inst_source = None
        is_no_write = False
        is_no_piano_guitar = False
        is_no_cache = False
        
        if self.primary_model_name == self.model_basename and isinstance(self.primary_sources, np.ndarray) and not self.pre_proc_model:
            source = self.primary_sources
            self.load_cached_sources()
        else:
            self.start_inference_console_write()
            is_no_cache = True

        mix = prepare_mix(self.audio_file)

        if is_no_cache:
            precision = getattr(self.model_data, 'model_precision', MODEL_PRECISION_DEFAULT)
            policy = SmartPrecisionPolicy(precision)
            self.demucs = MODEL_CACHE.get_model(self.model_path, precision_mode=precision)
            if self.demucs is None:
                if self.demucs_version == DEMUCS_V1:
                    if str(self.model_path).endswith(".gz"):
                        self.model_path = gzip.open(self.model_path, "rb")
                    klass, args, kwargs, state = torch.load(self.model_path)
                    self.demucs = klass(*args, **kwargs)
                    self.demucs.load_state_dict(state)
                elif self.demucs_version == DEMUCS_V2:
                    self.demucs = auto_load_demucs_model_v2(self.demucs_source_list, self.model_path)
                    self.demucs.load_state_dict(torch.load(self.model_path))
                else:  
                    self.demucs = HDemucs(sources=self.demucs_source_list)
                    self.demucs = _gm(name=os.path.splitext(os.path.basename(self.model_path))[0], 
                                      repo=Path(os.path.dirname(self.model_path)))
                    self.demucs = demucs_segments(self.segment, self.demucs)
                
                self.demucs = policy.cast_model(self.demucs)
                self.demucs.to(self.device)
                self.demucs.eval()
                MODEL_CACHE.set_model(self.model_path, self.demucs, size_bytes=os.path.getsize(self.model_path), precision_mode=precision)

            policy.diagnose(self.demucs, 'PyTorch-Demucs', execution_provider=str(getattr(self, 'run_type', '')))

            if self.pre_proc_model:
                if self.primary_stem not in [VOCAL_STEM, INST_STEM]:
                    is_no_write = True
                    self.write_to_console(DONE, base_text='')
                    mix_no_voc = process_secondary_model(self.pre_proc_model, self.process_data, is_pre_proc_model=True)
                    inst_mix = prepare_mix(mix_no_voc[INST_STEM])
                    self.process_iteration()
                    self.running_inference_console_write(is_no_write=is_no_write)
                    inst_source = self.demix_demucs(inst_mix)
                    self.process_iteration()

            self.running_inference_console_write(is_no_write=is_no_write) if not self.pre_proc_model else None
            
            if self.primary_model_name == self.model_basename and isinstance(self.primary_sources, np.ndarray) and self.pre_proc_model:
                source = self.primary_sources
            else:
                source = self.demix_demucs(mix)
            
            self.write_to_console(DONE, base_text='')
            
            del self.demucs
            clear_gpu_cache()
            
        if isinstance(inst_source, np.ndarray):
            source_reshape = spec_utils.reshape_sources(inst_source[self.demucs_source_map[VOCAL_STEM]], source[self.demucs_source_map[VOCAL_STEM]])
            inst_source[self.demucs_source_map[VOCAL_STEM]] = source_reshape
            source = inst_source

        if isinstance(source, np.ndarray):
            
            if len(source) == 2:
                self.demucs_source_map = DEMUCS_2_SOURCE_MAPPER
            else:
                self.demucs_source_map = DEMUCS_6_SOURCE_MAPPER if len(source) == 6 else DEMUCS_4_SOURCE_MAPPER

                if len(source) == 6 and self.process_data['is_ensemble_master'] or len(source) == 6 and self.is_secondary_model:
                    is_no_piano_guitar = True
                    six_stem_other_source = list(source)
                    six_stem_other_source = [i for n, i in enumerate(source) if n in [self.demucs_source_map[OTHER_STEM], self.demucs_source_map[GUITAR_STEM], self.demucs_source_map[PIANO_STEM]]]
                    other_source = np.zeros_like(six_stem_other_source[0])
                    for i in six_stem_other_source:
                        other_source += i
                    source_reshape = spec_utils.reshape_sources(source[self.demucs_source_map[OTHER_STEM]], other_source)
                    source[self.demucs_source_map[OTHER_STEM]] = source_reshape
                    
        if not self.is_vocal_split_model:
            self.cache_source(source)
        
        if (self.demucs_stems == ALL_STEMS and not self.process_data['is_ensemble_master']) or self.is_4_stem_ensemble and not self.is_return_dual:
            for stem_name, stem_value in self.demucs_source_map.items():
                if self.is_secondary_model_activated and not self.is_secondary_model and not stem_value >= 4:
                    if self.secondary_model_4_stem[stem_value]:
                        model_scale = self.secondary_model_4_stem_scale[stem_value]
                        stem_source_secondary = process_secondary_model(self.secondary_model_4_stem[stem_value], self.process_data, main_model_primary_stem_4_stem=stem_name, is_source_load=True, is_return_dual=False)
                        if isinstance(stem_source_secondary, np.ndarray):
                            stem_source_secondary = stem_source_secondary[1 if self.secondary_model_4_stem[stem_value].demucs_stem_count == 2 else stem_value].T
                        elif type(stem_source_secondary) is dict:
                            stem_source_secondary = stem_source_secondary[stem_name]
                            
                stem_source_secondary = None if stem_value >= 4 else stem_source_secondary
                stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({stem_name}).wav')
                stem_source = source[stem_value].T
                
                stem_source = self.process_secondary_stem(stem_source, secondary_model_source=stem_source_secondary, model_scale=model_scale)
                self.write_audio(stem_path, stem_source, samplerate, stem_name=stem_name)
                
                if stem_name == VOCAL_STEM and not self.is_sec_bv_rebalance:
                    self.process_vocal_split_chain({VOCAL_STEM:stem_source})
                
            if self.is_secondary_model:    
                return source
        else:
            if self.is_secondary_model_activated and self.secondary_model:
                    self.secondary_source_primary, self.secondary_source_secondary = process_secondary_model(self.secondary_model, self.process_data, main_process_method=self.process_method)
                    
            if not self.is_primary_stem_only:
                def secondary_save(sec_stem_name, source, raw_mixture=None, is_inst_mixture=False):
                    secondary_source = self.secondary_source if not is_inst_mixture else None
                    secondary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({sec_stem_name}).wav')
                    secondary_source_secondary = None
                    
                    if not isinstance(secondary_source, np.ndarray):
                        if self.is_demucs_combine_stems:
                            source = list(source)
                            if is_inst_mixture:
                                source = [i for n, i in enumerate(source) if not n in [self.demucs_source_map[self.primary_stem], self.demucs_source_map[VOCAL_STEM]]]
                            else:
                                source.pop(self.demucs_source_map[self.primary_stem])
                                
                            source = source[:len(source) - 2] if is_no_piano_guitar else source
                            secondary_source = np.zeros_like(source[0])
                            for i in source:
                                secondary_source += i
                            secondary_source = secondary_source.T
                        else:
                            if not isinstance(raw_mixture, np.ndarray):
                                raw_mixture = prepare_mix(self.audio_file)
       
                            secondary_source = source[self.demucs_source_map[self.primary_stem]]
                            
                            if self.is_invert_spec:
                                secondary_source = spec_utils.invert_stem(raw_mixture, secondary_source)
                            else:
                                raw_mixture = spec_utils.reshape_sources(secondary_source, raw_mixture)
                                secondary_source = (-secondary_source.T+raw_mixture.T)
                            
                    if not is_inst_mixture:
                        self.secondary_source = secondary_source
                        secondary_source_secondary = self.secondary_source_secondary
                        self.secondary_source = self.process_secondary_stem(secondary_source, secondary_source_secondary)
                        self.secondary_source_map = {self.secondary_stem: self.secondary_source}

                    self.write_audio(secondary_stem_path, secondary_source, samplerate, stem_name=sec_stem_name)

                secondary_save(self.secondary_stem, source, raw_mixture=mix)
                
                if self.is_demucs_pre_proc_model_inst_mix and self.pre_proc_model and not self.is_4_stem_ensemble:
                    secondary_save(f"{self.secondary_stem} {INST_STEM}", source, raw_mixture=inst_mix, is_inst_mixture=True)

            if not self.is_secondary_stem_only:
                primary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.primary_stem}).wav')
                if not isinstance(self.primary_source, np.ndarray):
                    self.primary_source = source[self.demucs_source_map[self.primary_stem]].T
                
                self.primary_source_map = self.final_process(primary_stem_path, self.primary_source, self.secondary_source_primary, self.primary_stem, samplerate)

            secondary_sources = {**self.primary_source_map, **self.secondary_source_map}
            
            self.process_vocal_split_chain(secondary_sources)
            
            if self.is_secondary_model:    
                return secondary_sources
    
    def demix_demucs(self, mix):
        
        org_mix = mix
        
        if self.is_pitch_change:
            mix, sr_pitched = spec_utils.change_pitch_semitones(mix, 44100, semitone_shift=-self.semitone_shift)
        
        processed = {}
        mix = torch.tensor(mix, dtype=torch.float32)
        ref = mix.mean(0)        
        mix = (mix - ref.mean()) / ref.std()
        mix_infer = mix 
        
        precision = getattr(self.model_data, 'model_precision', MODEL_PRECISION_DEFAULT)
        policy = SmartPrecisionPolicy(precision)
        enabled_amp = policy.enabled_amp
        amp_dtype = policy.amp_dtype
        
        mix_infer_dev = mix_infer.to(self.device)
        if policy.is_fp16:
            mix_infer_dev = mix_infer_dev.half()
        elif policy.is_bf16:
            mix_infer_dev = mix_infer_dev.bfloat16()
            
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=enabled_amp, dtype=amp_dtype):
                if self.demucs_version == DEMUCS_V1:
                    sources = apply_model_v1(self.demucs, 
                                                mix_infer_dev, 
                                                self.shifts, 
                                                self.is_split_mode,
                                                set_progress_bar=self.set_progress_bar)
                elif self.demucs_version == DEMUCS_V2:
                    sources = apply_model_v2(self.demucs, 
                                                mix_infer_dev, 
                                                self.shifts,
                                                self.is_split_mode,
                                                self.overlap,
                                                set_progress_bar=self.set_progress_bar)
                else:
                    sources = apply_model(self.demucs, 
                                            mix_infer_dev[None], 
                                            self.shifts,
                                            self.is_split_mode,
                                            self.overlap,
                                            static_shifts=1 if self.shifts == 0 else self.shifts,
                                            set_progress_bar=self.set_progress_bar,
                                            device=self.device)[0]
            sources = sources.to(torch.float32)
        
        sources = (sources * ref.std() + ref.mean()).cpu().numpy()
        sources[[0,1]] = sources[[1,0]]
        processed[mix] = sources[:,:,0:None].copy()
        sources = list(processed.values())
        sources = [s[:,:,0:None] for s in sources]
        #sources = [self.pitch_fix(s[:,:,0:None], sr_pitched, org_mix) if self.is_pitch_change else s[:,:,0:None] for s in sources]
        sources = np.concatenate(sources, axis=-1)
                     
        if self.is_pitch_change:
            sources = np.stack([self.pitch_fix(stem, sr_pitched, org_mix) for stem in sources])
                        
        return sources

class SeperateVR(SeperateAttributes):        

    def seperate(self):
        if self.primary_model_name == self.model_basename and isinstance(self.primary_sources, tuple):
            y_spec, v_spec = self.primary_sources
            self.load_cached_sources()
        else:
            self.start_inference_console_write()

            device = self.device

            nn_arch_sizes = [
                31191, # default
                33966, 56817, 123821, 123812, 129605, 218409, 537238, 537227]
            vr_5_1_models = [56817, 218409]
            model_size = math.ceil(os.stat(self.model_path).st_size / 1024)
            nn_arch_size = min(nn_arch_sizes, key=lambda x:abs(x-model_size))

            precision = getattr(self.model_data, 'model_precision', MODEL_PRECISION_DEFAULT)
            policy = SmartPrecisionPolicy(precision)
            self.model_run = MODEL_CACHE.get_model(self.model_path, precision_mode=precision)
            if self.model_run is None:
                if nn_arch_size in vr_5_1_models or self.is_vr_51_model:
                    self.model_run = nets_new.CascadedNet(self.mp.param['bins'] * 2, 
                                                          nn_arch_size, 
                                                          nout=self.model_capacity[0], 
                                                          nout_lstm=self.model_capacity[1])
                    self.is_vr_51_model = True
                else:
                    self.model_run = nets.determine_model_capacity(self.mp.param['bins'] * 2, nn_arch_size)
                                
                self.model_run.load_state_dict(torch.load(self.model_path, map_location=cpu)) 
                self.model_run = policy.cast_model(self.model_run)
                self.model_run.to(device) 
                MODEL_CACHE.set_model(self.model_path, self.model_run, size_bytes=os.path.getsize(self.model_path), precision_mode=precision)

            policy.diagnose(self.model_run, 'PyTorch-VR', execution_provider=str(getattr(self, 'run_type', '')))

            self.running_inference_console_write()
            
            if getattr(self.model_data, 'is_adaptive_chunk', False):
                scheduler = AdaptiveChunkScheduler(
                    device=device,
                    precision_mode=precision,
                    model_size_bytes=os.stat(self.model_path).st_size,
                    batch_size=self.batch_size
                )
                sched_chunk = scheduler.determine_chunk_size()
                if sched_chunk >= 4096:
                    self.window_size = 1024
                elif sched_chunk >= 2048:
                    self.window_size = 512
                else:
                    self.window_size = 320
                print(f"Adaptive Chunk Scheduler: Selected window_size {self.window_size}")
                        
            y_spec, v_spec = self.inference_vr(self.loading_mix(), device, self.aggressiveness)
            if not self.is_vocal_split_model:
                self.cache_source((y_spec, v_spec))
            self.write_to_console(DONE, base_text='')
            
        if self.is_secondary_model_activated and self.secondary_model:
            self.secondary_source_primary, self.secondary_source_secondary = process_secondary_model(self.secondary_model, self.process_data, main_process_method=self.process_method, main_model_primary=self.primary_stem)

        if not self.is_secondary_stem_only:
            primary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.primary_stem}).wav')
            if not isinstance(self.primary_source, np.ndarray):
                self.primary_source = self.spec_to_wav(y_spec).T
                if not self.model_samplerate == 44100:
                    self.primary_source = librosa.resample(self.primary_source.T, orig_sr=self.model_samplerate, target_sr=44100).T
                
            self.primary_source_map = self.final_process(primary_stem_path, self.primary_source, self.secondary_source_primary, self.primary_stem, 44100)  

        if not self.is_primary_stem_only:
            secondary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.secondary_stem}).wav')
            if not isinstance(self.secondary_source, np.ndarray):
                self.secondary_source = self.spec_to_wav(v_spec).T
                if not self.model_samplerate == 44100:
                    self.secondary_source = librosa.resample(self.secondary_source.T, orig_sr=self.model_samplerate, target_sr=44100).T
            
            self.secondary_source_map = self.final_process(secondary_stem_path, self.secondary_source, self.secondary_source_secondary, self.secondary_stem, 44100)
            
        clear_gpu_cache()
        secondary_sources = {**self.primary_source_map, **self.secondary_source_map}
        
        self.process_vocal_split_chain(secondary_sources)
        
        if self.is_secondary_model:
            return secondary_sources
            
    def loading_mix(self):

        X_wave, X_spec_s = {}, {}
        
        bands_n = len(self.mp.param['band'])
        
        audio_file = spec_utils.write_array_to_mem(self.audio_file, subtype=self.wav_type_set)
        is_mp3 = audio_file.endswith('.mp3') if isinstance(audio_file, str) else False

        for d in range(bands_n, 0, -1):        
            bp = self.mp.param['band'][d]
        
            if OPERATING_SYSTEM == 'Darwin':
                wav_resolution = 'polyphase' if SYSTEM_PROC == ARM or ARM in SYSTEM_ARCH else bp['res_type']
            else:
                wav_resolution = bp['res_type']
        
            if d == bands_n: # high-end band
                X_wave[d], _ = librosa.load(audio_file, bp['sr'], False, dtype=np.float32, res_type=wav_resolution)
                X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp['hl'], bp['n_fft'], self.mp, band=d, is_v51_model=self.is_vr_51_model)
                    
                if not np.any(X_wave[d]) and is_mp3:
                    X_wave[d] = rerun_mp3(audio_file, bp['sr'])

                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
            else: # lower bands
                X_wave[d] = librosa.resample(X_wave[d+1], self.mp.param['band'][d+1]['sr'], bp['sr'], res_type=wav_resolution)
                X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp['hl'], bp['n_fft'], self.mp, band=d, is_v51_model=self.is_vr_51_model)

            if d == bands_n and self.high_end_process != 'none':
                self.input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (self.mp.param['pre_filter_stop'] - self.mp.param['pre_filter_start'])
                self.input_high_end = X_spec_s[d][:, bp['n_fft']//2-self.input_high_end_h:bp['n_fft']//2, :]

        X_spec = spec_utils.combine_spectrograms(X_spec_s, self.mp, is_v51_model=self.is_vr_51_model)
        
        del X_wave, X_spec_s, audio_file

        return X_spec

    def inference_vr(self, X_spec, device, aggressiveness):
        def _execute(X_mag_pad, roi_size):
            X_dataset = []
            patches = (X_mag_pad.shape[2] - 2 * self.model_run.offset) // roi_size
            total_iterations = patches//self.batch_size if not self.is_tta else (patches//self.batch_size)*2
            for i in range(patches):
                start = i * roi_size
                X_mag_window = X_mag_pad[:, :, start:start + self.window_size]
                X_dataset.append(X_mag_window)

            X_dataset = np.asarray(X_dataset)
            self.model_run.eval()
            with torch.no_grad():
                mask = []
                for i in range(0, patches, self.batch_size):
                    self.progress_value += 1
                    if self.progress_value >= total_iterations:
                        self.progress_value = total_iterations
                    self.set_progress_bar(0.1, 0.8/total_iterations*self.progress_value)
                    precision = getattr(self.model_data, 'model_precision', MODEL_PRECISION_DEFAULT)
                    policy = SmartPrecisionPolicy(precision)
                    enabled_amp = policy.enabled_amp
                    amp_dtype = policy.amp_dtype
                    X_batch = X_dataset[i: i + self.batch_size]
                    X_batch = torch.from_numpy(X_batch).to(device)
                    
                    if policy.is_fp16:
                        X_batch = X_batch.half()
                    elif policy.is_bf16:
                        X_batch = X_batch.bfloat16()
                    
                    with torch.cuda.amp.autocast(enabled=enabled_amp, dtype=amp_dtype):
                        pred = self.model_run.predict_mask(X_batch)
                    if not pred.size()[3] > 0:
                        raise Exception(ERROR_MAPPER[WINDOW_SIZE_ERROR])
                    pred = pred.to(torch.float32).detach().cpu().numpy()
                    pred = np.concatenate(pred, axis=2)
                    mask.append(pred)
                if len(mask) == 0:
                    raise Exception(ERROR_MAPPER[WINDOW_SIZE_ERROR])
                
                mask = np.concatenate(mask, axis=2)
            return mask

        def postprocess(mask, X_mag, X_phase):
            is_non_accom_stem = False
            for stem in NON_ACCOM_STEMS:
                if stem == self.primary_stem:
                    is_non_accom_stem = True
                    
            mask = spec_utils.adjust_aggr(mask, is_non_accom_stem, aggressiveness)

            if self.is_post_process:
                mask = spec_utils.merge_artifacts(mask, thres=self.post_process_threshold)

            y_spec = mask * X_mag * np.exp(1.j * X_phase)
            v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)
        
            return y_spec, v_spec
        
        X_mag, X_phase = spec_utils.preprocess(X_spec)
        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, self.window_size, self.model_run.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()
        mask = _execute(X_mag_pad, roi_size)
        
        if self.is_tta:
            pad_l += roi_size // 2
            pad_r += roi_size // 2
            X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
            X_mag_pad /= X_mag_pad.max()
            mask_tta = _execute(X_mag_pad, roi_size)
            mask_tta = mask_tta[:, :, roi_size // 2:]
            mask = (mask[:, :, :n_frame] + mask_tta[:, :, :n_frame]) * 0.5
        else:
            mask = mask[:, :, :n_frame]

        y_spec, v_spec = postprocess(mask, X_mag, X_phase)
        
        return y_spec, v_spec

    def spec_to_wav(self, spec):
        if self.high_end_process.startswith('mirroring') and isinstance(self.input_high_end, np.ndarray) and self.input_high_end_h:        
            input_high_end_ = spec_utils.mirroring(self.high_end_process, spec, self.input_high_end, self.mp)
            wav = spec_utils.cmb_spectrogram_to_wave(spec, self.mp, self.input_high_end_h, input_high_end_, is_v51_model=self.is_vr_51_model)       
        else:
            wav = spec_utils.cmb_spectrogram_to_wave(spec, self.mp, is_v51_model=self.is_vr_51_model)
            
        return wav

def process_secondary_model(secondary_model: ModelData, 
                            process_data, 
                            main_model_primary_stem_4_stem=None, 
                            is_source_load=False, 
                            main_process_method=None, 
                            is_pre_proc_model=False, 
                            is_return_dual=True, 
                            main_model_primary=None):
        
    if not is_pre_proc_model:
        process_iteration = process_data['process_iteration']
        process_iteration()
    
    if secondary_model.process_method == VR_ARCH_TYPE:
        seperator = SeperateVR(secondary_model, process_data, main_model_primary_stem_4_stem=main_model_primary_stem_4_stem, main_process_method=main_process_method, main_model_primary=main_model_primary)
    if secondary_model.process_method == MDX_ARCH_TYPE:
        if secondary_model.is_mdx_c:
            seperator = SeperateMDXC(secondary_model, process_data, main_model_primary_stem_4_stem=main_model_primary_stem_4_stem, main_process_method=main_process_method, is_return_dual=is_return_dual, main_model_primary=main_model_primary)
        else:
            seperator = SeperateMDX(secondary_model, process_data, main_model_primary_stem_4_stem=main_model_primary_stem_4_stem, main_process_method=main_process_method, main_model_primary=main_model_primary)
    if secondary_model.process_method == DEMUCS_ARCH_TYPE:
        seperator = SeperateDemucs(secondary_model, process_data, main_model_primary_stem_4_stem=main_model_primary_stem_4_stem, main_process_method=main_process_method, is_return_dual=is_return_dual, main_model_primary=main_model_primary)
        
    secondary_sources = seperator.seperate()

    if type(secondary_sources) is dict and not is_source_load and not is_pre_proc_model:
        return gather_sources(secondary_model.primary_model_primary_stem, secondary_stem(secondary_model.primary_model_primary_stem), secondary_sources)
    else:
        return secondary_sources
    
def process_chain_model(secondary_model: ModelData, 
                        process_data, 
                        vocal_stem_path, 
                        master_vocal_source, 
                        master_inst_source=None):
    
    process_iteration = process_data['process_iteration']
    process_iteration()
    
    if secondary_model.bv_model_rebalance:
        vocal_source = spec_utils.reduce_mix_bv(master_inst_source, master_vocal_source, reduction_rate=secondary_model.bv_model_rebalance)
    else:
        vocal_source = master_vocal_source
    
    vocal_stem_path = [vocal_source, os.path.splitext(os.path.basename(vocal_stem_path))[0]]

    if secondary_model.process_method == VR_ARCH_TYPE:
        seperator = SeperateVR(secondary_model, process_data, vocal_stem_path=vocal_stem_path, master_inst_source=master_inst_source, master_vocal_source=master_vocal_source)
    if secondary_model.process_method == MDX_ARCH_TYPE:
        if secondary_model.is_mdx_c:
            seperator = SeperateMDXC(secondary_model, process_data, vocal_stem_path=vocal_stem_path, master_inst_source=master_inst_source, master_vocal_source=master_vocal_source)
        else:
            seperator = SeperateMDX(secondary_model, process_data, vocal_stem_path=vocal_stem_path, master_inst_source=master_inst_source, master_vocal_source=master_vocal_source)
    if secondary_model.process_method == DEMUCS_ARCH_TYPE:
        seperator = SeperateDemucs(secondary_model, process_data, vocal_stem_path=vocal_stem_path, master_inst_source=master_inst_source, master_vocal_source=master_vocal_source)
        
    secondary_sources = seperator.seperate()
    
    if type(secondary_sources) is dict:
        return secondary_sources
    else:
        return None
    
def gather_sources(primary_stem_name, secondary_stem_name, secondary_sources: dict):
    
    source_primary = False
    source_secondary = False

    for key, value in secondary_sources.items():
        if key in primary_stem_name:
            source_primary = value
        if key in secondary_stem_name:
            source_secondary = value

    return source_primary, source_secondary
        
def prepare_mix(mix):
    
    audio_path = mix

    if not isinstance(mix, np.ndarray):
        mix, sr = librosa.load(mix, mono=False, sr=44100)
    else:
        mix = mix.T

    if isinstance(audio_path, str):
        if not np.any(mix) and audio_path.endswith('.mp3'):
            mix = rerun_mp3(audio_path)

    if mix.ndim == 1:
        mix = np.asfortranarray([mix,mix])

    return mix

def rerun_mp3(audio_file, sample_rate=44100):

    with audioread.audio_open(audio_file) as f:
        track_length = int(f.duration)

    return librosa.load(audio_file, duration=track_length, mono=False, sr=sample_rate)[0]

def save_format(audio_path, save_format, mp3_bit_set):
    
    if not save_format == WAV:
        
        if OPERATING_SYSTEM == 'Darwin':
            FFMPEG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ffmpeg')
            pydub.AudioSegment.converter = FFMPEG_PATH
        
        musfile = pydub.AudioSegment.from_wav(audio_path)
        
        if save_format == FLAC:
            audio_path_flac = audio_path.replace(".wav", ".flac")
            musfile.export(audio_path_flac, format="flac")  
        
        if save_format == MP3:
            audio_path_mp3 = audio_path.replace(".wav", ".mp3")
            try:
                musfile.export(audio_path_mp3, format="mp3", bitrate=mp3_bit_set, codec="libmp3lame")
            except Exception as e:
                print(e)
                musfile.export(audio_path_mp3, format="mp3", bitrate=mp3_bit_set)
        
        try:
            os.remove(audio_path)
        except Exception as e:
            print(e)
            
def pitch_shift(mix):
    new_sr = 31183

    # Resample audio file
    resampled_audio = signal.resample_poly(mix, new_sr, 44100)
    
    return resampled_audio

def list_to_dictionary(lst):
    dictionary = {item: index for index, item in enumerate(lst)}
    return dictionary

def vr_denoiser(X, device, hop_length=1024, n_fft=2048, cropsize=256, is_deverber=False, model_path=None):
    batchsize = 4

    if is_deverber:
        nout, nout_lstm = 64, 128
        mp = ModelParameters(os.path.join('lib_v5', 'vr_network', 'modelparams', '4band_v3.json'))
        n_fft = mp.param['bins'] * 2
    else:
        mp = None
        hop_length=1024
        nout, nout_lstm = 16, 128
    
    model = nets_new.CascadedNet(n_fft, nout=nout, nout_lstm=nout_lstm)
    model.load_state_dict(torch.load(model_path, map_location=cpu))
    model.to(device)

    if mp is None:
        X_spec = spec_utils.wave_to_spectrogram_old(X, hop_length, n_fft)
    else:
        X_spec = loading_mix(X.T, mp)
   
    #PreProcess
    X_mag = np.abs(X_spec)
    X_phase = np.angle(X_spec)

    #Sep
    n_frame = X_mag.shape[2]
    pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, cropsize, model.offset)
    X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
    X_mag_pad /= X_mag_pad.max()

    X_dataset = []
    patches = (X_mag_pad.shape[2] - 2 * model.offset) // roi_size
    for i in range(patches):
        start = i * roi_size
        X_mag_crop = X_mag_pad[:, :, start:start + cropsize]
        X_dataset.append(X_mag_crop)

    X_dataset = np.asarray(X_dataset)

    model.eval()
    
    with torch.no_grad():
        mask = []
        # To reduce the overhead, dataloader is not used.
        for i in range(0, patches, batchsize):
            X_batch = X_dataset[i: i + batchsize]
            X_batch = torch.from_numpy(X_batch).to(device)

            pred = model.predict_mask(X_batch)

            pred = pred.detach().cpu().numpy()
            pred = np.concatenate(pred, axis=2)
            mask.append(pred)

        mask = np.concatenate(mask, axis=2)
    
    mask = mask[:, :, :n_frame]

    #Post Proc
    if is_deverber:
        v_spec = mask * X_mag * np.exp(1.j * X_phase)
        y_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)
    else:
        v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)

    if mp is None:
        wave = spec_utils.spectrogram_to_wave_old(v_spec, hop_length=1024)
    else:
        wave = spec_utils.cmb_spectrogram_to_wave(v_spec, mp, is_v51_model=True).T
        
    wave = spec_utils.match_array_shapes(wave, X)

    if is_deverber:
        wave_2 = spec_utils.cmb_spectrogram_to_wave(y_spec, mp, is_v51_model=True).T
        wave_2 = spec_utils.match_array_shapes(wave_2, X)
        return wave, wave_2
    else:
        return wave

def loading_mix(X, mp):

    X_wave, X_spec_s = {}, {}
    
    bands_n = len(mp.param['band'])
    
    for d in range(bands_n, 0, -1):        
        bp = mp.param['band'][d]
    
        if OPERATING_SYSTEM == 'Darwin':
            wav_resolution = 'polyphase' if SYSTEM_PROC == ARM or ARM in SYSTEM_ARCH else bp['res_type']
        else:
            wav_resolution = 'polyphase'#bp['res_type']
    
        if d == bands_n: # high-end band
            X_wave[d] = X

        else: # lower bands
            X_wave[d] = librosa.resample(X_wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=wav_resolution)
            
        X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp['hl'], bp['n_fft'], mp, band=d, is_v51_model=True)
        
        # if d == bands_n and is_high_end_process:
        #     input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (mp.param['pre_filter_stop'] - mp.param['pre_filter_start'])
        #     input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]

    X_spec = spec_utils.combine_spectrograms(X_spec_s, mp)
    
    del X_wave, X_spec_s

    return X_spec


try:
    from benchmarks.gui_runtime_recorder import install_gui_runtime_benchmark_hooks

    install_gui_runtime_benchmark_hooks(globals())
except Exception as gui_benchmark_error:
    print(f"[Benchmark warning] GUI runtime recorder unavailable: {type(gui_benchmark_error).__name__}")
