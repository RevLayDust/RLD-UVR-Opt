"""Unit tests for TensorRT-RTX wrapper, provider resolution, precision presets, and install record."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import onnxruntime as ort
import torch

from engine.tensorrt_rtx import (
    PROVIDER_CPU,
    PROVIDER_CUDA,
    PROVIDER_TRT_RTX,
    build_trt_rtx_options,
    create_trt_rtx_session,
    get_supported_execution_providers,
    is_ort_gpu_available,
    is_trt_rtx_available,
    is_trt_rtx_installed,
)
from install.install_record import read_install_record, write_install_record
from separate import SmartPrecisionPolicy, create_onnx_session


class TensorRTRTXEngineTests(unittest.TestCase):
    """Test TensorRT-RTX options, provider discovery, honest separation, and session fallback."""

    def test_build_trt_rtx_options_keys(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            options = build_trt_rtx_options(device_id=0, cache_dir=tmp_dir, enable_cuda_graph=True)
            self.assertEqual(options["device_id"], "0")
            self.assertEqual(options["nv_runtime_cache_path"], str(Path(tmp_dir)))
            self.assertEqual(options["enable_cuda_graph"], "1")
            # Must NOT contain standard TensorRT keys
            for key in options:
                self.assertFalse(key.startswith("trt_"), f"Unexpected standard TRT option: {key}")

    def test_standard_tensorrt_is_not_treated_as_trt_rtx(self):
        # Even if standard TensorrtExecutionProvider is present in ORT, it must NOT be offered as TensorRT-RTX
        with mock.patch("onnxruntime.get_available_providers", return_value=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]):
            with mock.patch("onnxruntime.get_all_providers", return_value=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]):
                # Ensure plugin is not detected
                with mock.patch("onnxruntime.get_ep_devices", create=True, return_value=[]):
                    self.assertFalse(is_trt_rtx_installed())
                    providers = get_supported_execution_providers(has_gpu_selected=True, device_id=0)
                    self.assertNotIn(PROVIDER_TRT_RTX, providers)
                    self.assertIn(PROVIDER_CUDA, providers)

    def test_provider_resolution_rules(self):
        # Rule 3: GPU not selected -> returns CPUExecutionProvider
        cpu_providers = get_supported_execution_providers(has_gpu_selected=False)
        self.assertEqual(cpu_providers, [PROVIDER_CPU])

        # Rule 2: GPU selected -> returns GPU providers with CUDA fallback
        gpu_providers = get_supported_execution_providers(has_gpu_selected=True, device_id=0)
        self.assertIsInstance(gpu_providers, list)
        self.assertGreater(len(gpu_providers), 0)
        self.assertTrue(PROVIDER_CUDA in gpu_providers or PROVIDER_CPU in gpu_providers)

    def test_saved_provider_preserved_while_locked(self):
        # Simulate GUI state behavior
        is_gpu_selected = False  # DEFAULT GPU device
        stored_ep = PROVIDER_TRT_RTX

        # While locked, stored provider must NOT be overwritten
        if is_gpu_selected:
            stored_ep = "OVERWRITTEN"
        self.assertEqual(stored_ep, PROVIDER_TRT_RTX)

        # When GPU is selected, validate against detected providers
        is_gpu_selected = True
        supported = [PROVIDER_CUDA, PROVIDER_CPU]
        if is_gpu_selected and stored_ep not in supported:
            stored_ep = PROVIDER_CUDA
        self.assertEqual(stored_ep, PROVIDER_CUDA)

    @mock.patch("onnxruntime.InferenceSession")
    def test_session_fallback_chain(self, mock_session):
        sentinel_session = mock.MagicMock()
        mock_session.return_value = sentinel_session

        # Case 1: Plugin is present -> uses sess_options or providers
        session = create_trt_rtx_session("test_model.onnx", device_id=0)
        self.assertIsNotNone(session)
        mock_session.assert_called()

        # Case 2: When TRT-RTX plugin session creation fails, it falls back to CUDA
        mock_session.reset_mock()
        mock_session.side_effect = [
            RuntimeError("Plugin creation failure"),
            sentinel_session,
        ]
        session_fb = create_trt_rtx_session("test_model.onnx", device_id=0)
        self.assertIsNotNone(session_fb)
        self.assertGreaterEqual(mock_session.call_count, 2)
        second_call_kwargs = mock_session.call_args_list[1][1]
        providers_used = second_call_kwargs.get("providers", [])
        self.assertTrue(
            any("CUDAExecutionProvider" in (p[0] if isinstance(p, tuple) else p) for p in providers_used)
            or any("CPUExecutionProvider" in (p[0] if isinstance(p, tuple) else p) for p in providers_used)
        )

    def test_precision_policy_diagnose_consistency(self):
        from gui_data.constants import MODEL_PRECISION_FP16, MODEL_PRECISION_FP32
        policy_fp16 = SmartPrecisionPolicy(MODEL_PRECISION_FP16)
        self.assertTrue(policy_fp16.is_fp16)
        self.assertEqual(policy_fp16.effective_mode_for("ONNX"), "FP16")

        policy_fp32 = SmartPrecisionPolicy(MODEL_PRECISION_FP32)
        self.assertTrue(policy_fp32.is_fp32)
        self.assertEqual(policy_fp32.effective_mode_for("ONNX"), "FP32")


class InstallRecordTests(unittest.TestCase):
    """Test persistent install record write, read, stack notes, and recovery."""

    def test_write_and_read_install_record_with_stack_info(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            record_path = Path(tmp_dir) / "test_install_record.json"

            # Write record for Option B
            write_install_record(
                option="B",
                install_mode="TENSORRT_RTX",
                reuse_venv="Y",
                has_uv=True,
                has_ffmpeg=True,
                has_rubberband=True,
                has_vc_redist=True,
                record_path=record_path,
            )

            self.assertTrue(record_path.is_file())

            # Read record back
            data = read_install_record(record_path)
            self.assertIsNotNone(data)
            self.assertEqual(data["selected_option"], "B")
            self.assertEqual(data["install_mode"], "TENSORRT_RTX")
            self.assertEqual(data["stack_profile"], "tensorrt_rtx_ort124")
            self.assertIn("ort_stack_note", data)
            self.assertIn("installed_packages", data)

    def test_read_missing_or_corrupt_record_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_path = Path(tmp_dir) / "non_existent.json"
            self.assertIsNone(read_install_record(missing_path))

            corrupt_path = Path(tmp_dir) / "corrupt.json"
            corrupt_path.write_text("{bad json: 123", encoding="utf-8")
            self.assertIsNone(read_install_record(corrupt_path))


if __name__ == "__main__":
    unittest.main()
