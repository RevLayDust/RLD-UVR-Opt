"""Regression coverage for automatic GUI per-inference recording."""

from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace

from benchmarks.gui_runtime_recorder import (
    GUIRuntimeBenchmarkRecorder,
    install_gui_runtime_benchmark_hooks,
    sha256_file,
)


class FakeCache:
    def __init__(self):
        self.cache = {}

    def get_model(self, model_path, precision_mode=None):
        normalized = str(Path(model_path).resolve())
        entry = self.cache.get(normalized)
        if entry and entry["precision_mode"] == precision_mode:
            return entry["model_obj"]
        if entry:
            self.cache.pop(normalized)
        return None

    def set_model(self, model_path, model_obj, size_bytes=0, precision_mode=None):
        self.cache[str(Path(model_path).resolve())] = {
            "model_obj": model_obj,
            "precision_mode": precision_mode,
        }


class FakeOrtSession:
    def __init__(self, precision):
        self.precision = precision

    def get_inputs(self):
        input_type = "tensor(float16)" if "FP16" in self.precision else "tensor(float)"
        return [SimpleNamespace(type=input_type)]

    def get_providers(self):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]


class FakeSampler:
    def __init__(self, progress=None, **kwargs):
        self.progress = progress or (lambda snapshot: None)

    def start(self):
        self.progress({
            "elapsed_seconds": 0.0,
            "process_cpu_percent": 12.5,
            "gpu_utilization_percent": 75.0,
            "device_vram_mb": 1024.0,
        })

    def stop(self):
        return {
            "sampling_interval_ms": 250.0,
            "average_process_cpu_percent": 12.5,
            "peak_process_cpu_percent": 20.0,
            "average_system_cpu_percent": 10.0,
            "peak_system_cpu_percent": 15.0,
            "peak_process_ram_mb": 512.0,
            "average_gpu_utilization_percent": 75.0,
            "peak_gpu_utilization_percent": 90.0,
            "baseline_device_vram_mb": 512.0,
            "peak_device_vram_mb": 1536.0,
            "average_power_w": 100.0,
            "peak_power_w": 125.0,
            "torch_peak_allocated_vram_mb": 768.0,
            "torch_peak_reserved_vram_mb": 896.0,
            "warnings": [],
        }


class FailingSampler(FakeSampler):
    def start(self):
        raise RuntimeError("synthetic sampler failure")


class SeperateMDX:
    def __init__(self, root: Path, cache: FakeCache, precision: str):
        self.model_path = str(root / "UVR-MDX-NET-Test.onnx")
        self.audio_file = str(root / "input.wav")
        self.audio_file_base = "input"
        self.export_path = str(root / "outputs")
        self.model_data = SimpleNamespace(model_precision=precision, device_set="0")
        self.device = "cuda:0"
        self.process_data = {}
        self.cache = cache
        self.messages = []

    def write_to_console(self, message, **kwargs):
        self.messages.append(str(message))

    def demix(self):
        return "separated"

    def write_audio(self):
        output = Path(self.export_path)
        output.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output / "input_(Vocals).wav"), "wb") as stream:
            stream.setnchannels(2)
            stream.setsampwidth(2)
            stream.setframerate(44100)
            stream.writeframes(b"\0\0\0\0" * 4410)

    def seperate(self):
        precision = self.model_data.model_precision
        model = self.cache.get_model(self.model_path, precision_mode=precision)
        if model is None:
            model = FakeOrtSession(precision)
            self.cache.set_model(self.model_path, model, precision_mode=precision)
        self.demix()
        self.write_audio()
        return "done"


def fake_environment(device_index: int):
    return {
        "hardware": {
            "os": "Test Windows",
            "cpu": "Test CPU",
            "gpu_name": "Test NVIDIA GPU",
        },
        "software": {
            "python_version": "3.11",
            "torch_version": "test",
        },
        "warnings": [],
    }


class GUIRuntimeRecorderTests(unittest.TestCase):
    def tearDown(self):
        method = SeperateMDX.seperate
        original = getattr(method, "_uvr_gui_benchmark_original", None)
        if original is not None:
            SeperateMDX.seperate = original

    def test_blocks_are_separate_and_model_round_resets_on_precision_change(self):
        project_root = Path(__file__).resolve().parents[1]
        uvr_hash_before = sha256_file(project_root / "UVR.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "UVR-MDX-NET-Test.onnx").write_bytes(b"model")
            with wave.open(str(root / "input.wav"), "wb") as stream:
                stream.setnchannels(2)
                stream.setsampwidth(2)
                stream.setframerate(44100)
                stream.writeframes(b"\0\0\0\0" * 4410)
            cache = FakeCache()
            namespace = {
                "SeperateMDX": SeperateMDX,
                "MODEL_CACHE": cache,
                "prepare_mix": lambda value: value,
            }
            recorder = GUIRuntimeBenchmarkRecorder(
                namespace,
                output_root=root / "runtime",
                sampler_factory=FakeSampler,
                environment_collector=fake_environment,
                progress_interval_seconds=1.0,
            )
            install_gui_runtime_benchmark_hooks(namespace, recorder=recorder)

            first = SeperateMDX(root, cache, "Performance (FP16)")
            second = SeperateMDX(root, cache, "Performance (FP16)")
            third = SeperateMDX(root, cache, "Ultra Quality (FP32)")
            self.assertEqual(first.seperate(), "done")
            self.assertEqual(second.seperate(), "done")
            self.assertEqual(third.seperate(), "done")

            session_name = (root / "runtime" / "latest.txt").read_text(encoding="utf-8").strip()
            session = root / "runtime" / session_name
            combined = (session / "benchmark.log").read_text(encoding="utf-8")
            records = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted(session.glob("inference_*.json"))
            ]

            self.assertIn("1st Inference:", combined)
            self.assertIn("2nd Inference:", combined)
            self.assertIn("3rd Inference:", combined)
            self.assertEqual([record["model_round"] for record in records], [1, 2, 1])
            self.assertEqual([record["status"] for record in records], ["completed"] * 3)
            self.assertEqual(records[0]["runtime"]["effective_precision"], "fp16")
            self.assertEqual(records[2]["runtime"]["effective_precision"], "fp32")
            self.assertIn("Stage inference:", combined)
            self.assertIn("Total-time:", combined)
            self.assertGreaterEqual(combined.count("======="), 4)
            self.assertTrue(any("[Benchmark] Recording" in message for message in first.messages))

        self.assertEqual(sha256_file(project_root / "UVR.py"), uvr_hash_before)

    def test_sampler_setup_failure_keeps_processing_and_closes_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "UVR-MDX-NET-Test.onnx").write_bytes(b"model")
            with wave.open(str(root / "input.wav"), "wb") as stream:
                stream.setnchannels(2)
                stream.setsampwidth(2)
                stream.setframerate(44100)
                stream.writeframes(b"\0\0\0\0" * 100)
            cache = FakeCache()
            prepare_mix = lambda value: value
            namespace = {
                "SeperateMDX": SeperateMDX,
                "MODEL_CACHE": cache,
                "prepare_mix": prepare_mix,
            }
            recorder = GUIRuntimeBenchmarkRecorder(
                namespace,
                output_root=root / "runtime",
                sampler_factory=FailingSampler,
                environment_collector=fake_environment,
            )
            install_gui_runtime_benchmark_hooks(namespace, recorder=recorder)

            separator = SeperateMDX(root, cache, "Performance (FP16)")
            result = separator.seperate()

            session_name = (root / "runtime" / "latest.txt").read_text(encoding="utf-8").strip()
            session = root / "runtime" / session_name
            record_path = next(session.glob("inference_*.json"))
            record = json.loads(record_path.read_text(encoding="utf-8"))
            text_log = next(session.glob("inference_*.log")).read_text(encoding="utf-8")
            self.assertEqual(result, "done")
            self.assertEqual(record["status"], "completed")
            self.assertTrue(any("Recorder setup failed" in warning for warning in record["warnings"]))
            self.assertIn("Warning: Recorder setup failed: RuntimeError", text_log)
            self.assertIn("Status: COMPLETED", text_log)
            self.assertIn("Total-time:", text_log)
            self.assertIs(namespace["prepare_mix"], prepare_mix)
            self.assertNotIn("demix", separator.__dict__)


if __name__ == "__main__":
    unittest.main()
