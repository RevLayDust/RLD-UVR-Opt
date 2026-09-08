"""Automated unit and regression tests for UVR CLI and Benchmarking suite."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from uvr_cli.cli import build_parser, create_synthetic_audio, normalize_cli_precision
from uvr_cli.exporter import export_benchmark_csv, export_benchmark_json, sanitize_path
from uvr_cli.model_resolver import (
    HeadlessModelData,
    build_headless_model_data,
    list_available_models,
    resolve_model_file,
)
from uvr_cli.telemetry import TelemetrySampler, collect_system_info


class BenchmarkCLITests(unittest.TestCase):
    """Test coverage for model resolution, telemetry, CLI parsing, and export."""

    def test_model_discovery_finds_installed_models(self):
        models = list_available_models()
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0, "Should detect at least one installed UVR model")
        first = models[0]
        self.assertIn("name", first)
        self.assertIn("architecture", first)
        self.assertIn("path", first)

    def test_model_resolution_and_headless_data(self):
        models = list_available_models()
        target = models[0]["name"]
        path, arch = resolve_model_file(target)
        self.assertTrue(path.is_file())

        model_data = build_headless_model_data(
            model_path=path,
            architecture=arch,
            precision="Performance (FP16)",
            device="cpu",
        )
        self.assertIsInstance(model_data, HeadlessModelData)
        self.assertEqual(model_data.process_method, arch)
        self.assertEqual(model_data.model_precision, "Performance (FP16)")
        self.assertIsNotNone(model_data.model_hash)

    def test_telemetry_sampler_lifecycle(self):
        sampler = TelemetrySampler(device_index=0, interval_sec=0.05)
        sampler.start()
        # Brief pause to collect samples
        import time
        time.sleep(0.2)
        metrics = sampler.stop()

        self.assertIn("avg_process_cpu_percent", metrics)
        self.assertIn("peak_process_cpu_percent", metrics)
        self.assertIn("peak_process_ram_mb", metrics)
        self.assertIn("samples_count", metrics)
        self.assertGreaterEqual(metrics["samples_count"], 1)

    def test_system_info_collection(self):
        info = collect_system_info(device_index=0)
        self.assertIn("os", info)
        self.assertIn("cpu", info)
        self.assertIn("system_ram_gb", info)
        self.assertIn("python_version", info)

    def test_path_sanitization(self):
        raw_path = r"C:\Users\JohnDoe\AppData\Local\test.wav"
        sanitized = sanitize_path(raw_path)
        self.assertNotIn("JohnDoe", sanitized)
        self.assertIn("<user_dir>/", sanitized)

    def test_export_json_and_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            json_file = temp_path / "test_session.json"
            csv_file = temp_path / "summary.csv"

            dummy_data = {
                "session_id": "test_123",
                "system_info": {"os": "TestOS"},
                "summary": {"avg_total_time_sec": 1.23},
            }
            export_benchmark_json(dummy_data, json_file)
            self.assertTrue(json_file.is_file())
            with json_file.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["session_id"], "test_123")

            dummy_rows = [{
                "session_id": "test_123",
                "timestamp_utc": "2026-09-06T00:00:00Z",
                "gpu_name": "TestGPU",
                "cpu_name": "TestCPU",
                "model_name": "test_model.onnx",
                "model_hash": "abc123hash",
                "backend": "MDX-Net",
                "precision": "Performance (FP16)",
                "audio_file": "sample.wav",
                "audio_duration_sec": 10.0,
                "round": "1",
                "cache_state": "cold",
                "total_time_sec": 2.5,
                "inference_time_sec": 2.5,
                "speed_factor_rt": 4.0,
                "peak_torch_vram_mb": 1024.0,
                "peak_device_vram_mb": 2048.0,
                "avg_gpu_util_percent": 85.0,
                "peak_gpu_util_percent": 95.0,
                "avg_cpu_util_percent": 50.0,
                "peak_process_ram_mb": 512.0,
                "status": "COMPLETED",
            }]
            export_benchmark_csv(dummy_rows, csv_file)
            self.assertTrue(csv_file.is_file())
            with csv_file.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["model_name"], "test_model.onnx")

    def test_cli_argument_parsing(self):
        parser = build_parser()

        # Test 'list-models'
        args_list = parser.parse_args(["list-models"])
        self.assertEqual(args_list.command, "list-models")

        # Test 'run'
        args_run = parser.parse_args([
            "run",
            "--audio", "test.wav",
            "--model", "UVR-MDX-NET-Inst_HQ_4.onnx",
            "--precision", "fp16",
            "--device", "cuda:0",
        ])
        self.assertEqual(args_run.command, "run")
        self.assertEqual(args_run.precision, "fp16")

        # Test 'bench'
        args_bench = parser.parse_args([
            "bench",
            "--audio", "test.wav",
            "--model", "UVR-MDX-NET-Inst_HQ_4.onnx",
            "--rounds", "3",
            "--warmup", "1",
        ])
        self.assertEqual(args_bench.command, "bench")
        self.assertEqual(args_bench.rounds, 3)
        self.assertEqual(args_bench.warmup, 1)

    def test_synthetic_audio_creation(self):
        audio_path = create_synthetic_audio(duration_sec=1.0, samplerate=44100)
        try:
            self.assertTrue(audio_path.is_file())
            self.assertGreater(audio_path.stat().st_size, 1000)
        finally:
            if audio_path.is_file():
                audio_path.unlink()
            if audio_path.parent.is_dir():
                audio_path.parent.rmdir()


if __name__ == "__main__":
    unittest.main()
