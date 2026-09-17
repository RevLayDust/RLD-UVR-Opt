"""Automated unit and regression tests for UVR CLI and Benchmarking suite."""

from __future__ import annotations

import csv
import io
import json
import tempfile
import time
import unittest
from pathlib import Path

from uvr_cli.cli import build_parser, create_synthetic_audio, format_elapsed_time, normalize_cli_precision
from uvr_cli.exporter import export_benchmark_csv, export_benchmark_json, sanitize_path
from uvr_cli.model_resolver import (
    HeadlessModelData,
    build_headless_model_data,
    list_available_models,
    resolve_model_file,
)
from uvr_cli.progress import InferenceProgressBar
from uvr_cli.runner import get_existing_outputs, get_expected_stem_paths
from uvr_cli.telemetry import TelemetrySampler, collect_system_info, get_cpu_name


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

        # Test custom parameters (overlap, segment_size, batch_size)
        model_data_custom = build_headless_model_data(
            model_path=path,
            architecture=arch,
            precision="Ultra Quality (FP32)",
            device="cuda:0",
            segment_size=4000,
            overlap=0.75,
            batch_size=2,
        )
        self.assertEqual(model_data_custom.overlap, 0.75)
        self.assertEqual(model_data_custom.overlap_mdx, 0.75)
        self.assertEqual(model_data_custom.mdx_segment_size, 4000)
        self.assertEqual(model_data_custom.segment_size, 4000)
        self.assertEqual(model_data_custom.mdx_batch_size, 2)
        self.assertEqual(model_data_custom.batch_size, 2)
        self.assertEqual(model_data_custom.device_name, "cuda:0")

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
        self.assertIn("cpu_name", info)
        self.assertIn("system_ram_gb", info)
        self.assertIn("python_version", info)
        self.assertEqual(info["cpu"], info["cpu_name"])
        self.assertGreater(len(info["cpu_name"]), 0)

    def test_get_cpu_name(self):
        cpu_name = get_cpu_name()
        self.assertIsInstance(cpu_name, str)
        self.assertGreater(len(cpu_name), 0)

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
                "device": "cuda:0",
                "segment_size": 256,
                "overlap": 0.25,
                "batch_size": 1,
                "audio_file": "sample.wav",
                "audio_duration_sec": 10.0,
                "round": "1",
                "total_time_sec": 2.5,
                "inference_time_sec": 2.5,
                "speed_factor_rt": 4.0,
                "peak_torch_vram_mb": 1024.0,
                "peak_device_vram_mb": 2048.0,
                "avg_gpu_util_percent": 85.0,
                "peak_gpu_util_percent": 95.0,
                "avg_cpu_util_percent": 50.0,
                "peak_process_ram_mb": 512.0,
                "git_commit": "abc1234deadbeef",
                "python_version": "3.11.0",
                "input_hash": "sha256_dummy_hash",
                "status": "COMPLETED",
            }]
            export_benchmark_csv(dummy_rows, csv_file)
            self.assertTrue(csv_file.is_file())
            with csv_file.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["model_name"], "test_model.onnx")
            self.assertEqual(rows[0]["device"], "cuda:0")
            self.assertEqual(rows[0]["segment_size"], "256")
            self.assertEqual(rows[0]["overlap"], "0.25")
            self.assertEqual(rows[0]["batch_size"], "1")
            self.assertEqual(rows[0]["git_commit"], "abc1234deadbeef")
            self.assertEqual(rows[0]["python_version"], "3.11.0")
            self.assertEqual(rows[0]["input_hash"], "sha256_dummy_hash")

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
            "--segment-size", "4000",
            "--overlap", "0.75",
            "--batch-size", "2",
        ])
        self.assertEqual(args_bench.command, "bench")
        self.assertEqual(args_bench.rounds, 3)
        self.assertEqual(args_bench.warmup, 1)
        self.assertEqual(args_bench.segment_size, 4000)
        self.assertEqual(args_bench.overlap, 0.75)
        self.assertEqual(args_bench.batch_size, 2)

        # Test 'run' with --overwrite and -y
        args_run_ow = parser.parse_args([
            "run",
            "--audio", "test.wav",
            "--model", "UVR-MDX-NET-Inst_HQ_4.onnx",
            "--overwrite",
        ])
        self.assertTrue(args_run_ow.overwrite)

        args_run_y = parser.parse_args([
            "run",
            "-a", "test.wav",
            "-m", "UVR-MDX-NET-Inst_HQ_4.onnx",
            "-y",
        ])
        self.assertTrue(args_run_y.overwrite)

    def test_expected_stem_paths_and_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dummy_audio = temp_path / "song.wav"
            dummy_audio.touch()

            model_data = HeadlessModelData(
                primary_stem="Instrumental",
                secondary_stem="Vocals",
            )

            expected = get_expected_stem_paths(model_data, dummy_audio, temp_path)
            expected_names = [p.name for p in expected]
            self.assertIn("song_(Instrumental).wav", expected_names)
            self.assertIn("song_(Vocals).wav", expected_names)

            # Initially no files exist
            existing = get_existing_outputs(model_data, dummy_audio, temp_path)
            self.assertEqual(len(existing), 0)

            # Create one stem
            stem1 = temp_path / "song_(Instrumental).wav"
            stem1.touch()

            existing = get_existing_outputs(model_data, dummy_audio, temp_path)
            self.assertEqual(len(existing), 1)
            self.assertEqual(existing[0].name, "song_(Instrumental).wav")

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

    def test_progress_bar_renders_block_style_and_fallback(self):
        # Unicode block mode
        buf_unicode = io.StringIO()
        bar_uni = InferenceProgressBar(stream=buf_unicode, bar_width=10, ascii_only=False)
        bar_uni.update(step=0.1, inference_iterations=0.4, current_chunk=5, total_chunks=10)
        content_uni = buf_unicode.getvalue()
        self.assertIn("5/10 chunks (50.0%)", content_uni)
        # Should contain full blocks
        self.assertTrue("█" in content_uni or "#" in content_uni)

        # ASCII fallback mode
        buf_ascii = io.StringIO()
        bar_ascii = InferenceProgressBar(stream=buf_ascii, bar_width=10, ascii_only=True)
        bar_ascii.update(step=0.1, inference_iterations=0.4, current_chunk=5, total_chunks=10)
        content_ascii = buf_ascii.getvalue()
        self.assertIn("[#####-----]", content_ascii)
        self.assertIn("5/10 chunks (50.0%)", content_ascii)

    def test_progress_bar_speed_and_eta_calculation(self):
        buf = io.StringIO()
        bar = InferenceProgressBar(stream=buf, bar_width=10, ascii_only=True)
        
        # Chunk 1: speed might not be established yet or ETA --:--
        bar.update(step=0.1, inference_iterations=0.08, current_chunk=1, total_chunks=10)
        content1 = buf.getvalue()
        self.assertIn("1/10 chunks (10.0%)", content1)

        # Simulate time passing for subsequent chunks
        time.sleep(0.06)
        bar.update(step=0.1, inference_iterations=0.40, current_chunk=5, total_chunks=10)
        content2 = buf.getvalue()
        self.assertIn("5/10 chunks (50.0%)", content2)
        self.assertIn("chunks/s", content2)
        self.assertIn("ETA:", content2)

    def test_progress_bar_in_place_updates_and_no_newline_spam(self):
        buf = io.StringIO()
        bar = InferenceProgressBar(stream=buf, bar_width=10, ascii_only=True)

        for i in range(1, 5):
            bar.update(step=0.1, inference_iterations=0.2 * i, current_chunk=i, total_chunks=5)
        
        output = buf.getvalue()
        # All in-progress updates must start with \r carriage returns
        self.assertIn("\r", output)
        # Should not have finished with newline yet before reaching final chunk
        self.assertFalse(output.endswith("\n"))

        # Reach final chunk (5/5)
        bar.update(step=0.1, inference_iterations=0.8, current_chunk=5, total_chunks=5)
        final_output = buf.getvalue()
        # Must conclude with a single newline when finished
        self.assertTrue(final_output.endswith("\n"))

    def test_progress_bar_write_message_interleaving(self):
        buf = io.StringIO()
        bar = InferenceProgressBar(stream=buf, bar_width=10, ascii_only=True)
        bar.update(step=0.1, inference_iterations=0.2, current_chunk=1, total_chunks=4)

        # Interleave a console message — write_message closes the bar first
        bar.write_message("[UVR Core] Intermediate notice")
        output = buf.getvalue()
        self.assertIn("[UVR Core] Intermediate notice\n", output)
        # Bar should be closed after write_message
        self.assertTrue(bar._closed)

    def test_format_elapsed_time(self):
        # Under 60 seconds
        self.assertEqual(format_elapsed_time(0.0), "0.00s")
        self.assertEqual(format_elapsed_time(3.42), "3.42s")
        self.assertEqual(format_elapsed_time(59.99), "59.99s")
        # 60 seconds and above
        self.assertEqual(format_elapsed_time(60.0), "1m 0.00s")
        self.assertEqual(format_elapsed_time(72.38), "1m 12.38s")
        self.assertEqual(format_elapsed_time(125.5), "2m 5.50s")
        self.assertEqual(format_elapsed_time(3661.12), "61m 1.12s")

    def test_cli_startup_isolation_does_not_load_heavy_backends(self):
        """Verify that importing cli and building parser does not load heavy inference backends."""
        import subprocess
        import sys

        code = (
            "import sys; "
            "import uvr_cli.cli; "
            "parser = uvr_cli.cli.build_parser(); "
            "loaded = set(sys.modules.keys()); "
            "heavy = {'torch', 'separate', 'onnxruntime', 'librosa', 'scipy'} & loaded; "
            "assert not heavy, f'Heavy modules unexpectedly loaded: {heavy}'; "
            "print('ISOLATED_OK')"
        )
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("ISOLATED_OK", res.stdout)

    def test_lazy_package_exports(self):
        """Verify that uvr_cli exports attributes lazily and correctly."""
        import uvr_cli

        self.assertEqual(uvr_cli.__version__, "2.0.0")
        self.assertTrue(callable(uvr_cli.resolve_model_file))
        self.assertTrue(callable(uvr_cli.InferenceProgressBar))
        self.assertTrue(callable(uvr_cli.collect_system_info))
        self.assertTrue(callable(uvr_cli.BenchmarkEngine))
        self.assertIn("BenchmarkEngine", dir(uvr_cli))
        self.assertIn("execute_inference", uvr_cli.__all__)

        with self.assertRaises(AttributeError):
            _ = uvr_cli.non_existent_attribute_xyz

    def test_inference_result_structure(self):
        """Verify that InferenceResult unpacks as a 2-tuple while providing inference_time."""
        from uvr_cli.runner import InferenceResult
        from pathlib import Path

        sample_paths = [Path("test_(Vocals).wav"), Path("test_(Instrumental).wav")]
        res = InferenceResult(sample_paths, True, inference_time=3.42)

        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        stems, is_valid = res
        self.assertEqual(stems, sample_paths)
        self.assertTrue(is_valid)
        self.assertEqual(res.inference_time, 3.42)
        self.assertEqual(res.output_files, sample_paths)
        self.assertTrue(res.is_valid)


if __name__ == "__main__":
    unittest.main()

