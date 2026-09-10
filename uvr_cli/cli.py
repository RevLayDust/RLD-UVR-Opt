"""Command-Line Interface for UVR CLI-Only inference and performance benchmarking."""

from __future__ import annotations

import argparse
import sys
import time
import tempfile
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from gui_data.constants import (
    MODEL_PRECISION_BF16,
    MODEL_PRECISION_DEFAULT,
    MODEL_PRECISION_FP16,
    MODEL_PRECISION_FP32,
    MODEL_PRECISION_OPTIONS,
)
from uvr_cli.engine import BenchmarkEngine
from uvr_cli.model_resolver import (
    build_headless_model_data,
    list_available_models,
    resolve_model_file,
)
from uvr_cli.progress import InferenceProgressBar
from uvr_cli.runner import execute_inference, get_existing_outputs

# Map shorthand precision arguments to UVR canonical precision strings
PRECISION_MAP = {
    "fp32": MODEL_PRECISION_FP32,
    "fp16": MODEL_PRECISION_FP16,
    "bf16": MODEL_PRECISION_BF16,
}


def normalize_cli_precision(prec_arg: Optional[str]) -> str:
    """Map user CLI input (e.g. 'fp16') to UVR canonical precision name."""
    if not prec_arg:
        return MODEL_PRECISION_DEFAULT
    clean = prec_arg.strip().lower()
    if clean in PRECISION_MAP:
        return PRECISION_MAP[clean]
    # Check if user passed the full canonical string
    for opt in MODEL_PRECISION_OPTIONS:
        if clean == opt.lower():
            return opt
    return MODEL_PRECISION_DEFAULT


def format_elapsed_time(seconds: float) -> str:
    """Format elapsed seconds into a human-readable string.

    Under 60s:  '3.42s'
    60s or more: '1m 12.38s'
    """
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    remaining = seconds - (minutes * 60)
    return f"{minutes}m {remaining:.2f}s"


def create_synthetic_audio(duration_sec: float = 2.0, samplerate: int = 44100) -> Path:
    """Create a temporary sine-wave test WAV file for smoke testing."""
    temp_dir = Path(tempfile.mkdtemp(prefix="uvr_smoke_test_"))
    audio_path = temp_dir / "synthetic_test_input.wav"

    total_samples = int(duration_sec * samplerate)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    # 440 Hz (A4) sine wave in stereo
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    stereo_signal = np.vstack((signal, signal)).T
    int_signal = (stereo_signal * 32767).astype(np.int16)

    with wave.open(str(audio_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(int_signal.tobytes())

    return audio_path


def handle_list_models(args: argparse.Namespace) -> int:
    """List all installed UVR models."""
    models = list_available_models()
    if not models:
        print("No models found in models/ directories.")
        return 0

    print(f"\n{'='*75}")
    print(f" INSTALLED UVR MODELS ({len(models)} found)")
    print(f"{'='*75}")
    print(f"{'Architecture':<15} | {'Size (MB)':<10} | {'Model Name'}")
    print(f"{'-'*15}-+-{'-'*10}-+-{'-'*45}")
    for m in models:
        print(f"{m['architecture']:<15} | {m['size_mb']:<10.2f} | {m['name']}")
    print(f"{'='*75}\n")
    return 0


def handle_run(args: argparse.Namespace) -> int:
    """Perform CLI-only separation directly through UVR core."""
    model_path, arch = resolve_model_file(args.model)
    precision = normalize_cli_precision(args.precision)

    print(f"Resolving model: {model_path.name} ({arch})")
    model_data = build_headless_model_data(
        model_path=model_path,
        architecture=arch,
        precision=precision,
        device=args.device,
        segment_size=args.segment_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )

    audio_path = Path(args.audio).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not audio_path.is_file():
        print(f"[ERROR] Input audio file not found: {audio_path}")
        return 1

    print(f"Separating     : {audio_path.name}")
    print(f"Precision      : {precision}")
    print(f"Output folder  : {output_dir}")

    # Check for existing conflicting output stems
    existing_outputs = get_existing_outputs(model_data, audio_path, output_dir)
    overwrite = getattr(args, "overwrite", False)

    if existing_outputs and not overwrite:
        print(f"\n[!] Output file(s) already exist:")
        for f in existing_outputs:
            print(f"  -> {f.name}")

        if not sys.stdin.isatty():
            print(f"\n[ERROR] Output files already exist. Use '--overwrite' or '-y' to overwrite.")
            return 1

        try:
            choice = input("\nOverwrite? [Y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Operation cancelled.")
            return 0

        if choice in ("y", "yes"):
            overwrite = True
        else:
            print("[INFO] Separation cancelled. Existing files were preserved.")
            return 0

    progress_bar = InferenceProgressBar()

    def console_handler(text: str, base: str = "") -> None:
        progress_bar.write_message(f"[UVR Core] {text}")

    t_start = time.perf_counter()
    stems, is_valid = execute_inference(
        model_data=model_data,
        audio_path=audio_path,
        export_dir=output_dir,
        progress_callback=progress_bar,
        console_callback=console_handler,
        overwrite=overwrite,
    )
    t_elapsed = time.perf_counter() - t_start

    if is_valid and stems:
        print(f"\n[OK] Separation completed successfully! Generated {len(stems)} stem(s):")
        for stem in stems:
            print(f"  -> {stem.name} ({round(stem.stat().st_size / (1024*1024), 2)} MB)")
        print(f"\nProcess complete")
        print(f"Time elapsed: {format_elapsed_time(t_elapsed)}")
        return 0
    else:
        print(f"\n[ERROR] Separation completed but no valid outputs were generated.")
        return 1


def handle_bench(args: argparse.Namespace) -> int:
    """Run real performance benchmark on UVR models."""
    model_path, arch = resolve_model_file(args.model)
    precision = normalize_cli_precision(args.precision)

    model_data = build_headless_model_data(
        model_path=model_path,
        architecture=arch,
        precision=precision,
        device=args.device,
        segment_size=args.segment_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )

    device_idx = int(args.device.split(":")[-1]) if ":" in args.device else 0

    engine = BenchmarkEngine(
        model_data=model_data,
        audio_path=args.audio,
        results_dir=args.export_dir,
        rounds=args.rounds,
        warmup=args.warmup,
        device_index=device_idx,
    )

    results = engine.run(verbose=True)
    return 0 if results["summary"]["rounds_measured"] > 0 else 1


def handle_smoke_test(args: argparse.Namespace) -> int:
    """Run end-to-end smoke test with synthetic audio."""
    print("Generating 2.0-second synthetic stereo audio for smoke test...")
    audio_path = create_synthetic_audio(duration_sec=2.0)

    try:
        # Select model: use specified or pick first available
        if args.model:
            model_target = args.model
        else:
            models = list_available_models()
            if not models:
                print("[ERROR] No models installed to test with. Download a model first.")
                return 1
            model_target = models[0]["path"]

        model_path, arch = resolve_model_file(model_target)
        precision = normalize_cli_precision(args.precision)

        print(f"Selected smoke test model: {model_path.name} ({arch})")

        model_data = build_headless_model_data(
            model_path=model_path,
            architecture=arch,
            precision=precision,
            device=args.device,
        )

        device_idx = int(args.device.split(":")[-1]) if ":" in args.device else 0

        engine = BenchmarkEngine(
            model_data=model_data,
            audio_path=audio_path,
            results_dir=args.export_dir,
            rounds=1,
            warmup=1,
            device_index=device_idx,
        )

        results = engine.run(verbose=True)
        if results["summary"]["rounds_measured"] > 0:
            print("\n[OK] Smoke test passed successfully!")
            return 0
        else:
            print("\n[FAIL] Smoke test completed without successful rounds.")
            return 1

    finally:
        # Clean up synthetic audio file and its folder
        try:
            if audio_path.is_file():
                audio_path.unlink()
            if audio_path.parent.is_dir():
                audio_path.parent.rmdir()
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(
        prog="uvr_cli",
        description="Ultimate Vocal Remover - CLI-Only Inference & Performance Benchmarking Suite",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: list-models
    subparsers.add_parser("list-models", help="List all available installed models")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run UVR separation directly via CLI (No GUI)")
    run_parser.add_argument("--audio", "-a", required=True, help="Path to input audio file")
    run_parser.add_argument("--model", "-m", required=True, help="Model name, filename, or absolute path")
    run_parser.add_argument(
        "--precision",
        "-p",
        default="fp16",
        help="Inference precision: fp32, fp16, bf16 (default: fp16)",
    )
    run_parser.add_argument("--device", "-d", default="cuda:0", help="Target device: cuda:0, cpu (default: cuda:0)")
    run_parser.add_argument("--output-dir", "-o", default="separated_outputs", help="Output directory for stems")
    run_parser.add_argument("--segment-size", type=int, default=None, help="Segment size (MDX chunk size)")
    run_parser.add_argument("--overlap", type=float, default=None, help="Overlap ratio (e.g. 0.25)")
    run_parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size (default: 1)")
    run_parser.add_argument(
        "--overwrite",
        "-y",
        action="store_true",
        help="Overwrite existing output stems without prompting",
    )

    # Command: bench
    bench_parser = subparsers.add_parser("bench", help="Run real benchmark with hardware telemetry & metric export")
    bench_parser.add_argument("--audio", "-a", required=True, help="Path to input audio file")
    bench_parser.add_argument("--model", "-m", required=True, help="Model name, filename, or absolute path")
    bench_parser.add_argument(
        "--precision",
        "-p",
        default="fp16",
        help="Inference precision: fp32, fp16, bf16 (default: fp16)",
    )
    bench_parser.add_argument("--device", "-d", default="cuda:0", help="Target device: cuda:0, cpu (default: cuda:0)")
    bench_parser.add_argument("--rounds", "-r", type=int, default=3, help="Number of measurement rounds (default: 3)")
    bench_parser.add_argument("--warmup", "-w", type=int, default=1, help="Number of warmup rounds (default: 1)")
    bench_parser.add_argument(
        "--export-dir",
        "-e",
        default="benchmark_results",
        help="Directory to save JSON & CSV reports (default: benchmark_results)",
    )
    bench_parser.add_argument("--segment-size", type=int, default=None, help="Segment size (MDX chunk size)")
    bench_parser.add_argument("--overlap", type=float, default=None, help="Overlap ratio (e.g. 0.25)")
    bench_parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size (default: 1)")

    # Command: smoke-test
    smoke_parser = subparsers.add_parser("smoke-test", help="Run automated smoke test using synthetic audio")
    smoke_parser.add_argument("--model", "-m", default=None, help="Optional model name (defaults to first available)")
    smoke_parser.add_argument("--precision", "-p", default="fp16", help="Inference precision (default: fp16)")
    smoke_parser.add_argument("--device", "-d", default="cuda:0", help="Target device: cuda:0, cpu (default: cuda:0)")
    smoke_parser.add_argument(
        "--export-dir",
        "-e",
        default="benchmark_results",
        help="Directory to save smoke test reports (default: benchmark_results)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "list-models":
        sys.exit(handle_list_models(args))
    elif args.command == "run":
        sys.exit(handle_run(args))
    elif args.command == "bench":
        sys.exit(handle_bench(args))
    elif args.command == "smoke-test":
        sys.exit(handle_smoke_test(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
