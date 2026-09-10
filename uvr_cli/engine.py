"""Benchmarking engine managing warmup, measurement rounds, timing, and export."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from .exporter import export_benchmark_csv, export_benchmark_json
from .model_resolver import HeadlessModelData
from .runner import execute_inference, get_audio_metadata
from .telemetry import TelemetrySampler, collect_system_info, sync_cuda


def _get_git_commit() -> str:
    """Return the current Git commit SHA, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _compute_file_sha256(file_path: Path) -> str:
    """Compute a SHA-256 hash of a file."""
    sha = hashlib.sha256()
    try:
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()
    except Exception:
        return "unknown"


class BenchmarkEngine:
    """Orchestrates end-to-end performance benchmarking for UVR inference."""

    def __init__(
        self,
        model_data: HeadlessModelData,
        audio_path: str | Path,
        results_dir: str | Path = "benchmark_results",
        rounds: int = 1,
        warmup: int = 0,
        device_index: int = 0,
    ) -> None:
        self.model_data = model_data
        self.audio_path = Path(audio_path).resolve()
        self.results_dir = Path(results_dir).resolve()
        self.rounds = max(1, rounds)
        self.warmup = max(0, warmup)
        self.device_index = device_index

        self.session_id = f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """Run warmup and measurement rounds, gather telemetry, and export results."""
        if verbose:
            print(f"\n{'='*70}")
            print(f" UVR BENCHMARK SESSION: {self.session_id}")
            print(f"{'='*70}")
            print(f"Model        : {self.model_data.model_name}")
            print(f"Backend      : {self.model_data.process_method}")
            print(f"Precision    : {self.model_data.model_precision}")
            print(f"Audio File   : {self.audio_path.name}")
            print(f"Rounds       : {self.rounds} (Warmup: {self.warmup})")
            print(f"Device Index : {self.device_index}")
            print(f"{'='*70}\n")

        system_info = collect_system_info(self.device_index)
        audio_info = get_audio_metadata(self.audio_path)

        git_commit = _get_git_commit()
        python_version = sys.version.split()[0]
        input_hash = _compute_file_sha256(self.audio_path)

        all_runs_data: List[Dict[str, Any]] = []
        csv_rows: List[Dict[str, Any]] = []

        total_iterations = self.warmup + self.rounds
        current_iter = 0

        temp_stems_dir = self.results_dir / "temp_stems" / self.session_id
        temp_stems_dir.mkdir(parents=True, exist_ok=True)

        try:
            for i in range(total_iterations):
                current_iter += 1
                is_warmup_run = current_iter <= self.warmup
                round_number = current_iter if is_warmup_run else (current_iter - self.warmup)
                run_label = f"Warmup {round_number}" if is_warmup_run else f"Measurement Round {round_number}/{self.rounds}"

                if verbose:
                    print(f"--> Starting {run_label}...")

                round_export_dir = temp_stems_dir / f"round_{current_iter}"
                round_export_dir.mkdir(parents=True, exist_ok=True)

                sampler = TelemetrySampler(device_index=self.device_index, interval_sec=0.1)
                sync_cuda(self.device_index)

                sampler.start()
                start_time = time.perf_counter()

                stems, is_valid = execute_inference(
                    model_data=self.model_data,
                    audio_path=self.audio_path,
                    export_dir=round_export_dir,
                )

                sync_cuda(self.device_index)
                end_time = time.perf_counter()
                telemetry = sampler.stop()

                total_time = round(end_time - start_time, 4)
                duration_sec = audio_info.get("duration_sec", 0.0)
                speed_factor = round(duration_sec / total_time, 2) if total_time > 0 and duration_sec > 0 else 0.0

                run_record = {
                    "iteration": current_iter,
                    "is_warmup": is_warmup_run,
                    "round": round_number,
                    "total_time_sec": total_time,
                    "speed_factor_rt": speed_factor,
                    "is_valid": is_valid,
                    "output_stems": [s.name for s in stems],
                    "telemetry": telemetry,
                }
                all_runs_data.append(run_record)

                csv_row = {
                    "session_id": self.session_id,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "gpu_name": system_info.get("gpu_name", "N/A"),
                    "cpu_name": system_info.get("cpu_name") or system_info.get("cpu", "N/A"),
                    "model_name": self.model_data.model_name,
                    "model_hash": self.model_data.model_hash,
                    "backend": self.model_data.process_method,
                    "precision": self.model_data.model_precision,
                    "device": getattr(self.model_data, "device_name", "cpu"),
                    "segment_size": getattr(self.model_data, "mdx_segment_size", None),
                    "overlap": getattr(self.model_data, "overlap", None),
                    "batch_size": getattr(self.model_data, "mdx_batch_size", 1),
                    "audio_file": self.audio_path.name,
                    "audio_duration_sec": duration_sec,
                    "round": f"warmup_{round_number}" if is_warmup_run else str(round_number),
                    "total_time_sec": total_time,
                    "inference_time_sec": total_time,  # Pure separator execution
                    "speed_factor_rt": speed_factor,
                    "peak_torch_vram_mb": telemetry.get("torch_peak_allocated_vram_mb", 0.0),
                    "peak_device_vram_mb": telemetry.get("peak_device_vram_mb", 0.0),
                    "avg_gpu_util_percent": telemetry.get("avg_gpu_utilization_percent", 0.0),
                    "peak_gpu_util_percent": telemetry.get("peak_gpu_utilization_percent", 0.0),
                    "avg_cpu_util_percent": telemetry.get("avg_process_cpu_percent", 0.0),
                    "peak_process_ram_mb": telemetry.get("peak_process_ram_mb", 0.0),
                    "git_commit": git_commit,
                    "python_version": python_version,
                    "input_hash": input_hash,
                    "status": "COMPLETED" if is_valid else "VALIDATION_FAILED",
                }
                csv_rows.append(csv_row)

                if verbose:
                    print(
                        f"    Time: {total_time:.2f}s ({speed_factor}x RT) | "
                        f"Peak Torch VRAM: {telemetry.get('torch_peak_allocated_vram_mb', 0):.1f} MiB | "
                        f"Peak Device VRAM: {telemetry.get('peak_device_vram_mb', 0):.1f} MiB | "
                        f"GPU Util: {telemetry.get('avg_gpu_utilization_percent', 0):.1f}% (Peak: {telemetry.get('peak_gpu_utilization_percent', 0):.1f}%)"
                    )

        finally:
            # Clean up temporary stems after benchmarking
            try:
                shutil.rmtree(temp_stems_dir, ignore_errors=True)
            except Exception:
                pass

        # Compute summary metrics across non-warmup rounds (or all if only warmup was run)
        measured_runs = [r for r in all_runs_data if not r["is_warmup"]]
        if not measured_runs:
            measured_runs = all_runs_data

        measured_times = [r["total_time_sec"] for r in measured_runs]
        measured_speed_factors = [r["speed_factor_rt"] for r in measured_runs]
        measured_torch_vram = [r["telemetry"].get("torch_peak_allocated_vram_mb", 0.0) for r in measured_runs]
        measured_device_vram = [r["telemetry"].get("peak_device_vram_mb", 0.0) for r in measured_runs]

        summary = {
            "rounds_measured": len(measured_runs),
            "warmup_rounds": self.warmup,
            "avg_total_time_sec": round(sum(measured_times) / len(measured_times), 4),
            "min_total_time_sec": min(measured_times),
            "max_total_time_sec": max(measured_times),
            "avg_speed_factor_rt": round(sum(measured_speed_factors) / len(measured_speed_factors), 2),
            "peak_torch_allocated_vram_mb": max(measured_torch_vram) if measured_torch_vram else 0.0,
            "peak_device_vram_mb": max(measured_device_vram) if measured_device_vram else 0.0,
        }

        report_data: Dict[str, Any] = {
            "session_id": self.session_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "system_info": system_info,
            "audio_info": audio_info,
            "model_info": {
                "name": self.model_data.model_name,
                "hash": self.model_data.model_hash,
                "backend": self.model_data.process_method,
                "precision": self.model_data.model_precision,
                "device": getattr(self.model_data, "device_name", "cpu"),
                "segment_size": getattr(self.model_data, "mdx_segment_size", None),
                "overlap": getattr(self.model_data, "overlap", None),
                "batch_size": getattr(self.model_data, "mdx_batch_size", 1),
            },
            "git_commit": git_commit,
            "python_version": python_version,
            "input_hash": input_hash,
            "summary": summary,
            "runs": all_runs_data,
        }

        # Export JSON and CSV
        json_file = self.results_dir / f"{self.session_id}.json"
        csv_file = self.results_dir / "summary.csv"

        export_benchmark_json(report_data, json_file)
        export_benchmark_csv(csv_rows, csv_file)

        if verbose:
            print(f"\n{'='*70}")
            print(f" BENCHMARK SUMMARY")
            print(f"{'='*70}")
            print(f"Avg Time     : {summary['avg_total_time_sec']:.2f}s (Min: {summary['min_total_time_sec']:.2f}s, Max: {summary['max_total_time_sec']:.2f}s)")
            print(f"Speed Factor : {summary['avg_speed_factor_rt']:.2f}x real-time")
            print(f"Peak VRAM    : Torch: {summary['peak_torch_allocated_vram_mb']:.1f} MiB | Device: {summary['peak_device_vram_mb']:.1f} MiB")
            print(f"JSON Report  : {json_file}")
            print(f"CSV Summary  : {csv_file}")
            print(f"{'='*70}\n")

        return report_data
