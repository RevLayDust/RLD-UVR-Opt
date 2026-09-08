"""Machine-readable JSON and CSV exporters for UVR benchmarks with privacy sanitization."""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

_WINDOWS_USER_PATH = re.compile(r"(?i)[A-Z]:[\\/]Users[\\/][^\\/\r\n\"']+[\\/]")
_GENERIC_ABS_PATH = re.compile(r"(?i)[A-Z]:[\\/][^\r\n\"']+")


def sanitize_path(value: Any) -> Any:
    """Sanitize paths and redact local username for safe GitHub uploads."""
    if isinstance(value, str):
        # Redact specific user profile directory while keeping project relative path
        redacted = _WINDOWS_USER_PATH.sub("<user_dir>/", value)
        return redacted.replace("\\", "/")
    if isinstance(value, dict):
        return {k: sanitize_path(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_path(v) for v in value]
    return value


def export_benchmark_json(data: Dict[str, Any], output_path: str | Path) -> Path:
    """Export benchmark result to structured machine-readable JSON."""
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = sanitize_path(data)
    with path.open("w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2, sort_keys=True)
    return path


CSV_FIELDS = [
    "session_id",
    "timestamp_utc",
    "gpu_name",
    "cpu_name",
    "model_name",
    "model_hash",
    "backend",
    "precision",
    "audio_file",
    "audio_duration_sec",
    "round",
    "total_time_sec",
    "inference_time_sec",
    "speed_factor_rt",
    "peak_torch_vram_mb",
    "peak_device_vram_mb",
    "avg_gpu_util_percent",
    "peak_gpu_util_percent",
    "avg_cpu_util_percent",
    "peak_process_ram_mb",
    "status",
]


def export_benchmark_csv(runs_summary: List[Dict[str, Any]], csv_path: str | Path) -> Path:
    """Append benchmark runs to a tabular summary CSV file."""
    path = Path(csv_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.is_file() and path.stat().st_size > 0

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()

        for run in runs_summary:
            clean_run = sanitize_path(run)
            writer.writerow(clean_run)

    return path
