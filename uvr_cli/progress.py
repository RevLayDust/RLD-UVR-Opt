"""Inference progress display with block-style progress bar, speed, and ETA."""

from __future__ import annotations

import shutil
import sys
import time
from typing import Any, Optional, TextIO

# Unicode block characters and safe ASCII fallbacks
BLOCK_FULL = "█"
BLOCK_EMPTY = "░"
ASCII_FULL = "#"
ASCII_EMPTY = "-"


def _can_encode_stream(stream: TextIO, text: str) -> bool:
    """Check if stream encoding supports the specified characters."""
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def _init_stream_encoding(stream: TextIO) -> None:
    """Ensure stream is configured for UTF-8 when supported."""
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


class InferenceProgressBar:
    """Terminal progress bar display for UVR model inference chunks."""

    def __init__(
        self,
        stream: Optional[TextIO] = None,
        bar_width: Optional[int] = None,
        prefix: str = "Progress: ",
        ascii_only: bool = False,
    ) -> None:
        self.stream = stream or sys.stdout
        _init_stream_encoding(self.stream)

        # Select block characters based on encoding capability
        if ascii_only or not _can_encode_stream(self.stream, BLOCK_FULL + BLOCK_EMPTY):
            self.char_full = ASCII_FULL
            self.char_empty = ASCII_EMPTY
        else:
            self.char_full = BLOCK_FULL
            self.char_empty = BLOCK_EMPTY

        self.custom_bar_width = bar_width
        self.prefix = prefix

        self.current_chunk: int = 0
        self.total_chunks: int = 0
        self.last_ratio: float = 0.0

        self.start_time: Optional[float] = None
        self.first_chunk_time: Optional[float] = None
        self.first_chunk_num: int = 0
        self.last_update_time: float = 0.0

        self.speed_chunks_per_sec: Optional[float] = None
        self.eta_sec: Optional[float] = None

        self._active: bool = False
        self._closed: bool = False
        self._last_line_len: int = 0

    def _determine_bar_width(self) -> int:
        """Compute visual bar width adapted to terminal columns."""
        if self.custom_bar_width is not None and self.custom_bar_width > 0:
            return self.custom_bar_width

        try:
            cols = shutil.get_terminal_size((80, 20)).columns
        except Exception:
            cols = 80

        if cols >= 90:
            return 25
        elif cols >= 75:
            return 20
        else:
            return 15

    def _format_time(self, seconds: float) -> str:
        """Format seconds into MM:SS or HH:MM:SS string."""
        if seconds < 0 or seconds != seconds:  # NaN check
            return "--:--"
        total_sec = int(round(seconds))
        if total_sec < 3600:
            m = total_sec // 60
            s = total_sec % 60
            return f"{m:02d}:{s:02d}"
        else:
            h = total_sec // 3600
            m = (total_sec % 3600) // 60
            s = total_sec % 60
            return f"{h:d}:{m:02d}:{s:02d}"

    def update(
        self,
        step: float = 0.0,
        inference_iterations: float = 0.0,
        current_chunk: int = 0,
        total_chunks: int = 0,
        **kwargs: Any,
    ) -> None:
        """Update progress state from UVR inference callbacks."""
        if self._closed:
            return

        now = time.perf_counter()

        # Handle completion signal from UVR (step >= 0.95 or 1.0)
        if step >= 0.95 and inference_iterations == 0.0:
            if self._active and not self._closed:
                # Render 100% completion bar if it was active
                if self.total_chunks > 0:
                    self.current_chunk = self.total_chunks
                    self._render_line(now)
                self.close()
            return

        # Skip updates before inference chunk processing starts
        if inference_iterations <= 0.0:
            return

        self._active = True

        # Calculate chunk numbers if not provided directly
        if current_chunk > 0:
            self.current_chunk = current_chunk
        else:
            self.current_chunk += 1

        if total_chunks > 0:
            if self.total_chunks == 0:
                self.total_chunks = total_chunks
            else:
                self.total_chunks = max(self.total_chunks, self.current_chunk)
        elif self.total_chunks == 0:
            ratio = inference_iterations / 0.8
            if ratio > 0.0 and self.current_chunk > 0:
                self.total_chunks = max(self.current_chunk, round(self.current_chunk / ratio))
        else:
            self.total_chunks = max(self.total_chunks, self.current_chunk)

        # Speed and ETA estimation
        if self.start_time is None:
            self.start_time = now
            self.first_chunk_time = now
            self.first_chunk_num = self.current_chunk
            self.speed_chunks_per_sec = None
            self.eta_sec = None
        else:
            elapsed = now - self.start_time
            chunks_processed = self.current_chunk - self.first_chunk_num + 1

            if elapsed >= 0.05 and chunks_processed > 0:
                self.speed_chunks_per_sec = chunks_processed / elapsed
            else:
                self.speed_chunks_per_sec = None

            # Estimate ETA only when speed can be estimated reliably
            if (
                self.speed_chunks_per_sec is not None
                and self.speed_chunks_per_sec > 0
                and self.total_chunks > 0
            ):
                if self.current_chunk >= self.total_chunks:
                    self.eta_sec = 0.0
                else:
                    remaining_chunks = max(0, self.total_chunks - self.current_chunk)
                    self.eta_sec = remaining_chunks / self.speed_chunks_per_sec
            else:
                self.eta_sec = None

        self._render_line(now)

        # Automatically finalize if last chunk completed
        if self.total_chunks > 0 and self.current_chunk >= self.total_chunks:
            self.close()

    def _render_line(self, now: float) -> None:
        """Render single line in terminal using carriage return without spamming newlines."""
        self.last_update_time = now

        bar_width = self._determine_bar_width()

        if self.total_chunks > 0:
            fraction = min(1.0, max(0.0, self.current_chunk / self.total_chunks))
            pct = fraction * 100.0
            filled_len = int(round(bar_width * fraction))
            filled_len = min(bar_width, max(0, filled_len))
            empty_len = bar_width - filled_len
            bar_visual = self.char_full * filled_len + self.char_empty * empty_len
            chunk_info = f"{self.current_chunk}/{self.total_chunks} chunks ({pct:.1f}%)"
        else:
            bar_visual = self.char_empty * bar_width
            chunk_info = f"{self.current_chunk} chunks"

        # Speed string
        if self.speed_chunks_per_sec is not None and self.speed_chunks_per_sec > 0:
            speed_str = f"{self.speed_chunks_per_sec:.2f} chunks/s"
        else:
            speed_str = "-- chunks/s"

        # ETA string
        if self.eta_sec is not None:
            eta_str = f"ETA: {self._format_time(self.eta_sec)}"
        else:
            eta_str = "ETA: --:--"

        content = f"{self.prefix}[{bar_visual}] {chunk_info} | {speed_str} | {eta_str}"

        # Overwrite previous line using carriage return with trailing space padding
        line_len = len(content)
        pad = " " * max(0, self._last_line_len - line_len)
        self._last_line_len = max(self._last_line_len, line_len)

        try:
            self.stream.write(f"\r{content}{pad}")
            self.stream.flush()
        except Exception:
            pass

    def write_message(self, message: str) -> None:
        """Print an external log message cleanly without corrupting the progress line."""
        if self._active and not self._closed:
            # Conclude progress bar cleanly before printing following messages
            self.close()

        print(message, file=self.stream)
        if hasattr(self.stream, "flush"):
            try:
                self.stream.flush()
            except Exception:
                pass

    def close(self) -> None:
        """Finalize progress bar display by appending a newline."""
        if self._closed:
            return
        self._closed = True
        if self._active:
            try:
                self.stream.write("\n")
                self.stream.flush()
            except Exception:
                pass

    def __call__(
        self,
        step: float = 0.0,
        inference_iterations: float = 0.0,
        current_chunk: int = 0,
        total_chunks: int = 0,
        **kwargs: Any,
    ) -> None:
        """Make class callable as progress_callback."""
        self.update(
            step=step,
            inference_iterations=inference_iterations,
            current_chunk=current_chunk,
            total_chunks=total_chunks,
            **kwargs,
        )
