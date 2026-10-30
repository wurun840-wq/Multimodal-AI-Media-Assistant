"""Small shared helpers: timestamp formatting, seeding, memory release.

Timestamps are numeric everywhere inside the system. They are formatted
only at the boundary where a human reads them -- a printed segment, a
citation, a report table.
"""

from __future__ import annotations

import gc
import logging
import random
from typing import Any, Iterable

LOGGER = logging.getLogger("media_assistant")


def configure_logging(level: int = logging.INFO, quiet_dependencies: bool = True) -> None:
    """Set up readable logging for scripts and notebooks.

    transformers and huggingface_hub are chatty at INFO and emit several
    warnings per Whisper call that are expected and not actionable. Our
    own logger is left at ``level`` so pipeline progress stays visible.
    """
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")
    LOGGER.setLevel(level)
    if quiet_dependencies:
        for name in ("transformers", "httpx", "huggingface_hub", "sentence_transformers", "urllib3"):
            logging.getLogger(name).setLevel(logging.ERROR)


def format_timestamp(seconds: float, decimals: int = 1) -> str:
    """Format a number of seconds as ``MM:SS.s`` (or ``HH:MM:SS.s``).

    >>> format_timestamp(95.24)
    '01:35.2'
    >>> format_timestamp(95.24, decimals=0)
    '01:35'
    """
    if seconds is None:
        return "--:--"
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if decimals > 0:
        sec_text = f"{secs:0{3 + decimals}.{decimals}f}"
    else:
        sec_text = f"{int(secs):02d}"
    if hours >= 1:
        return f"{int(hours):02d}:{int(minutes):02d}:{sec_text}"
    return f"{int(minutes):02d}:{sec_text}"


def format_span(start: float, end: float, decimals: int = 1, separator: str = "–") -> str:
    """Format a time range the way citations and transcripts show it.

    >>> format_span(95.2, 113.8)
    '01:35.2–01:53.8'
    """
    return f"{format_timestamp(start, decimals)}{separator}{format_timestamp(end, decimals)}"


def format_segment_line(segment: dict, decimals: int = 1) -> str:
    """Render one transcript segment the way the project brief shows it."""
    start = format_timestamp(segment.get("start"), decimals)
    end = format_timestamp(segment.get("end"), decimals)
    return f"[{start} → {end}] {segment.get('text', '').strip()}"


def set_seed(seed: int) -> None:
    """Seed every RNG we can reach, for reproducible runs."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass


def free_memory(device: str | None = None) -> None:
    """Release cached accelerator memory after a model has been deleted.

    Called by every ``_release_*`` method. On a 16 GB unified-memory Mac
    the recommended model stack does not fit all at once, so dropping a
    reference is not enough -- the allocator cache has to be emptied too.
    """
    gc.collect()
    try:
        import torch
    except ImportError:  # pragma: no cover
        return
    if device in (None, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if device in (None, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def process_memory_gb() -> float:
    """Resident memory of this process, in GB (best effort, 0.0 if unknown)."""
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:  # pragma: no cover - platform dependent
        return 0.0
    # macOS reports bytes, Linux reports kilobytes.
    import sys

    divisor = 1024**3 if sys.platform == "darwin" else 1024**2
    return usage / divisor


def truncate(text: str, limit: int = 160) -> str:
    """Shorten text for log lines and tables without breaking layout."""
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def coerce_float(value: Any, name: str) -> float:
    """Convert a value to float or raise a message a user can act on."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r} ({type(value).__name__})")


def total_duration(segments: Iterable[dict]) -> float:
    """Latest end timestamp across segments, or 0.0 when there are none."""
    ends = [float(s["end"]) for s in segments if s.get("end") is not None]
    return max(ends) if ends else 0.0
