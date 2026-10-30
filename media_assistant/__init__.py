"""Multimodal AI media assistant.

A small end-to-end system that turns a 3-8 minute recording into a
timestamped transcript, a validated structured analysis, a semantic
index, grounded question answering with timestamp citations, a bounded
tool-using agent, and a multilingual extension.

The package is deliberately import-cheap: heavy dependencies (torch,
transformers, sentence-transformers) are only pulled in when the
component that needs them is first touched.
"""

from __future__ import annotations

import os

# Must be set before torch is imported anywhere. Some operators used by
# Whisper and Qwen are not implemented for the MPS backend; without this
# they raise instead of quietly running on the CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Tokenizers' Rust parallelism warns loudly once a fork happens (e.g. the
# ffmpeg subprocess). We tokenize in a single process, so turn it off.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .config import Config  # noqa: E402
from .exceptions import (  # noqa: E402
    AgentError,
    AnalysisValidationError,
    MediaAssistantError,
    MediaError,
    ToolError,
)

__version__ = "1.0.0"

__all__ = [
    "Config",
    "MediaAssistant",
    "MediaAssistantError",
    "MediaError",
    "AnalysisValidationError",
    "ToolError",
    "AgentError",
]

_LAZY = {
    "MediaAssistant": ".assistant",
}


def __getattr__(name: str):
    """Import the heavy facade only when it is actually requested."""
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
