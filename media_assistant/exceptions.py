"""Exception types.

Every failure mode the system can reach on purpose has a named type, so
callers can tell "the file is broken" apart from "the model produced
something we refuse to accept" without matching on message strings.
"""

from __future__ import annotations


class MediaAssistantError(Exception):
    """Base class for every error raised by this package."""


class MediaError(MediaAssistantError):
    """The input media could not be read, decoded, or used.

    Covers a missing audio track, an unreadable container, and a file
    that is not media at all. A missing path raises FileNotFoundError
    instead, since that is what callers already expect.
    """


class TranscriptionError(MediaAssistantError):
    """Speech recognition ran but produced nothing usable."""


class AnalysisValidationError(MediaAssistantError):
    """The LLM analysis failed schema validation after every retry.

    Raised rather than returning a partial dictionary: silently accepting
    malformed structured output is exactly the failure this project is
    supposed to guard against.
    """

    def __init__(self, message: str, *, attempts: int = 0, problems: list[str] | None = None):
        super().__init__(message)
        self.attempts = attempts
        self.problems = problems or []


class ToolError(MediaAssistantError):
    """A tool was called with arguments it cannot accept.

    Inside the agent loop this is caught and turned into an observation
    so the model can correct itself; it never aborts the run.
    """


class AgentError(MediaAssistantError):
    """The agent loop itself failed (not an individual tool)."""
