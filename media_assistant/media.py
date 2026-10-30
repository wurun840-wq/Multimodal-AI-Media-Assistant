"""Part 1a -- media ingestion.

Accepts audio or video, reports what is actually inside the container,
and converts the audio track to the mono 16 kHz WAV that Whisper expects.

There is no system ``ffmpeg`` on the target machine, so every call goes
through the binary bundled with ``imageio-ffmpeg``. That also matters for
speech recognition: the transformers ASR pipeline shells out to a system
``ffmpeg`` when handed a file path, so :func:`load_audio` decodes the
audio ourselves and passes an array instead.
"""

from __future__ import annotations

import functools
import os
import re
import subprocess
from pathlib import Path

import numpy as np

from .exceptions import MediaError
from .utils import LOGGER

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_AUDIO_RE = re.compile(
    r"Stream #\d+:\d+.*?: Audio:\s*(?P<codec>[\w.]+).*?,\s*(?P<rate>\d+)\s*Hz,\s*(?P<layout>[^,]+),"
)
_VIDEO_RE = re.compile(
    r"Stream #\d+:\d+.*?: Video:\s*(?P<codec>[\w.]+).*?,\s*(?P<width>\d{2,5})x(?P<height>\d{2,5})"
)

_FATAL_MARKERS = (
    "Invalid data found when processing input",
    "moov atom not found",
    "End of file",
    "Unknown format",
)


@functools.lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    """Absolute path to the bundled ffmpeg binary."""
    try:
        import imageio_ffmpeg
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise MediaError(
            "imageio-ffmpeg is not installed; run `uv sync` before using media ingestion"
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def _require_readable(path: str | os.PathLike) -> Path:
    """Resolve a media path, with errors a user can act on."""
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Media file not found: {resolved}")
    if resolved.is_dir():
        raise MediaError(f"Expected a media file but got a directory: {resolved}")
    if resolved.stat().st_size == 0:
        raise MediaError(f"Media file is empty (0 bytes): {resolved}")
    return resolved


def _run_ffmpeg(args: list[str], *, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run the bundled ffmpeg, capturing output as text."""
    try:
        return subprocess.run(
            [ffmpeg_exe(), *args],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"ffmpeg timed out after {timeout}s: {' '.join(args)}") from exc


def probe_metadata(path: str | os.PathLike) -> dict:
    """Describe a media file without decoding it.

    Returns duration, container, and per-stream details. ``has_audio`` is
    the field callers must check: a video with no audio track cannot be
    transcribed, and that is a real case rather than a hypothetical one.

    Raises:
        FileNotFoundError: the path does not exist.
        MediaError: the file exists but is not decodable media.
    """
    resolved = _require_readable(path)
    # `ffmpeg -i` with no output writes the stream table to stderr and
    # exits non-zero by design, so the exit code is not the signal here.
    result = _run_ffmpeg(["-hide_banner", "-i", str(resolved)])
    text = result.stderr

    for marker in _FATAL_MARKERS:
        if marker in text:
            raise MediaError(
                f"Could not decode {resolved.name}: ffmpeg reported {marker!r}. "
                "The file may be corrupt or may not be a media file."
            )

    duration_match = _DURATION_RE.search(text)
    if duration_match is None:
        raise MediaError(
            f"Could not read a duration from {resolved.name}; it does not look like "
            "a media file ffmpeg can decode."
        )
    hours, minutes, seconds = duration_match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    audio = _AUDIO_RE.search(text)
    video = _VIDEO_RE.search(text)

    metadata = {
        "path": str(resolved),
        "filename": resolved.name,
        "container": resolved.suffix.lstrip(".").lower(),
        "size_bytes": resolved.stat().st_size,
        "duration_seconds": round(duration, 3),
        "has_audio": audio is not None,
        "has_video": video is not None,
    }
    if audio:
        metadata.update(
            audio_codec=audio.group("codec"),
            audio_sample_rate=int(audio.group("rate")),
            audio_layout=audio.group("layout").strip(),
        )
    if video:
        metadata.update(
            video_codec=video.group("codec"),
            video_width=int(video.group("width")),
            video_height=int(video.group("height")),
        )
    return metadata


def extract_audio(
    input_path: str | os.PathLike,
    output_wav: str | os.PathLike,
    sample_rate: int = 16_000,
) -> str:
    """Convert any supported media to mono PCM WAV at ``sample_rate``.

    Video inputs have their audio track extracted; audio inputs are
    resampled and downmixed. 16 kHz mono is Whisper's native input, so
    doing it once here keeps the ASR stage from resampling per call.

    Raises:
        FileNotFoundError: the input path does not exist.
        MediaError: the input has no audio track, or conversion failed.
    """
    metadata = probe_metadata(input_path)
    if not metadata["has_audio"]:
        raise MediaError(
            f"{metadata['filename']} has no audio track "
            f"({'video only' if metadata['has_video'] else 'no decodable streams'}); "
            "there is nothing to transcribe."
        )

    destination = Path(output_wav).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)

    result = _run_ffmpeg(
        [
            "-hide_banner", "-loglevel", "error", "-y",
            "-i", metadata["path"],
            "-vn",                      # drop video
            "-ac", "1",                 # mono
            "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            "-f", "wav",
            str(destination),
        ]
    )
    if result.returncode != 0 or not destination.exists():
        raise MediaError(
            f"Audio extraction failed for {metadata['filename']}: "
            f"{result.stderr.strip() or 'ffmpeg returned no error text'}"
        )

    LOGGER.info(
        "Extracted %.1fs of audio from %s to %s",
        metadata["duration_seconds"], metadata["filename"], destination.name,
    )
    return str(destination)


def load_audio(path: str | os.PathLike, sample_rate: int = 16_000) -> np.ndarray:
    """Decode audio to a mono float32 array at ``sample_rate``.

    The transformers ASR pipeline invokes a *system* ffmpeg when given a
    path, which does not exist here, so audio is always handed to it as
    an array produced by this function.
    """
    resolved = _require_readable(path)
    import librosa

    try:
        audio, _ = librosa.load(str(resolved), sr=sample_rate, mono=True)
    except Exception as exc:
        raise MediaError(f"Could not decode audio from {resolved.name}: {exc}") from exc

    if audio.size == 0:
        raise MediaError(f"{resolved.name} decoded to zero audio samples.")
    return audio.astype(np.float32, copy=False)


def prepare_audio(
    input_path: str | os.PathLike,
    workdir: str | os.PathLike = ".cache_media",
    sample_rate: int = 16_000,
) -> tuple[str, dict]:
    """Probe a media file and produce the WAV the ASR stage consumes.

    Returns ``(wav_path, metadata)``. Inputs that are already mono WAV at
    the target rate are used as-is instead of being rewritten.
    """
    metadata = probe_metadata(input_path)
    if not metadata["has_audio"]:
        raise MediaError(
            f"{metadata['filename']} has no audio track; there is nothing to transcribe."
        )

    already_prepared = (
        metadata["container"] == "wav"
        and metadata.get("audio_sample_rate") == sample_rate
        and metadata.get("audio_layout") == "mono"
    )
    if already_prepared:
        LOGGER.info("%s is already mono %d Hz WAV; using it directly", metadata["filename"], sample_rate)
        return metadata["path"], metadata

    destination = Path(workdir) / (Path(metadata["filename"]).stem + f".{sample_rate}.mono.wav")
    wav_path = extract_audio(metadata["path"], destination, sample_rate=sample_rate)
    return wav_path, metadata
