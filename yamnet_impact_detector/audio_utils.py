"""Utility helpers for working with audio extracted from video files."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AudioExtractionResult:
    """Container for extracted audio data."""

    waveform: np.ndarray
    sample_rate: int


def extract_audio_waveform(
    video_path: str | Path,
    target_sample_rate: int = 16000,
    mono: bool = True,
) -> AudioExtractionResult:
    """Extract the audio waveform from ``video_path`` using ffmpeg.

    The audio stream is decoded with ``ffmpeg`` and resampled to
    ``target_sample_rate`` which is the expected input rate for YAMNet.  By
    default the audio is converted to mono because YAMNet expects a
    single-channel waveform.
    """

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required to extract audio. "
            "Please install it and ensure it is available on your PATH."
        )

    if mono:
        channels = 1
    else:
        channels = _probe_audio_channels(path)

    ffmpeg_command = [
        ffmpeg,
        "-nostdin",
        "-i",
        str(path),
        "-vn",
        "-acodec",
        "pcm_f32le",
        "-ac",
        str(channels),
        "-ar",
        str(target_sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]

    try:
        process = subprocess.run(
            ffmpeg_command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to extract audio with ffmpeg. "
            f"Command output: {exc.stderr.decode(errors='ignore')}"
        ) from exc

    audio_bytes = process.stdout
    if not audio_bytes:
        raise RuntimeError("No audio data was produced when decoding the video")

    waveform = np.frombuffer(audio_bytes, dtype=np.float32)

    if channels > 1 and not mono:
        waveform = waveform.reshape(-1, channels).T
    else:
        waveform = waveform.reshape(-1)

    waveform = waveform.astype(np.float32, copy=False)

    return AudioExtractionResult(waveform=waveform, sample_rate=target_sample_rate)


def _probe_audio_channels(path: Path) -> int:
    """Return the number of channels in the first audio stream of ``path``."""

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        # Fall back to mono if ffprobe is unavailable.
        return 1

    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if process.returncode != 0:
        return 1

    output = process.stdout.decode().strip()
    try:
        return max(1, int(output))
    except ValueError:
        return 1
