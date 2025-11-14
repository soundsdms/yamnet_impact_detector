"""Utility helpers for working with audio extracted from video files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from moviepy.editor import VideoFileClip


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
    """Extract the audio waveform from ``video_path``.

    The audio stream is resampled to ``target_sample_rate`` which is the
    expected input rate for YAMNet.  By default the audio is converted to mono
    because YAMNet expects a single-channel waveform.
    """

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    with VideoFileClip(str(path)) as clip:
        audio_clip = clip.audio
        if audio_clip is None:
            raise ValueError(f"No audio track found in {path}")
        fps = audio_clip.fps
        audio_array = audio_clip.to_soundarray(fps=fps)

    if audio_array.ndim == 2 and mono:
        audio_array = np.mean(audio_array, axis=1)
    elif audio_array.ndim == 2:
        audio_array = audio_array.T

    waveform = np.asarray(audio_array, dtype=np.float32)

    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=0)

    if fps != target_sample_rate:
        waveform = librosa.resample(waveform, orig_sr=fps, target_sr=target_sample_rate)

    return AudioExtractionResult(waveform=waveform.astype(np.float32), sample_rate=target_sample_rate)
