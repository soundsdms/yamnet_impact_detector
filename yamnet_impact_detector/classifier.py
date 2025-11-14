"""Impact detection using Google's YAMNet audio event classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


# NOTE: YAMNet probabilities for impact-like events are typically well below
# 0.5, so the CLI now defaults to a lower threshold (0.15) to make detections
# easier to surface.  Users can always raise the threshold if they get too many
# false positives.
DEFAULT_IMPACT_KEYWORDS = (
    "impact",
    "collision",
    "crash",
    "smash",
    "bang",
    "hit",
    "thump",
    "thud",
    "slam",
)


@dataclass
class ImpactDetectionResult:
    """Represents a single impact event detected by the classifier."""

    timestamp: float
    confidence: float


@dataclass
class ImpactDetectionSummary:
    """Aggregate information about detections produced for a waveform."""

    detections: List[ImpactDetectionResult]
    max_confidence: float


class YamnetImpactDetector:
    """Detect impact-like events in an audio waveform using YAMNet."""

    def __init__(
        self,
        model_handle: str = "https://tfhub.dev/google/yamnet/1",
        impact_keywords: Sequence[str] = DEFAULT_IMPACT_KEYWORDS,
    ) -> None:
        self._model = hub.load(model_handle)
        self._class_names = self._load_class_map()
        self._sample_rate = self._load_sample_rate()
        self._hop_seconds = self._load_hop_seconds()
        self._impact_indices = self._find_impact_indices(impact_keywords)
        if not self._impact_indices:
            raise ValueError(
                "No classes matched the provided impact keywords. "
                "Try adding more keywords or inspect the YAMNet class map."
            )

    def _load_class_map(self) -> List[str]:
        class_map_path = self._model.class_map_path().numpy().decode("utf-8")
        with tf.io.gfile.GFile(class_map_path) as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def _find_impact_indices(self, keywords: Sequence[str]) -> List[int]:
        lowered_keywords = [kw.lower() for kw in keywords]
        indices = []
        for index, name in enumerate(self._class_names):
            lower_name = name.lower()
            if any(keyword in lower_name for keyword in lowered_keywords):
                indices.append(index)
        return indices

    def _load_sample_rate(self) -> int:
        try:
            return int(self._model.sample_rate().numpy())
        except AttributeError:  # pragma: no cover - defensive fallback
            return 16000

    def _load_hop_seconds(self) -> float:
        # YAMNet exports a convenience attribute with the frame hop size.
        try:
            return float(self._model.patch_hop_seconds().numpy())
        except AttributeError:  # pragma: no cover - default from YAMNet paper
            return 0.48

    def summarize_impacts(
        self,
        waveform: np.ndarray,
        sample_rate: int | None = None,
        threshold: float = 0.15,
        smoothing_seconds: float = 0.25,
        min_gap_seconds: float = 0.2,
    ) -> ImpactDetectionSummary:
        """Run impact detection on ``waveform`` and return a summary.

        Parameters
        ----------
        waveform:
            A mono floating-point waveform normalised to the range ``[-1, 1]``.
        sample_rate:
            Sampling rate of ``waveform``.  If this differs from the YAMNet
            expected sample rate (16kHz) the waveform is resampled.
        threshold:
            Score threshold that the smoothed impact probability must exceed for
            a detection to be emitted.
        smoothing_seconds:
            Width of the moving-average smoothing window applied to the impact
            scores.  Smaller values respond faster but may be noisy.
        min_gap_seconds:
            Minimum time between successive detections.  Detections that occur
            within this window of a previous detection are merged.
        """

        if waveform.ndim != 1:
            raise ValueError("waveform must be a mono (1-D) signal")

        if sample_rate is None:
            sample_rate = self._sample_rate

        if sample_rate != self._sample_rate:
            waveform = librosa.resample(
                waveform, orig_sr=sample_rate, target_sr=self._sample_rate
            )
            sample_rate = self._sample_rate

        waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
        scores, _, _ = self._model(waveform_tensor)
        scores = scores.numpy()

        impact_scores = scores[:, self._impact_indices]
        impact_probabilities = np.max(impact_scores, axis=1)

        smoothed_scores = self._smooth_scores(impact_probabilities, smoothing_seconds)
        timestamps = self._scores_to_timestamps(
            smoothed_scores, threshold, min_gap_seconds
        )

        detections = [
            ImpactDetectionResult(timestamp=ts, confidence=float(smoothed_scores[idx]))
            for idx, ts in timestamps
        ]

        max_confidence = float(np.max(smoothed_scores)) if smoothed_scores.size else 0.0

        return ImpactDetectionSummary(detections=detections, max_confidence=max_confidence)

    def detect_impacts(
        self,
        waveform: np.ndarray,
        sample_rate: int | None = None,
        threshold: float = 0.15,
        smoothing_seconds: float = 0.25,
        min_gap_seconds: float = 0.2,
    ) -> List[ImpactDetectionResult]:
        """Return only the list of detections for backwards compatibility."""

        summary = self.summarize_impacts(
            waveform,
            sample_rate=sample_rate,
            threshold=threshold,
            smoothing_seconds=smoothing_seconds,
            min_gap_seconds=min_gap_seconds,
        )
        return summary.detections

    def _smooth_scores(self, scores: np.ndarray, smoothing_seconds: float) -> np.ndarray:
        if smoothing_seconds <= 0:
            return scores
        window_size = max(1, int(round(smoothing_seconds / self._hop_seconds)))
        window = np.ones(window_size) / window_size
        return np.convolve(scores, window, mode="same")

    def _scores_to_timestamps(
        self,
        scores: np.ndarray,
        threshold: float,
        min_gap_seconds: float,
    ) -> List[tuple[int, float]]:
        above = scores >= threshold
        transitions = np.diff(above.astype(np.int8), prepend=0)
        onset_indices = np.where(transitions == 1)[0]

        events: List[tuple[int, float]] = []
        last_ts = -np.inf
        for index in onset_indices:
            timestamp = index * self._hop_seconds
            if timestamp - last_ts < min_gap_seconds:
                continue
            events.append((index, timestamp))
            last_ts = timestamp
        return events
