"""Command line interface for running the YAMNet impact detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .audio_utils import extract_audio_waveform
from .classifier import (
    DEFAULT_IMPACT_KEYWORDS,
    ImpactDetectionResult,
    YamnetImpactDetector,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to the input video file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help=(
            "Score threshold for classifying an impact (default: 0.15). "
            "Raise this if you receive too many false positives."
        ),
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.25,
        help="Smoothing window size in seconds (default: 0.25)",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.2,
        help="Minimum gap between successive detections in seconds (default: 0.2)",
    )
    parser.add_argument(
        "--keywords",
        nargs="*",
        default=list(DEFAULT_IMPACT_KEYWORDS),
        help="Optional override for the list of impact keywords",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to a JSON file where detections will be written",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output when --output is provided",
    )
    return parser.parse_args(argv)


def format_detections(detections: Sequence[ImpactDetectionResult]) -> list[dict[str, float]]:
    return [
        {"timestamp": float(det.timestamp), "confidence": float(det.confidence)}
        for det in detections
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    audio = extract_audio_waveform(args.video)
    detector = YamnetImpactDetector(impact_keywords=args.keywords)
    summary = detector.summarize_impacts(
        audio.waveform,
        sample_rate=audio.sample_rate,
        threshold=args.threshold,
        smoothing_seconds=args.smoothing,
        min_gap_seconds=args.min_gap,
    )

    detections = summary.detections
    detection_dicts = format_detections(detections)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        indent = 2 if args.pretty else None
        args.output.write_text(json.dumps(detection_dicts, indent=indent))
    else:
        for det in detection_dicts:
            print(f"{det['timestamp']:.3f}s\tconfidence={det['confidence']:.3f}")

    if not detections:
        print(
            "No impacts exceeded the threshold. Max confidence was"
            f" {summary.max_confidence:.3f}. Consider lowering --threshold "
            "or adjusting --keywords if you expected detections.",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
