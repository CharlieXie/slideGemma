"""Scene-change detection and temporal segmentation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """A contiguous segment of the video between two detected transitions."""
    index: int
    start_time: float
    end_time: float
    representative_frame: Image.Image
    frame_path: Optional[str] = None


_CMP_SIZE = (160, 120)


def compute_frame_diff(frame1: Image.Image, frame2: Image.Image) -> float:
    """Return the mean absolute pixel difference normalised to [0, 1]."""
    g1 = np.asarray(frame1.resize(_CMP_SIZE).convert("L"), dtype=np.float32)
    g2 = np.asarray(frame2.resize(_CMP_SIZE).convert("L"), dtype=np.float32)
    return float(np.mean(np.abs(g1 - g2)) / 255.0)


def detect_segments(
    frames: list[Image.Image],
    timestamps: list[float],
    threshold: float = 0.15,
    min_duration: float = 2.0,
    representative: str = "first",
) -> list[Segment]:
    """Split *frames* at points where the visual difference exceeds *threshold*.

    *representative* selects the frame representing each segment:
      ``"first"`` -- first frame after transition (slides)
      ``"last"``  -- last frame (whiteboards)
      ``"middle"``-- middle frame
    """
    if not frames:
        return []

    change_points = [0]

    for i in range(1, len(frames)):
        diff = compute_frame_diff(frames[i - 1], frames[i])
        if diff > threshold:
            elapsed = timestamps[i] - timestamps[change_points[-1]]
            if elapsed >= min_duration:
                change_points.append(i)

    segments: list[Segment] = []
    for seg_idx in range(len(change_points)):
        start_idx = change_points[seg_idx]
        end_idx = (change_points[seg_idx + 1] - 1
                   if seg_idx + 1 < len(change_points)
                   else len(frames) - 1)

        if representative == "last":
            rep_idx = end_idx
        elif representative == "middle":
            rep_idx = (start_idx + end_idx) // 2
        else:
            rep_idx = min(start_idx + 1, end_idx) if start_idx < end_idx else start_idx

        segments.append(Segment(
            index=seg_idx,
            start_time=timestamps[start_idx],
            end_time=timestamps[end_idx],
            representative_frame=frames[rep_idx],
        ))

    logger.info("Detected %d segments (threshold=%.3f)", len(segments), threshold)
    return segments


def time_based_segments(
    frames: list[Image.Image],
    timestamps: list[float],
    interval: float = 30.0,
) -> list[Segment]:
    """Create segments at fixed *interval* seconds -- useful for videos with
    gradual changes (whiteboard, teacher-only)."""
    if not frames:
        return []

    segments: list[Segment] = []
    seg_start = 0

    for i, ts in enumerate(timestamps):
        if ts - timestamps[seg_start] >= interval:
            segments.append(Segment(
                index=len(segments),
                start_time=timestamps[seg_start],
                end_time=timestamps[i - 1],
                representative_frame=frames[i - 1],
            ))
            seg_start = i

    segments.append(Segment(
        index=len(segments),
        start_time=timestamps[seg_start],
        end_time=timestamps[-1],
        representative_frame=frames[-1],
    ))

    logger.info("Created %d time-based segments (interval=%.0fs)", len(segments), interval)
    return segments


def adaptive_detect(
    frames: list[Image.Image],
    timestamps: list[float],
    initial_threshold: float = 0.15,
    min_segments: int = 3,
    max_segments: int = 100,
    min_duration: float = 2.0,
    representative: str = "first",
) -> list[Segment]:
    """Try progressively lower thresholds until at least *min_segments* are
    found, then clamp at *max_segments*.  Falls back to time-based sampling
    when the video is long but visually static."""
    video_duration = (timestamps[-1] - timestamps[0]) if timestamps else 0.0

    segments = detect_segments(frames, timestamps, initial_threshold,
                               min_duration, representative)

    if len(segments) < min_segments and video_duration > 60:
        for thr in (0.10, 0.07, 0.05, 0.03):
            segments = detect_segments(frames, timestamps, thr,
                                       min_duration, representative)
            if len(segments) >= min_segments:
                break

    if len(segments) < min_segments and video_duration > 60:
        segments = time_based_segments(frames, timestamps,
                                       interval=max(30.0, video_duration / 20))

    while len(segments) > max_segments:
        initial_threshold *= 1.5
        segments = detect_segments(frames, timestamps, initial_threshold,
                                   min_duration, representative)

    return segments
