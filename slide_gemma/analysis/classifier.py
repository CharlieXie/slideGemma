"""Video type classification via Gemma 4."""

from __future__ import annotations

import logging
from enum import Enum

import numpy as np

from ..models.loader import generate

logger = logging.getLogger(__name__)


class VideoType(str, Enum):
    SLIDES = "SLIDES"
    TEACHER_SLIDES = "TEACHER_SLIDES"
    WHITEBOARD = "WHITEBOARD"
    TEACHER_ONLY = "TEACHER_ONLY"
    SCREEN_RECORDING = "SCREEN_RECORDING"

    @classmethod
    def from_string(cls, text: str) -> "VideoType":
        cleaned = text.strip().upper().replace(" ", "_")
        for member in cls:
            if member.value in cleaned:
                return member
        return cls.SLIDES


_CLASSIFY_PROMPT = """\
Look at these sample frames from a lecture / educational video.
Classify the video into exactly ONE of these categories:

- SLIDES -- presentation slides or screen-shared slides occupy most of the frame
- TEACHER_SLIDES -- a teacher / presenter is visible AND slides or projected content are also shown
- WHITEBOARD -- a teacher writes on a whiteboard, blackboard, or similar surface
- TEACHER_ONLY -- only a teacher / presenter speaking to camera, no visual aids
- SCREEN_RECORDING -- screen recording of software, code, or digital content that is NOT slides

Respond with ONLY the category name, nothing else."""


def classify_video_type(
    model,
    processor,
    frame_paths: list[str],
    max_samples: int = 5,
) -> VideoType:
    """Send sample frames to Gemma 4 and return the detected VideoType."""
    indices = np.linspace(0, len(frame_paths) - 1, min(max_samples, len(frame_paths)), dtype=int)
    content: list[dict] = []
    for idx in indices:
        content.append({"type": "image", "url": frame_paths[int(idx)]})
    content.append({"type": "text", "text": _CLASSIFY_PROMPT})

    messages = [{"role": "user", "content": content}]
    response = generate(model, processor, messages, max_tokens=30)
    vtype = VideoType.from_string(response)
    logger.info("Video classified as %s (raw: %r)", vtype.value, response)
    return vtype
