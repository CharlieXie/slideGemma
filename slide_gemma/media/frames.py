"""Video metadata retrieval and frame extraction via PyAV."""

from __future__ import annotations

import logging

import av
from PIL import Image

logger = logging.getLogger(__name__)


def get_video_info(video_path: str) -> dict:
    """Return a dict with duration, fps, width, height, total_frames, has_audio."""
    container = av.open(video_path)
    vstream = container.streams.video[0]

    duration = float(container.duration / av.time_base) if container.duration else 0.0
    info = {
        "duration": duration,
        "fps": float(vstream.average_rate or 24),
        "width": vstream.codec_context.width,
        "height": vstream.codec_context.height,
        "total_frames": vstream.frames or 0,
        "has_audio": len(container.streams.audio) > 0,
    }
    container.close()
    return info


def extract_frames(video_path: str, fps: float = 1.0) -> tuple[list[Image.Image], list[float]]:
    """Decode *video_path* and return one PIL frame every ``1/fps`` seconds."""
    container = av.open(video_path)
    vstream = container.streams.video[0]

    video_fps = float(vstream.average_rate or 24)
    frame_interval = max(1, int(video_fps / fps))

    frames: list[Image.Image] = []
    timestamps: list[float] = []

    for i, frame in enumerate(container.decode(video=0)):
        if i % frame_interval == 0:
            img = frame.to_image()
            ts = float(frame.time) if frame.time is not None else i / video_fps
            frames.append(img)
            timestamps.append(ts)

    container.close()
    logger.info("Extracted %d frames at %.1f fps from %s", len(frames), fps, video_path)
    return frames, timestamps
