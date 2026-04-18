"""Per-segment analysis and final summary generation."""

from __future__ import annotations

import logging
import re

from ..models.loader import generate
from ..media.segments import Segment
from .classifier import VideoType
from .context import LectureContext
from .prompts import (
    build_slide_prompt,
    build_whiteboard_prompt,
    build_teacher_prompt,
    SUMMARY_PROMPT,
)

logger = logging.getLogger(__name__)


def analyze_segment(
    model,
    processor,
    segment: Segment,
    context: LectureContext,
    video_type: VideoType,
    total_segments: int,
    audio_text: str | None = None,
    prev_frame_path: str | None = None,
    max_tokens: int = 1024,
    language: str | None = None,
) -> str:
    """Analyse a single segment and return the generated interpretation."""
    assert segment.frame_path, "segment.frame_path must be set before calling analyze_segment"
    ctx = context.get_context_text()

    if video_type == VideoType.WHITEBOARD:
        prompt = build_whiteboard_prompt(
            ctx, segment.index, total_segments,
            segment.start_time, segment.end_time,
            has_prev_frame=prev_frame_path is not None,
            audio_text=audio_text, language=language,
        )
    elif video_type in (VideoType.TEACHER_SLIDES, VideoType.TEACHER_ONLY):
        prompt = build_teacher_prompt(
            ctx, segment.index, total_segments,
            segment.start_time, segment.end_time,
            audio_text=audio_text, language=language,
        )
    else:
        prompt = build_slide_prompt(
            ctx, segment.index, total_segments,
            segment.start_time, segment.end_time,
            audio_text=audio_text, language=language,
        )

    content: list[dict] = []
    if prev_frame_path and video_type == VideoType.WHITEBOARD:
        content.append({"type": "image", "url": prev_frame_path})
    content.append({"type": "image", "url": segment.frame_path})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    analysis = generate(model, processor, messages, max_tokens=max_tokens)
    analysis = _strip_preamble(analysis)
    analysis = _clean_markdown(analysis)
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return analysis


def generate_summary(
    model,
    processor,
    analyses: list[str],
    max_tokens: int = 1024,
) -> str:
    """Generate an overall study guide from all per-segment analyses."""
    combined = "\n\n".join(
        f"[Segment {i + 1}]\n{text}" for i, text in enumerate(analyses)
    )
    if len(combined) > 300_000:
        combined = combined[:300_000] + "\n... (truncated)"

    prompt = SUMMARY_PROMPT.format(analyses=combined)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return generate(model, processor, messages, max_tokens=max_tokens)


# ── Text post-processing ─────────────────────────────────────────────────

def _strip_preamble(text: str) -> str:
    """Remove filler sentences before the first bullet point."""
    lines = text.split("\n")
    first_bullet = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^[\u2022\-\*]\s", stripped) or re.match(r"^\d+\.\s", stripped):
            first_bullet = i
            break
    if first_bullet > 0:
        return "\n".join(lines[first_bullet:])
    return text


def _clean_markdown(text: str) -> str:
    """Strip unwanted markdown syntax while preserving **bold** labels."""
    text = text.replace("**", "\x00B\x00")
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = text.replace("\x00B\x00", "**")
    return text
