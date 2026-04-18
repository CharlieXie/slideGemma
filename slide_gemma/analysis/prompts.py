"""Prompt templates for different video types and the final summary."""

from __future__ import annotations

from .classifier import VideoType

# ── Per-type defaults ─────────────────────────────────────────────────────

_TYPE_DEFAULTS: dict[VideoType, dict] = {
    VideoType.SLIDES:           {"threshold": 0.15, "representative": "first", "min_duration": 2.0},
    VideoType.TEACHER_SLIDES:   {"threshold": 0.12, "representative": "middle", "min_duration": 3.0},
    VideoType.WHITEBOARD:       {"threshold": 0.03, "representative": "last",  "min_duration": 5.0},
    VideoType.TEACHER_ONLY:     {"threshold": 0.08, "representative": "middle", "min_duration": 10.0},
    VideoType.SCREEN_RECORDING: {"threshold": 0.10, "representative": "first", "min_duration": 2.0},
}


def get_defaults_for_type(vtype: VideoType) -> dict:
    return dict(_TYPE_DEFAULTS.get(vtype, _TYPE_DEFAULTS[VideoType.SLIDES]))


# ── Helpers ───────────────────────────────────────────────────────────────

def _ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _lang_instruction(language: str | None) -> str:
    if language:
        return f"\nRespond in {language}."
    return "\nRespond in the same language as the content shown."


# ── Slide prompt ──────────────────────────────────────────────────────────

def build_slide_prompt(
    context_text: str,
    seg_index: int,
    total: int,
    start: float,
    end: float,
    audio_text: str | None = None,
    language: str | None = None,
) -> str:
    parts = [
        "You are an expert tutor helping a student understand a lecture video "
        "in real time. The student can already see the current slide -- do NOT "
        "describe what is visible. Instead, TEACH the subject matter: explain "
        "the concepts, formulas, and ideas being presented.\n\n",
        context_text, "\n\n",
        f"Analyzing slide {seg_index + 1}/{total} "
        f"(time: {_ts(start)} -- {_ts(end)}).\n",
    ]
    if audio_text:
        parts.append(f'\nThe lecturer said: "{audio_text}"\n')
    parts.append(
        "\nStructure your response as bullet points:\n\n"
        "* **Topic**: State the main concept in a clear phrase, "
        "then summarize in 1-2 sentences.\n\n"
        "* **Key Concepts**: Identify the important ideas, terms, or formulas. "
        "For formulas, explain what each variable means and the intuition.\n\n"
        "* **Deep Dive**: Teach the underlying concept -- provide reasoning, "
        "analogies, or examples that go beyond what is on the slide.\n\n"
        "* **Connection**: How does this relate to earlier content "
        "and the broader course? What should the student remember?\n\n"
        "Keep each bullet to 1-2 concise sentences. "
        "Do NOT repeat or list what is written on the slide.\n\n"
        "IMPORTANT: Write plain text only. Do NOT use markdown formatting "
        "such as *italic*, _underline_, # headings, or `code`. "
        "Only use **bold** for the four section labels above."
    )
    parts.append(_lang_instruction(language))
    return "".join(parts)


# ── Whiteboard prompt ─────────────────────────────────────────────────────

def build_whiteboard_prompt(
    context_text: str,
    seg_index: int,
    total: int,
    start: float,
    end: float,
    has_prev_frame: bool = False,
    audio_text: str | None = None,
    language: str | None = None,
) -> str:
    parts = [
        "You are an expert tutor helping a student understand a blackboard / "
        "whiteboard lecture in real time. The student can see the board -- "
        "do NOT simply transcribe what is written. Instead, TEACH the subject "
        "matter: explain the mathematics, reasoning, and ideas.\n\n",
        context_text, "\n\n",
        f"Analyzing segment {seg_index + 1}/{total} "
        f"(time: {_ts(start)} -- {_ts(end)}).\n",
    ]
    if has_prev_frame:
        parts.append(
            "\nThe FIRST image is the board earlier; the SECOND is the "
            "current state. Focus on what is new.\n"
        )
    if audio_text:
        parts.append(f'\nThe lecturer said: "{audio_text}"\n')
    parts.append(
        "\nStructure your response as bullet points:\n\n"
        "* **New Content**: What new concept, equation, or diagram has "
        "appeared on the board? Summarize briefly.\n\n"
        "* **Explanation**: Teach the new material step by step. For "
        "equations, explain the reasoning behind each step.\n\n"
        "* **Intuition**: Provide the deeper \"why.\" Use analogies, visual "
        "reasoning, or real-world examples to build understanding.\n\n"
        "* **Connection**: How does this fit into the overall topic and "
        "build on what came before?\n\n"
        "Keep each bullet to 1-2 concise sentences. "
        "Do NOT just transcribe what is on the board.\n\n"
        "IMPORTANT: Write plain text only. Do NOT use markdown formatting "
        "such as *italic*, _underline_, # headings, or `code`. "
        "Only use **bold** for the four section labels above."
    )
    parts.append(_lang_instruction(language))
    return "".join(parts)


# ── Teacher prompt ────────────────────────────────────────────────────────

def build_teacher_prompt(
    context_text: str,
    seg_index: int,
    total: int,
    start: float,
    end: float,
    audio_text: str | None = None,
    language: str | None = None,
) -> str:
    parts = [
        "You are an expert tutor helping a student understand a lecture "
        "in real time. The student can see the lecturer -- focus entirely "
        "on TEACHING the subject matter being discussed, not on describing "
        "the scene.\n\n",
        context_text, "\n\n",
        f"Analyzing segment {seg_index + 1}/{total} "
        f"(time: {_ts(start)} -- {_ts(end)}).\n",
    ]
    if audio_text:
        parts.append(f'\nThe lecturer said: "{audio_text}"\n')
    parts.append(
        "\nStructure your response as bullet points:\n\n"
        "* **Topic**: What concept is being taught right now? "
        "State it in 1-2 clear sentences.\n\n"
        "* **Key Points**: What are the most important ideas being "
        "conveyed? Explain them.\n\n"
        "* **Deep Dive**: Teach the concept in detail -- provide clear "
        "explanations, examples, and intuition.\n\n"
        "* **Connection**: How does this relate to the broader subject "
        "and earlier content?\n\n"
        "Keep each bullet to 1-2 concise sentences. "
        "Do NOT describe the teacher's appearance or actions.\n\n"
        "IMPORTANT: Write plain text only. Do NOT use markdown formatting "
        "such as *italic*, _underline_, # headings, or `code`. "
        "Only use **bold** for the four section labels above."
    )
    parts.append(_lang_instruction(language))
    return "".join(parts)


# ── Summary prompt ────────────────────────────────────────────────────────

SUMMARY_PROMPT = """\
You are an expert tutor. Below are your explanations for each segment of a
lecture. Write a concise **study guide** (5-10 sentences) that a student can
use to review the material.

Cover the main topics, key concepts, important formulas, and the logical
flow of ideas. Highlight what is most important to understand and remember.

Respond in the same language as the segment analyses.

---
{analyses}
---"""
