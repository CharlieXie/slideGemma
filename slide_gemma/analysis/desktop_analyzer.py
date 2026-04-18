"""Desktop screenshot analysis -- structured JSON output for the study overlay.

Sends a screenshot to the model (llama.cpp or HF) and parses structured
fields: page_type, title, formula, summary, key_points, next_action, etc.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass

from ..models.llamacpp_client import LlamaCppServerClient
from ..models.loader import generate


@dataclass
class DesktopAnalysis:
    page_type: str
    title: str
    line1: str
    line2: str
    formula_text: str
    summary: str
    formula_spotlight: str
    key_points: list[str]
    next_action: str


class DesktopContext:
    """Short rolling memory of recent desktop analyses."""

    def __init__(self, max_entries: int = 5):
        self.max_entries = max_entries
        self._entries: list[str] = []

    def add(self, summary: str) -> None:
        if self.max_entries <= 0:
            return
        text = summary.strip()
        if text:
            self._entries.append(text)
            self._entries = self._entries[-self.max_entries:]

    def to_prompt_text(self) -> str:
        if self.max_entries <= 0 or not self._entries:
            return "No prior context."
        lines = [f"- {item}" for item in self._entries]
        return "Recent study context:\n" + "\n".join(lines)


def analyze_desktop_image(
    model,
    processor,
    image_path: str,
    context: DesktopContext,
    language: str = "English",
    max_tokens: int = 384,
) -> DesktopAnalysis:
    """Analyse a desktop screenshot using HF model directly."""
    prompt = _build_prompt(context.to_prompt_text(), language)
    messages = [{"role": "user", "content": [
        {"type": "image", "url": image_path},
        {"type": "text", "text": prompt},
    ]}]
    raw = generate(model, processor, messages, max_tokens=max_tokens)
    return _analysis_from_raw_text(raw, language=language)


def analyze_desktop_image_via_llamacpp(
    client: LlamaCppServerClient,
    image_path: str,
    context: DesktopContext,
    language: str = "English",
    max_tokens: int = 384,
) -> DesktopAnalysis:
    """Analyse a desktop screenshot using llama.cpp server."""
    prompt = _build_prompt(context.to_prompt_text(), language)
    message = client.build_multimodal_message(prompt, image_path)
    raw = client.generate([message], max_tokens=max_tokens)
    return _analysis_from_raw_text(raw, language=language)


def format_analysis_text(result: DesktopAnalysis, language: str = "English") -> str:
    labels = {
        "page_type": "Page type", "title": "Title",
        "formula": "Recognized formula / matrix",
        "spotlight": "Formula / diagram explanation",
        "summary": "Summary", "key_points": "Key points",
        "next_action": "Next step", "untitled": "Not recognized",
        "no_summary": "No summary available",
        "no_points": "No key points available",
        "fallback_next": "Keep reading, or ask about a term, chart, or formula.",
    }

    lines = [f"{labels['page_type']}: {result.page_type}",
             f"{labels['title']}: {result.title or labels['untitled']}", ""]
    if result.formula_text:
        lines.extend([f"{labels['formula']}:", _clean(result.formula_text), ""])
    if result.formula_spotlight:
        lines.extend([f"{labels['spotlight']}:", _clean(result.formula_spotlight), ""])
    lines.extend([f"{labels['summary']}:", _clean(result.summary) or labels["no_summary"],
                   "", f"{labels['key_points']}:"])
    if result.key_points:
        lines.extend(f"- {_clean(p)}" for p in result.key_points)
    else:
        lines.append(f"- {labels['no_points']}")
    lines.extend(["", f"{labels['next_action']}:", _clean(result.next_action) or labels["fallback_next"]])
    return "\n".join(lines)


def format_payload_text(payload: dict, language: str = "English") -> str:
    result = DesktopAnalysis(
        page_type=str(payload.get("page_type", "other")),
        title=str(payload.get("title", "")),
        line1=str(payload.get("line1", "")),
        line2=str(payload.get("line2", "")),
        formula_text=str(payload.get("formula_text", "")),
        summary=str(payload.get("summary_raw", payload.get("summary", ""))),
        formula_spotlight=str(payload.get("formula_spotlight", "")),
        key_points=list(payload.get("key_points", [])),
        next_action=str(payload.get("next_action", "")),
    )
    return format_analysis_text(result, language=language)


def analysis_to_payload(result: DesktopAnalysis, language: str = "English") -> dict:
    payload = asdict(result)
    payload["formula_text"] = _clean_formula(result.formula_text)
    payload["formula_spotlight"] = _clean(result.formula_spotlight)
    payload["summary_raw"] = _clean(result.summary)
    payload["summary"] = _overlay_summary(result)
    payload["key_points"] = [_clean(p) for p in result.key_points]
    payload["next_action"] = _clean(result.next_action)
    payload["display_text"] = format_analysis_text(result, language=language)
    return payload


# ── Internals ─────────────────────────────────────────────────────────────

def _build_prompt(context_text: str, language: str) -> str:
    return f"""You are an academic study assistant.

The image is a screenshot from a student's desktop. It may show a paper, slide,
PDF, textbook, course website, chart, code notebook, whiteboard image, or study
material. Do NOT just describe the screen. Help the student understand the
content at a high level.

If the screenshot contains equations, matrices, vectors, coordinate axes, graphs,
derivations, or other math content, you MUST prioritize explaining them.

{context_text}

Return a JSON object with exactly these fields:
- page_type: one of ["paper", "slides", "document", "webpage", "code", "whiteboard", "chart", "other"]
- title: short title for the current content
- line1: very short {language} subtitle for the current page, <= 24 characters
- line2: a second short {language} subtitle with the most useful insight, <= 32 characters
- formula_text: if a formula is clearly visible, copy one important expression; otherwise ""
- summary: 2-4 {language} sentences explaining the main academic content
- formula_spotlight: 1-2 {language} sentences explaining the most important visible formula or diagram; "" if none
- key_points: array of 2-4 short {language} bullet points
- next_action: one short {language} suggestion for what the student can ask next

Rules:
- Focus on teaching, not scene description.
- Prefer explaining visible formulas and relationships.
- If text is blurry, infer conservatively.
- Respond in {language}.
- Output JSON only, no markdown fences."""


def _analysis_from_raw_text(raw: str, language: str = "English") -> DesktopAnalysis:
    data = _parse_response(raw)
    return DesktopAnalysis(
        page_type=str(data.get("page_type", "other")).strip() or "other",
        title=_clean(str(data.get("title", ""))),
        line1=_clean(str(data.get("line1", "Detected study page")))[:24],
        line2=_clean(str(data.get("line2", "")))[:48],
        formula_text=_clean_formula(str(data.get("formula_text", ""))),
        summary=_clean(str(data.get("summary", ""))),
        formula_spotlight=_clean(str(data.get("formula_spotlight", ""))),
        key_points=_normalize_points(data.get("key_points")),
        next_action=_clean(str(data.get("next_action", ""))),
    )


def _parse_response(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    text = text.strip()

    for candidate in [text, _repair_escapes(text)]:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        for candidate in [m.group(0), _repair_escapes(m.group(0))]:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        try:
            parsed = ast.literal_eval(_repair_escapes(m.group(0)))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return {"page_type": "other", "summary": text[:400]}


def _normalize_points(value) -> list[str]:
    if isinstance(value, list):
        return [_clean(str(i)) for i in value if _clean(str(i))]
    if isinstance(value, str) and value.strip():
        return [_clean(p) for p in re.split(r"[;\n]+", value) if _clean(p)]
    return []


def _overlay_summary(result: DesktopAnalysis) -> str:
    parts = []
    if result.formula_text:
        parts.append(f"Formula: {_clean_formula(result.formula_text)}")
    if result.formula_spotlight:
        parts.append(_clean(result.formula_spotlight))
    if result.summary:
        parts.append(_clean(result.summary))
    return "\n".join(p for p in parts if p).strip()


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().replace("```", ""))


def _clean_formula(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().replace("```", ""))


def _repair_escapes(text: str) -> str:
    return re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", text)
