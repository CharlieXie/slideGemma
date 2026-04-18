"""Rolling context management for multi-segment lecture analysis."""

from __future__ import annotations

import re


class LectureContext:
    """Bounded window of segment summaries so each new analysis prompt
    receives enough background without overflowing the context."""

    def __init__(self, max_entries: int = 10):
        self._entries: list[tuple[int, str]] = []
        self.max_entries = max_entries

    def add(self, segment_index: int, summary: str) -> None:
        self._entries.append((segment_index, summary))

    def get_context_text(self) -> str:
        if not self._entries:
            return "This is the very beginning of the lecture -- no prior content."
        recent = self._entries[-self.max_entries:]
        lines = [f"  * Segment {idx + 1}: {s}" for idx, s in recent]
        header = "What has been covered so far:\n"
        if len(self._entries) > self.max_entries:
            header = (f"What has been covered so far "
                      f"(last {self.max_entries} of {len(self._entries)} segments):\n")
        return header + "\n".join(lines)

    @staticmethod
    def extract_summary(full_analysis: str, max_length: int = 200) -> str:
        """Extract a concise summary from the analysis for rolling context.

        Prefers the **Topic** line when available, falls back to the first
        substantive line.
        """
        for line in full_analysis.split("\n"):
            stripped = line.strip()
            if "**Topic**" in stripped or "**topic**" in stripped.lower():
                text = re.sub(r"\*\*.*?\*\*:?\s*", "", stripped).strip()
                if len(text) > 10:
                    return text[:max_length]
        for line in full_analysis.split("\n"):
            stripped = line.strip().lstrip("#*\u2022- ")
            if len(stripped) > 15:
                return stripped[:max_length] + ("..." if len(stripped) > max_length else "")
        clean = full_analysis.replace("\n", " ").strip()
        return clean[:max_length] + ("..." if len(clean) > max_length else "")
