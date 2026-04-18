"""M3AV dataset loader for Gemma 4 QLoRA fine-tuning.

M3AV (ACL 2024) is a multimodal, multigenre, multipurpose audio-visual
academic lecture dataset with ~367 hours of videos from five sources.

GitHub: Jack-ZC8/M3AV-dataset
Paper:  https://arxiv.org/abs/2403.14168
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from datasets import Dataset

from .base import LectureDataset

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert academic tutor. Given a lecture slide and its context, "
    "provide a clear, educational explanation of the content shown."
)


class M3AVDataset(LectureDataset):
    """Load M3AV dataset and format for Gemma 4 fine-tuning.

    M3AV is distributed via GitHub (not HuggingFace), so ``data_dir``
    must point to the local clone of the repository.

    Expected structure::

        data_dir/
          slides/
            <video_id>/
              slide_0001.png
              ...
          speech/
            <video_id>.json        # word-level timestamps + text
          metadata.json            # video-level metadata
    """

    name = "m3av"

    def __init__(self, data_dir: str | None = None):
        self.data_dir = data_dir

    def load(
        self,
        split: str = "train",
        max_samples: int | None = None,
        data_dir: str | None = None,
    ) -> Dataset:
        root = Path(data_dir or self.data_dir or "./data/m3av")
        logger.info("Loading M3AV from %s (split=%s) ...", root, split)

        if not root.exists():
            logger.warning(
                "M3AV data directory not found at %s. "
                "Clone the repo first: git clone https://github.com/Jack-ZC8/M3AV-dataset %s",
                root, root,
            )
            return self._empty_dataset()

        records = self._build_records(root)

        if max_samples is not None and max_samples < len(records):
            records = records[:max_samples]

        dataset = Dataset.from_list(records)
        logger.info("M3AV ready: %d examples", len(dataset))
        return dataset

    def _build_records(self, root: Path) -> list[dict]:
        """Walk the M3AV directory and pair slides with speech segments."""
        records: list[dict] = []
        slides_dir = root / "slides"
        speech_dir = root / "speech"

        if not slides_dir.exists():
            logger.warning("No slides/ directory in %s", root)
            return records

        for video_dir in sorted(slides_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            video_id = video_dir.name

            speech_file = speech_dir / f"{video_id}.json"
            speech_segments = self._load_speech(speech_file) if speech_file.exists() else []

            slide_files = sorted(video_dir.glob("*.png"))
            for i, slide_path in enumerate(slide_files):
                transcript = speech_segments[i] if i < len(speech_segments) else ""

                question = "Explain the content of this lecture slide."
                if transcript:
                    question += f'\n\nThe lecturer said: "{transcript}"'

                try:
                    from PIL import Image
                    img = Image.open(slide_path).convert("RGB")
                except Exception:
                    continue

                messages = LectureDataset._build_vqa_messages(
                    question=question,
                    answer=f"[Explanation for slide {i + 1} of {video_id}]",
                    system_prompt=_SYSTEM_PROMPT,
                    num_images=1,
                )
                records.append({
                    "messages_json": LectureDataset.messages_to_json(messages),
                    "images": [img],
                })

        return records

    @staticmethod
    def _load_speech(path: Path) -> list[str]:
        """Load speech segments from a JSON file, returning a list of text strings."""
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                return [seg.get("text", "") if isinstance(seg, dict) else str(seg)
                        for seg in data]
            if isinstance(data, dict) and "segments" in data:
                return [seg.get("text", "") for seg in data["segments"]]
        except Exception as e:
            logger.warning("Failed to load speech from %s: %s", path, e)
        return []

    @staticmethod
    def _empty_dataset() -> Dataset:
        return Dataset.from_dict({"messages_json": [], "images": []})
