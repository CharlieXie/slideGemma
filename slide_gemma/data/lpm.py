"""LPM (Lecture Presentations Multimodal) dataset loader.

LPM (ICCV 2023) contains 9,000+ slides with aligned spoken language from
180+ hours of educational videos across anatomy, biology, psychology,
computer science, and more.

GitHub: dondongwon/LPMDataset
Paper:  https://openaccess.thecvf.com/content/ICCV2023/html/Lee_Lecture_Presentations_Multimodal_Dataset_...
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import Dataset

from .base import LectureDataset

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert tutor. Given a lecture slide and the lecturer's "
    "spoken explanation, provide a clear, structured explanation that helps "
    "a student understand the material."
)


class LPMDataset(LectureDataset):
    """Load the Lecture Presentations Multimodal dataset.

    LPM is distributed via GitHub.  ``data_dir`` must point to the local
    clone / download directory.

    Expected structure::

        data_dir/
          slides/
            <lecture_id>/
              slide_0001.png
              ...
          transcripts/
            <lecture_id>.json      # slide-aligned transcripts
    """

    name = "lpm"

    def __init__(self, data_dir: str | None = None):
        self.data_dir = data_dir

    def load(
        self,
        split: str = "train",
        max_samples: int | None = None,
        data_dir: str | None = None,
    ) -> Dataset:
        root = Path(data_dir or self.data_dir or "./data/lpm")
        logger.info("Loading LPM from %s (split=%s) ...", root, split)

        if not root.exists():
            logger.warning(
                "LPM data directory not found at %s. "
                "Clone the repo first: git clone https://github.com/dondongwon/LPMDataset %s",
                root, root,
            )
            return self._empty_dataset()

        records = self._build_records(root)

        if max_samples is not None and max_samples < len(records):
            records = records[:max_samples]

        dataset = Dataset.from_list(records)
        logger.info("LPM ready: %d examples", len(dataset))
        return dataset

    def _build_records(self, root: Path) -> list[dict]:
        """Pair slides with their aligned spoken text."""
        records: list[dict] = []
        slides_dir = root / "slides"
        transcripts_dir = root / "transcripts"

        if not slides_dir.exists():
            logger.warning("No slides/ directory in %s", root)
            return records

        for lecture_dir in sorted(slides_dir.iterdir()):
            if not lecture_dir.is_dir():
                continue
            lecture_id = lecture_dir.name

            transcript_file = transcripts_dir / f"{lecture_id}.json"
            transcripts = self._load_transcripts(transcript_file) if transcript_file.exists() else {}

            slide_files = sorted(lecture_dir.glob("*.png"))
            for slide_path in slide_files:
                slide_name = slide_path.stem
                spoken_text = transcripts.get(slide_name, "")

                question = "Explain the content of this lecture slide in detail."
                if spoken_text:
                    question = (
                        f"Given this lecture slide and the lecturer's explanation, "
                        f"provide a clear summary.\n\n"
                        f'The lecturer said: "{spoken_text}"'
                    )

                try:
                    from PIL import Image
                    img = Image.open(slide_path).convert("RGB")
                except Exception:
                    continue

                answer = (
                    f"[Explanation for {slide_name} of {lecture_id}]"
                    if not spoken_text
                    else f"The slide discusses the following concepts: {spoken_text[:200]}"
                )

                messages = LectureDataset._build_vqa_messages(
                    question=question,
                    answer=answer,
                    system_prompt=_SYSTEM_PROMPT,
                    num_images=1,
                )
                records.append({
                    "messages_json": LectureDataset.messages_to_json(messages),
                    "images": [img],
                })

        return records

    @staticmethod
    def _load_transcripts(path: Path) -> dict[str, str]:
        """Load slide-aligned transcripts. Returns {slide_name: text}."""
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
            if isinstance(data, list):
                return {f"slide_{i:04d}": entry.get("text", str(entry))
                        for i, entry in enumerate(data)}
        except Exception as e:
            logger.warning("Failed to load transcripts from %s: %s", path, e)
        return {}

    @staticmethod
    def _empty_dataset() -> Dataset:
        return Dataset.from_dict({"messages_json": [], "images": []})
