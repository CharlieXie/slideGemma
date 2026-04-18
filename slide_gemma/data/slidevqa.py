"""SlideVQA dataset loader for Gemma 4 QLoRA fine-tuning.

SlideVQA (AAAI 2023) contains 2,600+ slide decks with 14,500+ questions
requiring multi-image reasoning over presentation slides.

HuggingFace: NTT-hil-insight/SlideVQA  (gated -- requires HF login)
GitHub:      https://github.com/nttmdlab-nlp/SlideVQA
Paper:       https://arxiv.org/abs/2301.04883
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import Dataset

from .base import LectureDataset

logger = logging.getLogger(__name__)

_HF_REPO = "NTT-hil-insight/SlideVQA"

_SYSTEM_PROMPT = (
    "You are an expert tutor analyzing presentation slides. "
    "Answer the question about the slide content accurately and concisely."
)


class SlideVQADataset(LectureDataset):
    """Load SlideVQA and format for Gemma 4 fine-tuning.

    Supports two loading modes:

    1. **HuggingFace** (default) -- requires ``huggingface-cli login`` to
       accept the NTT license agreement first.
    2. **Local directory** -- pass ``data_dir`` pointing to a checkout of the
       GitHub repo containing ``qa/`` JSON files and ``images/`` slide decks.
    """

    name = "slidevqa"

    def __init__(self, data_dir: str | None = None):
        self.data_dir = data_dir

    def load(
        self,
        split: str = "train",
        max_samples: int | None = None,
        data_dir: str | None = None,
    ) -> Dataset:
        root = data_dir or self.data_dir

        if root and Path(root).exists():
            return self._load_local(Path(root), split, max_samples)

        return self._load_hf(split, max_samples)

    # ── HuggingFace loading ───────────────────────────────────────────────

    def _load_hf(self, split: str, max_samples: int | None) -> Dataset:
        from datasets import load_dataset as hf_load_dataset

        logger.info("Loading SlideVQA from HuggingFace (%s, split=%s) ...", _HF_REPO, split)
        logger.info(
            "This dataset is gated. If you get an authentication error, run: "
            "huggingface-cli login"
        )

        try:
            raw = hf_load_dataset(_HF_REPO, split=split)
        except Exception as e:
            if "gated" in str(e).lower() or "authentication" in str(e).lower():
                logger.error(
                    "SlideVQA requires HuggingFace authentication. Steps:\n"
                    "  1. Accept the license at https://huggingface.co/datasets/%s\n"
                    "  2. Run: huggingface-cli login\n"
                    "  OR download from GitHub and pass --data-dir.",
                    _HF_REPO,
                )
            raise

        if max_samples is not None and max_samples < len(raw):
            raw = raw.select(range(max_samples))

        formatted = raw.map(
            self._format_hf_example,
            remove_columns=raw.column_names,
            desc="Formatting SlideVQA",
        )

        logger.info("SlideVQA ready: %d examples", len(formatted))
        return formatted

    @staticmethod
    def _format_hf_example(example: dict) -> dict:
        """Convert a HuggingFace SlideVQA row to chat format."""
        question = example.get("question", "")
        answer = str(example.get("answer", ""))

        images = []
        if "image" in example and example["image"] is not None:
            images = [example["image"]]
        elif "images" in example and example["images"] is not None:
            images = list(example["images"])

        num_images = max(1, len(images))
        messages = LectureDataset._build_vqa_messages(
            question=question,
            answer=answer,
            system_prompt=_SYSTEM_PROMPT,
            num_images=num_images,
        )

        return {
            "messages_json": LectureDataset.messages_to_json(messages),
            "images": images,
        }

    # ── Local directory loading ───────────────────────────────────────────

    def _load_local(self, root: Path, split: str, max_samples: int | None) -> Dataset:
        """Load from a local SlideVQA checkout (GitHub format).

        Expected structure::

            root/
              qa/
                train.json | val.json | test.json
              images/
                <deck_name>/
                  0001.png ... 0020.png
        """
        logger.info("Loading SlideVQA from local dir %s (split=%s) ...", root, split)

        split_map = {"train": "train", "validation": "val", "val": "val", "test": "test"}
        qa_name = split_map.get(split, split)

        qa_file = root / "qa" / f"{qa_name}.json"
        if not qa_file.exists():
            possible = list((root / "qa").glob("*.json")) if (root / "qa").exists() else []
            raise FileNotFoundError(
                f"QA file not found: {qa_file}. "
                f"Available: {[p.name for p in possible]}"
            )

        with open(qa_file) as f:
            qa_data = json.load(f)

        records: list[dict] = []
        images_dir = root / "images"

        for item in qa_data:
            question = item.get("question", "")
            answer = str(item.get("answer", ""))
            deck_name = item.get("deck_name", "")

            images = []
            evidence_pages = item.get("evidence", [])
            if evidence_pages and images_dir.exists():
                for page_num in evidence_pages:
                    img_path = images_dir / deck_name / f"{page_num:04d}.png"
                    if img_path.exists():
                        from PIL import Image
                        images.append(Image.open(img_path).convert("RGB"))

            num_images = max(1, len(images))
            messages = LectureDataset._build_vqa_messages(
                question=question,
                answer=answer,
                system_prompt=_SYSTEM_PROMPT,
                num_images=num_images,
            )
            records.append({
                "messages_json": LectureDataset.messages_to_json(messages),
                "images": images,
            })

        if max_samples is not None and max_samples < len(records):
            records = records[:max_samples]

        dataset = Dataset.from_list(records)
        logger.info("SlideVQA (local) ready: %d examples", len(dataset))
        return dataset
