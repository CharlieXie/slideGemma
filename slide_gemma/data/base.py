"""Base class for lecture-domain datasets."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

from datasets import Dataset

logger = logging.getLogger(__name__)


class LectureDataset(ABC):
    """Abstract base for datasets used to fine-tune Gemma 4 on lecture tasks.

    Subclasses implement :meth:`load` which returns a HuggingFace ``Dataset``
    whose rows contain:

    * ``messages_json`` -- JSON-serialized list of chat turns (avoids PyArrow
      mixed-type issues with multimodal content)
    * ``images``        -- list of PIL images (may be empty for text-only tasks)
    """

    name: str = ""

    @abstractmethod
    def load(
        self,
        split: str = "train",
        max_samples: int | None = None,
    ) -> Dataset:
        """Load and format the dataset.

        Args:
            split: Dataset split (``"train"`` / ``"validation"`` / ``"test"``).
            max_samples: Cap the number of rows (useful for debugging).

        Returns:
            A ``datasets.Dataset`` with ``messages_json`` and ``images`` columns.
        """
        ...

    @staticmethod
    def _build_vqa_messages(
        question: str,
        answer: str,
        system_prompt: str | None = None,
        num_images: int = 1,
    ) -> list[dict]:
        """Build a user/model turn pair for a VQA example.

        Image placeholders are inserted as ``{"type": "image"}`` and
        resolved at collation time from the ``images`` column.
        """
        user_content: list[dict] = []
        for _ in range(num_images):
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": question})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": answer})
        return messages

    @staticmethod
    def messages_to_json(messages: list[dict]) -> str:
        """Serialize messages for Arrow-safe storage."""
        return json.dumps(messages, ensure_ascii=False)

    @staticmethod
    def messages_from_json(messages_json: str) -> list[dict]:
        """Deserialize messages from JSON string."""
        return json.loads(messages_json)
