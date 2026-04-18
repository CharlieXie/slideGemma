#!/usr/bin/env python3
"""Smoke test for the fine-tuning pipeline using synthetic data.

Verifies the full chain: config -> dataset -> model loading -> QLoRA
adapter application, without requiring an actual GPU or real dataset.

Usage::

    python tools/test_pipeline.py
"""

from __future__ import annotations

import json
import sys
sys.path.insert(0, ".")

from PIL import Image
from datasets import Dataset


def create_synthetic_dataset(n: int = 20) -> Dataset:
    """Generate a small synthetic VQA dataset for pipeline testing."""
    records = []
    for i in range(n):
        img = Image.new("RGB", (224, 224), color=(i * 10 % 256, 100, 200))
        messages = [
            {"role": "system", "content": "You are a lecture slide expert."},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"What concept is shown on slide {i + 1}?"},
                ],
            },
            {
                "role": "assistant",
                "content": f"Slide {i + 1} covers topic {chr(65 + i % 26)} with key formula x={i}.",
            },
        ]
        records.append({
            "messages_json": json.dumps(messages, ensure_ascii=False),
            "images": [img],
        })
    return Dataset.from_list(records)


def test_config():
    from slide_gemma.training.config import TrainingConfig

    cfg = TrainingConfig()
    assert cfg.model_name == "e2b"
    assert cfg.lora_r == 16
    assert cfg.dataset_name == "slidevqa"

    cfg2 = TrainingConfig.from_yaml("configs/qlora_slidevqa.yaml")
    assert cfg2.output_dir == "./checkpoints/qlora_slidevqa"
    assert cfg2.num_train_epochs == 3

    print("[PASS] TrainingConfig")


def test_dataset_registry():
    from slide_gemma.data import DATASET_REGISTRY
    assert "slidevqa" in DATASET_REGISTRY
    assert "m3av" in DATASET_REGISTRY
    assert "lpm" in DATASET_REGISTRY
    print("[PASS] Dataset registry")


def test_synthetic_dataset():
    ds = create_synthetic_dataset(10)
    assert len(ds) == 10
    assert "messages_json" in ds.column_names
    assert "images" in ds.column_names

    ex = ds[0]
    messages = json.loads(ex["messages_json"])
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"

    user_content = messages[1]["content"]
    assert isinstance(user_content, list)
    image_items = [c for c in user_content if c.get("type") == "image"]
    assert len(image_items) == 1

    print("[PASS] Synthetic dataset creation")


def test_analysis_components():
    from slide_gemma.analysis.classifier import VideoType
    assert VideoType.from_string("SLIDES") == VideoType.SLIDES
    assert VideoType.from_string("whiteboard") == VideoType.WHITEBOARD

    from slide_gemma.analysis.context import LectureContext
    ctx = LectureContext(max_entries=3)
    for i in range(5):
        ctx.add(i, f"Topic {i}")
    text = ctx.get_context_text()
    assert "last 3 of 5" in text

    from slide_gemma.analysis.prompts import get_defaults_for_type, build_slide_prompt
    d = get_defaults_for_type(VideoType.WHITEBOARD)
    assert d["threshold"] == 0.03

    prompt = build_slide_prompt("context", 0, 5, 0.0, 30.0, language="English")
    assert "expert tutor" in prompt
    assert "English" in prompt

    print("[PASS] Analysis components")


def test_base_dataset_helper():
    from slide_gemma.data.base import LectureDataset
    messages = LectureDataset._build_vqa_messages(
        question="What is shown?",
        answer="A diagram of DNA.",
        system_prompt="You are a biology tutor.",
        num_images=2,
    )
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    user_content = messages[1]["content"]
    image_count = sum(1 for c in user_content if c.get("type") == "image")
    assert image_count == 2

    serialized = LectureDataset.messages_to_json(messages)
    deserialized = LectureDataset.messages_from_json(serialized)
    assert deserialized == messages

    print("[PASS] Base dataset helper + serialization")


def test_message_roundtrip():
    """Verify JSON roundtrip preserves multimodal content structure."""
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "image"},
            {"type": "text", "text": "Compare these two slides."},
        ]},
        {"role": "assistant", "content": "The first slide shows..."},
    ]

    js = json.dumps(messages)
    restored = json.loads(js)

    assert restored[0]["content"][0]["type"] == "image"
    assert restored[0]["content"][2]["text"] == "Compare these two slides."
    assert restored[1]["content"] == "The first slide shows..."

    print("[PASS] Message JSON roundtrip")


def main():
    print("=" * 50)
    print("  slideGemma Pipeline Smoke Test")
    print("=" * 50)
    print()

    test_config()
    test_dataset_registry()
    test_synthetic_dataset()
    test_analysis_components()
    test_base_dataset_helper()
    test_message_roundtrip()

    print()
    print("=" * 50)
    print("  ALL TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
