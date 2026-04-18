#!/usr/bin/env python3
"""Minimal demo: describe a short video clip with Gemma 4.

Usage::

    python tools/demo_video.py VIDEO_PATH [--model e2b|e4b] [--gpu 0]
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    p = argparse.ArgumentParser(description="Describe a video clip with Gemma 4.")
    p.add_argument("video", help="Path to the video file")
    p.add_argument("--model", default="e2b", choices=["e2b", "e4b"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=512)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from slide_gemma.models import load_model

    video_path = os.path.abspath(args.video)
    assert os.path.isfile(video_path), f"Video not found: {video_path}"

    print(f"Loading Gemma 4 ({args.model.upper()}) ...")
    model, processor = load_model(args.model)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "url": video_path},
                {
                    "type": "text",
                    "text": (
                        "Provide a detailed description of this video. "
                        "Include what you see, the setting, any people or objects, "
                        "actions taking place, and notable details. "
                        "Write 3-5 sentences."
                    ),
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        load_audio_from_video=False,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)

    text = processor.decode(output[0][input_len:], skip_special_tokens=True)
    try:
        result = processor.parse_response(text)
        description = result.get("content", text.strip())
    except Exception:
        description = text.strip()

    print(f"\n--- Description ---\n{description}\n-------------------")


if __name__ == "__main__":
    main()
