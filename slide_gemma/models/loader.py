"""Model loading for inference (HF or llama.cpp server) and QLoRA fine-tuning."""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "e2b": "google/gemma-4-E2B-it",
    "e4b": "google/gemma-4-E4B-it",
}

_LOCAL_DIR_PATTERNS = [
    "./gemma4_{name}_model",
    "./gemma4-{name}-model",
    "./gemma4_{name}",
]


def _find_local_model(model_name: str) -> str | None:
    for pattern in _LOCAL_DIR_PATTERNS:
        path = pattern.format(name=model_name)
        if os.path.isdir(path):
            return path
    return None


def _resolve_model_path(model_name: str, local_dir: str | None = None) -> str:
    if local_dir and os.path.isdir(local_dir):
        return local_dir
    local = _find_local_model(model_name)
    return local if local else MODEL_REGISTRY.get(model_name, model_name)


def load_model(
    model_name: str = "e2b",
    local_dir: str | None = None,
    server_url: str | None = None,
):
    """Load a Gemma 4 model for inference.

    If *server_url* is provided, returns ``(LlamaCppServerClient, None)``.
    Otherwise loads via HuggingFace Transformers.

    Returns:
        (model, processor) tuple.
    """
    if server_url:
        from .llamacpp_client import LlamaCppServerClient
        client = LlamaCppServerClient(server_url)
        logger.info("Using llama.cpp server at %s", client.endpoint)
        return client, None

    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    model_path = _resolve_model_path(model_name, local_dir)
    logger.info("Loading model from %s ...", model_path)

    model = AutoModelForMultimodalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(model_path, padding_side="left")

    logger.info("Model loaded on %s", model.device)
    return model, processor


def load_model_for_training(
    model_name: str = "e2b",
    local_dir: str | None = None,
    load_in_4bit: bool = True,
):
    """Load a Gemma 4 model with QLoRA quantization for fine-tuning.

    Returns:
        (model, processor) tuple with 4-bit quantized base weights.
    """
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

    model_path = _resolve_model_path(model_name, local_dir)
    logger.info("Loading model for training from %s (4bit=%s) ...", model_path, load_in_4bit)

    kwargs: dict = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": "sdpa",
    }

    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForMultimodalLM.from_pretrained(model_path, **kwargs)
    processor = AutoProcessor.from_pretrained(model_path, padding_side="left")

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    logger.info("Training model loaded (4bit=%s)", load_in_4bit)
    return model, processor


def generate(model, processor, messages: list[dict], max_tokens: int = 1024) -> str:
    """Run a single chat-completion turn.

    Dispatches to llama.cpp HTTP client or local HF model transparently.
    """
    from .llamacpp_client import LlamaCppServerClient

    if isinstance(model, LlamaCppServerClient):
        converted = _convert_messages_for_llamacpp(messages)
        return model.generate(converted, max_tokens=max_tokens)

    import torch

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

    generated_ids = output[0][input_len:]
    text = processor.decode(generated_ids, skip_special_tokens=True)

    try:
        result = processor.parse_response(text)
        return result.get("content", text.strip())
    except Exception:
        return text.strip()


def _convert_messages_for_llamacpp(messages: list[dict]) -> list[dict]:
    """Convert HF-style messages to OpenAI-style for llama.cpp."""
    converted: list[dict] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content")

        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            converted.append({"role": role, "content": str(content or "")})
            continue

        items: list[dict] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t == "text":
                items.append({"type": "text", "text": str(item.get("text", ""))})
            elif t == "image":
                url = item.get("url")
                if url:
                    from .llamacpp_client import image_path_to_data_uri
                    items.append({"type": "image_url", "image_url": {"url": image_path_to_data_uri(str(url))}})
            elif t == "image_url":
                items.append(item)

        converted.append({"role": role, "content": items})
    return converted
