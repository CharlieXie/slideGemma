"""HTTP client for a local llama.cpp OpenAI-compatible server."""

from __future__ import annotations

import base64
import json
import mimetypes
import socket
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class LlamaCppServerClient:
    """Wrapper around llama-server's chat-completions endpoint."""

    def __init__(self, server_url: str, timeout_seconds: int = 120):
        self.endpoint = self._normalize_endpoint(server_url)
        self.timeout_seconds = timeout_seconds

    def generate(self, messages: list[dict], max_tokens: int = 384) -> str:
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                status_code = response.status
                response_text = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            if _looks_like_memory_error(detail):
                raise RuntimeError(
                    f"AI service inference failed (possible OOM).\n"
                    f"Endpoint: {self.endpoint}\nHTTP {exc.code}\nDetails: {_safe(detail)}"
                ) from exc
            raise RuntimeError(
                f"AI service returned an error.\n"
                f"Endpoint: {self.endpoint}\nHTTP {exc.code}\nDetails: {_safe(detail)}"
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError(
                f"AI service timed out.\nEndpoint: {self.endpoint}\n"
                "Try again later or reduce input image size."
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Could not connect to AI service.\nEndpoint: {self.endpoint}\n\n"
                "Start the local AI service first, e.g.:\n"
                "  llama-server -hf ggml-org/gemma-4-E2B-it-GGUF --reasoning off"
            ) from exc

        if status_code >= 400:
            raise RuntimeError(
                f"AI service error HTTP {status_code}.\nDetails: {_safe(response_text)}"
            )

        try:
            data = json.loads(response_text)
        except ValueError as exc:
            raise RuntimeError(f"Unparseable AI response: {_safe(response_text)}") from exc

        try:
            message = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response shape: {data}") from exc

        reasoning = None
        try:
            reasoning = data["choices"][0]["message"].get("reasoning_content")
        except Exception:
            pass

        if (message == "" or message == [] or message is None) and reasoning:
            raise RuntimeError(
                "AI service is in reasoning mode and did not return a final answer.\n"
                "Restart with: llama-server -hf ggml-org/gemma-4-E2B-it-GGUF --reasoning off"
            )

        if isinstance(message, list):
            parts = [str(p.get("text", "")) for p in message if isinstance(p, dict) and p.get("type") == "text"]
            return "\n".join(p for p in parts if p.strip()).strip()

        return str(message).strip()

    @staticmethod
    def build_multimodal_message(prompt: str, image_path: str) -> dict:
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_path_to_data_uri(image_path)}},
            ],
        }

    @staticmethod
    def _normalize_endpoint(server_url: str) -> str:
        text = server_url.strip() or "http://127.0.0.1:8080"
        if not text.startswith(("http://", "https://")):
            text = f"http://{text}"
        parsed = urlparse(text)
        path = parsed.path.rstrip("/")
        if path.endswith("/v1/chat/completions"):
            final = path
        elif path.endswith("/v1"):
            final = f"{path}/chat/completions"
        elif path:
            final = f"{path}/v1/chat/completions"
        else:
            final = "/v1/chat/completions"
        return parsed._replace(path=final, params="", query="", fragment="").geturl()


def image_path_to_data_uri(image_path: str) -> str:
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    return f"data:{mime_type or 'image/png'};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _safe(text: str) -> str:
    text = text.strip()
    return text[:500] + "..." if len(text) > 500 else text


def _looks_like_memory_error(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in ["out of memory", "cuda error", "insufficient memory", "failed to allocate", "vram"])
