"""Ollama /api/generate provider — extracted from core.py for factory support."""

import json
import logging
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from ._retry import retry_request

logger = logging.getLogger("hydrag")


def _validate_url_scheme(url: str) -> None:
    """Reject non-HTTP(S) URL schemes (SR-4)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme {parsed.scheme!r} — only http/https allowed")


class OllamaProvider:
    """Ollama /api/generate provider with 3-attempt retry.

    Args:
        host: Ollama server base URL (default: http://localhost:11434).
    """

    def __init__(self, host: str = "http://localhost:11434") -> None:
        _validate_url_scheme(host)
        self._host = host.rstrip("/")

    @staticmethod
    def _parse_ollama_response(data: dict) -> str | None:  # type: ignore[type-arg]
        return str(data.get("response", "")) or None

    def generate(self, prompt: str, model: str = "", timeout: int = 30) -> str | None:
        effective_model = model or "llama3.2:latest"
        payload = json.dumps(
            {"model": effective_model, "prompt": prompt, "stream": False}
        ).encode()
        return retry_request(
            url=f"{self._host}/api/generate",
            payload=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
            provider_name="ollama",
            parse_response=self._parse_ollama_response,
        )

    def generate_stream(
        self, prompt: str, model: str = "", timeout: int = 30,
    ) -> str | None:
        """Streaming generate — returns full response but reads line-by-line."""
        effective_model = model or "llama3.2:latest"
        payload = json.dumps(
            {"model": effective_model, "prompt": prompt, "stream": True}
        ).encode()
        headers = {"Content-Type": "application/json"}
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    f"{self._host}/api/generate",
                    data=payload,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    tokens: list[str] = []
                    for line in resp:
                        chunk = json.loads(line.decode())
                        token = chunk.get("response", "")
                        tokens.append(token)
                        accumulated = "".join(tokens).strip().upper()
                        if "SUFFICIENT" in accumulated or "INSUFFICIENT" in accumulated:
                            return "".join(tokens)
                        if chunk.get("done"):
                            break
                    return "".join(tokens)
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    logger.log(
                        logging.WARNING,
                        "Ollama auth error (stream): HTTP %d",
                        exc.code,
                        extra={"provider": "ollama", "category": "auth_error", "detail": f"HTTP {exc.code}"},
                    )
                    raise
                if attempt < 2:
                    time.sleep(0.5)
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt < 2:
                    time.sleep(0.5)
            except ValueError:
                if attempt < 2:
                    time.sleep(0.5)
        return None
