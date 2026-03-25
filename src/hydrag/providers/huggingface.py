"""HuggingFace TGI /generate provider — stdlib-only HTTP."""

import json
import logging
import os
import urllib.error
from urllib.parse import urlparse

from ._retry import retry_request

logger = logging.getLogger("hydrag")


def _validate_url_scheme(url: str) -> None:
    """Reject non-HTTP(S) URL schemes (SR-4)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme {parsed.scheme!r} — only http/https allowed")


class HuggingFaceProvider:
    """HuggingFace TGI /generate provider via urllib.request.

    Args:
        api_base: TGI server base URL (e.g. "http://localhost:8080").
        model_id: Model identifier (logged, not sent to TGI — endpoint determines model).
        timeout: Request timeout in seconds.
        api_token: Bearer token. If None, reads HYDRAG_HF_API_TOKEN from os.environ.
    """

    def __init__(
        self,
        api_base: str,
        model_id: str = "",
        timeout: int = 30,
        api_token: str | None = None,
    ) -> None:
        if not api_base:
            raise ValueError("HuggingFaceProvider requires api_base (got empty string)")
        _validate_url_scheme(api_base)
        self._api_base = api_base.rstrip("/")
        self._model_id = model_id
        self._timeout = timeout
        self._api_token = api_token if api_token is not None else os.environ.get("HYDRAG_HF_API_TOKEN", "")

    @staticmethod
    def _parse_hf_response(data: dict | list) -> str | None:  # type: ignore[type-arg]
        if isinstance(data, list) and data:
            text = data[0].get("generated_text", "")
        elif isinstance(data, dict):
            text = data.get("generated_text", "")
        else:
            return None
        return str(text) or None

    def generate(self, prompt: str, model: str = "", timeout: int = 30) -> str | None:
        effective_timeout = timeout or self._timeout
        if model and model != self._model_id and self._model_id:
            logger.info(
                "HuggingFace: crag_model=%r ignored — TGI endpoint determines model (configured: %s)",
                model, self._model_id,
                extra={"provider": "huggingface", "category": "configuration_error", "detail": "crag_model ignored"},
            )
        payload = json.dumps({
            "inputs": prompt,
            "parameters": {"max_new_tokens": 50, "do_sample": False},
        }).encode()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_token:
            headers["Authorization"] = f"Bearer {self._api_token}"

        return retry_request(
            url=f"{self._api_base}/generate",
            payload=payload,
            headers=headers,
            timeout=effective_timeout,
            provider_name="huggingface",
            parse_response=self._parse_hf_response,
        )
