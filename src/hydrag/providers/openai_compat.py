"""OpenAI-compatible /v1/chat/completions provider — stdlib-only HTTP."""

import json
import logging
import os
from urllib.parse import urlparse

from ._retry import retry_request

logger = logging.getLogger("hydrag")


def _validate_url_scheme(url: str) -> None:
    """Reject non-HTTP(S) URL schemes (SR-4)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme {parsed.scheme!r} — only http/https allowed")


class OpenAICompatProvider:
    """OpenAI-compatible provider via urllib.request.

    Supports /v1/chat/completions (default) and /v1/completions endpoints.
    Provider owns serializing prompt into the backend-specific payload format (FR-11).

    Args:
        api_base: Server base URL (e.g. "http://localhost:8080").
        model: Model name (required — passed in request body).
        timeout: Request timeout in seconds.
        api_token: API key. If None, reads HYDRAG_OPENAI_COMPAT_API_KEY from os.environ.
        endpoint: API endpoint path (default: /v1/chat/completions).
    """

    def __init__(
        self,
        api_base: str,
        model: str,
        timeout: int = 30,
        api_token: str | None = None,
        endpoint: str = "/v1/chat/completions",
    ) -> None:
        if not api_base:
            raise ValueError("OpenAICompatProvider requires api_base (got empty string)")
        if not model:
            raise ValueError("OpenAICompatProvider requires model (got empty string)")
        _validate_url_scheme(api_base)
        self._api_base = api_base.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._api_token = api_token if api_token is not None else os.environ.get("HYDRAG_OPENAI_COMPAT_API_KEY", "")
        self._endpoint = endpoint

    def _build_payload(self, prompt: str) -> bytes:
        """Serialize prompt into backend-specific payload (FR-11)."""
        if "/chat/completions" in self._endpoint:
            body = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50,
                "temperature": 0.0,
            }
        else:
            body = {
                "model": self._model,
                "prompt": prompt,
                "max_tokens": 50,
                "temperature": 0.0,
            }
        return json.dumps(body).encode()

    def _parse_response(self, data: dict) -> str | None:  # type: ignore[type-arg]
        """Extract text from response based on endpoint format."""
        if "/chat/completions" in self._endpoint:
            choices = data.get("choices", [])
            if choices and isinstance(choices, list):
                message = choices[0].get("message", {})
                text = message.get("content", "")
                return str(text) or None
        else:
            choices = data.get("choices", [])
            if choices and isinstance(choices, list):
                text = choices[0].get("text", "")
                return str(text) or None
        return None

    def generate(self, prompt: str, model: str = "", timeout: int = 30) -> str | None:
        effective_timeout = timeout or self._timeout
        effective_model = model or self._model
        payload = self._build_payload(prompt)
        if model and model != self._model:
            # Override model in payload if caller passes different model (e.g. crag_model)
            body = json.loads(payload.decode())
            body["model"] = effective_model
            payload = json.dumps(body).encode()

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_token:
            headers["Authorization"] = f"Bearer {self._api_token}"

        url = f"{self._api_base}{self._endpoint}"
        return retry_request(
            url=url,
            payload=payload,
            headers=headers,
            timeout=effective_timeout,
            provider_name="openai_compat",
            parse_response=self._parse_response,
        )
