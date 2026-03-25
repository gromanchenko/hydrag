"""Shared retry helper for LLM providers."""

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger("hydrag")


def retry_request(
    url: str,
    payload: bytes,
    headers: dict[str, str],
    timeout: int,
    provider_name: str,
    max_attempts: int = 3,
    parse_response: Any = None,
) -> str | None:
    """Execute an HTTP POST with retry logic shared across all providers.

    Args:
        url: Target URL.
        payload: JSON-encoded request body.
        headers: HTTP headers.
        timeout: Request timeout in seconds.
        provider_name: Name for structured logging (e.g. "ollama").
        max_attempts: Number of attempts before giving up.
        parse_response: Optional callable(data: dict) -> str | None.
            If provided, called on the parsed JSON response.
            If None, raw JSON dict is returned as str via json.dumps.

    Returns:
        Parsed response string, or None on exhausted retries.

    Raises:
        urllib.error.HTTPError: On 401/403 auth errors (not retried).
    """
    for attempt in range(max_attempts):
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                if parse_response is not None:
                    result = parse_response(data)
                    if result is None:
                        logger.log(
                            logging.WARNING,
                            "%s malformed response",
                            provider_name,
                            extra={
                                "provider": provider_name,
                                "category": "malformed_response",
                                "detail": "parse returned None",
                            },
                        )
                        if attempt < max_attempts - 1:
                            time.sleep(0.5)
                        continue
                    return result
                return json.dumps(data)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                logger.log(
                    logging.WARNING,
                    "%s auth error: HTTP %d",
                    provider_name, exc.code,
                    extra={"provider": provider_name, "category": "auth_error", "detail": f"HTTP {exc.code}"},
                )
                raise
            if exc.code >= 500:
                logger.log(
                    logging.DEBUG if attempt < max_attempts - 1 else logging.WARNING,
                    "%s server error: HTTP %d (attempt %d/%d)",
                    provider_name, exc.code, attempt + 1, max_attempts,
                    extra={"provider": provider_name, "category": "server_error", "detail": f"HTTP {exc.code}"},
                )
                if attempt < max_attempts - 1:
                    time.sleep(0.5)
                continue
            logger.log(
                logging.WARNING,
                "%s HTTP error: %d",
                provider_name, exc.code,
                extra={"provider": provider_name, "category": "unknown", "detail": f"HTTP {exc.code}"},
            )
            if attempt < max_attempts - 1:
                time.sleep(0.5)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            cat = "timeout" if isinstance(exc, TimeoutError) else "connection_refused"
            logger.log(
                logging.DEBUG,
                "%s transient error: %s (attempt %d/%d)",
                provider_name, exc, attempt + 1, max_attempts,
                extra={"provider": provider_name, "category": cat, "detail": str(exc)},
            )
            if attempt < max_attempts - 1:
                time.sleep(0.5)
        except ValueError as exc:
            logger.log(
                logging.WARNING,
                "%s malformed response: %s",
                provider_name, exc,
                extra={"provider": provider_name, "category": "malformed_response", "detail": str(exc)},
            )
            if attempt < max_attempts - 1:
                time.sleep(0.5)
    return None
