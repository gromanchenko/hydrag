"""Provider factory — resolves LLMProvider from HydRAGConfig (§6.1)."""

import logging

from ..config import HydRAGConfig
from ..protocols import LLMProvider
from .huggingface import HuggingFaceProvider
from .ollama import OllamaProvider
from .openai_compat import OpenAICompatProvider

logger = logging.getLogger("hydrag")

_PROVIDER_NAMES = frozenset({"ollama", "huggingface", "openai_compat"})


def create_llm_provider(
    config: HydRAGConfig,
    api_tokens: dict[str, str] | None = None,
) -> LLMProvider:
    """Resolve provider from config.

    Args:
        config: HydRAG configuration with provider selection fields.
        api_tokens: Optional mapping of provider name -> API token.
            If provided, the token for the selected provider is passed
            to the provider's ``api_token`` kwarg. If absent, providers
            fall back to reading secrets from ``os.environ``.

    Returns:
        An LLMProvider instance for the selected provider.

    Raises:
        ValueError: Unknown provider name, or missing required fields
            for the selected provider.
    """
    name = config.llm_provider
    tokens = api_tokens or {}

    if name not in _PROVIDER_NAMES:
        raise ValueError(
            f"Unknown llm_provider {name!r} — must be one of: {', '.join(sorted(_PROVIDER_NAMES))}"
        )

    if name == "ollama":
        return OllamaProvider(host=config.ollama_host)

    if name == "huggingface":
        if not config.hf_api_base:
            raise ValueError(
                "llm_provider='huggingface' requires hf_api_base to be set "
                "(via constructor or HYDRAG_HF_API_BASE env var)"
            )
        token = tokens.get("huggingface")
        kwargs: dict[str, object] = {
            "api_base": config.hf_api_base,
            "model_id": config.hf_model_id,
            "timeout": config.hf_timeout,
        }
        if token is not None:
            kwargs["api_token"] = token
        return HuggingFaceProvider(**kwargs)  # type: ignore[arg-type]

    # openai_compat
    if not config.openai_compat_api_base:
        raise ValueError(
            "llm_provider='openai_compat' requires openai_compat_api_base to be set "
            "(via constructor or HYDRAG_OPENAI_COMPAT_API_BASE env var)"
        )
    if not config.openai_compat_model:
        raise ValueError(
            "llm_provider='openai_compat' requires openai_compat_model to be set "
            "(via constructor or HYDRAG_OPENAI_COMPAT_MODEL env var)"
        )
    token = tokens.get("openai_compat")
    kwargs_oai: dict[str, object] = {
        "api_base": config.openai_compat_api_base,
        "model": config.openai_compat_model,
        "timeout": config.openai_compat_timeout,
        "endpoint": config.openai_compat_endpoint,
    }
    if token is not None:
        kwargs_oai["api_token"] = token
    return OpenAICompatProvider(**kwargs_oai)  # type: ignore[arg-type]
