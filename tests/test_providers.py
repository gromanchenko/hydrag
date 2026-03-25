"""Conformance + unit tests for multi-provider LLM support (T-541, RFC V3).

§12.1 Conformance harness — parametrized across all providers with mock HTTP.
§12.2 Unit tests — config, factory, serialization, secrets, error categories.
§12.3 Integration — crag_supervisor + hydrag_search with factory.
§12.4 Regression — Ollama-only default path unchanged.
"""

import json
import logging
import os
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from unittest.mock import patch

import pytest

# Re-use conftest fixtures
from conftest import MockAdapter, MockLLM, make_config

from hydrag import (
    HydRAG,
    HydRAGConfig,
    LLMProvider,
    OllamaProvider,
    StreamingLLMProvider,
    crag_supervisor,
    create_llm_provider,
    hydrag_search,
)
from hydrag.providers.huggingface import HuggingFaceProvider
from hydrag.providers.openai_compat import OpenAICompatProvider

# ── Mock HTTP server for provider conformance ─────────────────────


class _MockProviderHandler(BaseHTTPRequestHandler):
    """Configurable mock HTTP handler for provider tests."""

    # Class-level config — set before each test
    response_body: bytes = b""
    response_code: int = 200
    response_headers: dict[str, str] = {}
    last_request_body: bytes = b""
    last_request_headers: dict[str, str] = {}

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        _MockProviderHandler.last_request_body = self.rfile.read(length)
        _MockProviderHandler.last_request_headers = dict(self.headers)
        self.send_response(self.response_code)
        for k, v in _MockProviderHandler.response_headers.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.response_body)

    def log_message(self, format: str, *args: object) -> None:
        pass  # suppress stderr


@pytest.fixture(scope="module")
def mock_server() -> tuple[str, HTTPServer]:
    """Start a mock HTTP server on a free port, shared across module."""
    server = HTTPServer(("127.0.0.1", 0), _MockProviderHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    yield base, server
    server.shutdown()


def _set_mock(body: dict | list | str, code: int = 200) -> None:  # type: ignore[type-arg]
    if isinstance(body, (dict, list)):
        _MockProviderHandler.response_body = json.dumps(body).encode()
    else:
        _MockProviderHandler.response_body = body.encode()
    _MockProviderHandler.response_code = code
    _MockProviderHandler.response_headers = {}


# ═══════════════════════════════════════════════════════════════════
# §12.1 CONFORMANCE HARNESS — parametrized across all providers
# ═══════════════════════════════════════════════════════════════════


def _make_provider(name: str, base_url: str) -> LLMProvider:
    """Create a provider instance for conformance tests."""
    if name == "ollama":
        return OllamaProvider(host=base_url)
    if name == "huggingface":
        return HuggingFaceProvider(api_base=base_url, model_id="test-model")
    if name == "openai_compat":
        return OpenAICompatProvider(api_base=base_url, model="test-model")
    raise ValueError(name)


def _success_response(name: str) -> dict | list:  # type: ignore[type-arg]
    """Build a valid success response for each provider."""
    if name == "ollama":
        return {"response": "SUFFICIENT"}
    if name == "huggingface":
        return [{"generated_text": "SUFFICIENT"}]
    if name == "openai_compat":
        return {"choices": [{"message": {"content": "SUFFICIENT"}}]}
    raise ValueError(name)


@pytest.mark.parametrize("provider_name", ["ollama", "huggingface", "openai_compat"])
class TestConformanceHarness:
    """§12.1: Conformance tests parametrized across all providers."""

    def test_generate_returns_str_on_success(self, provider_name: str, mock_server: tuple[str, HTTPServer]) -> None:
        """§12.1 #1: generate() returns str on success."""
        base, _ = mock_server
        _set_mock(_success_response(provider_name))
        provider = _make_provider(provider_name, base)
        result = provider.generate("test prompt", model="test-model", timeout=5)
        assert isinstance(result, str)
        assert "SUFFICIENT" in result

    def test_generate_returns_none_on_timeout(self, provider_name: str, mock_server: tuple[str, HTTPServer]) -> None:
        """§12.1 #2: generate() returns None on timeout (transient)."""
        provider = _make_provider(provider_name, "http://192.0.2.1:1")  # non-routable
        result = provider.generate("test", model="m", timeout=1)
        assert result is None

    def test_generate_returns_none_on_malformed(self, provider_name: str, mock_server: tuple[str, HTTPServer]) -> None:
        """§12.1 #4: generate() returns None on malformed response."""
        base, _ = mock_server
        _set_mock("not json at all{{{")
        provider = _make_provider(provider_name, base)
        result = provider.generate("test", model="m", timeout=5)
        assert result is None

    def test_generate_respects_timeout(self, provider_name: str, mock_server: tuple[str, HTTPServer]) -> None:
        """§12.1 #6: generate() respects timeout parameter."""
        # Using non-routable to test timeout is respected (fast fail)
        provider = _make_provider(provider_name, "http://192.0.2.1:1")
        result = provider.generate("test", model="m", timeout=1)
        assert result is None

    def test_generate_raises_on_auth_error(self, provider_name: str, mock_server: tuple[str, HTTPServer]) -> None:
        """§12.1 #8: generate() raises on HTTP 401/403 (auth_error)."""
        base, _ = mock_server
        _set_mock({"error": "unauthorized"}, code=401)
        provider = _make_provider(provider_name, base)
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            provider.generate("test", model="m", timeout=5)
        assert exc_info.value.code == 401

    def test_generate_raises_on_forbidden(self, provider_name: str, mock_server: tuple[str, HTTPServer]) -> None:
        """§12.1 #8 variant: HTTP 403."""
        base, _ = mock_server
        _set_mock({"error": "forbidden"}, code=403)
        provider = _make_provider(provider_name, base)
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            provider.generate("test", model="m", timeout=5)
        assert exc_info.value.code == 403

    def test_generate_returns_none_on_server_error(
        self, provider_name: str, mock_server: tuple[str, HTTPServer]
    ) -> None:
        """§12.1 #2 variant: HTTP 500 → None (transient)."""
        base, _ = mock_server
        _set_mock({"error": "internal"}, code=500)
        provider = _make_provider(provider_name, base)
        result = provider.generate("test", model="m", timeout=5)
        assert result is None


# ═══════════════════════════════════════════════════════════════════
# §12.2 UNIT TESTS
# ═══════════════════════════════════════════════════════════════════


class TestConfigFields:
    """§12.2 #1: Config construction via kwargs for all new fields."""

    def test_new_fields_defaults(self) -> None:
        cfg = HydRAGConfig()
        assert cfg.llm_provider == "ollama"
        assert cfg.hf_model_id == ""
        assert cfg.hf_api_base == ""
        assert cfg.hf_timeout == 30
        assert cfg.openai_compat_api_base == ""
        assert cfg.openai_compat_model == ""
        assert cfg.openai_compat_timeout == 30
        assert cfg.openai_compat_endpoint == "/v1/chat/completions"

    def test_new_fields_via_kwargs(self) -> None:
        cfg = HydRAGConfig(
            llm_provider="huggingface",
            hf_model_id="bigscience/bloom",
            hf_api_base="http://hf:8080",
            hf_timeout=60,
            openai_compat_endpoint="/v1/completions",
        )
        assert cfg.llm_provider == "huggingface"
        assert cfg.hf_model_id == "bigscience/bloom"
        assert cfg.hf_api_base == "http://hf:8080"
        assert cfg.hf_timeout == 60
        assert cfg.openai_compat_endpoint == "/v1/completions"

    def test_from_env_new_fields(self) -> None:
        """§12.2 #2: from_env() parsing for all new env vars."""
        env = {
            "HYDRAG_LLM_PROVIDER": "openai_compat",
            "HYDRAG_HF_MODEL_ID": "test-hf",
            "HYDRAG_HF_API_BASE": "http://hf-env:8080",
            "HYDRAG_HF_TIMEOUT": "45",
            "HYDRAG_OPENAI_COMPAT_API_BASE": "http://oai-env:8080",
            "HYDRAG_OPENAI_COMPAT_MODEL": "gpt-local",
            "HYDRAG_OPENAI_COMPAT_TIMEOUT": "60",
            "HYDRAG_OPENAI_COMPAT_ENDPOINT": "/v1/completions",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = HydRAGConfig.from_env()
        assert cfg.llm_provider == "openai_compat"
        assert cfg.hf_model_id == "test-hf"
        assert cfg.hf_api_base == "http://hf-env:8080"
        assert cfg.hf_timeout == 45
        assert cfg.openai_compat_api_base == "http://oai-env:8080"
        assert cfg.openai_compat_model == "gpt-local"
        assert cfg.openai_compat_timeout == 60
        assert cfg.openai_compat_endpoint == "/v1/completions"


class TestFactory:
    """§12.2 #3-4: Factory selection and api_tokens passthrough."""

    def test_factory_ollama_default(self) -> None:
        cfg = HydRAGConfig()
        provider = create_llm_provider(cfg)
        assert isinstance(provider, OllamaProvider)

    def test_factory_huggingface(self) -> None:
        cfg = HydRAGConfig(llm_provider="huggingface", hf_api_base="http://hf:8080")
        provider = create_llm_provider(cfg)
        assert isinstance(provider, HuggingFaceProvider)

    def test_factory_openai_compat(self) -> None:
        cfg = HydRAGConfig(
            llm_provider="openai_compat",
            openai_compat_api_base="http://oai:8080",
            openai_compat_model="test-model",
        )
        provider = create_llm_provider(cfg)
        assert isinstance(provider, OpenAICompatProvider)

    def test_factory_unknown_raises(self) -> None:
        """FR-7: Unknown provider → ValueError."""
        cfg = HydRAGConfig(llm_provider="nonexistent")
        with pytest.raises(ValueError, match="Unknown llm_provider"):
            create_llm_provider(cfg)

    def test_factory_hf_missing_api_base_raises(self) -> None:
        """FR-8 / AC-5: huggingface without hf_api_base → ValueError."""
        cfg = HydRAGConfig(llm_provider="huggingface")
        with pytest.raises(ValueError, match="hf_api_base"):
            create_llm_provider(cfg)

    def test_factory_openai_missing_model_raises(self) -> None:
        """FR-9 / AC-6: openai_compat without model → ValueError."""
        cfg = HydRAGConfig(
            llm_provider="openai_compat",
            openai_compat_api_base="http://oai:8080",
        )
        with pytest.raises(ValueError, match="openai_compat_model"):
            create_llm_provider(cfg)

    def test_factory_openai_missing_api_base_raises(self) -> None:
        cfg = HydRAGConfig(
            llm_provider="openai_compat",
            openai_compat_model="test",
        )
        with pytest.raises(ValueError, match="openai_compat_api_base"):
            create_llm_provider(cfg)

    def test_factory_api_tokens_passthrough(self) -> None:
        """FR-10 / AC-13: api_tokens forwarded to provider; env ignored."""
        cfg = HydRAGConfig(llm_provider="huggingface", hf_api_base="http://hf:8080")
        with patch.dict(os.environ, {"HYDRAG_HF_API_TOKEN": "env-token"}, clear=False):
            provider = create_llm_provider(cfg, api_tokens={"huggingface": "factory-token"})
        assert isinstance(provider, HuggingFaceProvider)
        assert provider._api_token == "factory-token"

    def test_factory_api_tokens_env_fallback(self) -> None:
        """When api_tokens absent, provider reads from env."""
        cfg = HydRAGConfig(llm_provider="huggingface", hf_api_base="http://hf:8080")
        with patch.dict(os.environ, {"HYDRAG_HF_API_TOKEN": "env-token"}, clear=False):
            provider = create_llm_provider(cfg)
        assert isinstance(provider, HuggingFaceProvider)
        assert provider._api_token == "env-token"


class TestProviderSerialization:
    """§12.2 #5: Request serialization per endpoint contract."""

    def test_ollama_request_body(self, mock_server: tuple[str, HTTPServer]) -> None:
        base, _ = mock_server
        _set_mock({"response": "ok"})
        provider = OllamaProvider(host=base)
        provider.generate("hello", model="test-model", timeout=5)
        body = json.loads(_MockProviderHandler.last_request_body)
        assert body["model"] == "test-model"
        assert body["prompt"] == "hello"
        assert body["stream"] is False

    def test_hf_request_body(self, mock_server: tuple[str, HTTPServer]) -> None:
        base, _ = mock_server
        _set_mock([{"generated_text": "ok"}])
        provider = HuggingFaceProvider(api_base=base, model_id="test")
        provider.generate("hello", timeout=5)
        body = json.loads(_MockProviderHandler.last_request_body)
        assert body["inputs"] == "hello"
        assert "parameters" in body

    def test_openai_chat_completions_body(self, mock_server: tuple[str, HTTPServer]) -> None:
        base, _ = mock_server
        _set_mock({"choices": [{"message": {"content": "ok"}}]})
        provider = OpenAICompatProvider(api_base=base, model="test-model")
        provider.generate("hello", timeout=5)
        body = json.loads(_MockProviderHandler.last_request_body)
        assert body["model"] == "test-model"
        assert body["messages"] == [{"role": "user", "content": "hello"}]
        assert "prompt" not in body

    def test_openai_completions_body(self, mock_server: tuple[str, HTTPServer]) -> None:
        base, _ = mock_server
        _set_mock({"choices": [{"text": "ok"}]})
        provider = OpenAICompatProvider(
            api_base=base, model="test-model", endpoint="/v1/completions"
        )
        provider.generate("hello", timeout=5)
        body = json.loads(_MockProviderHandler.last_request_body)
        assert body["model"] == "test-model"
        assert body["prompt"] == "hello"
        assert "messages" not in body

    def test_hf_bearer_token_sent(self, mock_server: tuple[str, HTTPServer]) -> None:
        base, _ = mock_server
        _set_mock([{"generated_text": "ok"}])
        provider = HuggingFaceProvider(api_base=base, api_token="my-token")
        provider.generate("test", timeout=5)
        assert _MockProviderHandler.last_request_headers.get("Authorization") == "Bearer my-token"

    def test_openai_bearer_token_sent(self, mock_server: tuple[str, HTTPServer]) -> None:
        base, _ = mock_server
        _set_mock({"choices": [{"message": {"content": "ok"}}]})
        provider = OpenAICompatProvider(api_base=base, model="m", api_token="my-key")
        provider.generate("test", timeout=5)
        assert _MockProviderHandler.last_request_headers.get("Authorization") == "Bearer my-key"


class TestSecretHandling:
    """§12.2 #6: Secret reading and logging safety."""

    def test_hf_reads_from_env_when_no_kwarg(self) -> None:
        with patch.dict(os.environ, {"HYDRAG_HF_API_TOKEN": "env-secret"}, clear=False):
            provider = HuggingFaceProvider(api_base="http://test:8080")
        assert provider._api_token == "env-secret"

    def test_hf_constructor_token_wins(self) -> None:
        """AC-13: Constructor token wins over env silently."""
        with patch.dict(os.environ, {"HYDRAG_HF_API_TOKEN": "env-secret"}, clear=False):
            provider = HuggingFaceProvider(api_base="http://test:8080", api_token="ctor-secret")
        assert provider._api_token == "ctor-secret"

    def test_openai_reads_env(self) -> None:
        with patch.dict(os.environ, {"HYDRAG_OPENAI_COMPAT_API_KEY": "env-key"}, clear=False):
            provider = OpenAICompatProvider(api_base="http://test:8080", model="m")
        assert provider._api_token == "env-key"

    def test_openai_constructor_wins(self) -> None:
        with patch.dict(os.environ, {"HYDRAG_OPENAI_COMPAT_API_KEY": "env-key"}, clear=False):
            provider = OpenAICompatProvider(api_base="http://test:8080", model="m", api_token="ctor-key")
        assert provider._api_token == "ctor-key"

    def test_token_never_in_logs(self, mock_server: tuple[str, HTTPServer], caplog: pytest.LogCaptureFixture) -> None:
        """SR-2: Token value never logged."""
        base, _ = mock_server
        _set_mock({"error": "unauthorized"}, code=401)
        provider = HuggingFaceProvider(api_base=base, api_token="super-secret-token-12345")
        with caplog.at_level(logging.DEBUG, logger="hydrag"):
            with pytest.raises(urllib.error.HTTPError):
                provider.generate("test", timeout=5)
        full_log = caplog.text
        assert "super-secret-token-12345" not in full_log


class TestErrorCategoryLogging:
    """§12.2 #7: Error category logging with extra dict keys."""

    def test_auth_error_logged_with_category(
        self,
        mock_server: tuple[str, HTTPServer],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        base, _ = mock_server
        _set_mock({"error": "unauthorized"}, code=401)
        provider = OllamaProvider(host=base)
        with caplog.at_level(logging.DEBUG, logger="hydrag"):
            with pytest.raises(urllib.error.HTTPError):
                provider.generate("test", timeout=5)
        # Check log record extra
        auth_records = [r for r in caplog.records if hasattr(r, "category") and r.category == "auth_error"]  # type: ignore[attr-defined]
        assert len(auth_records) >= 1
        assert auth_records[0].provider == "ollama"  # type: ignore[attr-defined]

    def test_server_error_logged(self, mock_server: tuple[str, HTTPServer], caplog: pytest.LogCaptureFixture) -> None:
        base, _ = mock_server
        _set_mock({"error": "internal"}, code=500)
        provider = HuggingFaceProvider(api_base=base)
        with caplog.at_level(logging.DEBUG, logger="hydrag"):
            provider.generate("test", timeout=5)
        server_records = [r for r in caplog.records if hasattr(r, "category") and r.category == "server_error"]  # type: ignore[attr-defined]
        assert len(server_records) >= 1


class TestCragModelIgnored:
    """§12.2 #8: crag_model ignored → INFO log emitted (AC-11)."""

    def test_hf_logs_crag_model_ignored(
        self,
        mock_server: tuple[str, HTTPServer],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        base, _ = mock_server
        _set_mock([{"generated_text": "SUFFICIENT"}])
        provider = HuggingFaceProvider(api_base=base, model_id="hf-default")
        with caplog.at_level(logging.INFO, logger="hydrag"):
            provider.generate("test", model="some-other-model", timeout=5)
        assert any("ignored" in r.message.lower() for r in caplog.records)


class TestURLSchemeValidation:
    """SR-4: Reject file:// and ftp:// schemes."""

    def test_ollama_rejects_file_scheme(self) -> None:
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            OllamaProvider(host="file:///etc/passwd")

    def test_hf_rejects_ftp_scheme(self) -> None:
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            HuggingFaceProvider(api_base="ftp://evil.com")

    def test_openai_rejects_file_scheme(self) -> None:
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            OpenAICompatProvider(api_base="file:///tmp/x", model="m")


class TestProtocolConformance:
    """Verify providers satisfy LLMProvider / StreamingLLMProvider protocols."""

    def test_ollama_is_llm_provider(self) -> None:
        assert isinstance(OllamaProvider(), LLMProvider)

    def test_ollama_is_streaming(self) -> None:
        assert isinstance(OllamaProvider(), StreamingLLMProvider)

    def test_hf_is_llm_provider(self) -> None:
        assert isinstance(HuggingFaceProvider(api_base="http://test:8080"), LLMProvider)

    def test_openai_is_llm_provider(self) -> None:
        assert isinstance(OpenAICompatProvider(api_base="http://test:8080", model="m"), LLMProvider)


# ═══════════════════════════════════════════════════════════════════
# §12.3 INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════


class TestCragSupervisorIntegration:
    """§12.3 #1: crag_supervisor with factory-created provider."""

    def test_crag_with_mock_provider(self) -> None:
        llm = MockLLM(response="SUFFICIENT")
        cfg = make_config()
        verdict = crag_supervisor("test query", ["chunk1", "chunk2"], llm=llm, config=cfg)
        assert verdict.sufficient is True

    def test_crag_with_none_response(self) -> None:
        llm = MockLLM(response=None)
        cfg = make_config()
        verdict = crag_supervisor("test query", ["chunk1"], llm=llm, config=cfg)
        assert verdict.sufficient is True
        assert verdict.reason == "model_unreachable"


class TestHydragSearchFactory:
    """§12.3 #2-3: hydrag_search default construction via factory and llm= override."""

    def test_default_factory_path(self) -> None:
        """§12.3 #2: hydrag_search defaults to factory provider."""
        adapter = MockAdapter(hybrid_results=["result1", "result2", "result3", "result4", "result5"])
        cfg = make_config(enable_head_2_crag=False)
        results = hydrag_search(adapter, "test query", config=cfg)
        assert len(results) > 0

    def test_explicit_llm_override(self) -> None:
        """§12.3 #3: llm= kwarg bypasses factory (FR-2)."""
        adapter = MockAdapter(
            hybrid_results=["r1", "r2", "r3", "r4", "r5"],
            keyword_results=["r1", "r2"],  # below fast_path threshold
        )
        custom_llm = MockLLM(response="SUFFICIENT")
        cfg = make_config(enable_fast_path=False)  # disable fast path so CRAG runs
        results = hydrag_search(adapter, "test query", config=cfg, llm=custom_llm)
        assert len(results) > 0
        assert len(custom_llm._calls) > 0  # custom LLM was actually used


# ═══════════════════════════════════════════════════════════════════
# §12.4 REGRESSION TESTS
# ═══════════════════════════════════════════════════════════════════


class TestOllamaDefaultRegression:
    """§12.4: Ollama-only config → OllamaProvider via factory (AC-1)."""

    def test_default_config_creates_ollama(self) -> None:
        """§12.4 #1: No llm_provider set → OllamaProvider."""
        cfg = HydRAGConfig()
        provider = create_llm_provider(cfg)
        assert isinstance(provider, OllamaProvider)

    def test_default_config_preserves_host(self) -> None:
        cfg = HydRAGConfig(ollama_host="http://custom:11434")
        provider = create_llm_provider(cfg)
        assert isinstance(provider, OllamaProvider)
        assert provider._host == "http://custom:11434"

    def test_hydrag_class_default_path(self) -> None:
        """§12.4 #2: HydRAG class wrapper works with default config."""
        adapter = MockAdapter(
            hybrid_results=["r1", "r2", "r3"],
            keyword_results=["r1", "r2", "r3"],
        )
        cfg = make_config(enable_head_2_crag=False)
        engine = HydRAG(adapter, config=cfg)
        results = engine.search("test")
        assert len(results) > 0

    def test_head_gating_unchanged(self) -> None:
        """§12.4 #3: Head gating unchanged."""
        adapter = MockAdapter(
            hybrid_results=["r1", "r2", "r3"],
            keyword_results=["r1", "r2", "r3"],
        )
        cfg = make_config(enable_head_2_crag=False, enable_head_3a_semantic=False)
        results = hydrag_search(adapter, "test", config=cfg)
        assert len(results) > 0
