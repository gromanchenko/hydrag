"""Shared test fixtures for hydrag-core tests."""

from dataclasses import dataclass, field

from hydrag import HydRAGConfig, LLMProvider, VectorStoreAdapter


@dataclass
class MockAdapter:
    """Minimal VectorStoreAdapter for testing."""

    semantic_results: list[str] = field(default_factory=list)
    keyword_results: list[str] = field(default_factory=list)
    hybrid_results: list[str] = field(default_factory=list)
    crag_results: list[str] = field(default_factory=list)
    graph_results: list[str] = field(default_factory=list)
    rewrite_result: str = ""
    _calls: list[tuple[str, str, int]] = field(default_factory=list)

    def semantic_search(self, query: str, n_results: int = 5) -> list[str]:
        self._calls.append(("semantic_search", query, n_results))
        return self.semantic_results[:n_results]

    def keyword_search(self, query: str, n_results: int = 5) -> list[str]:
        self._calls.append(("keyword_search", query, n_results))
        return self.keyword_results[:n_results]

    def hybrid_search(self, query: str, n_results: int = 5) -> list[str]:
        self._calls.append(("hybrid_search", query, n_results))
        return self.hybrid_results[:n_results]

    def crag_search(self, query: str, n_results: int = 5) -> list[str]:
        self._calls.append(("crag_search", query, n_results))
        return self.crag_results[:n_results]

    def graph_search(self, query: str, n_results: int = 5) -> list[str]:
        self._calls.append(("graph_search", query, n_results))
        return self.graph_results[:n_results]

    def rewrite_query(self, query: str) -> str:
        self._calls.append(("rewrite_query", query, 0))
        return self.rewrite_result or query


class MinimalAdapter:
    """Adapter with only the 3 required methods (no crag/graph/rewrite)."""

    def __init__(self, results: list[str] | None = None) -> None:
        self._results = results or []

    def semantic_search(self, query: str, n_results: int = 5) -> list[str]:
        return self._results[:n_results]

    def keyword_search(self, query: str, n_results: int = 5) -> list[str]:
        return self._results[:n_results]

    def hybrid_search(self, query: str, n_results: int = 5) -> list[str]:
        return self._results[:n_results]


@dataclass
class MockLLM:
    """LLMProvider that returns pre-configured responses."""

    response: str | None = "SUFFICIENT"
    _calls: list[tuple[str, str, int]] = field(default_factory=list)

    def generate(self, prompt: str, model: str = "", timeout: int = 30) -> str | None:
        self._calls.append(("generate", model, timeout))
        return self.response

    def generate_stream(self, prompt: str, model: str = "", timeout: int = 30) -> str | None:
        self._calls.append(("generate_stream", model, timeout))
        return self.response


# Verify protocol conformance at import time
assert isinstance(MockAdapter(), VectorStoreAdapter)
assert isinstance(MinimalAdapter(), VectorStoreAdapter)
assert isinstance(MockLLM(), LLMProvider)


def make_config(**overrides: object) -> HydRAGConfig:
    """Create a test config with sensible defaults."""
    defaults = {
        "profile": "prose",
        "crag_model": "test-model",
        "crag_timeout": 5,
        "ollama_host": "http://localhost:11434",
        "enable_web_fallback": False,
        "rrf_k": 60,
        "min_candidate_pool": 8,
    }
    defaults.update(overrides)
    return HydRAGConfig(**defaults)  # type: ignore[arg-type]
