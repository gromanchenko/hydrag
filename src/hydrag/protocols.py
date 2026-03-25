"""Adapter protocols for pluggable vector stores and LLM providers."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class VectorStoreAdapter(Protocol):
    """Minimal vector-store contract for HydRAG.

    Required methods (must be implemented):
        semantic_search, keyword_search, hybrid_search

    Optional methods (gracefully handled if missing):
        crag_search  — falls back to hybrid_search
        graph_search — skipped when absent
        rewrite_query — returns original query unchanged when absent
    """

    def semantic_search(self, query: str, n_results: int = 5) -> list[str]: ...

    def keyword_search(self, query: str, n_results: int = 5) -> list[str]: ...

    def hybrid_search(self, query: str, n_results: int = 5) -> list[str]: ...


@runtime_checkable
class LLMProvider(Protocol):
    """Base contract for LLM inference used by the CRAG supervisor."""

    def generate(self, prompt: str, model: str = "", timeout: int = 30) -> str | None: ...


@runtime_checkable
class StreamingLLMProvider(LLMProvider, Protocol):
    """Optional streaming extension for early verdict detection.

    Providers implementing this protocol can be used with
    ``crag_stream=True`` for lower latency.
    """

    def generate_stream(self, prompt: str, model: str = "", timeout: int = 30) -> str | None:
        ...
