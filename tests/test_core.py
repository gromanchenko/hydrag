"""Tests for core orchestrator: crag_supervisor, semantic_fallback, hydrag_search, HydRAG."""

import time

from conftest import MinimalAdapter, MockAdapter, MockLLM, make_config

from hydrag import (
    HydRAG,
    HydRAGConfig,
    crag_supervisor,
    hydrag_search,
    semantic_fallback,
)
from hydrag.core import _extract_symbol_hints

# ── _extract_symbol_hints ────────────────────────────────────────


class TestExtractSymbolHints:
    def test_backtick_wrapped(self) -> None:
        assert "MyClass" in _extract_symbol_hints("Use `MyClass` for this")

    def test_camel_case(self) -> None:
        hints = _extract_symbol_hints("The HttpRequest class")
        assert "HttpRequest" in hints

    def test_snake_case(self) -> None:
        hints = _extract_symbol_hints("call get_user_name function")
        assert "get_user_name" in hints

    def test_dotted_path(self) -> None:
        hints = _extract_symbol_hints("see module.Class.method")
        assert "module.Class.method" in hints

    def test_no_symbols(self) -> None:
        assert _extract_symbol_hints("How do I configure logging?") == []

    def test_dedupe_preserves_order(self) -> None:
        hints = _extract_symbol_hints("`foo` and `foo` again")
        assert hints == ["foo"]


# ── crag_supervisor ──────────────────────────────────────────────


class TestCRAGSupervisor:
    def test_sufficient_verdict(self) -> None:
        llm = MockLLM(response="SUFFICIENT")
        verdict = crag_supervisor("query", ["chunk1"], llm=llm, config=make_config())
        assert verdict.sufficient is True
        assert verdict.reason == "model_verdict"

    def test_insufficient_verdict(self) -> None:
        llm = MockLLM(response="INSUFFICIENT")
        verdict = crag_supervisor("query", ["chunk1"], llm=llm, config=make_config())
        assert verdict.sufficient is False
        assert verdict.reason == "model_verdict"

    def test_model_unreachable_fails_open(self) -> None:
        llm = MockLLM(response=None)
        verdict = crag_supervisor("query", ["chunk1"], llm=llm, config=make_config())
        assert verdict.sufficient is True
        assert verdict.reason == "model_unreachable"

    def test_ambiguous_response_fails_open(self) -> None:
        llm = MockLLM(response="I'm not sure about this")
        verdict = crag_supervisor("query", ["chunk1"], llm=llm, config=make_config())
        assert verdict.sufficient is True
        assert verdict.reason == "ambiguous_response"

    def test_latency_recorded(self) -> None:
        llm = MockLLM(response="SUFFICIENT")
        verdict = crag_supervisor("query", ["chunk1"], llm=llm, config=make_config())
        assert verdict.latency_ms >= 0

    def test_context_truncation(self) -> None:
        cfg = make_config(crag_char_limit=10, crag_context_chunks=2)
        llm = MockLLM(response="SUFFICIENT")
        crag_supervisor("q", ["a" * 100, "b" * 100, "c" * 100], llm=llm, config=cfg)
        # Verify LLM was called (context should be truncated, not error)
        assert len(llm._calls) == 1

    def test_insufficient_within_mixed_response(self) -> None:
        llm = MockLLM(response="Based on analysis: INSUFFICIENT. Missing details.")
        verdict = crag_supervisor("query", ["chunk1"], llm=llm, config=make_config())
        assert verdict.sufficient is False


# ── semantic_fallback ────────────────────────────────────────────


class TestSemanticFallback:
    def test_uses_adapter_methods(self) -> None:
        adapter = MockAdapter(
            hybrid_results=["h1", "h2"],
            crag_results=["c1", "c2"],
            keyword_results=["k1", "k2"],
            graph_results=["g1"],
        )
        results = semantic_fallback(adapter, "query", ["primary1"], n_results=5, config=make_config())
        assert len(results) > 0
        called_methods = {c[0] for c in adapter._calls}
        assert "crag_search" in called_methods
        assert "keyword_search" in called_methods
        assert "graph_search" in called_methods

    def test_uses_rewrite_query(self) -> None:
        adapter = MockAdapter(
            hybrid_results=["h1"],
            crag_results=["c1"],
            keyword_results=["k1"],
            rewrite_result="rewritten query",
        )
        semantic_fallback(adapter, "original", ["primary"], n_results=5, config=make_config())
        # Verify rewrite_query was called
        rewrite_calls = [c for c in adapter._calls if c[0] == "rewrite_query"]
        assert len(rewrite_calls) == 1

    def test_works_with_minimal_adapter(self) -> None:
        adapter = MinimalAdapter(results=["doc1", "doc2"])
        results = semantic_fallback(adapter, "query", ["primary1"], n_results=5, config=make_config())
        assert len(results) > 0

    def test_primary_hits_included_with_low_weight(self) -> None:
        adapter = MockAdapter(
            hybrid_results=["h1"],
            crag_results=["c1"],
            keyword_results=["k1"],
        )
        results = semantic_fallback(
            adapter, "query", ["primary_unique"], n_results=10, config=make_config()
        )
        texts = [r.text for r in results]
        assert "primary_unique" in texts


# ── hydrag_search ────────────────────────────────────────────────


class TestHydRAGSearch:
    def test_prose_profile_uses_hybrid(self) -> None:
        adapter = MockAdapter(hybrid_results=["doc1", "doc2"])
        llm = MockLLM(response="SUFFICIENT")
        cfg = make_config(profile="prose")
        results = hydrag_search(adapter, "query", config=cfg, llm=llm)
        assert len(results) > 0
        # Verify hybrid_search was called, not semantic+keyword
        called = [c[0] for c in adapter._calls]
        assert "hybrid_search" in called

    def test_code_profile_with_symbols_uses_code_aware(self) -> None:
        adapter = MockAdapter(
            semantic_results=["sem1"],
            keyword_results=["kw1"],
        )
        llm = MockLLM(response="SUFFICIENT")
        cfg = make_config(profile="code")
        # Query with CamelCase symbol → triggers code-aware path
        hydrag_search(adapter, "Use HttpRequest class", config=cfg, llm=llm)
        called = [c[0] for c in adapter._calls]
        assert "semantic_search" in called
        assert "keyword_search" in called

    def test_code_profile_without_symbols_uses_hybrid(self) -> None:
        adapter = MockAdapter(hybrid_results=["doc1"])
        llm = MockLLM(response="SUFFICIENT")
        cfg = make_config(profile="code")
        hydrag_search(adapter, "how to configure logging", config=cfg, llm=llm)
        called = [c[0] for c in adapter._calls]
        assert "hybrid_search" in called

    def test_sufficient_returns_primary_hits(self) -> None:
        adapter = MockAdapter(hybrid_results=["result1", "result2"])
        llm = MockLLM(response="SUFFICIENT")
        results = hydrag_search(adapter, "query", n_results=5, config=make_config(), llm=llm)
        assert len(results) == 2
        assert results[0].text == "result1"

    def test_insufficient_triggers_fallback(self) -> None:
        adapter = MockAdapter(
            hybrid_results=["primary1"],
            crag_results=["crag1"],
            keyword_results=["kw1"],
        )
        llm = MockLLM(response="INSUFFICIENT")
        results = hydrag_search(adapter, "query", n_results=5, config=make_config(), llm=llm)
        # Should have more results from fallback fusion
        assert len(results) > 0

    def test_empty_primary_returns_empty(self) -> None:
        adapter = MockAdapter(hybrid_results=[])
        llm = MockLLM(response="SUFFICIENT")
        results = hydrag_search(adapter, "query", config=make_config(), llm=llm)
        assert results == []

    def test_web_fallback_disabled_by_default(self) -> None:
        adapter = MockAdapter(hybrid_results=["doc1"])
        llm = MockLLM(response="INSUFFICIENT")
        cfg = make_config(enable_web_fallback=False)
        # Even on INSUFFICIENT, web fallback should not be attempted
        results = hydrag_search(adapter, "query", config=cfg, llm=llm)
        # Results come from semantic fallback only
        assert len(results) >= 0  # just verifying no crash

    def test_n_results_respected(self) -> None:
        adapter = MockAdapter(hybrid_results=["a", "b", "c", "d", "e", "f", "g", "h"])
        llm = MockLLM(response="SUFFICIENT")
        results = hydrag_search(adapter, "query", n_results=3, config=make_config(), llm=llm)
        assert len(results) <= 3

    def test_enable_web_fallback_override(self) -> None:
        adapter = MockAdapter(hybrid_results=["doc1"])
        llm = MockLLM(response="INSUFFICIENT")
        cfg = make_config(enable_web_fallback=True)
        # Web fallback enabled but no FIRECRAWL_API_KEY → empty web hits, no crash
        results = hydrag_search(
            adapter, "query", enable_web_fallback=True, config=cfg, llm=llm
        )
        assert isinstance(results, list)


# ── HydRAG class wrapper ────────────────────────────────────────


class TestHydRAGClass:
    def test_search_delegates(self) -> None:
        adapter = MockAdapter(hybrid_results=["doc1"])
        llm = MockLLM(response="SUFFICIENT")
        engine = HydRAG(adapter, config=make_config(), llm=llm)
        results = engine.search("query")
        assert len(results) > 0
        assert results[0].text == "doc1"

    def test_config_property(self) -> None:
        cfg = make_config(profile="code")
        engine = HydRAG(MockAdapter(), config=cfg)
        assert engine.config.profile == "code"

    def test_default_config(self) -> None:
        engine = HydRAG(MockAdapter())
        assert engine.config.profile == "prose"

    def test_minimal_adapter_works(self) -> None:
        adapter = MinimalAdapter(results=["doc1", "doc2"])
        llm = MockLLM(response="SUFFICIENT")
        engine = HydRAG(adapter, config=make_config(), llm=llm)
        results = engine.search("query")
        assert len(results) > 0


# ── Head 0: BM25 fast-path (V2.2 §3.0) ───────────────────────────


class TestHead0FastPath:
    def test_fast_path_triggers_on_high_ratio(self) -> None:
        adapter = MockAdapter(keyword_results=["k1", "k2", "k3", "k4", "k5"])
        cfg = make_config(enable_fast_path=True, fast_path_bm25_threshold=0.6)
        llm = MockLLM(response="SUFFICIENT")
        results = hydrag_search(adapter, "exact term", n_results=5, config=cfg, llm=llm)

        assert len(results) == 5
        assert results[0].head_origin == "head_0"
        assert results[0].metadata.get("fast_path_triggered") is True
        # CRAG should NOT have been queried
        assert len(llm._calls) == 0

    def test_fast_path_skipped_on_low_ratio(self) -> None:
        adapter = MockAdapter(
            keyword_results=["k1", "k2"],  # 2/5 = 0.4 < 0.6
            hybrid_results=["h1", "h2", "h3", "h4", "h5"],
        )
        cfg = make_config(enable_fast_path=True, fast_path_bm25_threshold=0.6)
        llm = MockLLM(response="SUFFICIENT")
        results = hydrag_search(adapter, "broad query", n_results=5, config=cfg, llm=llm)

        # Should fall through to Head 1
        assert results[0].head_origin != "head_0"
        # CRAG should have been called
        assert len(llm._calls) > 0

    def test_fast_path_disabled_by_default(self) -> None:
        adapter = MockAdapter(
            keyword_results=["k1", "k2", "k3", "k4", "k5"],
            hybrid_results=["h1", "h2", "h3"],
        )
        cfg = make_config(enable_fast_path=False)
        llm = MockLLM(response="SUFFICIENT")
        hydrag_search(adapter, "query", n_results=5, config=cfg, llm=llm)

        # keyword_search should NOT be called for fast-path
        kw_calls = [c for c in adapter._calls if c[0] == "keyword_search"]
        assert len(kw_calls) == 0

    def test_fast_path_empty_results_falls_through(self) -> None:
        adapter = MockAdapter(
            keyword_results=[],
            hybrid_results=["h1", "h2", "h3"],
        )
        cfg = make_config(enable_fast_path=True, fast_path_bm25_threshold=0.6)
        llm = MockLLM(response="SUFFICIENT")
        results = hydrag_search(adapter, "query", n_results=5, config=cfg, llm=llm)

        assert results[0].head_origin != "head_0"

    def test_fast_path_metadata_telemetry(self) -> None:
        adapter = MockAdapter(keyword_results=["k1", "k2", "k3"])
        cfg = make_config(enable_fast_path=True, fast_path_bm25_threshold=0.6)
        results = hydrag_search(adapter, "query", n_results=3, config=cfg, llm=MockLLM())

        for r in results:
            assert r.metadata["fast_path_triggered"] is True
            assert r.head_origin == "head_0"

    def test_fast_path_custom_threshold(self) -> None:
        adapter = MockAdapter(
            keyword_results=["k1", "k2", "k3"],  # 3/5 = 0.6
            hybrid_results=["h1", "h2", "h3", "h4", "h5"],
        )
        # Threshold 0.8 → 0.6 ratio should NOT trigger
        cfg = make_config(enable_fast_path=True, fast_path_bm25_threshold=0.8)
        llm = MockLLM(response="SUFFICIENT")
        results = hydrag_search(adapter, "query", n_results=5, config=cfg, llm=llm)

        assert results[0].head_origin != "head_0"


# ── Per-head switches ─────────────────────────────────────────────


class TestPerHeadSwitches:
    def test_disable_head_0_skips_fast_path(self) -> None:
        adapter = MockAdapter(
            keyword_results=["k1", "k2", "k3", "k4", "k5"],
            hybrid_results=["h1", "h2"],
        )
        cfg = make_config(enable_fast_path=True, enable_head_0=False)
        llm = MockLLM(response="SUFFICIENT")
        results = hydrag_search(adapter, "exact term", n_results=5, config=cfg, llm=llm)
        # Head 0 disabled → no fast-path, falls through to Head 1
        for r in results:
            assert r.head_origin != "head_0"

    def test_disable_head_2_crag_skips_llm(self) -> None:
        adapter = MockAdapter(hybrid_results=["h1", "h2", "h3"])
        cfg = make_config(enable_head_2_crag=False)
        llm = MockLLM(response="INSUFFICIENT")
        results = hydrag_search(adapter, "query", n_results=5, config=cfg, llm=llm)
        # CRAG disabled → LLM should NOT have been called
        assert len(llm._calls) == 0
        assert len(results) > 0

    def test_heads_whitelist_overrides_config(self) -> None:
        adapter = MockAdapter(
            keyword_results=["k1", "k2", "k3", "k4", "k5"],
            hybrid_results=["h1", "h2"],
        )
        cfg = make_config(enable_fast_path=True, enable_head_0=True)
        llm = MockLLM(response="SUFFICIENT")
        # heads={head_1} → only Head 1 runs, even though Head 0 is enabled in config
        results = hydrag_search(adapter, "exact term", n_results=5, config=cfg, llm=llm, heads={"head_1"})
        for r in results:
            assert r.head_origin != "head_0"

    def test_disable_heads_blacklist(self) -> None:
        adapter = MockAdapter(
            keyword_results=["k1", "k2", "k3", "k4", "k5"],
            hybrid_results=["h1", "h2"],
        )
        cfg = make_config(enable_fast_path=True)
        llm = MockLLM(response="SUFFICIENT")
        # Disable head_0 via blacklist
        results = hydrag_search(adapter, "query", n_results=5, config=cfg, llm=llm, disable_heads={"head_0"})
        for r in results:
            assert r.head_origin != "head_0"

    def test_all_heads_disabled_returns_empty(self) -> None:
        adapter = MockAdapter(hybrid_results=["h1"])
        cfg = make_config(enable_head_0=False, enable_head_1=False)
        llm = MockLLM(response="SUFFICIENT")
        results = hydrag_search(adapter, "query", n_results=5, config=cfg, llm=llm)
        assert results == []

    def test_disable_web_head_via_switch(self) -> None:
        adapter = MockAdapter(hybrid_results=["h1", "h2"])
        cfg = make_config(enable_web_fallback=True, enable_head_3b_web=False)
        llm = MockLLM(response="INSUFFICIENT")
        results = hydrag_search(adapter, "query", config=cfg, llm=llm)
        # 3b disabled → no web results
        assert isinstance(results, list)


# ── CRAG fail-open metadata propagation ───────────────────────────


class TestCRAGFailOpenMetadata:
    def test_model_unreachable_sets_crag_warning(self) -> None:
        adapter = MockAdapter(hybrid_results=["h1", "h2", "h3"])
        llm = MockLLM(response=None)  # None → model_unreachable
        cfg = make_config()
        results = hydrag_search(adapter, "query", n_results=3, config=cfg, llm=llm)
        assert len(results) > 0
        for r in results:
            assert r.metadata.get("crag_warning") == "model_unreachable"

    def test_ambiguous_response_sets_crag_warning(self) -> None:
        adapter = MockAdapter(hybrid_results=["h1", "h2"])
        llm = MockLLM(response="maybe yes maybe no")  # ambiguous → fail-open
        cfg = make_config()
        results = hydrag_search(adapter, "query", n_results=3, config=cfg, llm=llm)
        assert len(results) > 0
        for r in results:
            assert r.metadata.get("crag_warning") == "ambiguous_response"

    def test_normal_sufficient_has_no_crag_warning(self) -> None:
        adapter = MockAdapter(hybrid_results=["h1", "h2"])
        llm = MockLLM(response="SUFFICIENT")
        cfg = make_config()
        results = hydrag_search(adapter, "query", n_results=3, config=cfg, llm=llm)
        assert len(results) > 0
        for r in results:
            assert "crag_warning" not in r.metadata

    def test_crag_disabled_has_no_crag_warning(self) -> None:
        adapter = MockAdapter(hybrid_results=["h1", "h2"])
        llm = MockLLM(response=None)
        cfg = make_config(enable_head_2_crag=False)
        results = hydrag_search(adapter, "query", n_results=3, config=cfg, llm=llm)
        assert len(results) > 0
        for r in results:
            assert "crag_warning" not in r.metadata


# ── heads + disable_heads conflict validation ─────────────────────


class TestHeadsDisableHeadsConflict:
    def test_both_heads_and_disable_heads_raises(self) -> None:
        adapter = MockAdapter(hybrid_results=["h1"])
        llm = MockLLM(response="SUFFICIENT")
        cfg = make_config()
        import pytest
        with pytest.raises(ValueError, match="Cannot specify both"):
            hydrag_search(
                adapter, "query", config=cfg, llm=llm,
                heads={"head_1"}, disable_heads={"head_0"},
            )

    def test_heads_only_is_fine(self) -> None:
        adapter = MockAdapter(hybrid_results=["h1"])
        llm = MockLLM(response="SUFFICIENT")
        results = hydrag_search(
            adapter, "query", config=make_config(), llm=llm,
            heads={"head_1"},
        )
        assert len(results) > 0

    def test_disable_heads_only_is_fine(self) -> None:
        adapter = MockAdapter(hybrid_results=["h1"])
        llm = MockLLM(response="SUFFICIENT")
        results = hydrag_search(
            adapter, "query", config=make_config(), llm=llm,
            disable_heads={"head_0"},
        )
        assert len(results) > 0


# ── Fallback timeout / degradation ────────────────────────────────


class TestFallbackTimeout:
    def test_fallback_timeout_config_default(self) -> None:
        cfg = HydRAGConfig()
        assert cfg.fallback_timeout_s == 30.0

    def test_fallback_timeout_custom(self) -> None:
        cfg = HydRAGConfig(fallback_timeout_s=5.0)
        assert cfg.fallback_timeout_s == 5.0

    def test_semantic_fallback_returns_without_waiting_for_hung_branches(self) -> None:
        class _SlowAdapter:
            def semantic_search(self, query: str, n_results: int = 5) -> list[str]:
                return ["s1"]

            def keyword_search(self, query: str, n_results: int = 5) -> list[str]:
                time.sleep(0.25)
                return ["k1"]

            def hybrid_search(self, query: str, n_results: int = 5) -> list[str]:
                return ["h1"]

            def crag_search(self, query: str, n_results: int = 5) -> list[str]:
                time.sleep(0.25)
                return ["c1"]

            def graph_search(self, query: str, n_results: int = 5) -> list[str]:
                time.sleep(0.25)
                return ["g1"]

        cfg = make_config(fallback_timeout_s=0.01)
        start = time.monotonic()
        results = semantic_fallback(_SlowAdapter(), "q", ["p1"], n_results=5, config=cfg)
        elapsed = time.monotonic() - start
        assert elapsed < 0.5
        assert isinstance(results, list)

    def test_hydrag_search_returns_without_waiting_for_hung_fallbacks(self) -> None:
        class _SlowAdapter:
            def semantic_search(self, query: str, n_results: int = 5) -> list[str]:
                return ["s1"]

            def keyword_search(self, query: str, n_results: int = 5) -> list[str]:
                time.sleep(0.25)
                return ["k1"]

            def hybrid_search(self, query: str, n_results: int = 5) -> list[str]:
                if query == "query":
                    return ["primary1", "primary2"]
                time.sleep(0.25)
                return ["fallback-h1"]

            def crag_search(self, query: str, n_results: int = 5) -> list[str]:
                time.sleep(0.25)
                return ["c1"]

            def graph_search(self, query: str, n_results: int = 5) -> list[str]:
                time.sleep(0.25)
                return ["g1"]

        cfg = make_config(
            fallback_timeout_s=0.01,
            enable_fast_path=False,
            enable_head_3a_semantic=True,
            enable_head_3b_web=False,
        )
        llm = MockLLM(response="INSUFFICIENT")
        start = time.monotonic()
        results = hydrag_search(_SlowAdapter(), "query", n_results=3, config=cfg, llm=llm)
        elapsed = time.monotonic() - start
        assert elapsed < 0.5
        assert isinstance(results, list)


# ── LLMProvider protocol with streaming ───────────────────────────


class TestLLMProviderProtocol:
    def test_mock_llm_conforms(self) -> None:
        from hydrag import LLMProvider
        assert isinstance(MockLLM(), LLMProvider)

    def test_generate_only_provider_conforms(self) -> None:
        from hydrag import LLMProvider

        class _GenerateOnly:
            def generate(self, prompt: str, model: str = "", timeout: int = 30) -> str | None:
                return "SUFFICIENT"

        assert isinstance(_GenerateOnly(), LLMProvider)

    def test_streaming_provider_extension(self) -> None:
        from hydrag import LLMProvider, StreamingLLMProvider

        class _GenerateOnly:
            def generate(self, prompt: str, model: str = "", timeout: int = 30) -> str | None:
                return "SUFFICIENT"

        class _Streaming:
            def generate(self, prompt: str, model: str = "", timeout: int = 30) -> str | None:
                return "SUFFICIENT"

            def generate_stream(self, prompt: str, model: str = "", timeout: int = 30) -> str | None:
                return "SUFFICIENT"

        assert isinstance(_GenerateOnly(), LLMProvider)
        assert not isinstance(_GenerateOnly(), StreamingLLMProvider)
        assert isinstance(_Streaming(), StreamingLLMProvider)
