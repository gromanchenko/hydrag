"""Tests for fusion module: rrf_fuse, _text_of, _as_results, types."""

from hydrag import CRAGVerdict, RetrievalResult, rrf_fuse
from hydrag.fusion import _as_results, _text_of


class TestCRAGVerdict:
    def test_sufficient_verdict(self) -> None:
        v = CRAGVerdict(sufficient=True, reason="model_verdict", latency_ms=42.0)
        assert v.sufficient is True
        assert v.raw_response is None

    def test_insufficient_verdict_with_raw(self) -> None:
        v = CRAGVerdict(
            sufficient=False, reason="model_verdict", latency_ms=100.0, raw_response="INSUFFICIENT"
        )
        assert v.sufficient is False
        assert v.raw_response == "INSUFFICIENT"


class TestRetrievalResult:
    def test_defaults(self) -> None:
        r = RetrievalResult(text="hello", source="s", score=0.9, head_origin="h1", trust_level="local")
        assert r.metadata == {}
        assert r.crag_verdict is None

    def test_with_metadata(self) -> None:
        r = RetrievalResult(
            text="hello", source="s", score=0.9,
            head_origin="h1", trust_level="local", metadata={"k": "v"},
        )
        assert r.metadata["k"] == "v"


class TestTextOf:
    def test_string(self) -> None:
        assert _text_of("hello") == "hello"

    def test_retrieval_result(self) -> None:
        r = RetrievalResult(text="world", source="", score=0.5, head_origin="", trust_level="local")
        assert _text_of(r) == "world"


class TestAsResults:
    def test_wraps_strings(self) -> None:
        results = _as_results(["a", "b", "c"], head_origin="test")
        assert len(results) == 3
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].text == "a"
        assert results[0].head_origin == "test"
        assert results[0].trust_level == "local"

    def test_preserves_retrieval_results(self) -> None:
        rr = RetrievalResult(text="x", source="src", score=0.9, head_origin="orig", trust_level="web")
        results = _as_results([rr], head_origin="test")
        assert results[0] is rr  # same object, not re-wrapped

    def test_positional_score(self) -> None:
        results = _as_results(["a", "b"], head_origin="test")
        assert results[0].score == 1.0  # 1 - 0/2
        assert results[1].score == 0.5  # 1 - 1/2

    def test_web_trust_level(self) -> None:
        results = _as_results(["x"], head_origin="web", trust_level="web")
        assert results[0].trust_level == "web"


class TestRRFFuse:
    def test_single_source(self) -> None:
        results = rrf_fuse([([" a", "b", "c"], 1.0)], k=60, n_results=3)
        assert len(results) == 3
        assert results[0].score > results[1].score > results[2].score

    def test_empty_sources(self) -> None:
        results = rrf_fuse([([], 1.0)], k=60, n_results=5)
        assert results == []

    def test_multiple_sources_boost_shared_docs(self) -> None:
        # "shared" appears in both lists → should rank higher
        results = rrf_fuse(
            [
                (["shared", "only_a"], 1.0),
                (["shared", "only_b"], 1.0),
            ],
            k=60,
            n_results=5,
        )
        assert results[0].text == "shared"

    def test_weight_affects_ranking(self) -> None:
        results = rrf_fuse(
            [
                (["heavy"], 10.0),
                (["light"], 0.1),
            ],
            k=60,
            n_results=2,
        )
        assert results[0].text == "heavy"

    def test_n_results_limits_output(self) -> None:
        results = rrf_fuse(
            [(["a", "b", "c", "d", "e"], 1.0)],
            k=60,
            n_results=2,
        )
        assert len(results) == 2

    def test_tie_break_by_insertion_order(self) -> None:
        # Both have same score (appear once at rank 1 with weight 1.0)
        results = rrf_fuse(
            [
                (["first"], 1.0),
                (["second"], 1.0),
            ],
            k=60,
            n_results=2,
        )
        # Same score → first-seen wins
        assert results[0].text == "first"
        assert results[1].text == "second"

    def test_head_origin_and_trust_level_propagated(self) -> None:
        results = rrf_fuse(
            [(["doc"], 1.0)],
            k=60,
            n_results=1,
            head_origin="hydrag",
            trust_level="web",
        )
        assert results[0].head_origin == "hydrag"
        assert results[0].trust_level == "web"

    def test_mixed_string_and_retrieval_result(self) -> None:
        rr = RetrievalResult(text="typed", source="s", score=0.9, head_origin="x", trust_level="local")
        results = rrf_fuse(
            [(["plain", rr], 1.0)],
            k=60,
            n_results=2,
        )
        assert len(results) == 2
        assert results[0].text == "plain"
        assert results[1].text == "typed"

    def test_preserves_provenance_from_retrieval_result(self) -> None:
        """Regression: RRF must preserve source/metadata from first-seen RetrievalResult."""
        rr = RetrievalResult(
            text="doc_a",
            source="app.py",
            score=0.9,
            head_origin="head_1a",
            trust_level="local",
            metadata={"line": 42, "function": "main"},
        )
        results = rrf_fuse(
            [([rr, "plain_b"], 1.0)],
            k=60,
            n_results=2,
            head_origin="hydrag",
        )
        assert results[0].text == "doc_a"
        assert results[0].source == "app.py"
        assert results[0].metadata["line"] == 42
        assert results[0].metadata["function"] == "main"
        assert results[0].head_origin == "hydrag"  # overridden by caller
        # plain string gets empty source
        assert results[1].source == ""

    def test_provenance_uses_first_seen_object(self) -> None:
        """When same text from multiple sources, provenance comes from first insertion."""
        rr1 = RetrievalResult(
            text="shared",
            source="first.py",
            score=0.9,
            head_origin="head_1a",
            trust_level="local",
            metadata={"origin": "first"},
        )
        rr2 = RetrievalResult(
            text="shared",
            source="second.py",
            score=0.5,
            head_origin="head_3a",
            trust_level="web",
            metadata={"origin": "second"},
        )
        results = rrf_fuse(
            [([rr1], 1.0), ([rr2], 1.0)],
            k=60,
            n_results=1,
            head_origin="",
        )
        assert results[0].source == "first.py"
        assert results[0].metadata["origin"] == "first"

    def test_trust_level_preserved_for_web_results(self) -> None:
        """Regression: _rrf_fuse must NOT silently overwrite web trust with 'local'."""
        web_rr = RetrievalResult(
            text="web_doc",
            source="https://example.com",
            score=0.8,
            head_origin="head_3b",
            trust_level="web",
        )
        local_rr = RetrievalResult(
            text="local_doc",
            source="local.py",
            score=0.9,
            head_origin="head_1a",
            trust_level="local",
        )
        results = rrf_fuse(
            [([local_rr, web_rr], 1.0)],
            k=60,
            n_results=2,
            head_origin="hydrag",
        )
        by_text = {r.text: r for r in results}
        assert by_text["web_doc"].trust_level == "web"
        assert by_text["local_doc"].trust_level == "local"

    def test_trust_level_explicit_override(self) -> None:
        """When caller explicitly passes trust_level, it overrides originals."""
        web_rr = RetrievalResult(
            text="doc", source="", score=0.5,
            head_origin="head_3b", trust_level="web",
        )
        results = rrf_fuse(
            [([web_rr], 1.0)],
            k=60,
            n_results=1,
            trust_level="local",
        )
        assert results[0].trust_level == "local"

    def test_trust_level_none_defaults_local_for_plain_strings(self) -> None:
        """Plain string docs with no explicit trust_level get 'local'."""
        results = rrf_fuse(
            [(["plain"], 1.0)],
            k=60,
            n_results=1,
        )
        assert results[0].trust_level == "local"
