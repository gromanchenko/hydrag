"""Tests for SQLiteFTSStore (T-742) and enrichment (T-743)."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from hydrag.enrichment import OllamaKeywordExtractor
from hydrag.sqlite_store import (
    IndexedChunk,
    SQLiteFTSStore,
    _adaptive_keyword_count,
    _content_hash,
)

# ── Helpers ────────────────────────────────────────────────────────────


@dataclass
class FakeExtractor:
    """KeywordExtractor that returns deterministic results without LLM."""

    summary: str = "This is a test summary."
    keywords: list[str] = field(default_factory=lambda: [
        "python", "testing", "sqlite", "search", "indexing",
        "retrieval", "database", "fts5", "benchmark",
    ])
    call_count: int = 0

    def extract(self, text: str) -> dict[str, Any]:
        self.call_count += 1
        return {"summary": self.summary, "keywords": self.keywords}


def _make_chunks(n: int = 3) -> list[IndexedChunk]:
    return [
        IndexedChunk(
            chunk_id=f"doc::{i}",
            source=f"doc_{i}.md",
            title=f"Document {i}",
            raw_content=f"This is the content of document {i}. "
            f"It discusses python programming and sqlite databases. "
            f"Testing retrieval systems is important for search quality.",
        )
        for i in range(n)
    ]


# ── T-742: SQLiteFTSStore core ────────────────────────────────────────


class TestSQLiteFTSStore:
    def test_create_in_memory(self) -> None:
        with SQLiteFTSStore() as store:
            assert store.count() == 0
            stats = store.stats()
            assert stats["total_chunks"] == 0
            assert stats["enriched_chunks"] == 0

    def test_index_and_count(self) -> None:
        chunks = _make_chunks(5)
        with SQLiteFTSStore() as store:
            indexed = store.index_documents(chunks)
            assert indexed == 5
            assert store.count() == 5

    def test_index_deduplication(self) -> None:
        """Same content_hash should not re-index."""
        chunks = _make_chunks(2)
        with SQLiteFTSStore() as store:
            first = store.index_documents(chunks)
            assert first == 2
            second = store.index_documents(chunks)
            assert second == 0  # No new chunks

    def test_index_updates_changed_content(self) -> None:
        chunks = _make_chunks(1)
        with SQLiteFTSStore() as store:
            store.index_documents(chunks)
            chunks[0].raw_content = "Completely different content now."
            updated = store.index_documents(chunks)
            assert updated == 1

    def test_semantic_search_finds_results(self) -> None:
        chunks = _make_chunks(3)
        with SQLiteFTSStore() as store:
            store.index_documents(chunks)
            results = store.semantic_search("python programming")
            assert len(results) > 0
            assert any("python" in r.lower() for r in results)

    def test_keyword_search(self) -> None:
        chunks = _make_chunks(3)
        with SQLiteFTSStore() as store:
            store.index_documents(chunks)
            results = store.keyword_search("sqlite databases")
            assert len(results) > 0

    def test_hybrid_search(self) -> None:
        chunks = _make_chunks(3)
        with SQLiteFTSStore() as store:
            store.index_documents(chunks)
            results = store.hybrid_search("retrieval systems")
            assert len(results) > 0

    def test_search_empty_query(self) -> None:
        with SQLiteFTSStore() as store:
            store.index_documents(_make_chunks(1))
            assert store.semantic_search("") == []
            assert store.keyword_search("   ") == []

    def test_search_no_results(self) -> None:
        with SQLiteFTSStore() as store:
            store.index_documents(_make_chunks(1))
            results = store.semantic_search("xyzzyplugh")
            assert results == []

    def test_n_results_limit(self) -> None:
        chunks = _make_chunks(10)
        with SQLiteFTSStore() as store:
            store.index_documents(chunks)
            results = store.hybrid_search("python", n_results=3)
            assert len(results) <= 3

    def test_get_chunk(self) -> None:
        chunks = _make_chunks(1)
        with SQLiteFTSStore() as store:
            store.index_documents(chunks)
            retrieved = store.get_chunk("doc::0")
            assert retrieved is not None
            assert retrieved["source"] == "doc_0.md"

    def test_get_chunk_missing(self) -> None:
        with SQLiteFTSStore() as store:
            assert store.get_chunk("nonexistent") is None

    def test_persistent_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        chunks = _make_chunks(2)

        # Index
        with SQLiteFTSStore(db_path) as store:
            store.index_documents(chunks)

        # Reopen and verify
        with SQLiteFTSStore(db_path) as store:
            assert store.count() == 2
            results = store.hybrid_search("python")
            assert len(results) > 0

    def test_protocol_conformance(self) -> None:
        """SQLiteFTSStore must satisfy VectorStoreAdapter protocol."""
        from hydrag.protocols import VectorStoreAdapter

        store = SQLiteFTSStore()
        assert isinstance(store, VectorStoreAdapter)
        store.close()

    def test_fts_query_escaping(self) -> None:
        """Special characters in queries should not crash."""
        with SQLiteFTSStore() as store:
            store.index_documents(_make_chunks(1))
            # These should not raise
            store.hybrid_search("hello (world)")
            store.hybrid_search("AND OR NOT")
            store.hybrid_search("test*query")
            store.hybrid_search("")

    def test_trailing_period_no_syntax_error(self) -> None:
        """Queries ending with '.' must not raise OperationalError (regression T-FTS5-period)."""
        chunk = IndexedChunk("c1", "test", "", "biomaterials show inductive properties")
        with SQLiteFTSStore() as store:
            store.index_documents([chunk])
            # The period in "properties." previously caused FTS5 syntax error
            results = store.keyword_search("biomaterials show inductive properties.")
            assert results  # must return the document

    def test_hyphenated_query_tokenization(self) -> None:
        """Hyphens split tokens so both parts can match the index."""
        chunk = IndexedChunk("c1", "test", "", "0-dimensional biomaterials")
        with SQLiteFTSStore() as store:
            store.index_documents([chunk])
            results = store.keyword_search("0-dimensional biomaterials.")
            assert results  # must find the document via 'dimensional' or '0'

    def test_escape_fts_strips_period(self) -> None:
        """_escape_fts_query must not emit tokens ending in '.'."""
        escaped = SQLiteFTSStore._escape_fts_query("properties.")
        assert "." not in escaped, f"Period not stripped: {escaped!r}"

    def test_escape_fts_splits_hyphen(self) -> None:
        """_escape_fts_query must split on hyphens."""
        escaped = SQLiteFTSStore._escape_fts_query("0-dimensional")
        parts = escaped.split(" OR ")
        assert "0" in parts
        assert "dimensional" in parts


# ── T-742: Utility functions ──────────────────────────────────────────


class TestUtilities:
    def test_content_hash_deterministic(self) -> None:
        assert _content_hash("hello") == _content_hash("hello")
        assert _content_hash("hello") != _content_hash("world")

    def test_adaptive_keyword_count(self) -> None:
        assert _adaptive_keyword_count(50) == 5       # Floor
        assert _adaptive_keyword_count(200) == 5      # 200/200 = 1, clamped to 5
        assert _adaptive_keyword_count(2000) == 10    # 2000/200 = 10
        assert _adaptive_keyword_count(10000) == 30   # Ceiling


# ── T-743: Enrichment ─────────────────────────────────────────────────


class TestEnrichment:
    def test_index_with_extractor(self) -> None:
        """Enrichment should populate summary and keywords."""
        extractor = FakeExtractor()
        chunks = _make_chunks(2)
        with SQLiteFTSStore() as store:
            store.index_documents(chunks, extractor=extractor)
            assert extractor.call_count == 2

            chunk = store.get_chunk("doc::0")
            assert chunk is not None
            assert chunk["summary"] == "This is a test summary."
            assert chunk["keywords"] != ""

    def test_enrichment_skips_preenriched(self) -> None:
        """Chunks with existing summary/keywords skip LLM call."""
        extractor = FakeExtractor()
        chunks = _make_chunks(1)
        chunks[0].summary = "Pre-existing summary"
        chunks[0].keywords = "pre existing keywords"

        with SQLiteFTSStore() as store:
            store.index_documents(chunks, extractor=extractor)
            assert extractor.call_count == 0
            chunk = store.get_chunk("doc::0")
            assert chunk is not None
            assert chunk["summary"] == "Pre-existing summary"

    def test_enrichment_stats(self) -> None:
        extractor = FakeExtractor()
        with SQLiteFTSStore() as store:
            store.index_documents(_make_chunks(3), extractor=extractor)
            stats = store.stats()
            assert stats["enriched_chunks"] == 3

    def test_enriched_search_hits_keywords(self) -> None:
        """Keywords from enrichment should be searchable."""
        extractor = FakeExtractor(keywords=["benchmarking", "metrics", "evaluation"])
        chunks = [
            IndexedChunk(
                chunk_id="perf::0",
                source="perf.md",
                title="Performance",
                raw_content="The system runs fast under load.",
            )
        ]
        with SQLiteFTSStore() as store:
            store.index_documents(chunks, extractor=extractor)
            # "benchmarking" is not in raw_content but IS in keywords
            results = store.hybrid_search("benchmarking")
            assert len(results) > 0


# ── T-743: OllamaKeywordExtractor parsing ─────────────────────────────


class TestOllamaParseResponse:
    def test_valid_json(self) -> None:
        raw = '{"summary": "A test.", "keywords": ["a", "b", "c"]}'
        result = OllamaKeywordExtractor._parse_response(raw)
        assert result["summary"] == "A test."
        assert result["keywords"] == ["a", "b", "c"]

    def test_json_with_markdown_fences(self) -> None:
        raw = '```json\n{"summary": "Test.", "keywords": ["x"]}\n```'
        result = OllamaKeywordExtractor._parse_response(raw)
        assert result["summary"] == "Test."
        assert result["keywords"] == ["x"]

    def test_keywords_as_string(self) -> None:
        raw = '{"summary": "Sum.", "keywords": "a, b, c"}'
        result = OllamaKeywordExtractor._parse_response(raw)
        assert result["keywords"] == ["a", "b", "c"]

    def test_invalid_json(self) -> None:
        raw = "This is not JSON at all"
        result = OllamaKeywordExtractor._parse_response(raw)
        assert result["summary"] == ""
        assert result["keywords"] == []

    def test_empty_response(self) -> None:
        result = OllamaKeywordExtractor._parse_response("")
        assert result == {"summary": "", "keywords": []}


class TestFilterAnchored:
    def test_all_anchored(self) -> None:
        kws = ["python", "sqlite", "search"]
        source = "Python uses SQLite for search operations"
        result = OllamaKeywordExtractor._filter_anchored(kws, source)
        assert result == ["python", "sqlite", "search"]

    def test_partial_anchored(self) -> None:
        kws = ["python", "java", "rust", "sqlite"]
        source = "Python and SQLite are used here"
        result = OllamaKeywordExtractor._filter_anchored(kws, source)
        # 2 anchored (python, sqlite) >= 50% of 4, so anchored first
        assert result[0] == "python"
        assert result[1] == "sqlite"
        assert len(result) == 4  # keeps unanchored to fill

    def test_none_anchored(self) -> None:
        kws = ["java", "rust", "golang"]
        source = "Python and SQLite are used here"
        # 0 anchored < 50%, returns all with warning
        result = OllamaKeywordExtractor._filter_anchored(kws, source)
        assert len(result) == 3


# ── T-742: Indexer CLI ────────────────────────────────────────────────


class TestIndexerCLI:
    def test_index_directory(self, tmp_path: Path) -> None:
        """End-to-end: create files, index them, search them."""
        # Create test files
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "hello.md").write_text("# Hello\n\nThis is about Python programming.")
        (docs_dir / "world.md").write_text("# World\n\nSQLite databases are fast and reliable.")

        db_path = tmp_path / "test.db"

        from hydrag.indexer import main as indexer_main

        indexer_main([str(docs_dir), "--db", str(db_path)])

        assert db_path.exists()

        with SQLiteFTSStore(db_path) as store:
            assert store.count() >= 2
            results = store.hybrid_search("Python")
            assert len(results) > 0

    def test_index_empty_directory(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        db_path = tmp_path / "empty.db"

        from hydrag.indexer import main as indexer_main

        with pytest.raises(SystemExit) as exc_info:
            indexer_main([str(empty_dir), "--db", str(db_path)])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "No files" in captured.out


# ── T-742: Similarity Search CLI ──────────────────────────────────────


class TestSimilaritySearchCLI:
    def test_search_text_output(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        db_path = tmp_path / "search.db"
        with SQLiteFTSStore(db_path) as store:
            store.index_documents(_make_chunks(3))

        from hydrag.similarity_search import main as search_main

        search_main(["python", "--db", str(db_path)])
        captured = capsys.readouterr()
        assert "Results:" in captured.out
        assert "python" in captured.out.lower()

    def test_search_json_output(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        db_path = tmp_path / "search.db"
        with SQLiteFTSStore(db_path) as store:
            store.index_documents(_make_chunks(3))

        from hydrag.similarity_search import main as search_main

        search_main(["sqlite", "--db", str(db_path), "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "query" in data
        assert "results" in data
        assert "latency_ms" in data

    def test_search_missing_db(self, tmp_path: Path) -> None:
        from hydrag.similarity_search import main as search_main

        with pytest.raises(SystemExit):
            search_main(["test", "--db", str(tmp_path / "nonexistent.db")])
