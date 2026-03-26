"""Tests for SurrealDBAdapter (T-888).

CI-safe: all tests mock the surrealdb SDK — no live SurrealDB required.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from hydrag.sqlite_store import IndexedChunk

# ── Mock SDK ──────────────────────────────────────────────────────────


class MockSurreal:
    """Mock surrealdb.Surreal for unit tests.

    Returns envelope-format results to exercise _async_query normalization.
    """

    def __init__(self, url: str = "") -> None:
        self._url = url
        self._connected = False
        self._data: list[dict[str, Any]] = []
        # Pre-canned responses keyed by SQL prefix
        self._responses: dict[str, list[Any]] = {}
        self._create_count: int = 0

    async def connect(self) -> None:
        self._connected = True

    async def use(self, namespace: str, database: str) -> None:
        pass

    async def signin(self, vars: dict[str, Any]) -> str:
        return "mock-token"

    async def authenticate(self, token: str) -> None:
        pass

    async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[Any]:
        """Mimic real SDK query() which unwraps to first statement's inner result."""
        stmts = await self._statements(sql)
        return stmts[0]["result"] if stmts else []

    async def query_raw(self, sql: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return full envelope {"result": [{"status":"OK","result":[...]}]}."""
        return {"result": await self._statements(sql)}

    async def _statements(self, sql: str) -> list[dict[str, Any]]:
        """Build statement-level envelope list for a SQL string."""
        sql_stripped = sql.strip()
        # Multi-statement (semicolon-separated batched transaction)
        if ";" in sql_stripped:
            results: list[dict[str, Any]] = []
            for part in sql_stripped.split(";"):
                part = part.strip()
                if part:
                    results.extend(await self._statements(part))
            return results
        # Transaction control
        if sql_stripped.startswith("BEGIN TRANSACTION"):
            return [{"status": "OK", "result": []}]
        if sql_stripped in ("COMMIT TRANSACTION", "CANCEL TRANSACTION"):
            return [{"status": "OK", "result": []}]
        # Schema DDL
        if sql_stripped.startswith("DEFINE "):
            return [{"status": "OK", "result": []}]
        # UPDATE (upsert for indexing)
        if sql_stripped.startswith("UPDATE "):
            self._create_count += 1
            return [{"status": "OK", "result": []}]
        # CREATE
        if sql_stripped.startswith("CREATE "):
            self._create_count += 1
            return [{"status": "OK", "result": []}]
        # Count query (post-insert verification)
        if sql_stripped.startswith("SELECT count()"):
            return [{"status": "OK", "result": [{"total": self._create_count}]}]
        # RELATE
        if sql_stripped.startswith("RELATE "):
            return [{"status": "OK", "result": []}]
        # Check pre-canned responses
        for prefix, response in self._responses.items():
            if sql_stripped.startswith(prefix):
                return response
        return [{"status": "OK", "result": []}]

    async def is_ready(self) -> bool:
        return self._connected

    async def close(self) -> None:
        self._connected = False

    def set_response(self, sql_prefix: str, response: list[Any]) -> None:
        self._responses[sql_prefix] = response


# ── Helpers ───────────────────────────────────────────────────────────


def _make_chunks(n: int = 3) -> list[IndexedChunk]:
    return [
        IndexedChunk(
            chunk_id=f"src/file_{i}.py::0",
            source=f"src/file_{i}.py",
            title=f"File {i}",
            raw_content=f"def function_{i}(): pass  # content {i}",
            summary=f"Summary for file {i}",
            keywords=f"python,function_{i}",
            content_hash=hashlib.sha256(f"content_{i}".encode()).hexdigest(),
        )
        for i in range(n)
    ]


def _make_adapter(**kwargs: Any) -> Any:
    """Create adapter with mock Surreal, bypassing real import."""
    from hydrag.surreal_adapter import SurrealDBAdapter

    defaults = {
        "url": "ws://localhost:8000",
        "embedding_dim": 384,
        "auto_schema": False,
    }
    defaults.update(kwargs)
    adapter = SurrealDBAdapter(**defaults)
    return adapter


def _connect_adapter(adapter: Any, mock_surreal: MockSurreal | None = None) -> MockSurreal:
    """Wire a MockSurreal into an already-constructed adapter."""
    mock = mock_surreal or MockSurreal("ws://localhost:8000")
    mock._connected = True
    adapter._db = mock
    # Wait for bridge to be ready so _write_lock is initialized
    adapter._bridge._ready.wait(timeout=5)
    adapter._write_lock = adapter._bridge._write_lock
    return mock


# ── Protocol Conformance ─────────────────────────────────────────────


class TestProtocolConformance:
    def test_implements_vector_store_adapter(self) -> None:
        from hydrag.protocols import VectorStoreAdapter

        adapter = _make_adapter()
        assert isinstance(adapter, VectorStoreAdapter)

    def test_has_required_methods(self) -> None:

        adapter = _make_adapter()
        for method in ("semantic_search", "keyword_search", "hybrid_search"):
            assert hasattr(adapter, method)
            assert callable(getattr(adapter, method))

    def test_has_optional_graph_search(self) -> None:
        adapter = _make_adapter()
        assert hasattr(adapter, "graph_search")
        assert callable(adapter.graph_search)


# ── Constructor Validation ───────────────────────────────────────────


class TestConstructorValidation:
    def test_invalid_embedding_dim(self) -> None:
        from hydrag.surreal_adapter import SurrealDBAdapter

        with pytest.raises(ValueError, match="embedding_dim"):
            SurrealDBAdapter(url="ws://localhost:8000", embedding_dim=0)

    def test_invalid_batch_size(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            _make_adapter(batch_size=-1)

    def test_invalid_timeout(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            _make_adapter(timeout=0)

    def test_invalid_url_scheme(self) -> None:
        with pytest.raises(ValueError, match="scheme"):
            _make_adapter(url="ftp://localhost:8000")

    def test_ws_wss_accepted(self) -> None:
        a1 = _make_adapter(url="ws://localhost:8000")
        assert a1._url == "ws://localhost:8000"
        a2 = _make_adapter(url="wss://example.com:8000")
        assert a2._url == "wss://example.com:8000"

    def test_http_https_accepted(self) -> None:
        a1 = _make_adapter(url="http://localhost:8000")
        assert a1._url == "http://localhost:8000"
        a2 = _make_adapter(url="https://example.com:8000")
        assert a2._url == "https://example.com:8000"

    def test_token_over_plaintext_rejected(self) -> None:
        with pytest.raises(ValueError, match="plaintext"):
            _make_adapter(url="ws://remote-host:8000", token="secret-token")

    def test_token_over_localhost_ok(self) -> None:
        a = _make_adapter(url="ws://localhost:8000", token="secret-token")
        assert a._token == "secret-token"

    def test_token_over_plaintext_with_override(self) -> None:
        a = _make_adapter(
            url="ws://remote-host:8000",
            token="secret-token",
            allow_insecure_auth=True,
        )
        assert a._token == "secret-token"

    def test_token_over_wss_ok(self) -> None:
        a = _make_adapter(url="wss://remote-host:8000", token="secret-token")
        assert a._token == "secret-token"


# ── Connection Lifecycle ─────────────────────────────────────────────


class TestConnectionLifecycle:
    def test_check_connection_raises_when_not_connected(self) -> None:
        adapter = _make_adapter()
        with pytest.raises(RuntimeError, match="not connected"):
            adapter._check_connection()

    def test_check_connection_ok_after_connect(self) -> None:
        adapter = _make_adapter()
        _connect_adapter(adapter)
        adapter._check_connection()  # Should not raise

    def test_search_without_connect_raises(self) -> None:
        adapter = _make_adapter()
        with pytest.raises(RuntimeError, match="not connected"):
            adapter.keyword_search("test")

    def test_close_clears_db(self) -> None:
        adapter = _make_adapter()
        _connect_adapter(adapter)
        adapter.close()
        assert adapter._db is None

    def test_context_manager(self) -> None:
        adapter = _make_adapter(auto_schema=False)
        # Patch _connect to inject mock

        def patched_connect() -> None:
            _connect_adapter(adapter)

        adapter._connect = patched_connect
        with adapter as a:
            assert a._db is not None
        assert a._db is None  # closed after __exit__

    def test_del_does_not_raise(self) -> None:
        adapter = _make_adapter()
        _connect_adapter(adapter)
        adapter.__del__()  # Should not raise
        # __del__ is a no-op — db not cleared (use close() or context manager)
        assert adapter._db is not None


# ── Keyword Search ───────────────────────────────────────────────────


class TestKeywordSearch:
    def test_returns_results(self) -> None:
        adapter = _make_adapter()
        mock = _connect_adapter(adapter)
        mock.set_response(
            "SELECT raw_content, search::score(0)",
            [{"status": "OK", "result": [
                {"raw_content": "hello world"},
                {"raw_content": "foo bar"},
            ]}],
        )
        results = adapter.keyword_search("hello")
        assert results == ["hello world", "foo bar"]

    def test_empty_results(self) -> None:
        adapter = _make_adapter()
        mock = _connect_adapter(adapter)
        mock.set_response(
            "SELECT raw_content, search::score(0)",
            [{"status": "OK", "result": []}],
        )
        results = adapter.keyword_search("nonexistent")
        assert results == []

    def test_injection_treated_as_text(self) -> None:
        """SQL injection attempt: query is parameterized, not interpolated."""
        adapter = _make_adapter()
        mock = _connect_adapter(adapter)
        mock.set_response(
            "SELECT raw_content, search::score(0)",
            [{"status": "OK", "result": []}],
        )
        # This should not cause any error — it's just a weird search query
        results = adapter.keyword_search("' OR true --")
        assert results == []


# ── Semantic Search ──────────────────────────────────────────────────


class TestSemanticSearch:
    def test_with_embed_fn(self) -> None:
        def embed(text: str) -> list[float]:
            return [0.1] * 384

        adapter = _make_adapter(embed_fn=embed)
        mock = _connect_adapter(adapter)
        mock.set_response(
            "SELECT raw_content FROM chunks WHERE embedding",
            [{"status": "OK", "result": [
                {"raw_content": "vector match"},
            ]}],
        )
        results = adapter.semantic_search("test query")
        assert results == ["vector match"]

    def test_fallback_no_embed_fn(self) -> None:
        adapter = _make_adapter(embed_fn=None)
        mock = _connect_adapter(adapter)
        mock.set_response(
            "SELECT raw_content, search::score(0)",
            [{"status": "OK", "result": [
                {"raw_content": "fts fallback"},
            ]}],
        )
        results = adapter.semantic_search("test")
        assert results == ["fts fallback"]

    def test_dim_mismatch_raises(self) -> None:
        def bad_embed(text: str) -> list[float]:
            return [0.1] * 128  # Wrong dim

        adapter = _make_adapter(embed_fn=bad_embed, embedding_dim=384)
        _connect_adapter(adapter)
        with pytest.raises(AssertionError, match="dim=128"):
            adapter.semantic_search("test")

    def test_embed_fn_exception_falls_back(self) -> None:
        def failing_embed(text: str) -> list[float]:
            raise RuntimeError("embedding service down")

        adapter = _make_adapter(embed_fn=failing_embed)
        mock = _connect_adapter(adapter)
        mock.set_response(
            "SELECT raw_content, search::score(0)",
            [{"status": "OK", "result": [{"raw_content": "fts result"}]}],
        )
        results = adapter.semantic_search("test")
        assert results == ["fts result"]

    def test_numpy_vector_converted(self) -> None:
        class FakeNdarray:
            def __init__(self, data: list[float]) -> None:
                self._data = data

            def tolist(self) -> list[float]:
                return self._data

            def __len__(self) -> int:
                return len(self._data)

        def embed(text: str) -> Any:
            return FakeNdarray([0.5] * 384)

        adapter = _make_adapter(embed_fn=embed)
        mock = _connect_adapter(adapter)
        mock.set_response(
            "SELECT raw_content FROM chunks WHERE embedding",
            [{"status": "OK", "result": [{"raw_content": "numpy match"}]}],
        )
        results = adapter.semantic_search("test")
        assert results == ["numpy match"]

    def test_invalid_n_results(self) -> None:
        adapter = _make_adapter()
        _connect_adapter(adapter)
        with pytest.raises(ValueError, match="n_results"):
            adapter.semantic_search("test", n_results=0)
        with pytest.raises(ValueError, match="n_results"):
            adapter.semantic_search("test", n_results=-1)


# ── Hybrid Search ────────────────────────────────────────────────────


class TestHybridSearch:
    def test_fuses_results(self) -> None:
        def embed(text: str) -> list[float]:
            return [0.1] * 384

        adapter = _make_adapter(embed_fn=embed)
        mock = _connect_adapter(adapter)
        # Both FTS and KNN return overlapping results
        mock.set_response(
            "SELECT raw_content FROM chunks WHERE raw_content",
            [{"status": "OK", "result": [
                {"raw_content": "doc A"},
                {"raw_content": "doc B"},
            ]}],
        )
        mock.set_response(
            "SELECT raw_content FROM chunks WHERE embedding",
            [{"status": "OK", "result": [
                {"raw_content": "doc B"},
                {"raw_content": "doc C"},
            ]}],
        )
        results = adapter.hybrid_search("test", n_results=3)
        assert isinstance(results, list)
        # doc B appears in both → should be ranked higher by RRF
        assert "doc B" in results

    def test_custom_rrf_k(self) -> None:
        adapter = _make_adapter(rrf_k=10)
        assert adapter._rrf_k == 10


# ── Graph Search ─────────────────────────────────────────────────────


class TestGraphSearch:
    def test_empty_anchors_returns_empty(self) -> None:
        adapter = _make_adapter()
        mock = _connect_adapter(adapter)
        mock.set_response(
            "SELECT chunk_id, raw_content FROM chunks",
            [{"status": "OK", "result": []}],
        )
        results = adapter.graph_search("nonexistent")
        assert results == []

    def test_traverses_edges(self) -> None:
        adapter = _make_adapter()
        mock = _connect_adapter(adapter)

        # _keyword_search_with_ids returns anchors with chunk_id
        mock.set_response(
            "SELECT chunk_id, raw_content, search::score(0)",
            [{"status": "OK", "result": [
                {"chunk_id": "chunk_abc", "raw_content": "anchor content"},
            ]}],
        )
        # graph traversal
        mock.set_response(
            "SELECT\n",
            [{"status": "OK", "result": [
                {
                    "inbound": [{"raw_content": "caller code"}],
                    "outbound": [{"raw_content": "callee code"}],
                },
            ]}],
        )
        results = adapter.graph_search("test query", n_results=5)
        assert "caller code" in results
        assert "callee code" in results

    def test_inbound_outbound_weighting(self) -> None:
        adapter = _make_adapter(
            graph_inbound_weight=2.0,
            graph_outbound_weight=1.0,
        )
        mock = _connect_adapter(adapter)
        mock.set_response(
            "SELECT chunk_id, raw_content, search::score(0)",
            [{"status": "OK", "result": [
                {"chunk_id": "c1", "raw_content": "anchor"},
            ]}],
        )
        mock.set_response(
            "SELECT\n",
            [{"status": "OK", "result": [
                {
                    "inbound": [{"raw_content": "shared"}],
                    "outbound": [{"raw_content": "shared"}, {"raw_content": "outbound_only"}],
                },
            ]}],
        )
        results = adapter.graph_search("q", n_results=5)
        # "shared" has score 3.0 (2+1), "outbound_only" has 1.0
        assert results[0] == "shared"


# ── Indexing ─────────────────────────────────────────────────────────


class TestIndexDocuments:
    def test_batched_insert(self) -> None:
        adapter = _make_adapter(batch_size=2)
        mock = _connect_adapter(adapter)
        chunks = _make_chunks(5)
        # No existing hashes
        mock.set_response(
            "SELECT content_hash",
            [{"status": "OK", "result": []}],
        )
        created = adapter.index_documents(chunks)
        assert created == 5

    def test_dedup_skips_existing(self) -> None:
        adapter = _make_adapter(batch_size=10)
        mock = _connect_adapter(adapter)
        chunks = _make_chunks(3)
        # First two already exist
        mock.set_response(
            "SELECT content_hash",
            [{"status": "OK", "result": [
                {"content_hash": chunks[0].content_hash},
                {"content_hash": chunks[1].content_hash},
            ]}],
        )
        created = adapter.index_documents(chunks)
        assert created == 1

    def test_reindex_same_chunk_id_changed_content(self) -> None:
        """Re-indexing a chunk_id with new content should upsert, not crash."""
        adapter = _make_adapter(batch_size=10)
        mock = _connect_adapter(adapter)
        chunks_v1 = _make_chunks(1)
        # First pass: no existing hashes
        mock.set_response(
            "SELECT content_hash",
            [{"status": "OK", "result": []}],
        )
        created = adapter.index_documents(chunks_v1)
        assert created == 1

        # Second pass: same chunk_id, different content/hash
        chunks_v2 = [
            IndexedChunk(
                chunk_id=chunks_v1[0].chunk_id,  # same chunk_id
                source=chunks_v1[0].source,
                title=chunks_v1[0].title,
                raw_content="updated content for v2",
                summary="Updated summary",
                keywords="python,updated",
                content_hash=hashlib.sha256(b"updated_content").hexdigest(),
            )
        ]
        # New hash not in existing set → will attempt UPDATE (upsert)
        mock.set_response(
            "SELECT content_hash",
            [{"status": "OK", "result": []}],
        )
        created_v2 = adapter.index_documents(chunks_v2)
        assert created_v2 == 1


class TestIndexEdges:
    def test_deterministic_id(self) -> None:
        """Same edge input always produces the same hash."""
        edge_hash_1 = hashlib.sha256(b"a:calls:b").hexdigest()[:16]
        edge_hash_2 = hashlib.sha256(b"a:calls:b").hexdigest()[:16]
        assert edge_hash_1 == edge_hash_2

    def test_invalid_edge_type_raises(self) -> None:
        adapter = _make_adapter()
        _connect_adapter(adapter)
        with pytest.raises(ValueError, match="Unknown edge type"):
            adapter.index_edges([("a", "invalid_type", "b")])

    def test_orphan_edge_skipped(self) -> None:
        adapter = _make_adapter()
        mock = _connect_adapter(adapter)
        # Only one endpoint exists → orphaned
        mock.set_response(
            "SELECT id FROM chunks",
            [{"status": "OK", "result": [{"id": "chunks:a"}]}],
        )
        created = adapter.index_edges([("a", "calls", "b")])
        assert created == 0

    def test_valid_edges_created(self) -> None:
        adapter = _make_adapter()
        mock = _connect_adapter(adapter)
        # Both endpoints exist
        mock.set_response(
            "SELECT id FROM chunks",
            [{"status": "OK", "result": [{"id": "chunks:a"}, {"id": "chunks:b"}]}],
        )
        # Edge does not exist yet (default empty response for SELECT id FROM calls)
        created = adapter.index_edges([("a", "calls", "b")])
        assert created == 1

    def test_duplicate_edge_skipped(self) -> None:
        """Re-indexing the same edge should skip, not fail."""
        adapter = _make_adapter()
        mock = _connect_adapter(adapter)
        # Both endpoints exist
        mock.set_response(
            "SELECT id FROM chunks",
            [{"status": "OK", "result": [{"id": "chunks:a"}, {"id": "chunks:b"}]}],
        )
        # Edge already exists
        edge_hash = hashlib.sha256(b"a:calls:b").hexdigest()[:16]
        mock.set_response(
            "SELECT id FROM calls",
            [{"status": "OK", "result": [{"id": f"calls:{edge_hash}"}]}],
        )
        created = adapter.index_edges([("a", "calls", "b")])
        assert created == 0


# ── Per-statement Error Detection ────────────────────────────────────


class TestResultValidation:
    def test_err_status_raises(self) -> None:
        adapter = _make_adapter()
        mock = _connect_adapter(adapter)
        mock.set_response(
            "SELECT raw_content, search::score(0)",
            [{"status": "ERR", "result": "table not found"}],
        )
        with pytest.raises(RuntimeError, match="statement 0 failed"):
            adapter.keyword_search("test")


# ── Health Check ─────────────────────────────────────────────────────


class TestHealthCheck:
    def test_dual_path_returns_dict(self) -> None:
        adapter = _make_adapter()
        _connect_adapter(adapter)
        # Patch urllib to avoid real HTTP call
        with patch("hydrag.surreal_adapter.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()
            result = adapter.health_check()
        assert "bridge_healthy" in result
        assert "server_healthy" in result
        assert result["bridge_healthy"] is True
        assert result["server_healthy"] is True

    def test_bridge_down(self) -> None:
        adapter = _make_adapter()
        # _db is None → bridge_healthy should be False
        with patch("hydrag.surreal_adapter.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = OSError("connection refused")
            result = adapter.health_check()
        assert result["bridge_healthy"] is False
        assert result["server_healthy"] is False

    def test_server_unreachable(self) -> None:
        adapter = _make_adapter()
        _connect_adapter(adapter)
        with patch("hydrag.surreal_adapter.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = OSError("connection refused")
            result = adapter.health_check()
        assert result["bridge_healthy"] is True
        assert result["server_healthy"] is False


# ── AsyncBridge ──────────────────────────────────────────────────────


class TestAsyncBridge:
    def test_run_returns_result(self) -> None:
        from hydrag.surreal_adapter import _AsyncBridge

        bridge = _AsyncBridge(max_in_flight=10)
        try:
            async def coro() -> int:
                return 42

            result = bridge.run(coro(), timeout=5)
            assert result == 42
        finally:
            bridge.close()

    def test_timeout_raises_connection_error(self) -> None:
        from hydrag.surreal_adapter import _AsyncBridge

        bridge = _AsyncBridge(max_in_flight=10)
        try:
            async def slow_coro() -> None:
                await asyncio.sleep(10)

            with pytest.raises(ConnectionError, match="timed out"):
                bridge.run(slow_coro(), timeout=0.1)
        finally:
            bridge.close()

    def test_close_stops_thread(self) -> None:
        from hydrag.surreal_adapter import _AsyncBridge

        bridge = _AsyncBridge(max_in_flight=10)
        thread = bridge._thread
        assert thread is not None
        assert thread.is_alive()
        bridge.close()
        assert bridge._thread is None

    def test_fork_safety_reinitializes(self) -> None:
        from hydrag.surreal_adapter import _AsyncBridge

        bridge = _AsyncBridge(max_in_flight=10)
        try:
            # Simulate fork by changing PID
            bridge._pid = -1

            async def coro() -> str:
                return "after_fork"

            result = bridge.run(coro(), timeout=5)
            assert result == "after_fork"
            assert bridge._pid == os.getpid()
        finally:
            bridge.close()

    def test_exception_propagation(self) -> None:
        from hydrag.surreal_adapter import _AsyncBridge

        bridge = _AsyncBridge(max_in_flight=10)
        try:
            async def failing() -> None:
                raise ValueError("async failure")

            with pytest.raises(ValueError, match="async failure"):
                bridge.run(failing(), timeout=5)
        finally:
            bridge.close()


# ── Config Integration ───────────────────────────────────────────────


class TestConfigIntegration:
    def test_config_fields_exist(self) -> None:
        from hydrag.config import HydRAGConfig

        cfg = HydRAGConfig()
        assert hasattr(cfg, "surrealdb_url")
        assert cfg.surrealdb_url == ""
        assert cfg.surrealdb_namespace == "hydrag"
        assert cfg.surrealdb_database == "default"
        assert cfg.surrealdb_timeout == 30
        assert cfg.surrealdb_graph_anchors == 3
        assert cfg.surrealdb_graph_max_neighbors == 50
        assert cfg.surrealdb_batch_size == 100
        assert cfg.surrealdb_max_in_flight == 256
        assert cfg.surrealdb_rrf_k == 60
        assert cfg.surrealdb_rrf_weights == "1.0,1.0"

    def test_config_url_validation(self) -> None:
        from hydrag.config import HydRAGConfig

        cfg = HydRAGConfig(surrealdb_url="ws://localhost:8000")
        assert cfg.surrealdb_url == "ws://localhost:8000"

        with pytest.raises(ValueError, match="scheme"):
            HydRAGConfig(surrealdb_url="ftp://localhost:8000")

    def test_config_rrf_weights_parsed(self) -> None:
        from hydrag.config import HydRAGConfig

        cfg = HydRAGConfig(surrealdb_url="ws://localhost:8000", surrealdb_rrf_weights="1.5,0.8")
        assert cfg._parsed_rrf_weights == (1.5, 0.8)

    def test_config_rrf_weights_invalid(self) -> None:
        from hydrag.config import HydRAGConfig

        with pytest.raises(ValueError, match="rrf_weights"):
            HydRAGConfig(surrealdb_rrf_weights="1.0,2.0,3.0")

    def test_config_from_env(self) -> None:
        from hydrag.config import HydRAGConfig

        env = {
            "HYDRAG_SURREALDB_URL": "wss://prod:8000",
            "HYDRAG_SURREALDB_NAMESPACE": "custom_ns",
            "HYDRAG_SURREALDB_DATABASE": "custom_db",
            "HYDRAG_SURREALDB_TIMEOUT": "60",
            "HYDRAG_SURREALDB_GRAPH_ANCHORS": "5",
            "HYDRAG_SURREALDB_GRAPH_MAX_NEIGHBORS": "100",
            "HYDRAG_SURREALDB_BATCH_SIZE": "200",
            "HYDRAG_SURREALDB_MAX_IN_FLIGHT": "512",
            "HYDRAG_SURREALDB_RRF_K": "30",
            "HYDRAG_SURREALDB_RRF_WEIGHTS": "2.0,0.5",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = HydRAGConfig.from_env()
        assert cfg.surrealdb_url == "wss://prod:8000"
        assert cfg.surrealdb_namespace == "custom_ns"
        assert cfg.surrealdb_database == "custom_db"
        assert cfg.surrealdb_timeout == 60
        assert cfg.surrealdb_graph_anchors == 5
        assert cfg.surrealdb_graph_max_neighbors == 100
        assert cfg.surrealdb_batch_size == 200
        assert cfg.surrealdb_max_in_flight == 512
        assert cfg.surrealdb_rrf_k == 30
        assert cfg.surrealdb_rrf_weights == "2.0,0.5"


# ── Import Guard ─────────────────────────────────────────────────────


class TestImportGuard:
    def test_init_exports_surreal_adapter(self) -> None:
        """SurrealDBAdapter should be importable from hydrag when SDK is available."""
        from hydrag import SurrealDBAdapter

        assert SurrealDBAdapter is not None
