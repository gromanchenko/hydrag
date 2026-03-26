"""SurrealDB-backed adapter implementing VectorStoreAdapter.

Uses the ``surrealdb`` Python SDK over WebSocket, bridged to sync
via a daemon-thread event loop.  Requires ``pip install hydrag-core[surrealdb]``.

Ticket: T-888
RFC: docs/rfcs/SURREALDB_ADAPTER_RFC-Claude-Opus-4.6-v3.md
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import threading
import urllib.error
import urllib.request
from typing import Any, Callable, Coroutine, TypeVar
from urllib.parse import urlparse, urlunparse

try:
    # surrealdb>=1.0 exposes blocking Surreal and async AsyncSurreal.
    # The adapter expects async methods, so prefer AsyncSurreal when available.
    try:
        from surrealdb import AsyncSurreal as Surreal  # type: ignore[import-untyped]
    except ImportError:
        from surrealdb import Surreal  # type: ignore[import-untyped]
except ImportError as _surreal_err:
    raise ImportError(
        "SurrealDB adapter requires the 'surrealdb' package. "
        "Install it with: pip install hydrag-core[surrealdb]"
    ) from _surreal_err

from .fusion import rrf_fuse
from .sqlite_store import IndexedChunk

log = logging.getLogger("hydrag.surreal_adapter")

T = TypeVar("T")

# --------------------------------------------------------------------------- #
#  Allowed graph edge types (whitelist for f-string interpolation safety)
# --------------------------------------------------------------------------- #
_VALID_EDGE_TYPES = frozenset({"calls", "imports", "references"})

# --------------------------------------------------------------------------- #
#  Graph traversal SQL constant
# --------------------------------------------------------------------------- #
GRAPH_TRAVERSAL_SQL = """\
SELECT
    array::flatten([
        (SELECT raw_content FROM <-calls<-chunks LIMIT $limit),
        (SELECT raw_content FROM <-imports<-chunks LIMIT $limit),
        (SELECT raw_content FROM <-references<-chunks LIMIT $limit)
    ]) AS inbound,
    array::flatten([
        (SELECT raw_content FROM ->calls->chunks LIMIT $limit),
        (SELECT raw_content FROM ->imports->chunks LIMIT $limit),
        (SELECT raw_content FROM ->references->chunks LIMIT $limit)
    ]) AS outbound
FROM type::thing('chunks', $anchor_id);
"""


# =========================================================================== #
#  AsyncBridge — daemon-thread event loop for sync ↔ async bridging
# =========================================================================== #
class _AsyncBridge:
    """Runs an asyncio event loop in a dedicated daemon thread.

    Provides thread-safe submission of coroutines from any calling context.
    Handles lifecycle, backpressure, cancellation, and fork safety.
    """

    def __init__(self, max_in_flight: int = 256) -> None:
        self._max_in_flight = max_in_flight
        self._semaphore: asyncio.Semaphore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._pid: int = os.getpid()
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._write_lock: asyncio.Lock | None = None
        self._start()

    def _start(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._semaphore = None
        self._write_lock = None
        self._ready.clear()

        async def _init_sync_primitives() -> None:
            self._semaphore = asyncio.Semaphore(self._max_in_flight)
            self._write_lock = asyncio.Lock()

        def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_init_sync_primitives())
            self._ready.set()
            loop.run_forever()

        self._thread = threading.Thread(
            target=_run_loop,
            args=(self._loop,),
            daemon=True,
            name="hydrag-surreal-bridge",
        )
        self._thread.start()
        self._pid = os.getpid()

    def _ensure_alive(self) -> None:
        if os.getpid() != self._pid or self._thread is None or not self._thread.is_alive():
            with self._lock:
                if os.getpid() != self._pid or self._thread is None or not self._thread.is_alive():
                    self._start()
        if not self._ready.wait(timeout=5):
            raise RuntimeError("AsyncBridge failed to initialize within 5 seconds")

    def run(self, coro: Coroutine[Any, Any, T], timeout: float) -> T:
        self._ensure_alive()
        assert self._loop is not None

        async def _guarded() -> T:
            assert self._semaphore is not None
            async with self._semaphore:
                return await coro

        future = asyncio.run_coroutine_threadsafe(_guarded(), self._loop)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            raise ConnectionError(
                f"SurrealDB operation timed out after {timeout}s"
            ) from None
        except Exception:
            raise

    def close(self) -> None:
        loop = self._loop
        thread = self._thread
        if loop is not None and loop.is_running():
            async def _shutdown() -> None:
                tasks = [
                    t for t in asyncio.all_tasks(loop)
                    if t is not asyncio.current_task() and not t.done()
                ]
                for t in tasks:
                    t.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                loop.stop()

            asyncio.run_coroutine_threadsafe(_shutdown(), loop)
        if thread is not None and thread.is_alive():
            thread.join(timeout=2)
        self._loop = None
        self._thread = None


# =========================================================================== #
#  SurrealDBAdapter
# =========================================================================== #
class SurrealDBAdapter:
    """SurrealDB-backed adapter implementing VectorStoreAdapter.

    Uses the ``surrealdb`` Python SDK over WebSocket, bridged to sync
    via a daemon-thread event loop.
    """

    def __init__(
        self,
        url: str,
        *,
        embedding_dim: int,
        embed_fn: Callable[[str], list[float]] | None = None,
        namespace: str = "hydrag",
        database: str = "default",
        timeout: int = 30,
        auto_schema: bool = True,
        graph_anchor_count: int = 3,
        graph_max_neighbors: int = 50,
        rrf_k: int = 60,
        rrf_weights: tuple[float, float] = (1.0, 1.0),
        token: str | None = None,
        username: str | None = None,
        password: str | None = None,
        allow_insecure_auth: bool = False,
        batch_size: int = 100,
        max_in_flight: int = 256,
        graph_inbound_weight: float = 2.0,
        graph_outbound_weight: float = 1.0,
    ) -> None:
        # Validate structural parameters
        for name, val in [
            ("embedding_dim", embedding_dim),
            ("batch_size", batch_size),
            ("max_in_flight", max_in_flight),
            ("graph_anchor_count", graph_anchor_count),
            ("graph_max_neighbors", graph_max_neighbors),
            ("timeout", timeout),
            ("rrf_k", rrf_k),
        ]:
            if not isinstance(val, int) or val <= 0:
                raise ValueError(f"{name} must be a positive integer, got {val!r}")

        # URL validation
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https", "ws", "wss"):
            raise ValueError(
                f"surrealdb_url scheme must be http/https/ws/wss, got {parsed.scheme!r}"
            )

        # Credential-over-plaintext guard
        has_creds = token or (username and password)
        if has_creds and parsed.scheme in ("http", "ws"):
            if parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
                if not allow_insecure_auth:
                    raise ValueError(
                        "Auth over plaintext (http/ws) requires https/wss "
                        "or localhost. Set allow_insecure_auth=True to override."
                    )
                log.warning("auth over plaintext to %s", parsed.hostname)

        self._url = url
        self._embedding_dim = embedding_dim
        self._embed_fn = embed_fn
        self._namespace = namespace
        self._database = database
        self._timeout = timeout
        self._auto_schema = auto_schema
        self._graph_anchor_count = graph_anchor_count
        self._graph_max_neighbors = graph_max_neighbors
        self._rrf_k = rrf_k
        self._rrf_weights = rrf_weights
        self._token = token
        self._username = username
        self._password = password
        self._allow_insecure_auth = allow_insecure_auth
        self._batch_size = batch_size
        self._graph_inbound_weight = graph_inbound_weight
        self._graph_outbound_weight = graph_outbound_weight

        self._bridge = _AsyncBridge(max_in_flight=max_in_flight)
        self._db: Surreal | None = None
        self._write_lock: asyncio.Lock | None = None

    # ------------------------------------------------------------------ #
    #  Connection lifecycle
    # ------------------------------------------------------------------ #

    def _check_connection(self) -> None:
        if self._db is None:
            raise RuntimeError(
                "Adapter not connected. Use context manager or call connect()."
            )

    async def _async_connect(self) -> None:
        if self._db is not None:
            try:
                await self._db.close()
            except Exception:
                pass
        self._db = Surreal(self._url)
        await self._db.connect()
        if self._username and self._password:
            await self._db.signin({"username": self._username, "password": self._password})
            log.info("signin ok (username=%s)", self._username)
        elif self._token:
            await self._db.authenticate(self._token)
        await self._db.use(namespace=self._namespace, database=self._database)
        log.info("use ok (ns=%s, db=%s)", self._namespace, self._database)
        if self._auto_schema:
            await self._async_init_schema()

    def _connect(self) -> None:
        self._bridge.run(self._async_connect(), timeout=self._timeout)
        self._write_lock = self._bridge._write_lock

    async def _async_init_schema(self) -> None:
        schema_ddl = [
            "DEFINE TABLE IF NOT EXISTS _hydrag_meta SCHEMAFULL",
            "DEFINE FIELD IF NOT EXISTS version ON _hydrag_meta TYPE int",
            "UPDATE _hydrag_meta:current SET version = 1",
            "DEFINE TABLE IF NOT EXISTS chunks SCHEMAFULL",
            "DEFINE FIELD IF NOT EXISTS chunk_id ON chunks TYPE string",
            "DEFINE FIELD IF NOT EXISTS source ON chunks TYPE string",
            "DEFINE FIELD IF NOT EXISTS title ON chunks TYPE string",
            "DEFINE FIELD IF NOT EXISTS raw_content ON chunks TYPE string",
            "DEFINE FIELD IF NOT EXISTS summary ON chunks TYPE string",
            "DEFINE FIELD IF NOT EXISTS keywords ON chunks TYPE string",
            "DEFINE FIELD IF NOT EXISTS content_hash ON chunks TYPE string",
            "DEFINE FIELD IF NOT EXISTS embedding ON chunks TYPE array",
            "DEFINE FIELD IF NOT EXISTS metadata ON chunks TYPE object",
            "DEFINE ANALYZER IF NOT EXISTS hydrag_fts TOKENIZERS blank, class FILTERS lowercase, snowball(english)",
            (
                "DEFINE INDEX IF NOT EXISTS chunks_fts ON chunks"
                " FIELDS raw_content, keywords SEARCH ANALYZER hydrag_fts BM25"
            ),
            (
                f"DEFINE INDEX IF NOT EXISTS chunks_vec ON chunks"
                f" FIELDS embedding HNSW DIMENSION {self._embedding_dim} DIST COSINE"
            ),
            "DEFINE INDEX IF NOT EXISTS chunks_hash_idx ON chunks FIELDS content_hash UNIQUE",
            "DEFINE INDEX IF NOT EXISTS chunks_cid_idx ON chunks FIELDS chunk_id UNIQUE",
            "DEFINE TABLE IF NOT EXISTS calls SCHEMALESS",
            "DEFINE TABLE IF NOT EXISTS imports SCHEMALESS",
            "DEFINE TABLE IF NOT EXISTS references SCHEMALESS",
        ]
        for stmt in schema_ddl:
            await self._db.query_raw(stmt)  # type: ignore[union-attr]
        log.info("schema initialized (version=1, dim=%d)", self._embedding_dim)

    # ------------------------------------------------------------------ #
    #  Query layer
    # ------------------------------------------------------------------ #

    async def _async_query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        # Use query_raw() to get the full envelope [{"status":"OK","result":[...]}]
        # instead of query() which unwraps to only the first statement's inner result.
        raw = await self._db.query_raw(sql, params or {})  # type: ignore[union-attr]
        # Detect top-level RPC errors (e.g. auth failure, invalid syntax)
        if isinstance(raw, dict) and raw.get("error") is not None:
            raise RuntimeError(f"SurrealDB RPC error: {raw['error']}")
        result = raw.get("result", raw) if isinstance(raw, dict) else raw
        rows: list[dict[str, Any]] = []
        if isinstance(result, list):
            for i, stmt_result in enumerate(result):
                if isinstance(stmt_result, dict):
                    if stmt_result.get("status") == "ERR":
                        raise RuntimeError(
                            f"SurrealDB statement {i} failed: "
                            f"{stmt_result.get('result', 'unknown error')}"
                        )
                    inner = stmt_result.get("result", [])
                    if isinstance(inner, list):
                        rows.extend(inner)
                elif isinstance(stmt_result, list):
                    rows.extend(stmt_result)
                else:
                    rows.append(stmt_result)
        else:
            rows = result  # type: ignore[assignment]
        log.debug("query ok rows=%d sql=%s", len(rows), sql[:80])
        return rows

    def _query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self._check_connection()
        return self._bridge.run(self._async_query(sql, params), timeout=self._timeout)

    # ------------------------------------------------------------------ #
    #  Search methods (VectorStoreAdapter protocol)
    # ------------------------------------------------------------------ #

    def keyword_search(self, query: str, n_results: int = 5) -> list[str]:
        self._check_connection()
        result = self._query(
            "SELECT raw_content, search::score(0) + search::score(1) AS relevance "
            "FROM chunks "
            "WHERE raw_content @@ $query OR keywords @@ $query "
            "ORDER BY relevance DESC LIMIT $n_results",
            {"query": query, "n_results": n_results},
        )
        return [r["raw_content"] for r in result]

    def semantic_search(self, query: str, n_results: int = 5) -> list[str]:
        self._check_connection()
        if not isinstance(n_results, int) or n_results <= 0:
            raise ValueError(f"n_results must be a positive integer, got {n_results!r}")
        if self._embed_fn is None:
            log.debug("no embed_fn, falling back to FTS")
            return self.keyword_search(query, n_results=n_results)

        try:
            vec = self._embed_fn(query)
        except Exception:
            log.error("embed_fn failed for query, falling back to FTS", exc_info=True)
            return self.keyword_search(query, n_results=n_results)

        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        assert len(vec) == self._embedding_dim, (
            f"embed_fn returned dim={len(vec)}, expected {self._embedding_dim}"
        )
        result = self._query(
            f"SELECT raw_content FROM chunks WHERE embedding <|{int(n_results)}|> $vec",
            {"vec": vec},
        )
        return [r["raw_content"] for r in result]

    def hybrid_search(self, query: str, n_results: int = 5) -> list[str]:
        self._check_connection()
        kw_results = self.keyword_search(query, n_results=n_results * 2)
        sem_results = self.semantic_search(query, n_results=n_results * 2)
        fused = rrf_fuse(
            [(kw_results, self._rrf_weights[0]), (sem_results, self._rrf_weights[1])],
            k=self._rrf_k,
            n_results=n_results,
            head_origin="surreal_hybrid",
            trust_level="local",
        )
        return [r.text if hasattr(r, "text") else r for r in fused]

    # ------------------------------------------------------------------ #
    #  Optional: graph_search
    # ------------------------------------------------------------------ #

    def _keyword_search_with_ids(
        self, query: str, n_results: int = 5,
    ) -> list[dict[str, str]]:
        """Internal: keyword search returning chunk_id + raw_content."""
        self._check_connection()
        return self._query(
            "SELECT chunk_id, raw_content, search::score(0) + search::score(1) AS relevance "
            "FROM chunks "
            "WHERE raw_content @@ $query OR keywords @@ $query "
            "ORDER BY relevance DESC LIMIT $n_results",
            {"query": query, "n_results": n_results},
        )

    def graph_search(self, query: str, n_results: int = 5) -> list[str]:
        self._check_connection()
        anchor_rows = self._keyword_search_with_ids(
            query, n_results=self._graph_anchor_count,
        )
        if not anchor_rows:
            log.debug("graph_search: no anchors found, returning []")
            return []

        anchor_ids = [r["chunk_id"] for r in anchor_rows if "chunk_id" in r]

        neighbors: dict[str, list[Any]] = {"inbound": [], "outbound": []}
        for anchor_id in anchor_ids:
            result = self._query(
                GRAPH_TRAVERSAL_SQL,
                {"anchor_id": anchor_id, "limit": self._graph_max_neighbors},
            )
            for row in result:
                neighbors["inbound"].extend(row.get("inbound", []))
                neighbors["outbound"].extend(row.get("outbound", []))

        scored: dict[str, float] = {}
        for text in neighbors["inbound"]:
            content = text["raw_content"] if isinstance(text, dict) else text
            scored[content] = scored.get(content, 0) + self._graph_inbound_weight
        for text in neighbors["outbound"]:
            content = text["raw_content"] if isinstance(text, dict) else text
            scored[content] = scored.get(content, 0) + self._graph_outbound_weight
        ranked = sorted(scored, key=lambda t: scored[t], reverse=True)
        return ranked[:n_results]

    # ------------------------------------------------------------------ #
    #  Indexing
    # ------------------------------------------------------------------ #

    def index_documents(
        self,
        chunks: list[IndexedChunk],
        embeddings: list[list[float]] | None = None,
    ) -> int:
        self._check_connection()
        return self._bridge.run(
            self._async_index_documents(chunks, embeddings),
            timeout=self._timeout * max(1, len(chunks) // self._batch_size),
        )

    async def _async_index_documents(
        self,
        chunks: list[IndexedChunk],
        embeddings: list[list[float]] | None = None,
    ) -> int:
        assert self._write_lock is not None
        async with self._write_lock:
            total_created = 0
            for batch_start in range(0, len(chunks), self._batch_size):
                batch = chunks[batch_start: batch_start + self._batch_size]
                batch_embs = (
                    embeddings[batch_start: batch_start + self._batch_size]
                    if embeddings
                    else [None] * len(batch)
                )
                try:
                    batch_hashes = [chunk.content_hash for chunk in batch]
                    existing_rows = await self._async_query(
                        "SELECT content_hash FROM chunks "
                        "WHERE content_hash IN $hashes",
                        {"hashes": batch_hashes},
                    )
                    existing_hashes: set[str] = {
                        r["content_hash"]
                        for r in existing_rows
                        if isinstance(r, dict) and "content_hash" in r
                    }

                    # Build batch data for native INSERT (avoids multi-statement
                    # transaction queries which SurrealDB v2 may not support
                    # via the query RPC).
                    insert_batch: list[dict[str, Any]] = []
                    for chunk, emb in zip(batch, batch_embs):
                        if chunk.content_hash in existing_hashes:
                            continue
                        vec = emb
                        if vec is not None and hasattr(vec, "tolist"):
                            vec = vec.tolist()
                        insert_batch.append({
                            "id": chunk.chunk_id,
                            "chunk_id": chunk.chunk_id,
                            "source": chunk.source,
                            "title": chunk.title,
                            "raw_content": chunk.raw_content,
                            "summary": chunk.summary,
                            "keywords": chunk.keywords,
                            "content_hash": chunk.content_hash,
                            "embedding": vec or [],
                            "metadata": {},
                        })

                    if insert_batch:
                        result = await self._db.insert("chunks", insert_batch)  # type: ignore[union-attr]
                        total_created += len(insert_batch)
                        # Diagnostic: log first batch result + mid-batch count
                        if batch_start == 0:
                            log.info(
                                "first batch insert() returned %d records, "
                                "sample id: %s",
                                len(result) if isinstance(result, list) else -1,
                                result[0].get("id", "?") if isinstance(result, list) and result else "empty",
                            )
                            mid_rows = await self._async_query(
                                "SELECT count() AS total FROM chunks GROUP ALL", {},
                            )
                            mid_count = mid_rows[0]["total"] if mid_rows else 0
                            log.info("mid-batch count after first insert: %d", mid_count)
                except Exception as exc:
                    log.warning("batch insert failed: %s", exc)
                    raise

            # Post-insert count verification
            count_rows = await self._async_query(
                "SELECT count() AS total FROM chunks GROUP ALL", {},
            )
            db_count = count_rows[0]["total"] if count_rows else 0
            n_batches = (len(chunks) + self._batch_size - 1) // self._batch_size
            log.info(
                "indexed %d chunks in %d batches (db total: %d)",
                total_created, n_batches, db_count,
            )
            if db_count == 0 and total_created > 0:
                raise RuntimeError(
                    f"Post-insert verification failed: expected {total_created} "
                    f"chunks in DB but found {db_count}. Writes may be silently failing."
                )
            return total_created

    def index_edges(self, edges: list[tuple[str, str, str]]) -> int:
        self._check_connection()
        return self._bridge.run(
            self._async_index_edges(edges),
            timeout=self._timeout * max(1, len(edges) // 100),
        )

    async def _async_index_edges(self, edges: list[tuple[str, str, str]]) -> int:
        assert self._write_lock is not None
        async with self._write_lock:
            created = 0
            for src_id, edge_type, tgt_id in edges:
                if edge_type not in _VALID_EDGE_TYPES:
                    raise ValueError(f"Unknown edge type: {edge_type!r}")
                exists = await self._async_query(
                    "SELECT id FROM chunks WHERE chunk_id IN [$src, $tgt]",
                    {"src": src_id, "tgt": tgt_id},
                )
                if len(exists) < 2:
                    log.warning("orphaned edge %s->%s->%s", src_id, edge_type, tgt_id)
                    continue
                edge_hash = hashlib.sha256(
                    f"{src_id}:{edge_type}:{tgt_id}".encode()
                ).hexdigest()[:16]
                edge_exists = await self._async_query(
                    f"SELECT id FROM {edge_type} "
                    f"WHERE id = type::thing('{edge_type}', $edge_id)",
                    {"edge_id": edge_hash},
                )
                if edge_exists:
                    log.debug("edge %s already exists, skipping", edge_hash)
                    continue
                try:
                    await self._async_query(
                        f"RELATE type::thing('chunks', $src)->{edge_type}->type::thing('chunks', $tgt) "
                        f"SET id = type::thing('{edge_type}', $edge_id)",
                        {"src": src_id, "tgt": tgt_id, "edge_id": edge_hash},
                    )
                    created += 1
                except RuntimeError as exc:
                    if "already exists" in str(exc).lower():
                        log.debug("edge %s concurrently created, skipping", edge_hash)
                    else:
                        raise
            log.info("indexed %d edges (%d skipped)", created, len(edges) - created)
            return created

    # ------------------------------------------------------------------ #
    #  Health check — dual-path
    # ------------------------------------------------------------------ #

    def health_check(self) -> dict[str, bool]:
        # Path 1: Bridge health (SDK is_ready through daemon thread)
        bridge_ok = False
        try:
            if self._db is not None:
                bridge_ok = self._bridge.run(self._db.is_ready(), timeout=self._timeout)
        except Exception:
            log.warning("bridge health check failed")

        # Path 2: Direct HTTP liveness (bypasses WS bridge)
        server_ok = False
        parsed_url = urlparse(self._url)
        http_scheme = {"ws": "http", "wss": "https"}.get(
            parsed_url.scheme, parsed_url.scheme
        )
        http_url = urlunparse(parsed_url._replace(scheme=http_scheme))
        try:
            req = urllib.request.Request(f"{http_url}/health", method="GET")
            urllib.request.urlopen(req, timeout=self._timeout)  # noqa: S310
            server_ok = True
        except (urllib.error.URLError, OSError):
            log.warning("server liveness check failed")

        return {"bridge_healthy": bridge_ok, "server_healthy": server_ok}

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        try:
            if self._db is not None:
                self._bridge.run(self._db.close(), timeout=2)
        except Exception:
            pass
        self._bridge.close()
        self._db = None
        log.debug("adapter closed")

    def __enter__(self) -> SurrealDBAdapter:
        self._connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        # No-op: daemon thread dies with the process.
        # Use context manager or explicit close() for clean shutdown.
        pass
