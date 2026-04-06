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
    # RecordID is available in surrealdb>=1.0 (v2-compatible SDK).
    # Used in RELATE statements to avoid type::thing() parse failure in v2.2.1.
    try:
        from surrealdb import RecordID as _SurrealRecordID  # type: ignore[import-untyped]
        _HAS_SURREAL_RECORD_ID = True
    except ImportError:
        _HAS_SURREAL_RECORD_ID = False
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
#  Snowball(english) stop words — must match the server-side hydrag_fts analyzer
#  so that Python-side token generation does not produce dead score slots.
# --------------------------------------------------------------------------- #
_SNOWBALL_EN_STOP_WORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren", "arent", "as", "at", "be", "because",
    "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "couldn", "couldnt", "d", "did", "didn", "didnt", "do", "does",
    "doesn", "doesnt", "doing", "don", "dont", "down", "during", "each",
    "few", "for", "from", "further", "had", "hadn", "hadnt", "has", "hasn",
    "hasnt", "have", "haven", "havent", "having", "he", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
    "into", "is", "isn", "isnt", "it", "its", "itself", "just", "ll", "m",
    "ma", "me", "mightn", "mightnt", "more", "most", "mustn", "mustnt",
    "my", "myself", "needn", "neednt", "no", "nor", "not", "now", "o",
    "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shant",
    "she", "should", "shouldn", "shouldnt", "so", "some", "such", "t",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "ve", "very", "was", "wasn", "wasnt", "we",
    "were", "weren", "werent", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "won", "wont", "would", "wouldn",
    "wouldnt", "y", "you", "your", "yours", "yourself", "yourselves",
})

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
        assume_fresh: bool = False,
        deferred_index: bool = False,
        fts_fields: list[str] | None = None,
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
        self._assume_fresh = assume_fresh
        self._deferred_index = deferred_index
        self._fts_fields = fts_fields if fts_fields is not None else ["raw_content", "keywords"]

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

    # Maximum number of disjunctive term clauses to avoid query explosion.
    # Typical BEIR queries have 5-15 terms; cap protects against adversarial input.
    _MAX_DISJUNCTIVE_TERMS: int = 16

    @staticmethod
    def _normalize_fts_tokens(query: str) -> list[str]:
        """Normalize a natural-language query into clean tokens for SurrealDB FTS.

        Strips punctuation, splits hyphens, removes boolean operators
        (AND/OR/NOT/NEAR), and filters snowball(english) stop words to match
        the server-side ``hydrag_fts`` analyzer.  Returns a **list** of
        individual tokens suitable for disjunctive per-term ``@N@`` clauses.
        """
        _fts_operators = {"AND", "OR", "NOT", "NEAR"}
        clean: list[str] = []
        for raw_token in query.split():
            sub_tokens = raw_token.split("-")
            for t in sub_tokens:
                word = "".join(c for c in t if c.isalnum() or c == "_")
                if not word:
                    continue
                upper = word.upper()
                lower = word.lower()
                if upper in _fts_operators:
                    continue
                if lower in _SNOWBALL_EN_STOP_WORDS:
                    continue
                clean.append(word)
        return clean

    @staticmethod
    def _normalize_fts_query(query: str) -> str:
        """Convenience wrapper returning space-separated tokens as a string.

        Kept for backward compatibility with tests.
        """
        return " ".join(SurrealDBAdapter._normalize_fts_tokens(query))

    @classmethod
    def _build_disjunctive_fts_query(
        cls,
        tokens: list[str],
        *,
        select_fields: str = "raw_content",
        fts_fields: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Build a disjunctive (OR) FTS query with per-token score slots.

        SurrealDB ``@@`` with space-separated tokens uses **implicit AND**:
        all tokens must appear in the document.  To achieve OR-disjunctive
        recall (like SQLite FTS5 ``MATCH … OR …``), each token gets its own
        ``@N@`` score slot on each FTS field, joined by SQL-level ``OR``.

        When *fts_fields* is ``["raw_content"]`` only, keywords slots are
        skipped entirely — halving the number of BM25 evaluations.

        Score slots are allocated as:
        - field_0: slots 0 … N-1
        - field_1: slots N … 2N-1
        - …

        Returns ``(sql, params)`` for ``query_raw``.
        """
        fields = fts_fields if fts_fields is not None else ["raw_content", "keywords"]
        n = min(len(tokens), cls._MAX_DISJUNCTIVE_TERMS)
        tokens = tokens[:n]

        where_parts: list[str] = []
        score_parts: list[str] = []
        params: dict[str, Any] = {}

        for i, token in enumerate(tokens):
            for f_idx, field in enumerate(fields):
                slot = f_idx * n + i
                where_parts.append(f"{field} @{slot}@ $t{i}")
                score_parts.append(f"math::max([search::score({slot}), 0])")
            params[f"t{i}"] = token

        where_clause = " OR ".join(where_parts)
        score_expr = " + ".join(score_parts)

        sql = (
            f"SELECT {select_fields}, {score_expr} AS relevance "
            f"FROM chunks WHERE {where_clause} "
            f"ORDER BY relevance DESC LIMIT $n_results"
        )
        return sql, params

    async def _async_init_schema(self) -> None:
        # Check if a bulk_ingest sentinel exists — if so, a previous deferred-index
        # run crashed before rebuilding indexes.  Skip index DEFINE here and let
        # the next index_documents() call handle recovery.
        sentinel_rows = await self._async_query(
            "SELECT * FROM _hydrag_meta:bulk_ingest", {},
        )
        has_sentinel = bool(sentinel_rows and any(
            isinstance(r, dict) and r.get("active") for r in sentinel_rows
        ))

        schema_ddl = [
            "DEFINE TABLE IF NOT EXISTS _hydrag_meta SCHEMAFULL",
            "DEFINE FIELD IF NOT EXISTS version ON _hydrag_meta TYPE int",
            "DEFINE FIELD IF NOT EXISTS active ON _hydrag_meta TYPE option<bool>",
            "UPDATE _hydrag_meta:current SET version = 1",
            "DEFINE TABLE IF NOT EXISTS chunks SCHEMAFULL",
            "DEFINE FIELD IF NOT EXISTS chunk_id ON chunks TYPE string",
            "DEFINE FIELD IF NOT EXISTS source ON chunks TYPE string",
            "DEFINE FIELD IF NOT EXISTS title ON chunks TYPE string",
            "DEFINE FIELD IF NOT EXISTS raw_content ON chunks TYPE string",
            "DEFINE FIELD IF NOT EXISTS summary ON chunks TYPE string",
            "DEFINE FIELD IF NOT EXISTS keywords ON chunks TYPE string",
            "DEFINE FIELD IF NOT EXISTS content_hash ON chunks TYPE string",
            "DEFINE FIELD IF NOT EXISTS embedding ON chunks TYPE array<float>",
            "DEFINE FIELD IF NOT EXISTS metadata ON chunks TYPE object",
            "DEFINE ANALYZER IF NOT EXISTS hydrag_fts TOKENIZERS blank, class FILTERS lowercase, snowball(english)",
        ]

        # Only define indexes if no bulk_ingest sentinel (crash recovery guard)
        if not has_sentinel:
            schema_ddl.extend([
                (
                    "DEFINE INDEX IF NOT EXISTS chunks_fts_content ON chunks"
                    " FIELDS raw_content SEARCH ANALYZER hydrag_fts BM25"
                ),
                (
                    "DEFINE INDEX IF NOT EXISTS chunks_fts_keywords ON chunks"
                    " FIELDS keywords SEARCH ANALYZER hydrag_fts BM25"
                ),
                (
                    f"DEFINE INDEX IF NOT EXISTS chunks_vec ON chunks"
                    f" FIELDS embedding HNSW DIMENSION {self._embedding_dim} DIST COSINE"
                ),
                "DEFINE INDEX IF NOT EXISTS chunks_hash_idx ON chunks FIELDS content_hash UNIQUE",
                "DEFINE INDEX IF NOT EXISTS chunks_cid_idx ON chunks FIELDS chunk_id UNIQUE",
            ])
        else:
            log.warning(
                "bulk_ingest sentinel detected — skipping index DEFINE "
                "(indexes will be rebuilt on next index_documents call)"
            )
            # Still define UNIQUE indexes needed for insert correctness
            schema_ddl.extend([
                "DEFINE INDEX IF NOT EXISTS chunks_hash_idx ON chunks FIELDS content_hash UNIQUE",
                "DEFINE INDEX IF NOT EXISTS chunks_cid_idx ON chunks FIELDS chunk_id UNIQUE",
            ])

        schema_ddl.extend([
            "DEFINE TABLE IF NOT EXISTS calls SCHEMALESS",
            "DEFINE TABLE IF NOT EXISTS imports SCHEMALESS",
            "DEFINE TABLE IF NOT EXISTS references SCHEMALESS",
        ])

        for stmt in schema_ddl:
            await self._async_query(stmt)

        if not has_sentinel:
            await self._async_wait_for_indexes_ready(
                ("chunks_fts_content", "chunks_fts_keywords"),
                timeout_s=self._timeout,
            )
        log.info("schema initialized (version=1, dim=%d, sentinel=%s)", self._embedding_dim, has_sentinel)

    async def _async_wait_for_indexes_ready(
        self,
        index_names: tuple[str, ...],
        timeout_s: int,
    ) -> None:
        """Wait until Surreal fulltext indexes report ready.

        If the server does not expose readiness metadata, continue without
        blocking to remain backward-compatible.
        """
        if self._db is None:
            return

        loop = asyncio.get_running_loop()

        for index_name in index_names:
            deadline = loop.time() + timeout_s
            while True:
                raw = await self._db.query_raw(  # type: ignore[union-attr]
                    f"INFO FOR INDEX {index_name} ON chunks",
                )

                result = raw.get("result", raw) if isinstance(raw, dict) else raw
                info: Any = result
                if isinstance(result, list) and result:
                    stmt = result[0]
                    if isinstance(stmt, dict):
                        info = stmt.get("result", stmt)
                    else:
                        info = stmt
                if isinstance(info, list) and info:
                    info = info[0]

                status: str | None = None
                if isinstance(info, dict):
                    building = info.get("building")
                    if isinstance(building, dict):
                        val = building.get("status")
                        if isinstance(val, str):
                            status = val
                    if status is None:
                        val = info.get("status")
                        if isinstance(val, str):
                            status = val

                if status == "ready":
                    break

                # Some Surreal variants do not expose index build state.
                # Issue a probe FTS query to verify the index is functional
                # before proceeding — avoids silent 0-hit failures on cold start.
                if status is None:
                    log.warning(
                        "index readiness metadata unavailable for %s — probing",
                        index_name,
                    )
                    probe_ok = await self._async_probe_index(index_name)
                    if probe_ok:
                        break
                    if loop.time() >= deadline:
                        raise TimeoutError(
                            f"FTS index {index_name} not responding to probe queries "
                            f"within {timeout_s}s timeout",
                        )
                    await asyncio.sleep(0.5)
                    continue

                if status not in {"indexing", "cleaning", "started"}:
                    raise RuntimeError(
                        f"Unexpected index status for {index_name}: {status!r}",
                    )

                if loop.time() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for index {index_name} to become ready",
                    )

                await asyncio.sleep(0.2)

    async def _async_probe_index(self, index_name: str) -> bool:
        """Dispatch to FTS or HNSW probe based on index type."""
        if "vec" in index_name:
            return await self._async_probe_hnsw_index()
        return await self._async_probe_fts_index(index_name)

    async def _async_probe_hnsw_index(self) -> bool:
        """Probe HNSW vector index with a dummy nearest-neighbor query."""
        probe_vec = [0.0] * self._embedding_dim
        try:
            await self._async_query(
                "SELECT id FROM chunks WHERE embedding <|1|> $vec",
                {"vec": probe_vec},
            )
            log.info("HNSW probe succeeded")
            return True
        except Exception as exc:
            log.debug("HNSW probe failed: %s", exc)
            return False

    async def _async_probe_fts_index(self, index_name: str) -> bool:
        """Probe FTS index with a known-present token to verify build completion.

        Samples a real token from stored content rather than using a static probe
        word, avoiding false-positive readiness when the static word is absent.
        Returns True only when the probe returns at least one result.
        """
        field = "raw_content" if "content" in index_name else "keywords"
        try:
            sample_rows = await self._async_query(
                f"SELECT {field} FROM chunks LIMIT 1", {},
            )
        except Exception as exc:
            log.debug("FTS probe sample failed for %s: %s", index_name, exc)
            return False
        if not sample_rows:
            log.info("FTS probe for %s: no chunks, treating as ready", index_name)
            return True
        sample_text = sample_rows[0].get(field, "")
        probe_tokens = self._normalize_fts_tokens(sample_text)
        if not probe_tokens:
            log.info("FTS probe for %s: no tokens in sample, treating as ready", index_name)
            return True
        probe_word = probe_tokens[0]
        try:
            result = await self._async_query(
                f"SELECT count() AS n FROM chunks WHERE {field} @0@ $probe LIMIT 1",
                {"probe": probe_word},
            )
            n = result[0].get("n", 0) if result else 0
            if n > 0:
                log.info("FTS probe for %s ok (token=%r, n=%d)", index_name, probe_word, n)
                return True
            log.debug("FTS probe for %s: 0 results (token=%r)", index_name, probe_word)
            return False
        except Exception as exc:
            log.debug("FTS probe for %s failed: %s", index_name, exc)
            return False

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
        tokens = self._normalize_fts_tokens(query)
        if not tokens:
            return []
        sql, params = self._build_disjunctive_fts_query(
            tokens, select_fields="raw_content", fts_fields=self._fts_fields,
        )
        params["n_results"] = n_results
        result = self._query(sql, params)
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
        # Note: <|n|> KNN/HNSW operator is non-functional in SurrealDB v2.2.1
        # (returns empty results even when index is ready and data is present).
        # Brute-force fallback: SELECT the cosine score as an alias, ORDER BY it.
        # SurrealDB v2.2.1 requires the ORDER BY expression to appear in SELECT.
        result = self._query(
            "SELECT raw_content, vector::similarity::cosine(embedding, $vec) AS _score "
            "FROM chunks ORDER BY _score DESC LIMIT $n",
            {"vec": vec, "n": int(n_results)},
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
        """Internal: keyword search returning chunk_id + source + raw_content."""
        self._check_connection()
        tokens = self._normalize_fts_tokens(query)
        if not tokens:
            return []
        sql, params = self._build_disjunctive_fts_query(
            tokens, select_fields="chunk_id, source, raw_content",
            fts_fields=self._fts_fields,
        )
        params["n_results"] = n_results
        return self._query(sql, params)

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
        # Fixed timeout: 1 hour hard cap + rebuild budget for deferred index
        timeout = max(self._timeout, 3600)
        if self._deferred_index:
            timeout += max(0, len(chunks) // 10_000 * 30 + 60)
        return self._bridge.run(
            self._async_index_documents(chunks, embeddings),
            timeout=timeout,
        )

    async def _async_drop_indexes(self, include_unique: bool = False) -> None:
        """Drop FTS and HNSW indexes (for deferred rebuild).

        Args:
            include_unique: When True, also drops the two UNIQUE B-tree indexes
                (chunks_hash_idx, chunks_cid_idx).  Use only on fresh databases
                where insert deduplication is guaranteed by the caller (e.g. the
                microbenchmark with assume_fresh=True and a single writer).
                Production default is False — UNIQUE indexes guard against
                duplicate records on reconnect after a crash.
        """
        drop_stmts = [
            "REMOVE INDEX IF EXISTS chunks_fts_content ON chunks",
            "REMOVE INDEX IF EXISTS chunks_fts_keywords ON chunks",
            "REMOVE INDEX IF EXISTS chunks_vec ON chunks",
        ]
        if include_unique:
            drop_stmts += [
                "REMOVE INDEX IF EXISTS chunks_hash_idx ON chunks",
                "REMOVE INDEX IF EXISTS chunks_cid_idx ON chunks",
            ]
        for stmt in drop_stmts:
            await self._async_query(stmt)
        log.info(
            "deferred_index: dropped FTS + HNSW indexes%s",
            " + UNIQUE" if include_unique else "",
        )

    async def _async_rebuild_indexes(
        self,
        total_created: int,
        include_unique: bool = False,
    ) -> None:
        """Rebuild FTS and HNSW indexes after bulk insert.

        Args:
            total_created: Number of chunks inserted; drives dynamic rebuild timeout.
            include_unique: When True, also recreates the two UNIQUE B-tree indexes
                (chunks_hash_idx, chunks_cid_idx) that were dropped by a preceding
                _async_drop_indexes(include_unique=True) call.  Must be paired
                symmetrically — drop and rebuild both use the same value.
        """
        log.info("deferred_index: rebuilding indexes over %d chunks", total_created)
        rebuild_stmts = [
            (
                "DEFINE INDEX IF NOT EXISTS chunks_fts_content ON chunks"
                " FIELDS raw_content SEARCH ANALYZER hydrag_fts BM25"
            ),
        ]
        if "keywords" in self._fts_fields:
            rebuild_stmts.append(
                "DEFINE INDEX IF NOT EXISTS chunks_fts_keywords ON chunks"
                " FIELDS keywords SEARCH ANALYZER hydrag_fts BM25"
            )
        rebuild_stmts.append(
            f"DEFINE INDEX IF NOT EXISTS chunks_vec ON chunks"
            f" FIELDS embedding HNSW DIMENSION {self._embedding_dim} DIST COSINE"
        )
        if include_unique:
            rebuild_stmts += [
                "DEFINE INDEX IF NOT EXISTS chunks_hash_idx ON chunks FIELDS content_hash UNIQUE",
                "DEFINE INDEX IF NOT EXISTS chunks_cid_idx ON chunks FIELDS chunk_id UNIQUE",
            ]
        for stmt in rebuild_stmts:
            await self._async_query(stmt)

        # Dynamic timeout: at least self._timeout, or ~30s per 10K docs
        rebuild_timeout = max(self._timeout, total_created // 10_000 * 30 + 60)
        wait_indexes: list[str] = ["chunks_fts_content"]
        if "keywords" in self._fts_fields:
            wait_indexes.append("chunks_fts_keywords")
        if self._embedding_dim > 1:
            wait_indexes.append("chunks_vec")
        await self._async_wait_for_indexes_ready(
            tuple(wait_indexes),
            timeout_s=rebuild_timeout,
        )
        # Clear sentinel
        await self._async_query(
            "UPDATE _hydrag_meta:bulk_ingest SET active = false",
        )
        log.info(
            "deferred_index: indexes ready, sentinel cleared%s",
            " (UNIQUE rebuilt)" if include_unique else "",
        )

    async def _async_index_documents(
        self,
        chunks: list[IndexedChunk],
        embeddings: list[list[float]] | None = None,
    ) -> int:
        assert self._write_lock is not None
        async with self._write_lock:
            use_deferred = self._deferred_index

            # Check for stale sentinel from a previous crashed deferred run
            if not use_deferred:
                sentinel_rows = await self._async_query(
                    "SELECT active FROM _hydrag_meta:bulk_ingest", {},
                )
                stale = any(
                    isinstance(r, dict) and r.get("active")
                    for r in sentinel_rows
                )
                if stale:
                    log.warning(
                        "stale bulk_ingest sentinel detected — "
                        "forcing deferred index rebuild path"
                    )
                    use_deferred = True

            if use_deferred:
                # Set sentinel BEFORE dropping indexes (crash recovery)
                await self._async_query(
                    "UPDATE _hydrag_meta:bulk_ingest SET active = true",
                )
                await self._async_drop_indexes()

            total_created = 0
            try:
                for batch_start in range(0, len(chunks), self._batch_size):
                    batch = chunks[batch_start: batch_start + self._batch_size]
                    batch_embs = (
                        embeddings[batch_start: batch_start + self._batch_size]
                        if embeddings
                        else [None] * len(batch)
                    )
                    try:
                        # Fill missing content_hash (128-bit: MEDIUM-002 fix)
                        for chunk in batch:
                            if not chunk.content_hash:
                                chunk.content_hash = hashlib.sha256(
                                    chunk.raw_content.encode("utf-8"),
                                ).hexdigest()[:32]

                        existing_hashes: set[str] = set()
                        if not self._assume_fresh:
                            batch_hashes = [chunk.content_hash for chunk in batch]
                            existing_rows = await self._async_query(
                                "SELECT content_hash FROM chunks "
                                "WHERE content_hash IN $hashes",
                                {"hashes": batch_hashes},
                            )
                            existing_hashes = {
                                r["content_hash"]
                                for r in existing_rows
                                if isinstance(r, dict) and "content_hash" in r
                            }

                        insert_batch: list[dict[str, Any]] = []
                        batch_seen: set[str] = set()
                        for chunk, emb in zip(batch, batch_embs):
                            if chunk.content_hash in existing_hashes:
                                continue
                            if chunk.content_hash in batch_seen:
                                continue
                            batch_seen.add(chunk.content_hash)
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
                                "embedding": vec if vec else [0.0] * self._embedding_dim,
                                "metadata": {},
                            })

                        if insert_batch:
                            result = await self._async_query(
                                "INSERT INTO chunks $_data RETURN id",
                                {"_data": insert_batch},
                            )
                            total_created += len(insert_batch)
                            if batch_start == 0:
                                log.info(
                                    "first batch INSERT returned %d rows, "
                                    "sample: %s",
                                    len(result),
                                    repr(result[0])[:120] if result else "empty",
                                )
                    except Exception as exc:
                        log.warning(
                            "batch insert failed at offset %d (%d created so far): %s",
                            batch_start, total_created, exc,
                        )
                        raise
            finally:
                # Post-insert count verification (runs even on partial failure).
                # Wrapped in try/except so a transient query failure does not
                # prevent the deferred-index rebuild from running (P10-MEDIUM-001).
                try:
                    count_rows = await self._async_query(
                        "SELECT count() AS total FROM chunks GROUP ALL", {},
                    )
                    db_count = count_rows[0]["total"] if count_rows else 0
                except Exception as count_exc:
                    log.warning(
                        "post-insert count verification failed: %s", count_exc,
                    )
                    db_count = 0
                n_batches = (len(chunks) + self._batch_size - 1) // self._batch_size
                log.info(
                    "indexed %d chunks in %d batches (db total: %d)",
                    total_created, n_batches, db_count,
                )

                if use_deferred:
                    # Use max(total_created, db_count) so the rebuild timeout
                    # accounts for pre-existing rows during stale-sentinel
                    # recovery, not just the rows inserted in this call.
                    rebuild_row_estimate = max(total_created, db_count)
                    await self._async_rebuild_indexes(rebuild_row_estimate)

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
                    if _HAS_SURREAL_RECORD_ID:
                        # type::thing() fails in RELATE subject position in SurrealDB v2.2.1
                        # (Parse error: Unexpected token '::'). Use RecordID bound params instead.
                        await self._async_query(
                            f"RELATE $from_id->{edge_type}->$to_id SET id = $edge_id",
                            {
                                "from_id": _SurrealRecordID("chunks", src_id),
                                "to_id": _SurrealRecordID("chunks", tgt_id),
                                "edge_id": _SurrealRecordID(edge_type, edge_hash),
                            },
                        )
                    else:
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
        # Strip path (e.g. /rpc) so the health URL is http://host:port/health
        http_url = urlunparse(parsed_url._replace(scheme=http_scheme, path=""))
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
