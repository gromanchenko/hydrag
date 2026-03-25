"""SQLite FTS5 adapter — zero-dependency local search for HydRAG.

Implements VectorStoreAdapter using SQLite's built-in FTS5 engine.
No vectors, no GPU, no external services. Pure lexical search with
optional LLM-enriched metadata (summary + keywords) per chunk.

T-742: Core FTS5 store with indexer + search
T-743: LLM enrichment integration (summary + keyword extraction)
"""

import hashlib
import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class KeywordExtractor(Protocol):
    """Protocol for LLM-based keyword + summary generation (T-743)."""

    def extract(self, text: str) -> dict[str, Any]:
        """Return {"summary": str, "keywords": list[str]} for a chunk."""
        ...


@dataclass
class IndexedChunk:
    """A document chunk ready for FTS5 indexing."""

    chunk_id: str
    source: str
    title: str
    raw_content: str
    summary: str = ""
    keywords: str = ""
    content_hash: str = ""


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _adaptive_keyword_count(word_count: int) -> int:
    """Adaptive keyword count: ceil(word_count / 200), clamped to [5, 30]."""
    return max(5, min(30, math.ceil(word_count / 200)))


class SQLiteFTSStore:
    """SQLite FTS5-backed search store implementing VectorStoreAdapter.

    Fields indexed:
      - title: document/chunk title
      - raw_content: original text (weight 1.0)
      - summary: LLM-generated summary (weight 0.8) [T-743]
      - keywords: LLM-generated keywords (weight 1.4) [T-743]

    All four fields are searchable via FTS5. The BM25 ranking uses
    column weights to bias toward keywords (high recall via synonym
    expansion) while keeping raw_content as the primary signal.
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._init_schema()
        except Exception:
            self._conn.close()
            raise

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id   TEXT PRIMARY KEY,
                source     TEXT NOT NULL,
                title      TEXT NOT NULL DEFAULT '',
                raw_content TEXT NOT NULL,
                summary    TEXT NOT NULL DEFAULT '',
                keywords   TEXT NOT NULL DEFAULT '',
                content_hash TEXT NOT NULL DEFAULT '',
                model_id   TEXT NOT NULL DEFAULT '',
                prompt_hash TEXT NOT NULL DEFAULT '',
                enriched_at TEXT NOT NULL DEFAULT ''
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                title,
                raw_content,
                summary,
                keywords,
                content='chunks',
                content_rowid='rowid',
                tokenize='porter unicode61'
            );
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        self._conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)",
            ("schema_version", str(self.SCHEMA_VERSION)),
        )
        self._conn.commit()

    def index_documents(
        self,
        chunks: list[IndexedChunk],
        *,
        extractor: KeywordExtractor | None = None,
        model_id: str = "",
        prompt_hash: str = "",
    ) -> int:
        """Index chunks into FTS5. Returns count of newly indexed chunks.

        If extractor is provided (T-743), generates summary + keywords
        per chunk. Skips re-enrichment for unchanged content (content_hash match).
        """
        indexed = 0
        for chunk in chunks:
            chunk.content_hash = _content_hash(chunk.raw_content)

            # Skip if already indexed with same content
            existing = self._conn.execute(
                "SELECT content_hash FROM chunks WHERE chunk_id = ?",
                (chunk.chunk_id,),
            ).fetchone()
            if existing and existing["content_hash"] == chunk.content_hash:
                continue

            # T-743: LLM enrichment
            if extractor is not None and not chunk.summary and not chunk.keywords:
                try:
                    result = extractor.extract(chunk.raw_content)
                    chunk.summary = result.get("summary", "")
                    kw_list = result.get("keywords", [])
                    # Adaptive keyword count
                    word_count = len(chunk.raw_content.split())
                    target = _adaptive_keyword_count(word_count)
                    kw_list = kw_list[:target]
                    chunk.keywords = " ".join(kw_list)
                except Exception:
                    logger.warning("Enrichment failed for chunk %s", chunk.chunk_id, exc_info=True)

            # Upsert chunk
            self._conn.execute(
                """INSERT OR REPLACE INTO chunks
                   (chunk_id, source, title, raw_content, summary, keywords,
                    content_hash, model_id, prompt_hash, enriched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                (
                    chunk.chunk_id,
                    chunk.source,
                    chunk.title,
                    chunk.raw_content,
                    chunk.summary,
                    chunk.keywords,
                    chunk.content_hash,
                    model_id,
                    prompt_hash,
                ),
            )

            # Sync FTS index
            rowid = self._conn.execute(
                "SELECT rowid FROM chunks WHERE chunk_id = ?", (chunk.chunk_id,)
            ).fetchone()
            if rowid:
                rid = rowid[0]
                self._conn.execute(
                    "INSERT OR REPLACE INTO chunks_fts(rowid, title, raw_content, summary, keywords) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (rid, chunk.title, chunk.raw_content, chunk.summary, chunk.keywords),
                )
            indexed += 1

        self._conn.commit()
        return indexed

    def _fts_search(self, query: str, n_results: int = 5) -> list[str]:
        """Core FTS5 search with BM25 ranking and column weights."""
        if not query.strip():
            return []

        # Escape FTS5 special characters
        safe_query = self._escape_fts_query(query)
        if not safe_query:
            return []

        try:
            rows = self._conn.execute(
                """SELECT c.raw_content
                   FROM chunks_fts f
                   JOIN chunks c ON c.rowid = f.rowid
                   WHERE chunks_fts MATCH ?
                   ORDER BY bm25(chunks_fts, 1.0, 1.0, 0.8, 1.4)
                   LIMIT ?""",
                (safe_query, n_results),
            ).fetchall()
            return [row["raw_content"] for row in rows]
        except sqlite3.OperationalError:
            logger.debug("FTS query failed for: %s", safe_query, exc_info=True)
            return []

    @staticmethod
    def _escape_fts_query(query: str) -> str:
        """Convert a natural language query into safe FTS5 OR-joined tokens.

        Rules:
        - Keep alphanumeric characters and underscores.
        - Strip all punctuation (including periods) — FTS5 query syntax treats
          a trailing '.' as a syntax error even though the tokenizer handles it
          transparently at index time.
        - Hyphens split compound words into individual tokens so each part can
          match independently against the FTS5 index.
        - Drop FTS5 reserved operators to avoid ambiguous query syntax.
        """
        fts_operators = {"AND", "OR", "NOT", "NEAR"}
        clean: list[str] = []
        for raw_token in query.split():
            # Split on hyphens so "0-dimensional" → ["0", "dimensional"]
            sub_tokens = raw_token.split("-")
            for t in sub_tokens:
                word = "".join(c for c in t if c.isalnum() or c == "_")
                if word and word.upper() not in fts_operators:
                    clean.append(word)
        if not clean:
            return ""
        return " OR ".join(clean)

    # --- VectorStoreAdapter protocol ---

    def semantic_search(self, query: str, n_results: int = 5) -> list[str]:
        """FTS5 search — no vectors, purely lexical."""
        return self._fts_search(query, n_results)

    def keyword_search(self, query: str, n_results: int = 5) -> list[str]:
        """FTS5 search — identical to semantic for this store."""
        return self._fts_search(query, n_results)

    def hybrid_search(self, query: str, n_results: int = 5) -> list[str]:
        """FTS5 search — single search path, no hybrid distinction needed."""
        return self._fts_search(query, n_results)

    # --- Utility ---

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return dict(row) if row else None

    def stats(self) -> dict[str, Any]:
        total = self.count()
        enriched = self._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE keywords != ''"
        ).fetchone()
        return {
            "total_chunks": total,
            "enriched_chunks": enriched[0] if enriched else 0,
            "db_path": self._db_path,
        }

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SQLiteFTSStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
