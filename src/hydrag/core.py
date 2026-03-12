"""HydRAG core orchestrator — multi-headed retrieval with CRAG supervision."""

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .config import HydRAGConfig
from .fusion import CRAGVerdict, RetrievalResult, _as_results, _rrf_fuse, _text_of
from .protocols import LLMProvider, StreamingLLMProvider, VectorStoreAdapter
from .sanitize import _sanitize_web_content

logger = logging.getLogger("hydrag")


# ── Default LLM provider (Ollama) ────────────────────────────────


class OllamaProvider:
    """Ollama /api/generate provider with 3-attempt retry."""

    def __init__(self, host: str = "http://localhost:11434") -> None:
        self._host = host

    def generate(self, prompt: str, model: str = "", timeout: int = 30) -> str | None:
        effective_model = model or "llama3.2:latest"
        payload = json.dumps(
            {"model": effective_model, "prompt": prompt, "stream": False}
        ).encode()
        headers = {"Content-Type": "application/json"}
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    f"{self._host}/api/generate",
                    data=payload,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode())
                    return str(data.get("response", "")) or None
            except (urllib.error.URLError, TimeoutError, OSError, ValueError):
                if attempt < 2:
                    time.sleep(0.5)
        return None

    def generate_stream(
        self, prompt: str, model: str = "", timeout: int = 30,
    ) -> str | None:
        """Streaming generate — returns full response but reads line-by-line.

        This enables early termination once the verdict token is detected
        by the caller (via the streaming CRAG supervisor).
        """
        effective_model = model or "llama3.2:latest"
        payload = json.dumps(
            {"model": effective_model, "prompt": prompt, "stream": True}
        ).encode()
        headers = {"Content-Type": "application/json"}
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    f"{self._host}/api/generate",
                    data=payload,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    tokens: list[str] = []
                    for line in resp:
                        chunk = json.loads(line.decode())
                        token = chunk.get("response", "")
                        tokens.append(token)
                        accumulated = "".join(tokens).strip().upper()
                        if "SUFFICIENT" in accumulated or "INSUFFICIENT" in accumulated:
                            return "".join(tokens)
                        if chunk.get("done"):
                            break
                    return "".join(tokens)
            except (urllib.error.URLError, TimeoutError, OSError, ValueError):
                if attempt < 2:
                    time.sleep(0.5)
        return None


# ── CRAG prompt ──────────────────────────────────────────────────

_CRAG_PROMPT = """\
You are a retrieval quality evaluator.  Given a USER QUERY and RETRIEVED CONTEXT, \
decide whether the context is sufficient to answer the query.

Rules:
- Output EXACTLY one token: SUFFICIENT or INSUFFICIENT.
- SUFFICIENT = the context contains enough information to fully answer the query.
- INSUFFICIENT = the context is missing, irrelevant, or incomplete.

USER QUERY:
{query}

RETRIEVED CONTEXT (top chunks):
{context}

Verdict:"""


# ── Symbol detection (code profile only) ─────────────────────────


def _extract_symbol_hints(query: str) -> list[str]:
    """Extract likely code symbol references from a natural-language query.

    Catches: CamelCase, snake_case, dotted.paths, backtick-wrapped identifiers.
    Only used when ``profile="code"``.
    """
    hints: list[str] = []
    hints.extend(re.findall(r"`([^`]+)`", query))
    hints.extend(re.findall(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", query))
    hints.extend(re.findall(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b", query))
    hints.extend(re.findall(r"\b([a-zA-Z_]\w*(?:\.\w+)+)\b", query))
    return list(dict.fromkeys(hints))  # dedupe preserving order


# ── CRAG Supervisor ──────────────────────────────────────────────


def crag_supervisor(
    query: str,
    context_chunks: list[Any],
    llm: LLMProvider,
    config: HydRAGConfig | None = None,
) -> CRAGVerdict:
    """LLM-graded Corrective RAG supervisor.

    Calls a small/fast local model to produce a binary
    SUFFICIENT / INSUFFICIENT judgment on the retrieved context.

    When ``crag_mode="auto"`` (default) and a trained ONNX classifier
    exists at ``crag_classifier_path``, uses the classifier (~5-15ms)
    instead of the LLM (~7-14s). Falls back to LLM if classifier fails.

    When ``crag_stream=True`` (default) and the LLM provider supports
    ``generate_stream()``, uses streaming to detect the verdict token
    early, cutting latency by ~50%.

    Falls back to ``sufficient=True`` (pass-through) on model errors
    to avoid blocking retrieval when inference is unavailable.
    """
    cfg = config or HydRAGConfig()

    # ── Classifier path (crag_mode=auto or classifier) ──
    if cfg.crag_mode in ("auto", "classifier"):
        classifier_path = cfg.crag_classifier_path or str(
            Path.home() / ".hydrag" / "models" / "crag_classifier"
        )
        onnx_file = Path(classifier_path) / "model.onnx"
        if onnx_file.exists():
            try:
                from .tune import get_classifier
                classifier = get_classifier(classifier_path)
                context_text = "\n---\n".join(
                    _text_of(c)[: cfg.crag_char_limit]
                    for c in context_chunks[: cfg.crag_context_chunks]
                )
                return classifier.predict(query, context_text)
            except Exception as exc:
                if cfg.crag_mode == "classifier":
                    logger.error("CRAG classifier failed: %s", exc)
                    return CRAGVerdict(
                        sufficient=True, reason="classifier_error", latency_ms=0.0,
                    )
                logger.warning("CRAG classifier failed, falling back to LLM: %s", exc)
        elif cfg.crag_mode == "classifier":
            logger.error("CRAG classifier not found at %s", onnx_file)
            return CRAGVerdict(
                sufficient=True, reason="classifier_not_found", latency_ms=0.0,
            )

    # ── LLM path (crag_mode=auto fallback or crag_mode=llm) ──
    context_text = "\n---\n".join(
        _text_of(c)[: cfg.crag_char_limit] for c in context_chunks[: cfg.crag_context_chunks]
    )
    prompt = _CRAG_PROMPT.format(query=query, context=context_text)

    start = time.monotonic()
    # Use streaming when available for early verdict detection
    if cfg.crag_stream and isinstance(llm, StreamingLLMProvider):
        verdict_raw = llm.generate_stream(prompt, model=cfg.crag_model, timeout=cfg.crag_timeout)
    else:
        verdict_raw = llm.generate(prompt, model=cfg.crag_model, timeout=cfg.crag_timeout)
    latency_ms = (time.monotonic() - start) * 1000

    if verdict_raw is None:
        logger.warning("CRAG supervisor unavailable — falling back to SUFFICIENT")
        return CRAGVerdict(
            sufficient=True, reason="model_unreachable", latency_ms=latency_ms
        )

    normalized = verdict_raw.strip().upper()
    if "INSUFFICIENT" in normalized:
        return CRAGVerdict(
            sufficient=False,
            reason="model_verdict",
            latency_ms=latency_ms,
            raw_response=verdict_raw,
        )
    if "SUFFICIENT" in normalized:
        return CRAGVerdict(
            sufficient=True,
            reason="model_verdict",
            latency_ms=latency_ms,
            raw_response=verdict_raw,
        )

    # Ambiguous response — fail-open
    logger.warning(
        "CRAG supervisor returned ambiguous verdict: %r — treating as SUFFICIENT",
        verdict_raw[:100],
    )
    return CRAGVerdict(
        sufficient=True,
        reason="ambiguous_response",
        latency_ms=latency_ms,
        raw_response=verdict_raw,
    )


# ── Semantic Fallback ────────────────────────────────────────────


def semantic_fallback(
    adapter: VectorStoreAdapter,
    query: str,
    primary_hits: list[Any],
    n_results: int = 5,
    config: HydRAGConfig | None = None,
) -> list[RetrievalResult]:
    """Fallback retrieval: rewrite query, run CRAG + keyword + graph in parallel, fuse via RRF.

    Triggered when CRAG supervisor returns INSUFFICIENT.
    Merges fallback results with the original primary hits.
    """
    cfg = config or HydRAGConfig()

    # Rewrite query for broader coverage (optional adapter method)
    rewrite_fn = getattr(adapter, "rewrite_query", None)
    rewritten = rewrite_fn(query) if rewrite_fn else query

    pool_size = max(n_results, 8)

    def _safe_crag() -> list[Any]:
        crag_fn = getattr(adapter, "crag_search", None)
        if crag_fn:
            return crag_fn(rewritten, n_results=pool_size)  # type: ignore[no-any-return]
        return adapter.hybrid_search(rewritten, n_results=pool_size)

    def _safe_graph() -> list[Any]:
        graph_fn = getattr(adapter, "graph_search", None)
        if graph_fn:
            try:
                return graph_fn(rewritten, n_results=max(n_results, 6))  # type: ignore[no-any-return]
            except Exception:
                pass
        return []

    # Parallel retrieval — all three paths are independent
    timeout = cfg.fallback_timeout_s if cfg else 30.0
    crag_hits: list[Any] = []
    keyword_hits: list[Any] = []
    graph_hits: list[Any] = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        crag_future = pool.submit(_safe_crag)
        kw_future = pool.submit(adapter.keyword_search, rewritten, pool_size)
        graph_future = pool.submit(_safe_graph)
        try:
            crag_hits = crag_future.result(timeout=timeout)
        except Exception:
            logger.warning("semantic_fallback: CRAG branch timed out / failed")
            crag_future.cancel()
        try:
            keyword_hits = kw_future.result(timeout=timeout)
        except Exception:
            logger.warning("semantic_fallback: keyword branch timed out / failed")
            kw_future.cancel()
        try:
            graph_hits = graph_future.result(timeout=timeout)
        except Exception:
            logger.warning("semantic_fallback: graph branch timed out / failed")
            graph_future.cancel()

    return _rrf_fuse(
        [
            (primary_hits, 0.5),
            (crag_hits, 1.2),
            (keyword_hits, 1.0),
            (graph_hits, 0.9),
        ],
        k=cfg.rrf_k,
        n_results=n_results,
        head_origin="semantic_fallback",
    )


# ── Web Fallback ─────────────────────────────────────────────────


def web_fallback(
    query: str,
    limit: int = 3,
    config: HydRAGConfig | None = None,
) -> list[RetrievalResult]:
    """Retrieve external context via Firecrawl web search.

    Returns empty list if ``firecrawl-py`` is not installed or
    ``FIRECRAWL_API_KEY`` is not set.
    """
    cfg = config or HydRAGConfig()
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        logger.debug("web_fallback skipped: FIRECRAWL_API_KEY not set")
        return []

    try:
        from firecrawl import FirecrawlApp  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("web_fallback skipped: firecrawl-py not installed")
        return []

    try:
        app = FirecrawlApp(api_key=api_key)
        results = app.search(
            query,
            params={
                "pageOptions": {"fetchPageContent": True},
                "searchOptions": {"limit": limit},
            },
        )
        if not results or "data" not in results:
            return []

        chunks: list[str] = []
        for item in results.get("data", []):
            title = item.get("title", "")
            url = item.get("url", "")
            md = item.get("markdown", "") or item.get("content", "") or ""
            sanitized = _sanitize_web_content(
                md,
                max_chars=cfg.web_chunk_limit,
                allow_markdown=cfg.allow_markdown_in_web_fallback,
            )
            chunk = f"[Web: {_sanitize_web_content(title, max_chars=200)}]({url})\n{sanitized}"
            chunks.append(chunk)
        return _as_results(chunks, head_origin="web", trust_level="web")

    except Exception as exc:
        logger.warning("web_fallback error: %s", exc)
        return []


# ── HydRAG Orchestrator ─────────────────────────────────────────


def hydrag_search(
    adapter: VectorStoreAdapter,
    query: str,
    n_results: int = 5,
    enable_web_fallback: bool | None = None,
    config: HydRAGConfig | None = None,
    llm: LLMProvider | None = None,
    heads: set[str] | None = None,
    disable_heads: set[str] | None = None,
) -> list[RetrievalResult]:
    """HydRAG — Multi-Headed Retrieval pipeline.

    Head 0: BM25 fast path (keyword-only short-circuit).
    Head 1: Primary retrieval (hybrid by default; code-aware when
            ``profile="code"`` and symbol hints are detected).
    Head 2: CRAG Supervisor — LLM judges context sufficiency.
    Head 3a: Semantic fallback (on INSUFFICIENT).
    Head 3b: Web fallback via Firecrawl (on INSUFFICIENT, if enabled).

    Fuses all heads via Reciprocal Rank Fusion for the final ranked set.

    Parameters
    ----------
    adapter : VectorStoreAdapter
        Pluggable vector store implementing the required protocol methods.
    query : str
        User query.
    n_results : int
        Number of results to return.
    enable_web_fallback : bool | None
        Override config's ``enable_web_fallback``. ``None`` → use config.
    config : HydRAGConfig | None
        Pipeline configuration. Defaults to ``HydRAGConfig()`` (prose profile).
    llm : LLMProvider | None
        LLM provider for CRAG. Defaults to ``OllamaProvider``.
    heads : set[str] | None
        Whitelist of heads to enable (e.g. ``{"head_0", "head_1"}``).
        When set, only listed heads run. Overrides config-level switches.
        Valid names: ``head_0``, ``head_1``, ``head_2_crag``,
        ``head_3a_semantic``, ``head_3b_web``.
    disable_heads : set[str] | None
        Blacklist of heads to disable (e.g. ``{"head_2_crag"}``).
        Applied after config-level switches. Cannot be used with ``heads``.
    """
    cfg = config or HydRAGConfig()
    if heads is not None and disable_heads is not None:
        raise ValueError("Cannot specify both 'heads' and 'disable_heads' — use one or the other.")
    llm_provider = llm or OllamaProvider(host=cfg.ollama_host)
    weights = cfg.rrf_head_weights
    web_enabled = enable_web_fallback if enable_web_fallback is not None else cfg.enable_web_fallback

    # ── Resolve effective head set ──
    def _head_enabled(name: str, cfg_default: bool) -> bool:
        if heads is not None:
            return name in heads
        if disable_heads is not None and name in disable_heads:
            return False
        return cfg_default

    h0_on = _head_enabled("head_0", cfg.enable_head_0 and cfg.enable_fast_path)
    h1_on = _head_enabled("head_1", cfg.enable_head_1)
    h2_on = _head_enabled("head_2_crag", cfg.enable_head_2_crag)
    h3a_on = _head_enabled("head_3a_semantic", cfg.enable_head_3a_semantic)
    h3b_on = _head_enabled("head_3b_web", cfg.enable_head_3b_web or web_enabled)

    # ── Head 0: BM25 fast path (V2.2 §3.0) ──
    if h0_on and hasattr(adapter, "keyword_search"):
        fp_hits = adapter.keyword_search(query, n_results=n_results)
        if fp_hits:
            hit_ratio = len(fp_hits) / max(n_results, 1)
            if hit_ratio >= cfg.fast_path_bm25_threshold:
                # Confidence gate: high-ratio BM25 skips CRAG entirely
                if hit_ratio >= cfg.fast_path_confidence_threshold:
                    results = _as_results(fp_hits[:n_results], head_origin="head_0")
                    for r in results:
                        r.metadata["fast_path_triggered"] = True
                        r.metadata["crag_skipped"] = True
                    return results
                # Below confidence: fast-path hit but run CRAG to verify (if head 2 enabled)
                results = _as_results(fp_hits[:n_results], head_origin="head_0")
                for r in results:
                    r.metadata["fast_path_triggered"] = True
                if h2_on:
                    verdict = crag_supervisor(query, fp_hits, llm=llm_provider, config=cfg)
                    logger.info(
                        "HydRAG: fast-path CRAG gate verdict=%s (ratio=%.2f, conf_threshold=%.2f)",
                        "SUFFICIENT" if verdict.sufficient else "INSUFFICIENT",
                        hit_ratio, cfg.fast_path_confidence_threshold,
                    )
                    if verdict.sufficient:
                        return results
                else:
                    # CRAG disabled — accept fast-path results as-is
                    return results

    if not h1_on:
        # Head 1 disabled — return empty (no primary retrieval)
        return []

    # ── Head 1: Primary retrieval ──
    if cfg.profile == "code":
        symbol_hints = _extract_symbol_hints(query)
        if symbol_hints:
            head_1_key = "head_1a"
            sem_hits = adapter.semantic_search(
                query, n_results=max(n_results, cfg.min_candidate_pool)
            )
            sym_query = " ".join(symbol_hints)
            kw_hits = adapter.keyword_search(
                sym_query, n_results=max(n_results, cfg.min_candidate_pool)
            )
            primary_rr = _rrf_fuse(
                [(sem_hits, 1.5), (kw_hits, 1.0)],
                k=cfg.rrf_k,
                n_results=max(n_results, cfg.min_candidate_pool),
                head_origin="head_1a",
            )
            primary_hits: list[Any] = [r.text for r in primary_rr]
            primary_strategy = "code-aware"
        else:
            head_1_key = "head_1b"
            primary_hits = adapter.hybrid_search(
                query, n_results=max(n_results, cfg.min_candidate_pool)
            )
            primary_strategy = "hybrid"
    else:
        # prose (default): always hybrid, no symbol detection
        head_1_key = "head_1b"
        primary_hits = adapter.hybrid_search(
            query, n_results=max(n_results, cfg.min_candidate_pool)
        )
        primary_strategy = "hybrid"

    if not primary_hits:
        if cfg.allow_web_on_empty_primary and web_enabled:
            empty_web = web_fallback(query, limit=3, config=cfg)
            if empty_web:
                logger.info(
                    "HydRAG: primary empty, web fallback returned %d hits",
                    len(empty_web),
                )
                return empty_web[:n_results]
        logger.info("HydRAG: primary retrieval returned 0 hits")
        return []

    # ── Head 2: CRAG Supervisor ──
    if h2_on:
        verdict = crag_supervisor(query, primary_hits, llm=llm_provider, config=cfg)
        logger.info(
            "HydRAG: CRAG verdict=%s reason=%s (primary=%s, %d hits, %.0fms)",
            "SUFFICIENT" if verdict.sufficient else "INSUFFICIENT",
            verdict.reason,
            primary_strategy,
            len(primary_hits),
            verdict.latency_ms,
        )
    else:
        # CRAG disabled — treat primary as sufficient
        verdict = CRAGVerdict(sufficient=True, reason="head_disabled", latency_ms=0.0)

    if verdict.sufficient:
        results = _as_results(primary_hits[:n_results], head_origin=head_1_key)
        if verdict.reason != "model_verdict" and verdict.reason != "head_disabled":
            for r in results:
                r.metadata["crag_warning"] = verdict.reason
        return results

    # ── Head 3a: Semantic fallback + Head 3b: Web fallback (parallel) ──
    fallback_hits: list[Any] = []
    web_res: list[Any] = []
    fallback_future = None
    web_future = None
    with ThreadPoolExecutor(max_workers=2) as pool:
        if h3a_on:
            fallback_future = pool.submit(
                semantic_fallback, adapter, query, primary_hits,
                max(n_results, cfg.min_candidate_pool), cfg,
            )
        if h3b_on:
            web_future = pool.submit(web_fallback, query, 3, cfg)
        try:
            fallback_hits = fallback_future.result(timeout=cfg.fallback_timeout_s) if fallback_future else []
        except Exception:
            logger.warning("hydrag_search: semantic fallback timed out / failed")
            if fallback_future:
                fallback_future.cancel()
        try:
            web_res = web_future.result(timeout=cfg.fallback_timeout_s) if web_future else []
        except Exception:
            logger.warning("hydrag_search: web fallback timed out / failed")
            if web_future:
                web_future.cancel()

    # ── Final RRF fusion across all heads ──
    return _rrf_fuse(
        [
            (primary_hits, weights.get(head_1_key, 0.6)),
            (fallback_hits, weights.get("head_3a", 1.3)),
            (web_res, weights.get("head_3b", 0.8)),
        ],
        k=cfg.rrf_k,
        n_results=n_results,
        head_origin="hydrag",
    )


# ── HydRAG class wrapper (V1 spec §15.1) ────────────────────────


class HydRAG:
    """Convenience wrapper holding adapter + config + LLM provider.

    Usage::

        store = MyChromaAdapter(collection)
        engine = HydRAG(store)
        results = engine.search("How do I configure logging?")
    """

    def __init__(
        self,
        adapter: VectorStoreAdapter,
        config: HydRAGConfig | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self._adapter = adapter
        self._config = config or HydRAGConfig()
        self._llm = llm

    @property
    def config(self) -> HydRAGConfig:
        return self._config

    def search(
        self,
        query: str,
        n_results: int = 5,
        enable_web_fallback: bool | None = None,
        heads: set[str] | None = None,
        disable_heads: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """Run the full HydRAG pipeline and return ranked results."""
        return hydrag_search(
            self._adapter,
            query,
            n_results=n_results,
            enable_web_fallback=enable_web_fallback,
            config=self._config,
            llm=self._llm,
            heads=heads,
            disable_heads=disable_heads,
        )
