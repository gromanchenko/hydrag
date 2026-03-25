"""Typed retrieval containers and Reciprocal Rank Fusion engine."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CRAGVerdict:
    """Structured CRAG supervisor verdict."""

    sufficient: bool
    reason: str
    latency_ms: float
    raw_response: str | None = None


@dataclass
class RetrievalResult:
    """Typed retrieval result with provenance and score."""

    text: str
    source: str
    score: float
    head_origin: str  # "head_1a" | "head_1b" | "head_3a" | "head_3b" | "hydrag"
    trust_level: str  # "local" | "web"
    metadata: dict[str, Any] = field(default_factory=dict)
    crag_verdict: Optional[CRAGVerdict] = None


def _text_of(item: Any) -> str:
    """Extract plain text from a string or RetrievalResult."""
    return item.text if isinstance(item, RetrievalResult) else item


def _as_results(
    docs: list[Any],
    *,
    head_origin: str,
    trust_level: str = "local",
) -> list[RetrievalResult]:
    """Wrap raw strings into RetrievalResult with positional score."""
    n = max(len(docs), 1)
    out: list[RetrievalResult] = []
    for i, d in enumerate(docs):
        if isinstance(d, RetrievalResult):
            out.append(d)
        else:
            out.append(
                RetrievalResult(
                    text=d,
                    source="",
                    score=1.0 - i / n,
                    head_origin=head_origin,
                    trust_level=trust_level,
                )
            )
    return out


def rrf_fuse(
    sources: list[tuple[list[Any], float]],
    *,
    k: int = 60,
    n_results: int = 5,
    head_origin: str = "",
    trust_level: str | None = None,
) -> list[RetrievalResult]:
    """Reciprocal Rank Fusion across multiple ranked source lists.

    Each entry in *sources* is ``(ranked_docs, weight)``.
    Score: ``sum_s(weight_s / (k + rank_s(d)))``.
    Tie-break: first-seen source order, then stable doc identity.

    Items may be plain strings or :class:`RetrievalResult` — both accepted.
    Returns :class:`RetrievalResult` with fused RRF scores.
    """
    fused: dict[str, float] = {}
    insertion_order: dict[str, int] = {}
    first_seen: dict[str, RetrievalResult | None] = {}  # preserve provenance from first-seen object
    counter = 0
    for ranked_docs, weight in sources:
        for rank, doc in enumerate(ranked_docs, start=1):
            text = _text_of(doc)
            fused[text] = fused.get(text, 0.0) + weight / (k + rank)
            if text not in insertion_order:
                insertion_order[text] = counter
                first_seen[text] = doc if isinstance(doc, RetrievalResult) else None
                counter += 1
    if not fused:
        return []
    ranked = sorted(
        fused.items(),
        key=lambda kv: (-kv[1], insertion_order.get(kv[0], 0)),
    )
    results: list[RetrievalResult] = []
    for doc_text, score in ranked[:n_results]:
        original = first_seen.get(doc_text)
        effective_trust = trust_level if trust_level is not None else (
            original.trust_level if original is not None else "local"
        )
        if original is not None:
            results.append(
                RetrievalResult(
                    text=doc_text,
                    source=original.source,
                    score=score,
                    head_origin=head_origin or original.head_origin,
                    trust_level=effective_trust,
                    metadata={**original.metadata},
                    crag_verdict=original.crag_verdict,
                )
            )
        else:
            results.append(
                RetrievalResult(
                    text=doc_text,
                    source="",
                    score=score,
                    head_origin=head_origin,
                    trust_level=effective_trust,
                )
            )
    return results
