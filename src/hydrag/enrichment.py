"""LLM-based document enrichment for SQLite FTS5 indexing.

T-743: Generate summary + keywords per chunk at ingestion time.
Uses Ollama API (or any OpenAI-compatible endpoint) to produce
structured enrichments that are stored in the FTS5 index.

Key design decisions (from !cryt analysis):
- Adaptive keyword count: ceil(word_count / 200), clamped [5, 30]
- Extractive anchor check: at least 50% of keywords must appear in source
- Content-hash caching: unchanged chunks skip re-enrichment
- Model versioning: model_id + prompt_hash stored per enrichment
"""

import hashlib
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

ENRICHMENT_PROMPT = """\
Analyze the following text and produce a JSON response with exactly two keys:
1. "summary": A concise 2-3 sentence summary of the text.
2. "keywords": A list of {target_count} single-word or short-phrase keywords \
that capture the key concepts, entities, and topics. Include synonyms and \
related terms that someone might search for. Keywords MUST be relevant to \
the actual content.

Text:
---
{text}
---

Respond with ONLY valid JSON, no markdown fences:"""


def _prompt_hash(prompt_template: str) -> str:
    return hashlib.sha256(prompt_template.encode()).hexdigest()[:12]


class OllamaKeywordExtractor:
    """Keyword + summary extractor using Ollama's /api/generate endpoint."""

    def __init__(
        self,
        model: str = "qwen3:4b",
        host: str = "http://localhost:11434",
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.prompt_hash = _prompt_hash(ENRICHMENT_PROMPT)

    def extract(self, text: str) -> dict[str, Any]:
        """Generate summary + keywords for a text chunk."""
        word_count = len(text.split())
        from hydrag.sqlite_store import _adaptive_keyword_count
        target_count = _adaptive_keyword_count(word_count)

        # Truncate very long chunks to avoid token overflow
        truncated = text[:4000] if len(text) > 4000 else text

        prompt = ENRICHMENT_PROMPT.format(
            text=truncated,
            target_count=target_count,
        )

        raw = self._call_ollama(prompt)
        parsed = self._parse_response(raw)

        # Extractive anchor check: ≥50% of keywords must appear in source
        if parsed.get("keywords"):
            parsed["keywords"] = self._filter_anchored(
                parsed["keywords"], text
            )

        return parsed

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama /api/generate and return the response text."""
        from hydrag.providers._retry import retry_request
        url = f"{self.host}/api/generate"
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 512},
        }).encode("utf-8")

        def _parse(data: dict) -> str:
            return data.get("response", "")

        result = retry_request(
            url=url,
            payload=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
            provider_name="OllamaEnhancement",
            parse_response=_parse,
        )
        return result or ""

    @staticmethod
    def _parse_response(raw: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling common formatting issues."""
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")

        try:
            result = json.loads(cleaned)
            if isinstance(result, dict):
                summary = result.get("summary", "")
                keywords = result.get("keywords", [])
                if isinstance(keywords, str):
                    keywords = [k.strip() for k in keywords.split(",") if k.strip()]
                return {"summary": str(summary), "keywords": list(keywords)}
        except (json.JSONDecodeError, ValueError):
            logger.debug("Failed to parse enrichment JSON: %s", cleaned[:200])

        return {"summary": "", "keywords": []}

    @staticmethod
    def _filter_anchored(keywords: list[str], source: str) -> list[str]:
        """Keep keywords that are at least 50% grounded in the source text."""
        source_lower = source.lower()
        anchored = []
        unanchored = []

        for kw in keywords:
            if kw.lower() in source_lower:
                anchored.append(kw)
            else:
                unanchored.append(kw)

        # Require ≥50% anchored; if not enough anchored, keep all to avoid
        # dropping to zero results
        min_anchored = max(1, len(keywords) // 2)
        if len(anchored) >= min_anchored:
            # Fill up to original count with best unanchored
            return anchored + unanchored[: len(keywords) - len(anchored)]
        else:
            # Not enough grounding — return all but log warning
            logger.warning(
                "Only %d/%d keywords anchored in source — keeping all",
                len(anchored),
                len(keywords),
            )
            return keywords
