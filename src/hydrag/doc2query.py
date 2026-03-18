"""Doc2Query V2 — adaptive synthetic question generation per chunk.

Produces synthetic developer questions for each document chunk via LLM
(ollama by default). Supports adaptive n_questions based on chunk complexity
(RFC §2.3 lookup table) and smart boundary-aware truncation (RFC §2.4).

T-740: Adaptive n_questions and smart_truncate integrated from hydrag-benchmark.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger("hydrag.doc2query")

_PROMPT_TEMPLATE = """\
You are a developer documentation expert. Given the following code/documentation \
chunk, generate exactly {n} questions that a software developer would ask and \
that this chunk completely answers. Output only the questions, one per line.

Rules:
- Questions must be in natural language (not code).
- Questions must be answerable SOLELY from this chunk.
- Include at least one troubleshooting-style question ("Why does X fail when...").
- If the chunk describes system interactions, include an architectural question \
("How does X interact with Y?").
- Do NOT reference line numbers or chunk metadata.

CHUNK:
{chunk_text}

QUESTIONS:
"""

# RFC §2.3 — adaptive n_questions lookup table (token_threshold, max_n).
_ADAPTIVE_N_TABLE: list[tuple[int, int]] = [
    (50, 2),
    (200, 3),
    (500, 5),
]
_ADAPTIVE_N_LARGE = 7  # >500 tokens


def compute_adaptive_n(chunk_text: str, max_questions: int) -> int:
    """Compute question count based on chunk token length (RFC §2.3 lookup table).

    | Chunk tokens | Questions |
    |------------- |-----------|
    | < 50         | 1–2       |
    | 50–200       | 3         |
    | 200–500      | 4–5       |
    | > 500        | 5–7       |

    Result is capped at *max_questions* and floored at 1.
    """
    token_count = len(re.findall(r"\w+", chunk_text))
    for threshold, n in _ADAPTIVE_N_TABLE:
        if token_count < threshold:
            return max(1, min(n, max_questions))
    return max(1, min(_ADAPTIVE_N_LARGE, max_questions))


def smart_truncate(
    text: str,
    max_chars: int = 4000,
    overlap: int = 200,
) -> str:
    """Truncate at paragraph/sentence boundary, keep tail overlap (RFC §2.4).

    1. If ``text`` fits in *max_chars*, return as-is.
    2. Find the last ``\\n\\n`` (paragraph break) before *max_chars*.
    3. Fall back to the last ``". "`` (sentence end) before *max_chars*.
    4. Hard-cut at *max_chars* if no suitable boundary found.
    5. Append a tail snippet of *overlap* chars so the LLM sees conclusions.
    """
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n\n", 0, max_chars)
    if cut < max_chars // 2:
        dot_pos = text.rfind(". ", 0, max_chars)
        cut = dot_pos + 1 if dot_pos >= max_chars // 2 else -1
    if cut < max_chars // 2:
        cut = max_chars
    tail = text[-overlap:] if overlap > 0 and len(text) > max_chars + overlap else ""
    return text[:cut] + ("\n[...]\n" + tail if tail else "")


@dataclass
class Doc2QueryConfig:
    """Configuration for the Doc2Query LLM client."""

    model: str = "qwen3:4b"
    api_url: str = "http://localhost:11434"
    n_questions: int = 5
    timeout_s: float = 30.0
    max_retries: int = 2
    retry_backoff_s: float = 1.5
    num_predict: int = 256
    batch_size: int = 4
    custom_prompt: str = ""
    adaptive_n: bool = False
    max_questions_per_chunk: int = 12

    def config_fingerprint(self) -> str:
        """Hash of config params that affect question generation output."""
        key = (
            f"{self.model}:{self.n_questions}:{self.custom_prompt}"
            f":{self.adaptive_n}:{self.max_questions_per_chunk}"
        )
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


class Doc2QueryGenerator:
    """Generate synthetic questions from text chunks via an LLM API."""

    def __init__(self, config: Doc2QueryConfig | None = None) -> None:
        self._config = config or Doc2QueryConfig()

    @property
    def config_fingerprint(self) -> str:
        """Fingerprint of config params that affect generation output."""
        return self._config.config_fingerprint()

    def generate(self, chunk_text: str) -> list[str]:
        """Generate n questions for a single chunk. Returns list of question strings."""
        if self._config.adaptive_n:
            n = compute_adaptive_n(chunk_text, self._config.max_questions_per_chunk)
        else:
            n = self._config.n_questions
        prompt = self._build_prompt(chunk_text, n)
        try:
            response_text = self._call_llm(prompt, n)
        except Exception:
            logger.warning("LLM call failed for chunk (len=%d)", len(chunk_text))
            raise

        return self._parse_questions(response_text)

    def _build_prompt(self, chunk_text: str, n: int | None = None) -> str:
        """Build the LLM prompt, prepending custom_prompt if configured."""
        effective_n = n if n is not None else self._config.n_questions
        base = _PROMPT_TEMPLATE.format(
            n=effective_n,
            chunk_text=smart_truncate(chunk_text),
        )
        if self._config.custom_prompt:
            return f"{self._config.custom_prompt}\n\n{base}"
        return base

    def _call_llm(self, prompt: str, n_questions: int | None = None) -> str:
        """Call ollama /api/generate endpoint."""
        actual_n = n_questions if n_questions is not None else self._config.n_questions
        url = f"{self._config.api_url}/api/generate"
        payload = json.dumps({
            "model": self._config.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max(self._config.num_predict, actual_n * 50)},
        }).encode("utf-8")

        attempts = self._config.max_retries + 1
        for attempt in range(1, attempts + 1):
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self._config.timeout_s) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return body.get("response", "")
            except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError) as exc:
                if attempt >= attempts:
                    raise
                sleep_s = self._config.retry_backoff_s * attempt
                logger.warning(
                    "Doc2Query call attempt %d/%d failed: %s; retrying in %.1fs",
                    attempt,
                    attempts,
                    exc,
                    sleep_s,
                )
                time.sleep(sleep_s)

        # Unreachable: last attempt's except block re-raises
        raise RuntimeError("Doc2Query: all retry attempts exhausted")

    @staticmethod
    def _parse_questions(text: str) -> list[str]:
        """Parse LLM output into individual questions.

        Only lines ending with '?' are accepted. This prevents LLM preamble,
        statements, and partial code snippets from leaking into the pipeline.
        """
        lines = text.strip().splitlines()
        questions: list[str] = []
        for line in lines:
            # Strip leading numbering (1. 2. - * etc.)
            cleaned = re.sub(r"^\s*(?:\d+[.)]\s*|[*-]\s*)", "", line).strip()
            if cleaned and cleaned.endswith("?"):
                questions.append(cleaned)
        return questions


# ── Augmentation Cache ───────────────────────────────────────────────────────

_CACHE_SCHEMA_VERSION = 1


@dataclass
class CacheEntry:
    """A single cache entry for a chunk's Doc2Query results."""

    status: str  # "success" | "failed" | "absent"
    questions: list[str] = field(default_factory=list)
    attempts: int = 0


class AugmentationCache:
    """JSON-backed 3-state cache for Doc2Query augmentation results.

    Per RFC §4.2 step 2:
    - Key: chunk content-addressed hash
    - States: success (questions stored), failed (retry up to max), absent (unattempted)
    - Skip only if status=success
    - Retry if status=failed (up to max_retries attempts)
    """

    def __init__(self, cache_path: Path, max_retries: int = 3) -> None:
        self._path = cache_path
        self._max_retries = max_retries
        self._entries: dict[str, CacheEntry] = {}
        if cache_path.exists():
            self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            data.pop("_schema_version", None)
            for k, v in data.items():
                self._entries[k] = CacheEntry(
                    status=v["status"],
                    questions=v.get("questions", []),
                    attempts=v.get("attempts", 0),
                )
        except (json.JSONDecodeError, KeyError):
            logger.warning("Corrupt cache at %s — starting fresh", self._path)
            self._entries = {}

    def save(self) -> None:
        """Persist cache to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, object] = {"_schema_version": _CACHE_SCHEMA_VERSION}
        data.update({k: asdict(v) for k, v in self._entries.items()})
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def should_process(self, chunk_hash: str) -> bool:
        """Return True if this chunk needs (re-)processing."""
        entry = self._entries.get(chunk_hash)
        if entry is None:
            return True
        if entry.status == "success":
            return False
        if entry.status == "failed" and entry.attempts >= self._max_retries:
            return False
        return True

    def get(self, chunk_hash: str) -> CacheEntry | None:
        return self._entries.get(chunk_hash)

    def mark_success(self, chunk_hash: str, questions: list[str]) -> None:
        entry = self._entries.get(chunk_hash)
        attempts = (entry.attempts + 1) if entry else 1
        self._entries[chunk_hash] = CacheEntry(
            status="success", questions=questions, attempts=attempts,
        )

    def mark_failed(self, chunk_hash: str) -> None:
        entry = self._entries.get(chunk_hash)
        attempts = (entry.attempts + 1) if entry else 1
        self._entries[chunk_hash] = CacheEntry(
            status="failed", questions=[], attempts=attempts,
        )

    @property
    def stats(self) -> dict[str, int]:
        counts = {"success": 0, "failed": 0, "absent": 0, "total": len(self._entries)}
        for e in self._entries.values():
            counts[e.status] = counts.get(e.status, 0) + 1
        return counts
