"""HydRAG configuration with domain-agnostic defaults."""

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger("hydrag.config")

HYDRAG_SPEC_VERSION = "2.3"


@dataclass
class HydRAGConfig:
    """Central configuration for the HydRAG pipeline.

    Profiles
    --------
    - ``"prose"`` (default): Domain-agnostic. No symbol detection,
      always uses hybrid retrieval for the primary head. Suitable for
      documentation, legal, medical, or any non-code corpus.
    - ``"code"``: Enables regex-based symbol detection
      (CamelCase, snake_case, dotted paths, backtick identifiers).
      When symbols are found the primary head switches to a
      code-aware search (semantic + keyword fused via RRF).

    Load from environment variables (``HYDRAG_`` prefix) or pass
    values directly. See ``from_env()`` for the mapping.
    """

    profile: str = "prose"
    embedding_model: str = "Alibaba-NLP/gte-Qwen2-7B-instruct"
    crag_model: str = "qwen3:4b"
    crag_timeout: int = 30
    ollama_host: str = "http://localhost:11434"
    enable_web_fallback: bool = False  # Also gated by enable_head_3b_web per-head switch
    allow_web_on_empty_primary: bool = False
    allow_markdown_in_web_fallback: bool = False
    rrf_k: int = 60
    min_candidate_pool: int = 8
    web_chunk_limit: int = 3000
    crag_min_relevance: float = 0.12
    crag_context_chunks: int = 5
    crag_char_limit: int = 1500
    # V2.2: Head 0 BM25 fast-path (§3.0 spec) — on by default
    enable_fast_path: bool = True
    fast_path_bm25_threshold: float = 0.6
    # V2.3: Confidence-gated CRAG skip — if fast-path score >= threshold, skip LLM
    fast_path_confidence_threshold: float = 0.7
    # V2.3: Streaming CRAG — parse first token for early verdict
    crag_stream: bool = True
    # V2.3: CRAG mode — auto (classifier if available, else LLM), llm, classifier
    crag_mode: str = "auto"
    crag_classifier_path: str = ""
    # Per-head enable/disable switches
    enable_head_0: bool = True       # BM25 fast path
    enable_head_1: bool = True       # Primary retrieval (hybrid/code-aware)
    enable_head_2_crag: bool = True  # CRAG supervisor
    enable_head_3a_semantic: bool = True  # Semantic fallback
    enable_head_3b_web: bool = False      # Web fallback (also gated by enable_web_fallback)
    fallback_timeout_s: float = 30.0
    # V2.3+: Multi-provider LLM support (RFC V3)
    llm_provider: str = "ollama"
    hf_model_id: str = ""
    hf_api_base: str = ""
    hf_timeout: int = 30
    openai_compat_api_base: str = ""
    openai_compat_model: str = ""
    openai_compat_timeout: int = 30
    openai_compat_endpoint: str = "/v1/chat/completions"
    rrf_head_weights: dict[str, float] = field(
        default_factory=lambda: {
            "head_1a": 1.5,
            "head_1b": 1.0,
            "head_3a": 1.0,
            "head_3b": 0.8,
        }
    )

    def __post_init__(self) -> None:
        """Validate config fields."""
        if self.profile not in ("prose", "code"):
            raise ValueError(f"profile must be 'prose' or 'code', got {self.profile!r}")
        if self.crag_mode not in ("auto", "llm", "classifier"):
            raise ValueError(f"crag_mode must be 'auto', 'llm', or 'classifier', got {self.crag_mode!r}")
        if not (0 < self.fast_path_bm25_threshold <= 1.0):
            raise ValueError(f"fast_path_bm25_threshold must be in (0, 1.0], got {self.fast_path_bm25_threshold}")
        if not (0 <= self.fast_path_confidence_threshold <= 1.0):
            raise ValueError(f"fast_path_confidence_threshold must be in [0, 1.0], got {self.fast_path_confidence_threshold}")
        if self.rrf_k < 1:
            raise ValueError(f"rrf_k must be >= 1, got {self.rrf_k}")

    @classmethod
    def from_env(cls) -> "HydRAGConfig":
        """Create config from environment variables (HYDRAG_ prefix)."""
        cfg = cls(
            profile=os.environ.get("HYDRAG_PROFILE", cls.profile),
            embedding_model=os.environ.get("HYDRAG_EMBEDDING_MODEL", cls.embedding_model),
            crag_model=os.environ.get("CRAG_MODEL", cls.crag_model),
            crag_timeout=int(os.environ.get("CRAG_TIMEOUT", str(cls.crag_timeout))),
            ollama_host=os.environ.get("OLLAMA_HOST", cls.ollama_host),
            enable_web_fallback=os.environ.get(
                "HYDRAG_ENABLE_WEB_FALLBACK", ""
            ).lower()
            in ("1", "true"),
            allow_web_on_empty_primary=os.environ.get(
                "HYDRAG_ALLOW_WEB_ON_EMPTY", ""
            ).lower()
            in ("1", "true"),
            allow_markdown_in_web_fallback=os.environ.get(
                "HYDRAG_ALLOW_MARKDOWN_WEB", ""
            ).lower()
            in ("1", "true"),
            rrf_k=int(os.environ.get("HYDRAG_RRF_K", str(cls.rrf_k))),
            min_candidate_pool=int(
                os.environ.get("HYDRAG_MIN_CANDIDATE_POOL", str(cls.min_candidate_pool))
            ),
            web_chunk_limit=int(
                os.environ.get("HYDRAG_WEB_CHUNK_LIMIT", str(cls.web_chunk_limit))
            ),
            crag_min_relevance=float(
                os.environ.get("HYDRAG_CRAG_MIN_RELEVANCE", str(cls.crag_min_relevance))
            ),
            crag_context_chunks=int(
                os.environ.get("HYDRAG_CRAG_CONTEXT_CHUNKS", str(cls.crag_context_chunks))
            ),
            crag_char_limit=int(
                os.environ.get("HYDRAG_CRAG_CHAR_LIMIT", str(cls.crag_char_limit))
            ),
            fallback_timeout_s=float(
                os.environ.get("HYDRAG_FALLBACK_TIMEOUT_S", str(cls.fallback_timeout_s))
            ),
            enable_fast_path=os.environ.get(
                "HYDRAG_ENABLE_FAST_PATH", "true"
            ).lower()
            in ("1", "true"),
            fast_path_bm25_threshold=float(
                os.environ.get("HYDRAG_FAST_PATH_BM25_THRESHOLD", str(cls.fast_path_bm25_threshold))
            ),
            fast_path_confidence_threshold=float(
                os.environ.get("HYDRAG_FAST_PATH_CONFIDENCE_THRESHOLD", str(cls.fast_path_confidence_threshold))
            ),
            crag_stream=os.environ.get(
                "HYDRAG_CRAG_STREAM", "true"
            ).lower()
            in ("1", "true"),
            crag_mode=os.environ.get("HYDRAG_CRAG_MODE", cls.crag_mode),
            crag_classifier_path=os.environ.get("HYDRAG_CRAG_CLASSIFIER_PATH", cls.crag_classifier_path),
            enable_head_0=os.environ.get("HYDRAG_ENABLE_HEAD_0", "true").lower() in ("1", "true"),
            enable_head_1=os.environ.get("HYDRAG_ENABLE_HEAD_1", "true").lower() in ("1", "true"),
            enable_head_2_crag=os.environ.get("HYDRAG_ENABLE_HEAD_2_CRAG", "true").lower() in ("1", "true"),
            enable_head_3a_semantic=os.environ.get("HYDRAG_ENABLE_HEAD_3A_SEMANTIC", "true").lower() in ("1", "true"),
            enable_head_3b_web=os.environ.get("HYDRAG_ENABLE_HEAD_3B_WEB", "").lower() in ("1", "true"),
            llm_provider=os.environ.get("HYDRAG_LLM_PROVIDER", cls.llm_provider),
            hf_model_id=os.environ.get("HYDRAG_HF_MODEL_ID", cls.hf_model_id),
            hf_api_base=os.environ.get("HYDRAG_HF_API_BASE", cls.hf_api_base),
            hf_timeout=int(os.environ.get("HYDRAG_HF_TIMEOUT", str(cls.hf_timeout))),
            openai_compat_api_base=os.environ.get("HYDRAG_OPENAI_COMPAT_API_BASE", cls.openai_compat_api_base),
            openai_compat_model=os.environ.get("HYDRAG_OPENAI_COMPAT_MODEL", cls.openai_compat_model),
            openai_compat_timeout=int(os.environ.get("HYDRAG_OPENAI_COMPAT_TIMEOUT", str(cls.openai_compat_timeout))),
            openai_compat_endpoint=os.environ.get("HYDRAG_OPENAI_COMPAT_ENDPOINT", cls.openai_compat_endpoint),
        )
        # Handle JSON dictionary for RRF weights
        weights_env = os.environ.get("HYDRAG_RRF_HEAD_WEIGHTS")
        if weights_env:
            import json
            try:
                cfg.rrf_head_weights = json.loads(weights_env)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Malformed HYDRAG_RRF_HEAD_WEIGHTS=%r — using defaults",
                    weights_env,
                )
        return cfg

