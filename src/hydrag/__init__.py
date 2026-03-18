"""HydRAG — Multi-Headed Retrieval-Augmented Generation with CRAG supervision."""

from ._version import __version__
from .config import HYDRAG_SPEC_VERSION, HydRAGConfig
from .core import (
    HydRAG,
    OllamaProvider,
    crag_supervisor,
    hydrag_search,
    semantic_fallback,
    web_fallback,
)
from .doc2query import (
    AugmentationCache,
    CacheEntry,
    Doc2QueryConfig,
    Doc2QueryGenerator,
    compute_adaptive_n,
    smart_truncate,
)
from .fusion import CRAGVerdict, RetrievalResult
from .fusion import rrf_fuse
from .protocols import LLMProvider, StreamingLLMProvider, VectorStoreAdapter
from .providers.factory import create_llm_provider
from .providers.huggingface import HuggingFaceProvider
from .providers.openai_compat import OpenAICompatProvider
from .sanitize import sanitize_web_content
from .sqlite_store import IndexedChunk, SQLiteFTSStore

# Tune pipeline (optional — requires hydrag-core[tune] deps for training,
# but data structures and classifier loader are always importable)
try:
    from .tune import (
        CRAGClassifier,
        TrainingDataset,
        TrainingSample,
        export_onnx,
        generate_training_data,
        generate_training_data_from_logs,
        train_classifier,
        tune,
        tune_from_logs,
    )

    _TUNE_EXPORTS = [
        "CRAGClassifier",
        "TrainingDataset",
        "TrainingSample",
        "generate_training_data",
        "generate_training_data_from_logs",
        "train_classifier",
        "export_onnx",
        "tune",
        "tune_from_logs",
    ]
except ImportError:
    _TUNE_EXPORTS = []

__all__ = [
    "__version__",
    "HYDRAG_SPEC_VERSION",
    # Config
    "HydRAGConfig",
    # Protocols
    "VectorStoreAdapter",
    "LLMProvider",
    "StreamingLLMProvider",
    # Types
    "CRAGVerdict",
    "RetrievalResult",
    # Core functions
    "hydrag_search",
    "crag_supervisor",
    "semantic_fallback",
    "web_fallback",
    # Class wrapper
    "HydRAG",
    # Default provider
    "OllamaProvider",
    # Built-in providers (V2.3+ multi-provider)
    "HuggingFaceProvider",
    "OpenAICompatProvider",
    "create_llm_provider",
    # Utilities
    "rrf_fuse",
    "sanitize_web_content",
    # SQLite FTS5 (T-742/T-743)
    "SQLiteFTSStore",
    "IndexedChunk",
    # Doc2Query V2 (T-740)
    "Doc2QueryConfig",
    "Doc2QueryGenerator",
    "AugmentationCache",
    "CacheEntry",
    "compute_adaptive_n",
    "smart_truncate",
] + _TUNE_EXPORTS
