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
from .fusion import CRAGVerdict, RetrievalResult
from .fusion import _rrf_fuse as rrf_fuse
from .protocols import LLMProvider, StreamingLLMProvider, VectorStoreAdapter
from .sanitize import _sanitize_web_content as sanitize_web_content

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
    # Utilities
    "rrf_fuse",
    "sanitize_web_content",
] + _TUNE_EXPORTS
