---
id: HYDRAG-CORE-API-REFERENCE
category: reference
status: active
created: '2026-03-15'
updated: '2026-04-06'
summary: 'hydrag-core public API reference — HydRAG, HydRAGConfig, VectorStoreAdapter, IndexedChunk, SurrealDBAdapter'
keywords:
  hydrag-core: 9
  api: 8
  reference: 7
  surrealdb-adapter: 6
  vector-store: 6
  hydrag: 5
  indexed-chunk: 5
  rrf: 4
  crag: 4
---
# hydrag-core API Reference

Version: **See `hydrag.__version__`** | Spec: **HydRAG V2.3**

---

## Quick Start

```python
from hydrag import HydRAG, HydRAGConfig

class MyAdapter:
    def semantic_search(self, query: str, n_results: int = 5) -> list[str]: ...
    def keyword_search(self, query: str, n_results: int = 5) -> list[str]: ...
    def hybrid_search(self, query: str, n_results: int = 5) -> list[str]: ...

engine = HydRAG(MyAdapter())
results = engine.search("How does RRF fusion work?")
```

---

## Constants

| Name | Type | Value | Description |
|------|------|-------|-------------|
| `__version__` | `str` | runtime value | Package version (SemVer) |
| `HYDRAG_SPEC_VERSION` | `str` | `"2.3"` | HydRAG specification version this package implements |

---

## `HydRAGConfig`

Central configuration dataclass for the HydRAG pipeline.

```python
from hydrag import HydRAGConfig
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `profile` | `str` | `"prose"` | Domain profile: `"prose"` (default, domain-agnostic) or `"code"` (symbol-aware) |
| `embedding_model` | `str` | `"Alibaba-NLP/gte-Qwen2-7B-instruct"` | Embedding model name for vector operations |
| `crag_model` | `str` | `"qwen3:4b"` | Ollama model for CRAG supervisor |
| `crag_timeout` | `int` | `30` | CRAG inference timeout (seconds) |
| `ollama_host` | `str` | `"http://localhost:11434"` | Ollama server URL |
| `enable_web_fallback` | `bool` | `False` | Enable Head 3b web fallback via Firecrawl |
| `allow_web_on_empty_primary` | `bool` | `False` | Allow web fallback when Head 1 returns zero results |
| `allow_markdown_in_web_fallback` | `bool` | `False` | Preserve markdown in web fallback content |
| `rrf_k` | `int` | `60` | RRF smoothing constant |
| `min_candidate_pool` | `int` | `8` | Minimum candidates requested from each head |
| `web_chunk_limit` | `int` | `3000` | Max chars per web chunk after sanitization |
| `crag_min_relevance` | `float` | `0.12` | Minimum relevance threshold for CRAG evaluation |
| `crag_context_chunks` | `int` | `5` | Number of chunks sent to CRAG supervisor |
| `crag_char_limit` | `int` | `1500` | Per-chunk character limit in CRAG prompt |
| `enable_fast_path` | `bool` | `True` | Enable Head 0 BM25 fast-path |
| `fast_path_bm25_threshold` | `float` | `0.67` | BM25 hit ratio threshold for fast-path early exit |
| `fast_path_confidence_threshold` | `float` | `0.8` | BM25 score threshold to skip CRAG entirely |
| `crag_stream` | `bool` | `True` | Parse first LLM token for early CRAG verdict |
| `crag_mode` | `str` | `"auto"` | CRAG mode: `"auto"` (classifier if available, else LLM), `"llm"`, `"classifier"` |
| `crag_classifier_path` | `str` | `""` | Path to ONNX classifier model for CRAG |
| `enable_head_0` | `bool` | `True` | Enable BM25 fast-path head |
| `enable_head_1` | `bool` | `True` | Enable primary retrieval head |
| `enable_head_2_crag` | `bool` | `True` | Enable CRAG supervisor head |
| `enable_head_3a_semantic` | `bool` | `True` | Enable semantic fallback head |
| `enable_head_3b_web` | `bool` | `False` | Enable web fallback head (also gated by `enable_web_fallback`) |
| `fallback_timeout_s` | `float` | `5.0` | Timeout (seconds) for fallback head futures |
| `rrf_head_weights` | `dict[str, float]` | `{"head_1a": 1.5, "head_1b": 1.0, "head_3a": 1.0, "head_3b": 0.8}` | Per-head RRF weight map |

### Methods

#### `HydRAGConfig.from_env() -> HydRAGConfig`

Load configuration from environment variables.

| Environment Variable | Maps To |
|---------------------|---------|
| `HYDRAG_PROFILE` | `profile` |
| `HYDRAG_EMBEDDING_MODEL` | `embedding_model` |
| `CRAG_MODEL` | `crag_model` |
| `CRAG_TIMEOUT` | `crag_timeout` |
| `OLLAMA_HOST` | `ollama_host` |
| `HYDRAG_ENABLE_WEB_FALLBACK` | `enable_web_fallback` |
| `HYDRAG_ALLOW_WEB_ON_EMPTY` | `allow_web_on_empty_primary` |
| `HYDRAG_ALLOW_MARKDOWN_WEB` | `allow_markdown_in_web_fallback` |
| `HYDRAG_RRF_K` | `rrf_k` |
| `HYDRAG_MIN_CANDIDATE_POOL` | `min_candidate_pool` |
| `HYDRAG_WEB_CHUNK_LIMIT` | `web_chunk_limit` |
| `HYDRAG_CRAG_MIN_RELEVANCE` | `crag_min_relevance` |
| `HYDRAG_CRAG_CONTEXT_CHUNKS` | `crag_context_chunks` |
| `HYDRAG_CRAG_CHAR_LIMIT` | `crag_char_limit` |
| `HYDRAG_FALLBACK_TIMEOUT_S` | `fallback_timeout_s` |
| `HYDRAG_ENABLE_FAST_PATH` | `enable_fast_path` |
| `HYDRAG_FAST_PATH_BM25_THRESHOLD` | `fast_path_bm25_threshold` |
| `HYDRAG_FAST_PATH_CONFIDENCE_THRESHOLD` | `fast_path_confidence_threshold` |
| `HYDRAG_CRAG_STREAM` | `crag_stream` |
| `HYDRAG_CRAG_MODE` | `crag_mode` |
| `HYDRAG_CRAG_CLASSIFIER_PATH` | `crag_classifier_path` |
| `HYDRAG_ENABLE_HEAD_0` | `enable_head_0` |
| `HYDRAG_ENABLE_HEAD_1` | `enable_head_1` |
| `HYDRAG_ENABLE_HEAD_2_CRAG` | `enable_head_2_crag` |
| `HYDRAG_ENABLE_HEAD_3A_SEMANTIC` | `enable_head_3a_semantic` |
| `HYDRAG_ENABLE_HEAD_3B_WEB` | `enable_head_3b_web` |
| `HYDRAG_RRF_HEAD_WEIGHTS` | `rrf_head_weights` (JSON) |

---

## Protocols

### `VectorStoreAdapter` (Protocol)

Runtime-checkable protocol for vector store implementations.

```python
from hydrag import VectorStoreAdapter
```

**Required methods** (must implement all three):

| Method | Signature | Description |
|--------|-----------|-------------|
| `semantic_search` | `(query: str, n_results: int = 5) -> list[str]` | Embedding-based similarity search |
| `keyword_search` | `(query: str, n_results: int = 5) -> list[str]` | BM25 / inverted-index keyword search |
| `hybrid_search` | `(query: str, n_results: int = 5) -> list[str]` | Combined semantic + keyword search |

**Optional methods** (gracefully handled if missing):

| Method | Signature | Fallback |
|--------|-----------|----------|
| `crag_search` | `(query: str, n_results: int = 5) -> list[str]` | Falls back to `hybrid_search` |
| `graph_search` | `(query: str, n_results: int = 5) -> list[str]` | Skipped when absent |
| `rewrite_query` | `(query: str) -> str` | Returns original query unchanged |

### `LLMProvider` (Protocol)

Runtime-checkable protocol for LLM inference.

```python
from hydrag import LLMProvider
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `generate` | `(prompt: str, model: str = "", timeout: int = 30) -> str \| None` | Generate text; return `None` on failure |

### `StreamingLLMProvider` (Protocol)

Runtime-checkable extension of `LLMProvider` for streaming CRAG verdict parsing.

```python
from hydrag import StreamingLLMProvider
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `generate` | `(prompt: str, model: str = "", timeout: int = 30) -> str \| None` | Inherited from `LLMProvider` |
| `generate_stream` | `(prompt: str, model: str = "", timeout: int = 30) -> str \| None` | Generate with first-token parsing for early verdict |

When `crag_stream=True` (default), the CRAG supervisor checks if the provided LLM implements `StreamingLLMProvider`. If so, it calls `generate_stream()` to parse the first token for an early SUFFICIENT/INSUFFICIENT verdict, reducing latency.

---

## Core Functions

### `hydrag_search()`

Main orchestrator — multi-headed retrieval with CRAG supervision.

```python
from hydrag import hydrag_search

results = hydrag_search(
    adapter,              # VectorStoreAdapter
    query="...",          # User query
    n_results=5,          # Top-k results
    enable_web_fallback=None,  # Override config (None = use config)
    config=None,          # HydRAGConfig (None = defaults)
    llm=None,             # LLMProvider (None = OllamaProvider)
)
```

**Pipeline architecture:**

1. **Head 0** (BM25 fast-path): If `enable_fast_path=True` and keyword hit ratio ≥ `fast_path_bm25_threshold`, returns immediately. If score ≥ `fast_path_confidence_threshold`, skips CRAG.
2. **Head 1** (Primary): hybrid (prose) or code-aware (code profile + symbols detected)
3. **Head 2** (CRAG Supervisor): LLM or classifier judges SUFFICIENT / INSUFFICIENT
4. **Head 3a** (Semantic fallback): rewrite + CRAG search + keyword + graph, RRF-fused
5. **Head 3b** (Web fallback): Firecrawl search (requires `FIRECRAWL_API_KEY`)
6. **Final fusion**: RRF across all active heads with configurable weights

**Returns:** `list[RetrievalResult]`

### `crag_supervisor()`

LLM-graded or classifier-based Corrective RAG supervisor.

```python
from hydrag import crag_supervisor

verdict = crag_supervisor(query, context_chunks, llm, config)
# verdict.sufficient: bool
# verdict.reason: str  ("model_verdict", "model_unreachable", "ambiguous_response", "classifier_verdict")
# verdict.latency_ms: float
# verdict.raw_response: str | None
```

When `crag_mode="auto"` and `crag_classifier_path` is set, prefers the ONNX classifier (<15 ms) over LLM inference.

### `semantic_fallback()`

Triggered when CRAG returns INSUFFICIENT. Rewrites query, runs CRAG + keyword + graph search, fuses via RRF.

```python
from hydrag import semantic_fallback

results = semantic_fallback(adapter, query, primary_hits, n_results=5, config=None)
```

### `web_fallback()`

External web search via Firecrawl. Returns empty list if `firecrawl-py` not installed or `FIRECRAWL_API_KEY` not set.

```python
from hydrag import web_fallback

results = web_fallback(query, limit=3, config=None)
```

---

## Types

### `RetrievalResult`

```python
from hydrag import RetrievalResult

@dataclass
class RetrievalResult:
    text: str                    # Retrieved chunk text
    source: str                  # Source identifier
    score: float                 # RRF fusion score
    head_origin: str             # Which head produced this ("head_0", "head_1a", "head_1b", "hydrag", ...)
    trust_level: str             # "local" | "web"
    metadata: dict = field(default_factory=dict)  # Telemetry (e.g. fast_path_triggered, crag_skipped)
    crag_verdict: CRAGVerdict | None = None
```

### `CRAGVerdict`

```python
from hydrag import CRAGVerdict

@dataclass
class CRAGVerdict:
    sufficient: bool             # CRAG judgment
    reason: str                  # "model_verdict" | "model_unreachable" | "ambiguous_response" | "classifier_verdict"
    latency_ms: float            # Inference time
    raw_response: str | None = None  # Raw LLM output (None when classifier used)
```

---

## Built-in Providers

### `OllamaProvider`

```python
from hydrag import OllamaProvider

llm = OllamaProvider(host="http://localhost:11434")
response = llm.generate("Hello", model="qwen3:4b", timeout=30)
```

3-attempt retry with 0.5s backoff. Returns `None` on total failure.

---

## `HydRAG` Class

Convenience wrapper holding adapter + config + LLM provider.

```python
from hydrag import HydRAG, HydRAGConfig

config = HydRAGConfig(profile="code", enable_fast_path=True)
engine = HydRAG(adapter, config=config)

results = engine.search("How does rrf_fuse work?", n_results=5)
```

---

## Utilities

### `rrf_fuse()`

Reciprocal Rank Fusion with weighted multi-source and tie-breaking.

```python
from hydrag import rrf_fuse

# sources: list of (items, weight) tuples
results = rrf_fuse(
    sources=[
        (semantic_hits, 1.5),
        (keyword_hits, 1.0),
    ],
    k=60,
    n_results=10,
)
```

### `sanitize_web_content()`

Strip scripts, styles, HTML tags, and optionally markdown from raw web content.

```python
from hydrag import sanitize_web_content

clean = sanitize_web_content(raw_html, max_chars=3000, allow_markdown=False)
```

---

## Tune Module (Optional)

Requires `pip install hydrag-core[tune]` for training. Data structures and classifier loader are always importable.

### `CRAGClassifier`

ONNX-based binary classifier for fast CRAG inference (<15 ms).

```python
from hydrag import CRAGClassifier

classifier = CRAGClassifier("models/crag_classifier")
verdict = classifier.predict(query="...", context="...")
# verdict.sufficient: bool, verdict.latency_ms: float
```

### `tune()`

One-call pipeline: generate training data → fine-tune → export ONNX.

```python
from hydrag import tune

tune(
    adapter=adapter,
    llm=llm,
    output_dir="models/crag_classifier",
    n_samples=500,
    epochs=3,
)
```

### `tune_from_logs()`

Fine-tune from existing benchmark logs (no LLM teacher needed).

```python
from hydrag import tune_from_logs

tune_from_logs(
    log_dir="benchmarks/results/",
    output_dir="models/crag_classifier",
)
```

### Training Data Types

```python
from hydrag import TrainingSample, TrainingDataset

sample = TrainingSample(query="...", context="...", label=1)  # 1=SUFFICIENT, 0=INSUFFICIENT
dataset = TrainingDataset(samples=[sample])
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `config` | `HydRAGConfig` | Current configuration |

---

## Internal Functions (exported for testing)

| Function | Signature | Description |
|----------|-----------|-------------|
| `_rrf_fuse` | `(sources, *, k, n_results, head_origin) -> list[RetrievalResult]` | Reciprocal Rank Fusion with weighted multi-source |
| `_sanitize_web_content` | `(html, *, max_chars, allow_markdown) -> str` | Strip scripts/styles/whitespace from HTML |
