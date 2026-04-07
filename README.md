---
id: HYDRAG-CORE-README
category: guide
status: active
created: '2026-03-10'
updated: '2026-04-07'
summary: 'hydrag-core — Multi-Headed RAG with CRAG supervision; installation, quickstart, configuration, and adapter guide'
keywords:
  hydrag-core: 9
  retrieval-augmented-generation: 8
  crag: 7
  rrf: 7
  multi-headed: 6
  installation: 5
  vector-store-adapter: 5
  ollama: 4
  sqlite: 3
  surrealdb: 3
---

# hydrag-core

[![PyPI version](https://img.shields.io/pypi/v/hydrag-core.svg)](https://pypi.org/project/hydrag-core/)
[![Python](https://img.shields.io/pypi/pyversions/hydrag-core.svg)](https://pypi.org/project/hydrag-core/)
[![License](https://img.shields.io/pypi/l/hydrag-core.svg)](https://github.com/studio-playbook/hydrag-core/blob/main/LICENSE)

**HydRAG** — Multi-Headed Retrieval-Augmented Generation with CRAG supervision.

A standalone, domain-agnostic retrieval pipeline that fuses multiple retrieval
heads via Reciprocal Rank Fusion (RRF) and uses a Corrective RAG (CRAG)
supervisor to judge context sufficiency before triggering fallback strategies.

## Features

- **Multi-headed retrieval** — Five pipeline heads: BM25 fast-path → Primary (hybrid / code-aware) → CRAG supervisor → Semantic fallback → Web fallback
- **Per-head control** — Enable or disable individual heads at runtime via config or environment variables
- **Domain-agnostic by default** — `prose` profile works with any corpus (documentation, legal, medical, etc.)
- **Code-aware opt-in** — `code` profile detects symbols (CamelCase, snake_case, dotted paths) and routes to code-aware search
- **CRAG supervisor** — LLM-graded context sufficiency with streaming verdict parsing and confidence-gated fast-path skip
- **CRAG classifier** — Optional distilled DistilBERT binary classifier (<15 ms inference) replaces LLM-based CRAG via teacher-student training pipeline
- **Pluggable adapters** — Implement the `VectorStoreAdapter` protocol to connect any vector store
- **Multi-provider LLM support** — Select built-in `ollama`, `huggingface`, or `openai_compat` via config, or inject any custom `LLMProvider`
- **Zero required dependencies** — stdlib-only core; optional extras for ChromaDB, Firecrawl, fine-tuning, and dev tools
- **Fully typed** — PEP 561 compatible with `py.typed` marker; strict mypy passes

## Installation

```bash
pip install hydrag-core
```

With optional extras:

```bash
pip install hydrag-core[chromadb]     # ChromaDB adapter support
pip install hydrag-core[firecrawl]    # Web fallback via Firecrawl
pip install hydrag-core[tune]         # CRAG classifier fine-tuning (transformers, torch, onnxruntime)
pip install hydrag-core[dev]          # Development tools (pytest, ruff, mypy)
```

## Quick Start

```python
from hydrag import HydRAG, HydRAGConfig

# 1. Implement the VectorStoreAdapter protocol for your store
class MyAdapter:
    def semantic_search(self, query: str, n_results: int = 5) -> list[str]: ...
    def keyword_search(self, query: str, n_results: int = 5) -> list[str]: ...
    def hybrid_search(self, query: str, n_results: int = 5) -> list[str]: ...

# 2. Create engine and search
adapter = MyAdapter()
engine = HydRAG(adapter)
results = engine.search("How do I configure logging?")

for r in results:
    print(f"[{r.head_origin}] score={r.score:.4f}: {r.text[:80]}")
```

### Code Profile

```python
config = HydRAGConfig(profile="code")
engine = HydRAG(adapter, config=config)
# Queries with symbols (e.g. "HttpRequest class") trigger code-aware retrieval
results = engine.search("How does `HttpRequest` handle timeouts?")
```

### Custom LLM Provider

```python
from hydrag import LLMProvider

class MyLLM:
    def generate(self, prompt: str, model: str = "", timeout: int = 30) -> str | None:
        # Your LLM inference here
        ...

engine = HydRAG(adapter, llm=MyLLM())
```

### Built-In LLM Providers

HydRAG can construct the LLM provider from config without custom wiring.

```python
from hydrag import HydRAG, HydRAGConfig

# Default: Ollama
cfg = HydRAGConfig(llm_provider="ollama", ollama_host="http://localhost:11434")
engine = HydRAG(adapter, config=cfg)

# Hugging Face TGI-compatible endpoint
cfg = HydRAGConfig(
    llm_provider="huggingface",
    hf_api_base="http://localhost:8080",
    hf_model_id="meta-llama/Llama-3.1-8B-Instruct",
)
engine = HydRAG(adapter, config=cfg)

# OpenAI-compatible endpoint (vLLM, LM Studio, OpenRouter-compatible gateway, etc.)
cfg = HydRAGConfig(
    llm_provider="openai_compat",
    openai_compat_api_base="http://localhost:8000",
    openai_compat_model="Qwen/Qwen2.5-7B-Instruct",
)
engine = HydRAG(adapter, config=cfg)
```

Tokens are read from environment by default:

- `HYDRAG_HF_API_TOKEN` for `llm_provider="huggingface"`
- `HYDRAG_OPENAI_COMPAT_API_KEY` for `llm_provider="openai_compat"`

### Streaming CRAG Provider

For lower-latency CRAG verdict parsing, implement `StreamingLLMProvider`:

```python
from hydrag import StreamingLLMProvider

class MyStreamingLLM:
    def generate(self, prompt: str, model: str = "", timeout: int = 30) -> str | None: ...
    def generate_stream(self, prompt: str, model: str = "", timeout: int = 30) -> str | None:
        # Return full response; first token is parsed for early verdict
        ...

engine = HydRAG(adapter, llm=MyStreamingLLM())
```

### Per-Head Control

```python
# Run only BM25 fast-path + primary retrieval (skip CRAG, semantic, web)
config = HydRAGConfig(
    enable_head_0=True,
    enable_head_1=True,
    enable_head_2_crag=False,
    enable_head_3a_semantic=False,
    enable_head_3b_web=False,
)
engine = HydRAG(adapter, config=config)
results = engine.search("quick keyword lookup")
```

### CRAG Classifier (Optional)

```python
from hydrag import tune, CRAGClassifier

# Fine-tune a fast classifier from your corpus
tune(
    adapter=adapter,
    llm=llm,
    output_dir="models/crag_classifier",
    n_samples=500,
)

# Use it — auto-detected when crag_mode="auto" and path is set
config = HydRAGConfig(
    crag_mode="auto",
    crag_classifier_path="models/crag_classifier",
)
engine = HydRAG(adapter, config=config)
```

## Configuration

All settings can be set via environment variables (`HYDRAG_` prefix) or passed directly:

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `profile` | `HYDRAG_PROFILE` | `"prose"` | `"prose"` or `"code"` |
| `embedding_model` | `HYDRAG_EMBEDDING_MODEL` | `"Alibaba-NLP/gte-Qwen2-7B-instruct"` | Embedding model name |
| `crag_model` | `CRAG_MODEL` | `"qwen3:4b"` | Model for CRAG supervisor |
| `crag_timeout` | `CRAG_TIMEOUT` | `30` | Timeout (seconds) for CRAG calls |
| `ollama_host` | `OLLAMA_HOST` | `"http://localhost:11434"` | Ollama API endpoint |
| `llm_provider` | `HYDRAG_LLM_PROVIDER` | `"ollama"` | LLM backend selector: `ollama`, `huggingface`, `openai_compat` |
| `hf_model_id` | `HYDRAG_HF_MODEL_ID` | `""` | Optional model id for Hugging Face provider |
| `hf_api_base` | `HYDRAG_HF_API_BASE` | `""` | Hugging Face/TGI base URL (required when `llm_provider="huggingface"`) |
| `hf_timeout` | `HYDRAG_HF_TIMEOUT` | `30` | Timeout (seconds) for Hugging Face provider calls |
| `openai_compat_api_base` | `HYDRAG_OPENAI_COMPAT_API_BASE` | `""` | OpenAI-compatible API base URL (required when `llm_provider="openai_compat"`) |
| `openai_compat_model` | `HYDRAG_OPENAI_COMPAT_MODEL` | `""` | Model name for OpenAI-compatible provider (required when `llm_provider="openai_compat"`) |
| `openai_compat_timeout` | `HYDRAG_OPENAI_COMPAT_TIMEOUT` | `30` | Timeout (seconds) for OpenAI-compatible provider calls |
| `openai_compat_endpoint` | `HYDRAG_OPENAI_COMPAT_ENDPOINT` | `"/v1/chat/completions"` | Override OpenAI-compatible chat completions endpoint path |
| `enable_web_fallback` | `HYDRAG_ENABLE_WEB_FALLBACK` | `false` | Enable Firecrawl web fallback |
| `allow_web_on_empty_primary` | `HYDRAG_ALLOW_WEB_ON_EMPTY` | `false` | Allow web fallback when Head 1 returns empty |
| `allow_markdown_in_web_fallback` | `HYDRAG_ALLOW_MARKDOWN_WEB` | `false` | Preserve markdown in web fallback content |
| `rrf_k` | `HYDRAG_RRF_K` | `60` | RRF smoothing constant |
| `min_candidate_pool` | `HYDRAG_MIN_CANDIDATE_POOL` | `8` | Minimum candidates per head |
| `web_chunk_limit` | `HYDRAG_WEB_CHUNK_LIMIT` | `3000` | Max chars per web chunk after sanitization |
| `crag_min_relevance` | `HYDRAG_CRAG_MIN_RELEVANCE` | `0.67` | Minimum relevance threshold for CRAG |
| `crag_context_chunks` | `HYDRAG_CRAG_CONTEXT_CHUNKS` | `5` | Chunks sent to CRAG supervisor |
| `crag_char_limit` | `HYDRAG_CRAG_CHAR_LIMIT` | `1500` | Per-chunk character limit in CRAG prompt |
| `enable_fast_path` | `HYDRAG_ENABLE_FAST_PATH` | `true` | Head 0 BM25 fast-path |
| `fast_path_bm25_threshold` | `HYDRAG_FAST_PATH_BM25_THRESHOLD` | `0.67` | BM25 score threshold for fast-path |
| `fast_path_confidence_threshold` | `HYDRAG_FAST_PATH_CONFIDENCE_THRESHOLD` | `0.8` | Score threshold to skip CRAG entirely |
| `crag_stream` | `HYDRAG_CRAG_STREAM` | `true` | Parse first token for early CRAG verdict |
| `crag_mode` | `HYDRAG_CRAG_MODE` | `"auto"` | `"auto"`, `"llm"`, or `"classifier"` |
| `crag_classifier_path` | `HYDRAG_CRAG_CLASSIFIER_PATH` | `""` | Path to ONNX classifier model |
| `enable_head_0` | `HYDRAG_ENABLE_HEAD_0` | `true` | BM25 fast-path head |
| `enable_head_1` | `HYDRAG_ENABLE_HEAD_1` | `true` | Primary retrieval head |
| `enable_head_2_crag` | `HYDRAG_ENABLE_HEAD_2_CRAG` | `true` | CRAG supervisor head |
| `enable_head_3a_semantic` | `HYDRAG_ENABLE_HEAD_3A_SEMANTIC` | `true` | Semantic fallback head |
| `enable_head_3b_web` | `HYDRAG_ENABLE_HEAD_3B_WEB` | `false` | Web fallback head |
| `fallback_timeout_s` | `HYDRAG_FALLBACK_TIMEOUT_S` | `5.0` | Timeout for fallback head futures |
| `rrf_head_weights` | `HYDRAG_RRF_HEAD_WEIGHTS` | *(JSON)* | Per-head RRF weight map |

```python
# From environment
config = HydRAGConfig.from_env()

# Direct
config = HydRAGConfig(profile="code", rrf_k=100, enable_web_fallback=True)
```

## Adapter Protocol

Required methods (must implement all three):

```python
class VectorStoreAdapter(Protocol):
    def semantic_search(self, query: str, n_results: int = 5) -> list[str]: ...
    def keyword_search(self, query: str, n_results: int = 5) -> list[str]: ...
    def hybrid_search(self, query: str, n_results: int = 5) -> list[str]: ...
```

Optional methods (gracefully handled if missing):

```python
    def crag_search(self, query: str, n_results: int = 5) -> list[str]: ...
    def graph_search(self, query: str, n_results: int = 5) -> list[str]: ...
    def rewrite_query(self, query: str) -> str: ...
```

## Pipeline Architecture

```
Query → [Profile Router]
           │
           ├─ Head 0: BM25 Fast-Path
           │    └─ score ≥ threshold? → early return
           │    └─ score ≥ confidence threshold? → skip CRAG
           │
           ├─ Head 1: Primary Retrieval
           │    ├─ prose → hybrid_search()
           │    └─ code + symbols → code-aware (semantic + keyword RRF)
           │
           ├─ Head 2: CRAG Supervisor
           │    ├─ classifier (ONNX, <15ms) ← crag_mode="auto"|"classifier"
           │    └─ LLM (streaming verdict)  ← crag_mode="auto"|"llm"
           │         ├─ SUFFICIENT → return Head 1 results
           │         └─ INSUFFICIENT ↓
           │
           ├─ Head 3a: Semantic Fallback
           │    └─ rewrite + CRAG search + keyword + graph → RRF
           │
           └─ Head 3b: Web Fallback (optional)
                └─ Firecrawl search → sanitize → RRF

           Final: RRF Fusion across all active heads → list[RetrievalResult]
```

## Development

```bash
# Clone and install in development mode
git clone https://github.com/gromanchenko/hydrag.git
cd hydrag-core
pip install -e ".[dev]"

# Run tests
python -m pytest tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Versioning

HydRAG uses SemVer. Version sources that must stay in sync:

- `pyproject.toml` (`[project].version`)
- `src/hydrag/_version.py` (`__version__`)

## Type Checking

This package ships a PEP 561 `py.typed` marker. Downstream projects using
`mypy --strict` will get full type coverage out of the box.

## License

Apache-2.0 — see [LICENSE](LICENSE) for details.
