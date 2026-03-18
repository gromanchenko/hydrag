# Changelog

All notable changes to `hydrag-core` are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: [SemVer](https://semver.org/).

---

## [1.2.4] — 2026-03-18

### Added
- **Doc2Query V2 integration** (T-746): `compute_adaptive_n()` (RFC §2.3 lookup table), `smart_truncate()` (boundary-aware truncation with tail overlap), `Doc2QueryConfig`, `Doc2QueryGenerator`, `AugmentationCache`, `CacheEntry` — all integrated from hydrag-benchmark into core
- **SQLite FTS5 adapter** (T-742/T-743): `SQLiteFTSStore` and `IndexedChunk` for pure-stdlib full-text search without external dependencies
- **Multi-provider LLM support**: Factory-based provider selection via `llm_provider` config field
- **Built-in providers**: `ollama` (default, extracted from core.py), `huggingface` (TGI), `openai_compat` (OpenAI-compatible)
- **Provider factory**: `create_llm_provider(config, api_tokens=None)` resolves provider from config
- **8 new config fields**: `llm_provider`, `hf_model_id`, `hf_api_base`, `hf_timeout`, `openai_compat_api_base`, `openai_compat_model`, `openai_compat_timeout`, `openai_compat_endpoint`
- **Error contract split**: Auth/config errors raise; transient transport errors return `None` (CRAG fail-open)
- **Conformance test harness**: Parametrized across all providers with mock HTTP
- **Secret passthrough**: Factory `api_tokens` parameter for library-first usage without env vars
- 6 new public exports: `Doc2QueryConfig`, `Doc2QueryGenerator`, `AugmentationCache`, `CacheEntry`, `compute_adaptive_n`, `smart_truncate`
- 38 new Doc2Query tests (255 total)

### Changed
- `core.py` uses factory instead of hardcoded `OllamaProvider` for default LLM bootstrap
- `OllamaProvider` moved to `providers/ollama.py` (re-exported from original location for backward compat)

---

## [1.0.5] — 2026-03-13

### Fixed
- **Tracked scratch file**: Removed `test_profile4.py` (dev timing script) from repository; added to `.gitignore`
- **CHANGELOG accuracy**: Corrected false v1.0.3 entry that claimed `crag_min_relevance` was removed (field still exists and is wired in `from_env()`)
- **API Reference spec version**: Header now reads `V2.3` (was `V2.2+`)
- **API Reference dataclass defaults**: `RetrievalResult` and `CRAGVerdict` field signatures now match actual code (required fields no longer shown with false defaults)
- **API Reference**: Added `StreamingLLMProvider` protocol documentation and `fallback_timeout_s` to config fields table
- **README**: Added 9 missing config fields to the configuration table (`allow_web_on_empty_primary`, `allow_markdown_in_web_fallback`, `web_chunk_limit`, `crag_min_relevance`, `crag_context_chunks`, `crag_char_limit`, `fallback_timeout_s`, `rrf_head_weights`)
- **README**: Added `StreamingLLMProvider` usage example
- **README**: Removed stale `.githooks` versioning automation section (hooks live in parent monorepo, not this standalone package)
- **`enable_web_fallback` comment**: Replaced misleading "Deprecated" comment with accurate description
- **Internal ticket refs**: Stripped `T-515`, `T-516`, `T-459` references from CHANGELOG historical entries
- **numpy pin**: Relaxed `numpy>=1.24,<2.0` to `numpy>=1.24` in `[tune]` extras (NumPy 2.x compatible)

### Added
- `CONTRIBUTING.md` — contributor guide for public GitHub
- `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1
- `.github/workflows/ci.yml` — GitHub Actions CI (test + lint + type-check on Python 3.10–3.13)
- `py.typed` marker documented in README

## [1.0.4] — 2026-03-12

### Fixed
- Release readiness fixes from Gemini audit: import sort order (`ruff --fix`), ThreadPoolExecutor context managers, test coverage for fail-open metadata and head conflict resolution

## [1.0.3] — 2026-03-12

### Fixed
- **Documentation accuracy**: `tune()` code samples used `num_samples` instead of the actual parameter name `n_samples`
- **Documentation accuracy**: `CRAGClassifier` examples passed an `.onnx` file path instead of the required directory path
- **API Reference**: `HYDRAG_SPEC_VERSION` incorrectly listed as `"2.2"` (actual: `"2.3"`)
- **API Reference**: Removed non-existent `adapter` property from `HydRAG` class documentation
- **mypy --strict**: Fixed 25 type errors — bare `list` → `list[Any]`, stale `type: ignore` comments, `no-any-return`, variable redefinition
- **ruff**: Fixed 21 lint errors — unsorted imports, unused imports in tests, `noqa: F401` for re-exported tune symbols
- **`.gitignore`**: Expanded from `dist/` only to standard Python project ignores (`__pycache__/`, `*.pyc`, `.pytest_cache/`, etc.)
- **Internal references**: Removed ticket IDs from source docstrings; removed internal repo path from `_version.py`
- **`OllamaProvider`**: Fallback model now uses `cfg.crag_model` instead of hardcoded `"devstral-small-2:latest"`
- **`from_env()`**: Wired `crag_context_chunks` and `crag_char_limit` to `HYDRAG_CRAG_CONTEXT_CHUNKS` and `HYDRAG_CRAG_CHAR_LIMIT` environment variables
- **`fallback_timeout_s`**: Added to `API_REFERENCE.md` env var table

### Changed
- `ThreadPoolExecutor` usage in `semantic_fallback` and `hydrag_search` now uses context manager (`with`) for idiomatic cleanup
- Removed stale `dist/` build artifacts (1.0.0) from repository

## [1.0.2] — 2026-03-12

### Fixed
- `lightrag_search` missing timeout on `future.result()` calls (studio-side fix)

## [1.0.1] — 2026-03-12

### Fixed
- Corrective pass: timeout handler teardown, `LLMProvider` protocol backward compatibility, `fallback_timeout_s` env wiring, `teacher_retry_budget` parameter, timeout behavioral tests

## [1.0.0] — 2026-03-12

### Added
- **Per-head enable/disable switches**: `enable_head_0`, `enable_head_1`, `enable_head_2_crag`, `enable_head_3a_semantic`, `enable_head_3b_web` — individually toggle each pipeline head via config or `HYDRAG_ENABLE_HEAD_*` environment variables
- **CRAG classifier pipeline**: Teacher-student distillation replaces LLM-based CRAG with a DistilBERT binary classifier (<15 ms ONNX inference)
  - `CRAGClassifier` — ONNX-based binary classifier with cached singleton loader
  - `TrainingSample`, `TrainingDataset` — typed training data containers
  - `generate_training_data()` — synthetic query generation with LLM teacher labeling
  - `generate_training_data_from_logs()` — training data from existing benchmark logs
  - `train_classifier()` — HuggingFace Transformers fine-tuning
  - `export_onnx()` — export trained model to ONNX format
  - `tune()` — one-call data → train → export pipeline
  - `tune_from_logs()` — fine-tune from existing logs without LLM teacher
- **Confidence-gated CRAG skip**: When BM25 fast-path score ≥ `fast_path_confidence_threshold` (0.7), CRAG evaluation is skipped entirely
- **Streaming CRAG**: `crag_stream=True` (default) parses the first token of LLM response for early SUFFICIENT/INSUFFICIENT verdict
- **CRAG mode selection**: `crag_mode` field — `"auto"` (classifier if available, else LLM), `"llm"`, or `"classifier"`
- `HydRAGConfig.embedding_model` field (default: `"Alibaba-NLP/gte-Qwen2-7B-instruct"`)
- `HydRAGConfig.crag_classifier_path` field for ONNX model path
- `rrf_head_weights` — configurable per-head weight map for RRF fusion
- PEP 561 `py.typed` marker for downstream type-checking support
- Comprehensive PyPI packaging: badges, sdist includes, `Typing :: Typed` classifier

### Changed
- Default `crag_model` → `"qwen3:4b"` (was `"devstral-small-2:latest"`)
- Default `enable_fast_path` → `True` (was `False`; Head 0 BM25 fast-path now on by default)
- Development Status classifier → `5 - Production/Stable` (was `3 - Alpha`)
- Pipeline architecture: Head 0 fast-path can now gate both early return AND CRAG skip

### Fixed
- `from_env()` now maps all V2.3 fields including per-head switches and classifier settings

---

## [0.2.0] — 2026-03-12

### Added
- `HYDRAG_SPEC_VERSION = "2.2"` constant (aligns package with HydRAG spec V2.2)
- `HydRAGConfig.enable_fast_path: bool` — opt-in Head 0 BM25 fast-path (§3.0 spec)
- `HydRAGConfig.fast_path_bm25_threshold: float` — score threshold for fast-path early exit (default 0.6)
- `HYDRAG_ENABLE_FAST_PATH` and `HYDRAG_FAST_PATH_BM25_THRESHOLD` environment variable wiring in `from_env()`
- `HYDRAG_SPEC_VERSION` exported from top-level `hydrag` package
- **Head 0 runtime**: `hydrag_search()` now executes BM25 fast-path when `enable_fast_path=True` — short-circuits full pipeline for high-confidence keyword hits (§3.0 spec)
- Telemetry: `head_origin="head_0"` and `fast_path_triggered=True` in `RetrievalResult.metadata`
- 9 new tests for Head 0 fast-path (config + runtime); 78 total passing

### Notes
- Fast-path is **off by default** (`enable_fast_path=False`). Runtime wiring is a follow-up.
- API is backward-compatible; existing callers require no changes.

---

## [0.1.0a1] — 2026-03-10 (initial)

### Added
- `HydRAGConfig` with `profile` ("prose"/"code"), `crag_model`, `rrf_k`, `enable_web_fallback`, and all core settings
- `HydRAG` class — full multi-headed pipeline: Head 1 (primary hybrid/code), Head 2 (CRAG supervisor), Head 3a (semantic fallback), Head 3b (web fallback)
- `OllamaProvider` — built-in local LLM provider
- `VectorStoreAdapter` / `LLMProvider` protocols (runtime-checkable)
- `CRAGVerdict`, `RetrievalResult` typed dataclasses
- `_rrf_fuse` — Reciprocal Rank Fusion with weighted multi-source and tie-breaking
- `_sanitize_web_content` — strips scripts, styles, and excessive whitespace from HTML
- `HydRAGConfig.from_env()` — environment-variable config loading
- Zero required dependencies (stdlib-only core); `chromadb`, `firecrawl-py`, `dev` optional extras
- Apache-2.0 license
