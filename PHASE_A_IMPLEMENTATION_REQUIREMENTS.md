---
id: PHASE_A_IMPL_REQUIREMENTS
ticket: T-1093
category: implementation
status: in-progress
created: '2025-01-08'
updated: '2025-01-08'
author: Claude
summary: Phase A field additions required for hydrag-core 1.4.1 upstream parity
keywords:
  implementation: 9
  phase-a: 8
  fields: 7
  config: 6
  parity: 5
---

# Phase A Implementation Requirements for hydrag-core 1.4.1

## Summary

Comprehensive hydrag-core test suite has been created to validate Phase A RFC requirements. Current status: **8 fields missing from HydRAGConfig** that must be added for Phase A completion.

## Current State Analysis

### Fields Present (38/46 shown)
- Core: `embedding_model`, `crag_mode`, `llm_provider`, `profile`
- Head controls: `enable_head_0`, `enable_head_1`, `enable_head_2_crag`, `enable_head_3a_semantic`, `enable_head_3b_web`
- Fast path: `enable_fast_path`, `fast_path_bm25_threshold`, `fast_path_confidence_threshold`
- Providers: `ollama_host`, `openai_compat_*`, `hf_*`, `surrealdb_*`
- Web fallback: `enable_web_fallback`, `allow_web_on_empty_primary`, `allow_markdown_in_web_fallback`

### Fields Missing (Phase A Requirements)
```
MISSING FIELD                    EXACT TYPE         DEFAULT VALUE              RFC SOURCE
==================================================================================
enable_cross_encoder_rerank      bool               False                      §3.1, upstream row 1
cross_encoder_model              str                "cross-encoder/ms-marco..." §3.1, upstream row 2
cross_encoder_top_k              int                20                         §3.1, upstream row 3
hard_filter_insufficient         bool               False                      §3.1, upstream row 4
crag_fail_closed                 bool               False                      §3.1, upstream row 5
min_cosine_similarity            float              0.0                        §3.1, upstream row 6
enable_parent_child_retrieval    bool               False                      §3.1, upstream row 10
parent_context_window            int                1                          §3.1, upstream row 11
```

## Implementation Steps

### Step 1: Add Fields to HydRAGConfig (src/hydrag/config.py)

**Location:** HydRAGConfig @dataclass

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class HydRAGConfig:
    # ... existing fields ...
    
    # Phase A Upstream-Core Additions (RFC §3.1)
    # Cross-encoder reranking (rows 1-3)
    enable_cross_encoder_rerank: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_top_k: int = 20
    
    # Result-quality defenses (rows 4, 6)
    hard_filter_insufficient: bool = False
    min_cosine_similarity: float = 0.0
    
    # Fail-mode policy (row 5)
    crag_fail_closed: bool = False
    
    # Parent-child retrieval (rows 10-11)
    enable_parent_child_retrieval: bool = False
    parent_context_window: int = 1
    
    # ... existing fields continue ...
```

**Constraints:**
- `enable_cross_encoder_rerank`: boolean flag only, no special validation
- `cross_encoder_model`: string, must match HuggingFace model identifier format
- `cross_encoder_top_k`: must be positive integer (1 ≤ value)
- `hard_filter_insufficient`: boolean flag only
- `crag_fail_closed`: boolean flag only
- `min_cosine_similarity`: float, range [0.0, ∞) for Playbook parity (cosine similarity semantics)
- `enable_parent_child_retrieval`: boolean flag only
- `parent_context_window`: positive integer (1 ≤ value), represents sibling window size

### Step 2: Update from_env() Classmethod (src/hydrag/config.py)

**Location:** HydRAGConfig.from_env() classmethod

Add these environment variable bindings (maintain existing logic):

```python
@classmethod
def from_env(cls) -> "HydRAGConfig":
    """Load configuration from environment variables.
    
    Phase A additions (RFC §9.B.1 precedence):
    - HYDRAG_ENABLE_CROSS_ENCODER_RERANK -> enable_cross_encoder_rerank (bool)
    - HYDRAG_CROSS_ENCODER_MODEL -> cross_encoder_model (str)
    - HYDRAG_CROSS_ENCODER_TOP_K -> cross_encoder_top_k (int)
    - HYDRAG_HARD_FILTER_INSUFFICIENT -> hard_filter_insufficient (bool)
    - HYDRAG_CRAG_FAIL_CLOSED -> crag_fail_closed (bool)
    - HYDRAG_MIN_COSINE_SIMILARITY -> min_cosine_similarity (float)
    - HYDRAG_ENABLE_PARENT_CHILD_RETRIEVAL -> enable_parent_child_retrieval (bool)
    - HYDRAG_PARENT_CONTEXT_WINDOW -> parent_context_window (int)
    """
    
    # Existing parsing logic (preserve all current env vars)
    # ...
    
    # NEW: Phase A environment parsing
    enable_cross_encoder_rerank = (
        os.getenv("HYDRAG_ENABLE_CROSS_ENCODER_RERANK", "false").lower() == "true"
    )
    
    cross_encoder_model = os.getenv(
        "HYDRAG_CROSS_ENCODER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    
    cross_encoder_top_k = int(
        os.getenv("HYDRAG_CROSS_ENCODER_TOP_K", "20")
    )
    
    hard_filter_insufficient = (
        os.getenv("HYDRAG_HARD_FILTER_INSUFFICIENT", "false").lower() == "true"
    )
    
    crag_fail_closed = (
        os.getenv("HYDRAG_CRAG_FAIL_CLOSED", "false").lower() == "true"
    )
    
    min_cosine_similarity = float(
        os.getenv("HYDRAG_MIN_COSINE_SIMILARITY", "0.0")
    )
    
    enable_parent_child_retrieval = (
        os.getenv("HYDRAG_ENABLE_PARENT_CHILD_RETRIEVAL", "false").lower() == "true"
    )
    
    parent_context_window = int(
        os.getenv("HYDRAG_PARENT_CONTEXT_WINDOW", "1")
    )
    
    # Return instance with all fields (existing + Phase A)
    return cls(
        # ... existing fields ...
        enable_cross_encoder_rerank=enable_cross_encoder_rerank,
        cross_encoder_model=cross_encoder_model,
        cross_encoder_top_k=cross_encoder_top_k,
        hard_filter_insufficient=hard_filter_insufficient,
        crag_fail_closed=crag_fail_closed,
        min_cosine_similarity=min_cosine_similarity,
        enable_parent_child_retrieval=enable_parent_child_retrieval,
        parent_context_window=parent_context_window,
    )
```

**Environment Variable Precedence (RFC §9.B.1):**
1. Explicit constructor argument (highest)
2. Environment variable (HYDRAG_*)
3. Default in @dataclass (lowest)

### Step 3: Add __post_init__ Validation (if not present)

**Location:** HydRAGConfig.__post_init__()

```python
def __post_init__(self) -> None:
    """Validate config constraints."""
    
    # Existing validation logic
    # ...
    
    # Phase A validation (RFC §9)
    
    # cross_encoder_top_k must be positive
    if self.cross_encoder_top_k <= 0:
        raise ValueError(
            f"cross_encoder_top_k must be positive, got {self.cross_encoder_top_k}"
        )
    
    # parent_context_window must be positive
    if self.parent_context_window <= 0:
        raise ValueError(
            f"parent_context_window must be positive, got {self.parent_context_window}"
        )
    
    # min_cosine_similarity should be >= 0 for cosine similarity semantics
    if self.min_cosine_similarity < 0.0:
        raise ValueError(
            f"min_cosine_similarity must be >= 0.0 (cosine range), got {self.min_cosine_similarity}"
        )
```

### Step 4: Create hydrag/logging.py Module (Optional for Phase A, Required for Phase B)

**Location:** src/hydrag/logging.py

```python
"""Logging utilities for HydRAG applications.

This module provides logging infrastructure that is designed to work with
Playbook's StudioJSONFormatter for structured logging.
"""

import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger for HydRAG components.
    
    Returns a logger with:
    - Name starting with 'hydrag'
    - Handler-free by default (Playbook attaches formatter)
    - Propagation enabled (for parent/root handler attachment)
    
    Args:
        name: Optional logger name suffix. Full name: f"hydrag.{name}"
        
    Returns:
        logging.Logger configured for HydRAG
    """
    if name:
        full_name = f"hydrag.{name}"
    else:
        full_name = "hydrag"
    
    logger = logging.getLogger(full_name)
    
    # Ensure handler-free and propagation enabled
    logger.handlers = []
    logger.propagate = True
    
    return logger
```

**Export in src/hydrag/__init__.py:**

```python
# Add to __init__.py
from . import logging  # Make module accessible as hydrag.logging
```

### Step 5: Update pyproject.toml Version

**Location:** pyproject.toml

```toml
[project]
name = "hydrag-core"
version = "1.4.1"  # Bump from 1.4.0
description = "HydRAG core search engine with upstream parity for Playbook"
# ... rest of metadata ...
```

## Testing & Validation

### Run Full Phase A Test Suite

```bash
cd /Users/user/ll/hydrag-core

# Run all Phase A tests
python -m pytest tests/test_phase_a_requirements.py \
                 tests/test_phase_a_critical_c2_c3.py \
                 tests/test_phase_a_features_reranking_parent_child.py \
    -v --tb=short

# Expected: 89 tests pass with 0 failures
```

### Run Binary Gate Test Only

```bash
python -m pytest tests/test_phase_a_requirements.py::TestPhaseABinaryGate -v

# Expected: 2 tests pass (test_binary_gate_all_assertions, test_binary_gate_exit_code_0)
```

### Manual Verification

```bash
python -c "
from hydrag.config import HydRAGConfig

# Create config with defaults
c = HydRAGConfig()

# Verify all Phase A fields exist and have correct defaults
assert c.enable_cross_encoder_rerank is False
assert c.cross_encoder_model == 'cross-encoder/ms-marco-MiniLM-L-6-v2'
assert c.cross_encoder_top_k == 20
assert c.hard_filter_insufficient is False
assert c.crag_fail_closed is False
assert c.min_cosine_similarity == 0.0
assert c.enable_parent_child_retrieval is False
assert c.parent_context_window == 1

print('✓ All Phase A defaults verified')

# Test env var override
import os
os.environ['HYDRAG_ENABLE_CROSS_ENCODER_RERANK'] = 'true'
os.environ['HYDRAG_MIN_COSINE_SIMILARITY'] = '0.5'

c_env = HydRAGConfig.from_env()
assert c_env.enable_cross_encoder_rerank is True
assert c_env.min_cosine_similarity == 0.5

print('✓ All Phase A env var bindings verified')
"
```

## Test Coverage Summary

### test_phase_a_requirements.py (39 tests)
- ✓ Toggle parity: 8 tests (one per field)
- ✓ Environment binding: 8 tests (one per env var)
- ✓ Binary completion gate: 2 tests (assertions + exit code)
- ✓ Logging contract: 5 tests (if hydrag.logging implemented)
- ✓ Provider abstraction: 3 tests
- ✓ Config validation: 3 tests
- ✓ Version requirement: 1 test
- ✓ Integration: 2 tests

### test_phase_a_critical_c2_c3.py (22 tests)
- CRYT-C2 (Result Quality Defense): 8 tests
  - hard_filter_insufficient blocks low-confidence results
  - min_cosine_similarity filters unrelated results
- CRYT-C3 (Fail-Mode Policy): 8 tests
  - crag_fail_closed enables fail-closed behavior
  - Provider timeouts respect policy
- Integration: 6 tests

### test_phase_a_features_reranking_parent_child.py (28 tests)
- Cross-encoder reranking: 12 tests
- Parent-child retrieval: 12 tests
- Feature interactions: 4 tests

**Total: 89 comprehensive tests**

## Implementation Checklist

- [ ] Add 8 Phase A fields to HydRAGConfig @dataclass
- [ ] Update from_env() with 8 new environment variable bindings
- [ ] Add __post_init__() validation for new fields
- [ ] Create hydrag/logging.py module (optional for Phase A, required for Phase B)
- [ ] Export logging module in src/hydrag/__init__.py
- [ ] Update pyproject.toml version to 1.4.1
- [ ] Run full Phase A test suite (target: 89/89 pass)
- [ ] Run binary gate test (target: 2/2 pass)
- [ ] Capture exit code 0 evidence
- [ ] Document any implementation gaps or deviations
- [ ] Publish hydrag-core 1.4.1 to PyPI (when all tests pass)

## Timeline & Blockers

**Phase A Completion Gate:** All 89 tests pass with exit code 0

**Blocking Issues:**
- Missing fields in HydRAGConfig (8 required additions)
- Missing hydrag.logging module (optional, but recommended)
- Version not bumped in pyproject.toml

**Next Action:** Implement fields in src/hydrag/config.py and run Phase A test suite

## References

- **RFC:** docs/rfcs/HYDRAG_PYPI_HARD_CUTOVER_MIGRATION_RFC-GPT-5.3-Codex.md
- **Upstream Core Table:** RFC §3.1, rows 1-11
- **Parity Defaults Source:** tools/assistant/src/assistant/tools/rag_strategies.py
- **Binary Gate:** RFC §9, lines 251-265
- **Test Files:** tests/test_phase_a_*.py (3 files, 89 tests total)
