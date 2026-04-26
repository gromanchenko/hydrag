"""Phase A Comprehensive Testing Guide

This guide provides step-by-step instructions to validate hydrag-core 1.4.1
against all Phase A RFC requirements.

Test Coverage Summary
====================
- Phase A Requirements (39 tests): test_phase_a_requirements.py
- Critical Regressions C2/C3 (22 tests): test_phase_a_critical_c2_c3.py
- Feature Tests (28 tests): test_phase_a_features_reranking_parent_child.py
- TOTAL: 89 tests covering all upstream-core parity requirements

Prerequisites for Phase A Success
==================================

1. Hydrag-Core Config (src/hydrag/config.py)
   Add these fields to HydRAGConfig @dataclass:
   
   # Cross-encoder reranking (Phase A upstream-core rows 1-3)
   enable_cross_encoder_rerank: bool = False
   cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
   cross_encoder_top_k: int = 20
   
   # Result-quality defenses (Phase A upstream-core rows 4, 6)
   hard_filter_insufficient: bool = False
   min_cosine_similarity: float = 0.0
   
   # Fail-mode policy (Phase A upstream-core row 5)
   crag_fail_closed: bool = False
   
   # Parent-child retrieval (Phase A upstream-core rows 10-11)
   enable_parent_child_retrieval: bool = False
   parent_context_window: int = 1

2. Environment Variable Binding (src/hydrag/config.py::from_env)
   Update from_env() to read:
   - HYDRAG_ENABLE_CROSS_ENCODER_RERANK -> enable_cross_encoder_rerank
   - HYDRAG_CROSS_ENCODER_MODEL -> cross_encoder_model
   - HYDRAG_CROSS_ENCODER_TOP_K -> cross_encoder_top_k
   - HYDRAG_HARD_FILTER_INSUFFICIENT -> hard_filter_insufficient
   - HYDRAG_CRAG_FAIL_CLOSED -> crag_fail_closed
   - HYDRAG_MIN_COSINE_SIMILARITY -> min_cosine_similarity
   - HYDRAG_ENABLE_PARENT_CHILD_RETRIEVAL -> enable_parent_child_retrieval
   - HYDRAG_PARENT_CONTEXT_WINDOW -> parent_context_window

3. Logging Contract (src/hydrag/logging.py)
   Ensure hydrag.logging module exists with:
   - get_logger() function
   - Returns logger with name starting with "hydrag"
   - Logger is handler-free by default
   - Propagation enabled for Playbook formatter attachment

Running the Tests
=================

1. Full test suite (all 89 tests):
   
   $ python -m pytest tests/test_phase_a_requirements.py \\
                     tests/test_phase_a_critical_c2_c3.py \\
                     tests/test_phase_a_features_reranking_parent_child.py \\
     -v --tb=short

2. Only Phase A requirements (39 tests):
   
   $ python -m pytest tests/test_phase_a_requirements.py -v

3. Only critical regression tests (22 tests):
   
   $ python -m pytest tests/test_phase_a_critical_c2_c3.py -v

4. Only feature tests (28 tests):
   
   $ python -m pytest tests/test_phase_a_features_reranking_parent_child.py -v

5. Binary completion gate ONLY:
   
   $ python -m pytest tests/test_phase_a_requirements.py::TestPhaseABinaryGate -v

6. Run tests and generate coverage report:
   
   $ python -m pytest tests/test_phase_a_*.py \\
     --cov=hydrag.config \\
     --cov=hydrag.core \\
     --cov-report=html

Expected Test Results
=====================

Passing Indicators
------------------
✓ All 89 tests pass with exit code 0
✓ No warnings or deprecation alerts
✓ Binary completion gate passes (TestPhaseABinaryGate.test_binary_gate_all_assertions)
✓ All parity defaults match Playbook (0.0 distance)
✓ Logging contract validated
✓ Version requirement met (hydrag-core >= 1.4.1)

Test Execution Order
====================

Phase 1: Config Validation (10 min)
  1. Add fields to HydRAGConfig
  2. Run: pytest tests/test_phase_a_requirements.py::TestPhaseAToggleParity
  3. Fix: Any missing fields or wrong defaults

Phase 2: Environment Binding (10 min)
  1. Update from_env() in config.py
  2. Run: pytest tests/test_phase_a_requirements.py::TestPhaseAEnvironmentBinding
  3. Fix: Any env var mapping issues

Phase 3: Binary Gate (5 min)
  1. Run: pytest tests/test_phase_a_requirements.py::TestPhaseABinaryGate
  2. Must pass: gate exits 0, all assertions green
  3. This gate is blocking for Phase B

Phase 4: Logging (5 min)
  1. Verify src/hydrag/logging.py exists and exports get_logger()
  2. Run: pytest tests/test_phase_a_requirements.py::TestPhaseALoggingContract
  3. Logger must be handler-free, propagation enabled

Phase 5: Critical Regressions (10 min)
  1. Implement hard_filter_insufficient logic in CRAG supervisor
  2. Implement min_cosine_similarity filtering in semantic fallback
  3. Implement crag_fail_closed behavior in supervisor
  4. Run: pytest tests/test_phase_a_critical_c2_c3.py
  5. All 22 tests must pass

Phase 6: Features (10 min)
  1. Implement cross-encoder invocation (when enable_cross_encoder_rerank=True)
  2. Implement parent-child merging (when enable_parent_child_retrieval=True)
  3. Run: pytest tests/test_phase_a_features_reranking_parent_child.py
  4. All 28 tests must pass

Phase 7: Full Suite (5 min)
  1. Run: pytest tests/test_phase_a_*.py -v
  2. All 89 tests must pass
  3. Coverage >= 85% on updated code

Coverage Expectations by Module
===============================

src/hydrag/config.py
  - HydRAGConfig class: 100% (all fields tested)
  - from_env() method: 100% (all env vars tested)
  - __post_init__() validation: 95%+ (new fields validated)

src/hydrag/core.py
  - CRAG supervisor: 80%+ (fail-closed, hard-filter logic)
  - Semantic fallback: 80%+ (min_cosine_similarity filtering)
  - hydrag_search: 85%+ (cross-encoder, parent-child paths)

src/hydrag/logging.py
  - get_logger(): 100% (returns logger, name, propagation)

Test Debugging Tips
===================

If binary_gate test fails:
  1. Check each assertion individually
  2. Print actual vs expected values
  3. Verify dataclass fields are spelled correctly
  4. Ensure defaults are NOT aspirational (use actual Playbook values)

If hard_filter test fails:
  1. Check min_cosine_similarity is correctly used as filter threshold
  2. Verify filtering is OFF when min_cosine_similarity == 0.0
  3. Ensure hard_filter_insufficient blocks INSUFFICIENT results (low confidence)

If crag_fail_closed test fails:
  1. Check CRAG supervisor respects crag_fail_closed flag
  2. When True, exceptions should propagate (not fall back)
  3. When False, exceptions should allow fallback to continue

If parent_child test fails:
  1. Verify parent_context_window > 0 is valid
  2. Check merging logic respects window size
  3. Ensure merged results don't exceed context limits

Continuous Integration
======================

Add to CI/CD pipeline:

.github/workflows/phase-a-tests.yml
---
name: Phase A Comprehensive Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -e ".[dev]"
      - run: python -m pytest tests/test_phase_a_*.py -v --tb=short
      - run: python -m pytest tests/test_phase_a_requirements.py::TestPhaseABinaryGate::test_binary_gate_exit_code_0

Release Gate
============

Before publishing hydrag-core 1.4.1 to PyPI:

1. Run full test suite:
   $ python -m pytest tests/test_phase_a_*.py -v --tb=short

2. Verify binary gate PASSES:
   $ python -m pytest tests/test_phase_a_requirements.py::TestPhaseABinaryGate::test_binary_gate_exit_code_0

3. Capture version:
   $ python -c "import importlib.metadata; print(importlib.metadata.version('hydrag-core'))"
   Expected: 1.4.1 or higher

4. Record as Phase A completion evidence (T-1093):
   Command: python -m pytest tests/test_phase_a_*.py -v
   Exit code: 0
   Tests: 89/89 passed
   Binary gate: PASS
   Version: hydrag-core==1.4.1

Playbook Integration (Phase B)
==============================

Once hydrag-core 1.4.1 passes all Phase A tests:

1. Update playbook pyproject.toml:
   hydrag-core>=1.4.1,<1.5.0

2. Update playbook versions.lock with PyPI sha256

3. Run Playbook Phase B wrapper tests:
   $ pytest tools/assistant/tests/test_hydrag_compat.py -v
   $ pytest tools/assistant/tests/test_hydrag_config_field_mapping.py -v

4. Run Playbook Phase B integration:
   $ pytest tools/assistant/tests/test_hydrag_package_integration.py -v

5. Run Playbook CD gates:
   $ make cd-build && make cd-deploy && make sanity-ui

Resources
=========

- RFC: docs/rfcs/HYDRAG_PYPI_HARD_CUTOVER_MIGRATION_RFC-GPT-5.3-Codex.md
- Playbook config parity source: tools/assistant/src/assistant/tools/rag_strategies.py
- Phase A gate: §9 of RFC
- Phase B gate: §10 of RFC
"""

# Quick Start
if __name__ == "__main__":
    import subprocess
    import sys
    
    print("Phase A Comprehensive Test Execution")
    print("=" * 60)
    
    tests = [
        "tests/test_phase_a_requirements.py",
        "tests/test_phase_a_critical_c2_c3.py",
        "tests/test_phase_a_features_reranking_parent_child.py",
    ]
    
    print(f"Running {len(tests)} test files...")
    cmd = ["python", "-m", "pytest"] + tests + ["-v", "--tb=short"]
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✓ All Phase A tests PASSED")
        print("✓ Ready for Phase B (Playbook integration)")
    else:
        print(f"\n✗ Tests FAILED with exit code {result.returncode}")
        sys.exit(result.returncode)
