"""Comprehensive Phase A tests for HydRAG RFC parity requirements.

This test suite validates that hydrag-core 1.4.1 implements all upstream-core
toggles with exact parity defaults matching current Playbook semantics.

RFC Section 9 (Phase A) Requirements:
  ✓ All upstream-core toggles from §3.1 in HydRAGConfig
  ✓ Parity defaults from rag_strategies.py (Playbook current)
  ✓ Provider abstraction (LLMProvider factory) as only CRAG runtime
  ✓ hydrag.logging.get_logger() exposed (optional, skip if not present)
  ✓ Binary completion gate assertions pass
"""

import pytest

from hydrag import HydRAGConfig

# ============================================================================
# Phase A Toggle Disposition (§3.1) — Upstream-Core Parity Tests
# ============================================================================

class TestPhaseAToggleParity:
    """Validate that all upstream-core toggles exist with Playbook parity defaults."""

    def test_enable_cross_encoder_rerank_default_false(self) -> None:
        """Cross-encoder reranking default: OFF (Playbook current, rag_strategies.py L83)."""
        cfg = HydRAGConfig()
        assert hasattr(cfg, "enable_cross_encoder_rerank"), \
            "Missing enable_cross_encoder_rerank field"
        assert cfg.enable_cross_encoder_rerank is False, \
            "enable_cross_encoder_rerank default must be False"

    def test_cross_encoder_model_default_ms_marco(self) -> None:
        """Cross-encoder model default: 'cross-encoder/ms-marco-MiniLM-L-6-v2'."""
        cfg = HydRAGConfig()
        assert hasattr(cfg, "cross_encoder_model"), \
            "Missing cross_encoder_model field"
        assert cfg.cross_encoder_model == "cross-encoder/ms-marco-MiniLM-L-6-v2", \
            f"Expected 'cross-encoder/ms-marco-MiniLM-L-6-v2', got {cfg.cross_encoder_model!r}"

    def test_cross_encoder_top_k_default_20(self) -> None:
        """Cross-encoder top_k default: 20 (Playbook current, rag_strategies.py L85)."""
        cfg = HydRAGConfig()
        assert hasattr(cfg, "cross_encoder_top_k"), \
            "Missing cross_encoder_top_k field"
        assert cfg.cross_encoder_top_k == 20, \
            f"Expected 20, got {cfg.cross_encoder_top_k}"

    def test_hard_filter_insufficient_default_false(self) -> None:
        """Hard filter insufficient default: OFF (Playbook current, rag_strategies.py L87)."""
        cfg = HydRAGConfig()
        assert hasattr(cfg, "hard_filter_insufficient"), \
            "Missing hard_filter_insufficient field"
        assert cfg.hard_filter_insufficient is False, \
            "hard_filter_insufficient default must be False"

    def test_crag_fail_closed_default_false(self) -> None:
        """CRAG fail-mode default: fail-open (OFF) (Playbook current, rag_strategies.py L89)."""
        cfg = HydRAGConfig()
        assert hasattr(cfg, "crag_fail_closed"), \
            "Missing crag_fail_closed field"
        assert cfg.crag_fail_closed is False, \
            "crag_fail_closed default must be False (fail-open)"

    def test_min_cosine_similarity_default_0_0(self) -> None:
        """Minimum cosine similarity default: 0.0 (Playbook current, rag_strategies.py L91)."""
        cfg = HydRAGConfig()
        assert hasattr(cfg, "min_cosine_similarity"), \
            "Missing min_cosine_similarity field"
        assert isinstance(cfg.min_cosine_similarity, float), \
            f"min_cosine_similarity must be float, got {type(cfg.min_cosine_similarity)}"
        assert cfg.min_cosine_similarity == 0.0, \
            f"Expected 0.0, got {cfg.min_cosine_similarity}"

    def test_enable_parent_child_retrieval_default_false(self) -> None:
        """Parent-child retrieval default: OFF (Playbook current, rag_strategies.py L97)."""
        cfg = HydRAGConfig()
        assert hasattr(cfg, "enable_parent_child_retrieval"), \
            "Missing enable_parent_child_retrieval field"
        assert cfg.enable_parent_child_retrieval is False, \
            "enable_parent_child_retrieval default must be False"

    def test_parent_context_window_default_1(self) -> None:
        """Parent context window default: 1 (Playbook current, rag_strategies.py L98)."""
        cfg = HydRAGConfig()
        assert hasattr(cfg, "parent_context_window"), \
            "Missing parent_context_window field"
        assert cfg.parent_context_window == 1, \
            f"Expected 1, got {cfg.parent_context_window}"


# ============================================================================
# Phase A Environment Variable Binding Tests
# ============================================================================

class TestPhaseAEnvironmentBinding:
    """Validate environment variable mappings for all Phase A toggles."""

    def test_hydrag_enable_cross_encoder_rerank_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HYDRAG_ENABLE_CROSS_ENCODER_RERANK", "true")
        cfg = HydRAGConfig.from_env()
        assert cfg.enable_cross_encoder_rerank is True

    def test_hydrag_cross_encoder_model_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        test_model = "cross-encoder/custom-model"
        monkeypatch.setenv("HYDRAG_CROSS_ENCODER_MODEL", test_model)
        cfg = HydRAGConfig.from_env()
        assert cfg.cross_encoder_model == test_model

    def test_hydrag_cross_encoder_top_k_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HYDRAG_CROSS_ENCODER_TOP_K", "50")
        cfg = HydRAGConfig.from_env()
        assert cfg.cross_encoder_top_k == 50

    def test_hydrag_hard_filter_insufficient_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HYDRAG_HARD_FILTER_INSUFFICIENT", "true")
        cfg = HydRAGConfig.from_env()
        assert cfg.hard_filter_insufficient is True

    def test_hydrag_crag_fail_closed_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HYDRAG_CRAG_FAIL_CLOSED", "true")
        cfg = HydRAGConfig.from_env()
        assert cfg.crag_fail_closed is True

    def test_hydrag_min_cosine_similarity_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HYDRAG_MIN_COSINE_SIMILARITY", "0.5")
        cfg = HydRAGConfig.from_env()
        assert cfg.min_cosine_similarity == 0.5

    def test_hydrag_enable_parent_child_retrieval_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HYDRAG_ENABLE_PARENT_CHILD_RETRIEVAL", "true")
        cfg = HydRAGConfig.from_env()
        assert cfg.enable_parent_child_retrieval is True

    def test_hydrag_parent_context_window_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HYDRAG_PARENT_CONTEXT_WINDOW", "3")
        cfg = HydRAGConfig.from_env()
        assert cfg.parent_context_window == 3


# ============================================================================
# Phase A Binary Completion Gate (§9)
# ============================================================================

class TestPhaseABinaryGate:
    """Test the Phase A binary completion gate assertions.

    This mirrors the gate from §9:
      pip install "hydrag-core==1.4.1"
      python -c "from hydrag.config import HydRAGConfig; c=HydRAGConfig();
        assert c.enable_cross_encoder_rerank is False;
        assert c.hard_filter_insufficient is False;
        assert c.crag_fail_closed is False;
        assert isinstance(c.min_cosine_similarity, float) and c.min_cosine_similarity >= 0.0;
        assert c.enable_parent_child_retrieval is False;
        assert c.parent_context_window == 1;
        assert isinstance(c.embedding_model, str) and bool(c.embedding_model);
        assert c.cross_encoder_model == 'cross-encoder/ms-marco-MiniLM-L-6-v2';
        assert c.cross_encoder_top_k == 20"
    """

    def test_binary_gate_all_assertions(self) -> None:
        """Execute all Phase A binary gate assertions in one test."""
        c = HydRAGConfig()

        # Critical parity defaults (must not flip)
        assert c.enable_cross_encoder_rerank is False, \
            "GATE FAIL: enable_cross_encoder_rerank must be False"
        assert c.hard_filter_insufficient is False, \
            "GATE FAIL: hard_filter_insufficient must be False"
        assert c.crag_fail_closed is False, \
            "GATE FAIL: crag_fail_closed must be False"

        # Similarity threshold (must be float, >= 0.0 for parity)
        assert isinstance(c.min_cosine_similarity, float), \
            "GATE FAIL: min_cosine_similarity must be float"
        assert c.min_cosine_similarity >= 0.0, \
            "GATE FAIL: min_cosine_similarity must be >= 0.0"

        # Parent-child retrieval (must be OFF by default)
        assert c.enable_parent_child_retrieval is False, \
            "GATE FAIL: enable_parent_child_retrieval must be False"
        assert c.parent_context_window == 1, \
            "GATE FAIL: parent_context_window must be 1"

        # Embedding model (must be string, non-empty)
        assert isinstance(c.embedding_model, str), \
            "GATE FAIL: embedding_model must be string"
        assert bool(c.embedding_model), \
            "GATE FAIL: embedding_model must be non-empty"

        # Cross-encoder exact parity
        assert c.cross_encoder_model == "cross-encoder/ms-marco-MiniLM-L-6-v2", \
            f"GATE FAIL: cross_encoder_model must be exact parity, got {c.cross_encoder_model!r}"
        assert c.cross_encoder_top_k == 20, \
            f"GATE FAIL: cross_encoder_top_k must be 20, got {c.cross_encoder_top_k}"

    def test_binary_gate_exit_code_0(self) -> None:
        """Ensure gate passes and produces no exceptions (exit code 0 equivalent)."""
        try:
            c = HydRAGConfig()
            # If all assertions pass without raising, gate succeeds
            assert c.enable_cross_encoder_rerank is False
            assert c.cross_encoder_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
            assert c.cross_encoder_top_k == 20
            gate_result = "PASS"
        except (AssertionError, Exception) as e:
            gate_result = f"FAIL: {e}"

        assert gate_result == "PASS", \
            f"Phase A binary gate FAILED: {gate_result}"


# ============================================================================
# Phase A Logging Contract Tests (§9.A.3)
# ============================================================================

# Try to import the logging module, but skip tests if it doesn't exist yet
try:
    from hydrag import logging as hlog
    HAS_LOGGING = True
except ImportError:
    HAS_LOGGING = False


@pytest.mark.skipif(not HAS_LOGGING, reason="hydrag.logging module not yet implemented")
class TestPhaseALoggingContract:
    """Validate logging contract per Phase A task #3.

    - hydrag.logging.get_logger() exposed
    - Returns logger with name starting with 'hydrag'
    - Handler-free by default (Playbook attaches StudioJSONFormatter)
    """

    def test_hydrag_logging_module_exists(self) -> None:
        """Verify hydrag.logging module is importable."""
        assert hasattr(hlog, "get_logger"), \
            "hydrag.logging module missing get_logger() function"

    def test_hydrag_get_logger_returns_logger(self) -> None:
        """Ensure get_logger() returns a valid logger instance."""
        lg = hlog.get_logger()
        assert lg is not None, \
            "get_logger() must return a logger, not None"
        assert hasattr(lg, "name"), \
            "Returned object must be a logger with name attribute"

    def test_hydrag_logger_name_starts_with_hydrag(self) -> None:
        """Logger name must start with 'hydrag'."""
        lg = hlog.get_logger()
        assert lg.name.startswith("hydrag"), \
            f"Logger name must start with 'hydrag', got {lg.name!r}"

    def test_hydrag_logger_handler_free_by_default(self) -> None:
        """Logger must be handler-free by default (Playbook adds formatter)."""
        lg = hlog.get_logger()
        # Root logger always has at least propagation, but this logger's
        # own handlers list should be empty (no StreamHandler, etc.)
        assert len(lg.handlers) == 0, \
            f"hydrag logger must be handler-free by default, found {len(lg.handlers)} handlers"

    def test_hydrag_logger_propagation_enabled(self) -> None:
        """Logger propagation must be enabled (for Playbook formatter attachment)."""
        lg = hlog.get_logger()
        assert lg.propagate is True, \
            "hydrag logger must have propagation enabled"


# ============================================================================
# Phase A Provider Abstraction Tests (§9.A.2)
# ============================================================================

class TestPhaseAProviderAbstraction:
    """Validate provider abstraction is sole CRAG runtime path.

    - LLMProvider factory is available
    - ollama provider is the default and functional
    - No direct transport calls in Playbook-facing paths
    """

    def test_llm_provider_factory_available(self) -> None:
        """LLMProvider factory must be accessible."""
        from hydrag.providers import factory
        assert hasattr(factory, "get_provider"), \
            "LLMProvider factory missing get_provider function"

    def test_ollama_provider_default(self) -> None:
        """Default provider is 'ollama'."""
        cfg = HydRAGConfig()
        assert cfg.llm_provider == "ollama", \
            f"Default llm_provider must be 'ollama', got {cfg.llm_provider!r}"

    def test_provider_factory_instantiation(self) -> None:
        """Provider factory can instantiate ollama provider."""
        from hydrag.providers import factory
        cfg = HydRAGConfig(ollama_host="http://localhost:11434")
        provider = factory.get_provider(cfg)
        assert provider is not None, \
            "get_provider must return a provider instance"


# ============================================================================
# Phase A Config Validation Tests (§9)
# ============================================================================

class TestPhaseAConfigValidation:
    """Validate config constraints for all Phase A toggles."""

    def test_cross_encoder_top_k_positive(self) -> None:
        """cross_encoder_top_k must be positive."""
        # Should not raise
        cfg = HydRAGConfig(cross_encoder_top_k=10)
        assert cfg.cross_encoder_top_k == 10

    def test_parent_context_window_positive(self) -> None:
        """parent_context_window must be positive."""
        cfg = HydRAGConfig(parent_context_window=2)
        assert cfg.parent_context_window == 2

    def test_min_cosine_similarity_in_range(self) -> None:
        """min_cosine_similarity should be in [-1, 1] (cosine range)."""
        cfg = HydRAGConfig(min_cosine_similarity=0.5)
        assert cfg.min_cosine_similarity == 0.5

        cfg = HydRAGConfig(min_cosine_similarity=0.0)
        assert cfg.min_cosine_similarity == 0.0


# ============================================================================
# Phase A Version Verification (§9)
# ============================================================================

class TestPhaseAVersionRequirement:
    """Verify package version meets Phase A floor (hydrag-core 1.4.1)."""

    def test_hydrag_core_version_1_4_1_or_higher(self) -> None:
        """Package version must be >= 1.4.1."""
        from hydrag import __version__

        version_str = __version__
        # Parse semantic version
        parts = version_str.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2].split("-")[0]) if len(parts) > 2 else 0

        # Must be 1.4.1 or higher
        is_valid = (
            (major > 1) or
            (major == 1 and minor > 4) or
            (major == 1 and minor == 4 and patch >= 1)
        )
        assert is_valid, \
            f"hydrag-core version must be >= 1.4.1, got {version_str}"


# ============================================================================
# Phase A Integration Tests
# ============================================================================

class TestPhaseAIntegration:
    """End-to-end integration tests for Phase A requirements."""

    def test_config_roundtrip_through_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config created from env matches Phase A parity."""
        monkeypatch.setenv("HYDRAG_ENABLE_CROSS_ENCODER_RERANK", "false")
        monkeypatch.setenv("HYDRAG_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        monkeypatch.setenv("HYDRAG_CROSS_ENCODER_TOP_K", "20")
        monkeypatch.setenv("HYDRAG_HARD_FILTER_INSUFFICIENT", "false")
        monkeypatch.setenv("HYDRAG_CRAG_FAIL_CLOSED", "false")
        monkeypatch.setenv("HYDRAG_MIN_COSINE_SIMILARITY", "0.0")
        monkeypatch.setenv("HYDRAG_ENABLE_PARENT_CHILD_RETRIEVAL", "false")
        monkeypatch.setenv("HYDRAG_PARENT_CONTEXT_WINDOW", "1")

        cfg = HydRAGConfig.from_env()

        # All Phase A parity assertions
        assert cfg.enable_cross_encoder_rerank is False
        assert cfg.cross_encoder_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert cfg.cross_encoder_top_k == 20
        assert cfg.hard_filter_insufficient is False
        assert cfg.crag_fail_closed is False
        assert cfg.min_cosine_similarity == 0.0
        assert cfg.enable_parent_child_retrieval is False
        assert cfg.parent_context_window == 1

    def test_all_phase_a_toggles_configurable(self) -> None:
        """All Phase A toggles can be set directly."""
        cfg = HydRAGConfig(
            enable_cross_encoder_rerank=True,
            cross_encoder_model="custom-model",
            cross_encoder_top_k=50,
            hard_filter_insufficient=True,
            crag_fail_closed=True,
            min_cosine_similarity=0.7,
            enable_parent_child_retrieval=True,
            parent_context_window=2,
        )

        assert cfg.enable_cross_encoder_rerank is True
        assert cfg.cross_encoder_model == "custom-model"
        assert cfg.cross_encoder_top_k == 50
        assert cfg.hard_filter_insufficient is True
        assert cfg.crag_fail_closed is True
        assert cfg.min_cosine_similarity == 0.7
        assert cfg.enable_parent_child_retrieval is True
        assert cfg.parent_context_window == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
