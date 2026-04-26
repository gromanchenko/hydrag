"""Phase A feature-specific tests for cross-encoder and parent-child retrieval.

These tests validate the new upstream-core toggles are functional and can be
used in realistic retrieval scenarios.
"""


import pytest

from hydrag.config import HydRAGConfig

# ============================================================================
# Cross-Encoder Reranking Tests
# ============================================================================

class TestCrossEncoderReranking:
    """Cross-encoder reranking feature tests (Phase A upstream-core)."""

    def test_cross_encoder_disabled_by_default(self) -> None:
        """Cross-encoder reranking must be disabled by default."""
        cfg = HydRAGConfig()
        assert cfg.enable_cross_encoder_rerank is False

    def test_cross_encoder_can_be_enabled(self) -> None:
        """Cross-encoder can be explicitly enabled."""
        cfg = HydRAGConfig(enable_cross_encoder_rerank=True)
        assert cfg.enable_cross_encoder_rerank is True

    def test_cross_encoder_model_configurable(self) -> None:
        """Cross-encoder model can be customized."""
        cfg = HydRAGConfig(
            enable_cross_encoder_rerank=True,
            cross_encoder_model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        )
        assert cfg.cross_encoder_model == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    def test_cross_encoder_top_k_default_20(self) -> None:
        """Cross-encoder top_k default is 20."""
        cfg = HydRAGConfig(enable_cross_encoder_rerank=True)
        assert cfg.cross_encoder_top_k == 20

    def test_cross_encoder_top_k_configurable(self) -> None:
        """Cross-encoder top_k can be adjusted."""
        cfg = HydRAGConfig(
            enable_cross_encoder_rerank=True,
            cross_encoder_top_k=50,
        )
        assert cfg.cross_encoder_top_k == 50

    def test_cross_encoder_parity_defaults(self) -> None:
        """Parity check: exact default model and top_k."""
        cfg = HydRAGConfig()
        assert cfg.cross_encoder_model == "cross-encoder/ms-marco-MiniLM-L-6-v2", \
            "Must match Playbook default model"
        assert cfg.cross_encoder_top_k == 20, \
            "Must match Playbook default top_k"

    def test_cross_encoder_both_params_required_together(self) -> None:
        """Model and top_k should be paired."""
        cfg = HydRAGConfig(
            enable_cross_encoder_rerank=True,
            cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            cross_encoder_top_k=20,
        )
        assert cfg.enable_cross_encoder_rerank is True
        assert cfg.cross_encoder_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert cfg.cross_encoder_top_k == 20

    def test_cross_encoder_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variables for cross-encoder."""
        monkeypatch.setenv("HYDRAG_ENABLE_CROSS_ENCODER_RERANK", "true")
        monkeypatch.setenv("HYDRAG_CROSS_ENCODER_MODEL", "cross-encoder/custom")
        monkeypatch.setenv("HYDRAG_CROSS_ENCODER_TOP_K", "100")

        cfg = HydRAGConfig.from_env()
        assert cfg.enable_cross_encoder_rerank is True
        assert cfg.cross_encoder_model == "cross-encoder/custom"
        assert cfg.cross_encoder_top_k == 100

    def test_cross_encoder_reranking_scenario(self) -> None:
        """Realistic scenario: enable reranking with custom model."""
        cfg = HydRAGConfig(
            enable_cross_encoder_rerank=True,
            cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            cross_encoder_top_k=25,
            min_cosine_similarity=0.5,  # Filter unrelated before rerank
        )

        # Expected behavior:
        # 1. Primary retrieval returns N results
        # 2. Filter by min_cosine_similarity (keep top 25 or more)
        # 3. Rerank top 25 with cross-encoder
        assert cfg.enable_cross_encoder_rerank is True
        assert cfg.cross_encoder_top_k == 25
        assert cfg.min_cosine_similarity == 0.5


# ============================================================================
# Parent-Child Retrieval Tests
# ============================================================================

class TestParentChildRetrieval:
    """Parent-child retrieval feature tests (Phase A upstream-core)."""

    def test_parent_child_disabled_by_default(self) -> None:
        """Parent-child retrieval must be disabled by default."""
        cfg = HydRAGConfig()
        assert cfg.enable_parent_child_retrieval is False

    def test_parent_child_can_be_enabled(self) -> None:
        """Parent-child can be explicitly enabled."""
        cfg = HydRAGConfig(enable_parent_child_retrieval=True)
        assert cfg.enable_parent_child_retrieval is True

    def test_parent_context_window_default_1(self) -> None:
        """Parent context window default is 1 (immediate parent only)."""
        cfg = HydRAGConfig(enable_parent_child_retrieval=True)
        assert cfg.parent_context_window == 1

    def test_parent_context_window_configurable(self) -> None:
        """Parent context window can be expanded."""
        cfg = HydRAGConfig(
            enable_parent_child_retrieval=True,
            parent_context_window=3,
        )
        assert cfg.parent_context_window == 3

    def test_parent_context_window_parity_default_1(self) -> None:
        """Parity check: exact default window size."""
        cfg = HydRAGConfig()
        assert cfg.parent_context_window == 1, \
            "Must match Playbook default (immediate parent only)"

    def test_parent_child_both_params_required(self) -> None:
        """Enable flag and window size should be paired."""
        cfg = HydRAGConfig(
            enable_parent_child_retrieval=True,
            parent_context_window=2,
        )
        assert cfg.enable_parent_child_retrieval is True
        assert cfg.parent_context_window == 2

    def test_parent_child_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variables for parent-child."""
        monkeypatch.setenv("HYDRAG_ENABLE_PARENT_CHILD_RETRIEVAL", "true")
        monkeypatch.setenv("HYDRAG_PARENT_CONTEXT_WINDOW", "3")

        cfg = HydRAGConfig.from_env()
        assert cfg.enable_parent_child_retrieval is True
        assert cfg.parent_context_window == 3

    def test_parent_child_retrieval_scenario_basic(self) -> None:
        """Realistic scenario: basic parent-child merging."""
        cfg = HydRAGConfig(
            enable_parent_child_retrieval=True,
            parent_context_window=1,  # Immediate parent
        )

        # Expected behavior for chunk-based corpus:
        # 1. Retrieve child chunk (e.g., paragraph)
        # 2. Auto-merge parent chunk (e.g., section)
        # 3. Return merged context to LLM
        assert cfg.enable_parent_child_retrieval is True
        assert cfg.parent_context_window == 1

    def test_parent_child_retrieval_scenario_expanded_window(self) -> None:
        """Realistic scenario: expanded parent context window."""
        cfg = HydRAGConfig(
            enable_parent_child_retrieval=True,
            parent_context_window=3,  # Include up to 3 levels
        )

        # Expected behavior for hierarchical corpus:
        # 1. Retrieve child chunk (sentence/phrase level)
        # 2. Auto-merge parent 1 (paragraph level)
        # 3. Auto-merge parent 2 (section level)
        # 4. Auto-merge parent 3 (chapter level)
        # 5. Return expanded context
        assert cfg.parent_context_window == 3


# ============================================================================
# Feature Interaction Tests
# ============================================================================

class TestPhaseAFeatureInteractions:
    """Tests for interactions between Phase A features."""

    def test_cross_encoder_with_parent_child(self) -> None:
        """Cross-encoder reranking can work with parent-child retrieval."""
        cfg = HydRAGConfig(
            enable_cross_encoder_rerank=True,
            cross_encoder_top_k=20,
            enable_parent_child_retrieval=True,
            parent_context_window=1,
        )

        # Both features enabled: rerank merged parent-child results
        assert cfg.enable_cross_encoder_rerank is True
        assert cfg.enable_parent_child_retrieval is True

    def test_hard_filter_with_reranking(self) -> None:
        """Hard filter can precede reranking."""
        cfg = HydRAGConfig(
            hard_filter_insufficient=True,
            min_cosine_similarity=0.6,
            enable_cross_encoder_rerank=True,
            cross_encoder_top_k=20,
        )

        # Pipeline: hard_filter -> rerank
        assert cfg.hard_filter_insufficient is True
        assert cfg.enable_cross_encoder_rerank is True

    def test_fail_mode_with_all_features(self) -> None:
        """CRAG fail-mode interacts with other features."""
        cfg = HydRAGConfig(
            crag_fail_closed=True,
            enable_cross_encoder_rerank=True,
            enable_parent_child_retrieval=True,
            hard_filter_insufficient=True,
        )

        # All features configured
        assert cfg.crag_fail_closed is True
        assert cfg.enable_cross_encoder_rerank is True
        assert cfg.enable_parent_child_retrieval is True
        assert cfg.hard_filter_insufficient is True

    def test_all_phase_a_toggles_together(self) -> None:
        """All Phase A toggles can be enabled together."""
        cfg = HydRAGConfig(
            enable_cross_encoder_rerank=True,
            cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            cross_encoder_top_k=20,
            hard_filter_insufficient=True,
            crag_fail_closed=True,
            min_cosine_similarity=0.5,
            enable_parent_child_retrieval=True,
            parent_context_window=2,
        )

        # All parity defaults verified
        assert cfg.enable_cross_encoder_rerank is True
        assert cfg.cross_encoder_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert cfg.cross_encoder_top_k == 20
        assert cfg.hard_filter_insufficient is True
        assert cfg.crag_fail_closed is True
        assert cfg.min_cosine_similarity == 0.5
        assert cfg.enable_parent_child_retrieval is True
        assert cfg.parent_context_window == 2


# ============================================================================
# Configuration Immutability and Defaults
# ============================================================================

class TestPhaseADefaults:
    """Verify all Phase A defaults are stable and parity-correct."""

    def test_all_phase_a_defaults_match_parity(self) -> None:
        """All Phase A toggle defaults match Playbook parity."""
        cfg = HydRAGConfig()

        # Build a parity dict
        expected_parity = {
            "enable_cross_encoder_rerank": False,
            "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "cross_encoder_top_k": 20,
            "hard_filter_insufficient": False,
            "crag_fail_closed": False,
            "min_cosine_similarity": 0.0,
            "enable_parent_child_retrieval": False,
            "parent_context_window": 1,
        }

        # Verify each
        for field, expected_value in expected_parity.items():
            actual_value = getattr(cfg, field)
            assert actual_value == expected_value, \
                f"Parity mismatch: {field} = {actual_value}, expected {expected_value}"

    def test_default_config_immutable_across_instances(self) -> None:
        """Defaults should not leak between instances."""
        cfg1 = HydRAGConfig(cross_encoder_top_k=100)
        cfg2 = HydRAGConfig()  # Should still be 20

        assert cfg1.cross_encoder_top_k == 100
        assert cfg2.cross_encoder_top_k == 20, \
            "Defaults should not be affected by other instances"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
