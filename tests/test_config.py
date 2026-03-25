"""Tests for HydRAGConfig."""

import os

import pytest

from hydrag import HydRAGConfig


class TestHydRAGConfigDefaults:
    def test_default_profile_is_prose(self) -> None:
        cfg = HydRAGConfig()
        assert cfg.profile == "prose"

    def test_default_web_fallback_disabled(self) -> None:
        cfg = HydRAGConfig()
        assert cfg.enable_web_fallback is False
        assert cfg.allow_web_on_empty_primary is False

    def test_default_rrf_k(self) -> None:
        cfg = HydRAGConfig()
        assert cfg.rrf_k == 60

    def test_default_head_weights(self) -> None:
        cfg = HydRAGConfig()
        assert cfg.rrf_head_weights["head_1a"] == 1.5
        assert cfg.rrf_head_weights["head_1b"] == 1.0
        assert cfg.rrf_head_weights["head_3a"] == 1.0
        assert cfg.rrf_head_weights["head_3b"] == 0.8

    def test_code_profile(self) -> None:
        cfg = HydRAGConfig(profile="code")
        assert cfg.profile == "code"


class TestHydRAGConfigFromEnv:
    def test_profile_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HYDRAG_PROFILE", "code")
        cfg = HydRAGConfig.from_env()
        assert cfg.profile == "code"

    def test_web_fallback_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HYDRAG_ENABLE_WEB_FALLBACK", "true")
        cfg = HydRAGConfig.from_env()
        assert cfg.enable_web_fallback is True

    def test_rrf_k_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HYDRAG_RRF_K", "100")
        cfg = HydRAGConfig.from_env()
        assert cfg.rrf_k == 100

    def test_fallback_timeout_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HYDRAG_FALLBACK_TIMEOUT_S", "2.5")
        cfg = HydRAGConfig.from_env()
        assert cfg.fallback_timeout_s == 2.5

    def test_ollama_host_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_HOST", "http://myhost:11434")
        cfg = HydRAGConfig.from_env()
        assert cfg.ollama_host == "http://myhost:11434"

    def test_defaults_without_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear any HYDRAG_ vars that might leak from environment
        for key in list(os.environ):
            if key.startswith("HYDRAG_") or key in ("CRAG_MODEL", "CRAG_TIMEOUT", "OLLAMA_HOST"):
                monkeypatch.delenv(key, raising=False)
        cfg = HydRAGConfig.from_env()
        assert cfg.profile == "prose"
        assert cfg.crag_model == "qwen3:4b"
        assert cfg.embedding_model == "Alibaba-NLP/gte-Qwen2-7B-instruct"
        assert cfg.ollama_host == "http://localhost:11434"


class TestHydRAGConfigFastPath:
    """V2.2: Head 0 BM25 fast-path config (§3.0 spec)."""

    def test_defaults_enabled(self) -> None:
        cfg = HydRAGConfig()
        assert cfg.enable_fast_path is True
        assert cfg.fast_path_bm25_threshold == 0.6
        assert cfg.fast_path_confidence_threshold == 0.7
        assert cfg.crag_stream is True

    def test_direct_construction(self) -> None:
        cfg = HydRAGConfig(enable_fast_path=True, fast_path_bm25_threshold=0.75)
        assert cfg.enable_fast_path is True
        assert cfg.fast_path_bm25_threshold == 0.75

    def test_fast_path_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HYDRAG_ENABLE_FAST_PATH", "true")
        monkeypatch.setenv("HYDRAG_FAST_PATH_BM25_THRESHOLD", "0.8")
        cfg = HydRAGConfig.from_env()
        assert cfg.enable_fast_path is True
        assert cfg.fast_path_bm25_threshold == 0.8


class TestHydRAGConfigPerHead:
    """Per-head enable/disable switches."""

    def test_head_defaults(self) -> None:
        cfg = HydRAGConfig()
        assert cfg.enable_head_0 is True
        assert cfg.enable_head_1 is True
        assert cfg.enable_head_2_crag is True
        assert cfg.enable_head_3a_semantic is True
        assert cfg.enable_head_3b_web is False

    def test_disable_head_via_constructor(self) -> None:
        cfg = HydRAGConfig(enable_head_0=False, enable_head_2_crag=False)
        assert cfg.enable_head_0 is False
        assert cfg.enable_head_2_crag is False
        assert cfg.enable_head_1 is True

    def test_head_from_env(self) -> None:
        os.environ["HYDRAG_ENABLE_HEAD_0"] = "false"
        os.environ["HYDRAG_ENABLE_HEAD_3B_WEB"] = "true"
        try:
            cfg = HydRAGConfig.from_env()
            assert cfg.enable_head_0 is False
            assert cfg.enable_head_3b_web is True
        finally:
            del os.environ["HYDRAG_ENABLE_HEAD_0"]
            del os.environ["HYDRAG_ENABLE_HEAD_3B_WEB"]

    def test_crag_mode_default(self) -> None:
        cfg = HydRAGConfig()
        assert cfg.crag_mode == "auto"
        assert cfg.crag_classifier_path == ""

    def test_crag_mode_from_env(self) -> None:
        os.environ["HYDRAG_CRAG_MODE"] = "classifier"
        os.environ["HYDRAG_CRAG_CLASSIFIER_PATH"] = "/tmp/model"
        try:
            cfg = HydRAGConfig.from_env()
            assert cfg.crag_mode == "classifier"
            assert cfg.crag_classifier_path == "/tmp/model"
        finally:
            del os.environ["HYDRAG_CRAG_MODE"]
            del os.environ["HYDRAG_CRAG_CLASSIFIER_PATH"]
