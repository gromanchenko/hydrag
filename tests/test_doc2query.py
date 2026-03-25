"""Tests for Doc2Query V2 adaptive n, smart_truncate, config, and augmentation cache.

T-740: Integrated from hydrag-benchmark into hydrag-core.
"""

from __future__ import annotations

from pathlib import Path

from hydrag.doc2query import (
    AugmentationCache,
    Doc2QueryConfig,
    Doc2QueryGenerator,
    compute_adaptive_n,
    smart_truncate,
)

# ── compute_adaptive_n ───────────────────────────────────────────────────────


class TestComputeAdaptiveN:
    """RFC §2.3 lookup table: <50→1-2, 50-200→3, 200-500→5, >500→7."""

    def test_tiny_chunk(self) -> None:
        text = "short chunk with few words"
        assert compute_adaptive_n(text, max_questions=12) <= 2
        assert compute_adaptive_n(text, max_questions=12) >= 1

    def test_small_chunk(self) -> None:
        text = " ".join(f"word{i}" for i in range(100))
        assert compute_adaptive_n(text, max_questions=12) == 3

    def test_medium_chunk(self) -> None:
        text = " ".join(f"token{i}" for i in range(300))
        assert compute_adaptive_n(text, max_questions=12) == 5

    def test_large_chunk(self) -> None:
        text = " ".join(f"token{i}" for i in range(600))
        assert compute_adaptive_n(text, max_questions=12) == 7

    def test_cap_at_max_questions(self) -> None:
        text = " ".join(f"token{i}" for i in range(600))
        assert compute_adaptive_n(text, max_questions=4) == 4

    def test_cap_at_max_questions_small(self) -> None:
        text = " ".join(f"token{i}" for i in range(300))
        assert compute_adaptive_n(text, max_questions=2) == 2

    def test_floor_at_one(self) -> None:
        assert compute_adaptive_n("", max_questions=12) >= 1
        assert compute_adaptive_n("x", max_questions=0) >= 1

    def test_boundary_50_tokens(self) -> None:
        text = " ".join(f"w{i}" for i in range(50))
        assert compute_adaptive_n(text, max_questions=12) == 3

    def test_boundary_200_tokens(self) -> None:
        text = " ".join(f"w{i}" for i in range(200))
        assert compute_adaptive_n(text, max_questions=12) == 5

    def test_boundary_500_tokens(self) -> None:
        text = " ".join(f"w{i}" for i in range(500))
        assert compute_adaptive_n(text, max_questions=12) == 7

    def test_49_tokens(self) -> None:
        text = " ".join(f"w{i}" for i in range(49))
        assert compute_adaptive_n(text, max_questions=12) == 2


# ── smart_truncate ───────────────────────────────────────────────────────────


class TestSmartTruncate:
    def test_short_text_unchanged(self) -> None:
        text = "Short text under limit."
        assert smart_truncate(text, max_chars=4000) == text

    def test_exact_limit_unchanged(self) -> None:
        text = "x" * 4000
        assert smart_truncate(text, max_chars=4000) == text

    def test_paragraph_boundary_cut(self) -> None:
        para1 = "First paragraph. " * 100
        para2 = "Second paragraph. " * 100
        para3 = "Third paragraph. " * 100
        text = f"{para1}\n\n{para2}\n\n{para3}"
        result = smart_truncate(text, max_chars=4000, overlap=200)
        assert "\n[...]\n" in result
        assert result.endswith(text[-200:])

    def test_sentence_boundary_fallback(self) -> None:
        text = "Sentence one. " * 300
        result = smart_truncate(text, max_chars=4000, overlap=200)
        main_part = result.split("\n[...]\n")[0]
        assert main_part.rstrip().endswith(".")

    def test_hard_cut_fallback(self) -> None:
        text = "x" * 5000
        result = smart_truncate(text, max_chars=4000, overlap=200)
        assert "\n[...]\n" in result
        main_part = result.split("\n[...]\n")[0]
        assert len(main_part) == 4000

    def test_tail_overlap_present(self) -> None:
        tail_content = "TAIL_MARKER " * 20
        text = "x" * 5000 + tail_content
        result = smart_truncate(text, max_chars=4000, overlap=200)
        assert text[-200:] in result

    def test_custom_max_chars(self) -> None:
        text = "word " * 500
        result = smart_truncate(text, max_chars=1000, overlap=100)
        assert len(result) < len(text)

    def test_overlap_zero(self) -> None:
        text = "x" * 5000
        result = smart_truncate(text, max_chars=4000, overlap=0)
        assert len(result) <= 4000


# ── Doc2QueryConfig ──────────────────────────────────────────────────────────


class TestDoc2QueryConfig:
    def test_fingerprint_includes_adaptive_n(self) -> None:
        c1 = Doc2QueryConfig(adaptive_n=False)
        c2 = Doc2QueryConfig(adaptive_n=True)
        assert c1.config_fingerprint() != c2.config_fingerprint()

    def test_fingerprint_includes_max_questions(self) -> None:
        c1 = Doc2QueryConfig(max_questions_per_chunk=12)
        c2 = Doc2QueryConfig(max_questions_per_chunk=8)
        assert c1.config_fingerprint() != c2.config_fingerprint()

    def test_fingerprint_stable(self) -> None:
        c = Doc2QueryConfig()
        assert c.config_fingerprint() == c.config_fingerprint()

    def test_default_adaptive_n_false(self) -> None:
        c = Doc2QueryConfig()
        assert c.adaptive_n is False

    def test_default_max_questions(self) -> None:
        c = Doc2QueryConfig()
        assert c.max_questions_per_chunk == 12


# ── Doc2QueryGenerator ───────────────────────────────────────────────────────


class TestDoc2QueryGeneratorBuildPrompt:
    def test_smart_truncate_used(self) -> None:
        gen = Doc2QueryGenerator(Doc2QueryConfig())
        long_text = "x" * 6000
        prompt = gen._build_prompt(long_text)
        assert len(prompt) < 6000 + 500
        assert "[...]" in prompt

    def test_adaptive_n_in_prompt(self) -> None:
        gen = Doc2QueryGenerator(Doc2QueryConfig(adaptive_n=True))
        big_text = " ".join(f"word{i}" for i in range(600))
        prompt = gen._build_prompt(big_text, n=7)
        assert "exactly 7 questions" in prompt

    def test_non_adaptive_uses_baseline(self) -> None:
        gen = Doc2QueryGenerator(Doc2QueryConfig(n_questions=4, adaptive_n=False))
        prompt = gen._build_prompt("some text")
        assert "exactly 4 questions" in prompt

    def test_custom_prompt_prepended(self) -> None:
        gen = Doc2QueryGenerator(Doc2QueryConfig(custom_prompt="CONTEXT: K8s docs"))
        prompt = gen._build_prompt("chunk text")
        assert prompt.startswith("CONTEXT: K8s docs\n\n")


class TestDoc2QueryGeneratorParseQuestions:
    def test_basic_parsing(self) -> None:
        text = "1. What is X?\n2. How does Y work?\n3. Why does Z fail?"
        result = Doc2QueryGenerator._parse_questions(text)
        assert len(result) == 3
        assert result[0] == "What is X?"

    def test_rejects_non_question_lines(self) -> None:
        text = "Here is preamble.\n1. What is X?\nAnother statement."
        result = Doc2QueryGenerator._parse_questions(text)
        assert len(result) == 1
        assert result[0] == "What is X?"

    def test_strips_bullet_markers(self) -> None:
        text = "- What is X?\n* How does Y work?\n  - Why does Z fail?"
        result = Doc2QueryGenerator._parse_questions(text)
        assert len(result) == 3

    def test_empty_input(self) -> None:
        assert Doc2QueryGenerator._parse_questions("") == []
        assert Doc2QueryGenerator._parse_questions("  \n  ") == []


# ── AugmentationCache ────────────────────────────────────────────────────────


class TestAugmentationCache:
    def test_new_hash_should_process(self, tmp_path: Path) -> None:
        cache = AugmentationCache(tmp_path / "cache.json")
        assert cache.should_process("abc123") is True

    def test_success_skips_processing(self, tmp_path: Path) -> None:
        cache = AugmentationCache(tmp_path / "cache.json")
        cache.mark_success("abc123", ["Q1?", "Q2?"])
        assert cache.should_process("abc123") is False

    def test_failed_retries_up_to_max(self, tmp_path: Path) -> None:
        cache = AugmentationCache(tmp_path / "cache.json", max_retries=3)
        cache.mark_failed("abc123")
        assert cache.should_process("abc123") is True  # attempt 1 < 3
        cache.mark_failed("abc123")
        assert cache.should_process("abc123") is True  # attempt 2 < 3
        cache.mark_failed("abc123")
        assert cache.should_process("abc123") is False  # attempt 3 >= 3

    def test_persistence_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        c1 = AugmentationCache(path)
        c1.mark_success("h1", ["Q?"])
        c1.save()

        c2 = AugmentationCache(path)
        assert c2.should_process("h1") is False
        entry = c2.get("h1")
        assert entry is not None
        assert entry.questions == ["Q?"]

    def test_stats(self, tmp_path: Path) -> None:
        cache = AugmentationCache(tmp_path / "cache.json")
        cache.mark_success("a", ["Q?"])
        cache.mark_failed("b")
        stats = cache.stats
        assert stats["success"] == 1
        assert stats["failed"] == 1
        assert stats["total"] == 2

    def test_corrupt_cache_starts_fresh(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        path.write_text("{invalid json", encoding="utf-8")
        cache = AugmentationCache(path)
        assert cache.stats["total"] == 0
