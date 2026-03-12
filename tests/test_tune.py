"""Tests for hydrag.tune — CRAG classifier fine-tuning pipeline.

These tests use mocks to avoid requiring GPU, transformers, or ONNX runtime.
They verify the data structures, serialization, and control flow.
"""

import json
from pathlib import Path
from unittest.mock import patch

from hydrag.tune import (
    TrainingDataset,
    TrainingSample,
    generate_training_data,
)


class TestTrainingSample:
    def test_fields(self) -> None:
        s = TrainingSample(query="q1", context="ctx", label=1, metadata={"src": "test"})
        assert s.query == "q1"
        assert s.context == "ctx"
        assert s.label == 1
        assert s.metadata["src"] == "test"

    def test_default_metadata(self) -> None:
        s = TrainingSample(query="q1", context="ctx", label=0)
        assert s.metadata == {}


class TestTrainingDataset:
    def test_add_and_len(self) -> None:
        ds = TrainingDataset(samples=[
            TrainingSample(query="q1", context="c1", label=1),
            TrainingSample(query="q2", context="c2", label=0),
        ])
        assert len(ds.samples) == 2

    def test_save_and_load(self, tmp_path: Path) -> None:
        ds = TrainingDataset(samples=[
            TrainingSample(query="q1", context="c1", label=1, metadata={"conf": 0.95}),
            TrainingSample(query="q2", context="c2", label=0, metadata={"conf": 0.8}),
        ])

        out_file = tmp_path / "train.json"
        ds.save(out_file)
        assert out_file.exists()

        loaded = TrainingDataset.load(out_file)
        assert len(loaded.samples) == 2
        assert loaded.samples[0].query == "q1"
        assert loaded.samples[0].label == 1
        assert loaded.samples[1].metadata["conf"] == 0.8

    def test_save_json_format(self, tmp_path: Path) -> None:
        ds = TrainingDataset(samples=[
            TrainingSample(query="q", context="c", label=1),
        ])

        out_file = tmp_path / "data.json"
        ds.save(out_file)
        raw = json.loads(out_file.read_text())
        assert isinstance(raw, dict)
        assert "samples" in raw
        assert raw["samples"][0]["query"] == "q"
        assert raw["samples"][0]["label"] == 1


class TestCRAGClassifier:
    def test_import_available(self) -> None:
        """CRAGClassifier should be importable even without ONNX runtime."""
        from hydrag.tune import CRAGClassifier
        assert CRAGClassifier is not None

    def test_get_classifier_returns_instance(self, tmp_path: Path) -> None:
        """get_classifier should return a CRAGClassifier (lazy-loaded)."""
        from hydrag.tune import CRAGClassifier, get_classifier
        classifier = get_classifier(str(tmp_path / "nonexistent"))
        assert isinstance(classifier, CRAGClassifier)


class TestGenerateTrainingDataFromLogs:
    def test_loads_from_log_files(self, tmp_path: Path) -> None:
        """generate_training_data_from_logs should parse JSON log files."""
        from hydrag.tune import generate_training_data_from_logs

        # Create mock log files
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        log_data = [
            {"query": "test query 1", "context": "some context", "verdict": "SUFFICIENT"},
            {"query": "test query 2", "context": "other context", "verdict": "INSUFFICIENT"},
        ]
        for i, entry in enumerate(log_data):
            (log_dir / f"crag_log_{i}.json").write_text(json.dumps(entry))

        ds = generate_training_data_from_logs(log_dir)
        assert len(ds.samples) == 2


class TestGenerateTrainingDataRetries:
    def test_teacher_retry_budget_recovers_transient_failures(self) -> None:
        class _Adapter:
            def semantic_search(self, query: str, n_results: int = 5) -> list[str]:
                return ["s1"]

            def keyword_search(self, query: str, n_results: int = 5) -> list[str]:
                return ["k1"]

            def hybrid_search(self, query: str, n_results: int = 5) -> list[str]:
                return ["h1", "h2"]

        class _LLM:
            def generate(self, prompt: str, model: str = "", timeout: int = 30) -> str | None:
                return "SUFFICIENT"

        call_state = {"count": 0}

        def _flaky_crag(*args: object, **kwargs: object) -> object:
            call_state["count"] += 1
            if call_state["count"] == 1:
                raise RuntimeError("transient")
            from hydrag import CRAGVerdict
            return CRAGVerdict(sufficient=True, reason="model_verdict", latency_ms=1.0)

        with patch("hydrag.core.crag_supervisor", side_effect=_flaky_crag):
            ds = generate_training_data(
                _Adapter(),
                _LLM(),
                n_samples=1,
                parallel_workers=1,
                teacher_retry_budget=1,
            )
        assert len(ds.samples) == 1

    def test_teacher_retry_budget_zero_drops_failed_sample(self) -> None:
        class _Adapter:
            def semantic_search(self, query: str, n_results: int = 5) -> list[str]:
                return ["s1"]

            def keyword_search(self, query: str, n_results: int = 5) -> list[str]:
                return ["k1"]

            def hybrid_search(self, query: str, n_results: int = 5) -> list[str]:
                return ["h1", "h2"]

        class _LLM:
            def generate(self, prompt: str, model: str = "", timeout: int = 30) -> str | None:
                return "SUFFICIENT"

        with patch("hydrag.core.crag_supervisor", side_effect=RuntimeError("always bad")):
            ds = generate_training_data(
                _Adapter(),
                _LLM(),
                n_samples=1,
                parallel_workers=1,
                teacher_retry_budget=0,
            )
        assert len(ds.samples) == 0

    def test_empty_dir(self, tmp_path: Path) -> None:
        from hydrag.tune import generate_training_data_from_logs

        log_dir = tmp_path / "empty_logs"
        log_dir.mkdir()
        ds = generate_training_data_from_logs(log_dir)
        assert len(ds.samples) == 0

    def test_array_format(self, tmp_path: Path) -> None:
        """Should handle JSON files with arrays of records."""
        from hydrag.tune import generate_training_data_from_logs

        log_dir = tmp_path / "arr_logs"
        log_dir.mkdir()
        records = [
            {"query": "q1", "context": "c1", "verdict": "SUFFICIENT"},
            {"query": "q2", "context": "c2", "verdict": "INSUFFICIENT"},
        ]
        (log_dir / "batch.json").write_text(json.dumps(records))

        ds = generate_training_data_from_logs(log_dir)
        assert len(ds.samples) == 2
