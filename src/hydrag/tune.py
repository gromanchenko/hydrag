"""HydRAG CRAG classifier fine-tuning pipeline.

Replaces the LLM-based CRAG supervisor with a fast binary classifier.
Teacher-student distillation: run CRAG LLM on corpus to generate labels,
train a DistilBERT binary classifier, export to ONNX for <15ms inference.

Usage::

    from hydrag.tune import tune, generate_training_data, train_classifier

    # One-call convenience
    tune(adapter, llm=my_llm, output_dir="./crag_model/")

    # Step-by-step
    data = generate_training_data(adapter, llm=my_llm, n_samples=500)
    model_dir = train_classifier(data, output_dir="./crag_model/")
    onnx_path = export_onnx(model_dir)

Requires optional dependencies::

    pip install hydrag-core[tune]
    # installs: transformers, onnxruntime, datasets

At runtime, ``crag_supervisor()`` auto-detects the ONNX model and uses it
when ``crag_mode="auto"`` (default).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import HydRAGConfig
from .fusion import CRAGVerdict, _text_of
from .protocols import LLMProvider, VectorStoreAdapter

logger = logging.getLogger("hydrag.tune")

# Default output directory
DEFAULT_MODEL_DIR = Path.home() / ".hydrag" / "models" / "crag_classifier"

# Training data schema version
DATASET_VERSION = "1.0"


@dataclass
class TrainingSample:
    """Single training sample for the CRAG classifier."""

    query: str
    context: str
    label: int  # 1 = SUFFICIENT, 0 = INSUFFICIENT
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingDataset:
    """Collection of labeled samples for classifier training."""

    samples: list[TrainingSample]
    version: str = DATASET_VERSION
    created_at: str = ""
    config_hash: str = ""

    def save(self, path: Path) -> None:
        """Persist dataset to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for s in self.samples:
            records.append({
                "query": s.query,
                "context": s.context,
                "label": s.label,
                "metadata": s.metadata,
            })
        payload = {
            "version": self.version,
            "created_at": self.created_at,
            "config_hash": self.config_hash,
            "samples": records,
        }
        path.write_text(json.dumps(payload, indent=2))
        logger.info("Saved %d training samples to %s", len(self.samples), path)

    @classmethod
    def load(cls, path: Path) -> "TrainingDataset":
        """Load dataset from JSON."""
        raw = json.loads(path.read_text())
        samples = [
            TrainingSample(
                query=r["query"],
                context=r["context"],
                label=r["label"],
                metadata=r.get("metadata", {}),
            )
            for r in raw["samples"]
        ]
        return cls(
            samples=samples,
            version=raw.get("version", DATASET_VERSION),
            created_at=raw.get("created_at", ""),
            config_hash=raw.get("config_hash", ""),
        )


# ── Synthetic query templates for training data generation ───────

_QUERY_TEMPLATES = [
    "How does {topic} work?",
    "What is {topic}?",
    "Explain {topic} in detail",
    "What configuration controls {topic}?",
    "How to use {topic}",
    "What are the parameters for {topic}?",
    "Show me {topic}",
    "Where is {topic} defined?",
    "What does {topic} return?",
    "How is {topic} implemented?",
]


def _extract_topics_from_chunks(chunks: list[str], max_topics: int = 100) -> list[str]:
    """Extract topic-like phrases from document chunks for query synthesis."""
    import re

    topics: list[str] = []
    for chunk in chunks:
        # Extract function/class names
        topics.extend(re.findall(r"\bdef\s+(\w+)", chunk))
        topics.extend(re.findall(r"\bclass\s+(\w+)", chunk))
        # Extract config-like identifiers
        topics.extend(re.findall(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+){2,})\b", chunk))
        # Extract CamelCase identifiers
        topics.extend(re.findall(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", chunk))
    # Dedupe, limit
    seen: set[str] = set()
    unique: list[str] = []
    for t in topics:
        if t not in seen and len(t) > 2:
            seen.add(t)
            unique.append(t)
    return unique[:max_topics]


def generate_training_data(
    adapter: VectorStoreAdapter,
    llm: LLMProvider,
    n_samples: int = 500,
    config: HydRAGConfig | None = None,
    seed: int = 42,
    parallel_workers: int = 1,
    teacher_retry_budget: int = 1,
) -> TrainingDataset:
    """Generate labeled training data via teacher-student distillation.

    Retrieves documents from the adapter, synthesizes queries, runs the
    CRAG LLM supervisor to generate binary labels, and returns a
    ``TrainingDataset`` ready for classifier training.

    Parameters
    ----------
    adapter : VectorStoreAdapter
        Vector store to sample documents from.
    llm : LLMProvider
        LLM provider to use as the teacher (CRAG supervisor).
    n_samples : int
        Number of training samples to generate.
    config : HydRAGConfig | None
        Pipeline config (for CRAG model/timeout settings).
    seed : int
        Random seed for reproducibility.
    parallel_workers : int
        Number of parallel workers for teacher labeling (1 = serial).
        Higher values speed up generation proportionally but increase
        LLM load.
    teacher_retry_budget : int
        Number of retries for each teacher labeling attempt on transient
        failures (total attempts = retry_budget + 1).
    """
    from .core import crag_supervisor

    cfg = config or HydRAGConfig()
    rng = random.Random(seed)

    # Sample a broad set of documents from the adapter
    seed_queries = ["function", "class", "config", "error", "data", "model", "test", "api"]
    all_chunks: list[str] = []
    for sq in seed_queries:
        try:
            hits = adapter.hybrid_search(sq, n_results=20)
            all_chunks.extend(str(h) for h in hits)
        except Exception:
            pass

    if not all_chunks:
        logger.warning("No documents retrieved from adapter — returning empty dataset")
        return TrainingDataset(samples=[], created_at=_now_iso())

    # Extract topics for synthetic queries
    topics = _extract_topics_from_chunks(all_chunks, max_topics=n_samples * 2)
    if not topics:
        topics = ["general topic"]

    # Build work items: (query, context_hits, context_text) tuples
    work_items: list[tuple[str, list[Any], str, str]] = []
    templates = _QUERY_TEMPLATES

    for i in range(n_samples):
        topic = rng.choice(topics)
        template = rng.choice(templates)
        query = template.format(topic=topic)

        try:
            context_hits = adapter.hybrid_search(query, n_results=cfg.crag_context_chunks)
        except Exception:
            continue

        if not context_hits:
            continue

        context_text = "\n---\n".join(
            _text_of(c)[: cfg.crag_char_limit] for c in context_hits[: cfg.crag_context_chunks]
        )
        work_items.append((query, context_hits, context_text, topic))

    def _label_one(item: tuple[str, list[Any], str, str]) -> TrainingSample | None:
        q, hits, ctx, topic = item
        attempts = max(teacher_retry_budget, 0) + 1
        verdict = None
        for attempt in range(attempts):
            try:
                verdict = crag_supervisor(q, hits, llm=llm, config=cfg)
                break
            except Exception:
                if attempt >= attempts - 1:
                    return None
                continue
        if verdict is None:
            return None
        return TrainingSample(
            query=q,
            context=ctx,
            label=1 if verdict.sufficient else 0,
            metadata={
                "teacher_reason": verdict.reason,
                "teacher_latency_ms": verdict.latency_ms,
                "topic": topic,
            },
        )

    samples: list[TrainingSample] = []
    if parallel_workers > 1:
        from concurrent.futures import ThreadPoolExecutor as _Pool
        with _Pool(max_workers=parallel_workers) as pool:
            for i, result in enumerate(pool.map(_label_one, work_items)):
                if result is not None:
                    samples.append(result)
                if (i + 1) % 50 == 0:
                    logger.info("Generated %d/%d training samples", i + 1, len(work_items))
    else:
        for i, item in enumerate(work_items):
            result = _label_one(item)
            if result is not None:
                samples.append(result)
            if (i + 1) % 50 == 0:
                logger.info("Generated %d/%d training samples", i + 1, len(work_items))

    config_hash = hashlib.sha256(json.dumps({
        "crag_model": cfg.crag_model,
        "n_samples": n_samples,
        "seed": seed,
    }).encode()).hexdigest()[:12]

    logger.info(
        "Training data: %d samples (%d sufficient, %d insufficient)",
        len(samples),
        sum(1 for s in samples if s.label == 1),
        sum(1 for s in samples if s.label == 0),
    )

    return TrainingDataset(
        samples=samples,
        created_at=_now_iso(),
        config_hash=config_hash,
    )


def generate_training_data_from_logs(
    log_dir: Path,
) -> TrainingDataset:
    """Build training dataset from existing CRAG benchmark/run logs.

    Expects JSON files with ``query``, ``context``, and ``verdict`` fields
    (as produced by benchmark runners).
    """
    samples: list[TrainingSample] = []
    for path in sorted(log_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        # Support both single-record and array formats
        records = raw if isinstance(raw, list) else [raw]
        for rec in records:
            query = rec.get("query", "")
            context = rec.get("context", "")
            verdict = rec.get("verdict", "")
            if not query or not context:
                continue
            label = 1 if "SUFFICIENT" in str(verdict).upper() and "INSUFFICIENT" not in str(verdict).upper() else 0
            samples.append(TrainingSample(query=query, context=context, label=label))

    logger.info("Loaded %d training samples from logs in %s", len(samples), log_dir)
    return TrainingDataset(samples=samples, created_at=_now_iso())


def train_classifier(
    dataset: TrainingDataset,
    output_dir: str | Path | None = None,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 512,
) -> Path:
    """Train a binary classifier for CRAG verdict prediction.

    Uses HuggingFace Transformers ``Trainer`` API to fine-tune a
    DistilBERT (or compatible) model on the labeled dataset.

    Returns the path to the saved model directory.
    """
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError(
            "Training requires: pip install hydrag-core[tune]"
        ) from exc

    out = Path(output_dir) if output_dir else DEFAULT_MODEL_DIR
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore[no-untyped-call]
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2,
    )

    # Prepare dataset
    texts = [f"{s.query} [SEP] {s.context}" for s in dataset.samples]
    labels = [s.label for s in dataset.samples]

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    import torch as _torch

    class _CRAGDataset:
        def __init__(self, enc: Any, lab: list[int]) -> None:
            self._enc = enc
            self._lab = lab

        def __len__(self) -> int:
            return len(self._lab)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            item = {k: v[idx] for k, v in self._enc.items()}
            item["labels"] = _torch.tensor(self._lab[idx])
            return item

    train_dataset = _CRAGDataset(encodings, labels)

    training_args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=50,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    logger.info("Training %s on %d samples for %d epochs...", model_name, len(dataset.samples), epochs)
    trainer.train()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))

    # Save dataset metadata alongside model
    meta = {
        "model_name": model_name,
        "n_samples": len(dataset.samples),
        "epochs": epochs,
        "dataset_hash": dataset.config_hash,
        "created_at": _now_iso(),
    }
    (out / "tune_metadata.json").write_text(json.dumps(meta, indent=2))

    logger.info("Model saved to %s", out)
    return out


def export_onnx(
    model_dir: str | Path,
    output_path: str | Path | None = None,
    opset_version: int = 14,
) -> Path:
    """Export trained classifier to ONNX format for fast inference.

    Returns the path to the ONNX model file.
    """
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "ONNX export requires: pip install hydrag-core[tune]"
        ) from exc

    model_dir = Path(model_dir)
    onnx_path = Path(output_path) if output_path else model_dir / "model.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))  # type: ignore[no-untyped-call]
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    import torch

    dummy = tokenizer(
        "sample query [SEP] sample context",
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length",
    )

    torch.onnx.export(
        model,
        tuple(dummy.values()),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=opset_version,
    )

    logger.info("ONNX model exported to %s (%.1f MB)", onnx_path, onnx_path.stat().st_size / 1e6)
    return onnx_path


def tune(
    adapter: VectorStoreAdapter,
    llm: LLMProvider | None = None,
    output_dir: str | Path | None = None,
    n_samples: int = 500,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    export: bool = True,
    config: HydRAGConfig | None = None,
    seed: int = 42,
    parallel_workers: int = 1,
    teacher_retry_budget: int = 1,
) -> Path:
    """One-call fine-tuning: generate data → train classifier → export ONNX.

    Returns the path to the output model directory.

    Example::

        from hydrag import HydRAG
        from hydrag.tune import tune

        engine = HydRAG(my_adapter)
        model_path = tune(my_adapter, llm=my_llm)
        # Now crag_supervisor() will auto-detect and use the ONNX model

    Parameters
    ----------
    adapter : VectorStoreAdapter
        Vector store to sample documents from.
    llm : LLMProvider | None
        Teacher LLM. Defaults to OllamaProvider.
    output_dir : str | Path | None
        Where to save the model. Defaults to ``~/.hydrag/models/crag_classifier/``.
    n_samples : int
        Number of training samples to generate.
    model_name : str
        HuggingFace model name for the student classifier.
    epochs : int
        Training epochs.
    export : bool
        Whether to export to ONNX after training.
    config : HydRAGConfig | None
        Pipeline config (for teacher CRAG settings).
    seed : int
        Random seed.
    parallel_workers : int
        Number of parallel workers for teacher labeling.
    teacher_retry_budget : int
        Number of retries for each teacher labeling attempt.
    """
    from .core import OllamaProvider

    cfg = config or HydRAGConfig()
    llm_provider = llm or OllamaProvider(host=cfg.ollama_host)
    out = Path(output_dir) if output_dir else DEFAULT_MODEL_DIR

    logger.info("Starting CRAG classifier fine-tuning pipeline...")
    start = time.monotonic()

    # Step 1: Generate training data
    dataset = generate_training_data(
        adapter,
        llm_provider,
        n_samples=n_samples,
        config=cfg,
        seed=seed,
        parallel_workers=parallel_workers,
        teacher_retry_budget=teacher_retry_budget,
    )
    if not dataset.samples:
        raise ValueError("No training samples generated — check adapter contents")

    # Save dataset for reproducibility
    dataset.save(out / "training_data.json")

    # Step 2: Train classifier
    model_dir = train_classifier(
        dataset, output_dir=out, model_name=model_name, epochs=epochs,
    )

    # Step 3: Export to ONNX
    if export:
        export_onnx(model_dir)

    elapsed = time.monotonic() - start
    logger.info("Fine-tuning complete in %.1fs. Model at %s", elapsed, out)
    return out


def tune_from_logs(
    log_dir: str | Path,
    output_dir: str | Path | None = None,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    export: bool = True,
) -> Path:
    """Fine-tune from existing CRAG logs (no LLM teacher needed).

    Example::

        from hydrag.tune import tune_from_logs
        model_path = tune_from_logs("./bench_results/")
    """
    out = Path(output_dir) if output_dir else DEFAULT_MODEL_DIR

    dataset = generate_training_data_from_logs(Path(log_dir))
    if not dataset.samples:
        raise ValueError(f"No training samples found in {log_dir}")

    dataset.save(out / "training_data.json")
    model_dir = train_classifier(
        dataset, output_dir=out, model_name=model_name, epochs=epochs,
    )
    if export:
        export_onnx(model_dir)

    return out


# ── ONNX Runtime inference (used by crag_supervisor) ─────────────


class CRAGClassifier:
    """ONNX-based CRAG binary classifier for fast inference.

    Loads once, reuses session for subsequent predictions.
    Typical inference: 5-15ms on CPU.
    """

    def __init__(self, model_dir: str | Path) -> None:
        self._model_dir = Path(model_dir)
        self._session = None
        self._tokenizer = None
        self._max_length = 512

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return

        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Classifier inference requires: pip install hydrag-core[tune]"
            ) from exc

        onnx_path = self._model_dir / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"No ONNX model at {onnx_path}")

        self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir))  # type: ignore[no-untyped-call]
        self._session = ort.InferenceSession(str(onnx_path))
        logger.info("Loaded CRAG classifier from %s", onnx_path)

    def predict(self, query: str, context: str) -> CRAGVerdict:
        """Run classifier inference and return a CRAGVerdict."""
        self._ensure_loaded()
        if self._tokenizer is None or self._session is None:
            raise RuntimeError("Classifier not loaded — call _ensure_loaded() first")

        start = time.monotonic()
        text = f"{query} [SEP] {context}"
        inputs = self._tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=self._max_length,
            padding="max_length",
        )

        outputs = self._session.run(
            None,
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
        )

        import numpy as np
        logits = outputs[0][0]
        predicted = int(np.argmax(logits))
        latency_ms = (time.monotonic() - start) * 1000

        return CRAGVerdict(
            sufficient=predicted == 1,
            reason="classifier",
            latency_ms=latency_ms,
        )


# Cached singleton — loaded once per process (thread-safe)
_CLASSIFIER_CACHE: dict[str, CRAGClassifier] = {}
_CLASSIFIER_LOCK = threading.Lock()


def get_classifier(model_dir: str | Path) -> CRAGClassifier:
    """Get or create a cached CRAGClassifier instance."""
    resolved = Path(model_dir).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Classifier model directory not found: {resolved}")
    key = str(resolved)
    with _CLASSIFIER_LOCK:
        if key not in _CLASSIFIER_CACHE:
            _CLASSIFIER_CACHE[key] = CRAGClassifier(model_dir)
        return _CLASSIFIER_CACHE[key]


# ── Utility ──────────────────────────────────────────────────────


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
