"""Index a directory of text/markdown files into SQLite FTS5.

Usage:
    python -m hydrag.indexer /path/to/docs --db hydrag.db
    python -m hydrag.indexer /path/to/docs --db hydrag.db --enrich ollama

T-742: Pure FTS5 indexing (no --enrich flag)
T-743: With --enrich, uses LLM to generate summary + keywords per chunk
"""

import argparse
import sys
import uuid
from pathlib import Path

from hydrag.sqlite_store import IndexedChunk, SQLiteFTSStore


def _chunk_file(path: Path, max_chunk_chars: int = 2000) -> list[IndexedChunk]:
    """Split a file into chunks. Simple paragraph-based splitting."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"  SKIP {path}: {e}", file=sys.stderr)
        return []

    if not text.strip():
        return []

    chunks: list[IndexedChunk] = []
    paragraphs = text.split("\n\n")
    current = ""
    idx = 0

    for para in paragraphs:
        if len(current) + len(para) > max_chunk_chars and current:
            chunks.append(
                IndexedChunk(
                    chunk_id=f"{path.name}::{idx}",
                    source=str(path),
                    title=path.stem,
                    raw_content=current.strip(),
                )
            )
            idx += 1
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para

    if current.strip():
        chunks.append(
            IndexedChunk(
                chunk_id=f"{path.name}::{idx}",
                source=str(path),
                title=path.stem,
                raw_content=current.strip(),
            )
        )

    return chunks


def _build_extractor(provider: str, model: str, host: str) -> "KeywordExtractor | None":
    """Build a KeywordExtractor if --enrich is requested (T-743)."""
    if not provider:
        return None

    from hydrag.enrichment import OllamaKeywordExtractor

    return OllamaKeywordExtractor(model=model, host=host)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Index documents into SQLite FTS5")
    parser.add_argument("directory", type=Path, help="Directory of text/markdown files")
    parser.add_argument("--db", type=Path, default=Path("hydrag_fts.db"), help="SQLite DB path")
    parser.add_argument("--extensions", nargs="+", default=[".md", ".txt", ".py", ".rst"],
                        help="File extensions to index")
    parser.add_argument("--max-chunk-chars", type=int, default=2000, help="Max chars per chunk")
    parser.add_argument("--enrich", type=str, default="", metavar="PROVIDER",
                        help="LLM provider for enrichment (e.g. 'ollama')")
    parser.add_argument("--enrich-model", type=str, default="qwen3:4b",
                        help="Model for keyword/summary generation")
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434",
                        help="Ollama API host")
    args = parser.parse_args(argv)

    if not args.directory.is_dir():
        print(f"Error: {args.directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Collect files
    files: list[Path] = []
    for ext in args.extensions:
        files.extend(sorted(args.directory.rglob(f"*{ext}")))

    if not files:
        print(f"No files with extensions {args.extensions} found in {args.directory}")
        sys.exit(0)

    print(f"Found {len(files)} files to index")

    # Build extractor (T-743)
    extractor = _build_extractor(args.enrich, args.enrich_model, args.ollama_host)
    if extractor:
        print(f"LLM enrichment enabled: {args.enrich} / {args.enrich_model}")

    # Index
    all_chunks: list[IndexedChunk] = []
    for f in files:
        chunks = _chunk_file(f, args.max_chunk_chars)
        all_chunks.extend(chunks)

    print(f"Chunked into {len(all_chunks)} chunks")

    with SQLiteFTSStore(args.db) as store:
        indexed = store.index_documents(
            all_chunks,
            extractor=extractor,
            model_id=args.enrich_model if extractor else "",
        )
        stats = store.stats()
        print(f"Indexed {indexed} new chunks (total: {stats['total_chunks']}, "
              f"enriched: {stats['enriched_chunks']})")
        print(f"Database: {args.db}")


if __name__ == "__main__":
    main()
