"""Search a SQLite FTS5 index — no LLM, no vectors, pure lexical.

Usage:
    python -m hydrag.similarity_search "search query" --db hydrag_fts.db
    python -m hydrag.similarity_search "python decorators" --db hydrag_fts.db -n 10

T-742: Pure FTS5 similarity search without any LLM involved.
"""

import argparse
import sys
import time
from pathlib import Path

from hydrag.sqlite_store import SQLiteFTSStore


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Search SQLite FTS5 index")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--db", type=Path, default=Path("hydrag_fts.db"), help="SQLite DB path")
    parser.add_argument("-n", "--num-results", type=int, default=5, help="Number of results")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--max-chars", type=int, default=500,
                        help="Max chars to display per result (0 = unlimited)")
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"Error: Database {args.db} not found", file=sys.stderr)
        sys.exit(1)

    with SQLiteFTSStore(args.db) as store:
        stats = store.stats()
        t0 = time.perf_counter()
        results = store.hybrid_search(args.query, n_results=args.num_results)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if args.json:
            import json
            print(json.dumps({
                "query": args.query,
                "results": results,
                "count": len(results),
                "latency_ms": round(elapsed_ms, 3),
                "db_stats": stats,
            }, indent=2))
        else:
            print(f"Query: {args.query!r}")
            print(f"Results: {len(results)} / {stats['total_chunks']} chunks "
                  f"({elapsed_ms:.1f}ms)")
            print("-" * 60)
            for i, text in enumerate(results, 1):
                display = text[:args.max_chars] + "..." if args.max_chars and len(text) > args.max_chars else text
                print(f"\n[{i}] {display}\n")


if __name__ == "__main__":
    main()
