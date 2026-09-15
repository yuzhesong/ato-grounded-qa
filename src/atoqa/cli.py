"""Command line entry point: fetch, index, ask."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import REPO_ROOT, settings
from .index import HybridIndex
from .ingest import SEED_URLS, fetch, load_markdown_corpus, load_raw
from .pipeline import ask

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "corpus"


def build_index(corpus: str | None = None) -> HybridIndex:
    if corpus:
        path = Path(corpus)
        chunks = load_markdown_corpus(path) if any(path.glob("*.md")) else load_raw(path)
    elif (Path(settings.index_dir) / "chunks.json").exists():
        return HybridIndex.load()
    else:
        try:
            chunks = load_raw()
        except FileNotFoundError:
            print("No fetched corpus; using the fixture corpus.", file=sys.stderr)
            chunks = load_markdown_corpus(FIXTURES)
    return HybridIndex(chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="atoqa", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", help="download the seed ATO pages into data/raw")

    p_index = sub.add_parser("index", help="build and save the retrieval index")
    p_index.add_argument("--corpus", default=None)

    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--corpus", default=None)
    p_ask.add_argument("--json", action="store_true", help="emit the full Answer as JSON")

    args = parser.parse_args(argv)

    if args.command == "fetch":
        out = fetch(SEED_URLS)
        print(f"Fetched {len(SEED_URLS)} pages into {out}")
        return 0

    if args.command == "index":
        idx = build_index(args.corpus)
        path = idx.save()
        print(f"Indexed {len(idx.chunks)} chunks -> {path}")
        return 0

    idx = build_index(args.corpus)
    answer = ask(args.question, index=idx)

    if args.json:
        print(answer.model_dump_json(indent=2))
        return 0

    print(f"\n{answer.verdict.value.upper()}  (intent: {answer.intent.value}", end="")
    if answer.income_year:
        print(f", income year: {answer.income_year}"
              f"{' assumed' if answer.income_year_assumed else ''}", end="")
    print(")\n")
    print(answer.text)
    if answer.sources:
        print("\nSources")
        for s in answer.sources:
            print(f"  [{s['id']}] {s['citation']}")
            print(f"      {s['url']}")
    if answer.guards:
        print("\nVerification")
        for g in answer.guards:
            mark = "pass" if g.passed else "FAIL"
            score = "" if g.score is None else f" ({g.score})"
            print(f"  {mark:4} {g.name}{score}{(' - ' + g.detail) if g.detail else ''}")
    print(f"\ntrace {answer.trace_id}  {answer.latency_ms:.0f} ms")
    return 0 if answer.answered else 2


if __name__ == "__main__":
    raise SystemExit(main())
