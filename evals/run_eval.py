#!/usr/bin/env python3
"""Run the evaluation harness and gate on regression thresholds.

    python evals/run_eval.py                              # fixture corpus
    python evals/run_eval.py --corpus data/raw            # live fetched corpus
    python evals/run_eval.py --no-gate --json out.json

Exit status is 1 if any threshold in thresholds.yaml is breached, so this is
usable directly as a CI gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics import aggregate, check_thresholds, score_item  # noqa: E402

from atoqa.config import settings  # noqa: E402
from atoqa.index import HybridIndex  # noqa: E402
from atoqa.ingest import load_markdown_corpus, load_raw  # noqa: E402
from atoqa.llm import AdversarialStub, ExtractiveStub  # noqa: E402
from atoqa.pipeline import ask  # noqa: E402
from atoqa.schemas import Verdict  # noqa: E402

HERE = Path(__file__).resolve().parent
FIXTURES = ROOT / "tests" / "fixtures" / "corpus"


def load_index(corpus: str | None) -> HybridIndex:
    path = Path(corpus) if corpus else FIXTURES
    chunks = load_markdown_corpus(path) if any(path.glob("*.md")) else load_raw(path)
    return HybridIndex(chunks)


def run_task_suite(index: HybridIndex, items: list[dict]) -> list:
    llm = ExtractiveStub()
    return [
        score_item(item, ask(item["question"], index=index, llm=llm, trace=False))
        for item in items
    ]


def run_guard_suite(index: HybridIndex, items: list[dict]) -> dict:
    """Inject each fault mode into questions the system would otherwise answer.

    Sensitivity is recall over faults actually injected. Items where the
    adversarial stub had nothing to corrupt (because the base answer was empty)
    are excluded rather than counted as catches -- inflating the denominator
    with non-events would make the guards look better than they are.
    """
    answerable = [i for i in items if i["expect"] == "answer" and i["category"] != "calculation"]
    per_mode: dict[str, dict] = {}
    for mode in AdversarialStub.MODES:
        llm = AdversarialStub(mode)
        injected = caught = 0
        escaped: list[str] = []
        for item in answerable:
            baseline = ask(item["question"], index=index, llm=ExtractiveStub(), trace=False)
            if baseline.verdict not in (Verdict.ANSWER, Verdict.ANSWER_WITH_CAVEAT):
                continue
            answer = ask(item["question"], index=index, llm=llm, trace=False)
            if not llm.injected:
                continue          # this mode had nothing to corrupt on this item
            injected += 1
            if answer.verdict in (Verdict.ABSTAIN, Verdict.REFUSE):
                caught += 1
            else:
                escaped.append(item["id"])
        per_mode[mode] = {
            "injected": injected,
            "caught": caught,
            "recall": round(caught / injected, 4) if injected else None,
            "escaped": escaped,
        }
    total_injected = sum(m["injected"] for m in per_mode.values())
    total_caught = sum(m["caught"] for m in per_mode.values())
    return {
        "per_mode": per_mode,
        "injected": total_injected,
        "caught": total_caught,
        "sensitivity": round(total_caught / total_injected, 4) if total_injected else 1.0,
    }


def print_table(results: list) -> None:
    print(f"\n{'id':26} {'category':13} {'want':8} {'got':19} ok  notes")
    print("-" * 100)
    for r in results:
        ok = "y " if r.verdict_ok else "N "
        notes = []
        if r.numbers_missing:
            notes.append(f"missing {r.numbers_missing}")
        if r.forbidden_present:
            notes.append(f"FORBIDDEN {r.forbidden_present}")
        if not r.citations_ok:
            notes.append("citations unresolved")
        if r.guards_failed:
            notes.append("guards: " + ",".join(r.guards_failed))
        print(f"{r.id:26} {r.category:13} {r.expected:8} {r.actual:19} {ok} {'; '.join(notes)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=None, help="corpus directory (default: fixtures)")
    ap.add_argument("--questions", default=str(HERE / "questions.yaml"))
    ap.add_argument("--thresholds", default=str(HERE / "thresholds.yaml"))
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--no-gate", action="store_true", help="report without failing on regression")
    ap.add_argument("--skip-guards", action="store_true")
    args = ap.parse_args()

    items = yaml.safe_load(Path(args.questions).read_text("utf-8"))
    thresholds = yaml.safe_load(Path(args.thresholds).read_text("utf-8"))
    index = load_index(args.corpus)

    print(f"corpus: {args.corpus or FIXTURES}  ({len(index.chunks)} chunks)")
    print(f"provider: {settings.provider}   min_retrieval_score: {settings.min_retrieval_score}")

    results = run_task_suite(index, items)
    print_table(results)
    metrics = aggregate(results)

    guards = {"sensitivity": 1.0, "per_mode": {}, "injected": 0, "caught": 0}
    if not args.skip_guards:
        guards = run_guard_suite(index, items)

    print("\nTask metrics")
    for key in (
        "n_items", "verdict_accuracy", "false_answer_rate", "missed_answer_rate",
        "numeric_accuracy", "forbidden_number_rate", "citation_validity", "answer_rate",
    ):
        print(f"  {key:24} {metrics[key]}")
    print("  by category")
    for cat, acc in metrics["by_category"].items():
        print(f"    {cat:22} {acc}")

    if not args.skip_guards:
        print("\nGuard sensitivity (injected faults caught)")
        for mode, m in guards["per_mode"].items():
            esc = f"  escaped: {', '.join(m['escaped'])}" if m["escaped"] else ""
            print(f"  {mode:20} {m['caught']}/{m['injected']}  recall={m['recall']}{esc}")
        print(f"  {'OVERALL':20} {guards['caught']}/{guards['injected']}  "
              f"sensitivity={guards['sensitivity']}")

    failures = check_thresholds(metrics, guards["sensitivity"], thresholds)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus": str(args.corpus or FIXTURES),
        "provider": settings.provider,
        "chunks": len(index.chunks),
        "thresholds": thresholds,
        "metrics": metrics,
        "guards": guards,
        "failures": failures,
        "items": [asdict(r) for r in results],
    }
    out = Path(args.json_out) if args.json_out else HERE / "reports" / "latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport -> {out}")

    if failures:
        print("\nREGRESSION")
        for f in failures:
            print(f"  - {f}")
        return 0 if args.no_gate else 1
    print("\nall thresholds met")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
