"""The orchestrator: one function, one decision path, all controls visible.

Design intent, in the order the JD for this kind of work usually asks about it:

  * Deterministic first. Arithmetic and published thresholds are computed from
    `rates.py`, never recalled by a model. The model's job is wording.
  * Retrieval second, and only where explanation is actually needed.
  * Guards third, with the authority to downgrade an answer to a caveat or to
    refuse it outright.
  * Human hand-off is a real branch, not a disclaimer bolted onto the text:
    requests for personal advice are refused and routed.
  * Every path writes an audit trace.

A deterministic computation is wrapped as a citable `Chunk` so that it passes
through exactly the same guard stack as retrieved text. There is one code path
for "can this answer be trusted", not two.
"""

from __future__ import annotations

import time

from . import audit, guards
from .calculator import CalculationRefused, Computation, income_tax, lookup, marginal_rate
from .config import settings
from .index import HybridIndex
from .llm import LLM, get_llm
from .router import Route, detect_threshold, route
from .schemas import Answer, Chunk, Claim, DraftAnswer, Intent, Retrieved, Verdict

REFERRAL = (
    "This tool reports what published ATO guidance says. It cannot advise on your "
    "circumstances -- for that, speak to a registered tax agent."
)


def _computation_chunk(comp: Computation) -> Chunk:
    return Chunk(
        id=f"calc:{comp.kind}",
        text=" ".join(comp.working),
        title="Deterministic calculation from published ATO rates",
        source_url=comp.source_url,
        qc_ref=comp.qc_ref,
        heading_path=["computed"],
        income_years=[comp.income_year] if comp.income_year else [],
        retrieved_at=comp.checked_on,
    )


def _as_retrieved(chunk: Chunk) -> Retrieved:
    return Retrieved(chunk=chunk, score=1.0, rank=0)


def _prune(retrieved: list[Retrieved]) -> list[Retrieved]:
    """Drop passages far weaker than the best one before they reach the model.

    `top_k` is a retrieval parameter, not a judgement about relevance: asking
    for six passages returns six whether or not six are relevant. Showing the
    tail to a generator invites it to write a well-cited claim from a passage
    that merely shares vocabulary -- a question about the super guarantee rate
    retrieved the working-from-home page at 0.59 of the top score, on the
    strength of both pages mentioning a "rate", and the resulting answer cited
    it accurately and irrelevantly.

    So the cut is relative to the best passage, using the calibrated score.
    Relative rather than absolute because the top score varies with question
    length and phrasing; the shape of the drop-off is the stable signal.

    This sits in the pipeline rather than in the generator so that every
    provider gets the same context, and so the guards resolve citations against
    exactly the set the model was shown -- a model cannot cite a passage that
    was pruned, because it never saw it.
    """
    if not retrieved:
        return retrieved
    floor = retrieved[0].score * settings.context_relative_cutoff
    return [r for r in retrieved if r.score >= floor] or retrieved[:1]


def _render(claims: list[Claim], caveats: list[str]) -> str:
    lines = [f"{c.text} [{', '.join(c.citations)}]" for c in claims]
    body = "\n".join(lines)
    if caveats:
        body += "\n\n" + "\n".join(f"Note: {c}" for c in caveats)
    return body


def _sources(retrieved: list[Retrieved], claims: list[Claim]) -> list[dict]:
    used = {cid for c in claims for cid in c.citations}
    out = []
    for r in retrieved:
        if r.chunk.id not in used:
            continue
        out.append(
            {
                "id": r.chunk.id,
                "citation": r.chunk.citation,
                "url": r.chunk.source_url,
                "income_years": r.chunk.income_years,
                "checked_on": r.chunk.retrieved_at,
            }
        )
    return out


def _refuse(question: str, r: Route, reason: str, started: float) -> Answer:
    return Answer(
        question=question,
        verdict=Verdict.REFUSE,
        intent=r.intent,
        income_year=r.income_year,
        income_year_assumed=r.income_year_assumed,
        text=reason,
        trace_id=audit.new_trace_id(),
        latency_ms=(time.perf_counter() - started) * 1000,
    )


def _abstain(question: str, r: Route, reason: str, retrieved: list[Retrieved],
             started: float, guard_results=None) -> Answer:
    return Answer(
        question=question,
        verdict=Verdict.ABSTAIN,
        intent=r.intent,
        income_year=r.income_year,
        income_year_assumed=r.income_year_assumed,
        text=f"Not answered. {reason}",
        guards=guard_results or [],
        sources=[{"id": x.chunk.id, "citation": x.chunk.citation, "url": x.chunk.source_url,
                  "income_years": x.chunk.income_years, "checked_on": x.chunk.retrieved_at}
                 for x in retrieved[:3]],
        trace_id=audit.new_trace_id(),
        latency_ms=(time.perf_counter() - started) * 1000,
    )


def ask(
    question: str,
    index: HybridIndex | None = None,
    llm: LLM | None = None,
    trace: bool = True,
) -> Answer:
    started = time.perf_counter()
    llm = llm or get_llm()
    r = route(question)
    caveats: list[str] = []

    if r.intent is Intent.OUT_OF_SCOPE:
        # The router's keyword pre-filter is cheap and therefore imprecise: a
        # real tax question can avoid every keyword ("when do I need written
        # evidence for work-related expenses?"). So an apparent out-of-scope
        # question is confirmed against retrieval before being refused, and
        # re-routed as guidance if the corpus clearly does cover it. Refusing
        # on the keyword list alone was turning good questions away.
        rescued = False
        if index is not None:
            probe = index.search(question)
            # A higher bar than the answer threshold: rescuing on a marginal
            # match is how "how do I get a crust on sourdough?" ends up in the
            # guidance path at all. Clear coverage, or refuse.
            if probe and probe[0].score >= settings.min_rescue_score:
                r = Route(
                    Intent.GUIDANCE,
                    settings.default_income_year,
                    income_year_assumed=True,
                    threshold_key=detect_threshold(question),
                    reason="No tax keyword matched, but the corpus covers this closely.",
                )
                rescued = True
        if not rescued:
            ans = _refuse(question, r, "Out of scope: this tool only covers Australian tax "
                                       "topics documented in the indexed ATO guidance.", started)
            if trace:
                audit.write(ans, [], llm.name)
            return ans

    if r.intent is Intent.ADVICE_SEEKING:
        ans = _refuse(question, r, REFERRAL, started)
        if trace:
            audit.write(ans, [], llm.name)
        return ans

    if r.income_year_assumed and r.income_year:
        caveats.append(
            f"No income year was specified, so this answer is for {r.income_year}. "
            f"Rates and thresholds change between years."
        )

    # --- deterministic branch -------------------------------------------------
    if r.intent is Intent.CALCULATION and r.amount is not None:
        try:
            comp = (
                marginal_rate(r.amount, r.income_year)
                if "marginal" in question.lower()
                else income_tax(r.amount, r.income_year)
            )
        except CalculationRefused as exc:
            ans = _abstain(question, r, str(exc), [], started)
            if trace:
                audit.write(ans, [], llm.name)
            return ans

        chunk = _computation_chunk(comp)
        retrieved = [_as_retrieved(chunk)]
        summary = (
            f"Tax payable on ${r.amount:,.0f} for {comp.income_year} is "
            f"${comp.value:,.2f}."
            if comp.kind == "resident_income_tax"
            else f"The marginal rate on ${r.amount:,.0f} for {comp.income_year} is "
                 f"{comp.value:.4g}%."
        )
        claims = [Claim(text=summary, citations=[chunk.id])] + [
            Claim(text=w, citations=[chunk.id]) for w in comp.working
        ]
        if comp.excludes:
            caveats.append("Excludes " + ", ".join(comp.excludes) + ".")
        caveats.append(f"Rates checked against {comp.qc_ref} on {comp.checked_on}.")

        guard_results = guards.run_all(
            DraftAnswer(claims=claims, income_year=comp.income_year),
            retrieved, r.income_year, question,
        )
        ans = Answer(
            question=question, verdict=Verdict.ANSWER, intent=r.intent,
            income_year=comp.income_year, income_year_assumed=r.income_year_assumed,
            text=_render(claims, caveats), claims=claims,
            sources=_sources(retrieved, claims), guards=guard_results, caveats=caveats,
            trace_id=audit.new_trace_id(),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        if trace:
            audit.write(ans, retrieved, "deterministic-calculator")
        return ans

    # --- retrieval branch -----------------------------------------------------
    if index is None:
        raise ValueError("a retrieval index is required for guidance questions")

    deterministic: list[Claim] = []
    det_retrieved: list[Retrieved] = []
    if r.threshold_key:
        try:
            comp = lookup(r.threshold_key, r.income_year)
            chunk = _computation_chunk(comp)
            det_retrieved = [_as_retrieved(chunk)]
            deterministic = [Claim(text=comp.working[0], citations=[chunk.id])]
            if comp.notes:
                caveats.append(comp.notes)
        except CalculationRefused as exc:
            # A year-scoped value asked for outside its stated years is refused
            # rather than approximated. This is the superseded-rate case.
            ans = _abstain(question, r, str(exc), [], started)
            if trace:
                audit.write(ans, [], llm.name)
            return ans

    retrieved = _prune(index.search(question))
    if not retrieved or retrieved[0].score < settings.min_retrieval_score:
        best = retrieved[0].score if retrieved else 0.0
        ans = _abstain(
            question, r,
            f"The indexed guidance does not cover this closely enough to answer "
            f"(best retrieval score {best:.3f} < {settings.min_retrieval_score}).",
            retrieved, started,
        )
        if trace:
            audit.write(ans, retrieved, llm.name)
        return ans

    all_retrieved = det_retrieved + retrieved
    draft = llm.draft(question, retrieved, r.income_year)
    if deterministic:
        draft.claims = deterministic + draft.claims

    if draft.insufficient_context or not draft.claims:
        ans = _abstain(question, r, "The retrieved passages do not support an answer.",
                       retrieved, started)
        if trace:
            audit.write(ans, retrieved, llm.name)
        return ans

    guard_results = guards.run_all(
        draft, all_retrieved, r.income_year, question, index.vocabulary
    )
    failed = {g.name for g in guard_results if not g.passed}
    blocking = failed & set(guards.BLOCKING)

    if blocking:
        detail = "; ".join(g.detail for g in guard_results if g.name in blocking and g.detail)
        ans = _abstain(
            question, r,
            f"Draft answer failed verification ({', '.join(sorted(blocking))}): {detail}",
            all_retrieved, started, guard_results,
        )
        if trace:
            audit.write(ans, all_retrieved, llm.name)
        return ans

    advisory = failed & set(guards.ADVISORY)
    if advisory:
        caveats.append(
            "Some statements paraphrase the source loosely; check the cited passages."
        )
    caveats.append(REFERRAL)

    ans = Answer(
        question=question,
        verdict=Verdict.ANSWER_WITH_CAVEAT if advisory else Verdict.ANSWER,
        intent=r.intent,
        income_year=r.income_year,
        income_year_assumed=r.income_year_assumed,
        text=_render(draft.claims, caveats),
        claims=draft.claims,
        sources=_sources(all_retrieved, draft.claims),
        guards=guard_results,
        caveats=caveats,
        trace_id=audit.new_trace_id(),
        latency_ms=(time.perf_counter() - started) * 1000,
    )
    if trace:
        audit.write(ans, all_retrieved, llm.name)
    return ans
