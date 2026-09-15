"""Generation backends.

`ExtractiveStub` is the default and needs no credentials. It selects supporting
sentences from the retrieved passages, so it is grounded by construction. That
makes it a useful *floor*: any real model that scores worse than the stub on
the evaluation harness is doing harm, and the stub lets the whole pipeline --
routing, retrieval, guards, audit, metrics -- run deterministically in CI.

`AdversarialStub` deliberately produces the failure modes the guards exist to
catch. Without it, a green test suite would only prove the guards do not fire
on good output, which is the easy half. It is used by the guard-sensitivity
suite in the evaluation harness.
"""

from __future__ import annotations

import re
from typing import Protocol

from .config import settings
from .guards import _content_words, extract_numbers
from .prompts import SYSTEM, USER_TEMPLATE, format_context
from .schemas import Claim, DraftAnswer, Retrieved

_SENT = re.compile(r"(?<=[.!?])\s+")
MAX_CLAIMS = 3
# A supporting sentence must score at least this fraction of the best sentence
# to be included at all.
RELATIVE_CLAIM_CUTOFF = 0.55


class LLM(Protocol):
    name: str

    def draft(
        self, question: str, retrieved: list[Retrieved], income_year: str | None
    ) -> DraftAnswer: ...


class ExtractiveStub:
    name = "extractive-stub"

    def __init__(self, max_claims: int = MAX_CLAIMS) -> None:
        self.max_claims = max_claims

    def draft(
        self, question: str, retrieved: list[Retrieved], income_year: str | None
    ) -> DraftAnswer:
        if not retrieved or retrieved[0].score < settings.min_retrieval_score:
            return DraftAnswer(claims=[], income_year=income_year, insufficient_context=True)

        q_words = _content_words(question)
        scored: list[tuple[float, str, str]] = []
        for r in retrieved[:4]:
            for sent in _SENT.split(r.chunk.text):
                sent = sent.strip()
                if len(sent.split()) < 5:
                    continue
                s_words = _content_words(sent)
                if not s_words:
                    continue
                overlap = len(q_words & s_words) / max(len(q_words), 1)
                # Prefer sentences carrying a figure: for this corpus the answer
                # to a threshold or rate question is almost always numeric.
                bonus = 0.12 if extract_numbers(sent) else 0.0
                scored.append((overlap + bonus + 0.02 * (1.0 - r.rank / 10), sent, r.chunk.id))

        scored.sort(key=lambda t: -t[0])
        # Keep only sentences comparable to the best one. Without this, a third
        # and fourth claim get padded in on the strength of shared function
        # words -- a super guarantee question was picking up a working-from-home
        # sentence because both mention a "rate". Relative, not absolute,
        # because the absolute scale shifts with question length.
        cutoff = max(0.05, scored[0][0] * RELATIVE_CLAIM_CUTOFF) if scored else 0.05
        claims: list[Claim] = []
        seen: set[str] = set()
        for score, sent, cid in scored:
            if score < cutoff or sent in seen:
                continue
            seen.add(sent)
            claims.append(Claim(text=sent, citations=[cid]))
            if len(claims) >= self.max_claims:
                break

        if not claims:
            return DraftAnswer(claims=[], income_year=income_year, insufficient_context=True)
        return DraftAnswer(claims=claims, income_year=income_year)


class AdversarialStub:
    """Injects one specific failure mode into an otherwise grounded draft."""

    MODES = ("fabricate_number", "bogus_citation", "stale_year", "advice", "ungrounded")

    def __init__(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {self.MODES}")
        self.mode = mode
        self.name = f"adversarial-stub:{mode}"
        self._base = ExtractiveStub()
        # Set per call. Some modes cannot inject a fault into some questions
        # (e.g. stale_year when every retrieved passage is year-agnostic). The
        # evaluation harness reads this and excludes non-injections, so guard
        # recall is measured over faults that were actually introduced.
        self.injected = False

    def draft(
        self, question: str, retrieved: list[Retrieved], income_year: str | None
    ) -> DraftAnswer:
        draft = self._base.draft(question, retrieved, income_year)
        self.injected = False
        if not draft.claims:
            return draft
        first = draft.claims[0]

        if self.mode == "fabricate_number":
            nums = extract_numbers(first.text)
            if nums:
                raw = nums[0][2]
                digits = re.search(r"\d[\d,]*(?:\.\d+)?", raw)
                if digits:
                    bumped = str(int(float(digits.group(0).replace(",", "")) + 7))
                    first.text = first.text.replace(raw, raw.replace(digits.group(0), bumped), 1)
                    self.injected = True
            else:
                first.text += " The applicable threshold is $91,300."
                self.injected = True
        elif self.mode == "bogus_citation":
            first.citations = ["nonexistent#000"]
            self.injected = True
        elif self.mode == "stale_year":
            other = next(
                (r for r in retrieved if r.chunk.income_years and income_year
                 not in r.chunk.income_years),
                None,
            )
            if other is None:
                return draft
            first.citations = [other.chunk.id]
            first.text = other.chunk.text.split(".")[0].strip() + "."
            self.injected = True
        elif self.mode == "advice":
            first.text += " You should claim this method to reduce your tax."
            self.injected = True
        elif self.mode == "ungrounded":
            first.text = (
                "Eligibility is determined by the commissioner's discretionary "
                "assessment of proportionate apportionment across the relevant period."
            )
            self.injected = True
        return draft


class OpenAIChat:
    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model or settings.chat_model

    def _complete(self, question: str, retrieved: list[Retrieved], income_year: str | None) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": USER_TEMPLATE.format(
                        question=question,
                        income_year=income_year or "not specified",
                        context=format_context(retrieved),
                    ),
                },
            ],
        )
        return resp.choices[0].message.content or "{}"

    def draft(
        self, question: str, retrieved: list[Retrieved], income_year: str | None
    ) -> DraftAnswer:
        import json

        raw = self._complete(question, retrieved, income_year)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            # Malformed structured output is an abstention, not a crash.
            return DraftAnswer(claims=[], income_year=income_year, insufficient_context=True)
        try:
            return DraftAnswer.model_validate(payload)
        except Exception:
            return DraftAnswer(claims=[], income_year=income_year, insufficient_context=True)


class AzureOpenAIChat(OpenAIChat):
    name = "azure"

    def __init__(self, deployment: str | None = None) -> None:
        import os

        from openai import AzureOpenAI

        self._client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01"),
        )
        self._model = deployment or settings.chat_deployment


def get_llm() -> LLM:
    if settings.provider == "openai":
        return OpenAIChat()
    if settings.provider == "azure":
        return AzureOpenAIChat()
    return ExtractiveStub()
