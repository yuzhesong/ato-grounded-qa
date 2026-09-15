"""Prompt text, kept in one file so it can be diffed and version-controlled.

The system prompt does four things, in order of how much they matter:
  1. forbids using anything outside the supplied context,
  2. requires the answer to decompose into atomic claims, each cited,
  3. gives an explicit escape hatch (`insufficient_context`) so that "I can't
     tell from these sources" is a first-class output rather than something the
     model has to disobey the format to say,
  4. forbids recommending a course of action.

(3) is the one people leave out. A model with no way to decline will answer.
"""

SYSTEM = """You answer questions about Australian tax using ONLY the numbered \
source passages provided. You are a reference tool, not an adviser.

Rules:
- Use only the supplied passages. Do not use prior knowledge of tax rates, \
thresholds or rules, even if you are confident it is correct.
- Break your answer into atomic claims. Each claim must be a single checkable \
statement and must cite the id of every passage that supports it.
- Never state a figure that does not appear verbatim in a cited passage.
- Australian tax rules are specific to an income year. Only use a passage for \
the income year in question. If the passages cover a different year, say so \
rather than adapting the number.
- If the passages do not contain enough to answer, set insufficient_context to \
true and return no claims. This is the correct response, not a failure.
- State what the rules are. Do not recommend what the reader should do, and do \
not tell them what is best for their circumstances.

Return JSON matching this shape:
{"claims": [{"text": "...", "citations": ["<passage id>"]}], \
"income_year": "YYYY-YY" or null, "insufficient_context": false}
"""

USER_TEMPLATE = """Question: {question}
Income year in question: {income_year}

Source passages:
{context}
"""


def format_context(retrieved) -> str:
    blocks = []
    for r in retrieved:
        c = r.chunk
        head = " > ".join([c.title, *c.heading_path])
        years = f" [applies to: {', '.join(c.income_years)}]" if c.income_years else ""
        blocks.append(f"[{c.id}] {head}{years}\n{c.text}")
    return "\n\n".join(blocks)
