"""Ingestion: fetch, normalise, chunk.

ATO material is fetched on demand into ``data/raw/`` (git-ignored) and is
deliberately not redistributed in this repository -- see docs/governance.md.
Unit tests and CI run against the synthetic fixture corpus in
``tests/fixtures/corpus`` instead, so the test suite needs no network.

Chunking is structure-aware rather than fixed-width: ATO pages are organised
under headings that carry meaning ("Eligibility", "What the rate covers"), and
splitting across a heading boundary is what produces chunks that read as
authoritative while answering a different question than the one asked. Each
chunk keeps its heading path for citation and is stamped with the income years
it mentions, which is what the superseded-rate guard checks against.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from .config import settings
from .router import extract_all_income_years
from .schemas import Chunk

# A small, fixed set of public ATO pages. Kept explicit rather than crawled:
# a portfolio demo has no business spidering a government site.
SEED_URLS: tuple[str, ...] = (
    "https://www.ato.gov.au/tax-rates-and-codes/tax-rates-australian-residents",
    "https://www.ato.gov.au/businesses-and-organisations/gst-excise-and-indirect-taxes/gst/registering-for-gst",
    "https://www.ato.gov.au/businesses-and-organisations/super-for-employers/paying-super-contributions/how-much-super-to-pay",
    "https://www.ato.gov.au/individuals-and-families/income-deductions-offsets-and-records/deductions-you-can-claim/working-from-home-expenses/fixed-rate-method-67-cents",
)

_QC = re.compile(r"\bQC\s?(\d{4,6})\b")
_WS = re.compile(r"[ \t ]+")
MAX_WORDS = 180
OVERLAP_WORDS = 30


def _slug(url: str) -> str:
    return hashlib.blake2b(url.encode(), digest_size=8).hexdigest()


def fetch(urls: tuple[str, ...] = SEED_URLS, delay: float = 1.5) -> Path:
    """Download pages into data/raw with a manifest. Polite, serial, cached."""
    raw = Path(settings.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = raw / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}

    headers = {"User-Agent": settings.user_agent}
    with httpx.Client(headers=headers, timeout=30.0, follow_redirects=True) as client:
        for i, url in enumerate(urls):
            target = raw / f"{_slug(url)}.html"
            if target.exists():
                continue
            if i:
                time.sleep(delay)
            resp = client.get(url)
            resp.raise_for_status()
            target.write_text(resp.text, encoding="utf-8")
            manifest[target.name] = {
                "url": url,
                "fetched_at": datetime.now(UTC).isoformat(),
                "sha256": hashlib.sha256(resp.content).hexdigest(),
                "status": resp.status_code,
            }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return raw


def _clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


def parse_html(html: str, url: str, fetched_at: str | None = None) -> list[Chunk]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    h1 = soup.find("h1")
    title = _clean(h1.get_text()) if h1 else _clean(soup.title.get_text() if soup.title else url)
    page_text = soup.get_text(" ", strip=True)
    qc_match = _QC.search(page_text)
    qc_ref = f"QC{qc_match.group(1)}" if qc_match else None

    body = soup.find("main") or soup.body or soup
    sections: list[tuple[list[str], list[str]]] = [([], [])]
    path: list[str] = []
    for el in body.find_all(["h2", "h3", "h4", "p", "li", "td", "th"]):
        name = el.name
        text = _clean(el.get_text(" ", strip=True))
        if not text:
            continue
        if name in ("h2", "h3", "h4"):
            depth = int(name[1]) - 2
            path = path[:depth] + [text]
            sections.append((list(path), []))
        else:
            sections[-1][1].append(text)

    return _sections_to_chunks(sections, title, url, qc_ref, fetched_at)


def _sections_to_chunks(
    sections: list[tuple[list[str], list[str]]],
    title: str,
    url: str,
    qc_ref: str | None,
    fetched_at: str | None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path, paras in sections:
        words: list[str] = []
        for para in paras:
            words.extend(para.split())
        if not words:
            continue
        step = max(MAX_WORDS - OVERLAP_WORDS, 1)
        for start in range(0, len(words), step):
            window = words[start : start + MAX_WORDS]
            if len(window) < 12 and chunks:
                break
            text = " ".join(window)
            cid = f"{_slug(url)}#{len(chunks):03d}"
            chunks.append(
                Chunk(
                    id=cid,
                    text=text,
                    title=title,
                    source_url=url,
                    qc_ref=qc_ref,
                    heading_path=path,
                    income_years=extract_all_income_years(text),
                    retrieved_at=fetched_at,
                )
            )
            if start + MAX_WORDS >= len(words):
                break
    return chunks


def load_raw(directory: Path | None = None) -> list[Chunk]:
    raw = Path(directory or settings.raw_dir)
    manifest_path = raw / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No corpus at {raw}. Run `make fetch` first, or point at the fixture "
            f"corpus with --corpus tests/fixtures/corpus."
        )
    manifest = json.loads(manifest_path.read_text("utf-8"))
    chunks: list[Chunk] = []
    for name, meta in manifest.items():
        path = raw / name
        if not path.exists():
            continue
        chunks.extend(parse_html(path.read_text("utf-8"), meta["url"], meta.get("fetched_at")))
    return chunks


# --- fixture corpus ---------------------------------------------------------

_FRONT = re.compile(r"^---\n(.*?)\n---\n", re.S)


def load_markdown_corpus(directory: Path | str) -> list[Chunk]:
    """Load the synthetic fixture corpus: markdown with simple front matter.

    Front matter keys: title, source_url, qc_ref. Body sections are delimited
    by '## ' headings, mirroring the structure of a real ATO page.
    """
    import yaml

    directory = Path(directory)
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        if path.stem.lower() in {"readme", "index"} or path.name.startswith("_"):
            continue
        text = path.read_text("utf-8")
        m = _FRONT.match(text)
        meta = yaml.safe_load(m.group(1)) if m else {}
        body = text[m.end() :] if m else text

        sections: list[tuple[list[str], list[str]]] = [([], [])]
        for line in body.splitlines():
            if line.startswith("## "):
                sections.append(([line[3:].strip()], []))
            elif line.strip():
                sections[-1][1].append(line.strip())

        page = _sections_to_chunks(
            sections,
            title=meta.get("title", path.stem),
            url=meta.get("source_url", f"fixture://{path.name}"),
            qc_ref=meta.get("qc_ref"),
            # YAML parses a bare date into datetime.date; the schema wants a string.
            fetched_at=str(meta["checked_on"]) if meta.get("checked_on") else None,
        )
        chunks.extend(page)
    if not chunks:
        raise FileNotFoundError(f"No fixture documents found in {directory}")
    return chunks
