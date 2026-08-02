"""Strip legal/disclosure boilerplate from analyst reports before chunking.

Analyst research PDFs carry heavy non-content noise: a per-page licensing watermark, a
front-cover Reg-AC/"see disclosures" footer, and — most damagingly — a long trailing section
of legal disclosures, jurisdiction notices, rating-system explanations, and *other-ticker*
rating tables. Chunked and embedded, these compete with (and crowd out) the report's actual
thesis, and the other-ticker rating tables cause citation misattribution (e.g. answering
"Alphabet's rating" from a bank's coverage-disclosure table for an unrelated report).

This module removes that noise conservatively (high precision — never touch financial tables
or the page-1 thesis/rating). Two mechanisms:
  1. line filter  — drop repeated watermark / front-matter footer lines anywhere in the doc.
  2. section cut  — truncate the trailing disclosure section, but ONLY when its heading sits
                    in the latter part of the document and enough content remains, so a
                    body mention of "disclosure" can never gut the report.

Best-effort: on any unexpected input it returns the original text (ingestion must not fail
because cleaning tripped).
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("rag.ingestion.boilerplate")

# Lines matching any of these are pure front-matter/footer noise, dropped wherever they occur.
# High precision: each phrase is unambiguous boilerplate, not report content.
_LINE_NOISE = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"provided for the exclusive use of",              # per-page licensing watermark
        r"for reg\.?\s*ac certification",                  # front-cover pointer to disclosures
        r"see page\s+\d+\s+for analyst certification",     # front-cover pointer
        r"please see important information about this report",
        r"^\s*[*#>\s]*regulation ac\b.*\bcertif",          # Reg-AC certification sentence (cover)
    )
]

# STRONG heads: unambiguous legal-tail titles that never appear in thesis prose. Cutting at the
# first of these (heading-styled) needs only a minimum-content floor — no fraction guard — so
# reports that are mostly disclosures (e.g. a 14-page bank note) still get their tail removed.
_STRONG_HEADS = (
    "important disclosures",
    "required disclosures",
    "analyst certification",
    "explanation of equity research ratings",
    "legal entities disclosures",
    "disclosure appendix",
    "regulation ac",
    "reg ac",
)
# WEAK heads: plausible section titles that could also be phrases; only trusted as a cut point
# when they sit in the document's latter part (fraction guard), and only if no strong head found.
_WEAK_HEADS = (
    "other disclosures",
    "global research disclosures",
    "the argus research rating system",
    "rating system",
    "disclosures",
)

_HEAD_STRIP = " *#>_\t"

# Guards against removing signal.
_MIN_KEPT_CHARS = 500      # never reduce a document below this
_STRONG_FLOOR_FRAC = 0.12  # a strong head must still leave at least this much page-1 content
_WEAK_MIN_FRACTION = 0.55  # a weak head only counts as the tail if it's past this fraction


def _normalize_head(line: str) -> str:
    return line.strip().strip(_HEAD_STRIP).strip().lower()


def _head_tier(line: str) -> str | None:
    """Return 'strong' / 'weak' / None for whether the line is a disclosure section heading."""
    norm = _normalize_head(line)
    if len(norm) > 60:
        return None
    is_headingish = line.lstrip().startswith("#") or line.strip().startswith("**") or len(line) < 60
    if not is_headingish:
        return None
    if any(norm.startswith(h) for h in _STRONG_HEADS):
        return "strong"
    if any(norm.startswith(h) for h in _WEAK_HEADS):
        return "weak"
    return None


def _find_cut(text: str) -> int | None:
    """Character offset where the trailing disclosure section begins, or None."""
    total = len(text)
    strong_floor = max(_MIN_KEPT_CHARS, total * _STRONG_FLOOR_FRAC)
    weak_floor = total * _WEAK_MIN_FRACTION
    running = 0
    for ln in text.splitlines():
        start = running
        running += len(ln) + 1
        tier = _head_tier(ln)
        if tier == "strong" and start >= strong_floor:
            return start
        if tier == "weak" and start >= weak_floor:
            return start
    return None


def strip_boilerplate(text: str) -> str:
    """Return `text` with watermark/footer lines and the trailing disclosure section removed.
    Returns the input unchanged on empty/odd input or if the safeguards decline the cut."""
    if not text or not text.strip():
        return text
    try:
        cut_at = _find_cut(text)
        body = text[:cut_at] if (cut_at is not None and cut_at >= _MIN_KEPT_CHARS) else text
        kept = [ln for ln in body.splitlines() if not any(p.search(ln) for p in _LINE_NOISE)]
        cleaned = "\n".join(kept)
        return cleaned if cleaned.strip() else text
    except Exception:  # noqa: BLE001 - cleaning is best-effort; never fail ingestion over it
        logger.info("boilerplate stripping failed; using raw text")
        return text
