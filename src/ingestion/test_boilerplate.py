"""Tests for the analyst-report boilerplate stripper (Win 18)."""
from src.ingestion.boilerplate import strip_boilerplate


def test_drops_watermark_lines_anywhere():
    text = (
        "# Alphabet: Misunderestimated\n"
        "Provided for the exclusive use of Jane Doe at Acme on 02-Aug-2026\n"
        "Our thesis is that cloud growth is underappreciated.\n"
        "Provided for the exclusive use of Jane Doe at Acme on 02-Aug-2026\n"
        "We rate the stock Buy with a target of $250.\n"
    )
    out = strip_boilerplate(text)
    assert "exclusive use" not in out.lower()
    assert "cloud growth is underappreciated" in out
    assert "$250" in out


def test_cuts_trailing_disclosure_section_strong_head():
    body = "The company's revenue grew 20%. " * 40  # substantial content
    tail = (
        "\n##### **Important Disclosures**\n"
        "J.P. Morgan does and seeks to do business with companies covered.\n"
        "**Analysts' Compensation:** The research analysts responsible...\n"
        "Alphabet (Buy), Target Price $300 — coverage disclosure table\n"
    )
    out = strip_boilerplate(body + tail)
    assert "Important Disclosures" not in out
    assert "coverage disclosure table" not in out
    assert "revenue grew 20%" in out  # thesis kept


def test_keeps_short_docs_untouched_when_no_disclosure_tail():
    text = "Meta is rated BUY with a $800 target. Margins compressed to 31%."
    assert strip_boilerplate(text) == text


def test_body_mention_of_disclosure_does_not_gut_document():
    # A pointer early in the doc must NOT trigger a cut that removes the real content.
    text = (
        "See page 8 for analyst certification and important disclosures.\n"
        + "Our detailed investment thesis follows. " * 60
    )
    out = strip_boilerplate(text)
    assert "investment thesis follows" in out
    assert len(out) > 0.5 * len(text)  # the pointer line at the top didn't cause a big cut


def test_empty_and_none_safe():
    assert strip_boilerplate("") == ""
    assert strip_boilerplate("   ") == "   "
