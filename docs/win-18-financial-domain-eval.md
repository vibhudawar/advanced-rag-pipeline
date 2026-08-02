# Win 18 — Financial domain evaluation + boilerplate hygiene

## Why
Every quality number before this was measured on **scifact** (a scientific benchmark). "Reliable
like AlphaSense" was a claim, not a measurement, on the actual target domain (analyst reports).
This win builds a financial eval set + two new multi-doc metrics, measures the pipeline on it,
and fixes the biggest weakness the numbers exposed.

## What was added

### 1. Domain eval harness (committed — generic)
- `GoldenItem.expected_sources` — filenames the answer *should* cite. Chunk-level retrieval
  ground truth is impractical for multi-doc questions, so multi-doc recall is measured at the
  **document** level instead.
- `metrics.citation_validity` (pure) — every inline `[n]` resolves to a provided source, and an
  abstention carries **no** citations. Un-gameable.
- `metrics.source_coverage` (pure) — fraction of `expected_sources` actually cited; the
  document-level stand-in for multi-doc recall.
- `RunResult.citations` threaded through the production adapter; `evaluate.py` computes the new
  metrics and dumps **per-item** scores (answer, cited sources, contexts on low-faithfulness
  items) into the report for diagnosis without a re-run.
- `q_type` extended with `comparison` / `cross_company`.

### 2. Financial golden set (gitignored — `data/fin_golden.jsonl`)
16 questions over 6 real reports (Meta ×4 publishers, Alphabet ×2 sources): simple factoids,
named comparisons, aggregations across all four Meta reports, cross-company, and unanswerables.
Derived from copyrighted third-party research, so **never committed** (same rule as
`data/golden.jsonl`).

### 3. Boilerplate stripper (`src/ingestion/boilerplate.py`, `STRIP_BOILERPLATE=true`)
Analyst PDFs carry heavy non-content noise — a per-page licensing watermark, a front-cover
Reg-AC footer, and a long trailing **disclosures / jurisdiction / other-ticker rating-table**
section. Chunked and embedded, it crowds out the report's thesis and causes citation
misattribution (answering "Alphabet's rating" from a bank's coverage-disclosure table). The
stripper removes it conservatively:
- **line filter** — drop watermark / front-matter footer lines anywhere;
- **two-tier section cut** — truncate the trailing disclosure section at its heading. *Strong*
  headings (`Important Disclosures`, `Analyst Certification`, `Explanation of Equity Research
  Ratings`, …) are unambiguous legal-tail markers, cut at first occurrence with only a
  minimum-content floor; *weak* headings (`rating system`, …) require a latter-half position.
Never touches financial tables or the page-1 rating (verified: every rating/target survives).
Best-effort — returns raw text on any error, never fails ingestion.

## Measured

### Baseline — the scifact 1.0 was misleading
Production on `fin_golden` (full contextualization): **faithfulness 0.83–0.86**, answer relevance
~0.94, **citation validity 1.0**, source coverage ~0.90. Faithfulness on real reports is well
below the synthetic 1.0 — numbers/figures are harder to ground.

### Stripper A/B (contextualization held OFF in both arms to isolate the variable)
| Metric | no strip | strip | Δ |
|---|---|---|---|
| faithfulness | 0.692 | 0.692 | flat |
| answer relevance | 0.908 | 0.931 | +0.023 |
| context relevance | 0.700 | 0.700 | flat |
| abstention accuracy | 0.750 | 0.813 | +0.063 |
| citation validity | 1.000 | 1.000 | flat |
| source coverage | 0.615 | 0.692 | +0.077 |

Net positive, zero regression. Per-item, it **decisively fixed the headline failure**: fin-05
("why is Alphabet 'misunderestimated'?") went from a wrong abstention (retrieval returned a
financials table + disclosure page) to a correct, cited answer — faithfulness 0.0 → 1.0. It also
stopped fin-13 from citing a disclosure rating-table.

### Two findings for later
- **Contextualization is the dominant faithfulness lever** (0.69 without → 0.83–0.86 with) — far
  more than the stripper.
- **Contextualization has a TPM scaling limit**: a single large doc (122 chunks × full-doc
  prefix) exceeds 200k tokens/min and degrades to header-less chunks. A real blocker for the
  100–150-doc goal — Phase 3B (scale) should make contextualization batch/throttle-safe.

### Known hard case
`fin-10` ("range of price targets, most/least bullish") — numeric aggregation across documents.
Stripping the price-*history* table removed one source it had leaned on; the current per-analyst
targets remain on page 1. Cross-document numeric aggregation is a distinct, harder problem.
