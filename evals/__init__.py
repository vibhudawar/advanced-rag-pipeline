"""RAG evaluation harness (WIN 1 of plan-v2.md).

Two dataset sources:
  - own corpus  -> data/golden.jsonl        (evals.generate_golden)
  - BEIR subset -> data/beir/<name>/...      (evals.load_beir)

Metrics:
  - retrieval  : hit-rate@k, MRR, nDCG@k          (evals.metrics, pure math, no LLM)
  - generation : faithfulness, answer relevance,
                 context relevance, abstention     (evals.judges, hand-rolled LLM judge)

Run a baseline scoreboard with:
    python -m evals.evaluate --index <your-index-name>
"""
