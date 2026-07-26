"""Run a pipeline over the golden set and print the scoreboard.

    python -m evals.evaluate --index <name>                # baseline, full judge
    python -m evals.evaluate --index <name> --no-judge      # retrieval metrics only (fast, no LLM cost)
    python -m evals.evaluate --index <name> --limit 20      # quick smoke run

Retrieval metrics (hit@k / MRR / nDCG@k) are computed on the pre-rerank candidate set of
answerable items. Generation metrics (faithfulness / answer & context relevance) are judged
on answerable items only. Abstention accuracy is computed over ALL items: an unanswerable
item is correct iff the system abstained; an answerable item is correct iff it did not.

Writes a JSON report to evals/reports/ so runs are comparable over time — this is the
before/after scoreboard the README table is built from.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .metrics import hit_rate_at_k, mean, ndcg_at_k, reciprocal_rank
from .pipeline_adapter import PIPELINES
from .schema import ItemScore, read_jsonl

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    def tqdm(x, **_):  # type: ignore
        return x


def evaluate(index: str, pipeline_name: str, golden_path: str, k: int,
             use_judge: bool, judge_model: str | None, limit: int | None,
             retrieval_only: bool = False) -> dict:
    items = read_jsonl(golden_path)
    if limit:
        items = items[:limit]
    if retrieval_only:
        use_judge = False
    reviewed = sum(1 for it in items if not it.needs_review)
    print(f"[eval] {len(items)} golden items ({reviewed} human-reviewed) | pipeline={pipeline_name} | judge={'on' if use_judge else 'off'} | retrieval_only={retrieval_only}")

    kwargs = {"retrieval_only": True} if retrieval_only else {}
    pipeline = PIPELINES[pipeline_name](index_name=index, **kwargs)
    judge = None
    if use_judge:
        from .judges import Judge
        judge = Judge(model=judge_model)
        from config import OPENAI_GENERATION_MODEL
        print(f"[eval] judge model = {judge.model_name} (generator = {OPENAI_GENERATION_MODEL})")

    scores: list[ItemScore] = []
    errors = 0
    for it in tqdm(items, desc="scoring"):
        res = pipeline.run(it.query)
        if res.error:
            errors += 1
        s = ItemScore(id=it.id, q_type=it.q_type)

        if it.is_answerable:
            rel = set(it.relevant_chunk_hashes)
            s.hit_at_k = hit_rate_at_k(res.candidate_hashes, rel, k)
            s.mrr = reciprocal_rank(res.candidate_hashes, rel)
            s.ndcg_at_k = ndcg_at_k(res.candidate_hashes, rel, k)

        if judge is not None and not res.error:
            s.abstained = judge.detect_abstention(res.answer) if res.answer else True
            if it.q_type == "unanswerable":
                s.abstention_correct = bool(s.abstained)
            else:
                s.abstention_correct = not s.abstained
                # generation quality only meaningful for answerable items with an answer
                if res.answer and res.contexts:
                    s.faithfulness = judge.faithfulness(res.answer, res.contexts).score
                    s.answer_relevance = judge.answer_relevance(it.query, res.answer).score
                    s.context_relevance = judge.context_relevance(it.query, res.contexts).score

        s.notes["latency_s"] = round(res.latency_s, 2)
        scores.append(s)

    return _aggregate(scores, k, errors, pipeline_name, index)


def _aggregate(scores: list[ItemScore], k: int, errors: int, pipeline_name: str, index: str) -> dict:
    ans = [s for s in scores if s.q_type != "unanswerable"]
    report = {
        "pipeline": pipeline_name,
        "index": index,
        "n_items": len(scores),
        "n_errors": errors,
        "k": k,
        "retrieval": {
            f"hit_rate@{k}": round(mean([s.hit_at_k for s in ans]), 4),
            "mrr": round(mean([s.mrr for s in ans]), 4),
            f"ndcg@{k}": round(mean([s.ndcg_at_k for s in ans]), 4),
        },
        "generation": {
            "faithfulness": round(mean([s.faithfulness for s in ans]), 4),
            "answer_relevance": round(mean([s.answer_relevance for s in ans]), 4),
            "context_relevance": round(mean([s.context_relevance for s in ans]), 4),
        },
        "behaviour": {
            "abstention_accuracy": round(mean([1.0 if s.abstention_correct else 0.0
                                               for s in scores if s.abstention_correct is not None]), 4),
        },
        "latency_s_mean": round(mean([s.notes.get("latency_s") for s in scores]), 2),
    }
    return report


def _print_table(r: dict) -> None:
    ret, gen, beh = r["retrieval"], r["generation"], r["behaviour"]
    print("\n" + "=" * 60)
    print(f" SCOREBOARD — {r['pipeline']}  (n={r['n_items']}, errors={r['n_errors']})")
    print("=" * 60)
    rows = [
        (f"hit_rate@{r['k']}", ret[f"hit_rate@{r['k']}"]),
        ("MRR", ret["mrr"]),
        (f"nDCG@{r['k']}", ret[f"ndcg@{r['k']}"]),
        ("faithfulness", gen["faithfulness"]),
        ("answer_relevance", gen["answer_relevance"]),
        ("context_relevance", gen["context_relevance"]),
        ("abstention_accuracy", beh["abstention_accuracy"]),
        ("latency_s (mean)", r["latency_s_mean"]),
    ]
    for name, val in rows:
        print(f"  {name:<22} {val:>8}")
    print("=" * 60 + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Score a pipeline against the golden set.")
    p.add_argument("--index", default=os.getenv("EVAL_INDEX"), help="Pinecone index (or set EVAL_INDEX)")
    p.add_argument("--pipeline", default="baseline", choices=list(PIPELINES))
    p.add_argument("--golden", default="data/golden.jsonl")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--no-judge", action="store_true", help="skip LLM judging (still generates answers)")
    p.add_argument("--retrieval-only", action="store_true",
                   help="pure retrieval benchmark: skip rerank + generation + judge (near-free)")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    if not args.index:
        raise SystemExit("Provide --index <name> (or set EVAL_INDEX).")

    report = evaluate(args.index, args.pipeline, args.golden, args.k,
                      not args.no_judge, args.judge_model, args.limit,
                      retrieval_only=args.retrieval_only)
    _print_table(report)

    out_dir = Path("evals/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = out_dir / f"{args.pipeline}-{stamp}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[report] {out}")


if __name__ == "__main__":
    main()
