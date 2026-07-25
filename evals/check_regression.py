"""Compare a pipeline's retrieval metrics against a committed baseline; fail on regression.

This is the CI merge gate (see .github/workflows/ci.yml). Only **retrieval** metrics are
gated because they are deterministic (fixed index + embeddings) and therefore reliable;
generation/judge metrics are intentionally excluded — they cost money and vary run-to-run,
so they belong on a nightly trend, not a hard per-commit gate.

    python -m evals.check_regression --index beir-scifact           # gate: exit 1 on a >tol drop
    python -m evals.check_regression --index beir-scifact --update   # record current as new baseline

Baselines live in evals/baselines.json (committed). When you legitimately improve, run with
--update in the same PR to raise the bar.

Secrets (API keys) come from the environment via `config`; nothing is hardcoded and only
metric values are printed — never keys.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .evaluate import evaluate


def _load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def main() -> int:
    p = argparse.ArgumentParser(description="Retrieval-metric regression gate for CI.")
    p.add_argument("--index", required=True, help="Pinecone index to evaluate against")
    p.add_argument("--name", default=None, help="baseline key (default: the index name)")
    p.add_argument("--pipeline", default="baseline")
    p.add_argument("--golden", default="data/beir_scifact.jsonl")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--tolerance", type=float, default=0.02, help="max allowed relative drop (0.02 = 2%%)")
    p.add_argument("--baselines", default="evals/baselines.json")
    p.add_argument("--update", action="store_true", help="write current metrics as the new baseline")
    args = p.parse_args()
    name = args.name or args.index
    baselines_path = Path(args.baselines)

    report = evaluate(
        index=args.index, pipeline_name=args.pipeline, golden_path=args.golden,
        k=args.k, use_judge=False, judge_model=None, limit=None, retrieval_only=True,
    )
    current = report["retrieval"]

    baselines = _load(baselines_path)

    if args.update:
        baselines[name] = {
            "k": args.k, "golden": args.golden, "pipeline": args.pipeline,
            "metrics": current, "updated": time.strftime("%Y-%m-%d"),
        }
        baselines_path.parent.mkdir(parents=True, exist_ok=True)
        baselines_path.write_text(json.dumps(baselines, indent=2) + "\n")
        print(f"[baseline] updated '{name}' -> {current}")
        return 0

    if name not in baselines:
        print(f"[error] no baseline for '{name}' in {baselines_path}. "
              f"Record one first: python -m evals.check_regression --index {args.index} --update")
        return 2

    base = baselines[name]["metrics"]
    tol = args.tolerance
    regressions, rows = [], []
    for metric, val in current.items():
        b = base.get(metric)
        if b is None:
            rows.append((metric, "—", f"{val:.4f}", "new"))
            continue
        floor = b * (1 - tol)
        ok = val >= floor
        rows.append((metric, f"{b:.4f}", f"{val:.4f}", "ok" if ok else "REGRESSED"))
        if not ok:
            regressions.append((metric, b, val, floor))

    print("\n  metric                baseline   current    status")
    print("  " + "-" * 52)
    for metric, b, c, status in rows:
        print(f"  {metric:<20} {b:>9} {c:>9}    {status}")
    print()

    if regressions:
        print(f"[FAIL] {len(regressions)} metric(s) dropped more than {tol:.0%} vs baseline:")
        for m, b, c, floor in regressions:
            print(f"   - {m}: {c:.4f} < floor {floor:.4f} (baseline {b:.4f})")
        print("If this drop is expected/justified, update the baseline in the same PR "
              "(--update) and explain why in the PR description.")
        return 1

    print(f"[PASS] no retrieval regression beyond {tol:.0%}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
