"""Measure NLU intent-routing accuracy — the axis where NLU actually helps.

The BEIR retrieval benchmark showed NLU query-rewriting does NOT improve retrieval on clean,
well-formed queries (it dilutes precision). But that's the wrong axis to judge NLU on. Its
real value is *routing*: recognising when a message is a greeting or a support/transactional
request so the system can respond directly instead of running (and citing) a document search,
and enabling graceful non-answers.

This is a small, hand-labelled, non-PII set scoring the `understand()` step's `intent`
classification. Run: python -m evals.routing
"""

from __future__ import annotations

from src.retrieval.nlu import QueryUnderstander

# (message, expected_intent). Domain-neutral so the labels don't depend on a specific corpus.
ROUTING_SET: list[tuple[str, str]] = [
    # greetings / pleasantries -> should NOT trigger retrieval
    ("hi", "greeting"),
    ("hello there!", "greeting"),
    ("good morning", "greeting"),
    ("thanks, that's really helpful", "greeting"),
    ("thank you so much", "greeting"),
    ("hey how are you?", "greeting"),
    ("cheers, appreciate it", "greeting"),
    ("ok great, bye", "greeting"),
    # support / transactional -> off-topic for a knowledge-base, route elsewhere
    ("cancel my subscription", "off_topic"),
    ("I want a refund", "off_topic"),
    ("reset my password", "off_topic"),
    ("connect me to a human agent", "off_topic"),
    ("update my billing address", "off_topic"),
    # genuine knowledge questions -> rag_query
    ("what is the effect of metformin on cancer cell proliferation?", "rag_query"),
    ("how does reciprocal rank fusion combine ranked lists?", "rag_query"),
    ("summarize the main risks discussed in the annual report", "rag_query"),
    ("what were the revenue figures for the last three quarters?", "rag_query"),
    ("which studies link vitamin D to immune response?", "rag_query"),
    ("compare the side effects of the two treatments", "rag_query"),
    ("what does the contract say about termination clauses?", "rag_query"),
    ("explain the mechanism of action described in the paper", "rag_query"),
    ("who are the authors of the study on gut microbiota?", "rag_query"),
    ("what is the recommended dosage according to the guidelines?", "rag_query"),
]


def main() -> int:
    understander = QueryUnderstander()
    rows, correct = [], 0
    for message, expected in ROUTING_SET:
        got = understander.understand(message).intent
        ok = got == expected
        correct += ok
        rows.append((message, expected, got, ok))

    accuracy = correct / len(ROUTING_SET)
    print("\n  message                                   expected     got          ok")
    print("  " + "-" * 74)
    for message, expected, got, ok in rows:
        print(f"  {message[:40]:<40}  {expected:<11}  {got:<11}  {'✓' if ok else '✗'}")
    print("  " + "-" * 74)
    print(f"  intent routing accuracy: {accuracy:.1%}  ({correct}/{len(ROUTING_SET)})\n")

    misses = [(m, e, g) for m, e, g, ok in rows if not ok]
    if misses:
        print("  misclassified:")
        for m, e, g in misses:
            print(f"   - '{m}' expected {e}, got {g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
