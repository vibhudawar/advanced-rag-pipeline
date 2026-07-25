"""Hand-rolled LLM-as-judge for generation quality.

Design choices (see plan-v2.md WIN 1):
  - Judge model != generator model, to avoid a model grading its own output. The
    generator baseline is gpt-4o-mini; the judge defaults to a stronger model.
  - Each metric is one structured call returning a 0-1 score + a short reason, so the
    output is inspectable (you can read WHY a run scored low), not a black box.
  - Metrics mirror RAGAS semantics but are our own prompts:
      faithfulness       : are the answer's claims supported by the retrieved contexts?
      answer_relevance   : does the answer actually address the question?
      context_relevance  : are the retrieved contexts relevant to the question?
  - abstention detection is a cheap heuristic + a fallback judge call, used to score
    whether the system correctly says "I don't know" on unanswerable questions.

Keys come from config (env-backed). Nothing is hardcoded.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from config import GEMINI_API_KEY, OPENAI_API_KEY


class _Score(BaseModel):
    score: float = Field(description="A number from 0.0 (worst) to 1.0 (best).")
    reason: str = Field(description="One sentence explaining the score.")


def _make_judge_llm(model: str | None = None):
    """Return a LangChain chat model for judging, distinct from the generator model."""
    if OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or "gpt-4o", temperature=0.0,
                          openai_api_key=OPENAI_API_KEY)
    if GEMINI_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model or "gemini-2.5-pro", temperature=0.0,
                                      google_api_key=GEMINI_API_KEY)
    raise RuntimeError("No judge LLM key found (set OPENAI_API_KEY or GEMINI_API_KEY).")


_ABSTAIN_PATTERNS = [
    r"\bi (?:do not|don't) have (?:enough |sufficient )?(?:information|context)\b",
    r"\bno (?:relevant )?(?:information|context|documents?)\b",
    r"\bcannot (?:answer|be answered|find)\b",
    r"\bunable to (?:answer|find)\b",
    r"\bnot (?:enough|sufficient) (?:information|context)\b",
    r"\bthe (?:context|documents?) (?:do(?:es)? not|don't) (?:contain|mention|cover)\b",
]


class Judge:
    """Stateful judge holding one LLM client and structured-output chains."""

    def __init__(self, model: str | None = None):
        self.llm = _make_judge_llm(model)
        self.model_name = getattr(self.llm, "model_name", None) or getattr(self.llm, "model", "judge")
        self._chain = self.llm.with_structured_output(_Score)

    # ---- individual metrics -------------------------------------------------
    def faithfulness(self, answer: str, contexts: list[str]) -> _Score:
        ctx = _join(contexts)
        return self._chain.invoke(
            "You grade FAITHFULNESS: are ALL factual claims in the ANSWER supported by the "
            "CONTEXT? Score 1.0 if every claim is grounded, 0.0 if the answer is mostly "
            "unsupported or contradicts the context. Ignore fluency.\n\n"
            f"CONTEXT:\n{ctx}\n\nANSWER:\n{answer}"
        )

    def answer_relevance(self, question: str, answer: str) -> _Score:
        return self._chain.invoke(
            "You grade ANSWER RELEVANCE: does the ANSWER directly and completely address the "
            "QUESTION? Score 1.0 for a focused, complete answer; lower it for partial, evasive, "
            "or off-topic answers. A correct refusal to an unanswerable question is still "
            "relevant if it addresses the question.\n\n"
            f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
        )

    def context_relevance(self, question: str, contexts: list[str]) -> _Score:
        ctx = _join(contexts)
        return self._chain.invoke(
            "You grade CONTEXT RELEVANCE: how much of the retrieved CONTEXT is actually relevant "
            "to answering the QUESTION? Score 1.0 if the context is on-topic and useful, near 0.0 "
            "if it is mostly noise.\n\n"
            f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx}"
        )

    # ---- abstention ---------------------------------------------------------
    def detect_abstention(self, answer: str) -> bool:
        """Did the answer decline to answer / say it lacks information?"""
        low = answer.lower()
        if any(re.search(p, low) for p in _ABSTAIN_PATTERNS):
            return True
        # Fallback: ask the judge for ambiguous phrasings.
        verdict = self._chain.invoke(
            "Does the following ANSWER decline to answer or state it lacks the information to "
            "answer (as opposed to giving a substantive answer)? Score 1.0 if it declines, 0.0 "
            f"if it gives a real answer.\n\nANSWER:\n{answer}"
        )
        return verdict.score >= 0.5


def _join(contexts: list[str], max_chars: int = 8000) -> str:
    if not contexts:
        return "(no context retrieved)"
    out, total = [], 0
    for i, c in enumerate(contexts, 1):
        block = f"[{i}] {c.strip()}"
        total += len(block)
        if total > max_chars:
            break
        out.append(block)
    return "\n\n".join(out)
