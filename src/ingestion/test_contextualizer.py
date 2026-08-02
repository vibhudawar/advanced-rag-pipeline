"""Tests for contextualization TPM-safety: summarize-vs-full-doc decision + per-chunk failure.

The LLM is faked (monkeypatched) so these run offline and assert control flow, not model output.
"""
import sys
import types

import pytest

from src.ingestion import contextualizer


class _FakeResp:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    """Records prompts. invoke() -> summary marker; batch() -> one header per prompt,
    optionally raising for prompts whose chunk text is in `fail_on`."""
    def __init__(self, fail_on=()):
        self.fail_on = fail_on
        self.invoked = []
        self.batched = []

    def invoke(self, prompt):
        self.invoked.append(prompt)
        return _FakeResp("SUMMARY")

    def batch(self, prompts, config=None, return_exceptions=False):
        self.batched = prompts
        out = []
        for p in prompts:
            if any(f in p for f in self.fail_on):
                out.append(RuntimeError("429") if return_exceptions else None)
            else:
                out.append(_FakeResp("header"))
        return out


@pytest.fixture
def patch_llm(monkeypatch):
    def _install(fake):
        mod = types.ModuleType("langchain_openai")
        mod.ChatOpenAI = lambda **kw: fake
        monkeypatch.setitem(sys.modules, "langchain_openai", mod)
        # ensure the config key check passes so contextualization actually runs
        import config
        monkeypatch.setattr(config, "OPENAI_API_KEY", "x", raising=False)
    return _install


def _chunks(n, text="chunk body"):
    return [{"text": f"{text} {i}", "metadata": {"filename": "d.pdf"}} for i in range(n)]


def test_summarizes_on_token_blowup(patch_llm):
    install = patch_llm
    fake = _FakeLLM()
    install(fake)
    # small doc but many chunks -> blowup exceeds budget -> should summarize first
    big_doc = "word " * 4000  # ~20k chars
    out = contextualizer.contextualize_chunks(_chunks(60), big_doc)
    assert fake.invoked, "expected a one-shot summary call on token blowup"
    assert "SUMMARY" in fake.batched[0]      # chunk prompts carry the summary, not the full doc
    assert all(c["metadata"].get("context") == "header" for c in out)


def test_keeps_full_doc_when_small(patch_llm):
    install = patch_llm
    fake = _FakeLLM()
    install(fake)
    doc = "short document body that situates things"
    out = contextualizer.contextualize_chunks(_chunks(3), doc)
    assert not fake.invoked, "small/few-chunk docs should not be summarized"
    assert doc in fake.batched[0]            # full doc used as prefix
    assert len(out) == 3


def test_per_chunk_failure_is_isolated(patch_llm):
    install = patch_llm
    fake = _FakeLLM(fail_on=("chunk body 1",))  # only chunk index 1 fails
    install(fake)
    out = contextualizer.contextualize_chunks(_chunks(3, "chunk body"), "small doc")
    ctx = [c["metadata"].get("context") for c in out]
    assert ctx[0] == "header" and ctx[2] == "header"
    assert "context" not in out[1]["metadata"]        # failed chunk has no header
    assert out[1]["text"] == "chunk body 1"           # but still ingested (raw)


def test_empty_chunks_noop(patch_llm):
    install = patch_llm
    install(_FakeLLM())
    assert contextualizer.contextualize_chunks([], "doc") == []
