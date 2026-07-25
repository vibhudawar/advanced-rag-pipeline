"""Load a BEIR subset, ingest its corpus into a throwaway index, and emit a golden set.

    pip install -r evals/requirements-eval.txt
    python -m evals.load_beir --dataset scifact --index beir-scifact

This gives standardized, comparable retrieval numbers (nDCG@10 you can cite against
published BEIR results). It is heavier than the own-corpus path — it embeds the whole
subset corpus into a new Pinecone index — so treat it as step 2, after the own-corpus
scoreboard is running.

The doc text ingested and the text hashed for ground truth are the SAME string
(`_beir_doc_text`), so retrieval hashes line up exactly with qrels.
"""

from __future__ import annotations

import argparse

from config import EMBEDDING_PROVIDER
from src.ingestion.DBIngestion import get_vector_store
from src.ingestion.EmbeddingCreator import get_embedder
from src.utils.Helpers import ensure_index_exists

from .schema import GoldenItem, content_hash, write_jsonl


def _beir_doc_text(doc: dict) -> str:
    return f"{doc.get('title', '').strip()}\n{doc.get('text', '').strip()}".strip()


def _download(dataset: str) -> str:
    from beir import util
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    return util.download_and_unzip(url, "data/beir")


def load(dataset: str, index_name: str, split: str, max_corpus: int | None, out: str) -> None:
    from beir.datasets.data_loader import GenericDataLoader

    data_path = _download(dataset)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    print(f"[beir] {dataset}: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrel sets")

    # Ingest corpus into a fresh index.
    embedder = get_embedder(provider=EMBEDDING_PROVIDER)
    vs = get_vector_store()
    ensure_index_exists(vs, index_name, embedder)

    doc_ids = list(corpus)
    if max_corpus:
        # keep any docs referenced by qrels, plus fill up to max_corpus
        needed = {d for rels in qrels.values() for d in rels}
        rest = [d for d in doc_ids if d not in needed]
        doc_ids = list(needed) + rest[: max(0, max_corpus - len(needed))]

    batch = []
    for did in doc_ids:
        batch.append({"text": _beir_doc_text(corpus[did]), "metadata": {"beir_id": did}})
        if len(batch) >= 200:
            vs.add_documents(index_name, batch, embedder)
            batch = []
    if batch:
        vs.add_documents(index_name, batch, embedder)
    print(f"[beir] ingested {len(doc_ids)} docs into '{index_name}'")

    # Build golden items from qrels.
    items = []
    for qid, rels in qrels.items():
        rel_hashes = [content_hash(_beir_doc_text(corpus[d]))
                      for d, score in rels.items() if score > 0 and d in corpus]
        if not rel_hashes:
            continue
        items.append(GoldenItem(
            id=f"beir-{qid}", query=queries[qid], reference_answer="",
            relevant_chunk_hashes=rel_hashes, q_type="factoid",
            source="beir", needs_review=False,
        ))
    n = write_jsonl(items, out)
    print(f"[done] wrote {n} BEIR golden items to {out}")
    print(f"[next] python -m evals.evaluate --index {index_name} --golden {out} --no-judge")


def main() -> None:
    p = argparse.ArgumentParser(description="Load a BEIR subset into a throwaway index + golden set.")
    p.add_argument("--dataset", default="scifact", help="BEIR dataset name (e.g. scifact, fiqa, nfcorpus)")
    p.add_argument("--index", default="beir-scifact")
    p.add_argument("--split", default="test")
    p.add_argument("--max-corpus", type=int, default=None, help="cap corpus size to save cost")
    p.add_argument("--out", default="data/beir_scifact.jsonl")
    args = p.parse_args()
    load(args.dataset, args.index, args.split, args.max_corpus, args.out)


if __name__ == "__main__":
    main()
