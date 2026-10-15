"""Deterministic lexical retrieval for the RAG experiments (E2/E3).

A compact BM25 index over a small corpus of approved documents. Deterministic and
CPU-safe, so retrieval metrics are reproducible without external services. This
is a research-grade placeholder, not a production search system.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Document:
    id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] | None = None


class BM25Retriever:
    def __init__(self, documents: Sequence[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self._corpus = [tokenize(d.title + " " + d.text) for d in self.documents]
        self._doc_len = [len(t) for t in self._corpus]
        self._avg_len = sum(self._doc_len) / len(self._doc_len) if self._doc_len else 0.0
        self._df: Counter = Counter()
        for toks in self._corpus:
            self._df.update(set(toks))
        self._n = len(self.documents)

    def _score(self, query_toks: list[str], doc_idx: int) -> float:
        doc_toks = self._corpus[doc_idx]
        tf = Counter(doc_toks)
        dl = self._doc_len[doc_idx]
        score = 0.0
        for term in query_toks:
            if term not in tf:
                continue
            df = self._df.get(term, 0)
            idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1.0) if df else 0.0
            denom = tf[term] + self.k1 * (1 - self.b + self.b * dl / self._avg_len)
            score += idf * tf[term] * (self.k1 + 1) / denom
        return score

    def search(self, query: str, k: int = 5) -> list[str]:
        query_toks = tokenize(query)
        if not query_toks:
            return []
        scored = sorted(
            ((self._score(query_toks, i), self.documents[i].id) for i in range(self._n)),
            reverse=True,
        )
        return [doc_id for _, doc_id in scored if _ > 0][:k] or [doc_id for _, doc_id in scored][:k]


def build_retriever(documents: Sequence[dict[str, Any]]) -> BM25Retriever:
    docs = [
        Document(
            id=d["id"],
            text=d["text"],
            title=d.get("title", ""),
            metadata={k: v for k, v in d.items() if k not in ("id", "text", "title")},
        )
        for d in documents
    ]
    return BM25Retriever(docs)


def retrieval_context(retriever: BM25Retriever, query: str, k: int = 3) -> tuple[list[str], str]:
    """Return (retrieved_doc_ids, formatted context block) for prompt injection."""
    ids = retriever.search(query, k=k)
    by_id = {d.id: d for d in retriever.documents}
    lines = []
    for i, doc_id in enumerate(ids, 1):
        d = by_id.get(doc_id)
        if d:
            lines.append(f"[{i}] ({d.id}) {d.title}: {d.text}")
    return ids, "\n".join(lines)
