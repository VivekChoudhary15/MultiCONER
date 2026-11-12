"""
Offline Knowledge Base index for MultiCoNER retrieval (Stage A1).

Builds a BM25 index over Wikipedia/Wikidata entity entries and provides
retrieval. The index is built once (scripts/build_kb.py) and reused across
training/inference runs.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class KBEntry:
    """A single knowledge base entry."""

    qid: str
    title: str
    description: str
    aliases: List[str] = field(default_factory=list)
    entity_type: Optional[str] = None
    language: Optional[str] = None
    text: str = ""  # Concatenated searchable text (title + description + aliases)

    def to_dict(self) -> dict:
        return {
            "qid": self.qid,
            "title": self.title,
            "description": self.description,
            "aliases": self.aliases,
            "entity_type": self.entity_type,
            "language": self.language,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KBEntry":
        return cls(**d)


class _SparseBM25:
    """Vectorised BM25Okapi (same formula/defaults as ``rank_bm25.BM25Okapi``).

    Pre-computes a sparse (docs x vocab) term-weight matrix so that scoring a
    batch of queries is a single sparse matrix product instead of a Python
    loop over every document for every query token.
    """

    def __init__(
        self,
        corpus: Sequence[Sequence[str]],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        self.vocab: Dict[str, int] = {}
        rows: List[int] = []
        cols: List[int] = []
        tfs: List[float] = []
        doc_len = np.zeros(len(corpus), dtype=np.float64)
        for i, doc in enumerate(corpus):
            doc_len[i] = len(doc)
            for tok, c in Counter(doc).items():
                j = self.vocab.setdefault(tok, len(self.vocab))
                rows.append(i)
                cols.append(j)
                tfs.append(c)

        n_docs, n_vocab = len(corpus), len(self.vocab)
        self.n_docs = n_docs
        rows_a = np.asarray(rows, dtype=np.int64)
        cols_a = np.asarray(cols, dtype=np.int64)
        tf = np.asarray(tfs, dtype=np.float64)

        # IDF exactly as rank_bm25.BM25Okapi._calc_idf
        df = np.bincount(cols_a, minlength=n_vocab).astype(np.float64)
        idf = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
        if n_vocab:
            eps = epsilon * idf.mean()
            idf[idf < 0] = eps

        avgdl = doc_len.mean() if n_docs else 1.0
        norm = k1 * (1 - b + b * doc_len[rows_a] / avgdl)
        weights = idf[cols_a] * (tf * (k1 + 1) / (tf + norm))
        self.weights_t = csr_matrix(
            (weights.astype(np.float32), (cols_a, rows_a)), shape=(n_vocab, n_docs)
        )  # vocab x docs

    def score_batch(self, tokenized_queries: Sequence[Sequence[str]]) -> np.ndarray:
        """Return dense (n_queries x n_docs) BM25 scores."""
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        for i, q in enumerate(tokenized_queries):
            for tok, c in Counter(q).items():
                j = self.vocab.get(tok)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
                    vals.append(c)  # repeated query tokens count repeatedly, as in rank_bm25
        q_mat = csr_matrix(
            (np.asarray(vals, dtype=np.float32), (rows, cols)),
            shape=(len(tokenized_queries), len(self.vocab)),
        )
        return (q_mat @ self.weights_t).toarray()


class KBIndex:
    """
    Offline KB index with BM25 retrieval.

    Stores KBEntry objects and a BM25 index built over their ``text`` field.
    A separate BM25 index (own IDF statistics) is built lazily per language,
    plus one over all entries for ``language=None``.
    Supports saving/loading to disk for reuse across runs.
    """

    _ALL = "__all__"

    def __init__(self, entries: Optional[List[KBEntry]] = None):
        self.entries: List[KBEntry] = entries or []
        # language -> (entry indices, BM25 index over those entries)
        self._indexes: Dict[str, Tuple[np.ndarray, _SparseBM25]] = {}

    def add(self, entry: KBEntry) -> None:
        self.entries.append(entry)
        self._indexes.clear()

    def _get_index(self, language: Optional[str]) -> Optional[Tuple[np.ndarray, _SparseBM25]]:
        key = self._ALL if language is None else language
        if key not in self._indexes:
            idx = np.asarray(
                [
                    i for i, e in enumerate(self.entries)
                    if language is None or e.language == language
                ],
                dtype=np.int64,
            )
            if len(idx) == 0:
                return None
            corpus = [self.entries[i].text.lower().split() for i in idx]
            self._indexes[key] = (idx, _SparseBM25(corpus))
        return self._indexes[key]

    def search(
        self,
        query: str,
        language: Optional[str] = None,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[KBEntry, float]]:
        """Retrieve top-k (entry, score) pairs for a query using BM25.

        If ``language`` is given, only entries matching that language are
        considered, which keeps retrieval relevant for multilingual training.
        """
        return self.search_batch([query], language, top_k, threshold)[0]

    def search_batch(
        self,
        queries: Sequence[str],
        language: Optional[str] = None,
        top_k: int = 5,
        threshold: float = 0.0,
        chunk_size: int = 512,
    ) -> List[List[Tuple[KBEntry, float]]]:
        """Batched version of :meth:`search` (much faster for many queries)."""
        results: List[List[Tuple[KBEntry, float]]] = [[] for _ in queries]
        found = self._get_index(language)
        if found is None or top_k <= 0:
            return results
        entry_idx, bm25 = found
        k = min(top_k, bm25.n_docs)

        for start in range(0, len(queries), chunk_size):
            chunk = [q.lower().split() for q in queries[start:start + chunk_size]]
            scores = bm25.score_batch(chunk)
            if k < bm25.n_docs:
                cand = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            else:
                cand = np.tile(np.arange(bm25.n_docs), (len(chunk), 1))
            for r in range(len(chunk)):
                row = cand[r]
                row_scores = scores[r, row]
                order = np.lexsort((row, -row_scores))  # score desc, then index
                out = results[start + r]
                for o in order:
                    s = float(row_scores[o])
                    if s > threshold:
                        out.append((self.entries[entry_idx[row[o]]], s))
        return results

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        entries_path = directory / "entries.json"
        with open(entries_path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self.entries], f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, directory: str | Path) -> "KBIndex":
        directory = Path(directory)
        entries_path = directory / "entries.json"
        with open(entries_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = [KBEntry.from_dict(d) for d in data]
        return cls(entries)
