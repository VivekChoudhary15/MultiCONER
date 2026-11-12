"""
Offline Knowledge Base index for MultiCoNER retrieval (Stage A1).

Builds a BM25 index over Wikipedia/Wikidata entity entries and provides
retrieval. The index is built once (scripts/build_kb.py) and reused across
training/inference runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi


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


class KBIndex:
    """
    Offline KB index with BM25 retrieval.

    Stores KBEntry objects and a BM25 index built over their ``text`` field.
    Supports per-language filtering and saving/loading to disk for reuse
    across runs.
    """

    def __init__(self, entries: Optional[List[KBEntry]] = None):
        self.entries: List[KBEntry] = entries or []
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: Optional[List[List[str]]] = None
        if self.entries:
            self._build_bm25()

    def _build_bm25(self) -> None:
        self._tokenized_corpus = [e.text.lower().split() for e in self.entries]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def add(self, entry: KBEntry) -> None:
        self.entries.append(entry)
        self._build_bm25()

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
        if self._bm25 is None or not self.entries:
            return []

        if language is not None:
            candidate_indices = [
                i for i, e in enumerate(self.entries) if e.language == language
            ]
        else:
            candidate_indices = list(range(len(self.entries)))

        if not candidate_indices:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        candidate_scores = [(i, scores[i]) for i in candidate_indices]
        candidate_scores.sort(key=lambda x: x[1], reverse=True)

        results: List[Tuple[KBEntry, float]] = []
        for idx, score in candidate_scores[:top_k]:
            if score > threshold:
                results.append((self.entries[idx], float(score)))
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
