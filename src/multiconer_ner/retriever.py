"""
Retriever wrapper for Stage A1.

Wraps KBIndex and formats retrieved context into a string suitable for
prepending to model input.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

from .kb import KBEntry, KBIndex

Hit = Union[KBEntry, Tuple[KBEntry, float]]


class Retriever:
    """Thin wrapper around KBIndex with context formatting."""

    def __init__(self, kb_index: KBIndex):
        self.kb = kb_index

    def retrieve(
        self,
        sentence: str,
        language: Optional[str] = None,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[KBEntry, float]]:
        return self.kb.search(
            sentence, language=language, top_k=top_k, threshold=threshold
        )

    def retrieve_batch(
        self,
        sentences: Sequence[str],
        language: Optional[str] = None,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[List[Tuple[KBEntry, float]]]:
        return self.kb.search_batch(
            sentences, language=language, top_k=top_k, threshold=threshold
        )

    @staticmethod
    def context_string(
        entries: Sequence[Hit],
        max_entries: int = 5,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Format retrieved entries into a context string for model input.

        Accepts either KBEntry objects or (KBEntry, score) pairs as returned
        by KBIndex.search. If ``max_tokens`` is given, the string is cut to
        that many whitespace tokens (the same cut NERDataset applies), which
        keeps memory low when attaching context to millions of sentences.
        """
        if not entries:
            return ""
        parts = []
        for hit in list(entries)[:max_entries]:
            entry = hit[0] if isinstance(hit, tuple) else hit
            desc = entry.description or entry.title
            parts.append(f"[KB] {entry.title}: {desc}")
        ctx = " ".join(parts)
        if max_tokens is not None:
            ctx = " ".join(ctx.split()[:max_tokens])
        return ctx

    def attach_context(
        self,
        sentences: List,
        language: Optional[str] = None,
        top_k: int = 5,
        threshold: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> List:
        """Return a copy of ``sentences`` with retrieved context attached."""
        from dataclasses import replace

        queries = [" ".join(s.tokens) for s in sentences]
        hits = self.retrieve_batch(queries, language=language, top_k=top_k, threshold=threshold)
        out = []
        for sent, h in zip(sentences, hits):
            ctx = self.context_string(h, max_tokens=max_tokens)
            out.append(replace(sent, context=ctx) if ctx else sent)
        return out
