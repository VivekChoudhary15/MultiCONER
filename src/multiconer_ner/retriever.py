"""
Retriever wrapper for Stages A1 (retrieval) and A2 (context filter).

Wraps KBIndex and provides relevance-aware retrieval:
  * BM25 search over an offline Wikipedia/Wikidata index
  * Score-based filtering so irrelevant context is not injected
  * Context string formatting with max-entry and max-token caps
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Sequence, Tuple

from .kb import KBEntry, KBIndex


class Retriever:
    """Thin wrapper around KBIndex with relevance-aware context filtering."""

    def __init__(self, kb_index: KBIndex):
        self.kb = kb_index

    def retrieve(
        self,
        sentence: str,
        language: Optional[str] = None,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[KBEntry, float]]:
        """Retrieve top-k entries for a sentence, returned as (entry, score) pairs."""
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
        """Retrieve for many sentences at once."""
        return [
            self.retrieve(s, language=language, top_k=top_k, threshold=threshold)
            for s in sentences
        ]

    @staticmethod
    def filter_by_score(
        hits: Sequence[Tuple[KBEntry, float]],
        min_score: float,
        max_entries: int = 3,
    ) -> List[KBEntry]:
        """
        Keep only entries whose BM25 score >= min_score, capped at max_entries.

        This is the core A2 filter: irrelevant retrieval is dropped so it cannot
        poison the model input.
        """
        filtered = [e for e, s in hits if s >= min_score]
        return filtered[:max_entries]

    @staticmethod
    def context_string(
        entries: Sequence[KBEntry],
        max_entries: int = 3,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Format retrieved entries into a context string for model input."""
        if not entries:
            return ""
        parts = []
        for entry in list(entries)[:max_entries]:
            desc = entry.description or entry.title
            parts.append(f"[KB] {entry.title}: {desc}")
        ctx = " ".join(parts)
        if max_tokens is not None:
            ctx = " ".join(ctx.split()[:max_tokens])
        return ctx

    def attach_context(
        self,
        sentences: Sequence,
        language: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.0,
        max_entries: int = 3,
        max_tokens: Optional[int] = None,
        skip_low_confidence: bool = True,
    ) -> List:
        """
        Return a copy of ``sentences`` with retrieved context attached.

        A2 behavior:
          * Only attach context when the best BM25 score >= min_score.
          * When skip_low_confidence is True and no entry clears the threshold,
            the sentence is returned unchanged (pure A0 behavior).
          * Context is capped at max_entries entries and max_tokens tokens.
        """
        from dataclasses import replace

        queries = [" ".join(s.tokens) for s in sentences]
        hits = self.retrieve_batch(
            queries, language=language, top_k=top_k, threshold=0.0
        )
        out = []
        for sent, h in zip(sentences, hits):
            best_score = h[0][1] if h else 0.0
            if skip_low_confidence and best_score < min_score:
                # Retrieval is weak for this sentence -> do not attach context.
                out.append(sent)
                continue
            entries = self.filter_by_score(h, min_score=min_score, max_entries=max_entries)
            ctx = self.context_string(entries, max_entries=max_entries, max_tokens=max_tokens)
            out.append(replace(sent, context=ctx) if ctx else sent)
        return out
