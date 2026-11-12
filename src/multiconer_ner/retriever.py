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


def sentence_mentions(sent) -> set:
    """Gold entity mention strings of a BIO-tagged sentence (same rule as
    scripts/build_kb.py, so they match the KB entry titles)."""
    mentions, cur = set(), []
    for tok, tag in zip(sent.tokens, sent.tags):
        if tag == "O":
            if cur:
                mentions.add(" ".join(cur))
                cur = []
        elif tag.startswith("B-"):
            if cur:
                mentions.add(" ".join(cur))
            cur = [tok]
        elif tag.startswith("I-"):
            cur.append(tok)
    if cur:
        mentions.add(" ".join(cur))
    return mentions


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
        """Retrieve for many sentences at once (vectorised sparse BM25 in KBIndex)."""
        return self.kb.search_batch(
            sentences, language=language, top_k=top_k, threshold=threshold
        )

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
        queries = [" ".join(s.tokens) for s in sentences]
        hits = self.retrieve_batch(
            queries, language=language, top_k=top_k, threshold=0.0
        )
        return self._apply_filter(
            sentences, hits, min_score, max_entries, max_tokens, skip_low_confidence
        )

    def attach_context_kfold(
        self,
        sentences: Sequence,
        language: str,
        n_folds: int = 5,
        seed: int = 42,
        top_k: int = 5,
        min_score: float = 0.0,
        max_entries: int = 3,
        max_tokens: Optional[int] = None,
        skip_low_confidence: bool = True,
    ) -> List:
        """Leakage-free context for TRAINING sentences (K-fold KB).

        The KB is built from the training set's gold mentions, so plain
        retrieval on a training sentence almost always returns that sentence's
        own gold entities -- context the model never gets at test time. Here
        the training sentences are split into ``n_folds`` folds, and a sentence
        in fold f may only retrieve KB entries for mentions that also occur in
        some OTHER fold. An entity therefore appears in training context only
        when it was "seen elsewhere in training", just as at test time.
        KB entries whose title never occurs as a training mention are always
        allowed. The A2 score filter is applied afterwards as usual.
        """
        import numpy as np

        rng = np.random.RandomState(seed)
        fold_of = rng.randint(0, n_folds, size=len(sentences))

        # mention -> set of folds it occurs in
        mention_folds: dict = {}
        for sent, f in zip(sentences, fold_of):
            for m in sentence_mentions(sent):
                mention_folds.setdefault(m, set()).add(int(f))

        lang_entries = [
            i for i, e in enumerate(self.kb.entries) if e.language == language
        ]
        hits: List = [None] * len(sentences)
        for f in range(n_folds):
            allowed = [
                i for i in lang_entries
                if not (mention_folds.get(self.kb.entries[i].title, set()) <= {f})
            ]
            members = [j for j in range(len(sentences)) if fold_of[j] == f]
            if not members:
                continue
            fold_hits = self.kb.search_batch(
                [" ".join(sentences[j].tokens) for j in members],
                top_k=top_k, threshold=0.0, entry_indices=allowed,
            )
            for j, h in zip(members, fold_hits):
                hits[j] = h
        hits = [h or [] for h in hits]
        return self._apply_filter(
            sentences, hits, min_score, max_entries, max_tokens, skip_low_confidence
        )

    def _apply_filter(
        self,
        sentences: Sequence,
        hits: Sequence,
        min_score: float,
        max_entries: int,
        max_tokens: Optional[int],
        skip_low_confidence: bool,
    ) -> List:
        """A2 relevance filter + formatting, shared by both attach methods."""
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
