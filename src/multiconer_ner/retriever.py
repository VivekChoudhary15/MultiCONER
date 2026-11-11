"""
Retriever wrapper for Stage A1.

Wraps KBIndex and formats retrieved context into a string suitable for
prepending to model input.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .kb import KBEntry, KBIndex


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
    ) -> List[KBEntry]:
        return self.kb.search(
            sentence, language=language, top_k=top_k, threshold=threshold
        )

    @staticmethod
    def context_string(entries: List[KBEntry], max_entries: int = 5) -> str:
        """Format retrieved entries into a context string for model input."""
        if not entries:
            return ""
        parts = []
        for entry in entries[:max_entries]:
            desc = entry.description or entry.title
            parts.append(f"[KB] {entry.title}: {desc}")
        return " ".join(parts)
