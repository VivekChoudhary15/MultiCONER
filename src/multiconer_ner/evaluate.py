"""Evaluation utilities for NER using seqeval.

Computes precision / recall / F1 overall and per entity type from BIO tag
sequences (one list of tag strings per sentence).
"""

from __future__ import annotations

from typing import Dict, List

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from seqeval.scheme import IOB2


def evaluate_predictions(
    true_tags: List[List[str]],
    pred_tags: List[List[str]],
) -> dict:
    """Compute overall and per-entity-type precision/recall/F1 (seqeval)."""
    assert len(true_tags) == len(pred_tags), (
        f"true ({len(true_tags)}) and pred ({len(pred_tags)}) sentence counts differ"
    )

    overall_precision = precision_score(
        true_tags, pred_tags, scheme=IOB2, mode="strict"
    )
    overall_recall = recall_score(
        true_tags, pred_tags, scheme=IOB2, mode="strict"
    )
    overall_f1 = f1_score(
        true_tags, pred_tags, scheme=IOB2, mode="strict"
    )

    report = classification_report(
        true_tags,
        pred_tags,
        scheme=IOB2,
        mode="strict",
        output_dict=True,
        zero_division=0,
    )

    by_type: Dict[str, dict] = {}
    for entity_type, scores in report.items():
        if entity_type in ("macro_avg", "micro_avg", "weighted_avg", "accuracy"):
            continue
        by_type[entity_type] = {
            "precision": float(scores.get("precision", 0.0)),
            "recall": float(scores.get("recall", 0.0)),
            "f1": float(scores.get("f1-score", 0.0)),
            "support": int(scores.get("support", 0)),
        }

    return {
        "overall_precision": float(overall_precision),
        "overall_recall": float(overall_recall),
        "overall_f1": float(overall_f1),
        "by_type": by_type,
        "report": classification_report(
            true_tags, pred_tags, scheme=IOB2, mode="strict", zero_division=0
        ),
    }


def print_report(metrics: dict) -> None:
    """Pretty-print the seqeval classification report."""
    print(metrics["report"])


def save_metrics(metrics: dict, path: str) -> None:
    """Save metrics (including the report string) to JSON."""
    import json

    out = dict(metrics)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
