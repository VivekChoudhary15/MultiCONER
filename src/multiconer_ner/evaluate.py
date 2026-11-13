"""Evaluation utilities for NER using seqeval.

Computes precision / recall / F1 overall and per entity type from BIO tag
sequences (one list of tag strings per sentence).
"""

from __future__ import annotations

from typing import Dict, List

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2


def _format_report(report: dict, digits: int = 4) -> str:
    rows = [(k, v) for k, v in report.items() if not k.endswith(" avg")]
    avgs = [(k, v) for k, v in report.items() if k.endswith(" avg")]
    width = max([len(k) for k, _ in report.items()] + [12])
    head = f"{'':>{width}} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}"
    line = lambda k, v: (
        f"{k:>{width}} {v['precision']:>9.{digits}f} {v['recall']:>9.{digits}f} "
        f"{v['f1-score']:>9.{digits}f} {int(v['support']):>9}"
    )
    return "\n".join([head, ""] + [line(k, v) for k, v in rows] + [""] + [line(k, v) for k, v in avgs]) + "\n"


def evaluate_predictions(
    true_tags: List[List[str]],
    pred_tags: List[List[str]],
) -> dict:
    """Compute overall and per-entity-type precision/recall/F1 (seqeval,
    strict IOB2) with a single seqeval pass; overall = micro average."""
    assert len(true_tags) == len(pred_tags), (
        f"true ({len(true_tags)}) and pred ({len(pred_tags)}) sentence counts differ"
    )

    report = classification_report(
        true_tags,
        pred_tags,
        scheme=IOB2,
        mode="strict",
        output_dict=True,
        zero_division=0,
    )
    micro = report.get("micro avg", {"precision": 0.0, "recall": 0.0, "f1-score": 0.0})

    by_type: Dict[str, dict] = {}
    for entity_type, scores in report.items():
        if entity_type.endswith(" avg"):
            continue
        by_type[entity_type] = {
            "precision": float(scores.get("precision", 0.0)),
            "recall": float(scores.get("recall", 0.0)),
            "f1": float(scores.get("f1-score", 0.0)),
            "support": int(scores.get("support", 0)),
        }

    return {
        "overall_precision": float(micro["precision"]),
        "overall_recall": float(micro["recall"]),
        "overall_f1": float(micro["f1-score"]),
        "by_type": by_type,
        "report": _format_report(report) if by_type else "(no entities)\n",
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
