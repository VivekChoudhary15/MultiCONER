"""Inference: run a trained model on a split and save predictions in CoNLL
format (token + predicted tag per line, blank line between sentences).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from .data import NERDataset, Sentence, load_split_sentences, make_collate_fn
from .model import MultilingualNER


def predict_split(
    model: MultilingualNER,
    sentences: List[Sentence],
    tokenizer,
    label2id: dict,
    device: str,
    batch_size: int = 32,
    max_length: int = 256,
    lowercase: bool = False,
    amp: bool = False,
) -> List[Tuple[List[str], List[str], List[str]]]:
    """Predict tags for a list of Sentence objects.

    Returns a list of (tokens, true_tags, pred_tags) per sentence.
    """
    model.eval()
    use_amp = amp and str(device).startswith("cuda")
    dataset = NERDataset(
        sentences,
        tokenizer,
        label2id,
        max_length=max_length,
        lowercase=lowercase,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=make_collate_fn(tokenizer.pad_token_id),
    )
    id2label = model.id2label

    results: List[Tuple[List[str], List[str], List[str]]] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
            preds = model.decode(logits, (labels != -100).to(device))

            for i in range(labels.shape[0]):
                lab = labels[i].tolist()
                true_seq = [id2label[l] for l in lab if l != -100]
                results.append((None, true_seq, preds[i]))

    # Attach raw tokens (dataset order matches sentences order). Sentences
    # longer than max_length were truncated, so trim tokens to match.
    assert len(results) == len(sentences)
    for idx, sentence in enumerate(sentences):
        _, true_seq, pred_seq = results[idx]
        results[idx] = (sentence.tokens[: len(pred_seq)], true_seq, pred_seq)

    return results


def save_conll_predictions(
    results: List[Tuple[List[str], List[str], List[str]]],
    output_path: str | Path,
) -> None:
    """Write predictions to a CoNLL file: token<TAB>pred_tag per line,
    blank line between sentences."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tab = "\t"
    nl = "\n"
    with output_path.open("w", encoding="utf-8") as f:
        for tokens, true_tags, pred_tags in results:
            for token, pred_tag in zip(tokens, pred_tags):
                f.write(token + tab + pred_tag + nl)
            f.write(nl)


def save_predictions_json(
    results: List[Tuple[List[str], List[str], List[str]]],
    output_path: str | Path,
) -> None:
    """Save predictions as JSON for downstream inspection."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"tokens": tokens, "true_tags": true_tags, "pred_tags": pred_tags}
        for tokens, true_tags, pred_tags in results
    ]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
