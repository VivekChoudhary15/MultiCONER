"""
Model layer for the MultiCoNER multilingual NER pipeline.

Thin wrapper around a HuggingFace transformer backbone (XLM-RoBERTa,
mDeBERTa, ...) with a token-classification head and an optional linear-chain
CRF. Handles saving / loading of the model, tokenizer, config, and label
mappings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedTokenizerBase


class MultilingualNER(nn.Module):
    """
    Multilingual NER model.

    Architecture:
        transformer_backbone -> dropout -> linear_classifier (num_labels)
    Optionally appends a linear-chain CRF over the tag sequence.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        classifier_dropout: float = 0.1,
        use_crf: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id
        self.use_crf = use_crf

        hf_config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label={str(k): v for k, v in id2label.items()},
            label2id=label2id,
            classifier_dropout=classifier_dropout,
        )
        self.backbone = AutoModel.from_pretrained(model_name, config=hf_config)
        self.dropout = nn.Dropout(hf_config.classifier_dropout or classifier_dropout)
        self.classifier = nn.Linear(hf_config.hidden_size, num_labels)

        self.crf = None
        if use_crf:
            try:
                from torchcrf import CRF  # type: ignore

                self.crf = CRF(num_labels, batch_first=True)
            except Exception as exc:  # pragma: no cover - optional dependency
                print(
                    f"[model] torchcrf not available ({exc}); falling back to linear head."
                )
                self.use_crf = False

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)  # (B, T, num_labels)

        loss = None
        if labels is not None:
            if self.use_crf and self.crf is not None:
                mask = attention_mask.bool()
                crf_labels = labels.clamp(min=0)
                loss = -self.crf(logits, crf_labels, mask=mask, reduction="mean")
            else:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                valid = active_labels != -100
                if valid.any():
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(active_logits[valid], active_labels[valid])

        return {"loss": loss, "logits": logits}

    def predict_tags(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list:
        """Return predicted tag strings (list of lists), one per batch item."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids, attention_mask=attention_mask
            )
        logits = outputs["logits"]  # (B, T, L)
        mask = attention_mask.bool()
        predictions: list = []
        for i in range(logits.shape[0]):
            seq_mask = mask[i]
            seq_logits = logits[i][seq_mask]
            pred_ids = seq_logits.argmax(dim=-1).tolist()
            predictions.append([self.id2label[p] for p in pred_ids])
        return predictions

    def save(self, output_dir: str | Path, tokenizer: PreTrainedTokenizerBase) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Backbone weights.
        self.backbone.save_pretrained(output_dir)
        # Classifier + dropout weights.
        torch.save(self.classifier.state_dict(), output_dir / "classifier.bin")
        torch.save(self.dropout.state_dict(), output_dir / "dropout.bin")

        # Tokenizer.
        tokenizer.save_pretrained(output_dir)

        # Label maps + config.
        with open(output_dir / "label2id.json", "w", encoding="utf-8") as f:
            json.dump(self.label2id, f, indent=2, ensure_ascii=False)
        with open(output_dir / "id2label.json", "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.id2label.items()}, f, indent=2, ensure_ascii=False)

        model_cfg = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "use_crf": self.use_crf,
        }
        with open(output_dir / "model_config.json", "w", encoding="utf-8") as f:
            json.dump(model_cfg, f, indent=2)

    @classmethod
    def load(cls, output_dir: str | Path) -> "MultilingualNER":
        output_dir = Path(output_dir)
        with open(output_dir / "label2id.json", "r", encoding="utf-8") as f:
            label2id = json.load(f)
        with open(output_dir / "id2label.json", "r", encoding="utf-8") as f:
            id2label = {int(k): v for k, v in json.load(f).items()}
        with open(output_dir / "model_config.json", "r", encoding="utf-8") as f:
            model_cfg = json.load(f)

        model = cls(
            model_name=model_cfg["model_name"],
            num_labels=model_cfg["num_labels"],
            id2label=id2label,
            label2id=label2id,
            use_crf=model_cfg.get("use_crf", False),
        )
        model.backbone = AutoModel.from_pretrained(output_dir)
        classifier_path = output_dir / "classifier.bin"
        if classifier_path.exists():
            model.classifier.load_state_dict(
                torch.load(classifier_path, map_location="cpu")
            )
        dropout_path = output_dir / "dropout.bin"
        if dropout_path.exists():
            model.dropout.load_state_dict(
                torch.load(dropout_path, map_location="cpu")
            )
        return model
