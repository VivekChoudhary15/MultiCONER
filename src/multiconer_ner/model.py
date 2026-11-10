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


# ---------------------------------------------------------------------------
# Compatibility wrapper around the installed CRF package.
#
# PyPI distributes the CRF implementation under the mixed-case name "TorchCRF"
# (forward(h, labels, mask) + viterbi_decode(h, mask)), while the canonical
# "torchcrf" API uses (emissions, tags, mask) forward + decode(emissions, mask).
# This wrapper normalises both to a single interface used by MultilingualNER.
# ---------------------------------------------------------------------------
class _CRF(nn.Module):
    """Unified CRF interface: forward(emissions, tags, mask) -> loss;
    decode(emissions, mask) -> List[List[int]]."""

    def __init__(self, num_labels: int):
        super().__init__()
        backend = None
        try:
            from torchcrf import CRF as _CRFBackend  # type: ignore
            backend = _CRFBackend(num_labels, batch_first=True)
            self._kind = "torchcrf"
        except Exception:
            try:
                from TorchCRF import CRF as _CRFBackend  # type: ignore
                backend = _CRFBackend(num_labels)  # no batch_first arg
                self._kind = "TorchCRF"
            except Exception as exc:
                raise ImportError(
                    "Neither torchcrf nor TorchCRF is installable/importable: "
                    f"{exc}"
                ) from exc
        self.backend = backend

    def forward(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        """Return mean negative log-likelihood (scalar)."""
        if self._kind == "torchcrf":
            # torchcrf.forward already returns mean NLL over the batch.
            return self.backend(emissions, tags, mask=mask, reduction="mean")
        # TorchCRF: forward(h, labels, mask) may return per-sequence or summed NLL.
        # Force it to a scalar mean over the batch.
        nll = self.backend(emissions, tags, mask)
        if nll.dim() > 0:
            nll = nll.mean()
        return nll

    def decode(
        self, emissions: torch.Tensor, mask: torch.BoolTensor
    ) -> list:
        """Return best tag-index sequences (List[List[int]])."""
        if self._kind == "torchcrf":
            return self.backend.decode(emissions, mask)
        # TorchCRF: viterbi_decode(h, mask).
        return self.backend.viterbi_decode(emissions, mask)


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
                self.crf = _CRF(num_labels)
            except ImportError as exc:
                print(f"[model] {exc}; falling back to linear head.")
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
                # CRF negative log-likelihood (wrapper returns mean NLL).
                loss = self.crf(logits, crf_labels, mask)
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
        """Return predicted tag strings (list of lists), one per batch item.

        When a CRF is attached, Viterbi decoding is used so the output tag
        sequence is guaranteed to be a valid BIO path.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids, attention_mask=attention_mask
            )
        logits = outputs["logits"]  # (B, T, L)
        mask = attention_mask.bool()
        predictions: list = []

        if self.use_crf and self.crf is not None:
            # torchcrf.decode returns List[List[int]] (best tag index sequence).
            best_paths = self.crf.decode(logits, mask)
            for path in best_paths:
                predictions.append([self.id2label[p] for p in path])
        else:
            for i in range(logits.shape[0]):
                seq_mask = mask[i]
                seq_logits = logits[i][seq_mask]
                pred_ids = seq_logits.argmax(dim=-1).tolist()
                predictions.append([self.id2label[p] for p in pred_ids])

        return predictions

    def save(self, output_dir: str | Path, tokenizer: PreTrainedTokenizerBase) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Backbone weights + config.
        self.backbone.save_pretrained(output_dir)
        # Classifier + dropout weights.
        torch.save(self.classifier.state_dict(), output_dir / "classifier.bin")
        torch.save(self.dropout.state_dict(), output_dir / "dropout.bin")
        # CRF weights (if present).
        if self.use_crf and self.crf is not None:
            torch.save(self.crf.state_dict(), output_dir / "crf.bin")

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
        crf_path = output_dir / "crf.bin"
        if model.use_crf and crf_path.exists():
            model.crf.load_state_dict(torch.load(crf_path, map_location="cpu"))

        return model
