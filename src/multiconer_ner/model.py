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
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedTokenizerBase


# --------------------------------------------------------------------------- #
# Stage A3: Coarse-to-fine label mapping (67 fine -> 13 coarse tags)
# --------------------------------------------------------------------------- #

COARSE_LABELS: List[str] = [
    "O",
    "B-PER", "I-PER",      # Person
    "B-LOC", "I-LOC",      # Location
    "B-ORG", "I-ORG",      # Organization
    "B-PROD", "I-PROD",    # Product
    "B-EVT", "I-EVT",      # Event
    "B-MISC", "I-MISC",    # Miscellaneous
]

COARSE_LABEL2ID = {label: i for i, label in enumerate(COARSE_LABELS)}
COARSE_ID2LABEL = {i: label for i, label in enumerate(COARSE_LABELS)}

FINE_TO_COARSE = {
    "O": "O",
    # Person
    "B-Artist": "B-PER", "I-Artist": "I-PER",
    "B-Athlete": "B-PER", "I-Athlete": "I-PER",
    "B-Politician": "B-PER", "I-Politician": "I-PER",
    "B-Cleric": "B-PER", "I-Cleric": "I-PER",
    "B-Scientist": "B-PER", "I-Scientist": "I-PER",
    "B-OtherPER": "B-PER", "I-OtherPER": "I-PER",
    # Location
    "B-HumanSettlement": "B-LOC", "I-HumanSettlement": "I-LOC",
    "B-OtherLOC": "B-LOC", "I-OtherLOC": "I-LOC",
    "B-Facility": "B-LOC", "I-Facility": "I-LOC",
    "B-Station": "B-LOC", "I-Station": "I-LOC",
    # Organization
    "B-ORG": "B-ORG", "I-ORG": "I-ORG",
    "B-PublicCorp": "B-ORG", "I-PublicCorp": "I-ORG",
    "B-PrivateCorp": "B-ORG", "I-PrivateCorp": "I-ORG",
    "B-SportsGRP": "B-ORG", "I-SportsGRP": "I-ORG",
    "B-MusicalGRP": "B-ORG", "I-MusicalGRP": "I-ORG",
    # Product
    "B-WrittenWork": "B-PROD", "I-WrittenWork": "I-PROD",
    "B-VisualWork": "B-PROD", "I-VisualWork": "I-PROD",
    "B-MusicalWork": "B-PROD", "I-MusicalWork": "I-PROD",
    "B-ArtWork": "B-PROD", "I-ArtWork": "I-PROD",
    "B-OtherPROD": "B-PROD", "I-OtherPROD": "I-PROD",
    "B-Software": "B-PROD", "I-Software": "I-PROD",
    "B-Vehicle": "B-PROD", "I-Vehicle": "I-PROD",
    # Miscellaneous
    "B-Disease": "B-MISC", "I-Disease": "I-MISC",
    "B-Food": "B-MISC", "I-Food": "I-MISC",
    "B-Drink": "B-MISC", "I-Drink": "I-MISC",
    "B-Symptom": "B-MISC", "I-Symptom": "I-MISC",
    "B-AnatomicalStructure": "B-MISC", "I-AnatomicalStructure": "I-MISC",
    "B-MedicalProcedure": "B-MISC", "I-MedicalProcedure": "I-MISC",
    "B-Medication/Vaccine": "B-MISC", "I-Medication/Vaccine": "I-MISC",
    "B-AerospaceManufacturer": "B-MISC", "I-AerospaceManufacturer": "I-MISC",
    "B-CarManufacturer": "B-MISC", "I-CarManufacturer": "I-MISC",
    "B-SportsManager": "B-MISC", "I-SportsManager": "I-MISC",
    "B-Clothing": "B-MISC", "I-Clothing": "I-MISC",
}


def fine_to_coarse_labels(fine_labels: List[int], fine_id2label: Dict[int, str],
                          coarse_label2id: Dict[str, int]) -> List[int]:
    """Map fine-grained label IDs to coarse label IDs."""
    return [
        coarse_label2id[FINE_TO_COARSE.get(fine_id2label.get(l, "O"), "O")]
        for l in fine_labels
    ]


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
            # torchcrf.forward returns the LOG-LIKELIHOOD (higher = better), so
            # negate it to get a loss to minimise.
            return -self.backend(emissions, tags, mask=mask, reduction="mean")
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
        coarse_head: bool = False,
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
                print(
                    f"[model] WARNING: {exc}; falling back to linear head. "
                    "Install it with: pip install pytorch-crf"
                )
                self.use_crf = False

        # --- Stage A3: Auxiliary coarse head ---
        self.coarse_head = None
        if coarse_head:
            self.coarse_head = nn.Linear(hf_config.hidden_size, len(COARSE_LABELS))

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    @staticmethod
    def _compact(
        logits: torch.Tensor, word_mask: torch.Tensor, labels: Optional[torch.Tensor] = None
    ):
        """Gather the first-subword position of every word to the left.

        Labels are only defined on the first subword of each word (others are
        -100), so the CRF / predictions must run over those positions only.
        Otherwise predictions contain one tag per *subword* (incl. <s>, </s>)
        and no longer line up with the gold word-level tags.

        Returns (emissions [B, W, L], mask [B, W] bool, tags [B, W] or None).
        """
        word_mask = word_mask.bool()
        B, _, L = logits.shape
        lengths = word_mask.sum(dim=1)                      # (B,)
        W = max(int(lengths.max().item()), 1)
        # Stable sort puts True positions first, preserving their order.
        order = torch.argsort((~word_mask).to(torch.int8), dim=1, stable=True)[:, :W]
        emissions = logits.gather(1, order.unsqueeze(-1).expand(B, W, L))
        mask = torch.arange(W, device=logits.device).unsqueeze(0) < lengths.unsqueeze(1)
        tags = None
        if labels is not None:
            tags = labels.gather(1, order).clamp(min=0)
            tags = tags.masked_fill(~mask, 0)
        # torchcrf requires the first timestep of every sequence to be on.
        mask[:, 0] = True
        return emissions, mask, tags

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        coarse_labels: Optional[torch.Tensor] = None,
        no_context_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)  # (B, T, num_labels)

        # --- Stage A3: Coarse head ---
        coarse_logits = None
        if self.coarse_head is not None:
            coarse_logits = self.coarse_head(sequence_output)  # (B, T, num_coarse)

        loss = None
        loss_terms = {}
        if labels is not None:
            # --- Fine-grained loss ---
            if self.use_crf and self.crf is not None:
                emissions, mask, tags = self._compact(logits.float(), labels != -100, labels)
                # CRF negative log-likelihood over word-level positions (mean NLL).
                loss_fine = self.crf(emissions, tags, mask)
            else:
                flat_logits = logits.float().view(-1, self.num_labels)
                flat_labels = labels.view(-1)
                if (flat_labels != -100).any():
                    loss_fine = nn.functional.cross_entropy(
                        flat_logits, flat_labels, ignore_index=-100
                    )
                else:
                    loss_fine = torch.tensor(0.0, device=logits.device)

            # --- Coarse head loss ---
            loss_coarse = torch.tensor(0.0, device=logits.device)
            if self.coarse_head is not None and coarse_labels is not None:
                flat_coarse_logits = coarse_logits.float().view(-1, len(COARSE_LABELS))
                flat_coarse_labels = coarse_labels.view(-1)
                valid = flat_coarse_labels != -100
                if valid.any():
                    loss_coarse = nn.functional.cross_entropy(
                        flat_coarse_logits[valid], flat_coarse_labels[valid]
                    )

            # --- No-context loss (if context was used, mask it out) ---
            loss_no_context = torch.tensor(0.0, device=logits.device)
            if no_context_mask is not None:
                # no_context_mask: True for positions that are NOT context (i.e., real words)
                # We compute CE only on those positions to force the model to
                # not over-rely on retrieved context.
                flat_logits_nc = logits.float().view(-1, self.num_labels)
                flat_labels_nc = labels.view(-1)
                flat_mask = no_context_mask.view(-1).bool()
                valid = (flat_labels_nc != -100) & flat_mask
                if valid.any():
                    loss_no_context = nn.functional.cross_entropy(
                        flat_logits_nc[valid], flat_labels_nc[valid]
                    )

            loss = loss_fine + loss_coarse + loss_no_context
            loss_terms = {
                "loss_fine": loss_fine,
                "loss_coarse": loss_coarse,
                "loss_no_context": loss_no_context,
            }

        return {"loss": loss, "logits": logits, "coarse_logits": coarse_logits, **loss_terms}

    def decode(self, logits: torch.Tensor, word_mask: torch.Tensor) -> list:
        """Decode logits into word-level tag strings (one list per sentence).

        ``word_mask`` marks the first subword of every word (``labels != -100``).
        Output length per sentence == number of words kept after truncation,
        so it lines up exactly with the gold tags.
        """
        emissions, mask, _ = self._compact(logits.float(), word_mask)
        n_words = word_mask.bool().sum(dim=1).tolist()
        predictions: list = []
        if self.use_crf and self.crf is not None:
            # Viterbi decoding -> valid BIO path.
            best_paths = self.crf.decode(emissions, mask)
        else:
            best_paths = emissions.argmax(dim=-1).tolist()
        for path, n in zip(best_paths, n_words):
            predictions.append([self.id2label[p] for p in path[:n]])
        return predictions

    def predict_tags(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> list:
        """Return predicted word-level tag strings (list of lists)."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        return self.decode(outputs["logits"], word_mask)

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

    def load_weights(self, ckpt_dir: str | Path) -> None:
        """Load backbone/classifier/CRF weights from a checkpoint dir IN PLACE
        (keeps the same parameter objects, so an existing optimizer stays valid)."""
        ckpt_dir = Path(ckpt_dir)
        saved_backbone = AutoModel.from_pretrained(ckpt_dir)
        self.backbone.load_state_dict(saved_backbone.state_dict())
        del saved_backbone
        self.classifier.load_state_dict(
            torch.load(ckpt_dir / "classifier.bin", map_location="cpu")
        )
        crf_path = ckpt_dir / "crf.bin"
        if self.use_crf and self.crf is not None:
            if not crf_path.exists():
                raise FileNotFoundError(f"{crf_path} missing (checkpoint trained without CRF?)")
            self.crf.load_state_dict(torch.load(crf_path, map_location="cpu"))

    @classmethod
    def load(cls, output_dir: str | Path) -> "MultilingualNER":
        output_dir = Path(output_dir)
        with open(output_dir / "label2id.json", "r", encoding="utf-8") as f:
            label2id = json.load(f)
        with open(output_dir / "id2label.json", "r", encoding="utf-8") as f:
            id2label = {int(k): v for k, v in json.load(f).items()}
        with open(output_dir / "model_config.json", "r", encoding="utf-8") as f:
            model_cfg = json.load(f)

        # Build the backbone straight from the saved weights in output_dir.
        model = cls(
            model_name=str(output_dir),
            num_labels=model_cfg["num_labels"],
            id2label=id2label,
            label2id=label2id,
            use_crf=model_cfg.get("use_crf", False),
        )
        model.model_name = model_cfg["model_name"]

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
