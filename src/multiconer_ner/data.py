"""
Data layer for MultiCoNER multilingual NER.

Responsibilities:
  * Stream CoNLL files one sentence at a time (memory friendly for the
    multi-million-token test splits).
  * Build the BIO label set from the tags observed in the training data.
  * Align NER labels to subword tokens produced by a HF tokenizer.
  * Build PyTorch DataLoaders for train / val / test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

# A valid CoNLL token row: four whitespace-separated columns, last column a tag like O / B-X / I-X.
_TAG_RE = re.compile(r"^(?:O|[BI]-.+)$")
_META_RE = re.compile(r"\b(domain|corrupt)=([^\s]+)")


@dataclass(frozen=True)
class Sentence:
    """A single annotated sentence."""

    tokens: List[str]
    tags: List[str]
    language: str
    split: str
    doc_id: Optional[str] = None
    domain: Optional[str] = None


def _parse_meta(line: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (domain, corrupt) from a '# ...' metadata header line."""
    fields = dict(_META_RE.findall(line))
    domain = fields.get("domain")
    corrupt = fields.get("corrupt")
    return domain, corrupt


def stream_conll(
    path: Path,
    language: str,
    split: str,
) -> List[Sentence]:
    """Parse one CoNLL file into a list of Sentence objects.

    The file is read line-by-line; blank lines terminate a sentence so we
    never materialize the whole file as token-level records.
    """
    sentences: List[Sentence] = []
    tokens: List[str] = []
    tags: List[str] = []
    doc_id: Optional[str] = None
    domain: Optional[str] = None

    def flush() -> None:
        nonlocal tokens, tags, doc_id, domain
        if tokens:
            sentences.append(
                Sentence(
                    tokens=list(tokens),
                    tags=list(tags),
                    language=language,
                    split=split,
                    doc_id=doc_id,
                    domain=domain,
                )
            )
        tokens = []
        tags = []
        doc_id = None
        domain = None

    with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                flush()
                continue
            if line.startswith("#"):
                # A metadata header ends the current sentence if one is open.
                if tokens:
                    flush()
                domain, _corrupt = _parse_meta(line)
                m = re.search(r"\bid\s+(\S+)", line)
                if m:
                    doc_id = m.group(1)
                continue

            columns = line.split()
            if len(columns) != 4 or not _TAG_RE.fullmatch(columns[-1]):
                # Malformed row: treat as a sentence boundary to avoid
                # propagating misaligned tags.
                flush()
                continue

            tokens.append(columns[0])
            tags.append(columns[-1])

    flush()
    return sentences


def load_split_sentences(
    data_root: str | Path,
    language: str,
    split: str,
) -> List[Sentence]:
    """Load all sentences for a given (language, split) from data_root.

    Expected file: <data_root>/<language>/*_{split}.conll
    """
    data_root = Path(data_root)
    lang_folder = data_root / language
    pattern = f"*_{split}.conll"
    files = sorted(lang_folder.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No {split}.conll files found for language '{language}' under {lang_folder}"
        )
    sentences: List[Sentence] = []
    for fp in files:
        sentences.extend(stream_conll(fp, language, split))
    return sentences


def build_label_list(tag_sequences: Sequence[Sequence[str]]) -> List[str]:
    """Build the sorted BIO label list from observed tags.

    Order: "O" first, then B-<type> / I-<type> pairs grouped by entity type.
    """
    types: set[str] = set()
    for tags in tag_sequences:
        for tag in tags:
            if tag == "O":
                continue
            prefix, etype = tag.split("-", 1)
            types.add(etype)

    label_list: List[str] = ["O"]
    for etype in sorted(types):
        label_list.append(f"B-{etype}")
        label_list.append(f"I-{etype}")
    return label_list


class NERDataset(Dataset):
    """Map-style dataset that aligns NER tags to subword token ids."""

    def __init__(
        self,
        sentences: Sequence[Sentence],
        tokenizer: PreTrainedTokenizerBase,
        label2id: Dict[str, int],
        max_length: int = 256,
        lowercase: bool = False,
    ):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.lowercase = lowercase
        self.examples = self._encode(sentences)

    def _encode(self, sentences: Sequence[Sentence]) -> List[dict]:
        examples: List[dict] = []
        for sent in sentences:
            tokens = [t.lower() if self.lowercase else t for t in sent.tokens]
            tags = sent.tags

            encoding = self.tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors=None,
            )

            word_ids = encoding.word_ids()
            labels: List[int] = []
            previous_word_idx: Optional[int] = None
            for word_idx in word_ids:
                if word_idx is None:
                    # Special token ([CLS], [SEP], padding).
                    labels.append(-100)
                elif word_idx != previous_word_idx:
                    # First subword of a word -> keep the word's label.
                    labels.append(self.label2id[tags[word_idx]])
                else:
                    # Continuation subword -> ignore in loss (-100).
                    labels.append(-100)
                previous_word_idx = word_idx

            examples.append(
                {
                    "input_ids": encoding["input_ids"],
                    "attention_mask": encoding["attention_mask"],
                    "labels": labels,
                    "language": sent.language,
                    "split": sent.split,
                }
            )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(ex["labels"], dtype=torch.long),
        }


def make_dataloaders(
    *,
    train_sentences: Sequence[Sentence],
    val_sentences: Sequence[Sentence],
    test_sentences: Sequence[Sentence],
    tokenizer: PreTrainedTokenizerBase,
    label2id: Dict[str, int],
    data_config: DataConfig,
    train_batch_size: int,
    val_batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders sharing one label space."""
    train_ds = NERDataset(
        train_sentences,
        tokenizer,
        label2id,
        max_length=data_config.max_length,
        lowercase=data_config.lowercase,
    )
    val_ds = NERDataset(
        val_sentences,
        tokenizer,
        label2id,
        max_length=data_config.max_length,
        lowercase=data_config.lowercase,
    )
    test_ds = NERDataset(
        test_sentences,
        tokenizer,
        label2id,
        max_length=data_config.max_length,
        lowercase=data_config.lowercase,
    )

    train_loader = DataLoader(
        train_ds, batch_size=train_batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_batch_size, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=val_batch_size, shuffle=False, drop_last=False
    )
    return train_loader, val_loader, test_loader
