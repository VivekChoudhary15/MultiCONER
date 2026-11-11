"""
Data layer for MultiCoNER multilingual NER.

Responsibilities:
  * Stream CoNLL files one sentence at a time (memory friendly for the
    multi-million-token test splits).
  * Repair invalid BIO sequences so every training example is a valid BIO tag
    sequence.
  * Build the BIO label set from the tags observed in the training data.
  * Align NER labels to subword tokens produced by a HF tokenizer.
  * Build PyTorch DataLoaders for train / val / test, including a
    language-balanced sampler for multilingual training.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
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


def repair_bio_tags(tags: Sequence[str]) -> List[str]:
    """
    Repair an invalid BIO tag sequence into a valid one.

    Rules:
      * "O" stays "O" and resets the previous entity type.
      * "B-X" stays "B-X" and starts a new span of type X.
      * "I-X" stays "I-X" only if the previous tag was "B-X" or "I-X" of the
        same type. Otherwise it is converted to "B-X" (orphan or mismatched I-).

    Parameters
    ----------
    tags : sequence of str
        Input BIO tags (possibly invalid).

    Returns
    -------
    list of str
        Repaired BIO tags.
    """
    repaired: List[str] = []
    prev_type: Optional[str] = None  # entity type of the previous B-/I- tag

    for tag in tags:
        if tag == "O":
            repaired.append("O")
            prev_type = None
        elif tag.startswith("B-"):
            etype = tag[2:]
            repaired.append(tag)
            prev_type = etype
        elif tag.startswith("I-"):
            etype = tag[2:]
            if prev_type == etype:
                repaired.append(tag)
            else:
                # Orphan I- or mismatched I- (e.g. O I-PER, or B-LOC I-ORG).
                # Promote to B- so the sequence stays valid BIO.
                repaired.append(f"B-{etype}")
                prev_type = etype
        else:
            # Unknown tag format: keep as-is but break the current span.
            repaired.append(tag)
            prev_type = None

    return repaired


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
    never materialize the whole file as token-level records. Every sentence's
    BIO tags are repaired before being stored.
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
                    tags=repair_bio_tags(list(tags)),
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
            if line.startswith("# id ") or line.startswith("# id\t"):
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
    All returned sentences have repaired BIO tags.
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
                padding=False,  # dynamic padding happens in collate_batch
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
            "input_ids": ex["input_ids"],
            "attention_mask": ex["attention_mask"],
            "labels": ex["labels"],
        }


def make_collate_fn(pad_token_id: int):
    """Pad a batch to its own longest sequence (much faster than padding
    every sentence to max_length)."""

    def collate_batch(batch: Sequence[dict]) -> dict:
        max_len = max(len(ex["input_ids"]) for ex in batch)
        n = len(batch)
        input_ids = torch.full((n, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((n, max_len), dtype=torch.long)
        labels = torch.full((n, max_len), -100, dtype=torch.long)
        for i, ex in enumerate(batch):
            L = len(ex["input_ids"])
            input_ids[i, :L] = torch.tensor(ex["input_ids"], dtype=torch.long)
            attention_mask[i, :L] = torch.tensor(ex["attention_mask"], dtype=torch.long)
            labels[i, :L] = torch.tensor(ex["labels"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate_batch


class LanguageBalancedSampler(Sampler[int]):
    """
    Sampler that draws examples from a ConcatDataset such that each language
    is sampled with (approximately) equal probability, regardless of how many
    sentences that language contributes.

    This prevents high-resource languages (EN, ES, FR) from dominating
    multilingual training batches.
    """

    def __init__(
        self,
        dataset_sizes: Sequence[int],
        temperature: float = 1.0,
        num_samples: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ):
        self.dataset_sizes = list(dataset_sizes)
        self.num_languages = len(self.dataset_sizes)
        self.temperature = temperature
        self.num_samples = num_samples or sum(self.dataset_sizes)
        self.generator = generator or torch.Generator()

        # Cumulative offsets to map (language, local_index) -> global index.
        self.cumulative_sizes: List[int] = [0]
        for size in self.dataset_sizes:
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + size)

        # Equal probability per language, temperature-scaled.
        lang_weights = [1.0] * self.num_languages
        if temperature != 1.0:
            lang_weights = [w ** (1.0 / temperature) for w in lang_weights]
        total = sum(lang_weights)
        self.lang_weights = [w / total for w in lang_weights]

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.num_samples):
            # Pick a language according to lang_weights (seeded generator).
            r = float(torch.rand(1, generator=self.generator).item())
            cumulative = 0.0
            lang_idx = self.num_languages - 1
            for i, w in enumerate(self.lang_weights):
                cumulative += w
                if r < cumulative:
                    lang_idx = i
                    break

            # Pick a random sentence from that language.
            lang_size = self.dataset_sizes[lang_idx]
            local_idx = int(torch.randint(0, lang_size, (1,), generator=self.generator).item())
            global_idx = self.cumulative_sizes[lang_idx] + local_idx
            yield global_idx

    def __len__(self) -> int:
        return self.num_samples


def make_dataloaders(
    *,
    train_sentences: Sequence[Sentence],
    val_sentences: Sequence[Sentence],
    tokenizer: PreTrainedTokenizerBase,
    label2id: Dict[str, int],
    data_config: "DataConfig",
    train_batch_size: int,
    val_batch_size: int,
    language_sampling_temperature: float = 1.0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Build train / val / test DataLoaders sharing one label space.

    If ``data_config.multilingual`` is True, the train loader uses a
    language-balanced sampler so each language contributes equally to each
    epoch. The val loader is pooled. Test data is NOT tokenized here; it is
    encoded per language at prediction time (the test splits are huge).
    """
    collate = make_collate_fn(tokenizer.pad_token_id)

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

    # Train loader.
    if data_config.multilingual:
        # Per-language sizes, in the SAME order the sentences appear in
        # train_sentences (sentences are concatenated language by language).
        lang_order: List[str] = []
        lang_to_count: Dict[str, int] = {}
        for s in train_sentences:
            if s.language not in lang_to_count:
                lang_order.append(s.language)
                lang_to_count[s.language] = 0
            lang_to_count[s.language] += 1
        sizes = [lang_to_count[lang] for lang in lang_order]

        sampler = LanguageBalancedSampler(
            sizes,
            temperature=language_sampling_temperature,
            generator=torch.Generator().manual_seed(seed),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=train_batch_size,
            sampler=sampler,
            shuffle=False,  # sampler handles ordering
            collate_fn=collate,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=train_batch_size, shuffle=True, drop_last=False,
            collate_fn=collate,
            generator=torch.Generator().manual_seed(seed),
        )

    val_loader = DataLoader(
        val_ds, batch_size=val_batch_size, shuffle=False, drop_last=False,
        collate_fn=collate,
    )
    return train_loader, val_loader


def make_language_balanced_loader(
    datasets: Sequence[Dataset],
    batch_size: int,
    temperature: float = 1.0,
    shuffle: bool = True,
) -> DataLoader:
    """
    Build a DataLoader over a ConcatDataset of per-language datasets using
    LanguageBalancedSampler. Useful when you already have separate datasets
    per language and want a single balanced training stream.
    """
    from torch.utils.data import ConcatDataset

    concat = ConcatDataset(list(datasets))
    sizes = [len(d) for d in datasets]
    sampler = LanguageBalancedSampler(sizes, temperature=temperature)
    pad_id = datasets[0].tokenizer.pad_token_id  # type: ignore[attr-defined]
    return DataLoader(concat, batch_size=batch_size, sampler=sampler, shuffle=False,
                      collate_fn=make_collate_fn(pad_id))
