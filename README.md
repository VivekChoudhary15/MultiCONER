# MultiCoNER Multilingual NER Pipeline

A clean, modular, observable training pipeline for fine-grained multilingual Named
Entity Recognition on the **MultiCoNER 2** dataset (SemEval 2023 Task 2). Built on
HuggingFace Transformers (XLM-RoBERTa / mDeBERTa) with PyTorch.

## Implemented stages

| Stage | Name | Status | Description |
|-------|------|--------|-------------|
| **A0** | Baseline | ✅ Done | BIO repair, XLM-R + CRF, language-balanced sampling, checkpoints, TensorBoard, per-language + pooled eval |
| **A1** | Retrieval | ✅ Done | Offline Wikipedia/Wikidata BM25 index; retrieved context injected into model input; flag-gated (`--use-retrieval`) |
| **A2** | Context filter | ✅ Done | Relevance thresholding; limits context entries; skips low-confidence retrieval; flag-gated (A2 flags) |
| **A3** | Noise-robust training | ✅ Done | Character-level noise augmentation; KL divergence (clean vs noisy); auxiliary coarse head; flag-gated (`--noise-training`) |

The pipeline does **not yet** include A4–A5 (gated fusion, ensemble). Those are planned on top of A3.

## Repository layout

```
MultiCoNER/
├── dataset/multiconer2023/   # CoNLL files (your existing data)
├── scripts/
│   └── build_kb.py           # Build offline KB index (Stage A1)
├── src/multiconer_ner/       # pipeline package
│   ├── config.py             # all hyperparameters / paths (A0 + A1 + A2 + A3 config)
│   ├── data.py               # CoNLL parser, BIO repair, label alignment, DataLoaders, context support, NoiseAugmenter
│   ├── model.py              # MultilingualNER (XLM-R + CRF head, coarse head, save/load)
│   ├── trainer.py            # train/val loop, TensorBoard, checkpoints, early stopping on dev F1, A3 dual-pass training
│   ├── evaluate.py           # seqeval P/R/F1 overall & per entity type
│   ├── inference.py          # prediction + CoNLL/JSON output (context-aware)
│   ├── kb.py                 # KBIndex: offline BM25 index (Stage A1)
│   ├── retriever.py          # Retriever: context formatting + A2 relevance filter + A2 k-fold
│   ├── pipeline.py           # orchestrate train -> evaluate -> predict -> save (A0 + A1 + A2 + A3)
│   └── main.py               # CLI entry point (A0 + A1 + A2 + A3 flags)
├── models/                   # saved models (created at runtime)
├── runs/                     # TensorBoard event files (created at runtime)
├── outputs/                  # test predictions & metrics (created at runtime)
├── kb_index/                 # offline KB index (created by build_kb.py, Stage A1)
├── requirements.txt
├── README.md
└── multiconer2023_data_analytics.ipynb   # your existing notebook (untouched)
```

## Quick start (Linux GPU)

### 1. Activate the virtual environment

```bash
cd /path/to/MultiCoNER
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train A0 baseline (no retrieval) — current default behavior

```bash
# Single language
python -m src.multiconer_ner.main --language EN-English --epochs 5 --batch-size 16

# Multilingual (all 12 languages pooled, language-balanced sampling)
python -m src.multiconer_ner.main --multilingual --epochs 5 --batch-size 16
```

### 4. Stage A1: Retrieval-augmented NER

**Step 1 — Build the offline KB index (run once):**

```bash
# Build for specific languages
python scripts/build_kb.py --languages en bn hi de es fr it pt sv uk fa zh \
    --data-root dataset/multiconer2023 \
    --output ./kb_index

# Force rebuild if Wikipedia data changes
python scripts/build_kb.py --languages all --data-root dataset/multiconer2023 \
    --output ./kb_index --rebuild
```

**Step 2 — Train with retrieval (retrains model with context injected):**

```bash
# Single language with retrieval
python -m src.multiconer_ner.main \
    --language EN-English \
    --use-retrieval \
    --kb-index-dir ./kb_index \
    --retrieval-top-k 5 \
    --epochs 5 \
    --batch-size 16

# Multilingual with retrieval
python -m src.multiconer_ner.main \
    --multilingual \
    --use-retrieval \
    --kb-index-dir ./kb_index \
    --retrieval-top-k 3 \
    --epochs 5 \
    --batch-size 16
```

**Step 3 — Evaluate / infer with retrieval (uses saved KB index):**

```bash
# Inference automatically uses retrieval if --use-retrieval is set
# (the saved best_model already contains the retrieval-trained weights)
python -m src.multiconer_ner.main \
    --language EN-English \
    --use-retrieval \
    --kb-index-dir ./kb_index \
    --epochs 1  # evaluation only, uses checkpoint
```

### 5. Stage A2: Context filter (relevance gating)

A2 adds relevance filtering so that retrieved context is only injected when it
is actually useful. This directly addresses the problem where irrelevant BM25
matches poison the model input.

```bash
# Train with retrieval + context filter (skip low-confidence matches)
python -m src.multiconer_ner.main \
    --language EN-English \
    --use-retrieval \
    --kb-index-dir ./kb_index \
    --retrieval-min-score 1.0 \
    --max-context-entries 3 \
    --skip-low-confidence \
    --epochs 5 \
    --batch-size 16

# Multilingual with retrieval + filter
python -m src.multiconer_ner.main \
    --multilingual \
    --use-retrieval \
    --kb-index-dir ./kb_index \
    --retrieval-min-score 1.0 \
    --max-context-entries 3 \
    --skip-low-confidence \
    --epochs 5 \
    --batch-size 16
```

### 6. Stage A3: Noise-robust training

A3 adds noise augmentation and multi-component loss to make the model robust
to keyboard-layout noise (~30% of test instances are corrupted). Trains on both
clean and noisy versions of each sentence with KL divergence between their
predictions.

```bash
# Train with noise-robust training (clean + noisy dual pass)
python -m src.multiconer_ner.main \
    --language EN-English \
    --noise-training \
    --noise-rate 0.3 \
    --noise-types typo,swap,drop \
    --coarse-head \
    --epochs 5 \
    --batch-size 16

# Multilingual with noise-robust training
python -m src.multiconer_ner.main \
    --multilingual \
    --noise-training \
    --noise-rate 0.3 \
    --coarse-head \
    --epochs 5 \
    --batch-size 16

# Combine A1+A2+A3: retrieval + context filter + noise training
python -m src.multiconer_ner.main \
    --language EN-English \
    --use-retrieval \
    --kb-index-dir ./kb_index \
    --retrieval-min-score 1.0 \
    --max-context-entries 3 \
    --skip-low-confidence \
    --noise-training \
    --noise-rate 0.3 \
    --coarse-head \
    --epochs 5 \
    --batch-size 16
```

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--language` | `EN-English` | Language folder, or `all` for every folder. |
| `--multilingual` | off | Train on all language folders (pooled, language-balanced sampling). |
| `--model` | `xlm-roberta-base` | HF model id. |
| `--data-root` | `dataset/multiconer2023` | Path to the CoNLL dataset root. |
| `--epochs` | `5` | Training epochs. |
| `--batch-size` | `16` | Train batch size. |
| `--lr` | `3e-5` | Learning rate. |
| `--max-length` | `256` | Max sequence length (tokens). |
| `--experiment-name` | `multiconer_ner` | Subfolder under `runs/` and `models/`. |
| `--use-crf` | on | Linear-chain CRF head (default True). Use `--no-use-crf` to disable. |
| `--freeze-backbone` | off | Freeze the transformer backbone. |
| `--amp` | off | bf16 mixed precision on GPU. |
| `--use-retrieval` | off | **Stage A1:** Enable Wikipedia/Wikidata retrieval. |
| `--kb-index-dir` | `./kb_index` | **Stage A1:** KB index directory. |
| `--rebuild-kb` | off | **Stage A1:** Force rebuild KB index. |
| `--retrieval-top-k` | 5 | **Stage A1:** KB entries per sentence. |
| `--retrieval-threshold` | 0.0 | **Stage A1:** Min BM25 score to keep entry. |
| `--context-max-tokens` | 64 | **Stage A1:** Max tokens reserved for context. |
| `--retrieval-min-score` | 0.0 | **Stage A2:** Min BM25 score to accept context. |
| `--max-context-entries` | 3 | **Stage A2:** Max KB entries injected per sentence. |
| `--skip-low-confidence` | on | **Stage A2:** Skip context when best BM25 < `--retrieval-min-score`. |
| `--context-filter-mode` | threshold | **Stage A2:** `threshold` / `adaptive` / `none`. |
| `--kb-train-folds` | 0 | **Stage A2:** K-fold leakage-free retrieval for train (0 = off). |
| `--noise-training` | off | **Stage A3:** Enable noise-robust training. |
| `--noise-rate` | 0.3 | **Stage A3:** Fraction of entity tokens to corrupt. |
| `--noise-types` | typo,swap,drop | **Stage A3:** Comma-separated: typo,swap,drop. |
| `--coarse-head` | off | **Stage A3:** Auxiliary coarse head (13 tags). |
| `--noise-lambda-coarse` | 0.5 | **Stage A3:** Weight for coarse head loss. |
| `--noise-lambda-context` | 0.3 | **Stage A3:** Weight for no-context loss. |
| `--noise-lambda-kl` | 1.0 | **Stage A3:** Weight for KL divergence (clean vs noisy). |

## Design: flag-driven, retraining required

**Important:** Stages A1, A2, and A3 are **training-time changes**, not inference-only. The model
must be retrained with the relevant flags because they change the model input or training objective.
Without the flags, the pipeline runs exactly as A0 (no retrieval, no context, no noise, identical behavior).

```bash
# A0 baseline (no retrieval, no noise) — identical to before A1 was added
python -m src.multiconer_ner.main --language EN-English --epochs 5

# A1 retrieval (retrains with context)
python -m src.multiconer_ner.main --language EN-English --use-retrieval --epochs 5

# A3 noise training (retrains with noisy copies + KL loss)
python -m src.multiconer_ner.main --language EN-English --noise-training --epochs 5
```

## How A1/A2 work (data layer only)

No changes to `model.py` or `trainer.py` were needed for A1/A2. Retrieval and context filtering are handled
entirely at the data level:

1. **KB index** (`kb.py` + `scripts/build_kb.py`): BM25 index over Wikipedia/Wikidata
   entity entries (title + description). Built once, saved to disk, reused across runs.
2. **Context injection** (`data.py`): Retrieved context tokens are prepended to the
   sentence tokens. Context tokens get `-100` labels (ignored by loss); only sentence
   words are scored. Context is truncated to `context_max_tokens` to preserve the
   sentence within `max_length`.
3. **K-fold KB for training** (`retriever.py`): With `--kb-train-folds K`, train sentences
   only retrieve KB entries for mentions seen in *other* folds — prevents the model from
   looking up its own gold entities during training (leakage-free).
4. **CRF decoding** (`model.py`): The CRF decodes only word positions where labels
   are not `-100`, so context subwords are automatically excluded from the output tag
   sequence. No architecture changes needed.
5. **Inference** (`inference.py`): Same retrieval logic is applied at prediction time
   so the model sees context at test time exactly as it did during training.

**Stage A2 — Context filter (relevance gating):**

| A2 component | How it works |
|--------------|--------------|
| **Relevance threshold** (`--retrieval-min-score`) | Only inject context when the best BM25 score meets the threshold. Sentences below it get pure A0 (no context). |
| **Max context entries** (`--max-context-entries`) | Cap context at N entries (default 3) so irrelevant matches don't dominate the input. |
| **Skip low-confidence** (`--skip-low-confidence`) | When enabled and no entry clears the threshold, the sentence is left unchanged — effectively falling back to A0 for that instance. |
| **Per-language behavior** | Different languages have different Wikipedia coverage; tune `--retrieval-min-score` per language on dev F1. |

This directly addresses the finding that **unfiltered retrieval hurts performance** (irrelevant context confuses the model). PAI observed retrieval benefits only on clean subsets; A2 makes retrieval safe on noisy data by skipping low-confidence matches.

**Stage A3 — Noise-robust training (data + model + trainer):**

| A3 component | How it works |
|--------------|--------------|
| **Noise augmentation** (`--noise-training`) | Character-level typo/swap/drop noise applied to entity tokens during training. Matches the ~30% keyboard-layout noise in MultiCoNER 2 test sets. |
| **Dual forward pass** | Each training batch is passed through the model twice: once clean, once noisy. Loss is computed on both. |
| **KL divergence** (`--noise-lambda-kl`) | KL divergence between clean and noisy prediction distributions forces the model to produce similar outputs regardless of noise. |
| **Coarse head** (`--coarse-head`) | Auxiliary 13-way classification head (PER, LOC, ORG, PROD, EVT, MISC) provides regularization for fine-grained decisions. |
| **No-context loss** (`--noise-lambda-context`) | Forces the model to predict entities correctly even without relying on retrieved context. |

## Outputs

After a run you will find:

- `models/<experiment_name>/best_model/`
  - `pipeline_config.json` — full config used (includes all A1/A2/A3 flags)
  - `training_summary.json` — best dev F1, per-language + pooled test metrics
  - `label2id.json`, `id2label.json` — label mappings
  - `model_config.json` — backbone + num_labels + use_crf + coarse_head
  - tokenizer files + model weights + `crf.bin`
- `runs/<experiment_name>/` — TensorBoard logs (`tensorboard --logdir runs`)
- `outputs/<experiment_name>/<language>/`
  - `test_predictions.conlt` — token<TAB>pred_tag CoNLL predictions
  - `test_predictions.json` — predictions + true tags
  - `test_metrics.json` — per-language precision/recall/F1 by entity type
- `outputs/<experiment_name>/pooled_test_metrics.json` — pooled micro metrics
- `kb_index/entries.json` — offline KB index entries (Stage A1)

## Reload a saved model for inference

```python
from src.multiconer_ner.model import MultilingualNER

model = MultilingualNER.load("models/multiconer_ner/best_model")
model.eval()
tags = model.predict_tags(input_ids, attention_mask, word_mask)  # Viterbi-decoded via CRF
```

## Notes

- Your existing notebook `multiconer2023_data_analytics.ipynb` was **not modified**.
- The pipeline builds the BIO label set automatically from the tags in your
  training split (67 labels for the standard MultiCoNER 2 taxonomy).
- All sentences are **BIO-repaired** before training, so invalid `I-` tags
  (e.g. `O I-PER`, `B-LOC I-ORG`) are promoted to `B-` and the sequence stays valid.
- MultiCoNER 2 adds keyboard-layout noise to ~30% of test instances (EN, DE, IT,
  FR, ES, ZH, PT, SV). Stage A3 noise-robust training directly addresses this by
  training on synthetically corrupted data with KL divergence regularization.
- `pytorch-crf` is required for the CRF head; `rank_bm25` is required for Stage A1
  retrieval. Both are listed in `requirements.txt`.

## Requirements

```
torch>=2.0.0
transformers>=4.40.0
seqeval>=1.2.0
tensorboard>=2.0.0
tqdm>=4.60.0
pyyaml>=6.0
sentencepiece>=0.2.0
pytorch-crf>=0.7.2
rank_bm25>=0.2.2
requests>=2.28.0
numpy
```
