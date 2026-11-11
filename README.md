# MultiCoNER Multilingual NER Pipeline

A clean, modular, observable training pipeline for fine-grained multilingual Named
Entity Recognition on the **MultiCoNER 2** dataset (SemEval 2023 Task 2). Built on
HuggingFace Transformers (XLM-RoBERTa / mDeBERTa) with PyTorch.

## Implemented stages

| Stage | Name | Status | Description |
|-------|------|--------|-------------|
| **A0** | Baseline | ✅ Done | BIO repair, XLM-R + CRF, language-balanced sampling, checkpoints, TensorBoard, per-language + pooled eval |
| **A1** | Retrieval | ✅ Done | Offline Wikipedia/Wikidata BM25 index; retrieved context injected into model input; flag-gated (`--use-retrieval`) |

The pipeline does **not yet** include A2–A5 (context filter, noise training, gated fusion, ensemble). Those are planned on top of A1.

## Repository layout

```
MultiCoNER/
├── dataset/multiconer2023/   # CoNLL files (your existing data)
├── scripts/
│   └── build_kb.py           # Build offline KB index (Stage A1)
├── src/multiconer_ner/       # pipeline package
│   ├── config.py             # all hyperparameters / paths (A0 + A1 config)
│   ├── data.py               # CoNLL parser, BIO repair, label alignment, DataLoaders, context support
│   ├── model.py              # MultilingualNER (XLM-R + CRF head, save/load)
│   ├── trainer.py            # train/val loop, TensorBoard, checkpoints, early stopping on dev F1
│   ├── evaluate.py           # seqeval P/R/F1 overall & per entity type
│   ├── inference.py          # prediction + CoNLL/JSON output (context-aware)
│   ├── kb.py                 # KBIndex: offline BM25 index (Stage A1)
│   ├── retriever.py          # Retriever: context formatting (Stage A1)
│   ├── pipeline.py           # orchestrate train -> evaluate -> predict -> save (A0 + A1)
│   └── main.py               # CLI entry point (A0 + A1 flags)
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

## Design: flag-driven, retraining required

**Important:** Stage A1 is a **training-time change**, not inference-only. The model
must be retrained with `--use-retrieval` because retrieved context becomes part of
the model input. Without the flag, the pipeline runs exactly as A0 (no retrieval,
no context, identical behavior).

The retrieval flag is fully gated:

```bash
# A0 baseline (no retrieval) — identical to before A1 was added
python -m src.multiconer_ner.main --language EN-English --epochs 5

# A1 retrieval (retrains with context)
python -m src.multiconer_ner.main --language EN-English --use-retrieval --epochs 5
```

## How A1 works (data layer only)

No changes to `model.py` or `trainer.py` were needed. Retrieval is handled
entirely at the data level:

1. **KB index** (`kb.py` + `scripts/build_kb.py`): BM25 index over Wikipedia/Wikidata
   entity entries (title + description). Built once, saved to disk, reused across runs.
2. **Context injection** (`data.py`): Retrieved context tokens are prepended to the
   sentence tokens. Context tokens get `-100` labels (ignored by loss); only sentence
   words are scored. Context is truncated to `context_max_tokens` to preserve the
   sentence within `max_length`.
3. **CRF decoding** (`model.py`): The CRF decodes only word positions where labels
   are not `-100`, so context subwords are automatically excluded from the output tag
   sequence. No architecture changes needed.
4. **Inference** (`inference.py`): Same retrieval logic is applied at prediction time
   so the model sees context at test time exactly as it did during training.

## Outputs

After a run you will find:

- `models/<experiment_name>/best_model/`
  - `pipeline_config.json` — full config used (includes `use_retrieval`)
  - `training_summary.json` — best dev F1, per-language + pooled test metrics
  - `label2id.json`, `id2label.json` — label mappings
  - `model_config.json` — backbone + num_labels + use_crf
  - tokenizer files + model weights + `crf.bin`
- `runs/<experiment_name>/` — TensorBoard logs (`tensorboard --logdir runs`)
- `outputs/<experiment_name>/<language>/`
  - `test_predictions.conll` — token<TAB>pred_tag CoNLL predictions
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
  FR, ES, ZH, PT, SV). The A0/A1 baseline trains on clean train/dev and evaluates
  on the test split. Noise-robust training (A3) is planned as a future phase.
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
```
