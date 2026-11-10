# MultiCoNER Multilingual NER Pipeline

A clean, modular, observable training pipeline for fine-grained multilingual Named
Entity Recognition on the **MultiCoNER 2** dataset (SemEval 2023 Task 2). Built on
HuggingFace Transformers (XLM-RoBERTa / mDeBERTa) with PyTorch.

## Implemented baseline (Ablation A0)

The current pipeline implements **Ablation 0 (A0)**, the plain baseline from the
proposed multi-component approach:

| A0 component | Status | Details |
|--------------|--------|---------|
| **Data prep** | ✅ | CoNLL streaming parser; all tag sequences are **BIO-repaired** (orphan/mismatched `I-` tags promoted to `B-`). |
| **Baseline** | ✅ | **XLM-RoBERTa + linear-chain CRF** token classifier (`use_crf=True` by default). |
| **Language sampling** | ✅ | Multilingual training uses a **language-balanced sampler** so each language contributes equally per batch (prevents EN/ES/FR from dominating BN/HI/FA). |

The pipeline does **not yet** include S3–S7 (retrieval, context filter, noise training, gated fusion, ensemble). Those are planned as phases A1–A5 on top of A0.

## Repository layout

```
MultiCoNER/
├── dataset/multiconer2023/   # CoNLL files (your existing data)
├── src/multiconer_ner/       # pipeline package
│   ├── config.py             # all hyperparameters / paths (A0 defaults)
│   ├── data.py               # CoNLL parser, BIO repair, label alignment, DataLoaders, language-balanced sampler
│   ├── model.py              # MultilingualNER (XLM-R + CRF head, save/load)
│   ├── trainer.py            # train/val loop, TensorBoard, checkpoints, early stopping on dev F1
│   ├── evaluate.py           # seqeval P/R/F1 overall & per entity type
│   ├── inference.py          # prediction + CoNLL/JSON output
│   ├── pipeline.py           # orchestrate train -> evaluate -> predict -> save (A0)
│   └── main.py               # CLI entry point
├── models/                   # saved models (created at runtime)
├── runs/                     # TensorBoard event files (created at runtime)
├── outputs/                  # test predictions & metrics (created at runtime)
├── requirements.txt
├── README.md
└── multiconer2023_data_analytics.ipynb   # your existing notebook (untouched)
```

## Quick start

### 1. Activate the project virtual environment

```powershell
cd C:\Users\offic\Documents\Projects\MultiCoNER
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Train the A0 baseline on a single language (e.g. English)

```powershell
python -m src.multiconer_ner.main --language EN-English --epochs 5 --batch-size 16
```

### 4. Train A0 multilingual (all languages pooled, language-balanced sampling)

```powershell
python -m src.multiconer_ner.main --multilingual --epochs 5 --batch-size 16
```

### 5. Use a stronger backbone

```powershell
python -m src.multiconer_ner.main --language EN-English --model xlm-roberta-large --epochs 5
```

or `microsoft/mdeberta-v3-base` / `microsoft/mdeberta-v3-large`.

## CLI options

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
| `--use-crf` | on (A0 default) | Enable linear-chain CRF (default True for A0 baseline). |
| `--freeze-backbone` | off | Freeze the transformer backbone (feature extraction). |

## Outputs

After a run you will find:

- `models/<experiment_name>/best_model/`
  - `pipeline_config.json` — full config used
  - `training_summary.json` — best dev F1, per-language + pooled test metrics
  - `label2id.json`, `id2label.json` — label mappings
  - `model_config.json` — backbone + num_labels + use_crf
  - tokenizer files + model weights + `crf.bin`
- `runs/<experiment_name>/` — TensorBoard logs. Launch with:
  ```powershell
  tensorboard --logdir runs
  ```
- `outputs/<experiment_name>/<language>/`
  - `test_predictions.conll` — token<TAB>pred_tag CoNLL predictions
  - `test_predictions.json` — predictions + true tags
  - `test_metrics.json` — per-language precision/recall/F1 by entity type
- `outputs/<experiment_name>/pooled_test_metrics.json` — pooled micro metrics

## Reload a saved model for inference

```python
from src.multiconer_ner.model import MultilingualNER

model = MultilingualNER.load("models/multiconer_ner/best_model")
model.eval()
tags = model.predict_tags(input_ids, attention_mask)  # Viterbi-decoded via CRF
```

## Notes

- Your existing notebook `multiconer2023_data_analytics.ipynb` was **not modified**.
- The pipeline builds the BIO label set automatically from the tags in your
  training split (67 labels for the standard MultiCoNER 2 taxonomy).
- All sentences are **BIO-repaired** before training, so invalid `I-` tags
  (e.g. `O I-PER`, `B-LOC I-ORG`) are promoted to `B-` and the sequence stays valid.
- MultiCoNER 2 adds keyboard-layout noise to ~30% of test instances (EN, DE, IT,
  FR, ES, ZH, PT, SV). The A0 baseline trains on clean train/dev and evaluates on
  the test split. Noise-robust training (A3) is planned as a future phase.
- `torchcrf` is required for the CRF head and is listed in `requirements.txt`.

## Requirements

```
torch>=2.0.0
transformers>=4.40.0
seqeval>=1.2.0
tensorboard>=2.0.0
tqdm>=4.60.0
pyyaml>=6.0
sentencepiece>=0.2.0
torchcrf>=1.1.0
```
