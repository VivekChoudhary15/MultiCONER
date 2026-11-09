# MultiCoNER Multilingual NER Pipeline

A clean, modular, observable training pipeline for fine-grained multilingual Named
Entity Recognition on the **MultiCoNER 2** dataset (SemEval 2023 Task 2). Built on
HuggingFace Transformers (XLM-RoBERTa / mDeBERTa) with PyTorch.

## What this implements

A strong, reproducible baseline modeled on the top MultiCoNER shared-task systems:
a fine-tuned multilingual transformer for token-level NER over the fine-grained
taxonomy, trained end-to-end with validation-based model selection.

### Key capabilities
- **Multi-language / multilingual tracks**: train on one language or all language
  folders (incl. `MULTI-Multilingual`) in a single pooled model.
- **Configurable backbone**: `xlm-roberta-base` (default), `xlm-roberta-large`,
  `microsoft/mdeberta-v3-base`, etc.
- **Observability**: TensorBoard logging of train loss, dev loss, dev overall +
  per-entity-type precision/recall/F1, learning rate, and hyperparameters.
- **Model selection & checkpointing**: best model saved by dev F1; per-epoch
  checkpoints; early stopping on dev F1.
- **Clean modular pipeline**: `data.py`, `model.py`, `trainer.py`, `evaluate.py`,
  `inference.py`, `pipeline.py`, `config.py`, `main.py`.
- **Saved artifacts**: model weights, tokenizer, label maps (`label2id`/`id2label`),
  pipeline config, training history, and a training summary JSON.
- **Test predictions**: per-language CoNLL + JSON predictions and metrics, plus a
  pooled micro-average across languages.

## Repository layout

```
MultiCoNER/
├── dataset/multiconer2023/   # CoNLL files (your existing data)
├── src/multiconer_ner/       # pipeline package
├── models/                   # saved models (created at runtime)
├── runs/                     # TensorBoard event files (created at runtime)
├── outputs/                  # test predictions & metrics (created at runtime)
├── requirements.txt
└── multiconer2023_data_analytics.ipynb   # your existing notebook (untouched)
```

## Quick start

### 1. Activate the project virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Train on a single language (e.g. English)

```powershell
python -m src.multiconer_ner.main --language EN-English --epochs 5 --batch-size 16
```

### 4. Train multilingual (all languages pooled)

```powershell
python -m src.multiconer_ner.main --multilingual --epochs 5 --batch-size 16
```

### 5. Use a stronger backbone

```powershell
python -m src.multiconer_ner.main --language EN-English --model xlm-roberta-large --epochs 5
```

## CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--language` | `EN-English` | Language folder, or `all` for every folder. |
| `--multilingual` | off | Train on all language folders (pooled). |
| `--model` | `xlm-roberta-base` | HF model id. |
| `--data-root` | `dataset/multiconer2023` | Path to the CoNLL dataset root. |
| `--epochs` | `5` | Training epochs. |
| `--batch-size` | `16` | Train batch size. |
| `--lr` | `3e-5` | Learning rate. |
| `--max-length` | `256` | Max sequence length (tokens). |
| `--experiment-name` | `multiconer_ner` | Subfolder under `runs/` and `models/`. |
| `--use-crf` | off | Add a CRF layer (needs `torchcrf`). |
| `--freeze-backbone` | off | Freeze the transformer backbone. |

## Outputs

- `models/<experiment_name>/best_model/` — best model, tokenizer, label maps,
  config, training summary JSON.
- `runs/<experiment_name>/` — TensorBoard logs (`tensorboard --logdir runs`).
- `outputs/<experiment_name>/<language>/` — per-language CoNLL/JSON predictions
  and metrics.
- `outputs/<experiment_name>/pooled_test_metrics.json` — pooled micro metrics.

## Reload a saved model

```python
from src.multiconer_ner.model import MultilingualNER

model = MultilingualNER.load("models/multiconer_ner/best_model")
model.eval()
model.predict_tags(input_ids, attention_mask)
```

## Notes

- Your existing notebook `multiconer2023_data_analytics.ipynb` was **not modified**.
- The pipeline builds the BIO label set automatically from the tags in your
  training split, so it adapts to the exact taxonomy present.
- MultiCoNER 2 adds keyboard-layout noise to ~30% of test instances. The pipeline
  trains on clean train/dev and evaluates on the test split, matching the
  shared-task protocol.

## Requirements

```
torch>=2.0.0
transformers>=4.40.0
seqeval>=1.2.0
tensorboard>=2.0.0
tqdm>=4.60.0
pyyaml>=6.0
```

Optional: `torchcrf` (for `--use-crf`).