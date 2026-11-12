"""
Configuration for the MultiCoNER multilingual NER training pipeline.

All paths are resolved relative to the project root (the directory that
contains this `src` folder). Adjust `DATA_ROOT`, `MODEL_NAME`, and the
language/track settings to match the experiment you want to run.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


def project_root() -> Path:
    """Return the repository root (parent of the `src` package)."""
    return Path(__file__).resolve().parents[2]


@dataclass
class DataConfig:
    """Dataset / dataloader settings."""

    # Root folder that contains the per-language folders (BN-Bangla, ..., MULTI-Multilingual, ...)
    data_root: str = str(project_root() / "dataset" / "multiconer2023")
    # Language folder to train on, or "MULTI-Multilingual" / "all" for multilingual training.
    language: str = "EN-English"
    # Splits to use. Set to ["train", "dev"] for train+dev; ["train"] for train only.
    train_splits: List[str] = field(default_factory=lambda: ["train"])
    # Split used for validation / early stopping / model selection.
    val_split: str = "dev"
    # Split used for final inference / predictions.
    test_split: str = "test"
    # Maximum sequence length (tokens) the model will see.
    max_length: int = 256
    # Whether to lowercase tokens before tokenization (usually False for cased multilingual models).
    lowercase: bool = False
    # If True, train on every language folder found under data_root (multilingual track).
    multilingual: bool = False

    # --- Stage A1: Retrieval configuration ---
    use_retrieval: bool = False
    kb_index_dir: str = "./kb_index"
    rebuild_kb: bool = False
    retrieval_top_k: int = 5
    retrieval_threshold: float = 0.0
    context_max_tokens: int = 64

    # --- Stage A2: Context filter configuration ---
    retrieval_min_score: float = 0.0          # Minimum BM25 score to accept context
    max_context_entries: int = 3              # Max KB entries injected per sentence
    skip_low_confidence: bool = True          # Skip context entirely when below threshold
    context_filter_mode: str = "threshold"    # "threshold" | "adaptive" | "none"
    # Leakage-free training context: K-fold KB for train sentences (0/1 = off).
    kb_train_folds: int = 0


@dataclass
class ModelConfig:
    """Model architecture settings."""

    # HuggingFace model id. Good multi-language choices:
    #   "xlm-roberta-base"  (fast, strong multilingual baseline)
    #   "xlm-roberta-large" (stronger, slower)
    #   "microsoft/mdeberta-v3-base" / "...-v3-large"
    model_name: str = "xlm-roberta-base"
    # If True, load a CRF layer on top of the classifier.
    # The A0 baseline uses CRF, so this defaults to True.
    use_crf: bool = True
    # Dropout probability for the classification head.
    classifier_dropout: float = 0.1
    # Whether to freeze the transformer backbone (feature-extraction mode).
    freeze_backbone: bool = False


@dataclass
class TrainConfig:
    """Training settings."""

    seed: int = 42
    # Batch size per GPU/CPU.
    train_batch_size: int = 16
    val_batch_size: int = 32
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 5
    # Warmup proportion of total steps.
    warmup_ratio: float = 0.1
    # Gradient clipping max norm (0 = disabled).
    max_grad_norm: float = 1.0
    # Log to TensorBoard every N steps.
    log_every: int = 50
    # Evaluate on dev every N steps.
    eval_every: int = 500
    # Early-stopping patience (number of evals without dev-F1 improvement).
    early_stopping_patience: int = 3
    # Mixed precision (False on CPU).
    fp16: bool = False
    # Temperature for language-balanced sampling (1.0 = equal per language).
    # Only used when data_config.multilingual is True.
    language_sampling_temperature: float = 1.0
    # Checkpoint directory to resume from (e.g. models/<exp>/epoch3). None = fresh run.
    resume_from: Optional[str] = None


@dataclass
class LogConfig:
    """Observability / logging settings."""

    # Directory for TensorBoard event files.
    tb_dir: str = str(project_root() / "runs")
    # Directory for saved models and metrics.
    output_dir: str = str(project_root() / "models")
    # Directory for prediction outputs.
    predictions_dir: str = str(project_root() / "outputs")
    # Experiment name (used for tb subfolder & model subfolder).
    experiment_name: str = "multiconer_ner"
    # Save a checkpoint after every epoch.
    save_every_epoch: bool = True
    # Save the best model (by dev F1) automatically.
    save_best_model: bool = True
    # Store optimizer/scheduler/RNG state in each epoch checkpoint (needed for
    # an exact resume; ~2x model size extra per checkpoint).
    save_optimizer_state: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    log: LogConfig = field(default_factory=LogConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(
            data=DataConfig(**d.get("data", {})),
            model=ModelConfig(**d.get("model", {})),
            train=TrainConfig(**d.get("train", {})),
            log=LogConfig(**d.get("log", {})),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "Config":
        import json
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(d)
