"""Command-line entry point for the MultiCoNER multilingual NER pipeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import Config
from .pipeline import run_pipeline, setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train / evaluate a multilingual NER model on MultiCoNER 2."
    )
    parser.add_argument(
        "--language", default="EN-English",
        help="Language folder to use (e.g. EN-English) or 'all'/'MULTI-Multilingual'.",
    )
    parser.add_argument(
        "--multilingual", action="store_true",
        help="Train on all language folders under data_root (multilingual track).",
    )
    parser.add_argument(
        "--model", default="xlm-roberta-base",
        help="HF model id (xlm-roberta-base, xlm-roberta-large, microsoft/mdeberta-v3-base, ...).",
    )
    parser.add_argument(
        "--data-root", default=None,
        help="Path to dataset/multiconer2023 (default: <repo>/dataset/multiconer2023).",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Train batch size.",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-5, help="Learning rate.",
    )
    parser.add_argument(
        "--max-length", type=int, default=256, help="Max token length.",
    )
    parser.add_argument(
        "--experiment-name", default="multiconer_ner",
        help="Experiment name (used for runs/ and models/ subfolders).",
    )
    parser.add_argument(
        "--use-crf", action=argparse.BooleanOptionalAction, default=True,
        help="Linear-chain CRF head (A0 default: on). Use --no-use-crf to disable.",
    )
    parser.add_argument(
        "--amp", action="store_true",
        help="bf16 mixed precision on GPU (faster, less memory; good for *-large models).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--resume-from", default=None,
        help="Checkpoint dir to resume from, e.g. models/<exp>/epoch3. "
             "--epochs is the TOTAL number of epochs (not extra epochs).",
    )
    parser.add_argument(
        "--save-optimizer-state", action=argparse.BooleanOptionalAction, default=True,
        help="Save optimizer/scheduler/RNG state in epoch checkpoints (for exact resume).",
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=32, help="Dev/test batch size.",
    )
    parser.add_argument(
        "--eval-every", type=int, default=500,
        help="Evaluate on dev every N steps (0 = only at end of each epoch).",
    )
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Early-stopping patience (number of dev evals without F1 improvement).",
    )
    parser.add_argument(
        "--freeze-backbone", action="store_true",
        help="Freeze the transformer backbone (feature extraction).",
    )
    # --- Stage A1: Retrieval flags ---
    parser.add_argument(
        "--use-retrieval", action=argparse.BooleanOptionalAction, default=False,
        help="Enable Wikipedia/Wikidata retrieval (Stage A1). "
             "Retrains model with retrieved context injected into input.",
    )
    parser.add_argument(
        "--kb-index-dir", default="./kb_index",
        help="Directory to read/write the offline KB index (Stage A1).",
    )
    parser.add_argument(
        "--rebuild-kb", action="store_true",
        help="Force rebuild the KB index even if it exists (Stage A1).",
    )
    parser.add_argument(
        "--retrieval-top-k", type=int, default=5,
        help="Number of KB entries to retrieve per sentence (Stage A1).",
    )
    parser.add_argument(
        "--retrieval-threshold", type=float, default=0.0,
        help="Minimum BM25 score to keep a retrieved entry (Stage A1).",
    )
    parser.add_argument(
        "--context-max-tokens", type=int, default=64,
        help="Max tokens reserved for retrieved context (Stage A1).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging()

    config = Config()
    if args.data_root:
        config.data.data_root = args.data_root
    config.data.language = args.language
    config.data.multilingual = args.multilingual
    config.model.model_name = args.model
    config.train.num_train_epochs = args.epochs
    config.train.train_batch_size = args.batch_size
    config.train.learning_rate = args.lr
    config.data.max_length = args.max_length
    config.log.experiment_name = args.experiment_name
    config.model.use_crf = args.use_crf
    config.model.freeze_backbone = args.freeze_backbone
    config.train.fp16 = args.amp
    config.train.seed = args.seed
    config.train.resume_from = args.resume_from
    config.log.save_optimizer_state = args.save_optimizer_state
    config.train.val_batch_size = args.eval_batch_size
    config.train.eval_every = args.eval_every
    config.train.early_stopping_patience = args.patience

    # --- Stage A1: Retrieval config ---
    config.data.use_retrieval = args.use_retrieval
    config.data.kb_index_dir = args.kb_index_dir
    config.data.rebuild_kb = args.rebuild_kb
    config.data.retrieval_top_k = args.retrieval_top_k
    config.data.retrieval_threshold = args.retrieval_threshold
    config.data.context_max_tokens = args.context_max_tokens

    run_pipeline(config)


if __name__ == "__main__":
    main()
