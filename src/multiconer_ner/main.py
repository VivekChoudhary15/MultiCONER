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
        "--use-crf", action="store_true",
        help="Add a CRF layer (requires torchcrf).",
    )
    parser.add_argument(
        "--freeze-backbone", action="store_true",
        help="Freeze the transformer backbone (feature extraction).",
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

    run_pipeline(config)


if __name__ == "__main__":
    main()
