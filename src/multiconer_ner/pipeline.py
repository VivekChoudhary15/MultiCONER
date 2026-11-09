"""End-to-end pipeline: load CoNLL data, build label set, train, evaluate, save model + predictions."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoTokenizer

from .config import Config, DataConfig
from .data import (
    NERDataset,
    build_label_list,
    load_split_sentences,
    make_dataloaders,
)
from .evaluate import evaluate_predictions, print_report, save_metrics
from .inference import predict_split, save_conll_predictions, save_predictions_json
from .model import MultilingualNER
from .trainer import Trainer

logger = logging.getLogger("multiconer_ner")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_languages(data_root: str | Path, multilingual: bool, language: str) -> List[str]:
    """Resolve the list of language folders to train/evaluate on."""
    data_root = Path(data_root)
    if multilingual or language == "all":
        folders = sorted(
            p.name for p in data_root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
        return folders
    return [language]


def build_label_maps(train_sentences_by_lang: Dict[str, List]) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Build label list and id<->label maps from training sentences across languages."""
    all_tags: List[List[str]] = [
        s.tags for sentences in train_sentences_by_lang.values() for s in sentences
    ]
    label_list = build_label_list(all_tags)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    return label_list, label2id, id2label


def run_pipeline(config: Config) -> dict:
    setup_logging()
    data_cfg: DataConfig = config.data

    # 1. Resolve languages and load data.
    languages = load_languages(data_cfg.data_root, data_cfg.multilingual, data_cfg.language)
    logger.info(f"Languages: {languages}")

    train_sentences_by_lang: Dict[str, List] = {}
    val_sentences_by_lang: Dict[str, List] = {}
    test_sentences_by_lang: Dict[str, List] = {}

    for lang in languages:
        train_sentences_by_lang[lang] = []
        for split in data_cfg.train_splits:
            train_sentences_by_lang[lang].extend(
                load_split_sentences(data_cfg.data_root, lang, split)
            )
        val_sentences_by_lang[lang] = load_split_sentences(
            data_cfg.data_root, lang, data_cfg.val_split
        )
        test_sentences_by_lang[lang] = load_split_sentences(
            data_cfg.data_root, lang, data_cfg.test_split
        )
        logger.info(
            f"{lang}: train={len(train_sentences_by_lang[lang])} "
            f"dev={len(val_sentences_by_lang[lang])} "
            f"test={len(test_sentences_by_lang[lang])}"
        )

    # 2. Build label space from training data.
    label_list, label2id, id2label = build_label_maps(train_sentences_by_lang)
    logger.info(f"Number of labels: {len(label_list)} (incl. 'O')")

    # 3. Tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

    # 4. Pooled dataloaders across selected languages (multilingual training).
    pooled_train_ds = ConcatDataset([
        NERDataset(train_sentences_by_lang[lang], tokenizer, label2id,
                   max_length=data_cfg.max_length, lowercase=data_cfg.lowercase)
        for lang in languages
    ])
    pooled_val_ds = ConcatDataset([
        NERDataset(val_sentences_by_lang[lang], tokenizer, label2id,
                   max_length=data_cfg.max_length, lowercase=data_cfg.lowercase)
        for lang in languages
    ])

    pooled_train_loader = DataLoader(
        pooled_train_ds, batch_size=config.train.train_batch_size, shuffle=True
    )
    pooled_val_loader = DataLoader(
        pooled_val_ds, batch_size=config.train.val_batch_size, shuffle=False
    )

    # 5. Model.
    model = MultilingualNER(
        model_name=config.model.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        classifier_dropout=config.model.classifier_dropout,
        use_crf=config.model.use_crf,
        freeze_backbone=config.model.freeze_backbone,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 6. Train.
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=pooled_train_loader,
        val_loader=pooled_val_loader,
        tokenizer=tokenizer,
        id2label=id2label,
        device=device,
    )
    train_result = trainer.fit()
    logger.info(f"Training complete. Best dev F1: {train_result['best_dev_f1']:.4f}")

    # 7. Save best model + config to a stable location.
    best_dir = trainer.output_dir / "best"
    final_model_dir = (
        Path(config.log.output_dir) / config.log.experiment_name / "best_model"
    )
    if best_dir.exists():
        if final_model_dir.exists():
            shutil.rmtree(final_model_dir)
        shutil.copytree(best_dir, final_model_dir)
    else:
        model.save(final_model_dir, tokenizer)

    config.save_json(final_model_dir / "pipeline_config.json")
    summary = {
        "best_dev_f1": train_result["best_dev_f1"],
        "best_dev_epoch": train_result["best_dev_epoch"],
        "best_dev_step": train_result["best_dev_step"],
        "label_list": label_list,
        "label2id": label2id,
        "num_labels": len(label_list),
        "languages": languages,
        "model_name": config.model.model_name,
        "output_dir": str(final_model_dir),
    }
    with open(final_model_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Best model saved to: {final_model_dir}")

    # 8. Final evaluation on test set (per language + pooled).
    final_model = MultilingualNER.load(final_model_dir)
    final_model.to(device)

    pooled_results: List[dict] = []
    all_true: List[List[str]] = []
    all_pred: List[List[str]] = []

    for lang in languages:
        lang_test_sentences = test_sentences_by_lang[lang]
        results = predict_split(
            model=final_model,
            sentences=lang_test_sentences,
            tokenizer=tokenizer,
            label2id=label2id,
            device=device,
            batch_size=config.train.val_batch_size,
            max_length=data_cfg.max_length,
            lowercase=data_cfg.lowercase,
        )
        true_tags = [r[1] for r in results]
        pred_tags = [r[2] for r in results]
        metrics = evaluate_predictions(true_tags, pred_tags)

        lang_pred_dir = (
            Path(config.log.predictions_dir) / config.log.experiment_name / lang
        )
        save_conll_predictions(results, lang_pred_dir / "test_predictions.conll")
        save_predictions_json(results, lang_pred_dir / "test_predictions.json")
        save_metrics(metrics, lang_pred_dir / "test_metrics.json")

        logger.info(
            f"{lang} test -> precision={metrics['overall_precision']:.4f} "
            f"recall={metrics['overall_recall']:.4f} f1={metrics['overall_f1']:.4f}"
        )
        pooled_results.append(
            {
                "language": lang,
                "metrics": {k: v for k, v in metrics.items() if k != "report"},
            }
        )
        all_true.extend(true_tags)
        all_pred.extend(pred_tags)

    pooled_metrics = evaluate_predictions(all_true, all_pred)
    pooled_dir = Path(config.log.predictions_dir) / config.log.experiment_name
    save_metrics(pooled_metrics, pooled_dir / "pooled_test_metrics.json")
    logger.info(
        f"POOLED test -> precision={pooled_metrics['overall_precision']:.4f} "
        f"recall={pooled_metrics['overall_recall']:.4f} f1={pooled_metrics['overall_f1']:.4f}"
    )
    print_report(pooled_metrics)

    summary["pooled_test_metrics"] = {
        k: v for k, v in pooled_metrics.items() if k != "report"
    }
    summary["per_language_test"] = pooled_results
    with open(final_model_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
