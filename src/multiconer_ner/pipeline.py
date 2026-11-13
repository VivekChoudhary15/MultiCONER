"""
End-to-end pipeline: load CoNLL data, build label set, train, evaluate,
save model + predictions.

Implements ablation A0 (baseline) and A1/A2 (retrieval + context filter):

  * A0: BIO-repaired tag sequences, XLM-RoBERTa + CRF,
        language-balanced sampling, checkpoints, early stopping on dev F1,
        per-language + pooled test evaluation.
  * A1: Wikipedia/Wikidata retrieval; retrieved context injected into input.
  * A2: Relevance filtering -- skip context when BM25 score is below a
        threshold; cap context entries; per-language filtering.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from .config import Config, DataConfig
from .data import (
    build_label_list,
    load_split_sentences,
    make_dataloaders,
)
from .evaluate import evaluate_predictions, print_report, save_metrics
from .inference import predict_split, save_conll_predictions, save_predictions_json
from .data import NoiseAugmenter
from .kb import KBIndex
from .model import MultilingualNER
from .retriever import Retriever, sentence_mentions
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
        # MULTI-Multilingual is just a mix of the 12 monolingual sets; including
        # it would duplicate every language in train/dev/test.
        folders = sorted(
            p.name for p in data_root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
            and not p.name.upper().startswith("MULTI")
        )
        return folders
    return [language]


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_label_maps(
    train_sentences_by_lang: Dict[str, List],
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Build label list and id<->label maps from training sentences across languages."""
    all_tags: List[List[str]] = [
        s.tags
        for sentences in train_sentences_by_lang.values()
        for s in sentences
    ]
    label_list = build_label_list(all_tags)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    return label_list, label2id, id2label


def run_pipeline(config: Config) -> dict:
    setup_logging()
    set_seed(config.train.seed)
    data_cfg: DataConfig = config.data

    # ------------------------------------------------------------------ #
    # 1. Resolve languages and load data.
    #    All sentences are BIO-repaired inside stream_conll / load_split_sentences.
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # 2. Build label space from training data.
    # ------------------------------------------------------------------ #
    # Use train + dev + test tags so an entity type that is rare/absent in
    # train cannot crash encoding of dev/test with a KeyError.
    label_list, label2id, id2label = build_label_maps(
        {
            f"{lang}/{name}": sents
            for name, by_lang in (
                ("train", train_sentences_by_lang),
                ("dev", val_sentences_by_lang),
                ("test", test_sentences_by_lang),
            )
            for lang, sents in by_lang.items()
        }
    )
    logger.info(f"Number of labels: {len(label_list)} (incl. 'O')")

    # ------------------------------------------------------------------ #
    # 3. Tokenizer.
    # ------------------------------------------------------------------ #
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

    # ------------------------------------------------------------------ #
    # 3b. Stage A1+A2: Retrieval + context filter.
    #    Build/load KB index, then attach filtered context to sentences.
    # ------------------------------------------------------------------ #
    retriever: Optional[Retriever] = None
    if data_cfg.use_retrieval:
        kb_dir = Path(data_cfg.kb_index_dir)
        if data_cfg.rebuild_kb or not (kb_dir / "entries.json").exists():
            logger.info(
                f"[A1] Building KB index at {kb_dir} (languages: {languages})"
            )
            # scripts/build_kb.py is not part of the package.
            import sys
            from .config import project_root

            sys.path.insert(0, str(project_root() / "scripts"))
            from build_kb import build_kb_index

            kb = build_kb_index(
                languages=languages,
                data_root=str(data_cfg.data_root),
                output_dir=kb_dir,
                train_splits=data_cfg.train_splits,
                rebuild=True,
            )
        else:
            logger.info(f"[A1] Loading KB index from {kb_dir}")
            kb = KBIndex.load(kb_dir)
        retriever = Retriever(kb)

        # Attach retrieved context with A2 relevance filtering.
        # Sentences whose best BM25 score < retrieval_min_score are left
        # without context (pure A0 behavior for those instances).
        for lang in languages:
            for split_name, split_dict in (
                ("train", train_sentences_by_lang),
                ("dev", val_sentences_by_lang),
                ("test", test_sentences_by_lang),
            ):
                import time

                t0 = time.time()
                filter_kwargs = dict(
                    top_k=data_cfg.retrieval_top_k,
                    min_score=data_cfg.retrieval_min_score,
                    max_entries=data_cfg.max_context_entries,
                    max_tokens=data_cfg.context_max_tokens,
                    skip_low_confidence=data_cfg.skip_low_confidence,
                )
                kfold = split_name == "train" and data_cfg.kb_train_folds > 1
                if kfold:
                    # Leakage-free: train sentences only see KB entries of
                    # mentions that occur in other folds (see Retriever).
                    split_dict[lang] = retriever.attach_context_kfold(
                        split_dict[lang], language=lang,
                        n_folds=data_cfg.kb_train_folds, seed=config.train.seed,
                        **filter_kwargs,
                    )
                else:
                    split_dict[lang] = retriever.attach_context(
                        split_dict[lang], language=lang, **filter_kwargs
                    )
                with_ctx = [s for s in split_dict[lang] if s.context]
                # Diagnostic: how often the context names one of the
                # sentence's own gold entities (train should resemble test).
                n_own = sum(
                    1 for s in with_ctx
                    if any(f"[KB] {m}:" in s.context for m in sentence_mentions(s))
                )
                logger.info(
                    f"[A1+A2{'+kfold' if kfold else ''}] {lang}/{split_name}: context attached to "
                    f"{len(with_ctx)}/{len(split_dict[lang])} sentences; "
                    f"context names an own gold entity in "
                    f"{100 * n_own / max(len(with_ctx), 1):.1f}% of them "
                    f"({time.time() - t0:.1f}s)"
                )

    # ------------------------------------------------------------------ #
    # 3c. Stage A3: Noise augmentation for training (if enabled).
    # ------------------------------------------------------------------ #
    noise_augmenter: Optional[NoiseAugmenter] = None
    if data_cfg.noise_training:
        noise_augmenter = NoiseAugmenter(
            noise_rate=data_cfg.noise_rate,
            noise_types=data_cfg.noise_types,
            seed=config.train.seed,
        )
        logger.info(
            f"[A3] Noise-robust training enabled: rate={data_cfg.noise_rate}, "
            f"types={data_cfg.noise_types}, coarse_head={data_cfg.coarse_head}"
        )

    # ------------------------------------------------------------------ #
    # 4. Combine sentences across languages and build DataLoaders.
    #    If multilingual, the train loader uses LanguageBalancedSampler
    #    (equal probability per language) so high-resource languages do not
    #    dominate low-resource ones.
    # ------------------------------------------------------------------ #
    all_train: List = []
    all_val: List = []
    for lang in languages:
        all_train.extend(train_sentences_by_lang[lang])
        all_val.extend(val_sentences_by_lang[lang])

    train_loader, val_loader = make_dataloaders(
        train_sentences=all_train,
        val_sentences=all_val,
        tokenizer=tokenizer,
        label2id=label2id,
        data_config=data_cfg,
        train_batch_size=config.train.train_batch_size,
        val_batch_size=config.train.val_batch_size,
        context_max_tokens=data_cfg.context_max_tokens,
        language_sampling_temperature=config.train.language_sampling_temperature,
        seed=config.train.seed,
        noise_augmenter=noise_augmenter,
    )

    # ------------------------------------------------------------------ #
    # 5. Model.
    #    Default backbone: xlm-roberta-base. Default head: linear + CRF (A0 baseline).
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # 6. Train.
    # ------------------------------------------------------------------ #
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        id2label=id2label,
        device=device,
    )
    train_result = trainer.fit()
    logger.info(f"Training complete. Best dev F1: {train_result['best_dev_f1']:.4f}")

    # ------------------------------------------------------------------ #
    # 7. Save best model + config to a stable location.
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # 8. Final evaluation on test set (per language + pooled).
    #    Test sentences already carry retrieved context from step 3b, so we
    #    do NOT re-retrieve here (pass retriever=None).
    # ------------------------------------------------------------------ #
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
            amp=config.train.fp16,
            retriever=None,  # context already attached in step 3b
            language=lang,
            context_max_tokens=data_cfg.context_max_tokens,
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
