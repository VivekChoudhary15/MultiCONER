"""Re-score a trained model on one language's test split with the current
encoder, and recompute the pooled F1 using the run's saved predictions for
the other languages.

Used to correct A1/A2 Chinese scores: the old encoder truncated CJK sentences
away behind their KB context, so most ZH test sentences were never scored.

    python scripts/rescore_language.py --experiment a2_multi_kfold --language ZH-Chinese

Writes outputs/<exp>/<lang>_rescored/ and outputs/<exp>/pooled_test_metrics_rescored.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from multiconer_ner.data import load_split_sentences  # noqa: E402
from multiconer_ner.evaluate import evaluate_predictions, save_metrics  # noqa: E402
from multiconer_ner.inference import predict_split, save_conll_predictions, save_predictions_json  # noqa: E402
from multiconer_ner.kb import KBIndex  # noqa: E402
from multiconer_ner.model import MultilingualNER  # noqa: E402
from multiconer_ner.retriever import Retriever  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--language", default="ZH-Chinese")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    model_dir = ROOT / args.models_dir / args.experiment / "best_model"
    out_dir = ROOT / args.outputs_dir / args.experiment
    cfg = json.loads((model_dir / "pipeline_config.json").read_text())["data"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sentences = load_split_sentences(cfg["data_root"], args.language, cfg["test_split"])
    if cfg.get("use_retrieval"):
        top_k = cfg.get("retrieval_top_k", 5)
        # Runs from before A2 (A1) have no filter settings: plain top-k context.
        filt = dict(
            top_k=top_k,
            min_score=cfg.get("retrieval_min_score") or 0.0,
            max_entries=cfg.get("max_context_entries") or top_k,
            max_tokens=cfg.get("context_max_tokens", 64),
            skip_low_confidence=bool(cfg.get("skip_low_confidence")),
        )
        print(f"retrieval: {filt}", flush=True)
        sentences = Retriever(KBIndex.load(ROOT / cfg["kb_index_dir"])).attach_context(
            sentences, language=args.language, **filt
        )

    model = MultilingualNER.load(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    t0 = time.time()
    results = predict_split(
        model, sentences, tokenizer, model.label2id, device,
        batch_size=args.batch_size, max_length=cfg["max_length"],
        lowercase=cfg["lowercase"], amp=True,
        context_max_tokens=cfg.get("context_max_tokens", 64),
    )
    scored = sum(1 for r in results if r[1])
    metrics = evaluate_predictions([r[1] for r in results], [r[2] for r in results])
    lang_dir = out_dir / f"{args.language}_rescored"
    lang_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, str(lang_dir / "test_metrics.json"))
    save_predictions_json(results, lang_dir / "test_predictions.json")
    save_conll_predictions(results, lang_dir / "test_predictions.conll")
    old = json.loads((out_dir / args.language / "test_metrics.json").read_text())["overall_f1"]
    print(f"{args.language}: scored {scored}/{len(results)} sentences, "
          f"F1 {metrics['overall_f1']:.4f} (was {old:.4f})  [{time.time() - t0:.0f}s]", flush=True)

    all_true, all_pred = [r[1] for r in results], [r[2] for r in results]
    for lang_path in sorted(p for p in out_dir.iterdir() if p.is_dir() and not p.name.endswith("_rescored")):
        if lang_path.name == args.language:
            continue
        for e in json.loads((lang_path / "test_predictions.json").read_text(encoding="utf-8")):
            all_true.append(e["true_tags"])
            all_pred.append(e["pred_tags"])
    pooled = evaluate_predictions(all_true, all_pred)
    save_metrics(pooled, str(out_dir / "pooled_test_metrics_rescored.json"))
    old_pooled = json.loads((out_dir / "pooled_test_metrics.json").read_text())["overall_f1"]
    print(f"POOLED: F1 {pooled['overall_f1']:.4f} (was {old_pooled:.4f})", flush=True)


if __name__ == "__main__":
    main()
