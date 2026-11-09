"""
Training loop with validation, early stopping, checkpointing, and
TensorBoard observability.

Responsibilities:
  * Run train / validation epochs with AdamW + linear warmup schedule.
  * Log train loss, dev loss, and dev F1 (overall + per entity type) to
    TensorBoard every `log_every` steps.
  * Evaluate on dev every `eval_every` steps; track best model by dev F1.
  * Early-stop when dev F1 does not improve for `patience` evaluations.
  * Save the best model and the last checkpoint, plus a training-history JSON.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from .config import Config, DataConfig, TrainConfig, LogConfig
from .evaluate import evaluate_predictions
from .model import MultilingualNER


class Trainer:
    def __init__(
        self,
        model: MultilingualNER,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer,
        id2label: Dict[int, str],
        device: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.train_cfg: TrainConfig = config.train
        self.log_cfg: LogConfig = config.log
        self.data_cfg: DataConfig = config.data

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.id2label = id2label

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # TensorBoard writer.
        tb_dir = Path(self.log_cfg.tb_dir) / self.log_cfg.experiment_name
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tb_dir))

        # Output dirs.
        self.output_dir = Path(self.log_cfg.output_dir) / self.log_cfg.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer + scheduler.
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
        )
        num_training_steps = (
            len(self.train_loader) * self.train_cfg.num_train_epochs
        )
        num_warmup_steps = int(num_training_steps * self.train_cfg.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # State.
        self.global_step = 0
        self.epoch = 0
        self.best_dev_f1 = -1.0
        self.best_dev_epoch = -1
        self.best_dev_step = -1
        self.patience_counter = 0
        self.history: List[dict] = []
        self.start_time = time.time()

    def _move_batch(self, batch: dict) -> dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc="train", leave=False)
        for batch in pbar:
            batch = self._move_batch(batch)
            self.optimizer.zero_grad()

            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = out["loss"]
            if loss is None:
                continue
            loss.backward()

            if self.train_cfg.max_grad_norm and self.train_cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.train_cfg.max_grad_norm
                )

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if self.global_step % self.train_cfg.log_every == 0:
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.writer.add_scalar(
                    "train/lr", self.scheduler.get_last_lr()[0], self.global_step
                )

            if (
                self.train_cfg.eval_every > 0
                and self.global_step % self.train_cfg.eval_every == 0
            ):
                dev_metrics = self.evaluate()
                self._log_dev(dev_metrics, step=self.global_step)
                self._maybe_save_best(dev_metrics, step=self.global_step)
                self.model.train()

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None) -> dict:
        loader = loader or self.val_loader
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        true_tags: List[List[str]] = []
        pred_tags: List[List[str]] = []

        for batch in tqdm(loader, desc="eval", leave=False):
            batch = self._move_batch(batch)
            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            if out["loss"] is not None:
                total_loss += out["loss"].item()
                n_batches += 1

            labels = batch["labels"]
            preds = self.model.predict_tags(batch["input_ids"], batch["attention_mask"])

            for i in range(labels.shape[0]):
                lab = labels[i].tolist()
                t_seq = [self.id2label[l] for l in lab if l != -100]
                true_tags.append(t_seq)
                pred_tags.append(preds[i])

        avg_loss = total_loss / max(n_batches, 1)
        metrics = evaluate_predictions(true_tags, pred_tags)
        metrics["loss"] = avg_loss
        metrics["num_true_entities"] = sum(len(t) for t in true_tags)
        return metrics

    def _log_dev(self, metrics: dict, step: int) -> None:
        f1 = metrics.get("overall_f1", 0.0)
        self.writer.add_scalar("dev/loss", metrics.get("loss", 0.0), step)
        self.writer.add_scalar("dev/overall_f1", f1, step)
        self.writer.add_scalar(
            "dev/overall_precision", metrics.get("overall_precision", 0.0), step
        )
        self.writer.add_scalar(
            "dev/overall_recall", metrics.get("overall_recall", 0.0), step
        )
        for etype, vals in metrics.get("by_type", {}).items():
            self.writer.add_scalar(
                f"dev/f1_{etype}", vals.get("f1", 0.0), step
            )

    def _maybe_save_best(self, metrics: dict, step: int) -> None:
        f1 = metrics.get("overall_f1", 0.0)
        improved = f1 > self.best_dev_f1
        if improved:
            self.best_dev_f1 = f1
            self.best_dev_step = step
            self.best_dev_epoch = self.epoch
            self.patience_counter = 0
            if self.log_cfg.save_best_model:
                self._save_checkpoint(tag="best")
            print(
                f"[step {step}] New best dev F1 = {f1:.4f}; saved best checkpoint."
            )
        else:
            self.patience_counter += 1

    def _save_checkpoint(self, tag: str) -> None:
        ckpt_dir = self.output_dir / tag
        self.model.save(ckpt_dir, self.tokenizer)
        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_dev_f1": self.best_dev_f1,
            "best_dev_step": self.best_dev_step,
            "best_dev_epoch": self.best_dev_epoch,
            "history": self.history,
        }
        with open(ckpt_dir / "training_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def fit(self) -> dict:
        """Run the full training loop. Returns the training history."""
        for epoch in range(self.train_cfg.num_train_epochs):
            self.epoch = epoch + 1
            train_loss = self._train_epoch()
            print(f"Epoch {self.epoch}: train_loss={train_loss:.4f}")

            dev_metrics = self.evaluate()
            self._log_dev(dev_metrics, step=self.global_step)
            self._record_history(epoch, train_loss, dev_metrics)
            self._maybe_save_best(dev_metrics, step=self.global_step)

            if self.log_cfg.save_every_epoch:
                self._save_checkpoint(tag=f"epoch{self.epoch}")

            print(
                f"  dev_loss={dev_metrics.get('loss',0):.4f} "
                f"dev_f1={dev_metrics.get('overall_f1',0):.4f} "
                f"patience={self.patience_counter}/{self.train_cfg.early_stopping_patience}"
            )

            if self.patience_counter >= self.train_cfg.early_stopping_patience:
                print(
                    "Early stopping triggered (no dev-F1 improvement for "
                    f"{self.train_cfg.early_stopping_patience} evals)."
                )
                break

        # Persist full history.
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        self.writer.add_hparams(
            {
                "lr": self.train_cfg.learning_rate,
                "batch_size": self.train_cfg.train_batch_size,
                "epochs": self.train_cfg.num_train_epochs,
                "model": self.config.model.model_name,
            },
            {"dev/best_f1": self.best_dev_f1},
        )
        self.writer.close()

        return {
            "best_dev_f1": self.best_dev_f1,
            "best_dev_epoch": self.best_dev_epoch,
            "best_dev_step": self.best_dev_step,
            "history": self.history,
            "output_dir": str(self.output_dir),
        }

    def _record_history(self, epoch: int, train_loss: float, dev_metrics: dict) -> None:
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_metrics.get("loss"),
            "dev_overall_f1": dev_metrics.get("overall_f1"),
            "dev_overall_precision": dev_metrics.get("overall_precision"),
            "dev_overall_recall": dev_metrics.get("overall_recall"),
            "global_step": self.global_step,
            "elapsed_seconds": round(time.time() - self.start_time, 2),
        }
        self.history.append(entry)
