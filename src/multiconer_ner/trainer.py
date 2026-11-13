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
import math
import random
import shutil
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from .config import Config, DataConfig, TrainConfig, LogConfig
from .evaluate import evaluate_predictions
from .model import (
    COARSE_LABEL2ID,
    FINE_TO_COARSE,
    MultilingualNER,
    fine_to_coarse_labels,
)


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

        # --- Stage A3: Noise-robust training config ---
        self.noise_training = bool(self.data_cfg.noise_training)
        self.noise_rate = self.data_cfg.noise_rate
        self.lambda_coarse = self.data_cfg.noise_lambda_coarse
        self.lambda_context = self.data_cfg.noise_lambda_context
        self.lambda_kl = self.data_cfg.noise_lambda_kl
        self.coarse_head = self.model.coarse_head is not None

        # Pre-compute fine->coarse label mapping tensor for fast batch conversion.
        # Index = fine label ID, value = coarse label ID.
        self._fine_to_coarse_tensor: Optional[torch.Tensor] = None
        if self.coarse_head and self.id2label:
            mapping = []
            for fine_id in sorted(self.id2label.keys()):
                fine_label = self.id2label[fine_id]
                coarse_label = FINE_TO_COARSE.get(fine_label, "O")
                mapping.append(COARSE_LABEL2ID.get(coarse_label, 0))
            self._fine_to_coarse_tensor = torch.tensor(
                mapping, dtype=torch.long, device=self.device
            )

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

        if self.train_cfg.resume_from:
            self.resume(self.train_cfg.resume_from)

    def _autocast(self):
        """bf16 autocast on CUDA when --amp is set (no GradScaler needed)."""
        enabled = bool(self.train_cfg.fp16) and str(self.device).startswith("cuda")
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled)

    def _move_batch(self, batch: dict) -> dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _prepare_a3_batch(self, batch: dict) -> dict:
        """Add coarse_labels and no_context_mask to a batch for A3 training."""
        if not self.noise_training:
            return batch

        batch = dict(batch)  # shallow copy

        # Convert fine labels -> coarse labels for the coarse head.
        if self._fine_to_coarse_tensor is not None and "labels" in batch:
            fine_labels = batch["labels"]
            coarse_labels = self._fine_to_coarse_tensor[fine_labels.view(-1)].view_as(fine_labels)
            coarse_labels = torch.where(
                fine_labels == -100,
                torch.tensor(-100, device=self.device),
                coarse_labels,
            )
            batch["coarse_labels"] = coarse_labels

        # no_context_mask: True for real sentence words (not context, not padding/special).
        if "context_mask" in batch and "labels" in batch:
            no_context_mask = (~batch["context_mask"].bool()) & (batch["labels"] != -100)
            batch["no_context_mask"] = no_context_mask

        return batch

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc="train", leave=False)
        for batch in pbar:
            batch = self._move_batch(batch)
            batch = self._prepare_a3_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)

            with self._autocast():
                # --- Clean forward pass ---
                out = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    coarse_labels=batch.get("coarse_labels"),
                    no_context_mask=batch.get("no_context_mask"),
                )
                loss = out["loss"]

                # --- Stage A3: Noisy forward pass + KL divergence ---
                if self.noise_training and "noisy_input_ids" in batch:
                    noisy_batch = {
                        "input_ids": batch["noisy_input_ids"],
                        "attention_mask": batch["noisy_attention_mask"],
                        "labels": batch["noisy_labels"],
                        "coarse_labels": batch.get("noisy_coarse_labels"),
                        "no_context_mask": batch.get("noisy_no_context_mask"),
                    }
                    noisy_batch = self._prepare_a3_batch(noisy_batch)
                    out_noisy = self.model(
                        input_ids=noisy_batch["input_ids"],
                        attention_mask=noisy_batch["attention_mask"],
                        labels=noisy_batch["labels"],
                        coarse_labels=noisy_batch.get("coarse_labels"),
                        no_context_mask=noisy_batch.get("no_context_mask"),
                    )
                    loss_noisy = out_noisy["loss"]

                    # KL divergence: make predictions invariant to noise.
                    # Compute on valid (non -100) positions only.
                    clean_logits = out["logits"].float()
                    noisy_logits = out_noisy["logits"].float()
                    mask = batch["attention_mask"].view(-1).bool()
                    valid_labels = batch["labels"].view(-1) != -100
                    valid = mask & valid_labels

                    if valid.any():
                        clean_probs = F.softmax(
                            clean_logits.view(-1, self.model.num_labels)[valid], dim=-1
                        )
                        noisy_probs = F.softmax(
                            noisy_logits.view(-1, self.model.num_labels)[valid], dim=-1
                        )
                        kl_loss = F.kl_div(
                            noisy_probs.log(), clean_probs, reduction="batchmean"
                        )
                    else:
                        kl_loss = torch.tensor(0.0, device=self.device)

                    # Combine losses.
                    loss = (
                        loss
                        + self.lambda_kl * kl_loss
                        + self.lambda_coarse
                        * out.get("loss_coarse", torch.tensor(0.0, device=self.device))
                        + self.lambda_context
                        * out.get("loss_no_context", torch.tensor(0.0, device=self.device))
                    )
                else:
                    kl_loss = torch.tensor(0.0, device=self.device)

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
                if self.noise_training:
                    self.writer.add_scalar(
                        "train/kl_loss", kl_loss.item(), self.global_step
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
            with self._autocast():
                out = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
            if out["loss"] is not None:
                total_loss += out["loss"].item()
                n_batches += 1

            labels = batch["labels"]
            # Reuse the logits from the forward pass (no second forward).
            preds = self.model.decode(out["logits"], labels != -100)

            for i in range(labels.shape[0]):
                lab = labels[i].tolist()
                t_seq = [self.id2label[l] for l in lab if l != -100]
                true_tags.append(t_seq)
                pred_tags.append(preds[i])

        avg_loss = total_loss / max(n_batches, 1)
        metrics = evaluate_predictions(true_tags, pred_tags)
        metrics["loss"] = avg_loss
        metrics["num_true_tokens"] = sum(len(t) for t in true_tags)
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

    # ------------------------------------------------------------------ #
    # Checkpointing / resume
    # ------------------------------------------------------------------ #
    def _loader_generators(self) -> List[torch.Generator]:
        """torch.Generators that drive the train-loader shuffle/sampling order."""
        gens: List[torch.Generator] = []
        for g in (
            getattr(self.train_loader, "generator", None),
            getattr(getattr(self.train_loader, "sampler", None), "generator", None),
        ):
            if isinstance(g, torch.Generator) and all(g is not h for h in gens):
                gens.append(g)
        return gens

    def _save_trainer_state(self, ckpt_dir: Path) -> None:
        import numpy as np

        state = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_dev_f1": self.best_dev_f1,
            "best_dev_step": self.best_dev_step,
            "best_dev_epoch": self.best_dev_epoch,
            "patience_counter": self.patience_counter,
            "history": self.history,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "loader_generators": [g.get_state() for g in self._loader_generators()],
        }
        torch.save(state, ckpt_dir / "trainer_state.pt")

    @staticmethod
    def _best_f1_in(ckpt_dir: Path) -> Optional[float]:
        try:
            st = json.loads((ckpt_dir / "training_state.json").read_text(encoding="utf-8"))
            return float(st["best_dev_f1"])
        except Exception:
            return None

    def resume(self, ckpt_dir: str | Path) -> None:
        """Restore weights (+ optimizer/scheduler/RNG if saved) from a checkpoint.

        * epochN checkpoint with trainer_state.pt -> exact resume at epoch N+1.
        * any other checkpoint (old epochN, best, best_model) -> weights-only:
          fresh AdamW moments, LR schedule fast-forwarded, training continues
          from the start of the first epoch that was not fully completed.
        """
        import numpy as np

        ckpt_dir = Path(ckpt_dir).resolve()
        if not ckpt_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

        # --- sanity checks: same backbone / CRF / label space --------------
        mcfg = json.loads((ckpt_dir / "model_config.json").read_text(encoding="utf-8"))
        if mcfg["model_name"] != self.config.model.model_name:
            raise ValueError(
                f"Checkpoint was trained with --model {mcfg['model_name']} but this run "
                f"uses --model {self.config.model.model_name}. Pass the same --model."
            )
        if bool(mcfg.get("use_crf")) != bool(self.model.use_crf):
            raise ValueError(
                f"Checkpoint use_crf={mcfg.get('use_crf')} but this run use_crf={self.model.use_crf}."
            )
        saved_l2i = json.loads((ckpt_dir / "label2id.json").read_text(encoding="utf-8"))
        if saved_l2i != self.model.label2id:
            raise ValueError(
                "Label set differs from the checkpoint (different --language/--multilingual?)."
            )

        self.model.load_weights(ckpt_dir)
        self.model.to(self.device)

        steps_per_epoch = len(self.train_loader)
        full_state = ckpt_dir / "trainer_state.pt"
        if full_state.exists():
            st = torch.load(full_state, map_location="cpu", weights_only=False)
            self.optimizer.load_state_dict(st["optimizer"])
            # LambdaLR: restores the step counter; the lambda itself comes from
            # this run, so passing a larger --epochs extends the schedule.
            self.scheduler.load_state_dict(st["scheduler"])
            self.epoch = int(st["epoch"])
            self.global_step = int(st["global_step"])
            self.best_dev_f1 = float(st["best_dev_f1"])
            self.best_dev_step = int(st["best_dev_step"])
            self.best_dev_epoch = int(st["best_dev_epoch"])
            self.patience_counter = int(st["patience_counter"])
            self.history = list(st["history"])
            torch.set_rng_state(st["torch_rng"])
            if st.get("cuda_rng") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(st["cuda_rng"])
            random.setstate(st["python_rng"])
            np.random.set_state(st["numpy_rng"])
            for g, s in zip(self._loader_generators(), st.get("loader_generators", [])):
                g.set_state(s)
            mode = "FULL (weights + optimizer + scheduler + RNG)"
        else:
            st = json.loads((ckpt_dir / "training_state.json").read_text(encoding="utf-8"))
            completed = int(st["global_step"]) // steps_per_epoch
            self.epoch = completed
            self.global_step = completed * steps_per_epoch
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(self.global_step):
                    self.scheduler.step()
            self.best_dev_f1 = float(st.get("best_dev_f1", -1.0))
            self.best_dev_step = int(st.get("best_dev_step", -1))
            self.best_dev_epoch = int(st.get("best_dev_epoch", -1))
            self.patience_counter = 0
            self.history = list(st.get("history", []))[:completed]
            mode = "WEIGHTS-ONLY (no trainer_state.pt; optimizer moments start fresh)"

        # --- make sure output_dir/best really holds the restored best model ---
        dst_best = self.output_dir / "best"
        if self.best_dev_f1 >= 0:
            def same(d: Path) -> bool:
                f = self._best_f1_in(d)
                return f is not None and math.isclose(f, self.best_dev_f1, abs_tol=1e-9)

            if not same(dst_best):
                src_best = ckpt_dir.parent / "best"
                if src_best.resolve() != dst_best.resolve() and same(src_best):
                    if dst_best.exists():
                        shutil.rmtree(dst_best)
                    shutil.copytree(src_best, dst_best)
                elif self.best_dev_step == self.global_step:
                    # The checkpoint we resume from IS the best model so far.
                    self._save_checkpoint(tag="best")
                else:
                    print(
                        f"[resume] WARNING: no matching 'best' checkpoint for dev F1 "
                        f"{self.best_dev_f1:.4f}; best-model tracking restarts from this run."
                    )
                    self.best_dev_f1 = -1.0

        print(
            f"[resume] {mode}\n"
            f"[resume] from {ckpt_dir}: completed epochs={self.epoch}, "
            f"global_step={self.global_step}, best_dev_f1={self.best_dev_f1:.4f}, "
            f"lr now={self.scheduler.get_last_lr()[0]:.3e}"
        )
        if self.epoch >= self.train_cfg.num_train_epochs:
            print(
                f"[resume] WARNING: checkpoint already completed {self.epoch} epochs and "
                f"--epochs is {self.train_cfg.num_train_epochs}; nothing left to train. "
                f"Pass a larger --epochs to continue."
            )

    def _save_checkpoint(self, tag: str, full_state: bool = False) -> None:
        ckpt_dir = self.output_dir / tag
        self.model.save(ckpt_dir, self.tokenizer)
        if full_state and self.log_cfg.save_optimizer_state:
            self._save_trainer_state(ckpt_dir)
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
        for epoch in range(self.epoch, self.train_cfg.num_train_epochs):
            self.epoch = epoch + 1
            train_loss = self._train_epoch()
            print(f"Epoch {self.epoch}: train_loss={train_loss:.4f}")

            dev_metrics = self.evaluate()
            self._log_dev(dev_metrics, step=self.global_step)
            self._record_history(epoch, train_loss, dev_metrics)
            self._maybe_save_best(dev_metrics, step=self.global_step)

            if self.log_cfg.save_every_epoch:
                self._save_checkpoint(tag=f"epoch{self.epoch}", full_state=True)

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
