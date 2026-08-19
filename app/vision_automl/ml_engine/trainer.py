"""Training loop and Optuna search for the vision ML engine."""

import functools
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import lightning as L
import optuna
import torch
from torch import nn, optim
from tqdm import tqdm

from app.core.exceptions import AutoMLConfigError, AutoMLTrainingError
from app.vision_automl.ml_engine.configs import load_task_config

# Backward-compat import alias kept for external code that references
# ClassificationData / ClassificationModel directly.
from app.vision_automl.ml_engine.model import ImageClassificationModel  # noqa: F401

logger = logging.getLogger(__name__)

# Keys whose tensors should be treated as target/label dtype
_TARGET_KEYS: frozenset[str] = frozenset({"labels", "start_positions", "end_positions"})


class EarlyStopping:
    """Simple early stopping callback based on monitored metric."""

    def __init__(
        self, monitor: str = "val_loss", patience: int = 3, min_delta: float = 0.0
    ) -> None:
        self.monitor: str = monitor
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.best: float = float("inf")
        self.counter: int = 0

    def on_epoch_end(
        self, trainer: "FabricTrainer", epoch: int, logs: dict[str, float]
    ) -> None:
        """Update state after epoch; may signal stopping on trainer."""
        current: float | None = logs.get(self.monitor)
        if current is None:
            logger.warning(
                f"Metric '{self.monitor}' not found in logs. Skipping early stopping check."
            )
            return

        if current < self.best - self.min_delta:
            self.best = current
            self.counter = 0
            logger.info(f"New best {self.monitor}: {self.best:.4f}")
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                logger.info("Early stopping triggered!")
                trainer.epochs = epoch + 1


class FabricTrainer:
    """Minimal trainer using Lightning Fabric.

    Supports both:
    - Classification tasks where the model returns logits and the trainer
      computes the loss via ``loss_fn`` (``model_computes_loss=False``).
    - Generative / structured-prediction tasks where the model computes
      its own loss internally and returns a scalar tensor
      (``model_computes_loss=True``).
    """

    def __init__(
        self,
        datamodule: Any,
        model_class: type[nn.Module],
        model_kwargs: dict[str, Any] | None = None,
        optimizer_class: type[optim.Optimizer] = optim.AdamW,
        optimizer_kwargs: dict[str, Any] | None = None,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        lr: float = 0.001,
        epochs: int = 1,
        time_limit: float | None = None,
        device: str | int = "auto",
        num_threads: int | None = None,
        callbacks: list[Any] | None = None,
        input_dtype: torch.dtype = torch.float32,
        target_dtype: torch.dtype = torch.long,
        model_computes_loss: bool = False,
    ) -> None:
        self.datamodule: Any = datamodule
        self.model_class: type[nn.Module] = model_class
        self.model_kwargs: dict[str, Any] = model_kwargs or {}
        self.optimizer_class: type[optim.Optimizer] = optimizer_class
        self.optimizer_kwargs: dict[str, Any] = optimizer_kwargs or {"lr": lr}
        self.loss_fn: nn.Module = loss_fn
        self.epochs: int = epochs
        self.time_limit: float | None = time_limit
        self.device: str | int = device
        self.callbacks: list[Any] = callbacks or []
        self.input_dtype: torch.dtype = input_dtype
        self.target_dtype: torch.dtype = target_dtype
        self.model_computes_loss: bool = model_computes_loss

        if num_threads is not None:
            torch.set_num_threads(num_threads)

        self.fabric: L.Fabric = L.Fabric(devices=self.device)
        self._setup_model_optimizer()

    def _setup_model_optimizer(self) -> None:
        """Instantiate model and optimizer and prepare loaders with Fabric."""
        logger.info("Setting up model and optimizer.")
        self.model: nn.Module = self.model_class(**self.model_kwargs)
        self.optimizer: optim.Optimizer = self.optimizer_class(
            self.model.parameters(), **self.optimizer_kwargs
        )

        train_loader: Any = self.datamodule.train_dataloader()
        val_loader: Any = self.datamodule.val_dataloader()
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
        ) = self.fabric.setup(self.model, self.optimizer, train_loader, val_loader)
        self.test_loader: Any = self.datamodule.test_dataloader()
        logger.info("Model and optimizer setup complete.")

    def _move_batch(self, batch: Any) -> dict[str, Any]:
        """Move batch to the Fabric device.

        Handles arbitrary dict batches (all modalities) and legacy
        ``(images, labels)`` tuple batches.  Non-tensor values (e.g. list
        of annotation dicts for object detection) are passed through as-is.
        Integer tensors (``input_ids``, etc.) are moved without dtype coercion.
        """
        if isinstance(batch, dict):
            moved: dict[str, Any] = {}
            for k, v in batch.items():
                if not isinstance(v, torch.Tensor):
                    moved[k] = v  # keep non-tensors (e.g. list of dicts)
                elif k in _TARGET_KEYS:
                    moved[k] = v.to(self.fabric.device, dtype=self.target_dtype)
                elif v.dtype.is_floating_point:
                    moved[k] = v.to(self.fabric.device, dtype=self.input_dtype)
                else:
                    # int/long tensors (input_ids, etc.) — preserve dtype
                    moved[k] = v.to(self.fabric.device)
            return moved
        else:
            imgs, batch_labels = batch
            return {
                "pixel_values": imgs.to(self.fabric.device, dtype=self.input_dtype),
                "labels": batch_labels.to(self.fabric.device, dtype=self.target_dtype),
            }

    def _check_time_limit(self, start_time: float) -> bool:
        """Return True if configured time limit has been exceeded."""
        elapsed: float = time.time() - start_time
        if self.time_limit and elapsed > self.time_limit:
            logger.warning(f"Time limit reached ({elapsed:.2f}s). Stopping training.")
            return True
        return False

    def _compute_loss_and_logits(
        self, moved: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run forward pass, return (loss, logits_or_None)."""
        if self.model_computes_loss:
            outputs = self.model(**moved)
            loss = outputs if isinstance(outputs, torch.Tensor) else outputs.loss
            return loss, None
        else:
            labels = moved.pop("labels")
            outputs = self.model(**moved)
            loss = self.loss_fn(outputs, labels)
            return loss, outputs

    def train_epoch(self, epoch: int, start_time: float) -> float:
        """Train for a single epoch and return average training loss."""
        self.model.train()
        running_loss: float = 0.0
        batch_count: int = len(self.train_loader)

        for batch in tqdm(
            self.train_loader, desc=f"Epoch {epoch + 1} Training", leave=False
        ):
            if self._check_time_limit(start_time):
                return running_loss / max(1, batch_count)

            moved = self._move_batch(batch)
            self.optimizer.zero_grad()
            loss, _ = self._compute_loss_and_logits(moved)
            self.fabric.backward(loss)
            self.optimizer.step()
            running_loss += loss.item()

        avg_loss: float = running_loss / batch_count
        logger.info(f"Epoch {epoch + 1} Training Loss: {avg_loss:.4f}")
        return avg_loss

    def validate(self, start_time: float) -> tuple[float, float]:
        """Evaluate on validation set; return (avg_loss, accuracy)."""
        self.model.eval()
        val_loss: float = 0.0
        correct: int = 0
        total: int = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                if self._check_time_limit(start_time):
                    break

                moved = self._move_batch(batch)

                if self.model_computes_loss:
                    outputs = self.model(**moved)
                    loss = (
                        outputs if isinstance(outputs, torch.Tensor) else outputs.loss
                    )
                    val_loss += loss.item()
                else:
                    labels = moved.pop("labels")
                    outputs = self.model(**moved)
                    loss = self.loss_fn(outputs, labels)
                    val_loss += loss.item()
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

        avg_loss: float = val_loss / max(1, len(self.val_loader))
        accuracy: float = correct / max(1, total)
        logger.info(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy

    def test(self) -> tuple[float, float]:
        """Evaluate on test set; return (avg_loss, accuracy)."""
        self.model.eval()
        test_loss: float = 0.0
        correct: int = 0
        total: int = 0

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                moved = self._move_batch(batch)

                if self.model_computes_loss:
                    outputs = self.model(**moved)
                    loss = (
                        outputs if isinstance(outputs, torch.Tensor) else outputs.loss
                    )
                    test_loss += loss.item()
                else:
                    labels = moved.pop("labels")
                    outputs = self.model(**moved)
                    loss = self.loss_fn(outputs, labels)
                    test_loss += loss.item()
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

        avg_loss: float = test_loss / len(self.test_loader)
        accuracy: float = correct / max(1, total)
        logger.info(f"Test Results - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy

    def fit(self, trial: optuna.Trial | None = None) -> tuple[float, float]:
        """Run the full train/validate loop, then evaluate on the test set.

        Each epoch reports validation loss back to the Optuna trial (so bad
        trials get pruned early), gives callbacks a chance to stop training,
        and checks the wall-clock time limit. Returns the test loss and
        accuracy of the final model.
        """
        logger.info("Starting training loop.")
        start_time: float = time.time()

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch, start_time)
            val_loss, val_acc = self.validate(start_time)

            if trial is not None:
                trial.report(val_loss, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            logs = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }

            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, logs)

            if self._check_time_limit(start_time):
                break

        return self.test()


# ---------------------------------------------------------------------------
# run_optuna_search
# ---------------------------------------------------------------------------
def run_optuna_search(
    *,
    task_type: str = "image_classification",
    csv_path: Path,
    images_dir: Path | None = None,
    filename_column: str = "filename",
    label_column: str = "label",
    n_trials: int = 3,
    timeout: int | None = None,
    model_size: str = "small",
    workdir: Path,
    num_gpus: str | int = "auto",
    num_cpus: str | int = "auto",
    **extra_kwargs,
) -> dict:
    """Run an Optuna hyperparameter search for the given task type.

    Dispatches to the appropriate per-task objective via ``OBJECTIVE_REGISTRY``.
    ``extra_kwargs`` are forwarded to the objective (e.g. ``text_column`` for
    text tasks).

    Raises:
        ValueError: If ``task_type`` is not in ``OBJECTIVE_REGISTRY``.
    """
    from app.vision_automl.hpo.optuna_objectives import OBJECTIVE_REGISTRY

    if task_type not in OBJECTIVE_REGISTRY:
        raise AutoMLConfigError(
            f"Unknown task type '{task_type}'. Supported: {sorted(OBJECTIVE_REGISTRY)}"
        )

    config = load_task_config(task_type)
    objective_fn = OBJECTIVE_REGISTRY[task_type]

    run_dir = workdir / "optuna"
    run_dir.mkdir(exist_ok=True)

    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=10,
        reduction_factor=3,
        min_early_stopping_rate=0,
    )
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    timeout_per_trial = timeout / max(n_trials, 1) if timeout else None

    # Build keyword arguments for the objective
    objective_kwargs: dict = {
        "csv_path": csv_path,
        "images_dir": images_dir,
        "filename_column": filename_column,
        "label_column": label_column,
        "model_size": model_size,
        "timeout_per_trial": timeout_per_trial,
        "config": config,
        "num_gpus": num_gpus,
        "num_cpus": num_cpus,
        "workdir": workdir,
        "task_type": task_type,
        **extra_kwargs,
    }

    study.optimize(
        functools.partial(objective_fn, **objective_kwargs),
        n_trials=n_trials,
        timeout=timeout,
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise AutoMLTrainingError(
            f"All {len(study.trials)} Optuna trial(s) failed or were pruned. "
            "Check your dataset, model IDs, and time budget."
        )

    best_trial_dir = run_dir / f"trial_{study.best_trial.number}"
    _copy_best_trial_artifacts(best_trial_dir, workdir)

    return {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "model_dir": best_trial_dir,
    }


def _copy_best_trial_artifacts(best_trial_dir: Path, workdir: Path) -> None:
    """Copy the best trial's feature_mapping.json + model.pt into workdir/model/.

    ``serialize_and_zip_model`` zips ``workdir/model/``, so anything we want in
    the uploaded zip needs to land there. Missing sources are skipped silently
    (per-trial saves are best-effort and may not have run).
    """
    model_dir = Path(workdir) / "model"
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning("Failed to create model dir %s: %s", model_dir, e)
        return

    if not best_trial_dir.exists():
        logger.warning("Best trial artifact dir does not exist: %s", best_trial_dir)
        return

    for fname in ("feature_mapping.json", "model.pt"):
        src = best_trial_dir / fname
        if not src.exists():
            logger.debug("Best trial did not produce %s; skipping", fname)
            continue
        try:
            shutil.copy2(src, model_dir / fname)
            logger.debug("Copied %s -> %s", src, model_dir / fname)
        except Exception as e:
            logger.warning("Failed to copy %s into model dir: %s", src, e)
