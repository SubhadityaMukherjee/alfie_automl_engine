import logging
import time
from typing import Any

import lightning as L
import torch
from torch import nn, optim
from tqdm import tqdm
from pathlib import Path
import optuna

from app.vision_automl.ml_engine.datamodule import ClassificationData
from app.vision_automl.ml_engine.model import ClassificationModel

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s"))
    logger.addHandler(handler)


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
    """Minimal trainer using Lightning Fabric for classification tasks."""

    def __init__(
        self,
        datamodule: Any,
        model_class: type[nn.Module],
        model_kwargs: dict[str, Any] = {},
        optimizer_class: type[optim.Optimizer] = optim.AdamW,
        optimizer_kwargs: dict[str, Any] = {},
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        lr: float = 0.001,
        epochs: int = 1,
        time_limit: float | None = None,
        device: str = "auto",
        callbacks: list[Any] = [],
        input_dtype: torch.dtype = torch.float32,
        target_dtype: torch.dtype = torch.long,
    ) -> None:
        self.datamodule: Any = datamodule
        self.model_class: type[nn.Module] = model_class
        self.model_kwargs: dict[str, Any] = model_kwargs
        self.optimizer_class: type[optim.Optimizer] = optimizer_class
        self.optimizer_kwargs: dict[str, Any] = optimizer_kwargs or {"lr": lr}
        self.loss_fn: nn.Module = loss_fn
        self.epochs: int = epochs
        self.time_limit: float | None = time_limit
        self.device: str = device
        self.callbacks: list[Any] = callbacks
        self.input_dtype: torch.dtype = input_dtype
        self.target_dtype: torch.dtype = target_dtype

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
        self.model, self.optimizer, self.train_loader, self.val_loader = (
            self.fabric.setup(self.model, self.optimizer, train_loader, val_loader)
        )
        self.test_loader: Any = self.datamodule.test_dataloader()
        logger.info("Model and optimizer setup complete.")

    def _move_batch(self, batch: Any) -> dict[str, torch.Tensor]:
        """Move batch tensors to device and standardize batch dict."""
        pixel_values: torch.Tensor
        labels: torch.Tensor

        if isinstance(batch, dict):
            pixel_values = batch["pixel_values"].to(
                self.fabric.device, dtype=self.input_dtype
            )
            labels = batch["labels"].to(self.fabric.device, dtype=self.target_dtype)
        else:
            imgs, batch_labels = batch
            pixel_values = imgs.to(self.fabric.device, dtype=self.input_dtype)
            labels = batch_labels.to(self.fabric.device, dtype=self.target_dtype)

        moved_batch: dict[str, torch.Tensor] = {
            "pixel_values": pixel_values,
            "labels": labels,
        }
        return moved_batch

    def _check_time_limit(self, start_time: float) -> bool:
        """Return True if configured time limit has been exceeded."""
        elapsed: float = time.time() - start_time
        if self.time_limit and elapsed > self.time_limit:
            logger.warning(f"Time limit reached ({elapsed:.2f}s). Stopping training.")
            return True
        return False

    def train_epoch(self, epoch: int, start_time: float) -> float:
        """Train for a single epoch and return average training loss."""
        self.model.train()
        running_loss: float = 0.0
        batch_count: int = len(self.train_loader)

        for batch in tqdm(
            self.train_loader, desc=f"Epoch {epoch+1} Training", leave=False
        ):
            if self._check_time_limit(start_time):
                return running_loss / max(1, batch_count)

            moved: dict[str, torch.Tensor] = self._move_batch(batch)
            self.optimizer.zero_grad()
            outputs: torch.Tensor = self.model(moved["pixel_values"])
            loss: torch.Tensor = self.loss_fn(outputs, moved["labels"])
            self.fabric.backward(loss)
            self.optimizer.step()
            running_loss += loss.item()

        avg_loss: float = running_loss / batch_count
        logger.info(f"Epoch {epoch+1} Training Loss: {avg_loss:.4f}")
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

                moved: dict[str, torch.Tensor] = self._move_batch(batch)
                outputs: torch.Tensor = self.model(moved["pixel_values"])
                loss: torch.Tensor = self.loss_fn(outputs, moved["labels"])
                val_loss += loss.item()

                preds: torch.Tensor = outputs.argmax(dim=1)
                correct += (preds == moved["labels"]).sum().item()
                total += moved["labels"].size(0)

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
                moved: dict[str, torch.Tensor] = self._move_batch(batch)
                outputs: torch.Tensor = self.model(moved["pixel_values"])
                loss: torch.Tensor = self.loss_fn(outputs, moved["labels"])
                test_loss += loss.item()

                preds: torch.Tensor = outputs.argmax(dim=1)
                correct += (preds == moved["labels"]).sum().item()
                total += moved["labels"].size(0)

        avg_loss: float = test_loss / len(self.test_loader)
        accuracy: float = correct / total
        logger.info(f"Test Results - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy

    def fit(self, trial: optuna.Trial | None = None) -> tuple[float, float]:
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

def optuna_objective(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    images_dir: Path,
    filename_column: str,
    label_column: str,
):
    # -------------------------
    # Search space
    # -------------------------
    model_id = trial.suggest_categorical(
        "model_id",
        ["google/efficientnet-b0", "google/vit-base-patch16-224"],
    )

    lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    # -------------------------
    # Data
    # -------------------------
    datamodule = ClassificationData(
        csv_file=str(csv_path),
        root_dir=str(images_dir),
        img_col=filename_column,
        label_col=label_column,
        batch_size=batch_size,
        hf_model_id=model_id,
    )

    # -------------------------
    # Trainer
    # -------------------------
    trainer = FabricTrainer(
        datamodule=datamodule,
        model_class=ClassificationModel,
        model_kwargs={
            "model_id": model_id,
            "num_classes": datamodule.num_classes,
            "id2label": datamodule.id2label,
            "label2id": datamodule.label2id,
        },
        optimizer_class=optim.AdamW,
        optimizer_kwargs={
            "lr": lr,
            "weight_decay": weight_decay,
        },
        loss_fn=nn.CrossEntropyLoss(),
        epochs=20,  # short by design (Optuna!)
        callbacks=[EarlyStopping(patience=3)],
    )

    # -------------------------
    # Train
    # -------------------------
    test_loss, test_acc = trainer.fit(trial=trial)

    # We optimize validation loss proxy via test loss
    return test_loss

def run_optuna_search(
    *,
    csv_path: Path,
    images_dir: Path,
    filename_column: str,
    label_column: str,
    n_trials: int = 20,
    timeout: int | None = None,
):
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=3,
        n_warmup_steps=2,
    )

    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(
        lambda trial: optuna_objective(
            trial,
            csv_path=csv_path,
            images_dir=images_dir,
            filename_column=filename_column,
            label_column=label_column,
        ),
        n_trials=n_trials,
        timeout=timeout,
    )

    return {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
