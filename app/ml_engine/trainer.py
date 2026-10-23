"""Training loops for the consolidated ML engine.

Contains the generic engine's Lightning Fabric training loop and Optuna
search (``FabricTrainer`` / ``run_optuna_search``) plus the AutoGluon-based
tabular trainer (``AutoGluonTrainer``).
"""

import logging
import math
import shutil
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import lightning as L
import numpy as np
import optuna
import pandas as pd
import torch
from autogluon.tabular import TabularDataset, TabularPredictor
from dotenv import find_dotenv, load_dotenv
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    explained_variance_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from torch import nn, optim
from tqdm import tqdm

from app.core.config import get_settings
from app.core.exceptions import (
    AutoMLConfigError,
    AutoMLDataError,
    AutoMLLeaderboardError,
    AutoMLTrainingError,
)
from app.ml_engine.configs import load_task_config

# Backward-compat import alias kept for external code that references
# ClassificationData / ClassificationModel directly.
from app.ml_engine.model import ImageClassificationModel  # noqa: F401

logger = logging.getLogger(__name__)

# Keys whose tensors should be treated as target/label dtype
_TARGET_KEYS: frozenset[str] = frozenset({"labels", "start_positions", "end_positions"})


def _resolve_precision(precision: str) -> str:
    """Resolve the Fabric precision string, honouring "auto".

    "auto" picks bf16-mixed on CUDA hardware that supports it, 16-mixed on
    older CUDA GPUs, and full precision everywhere else (CPU / MPS).
    """
    if precision != "auto":
        return precision
    if torch.cuda.is_available():
        return "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
    return "32-true"


def _safe_metric(name: str, fn: Callable[[], float]) -> float | None:
    """Evaluate one metric function; return None when it is not computable.

    Degenerate validation batches (only one class present, a class missing
    from the predictions) make some metrics undefined — skip them instead of
    reporting misleading values.
    """
    try:
        with warnings.catch_warnings():
            # Degenerate batches are expected on tiny validation sets; the
            # resulting sklearn warnings are noise (we skip the metric).
            warnings.simplefilter("ignore")
            value: float = float(fn())
    except Exception as e:
        logger.debug("Metric %s not computable, skipping: %s", name, e)
        return None
    if not math.isfinite(value):
        logger.debug("Metric %s is non-finite, skipping", name)
        return None
    return value


def _classification_metrics_from_arrays(
    y_true: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    """General-purpose classification metrics on probability arrays.

    ``y_proba`` is a ``[N, num_classes]`` matrix of class probabilities and
    ``y_true`` integer class ids aligned with its column order. Shared by the
    generic-engine eval loop and the AutoGluon tabular trainer. Metrics that
    are undefined on the observed labels (e.g. ROC-AUC when a class is
    missing) are omitted rather than reported as zero.
    """
    if y_proba.shape[0] == 0:
        return {}
    y_pred = y_proba.argmax(axis=1)
    num_classes = int(y_proba.shape[1])

    metrics: dict[str, float] = {"accuracy": float((y_pred == y_true).mean())}

    candidates: dict[str, Callable[[], float]] = {
        "balanced_accuracy": lambda: balanced_accuracy_score(y_true, y_pred),
        "precision_macro": lambda: precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "precision_weighted": lambda: precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_macro": lambda: recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_weighted": lambda: recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_macro": lambda: f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": lambda: f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "cohen_kappa": lambda: cohen_kappa_score(y_true, y_pred),
        "matthews_corrcoef": lambda: matthews_corrcoef(y_true, y_pred),
    }
    for name, fn in candidates.items():
        if (value := _safe_metric(name, fn)) is not None:
            metrics[name] = value

    # Top-k accuracy is trivially 1.0 unless there are more classes than k.
    for k in (3, 5):
        if num_classes > k:
            topk = np.argsort(-y_proba, axis=1, kind="stable")[:, :k]
            metrics[f"top{k}_accuracy"] = float(
                (topk == y_true[:, None]).any(axis=1).mean()
            )

    # ROC-AUC: positive-class score for binary, macro OvR for multiclass.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if num_classes == 2:
                auc_value: float = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                auc_value = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
        if math.isfinite(auc_value):
            metrics["roc_auc"] = auc_value
    except Exception as e:
        logger.debug("ROC-AUC not computable on this validation set: %s", e)

    return metrics


def _compute_classification_metrics(
    probs: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    """Torch-tensor wrapper: softmax'd logits + label ids -> metrics dict."""
    return _classification_metrics_from_arrays(
        labels.detach().cpu().long().numpy(),
        probs.detach().cpu().float().numpy(),
    )


def _compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """General-purpose regression metrics on the held-out test set."""
    candidates: dict[str, Callable[[], float]] = {
        "r2": lambda: r2_score(y_true, y_pred),
        "mae": lambda: mean_absolute_error(y_true, y_pred),
        "rmse": lambda: math.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": lambda: mean_absolute_percentage_error(y_true, y_pred),
        "median_absolute_error": lambda: median_absolute_error(y_true, y_pred),
        "explained_variance": lambda: explained_variance_score(y_true, y_pred),
    }
    metrics: dict[str, float] = {}
    for name, fn in candidates.items():
        if (value := _safe_metric(name, fn)) is not None:
            metrics[name] = value
    return metrics


def _compute_tabular_test_metrics(
    predictor: Any, test_df: pd.DataFrame, target_column: str
) -> dict[str, float]:
    """Compute a metrics suite on the tabular test set (best-effort).

    Dispatches on the AutoGluon predictor's problem type: classification
    (binary/multiclass) gets the shared classification suite, regression gets
    ``_compute_regression_metrics``. Anything else (quantile, unknown) and any
    error degrade to an empty dict — extra metrics must never fail training.
    """
    try:
        y_true = test_df[target_column]
        problem_type = str(getattr(predictor, "problem_type", "") or "")

        if problem_type in ("binary", "multiclass"):
            proba = predictor.predict_proba(test_df)
            if isinstance(proba, pd.Series):
                proba = proba.to_frame()
            if not isinstance(proba, pd.DataFrame) or proba.empty:
                logger.warning("predict_proba returned no columns; skipping metrics")
                return {}

            if proba.shape[1] == 1:
                # Single-column output: probability of the positive class.
                positive_class = getattr(predictor, "positive_class", None)
                negative_class = getattr(predictor, "negative_class", None)
                names = (
                    str(negative_class) if negative_class is not None else "0",
                    str(positive_class) if positive_class is not None else "1",
                )
                positive = proba.iloc[:, 0].astype(float)
                proba = pd.DataFrame({names[0]: 1.0 - positive, names[1]: positive})
            else:
                class_labels = getattr(predictor, "class_labels", None)
                if class_labels is not None and len(class_labels) == proba.shape[1]:
                    proba.columns = [str(c) for c in class_labels]
                # Binary: the shared helper expects the positive class last.
                positive_class = getattr(predictor, "positive_class", None)
                if proba.shape[1] == 2 and positive_class is not None:
                    cols = [str(c) for c in proba.columns]
                    if cols[-1] != str(positive_class):
                        proba = proba.iloc[:, [1, 0]]

            # Map labels to column indices so y_true aligns with the column
            # order of y_proba regardless of how classes were sorted.
            columns = [str(c) for c in proba.columns]
            codes = pd.Categorical(y_true.astype(str), categories=columns).codes
            keep = codes >= 0
            if not keep.all():
                logger.warning(
                    "%d test row(s) have labels outside the predictor's "
                    "classes; excluded from test metrics",
                    int((~keep).sum()),
                )
            return _classification_metrics_from_arrays(
                codes[keep].astype(np.int64),
                proba.to_numpy(dtype=float)[keep],
            )

        if problem_type == "regression":
            y_pred = predictor.predict(test_df)
            return _compute_regression_metrics(
                y_true.to_numpy(dtype=float),
                np.asarray(y_pred, dtype=float).ravel(),
            )

        logger.info(
            "No extra test metrics computed for problem type '%s'", problem_type
        )
        return {}
    except Exception as e:
        logger.warning("Failed to compute tabular test metrics: %s", e)
        return {}


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
        precision: str | None = None,
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

        self._configure_backend()
        self.precision: str = _resolve_precision(
            precision if precision is not None else get_settings().training_precision
        )
        logger.info("Training with precision: %s", self.precision)
        self.fabric: L.Fabric = L.Fabric(
            devices=self.device, precision=cast(Any, self.precision)
        )
        self._setup_model_optimizer()

    @staticmethod
    def _configure_backend() -> None:
        """Enable fast matmul paths (TF32 on Ampere+, cuDNN autotuning)."""
        try:
            torch.set_float32_matmul_precision("high")
        except Exception as e:  # pragma: no cover - very old torch versions
            logger.debug("Could not set float32 matmul precision: %s", e)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

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
                    moved[k] = v.to(
                        self.fabric.device, dtype=self.target_dtype, non_blocking=True
                    )
                elif v.dtype.is_floating_point:
                    moved[k] = v.to(
                        self.fabric.device, dtype=self.input_dtype, non_blocking=True
                    )
                else:
                    # int/long tensors (input_ids, etc.) — preserve dtype
                    moved[k] = v.to(self.fabric.device, non_blocking=True)
            return moved
        else:
            imgs, batch_labels = batch
            return {
                "pixel_values": imgs.to(
                    self.fabric.device, dtype=self.input_dtype, non_blocking=True
                ),
                "labels": batch_labels.to(
                    self.fabric.device, dtype=self.target_dtype, non_blocking=True
                ),
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
        """Run forward pass, return (loss, logits_or_None).

        Under mixed precision the module forward runs inside Fabric's
        autocast context, but this loss computation does not — upcast the
        logits so the loss is computed in float32.
        """
        if self.model_computes_loss:
            outputs = self.model(**moved)
            loss = outputs if isinstance(outputs, torch.Tensor) else outputs.loss
            return loss, None
        else:
            labels = moved.pop("labels")
            outputs = self.model(**moved)
            loss = self.loss_fn(outputs.float(), labels)
            return loss, outputs

    def train_epoch(
        self, epoch: int, start_time: float
    ) -> tuple[float | None, float | None]:
        """Train for a single epoch; return (average training loss, train accuracy).

        Accuracy is a running argmax match over the epoch and is only
        computed when logits and a label tensor are available
        (``model_computes_loss=False``); LM-style tasks return ``None``.

        Returns ``None`` for the loss when not a single batch was trained
        (e.g. the time limit fired immediately) — callers must not treat
        that as a real result.
        """
        self.model.train()
        # Accumulate on-device and sync once per epoch (per-batch .item()
        # would force a device synchronization every step).
        loss_sum = torch.zeros((), device=self.fabric.device)
        correct_sum = torch.zeros((), device=self.fabric.device)
        total_count = torch.zeros((), device=self.fabric.device)
        batch_count: int = 0

        for batch in tqdm(
            self.train_loader, desc=f"Epoch {epoch + 1} Training", leave=False
        ):
            if self._check_time_limit(start_time):
                break

            moved = self._move_batch(batch)
            # Labels must be referenced before _compute_loss_and_logits pops
            # them off the batch dict.
            labels = moved.get("labels")
            self.optimizer.zero_grad()
            loss, outputs = self._compute_loss_and_logits(moved)
            self.fabric.backward(loss)
            self.optimizer.step()
            loss_sum += loss.detach().float()
            batch_count += 1

            if (
                not self.model_computes_loss
                and outputs is not None
                and labels is not None
            ):
                preds = outputs.detach().argmax(dim=1)
                correct_sum += (preds == labels).sum()
                total_count += labels.size(0)

        if not batch_count:
            logger.warning(
                "Epoch %d trained no batches (time limit); skipping", epoch + 1
            )
            return None, None

        avg_loss: float = (loss_sum / batch_count).item()
        total: int = int(total_count.item())
        train_acc: float | None = int(correct_sum.item()) / total if total else None
        if train_acc is not None:
            logger.info(
                f"Epoch {epoch + 1} Training Loss: {avg_loss:.4f}, "
                f"Training Accuracy: {train_acc:.4f}"
            )
        else:
            logger.info(f"Epoch {epoch + 1} Training Loss: {avg_loss:.4f}")
        return avg_loss, train_acc

    def validate(self, start_time: float) -> dict[str, float]:
        """Evaluate on the validation set; return a metrics dict.

        Always contains ``loss``. For classification tasks
        (``model_computes_loss=False``) it also carries the general
        classification metrics from ``_compute_classification_metrics``.
        """
        self.model.eval()
        val_loss = torch.zeros((), device=self.fabric.device)
        all_probs: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        batch_count: int = 0

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
                    val_loss += loss.detach().float()
                else:
                    labels = moved.pop("labels")
                    outputs = self.model(**moved)
                    loss = self.loss_fn(outputs.float(), labels)
                    val_loss += loss.detach().float()
                    # CPU fp32 copies keep memory bounded; metrics are
                    # computed once at the end of the epoch.
                    all_probs.append(torch.softmax(outputs.float(), dim=1).cpu())
                    all_labels.append(labels.detach().cpu())
                batch_count += 1

        if not batch_count:
            # An empty validation pass must never look like a perfect 0.0
            # loss — inf loses every best-epoch comparison in fit().
            logger.warning("Validation ran no batches; reporting loss=inf")
            return {"loss": float("inf")}

        avg_loss: float = (val_loss / batch_count).item()
        metrics: dict[str, float] = {"loss": avg_loss}

        if all_probs:
            metrics.update(
                _compute_classification_metrics(
                    torch.cat(all_probs), torch.cat(all_labels)
                )
            )

        logger.info(
            "Validation - Loss: %.4f, Accuracy: %.4f, F1 (macro): %.4f",
            avg_loss,
            metrics.get("accuracy", 0.0),
            metrics.get("f1_macro", 0.0),
        )
        return metrics

    def test(self) -> dict[str, float]:
        """Evaluate on the test set; returns the same metrics dict as validate()."""
        self.model.eval()
        test_loss = torch.zeros((), device=self.fabric.device)
        all_probs: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        batch_count: int = 0

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                moved = self._move_batch(batch)

                if self.model_computes_loss:
                    outputs = self.model(**moved)
                    loss = (
                        outputs if isinstance(outputs, torch.Tensor) else outputs.loss
                    )
                    test_loss += loss.detach().float()
                else:
                    labels = moved.pop("labels")
                    outputs = self.model(**moved)
                    loss = self.loss_fn(outputs.float(), labels)
                    test_loss += loss.detach().float()
                    all_probs.append(torch.softmax(outputs.float(), dim=1).cpu())
                    all_labels.append(labels.detach().cpu())
                batch_count += 1

        if not batch_count:
            logger.warning("Test set produced no batches; skipping test metrics")
            return {}

        avg_loss: float = (test_loss / batch_count).item()
        metrics: dict[str, float] = {"loss": avg_loss}

        if all_probs:
            metrics.update(
                _compute_classification_metrics(
                    torch.cat(all_probs), torch.cat(all_labels)
                )
            )

        logger.info(
            "Test Results - Loss: %.4f, Accuracy: %.4f, F1 (macro): %.4f",
            avg_loss,
            metrics.get("accuracy", 0.0),
            metrics.get("f1_macro", 0.0),
        )
        return metrics

    def fit(self, trial: optuna.Trial | None = None) -> tuple[float, dict[str, float]]:
        """Run the full train/validate loop; return (best_val_loss, best_metrics).

        Each epoch reports validation loss back to the Optuna trial (so bad
        trials get pruned early), gives callbacks a chance to stop training,
        and checks the wall-clock time limit. The metrics dict of the epoch
        with the best validation loss is returned so hyperparameter search
        selects on validation performance (the test split is only touched
        once, by the final evaluation after the loop).

        The returned metrics dict carries the best epoch's validation suite
        (``loss`` / ``accuracy`` / ...) plus ``train_loss`` and ``train_acc``
        (both from the last epoch; ``train_acc`` is absent for LM-style
        tasks), and ``val_loss`` / ``val_acc`` / ``test_loss`` / ``test_acc``
        aliases for the AutoDW metadata upload.

        Raises ``optuna.TrialPruned`` (inside a search) or
        ``AutoMLTrainingError`` (standalone) when the time limit fired before
        a single batch was trained — an untrained model must never be
        reported as a real result.
        """
        logger.info("Starting training loop.")
        start_time: float = time.time()

        best_val_loss: float = float("inf")
        best_val_metrics: dict[str, float] = {}
        last_val: tuple[float, dict[str, float]] = (float("inf"), {})
        last_train_loss: float = 0.0
        last_train_acc: float | None = None
        trained_any_batch = False

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(epoch, start_time)
            if train_loss is None:
                # Time limit fired before this epoch did any work.
                break
            trained_any_batch = True
            last_train_loss = train_loss
            last_train_acc = train_acc
            val_metrics = self.validate(start_time)
            val_loss = val_metrics["loss"]
            last_val = (val_loss, val_metrics)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_metrics = dict(val_metrics)

            if trial is not None:
                trial.report(val_loss, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            logs: dict[str, float] = {
                "train_loss": train_loss,
                "train_acc": float(train_acc) if train_acc is not None else 0.0,
                "val_loss": val_loss,
                "val_acc": val_metrics.get("accuracy", 0.0),
                "val_f1_macro": val_metrics.get("f1_macro", 0.0),
            }

            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, logs)

            if self._check_time_limit(start_time):
                break

        if not trained_any_batch:
            # No real training happened — any loss/accuracy would be fake.
            # A stopped-before-started trial is a pruned trial, not a result.
            if trial is not None:
                raise optuna.TrialPruned()
            raise AutoMLTrainingError(
                "Training stopped before any batch completed; no metrics to report"
            )

        if not math.isfinite(best_val_loss):
            best_val_loss, best_val_metrics = last_val

        # Final held-out evaluation — once per fit, after selection. Extra
        # metrics must never fail training, so a failure degrades to no test
        # entries rather than raising.
        try:
            test_metrics: dict[str, float] = self.test()
        except Exception as e:
            logger.warning("Final test evaluation failed; skipping test metrics: %s", e)
            test_metrics = {}

        best_val_metrics["train_loss"] = float(last_train_loss)
        if last_train_acc is not None:
            best_val_metrics["train_acc"] = float(last_train_acc)
        best_val_metrics["val_loss"] = float(best_val_loss)
        best_val_metrics["val_acc"] = float(best_val_metrics.get("accuracy", 0.0))
        best_val_metrics.update(
            {
                ("test_acc" if k == "accuracy" else f"test_{k}"): float(v)
                for k, v in test_metrics.items()
                if k != "loss"
            }
        )
        if "loss" in test_metrics:
            best_val_metrics["test_loss"] = float(test_metrics["loss"])
        return best_val_loss, best_val_metrics


# ---------------------------------------------------------------------------
# run_optuna_search
# ---------------------------------------------------------------------------
def _clamp_per_trial_timeout(
    timeout_per_trial: float | None,
    deadline: float | None,
    now: float,
) -> float | None:
    """Clamp a trial's budget to the study's remaining overall time.

    ``study.optimize(timeout=...)`` only stops *new* trials from starting; a
    running trial would otherwise finish in full and blow the caller's
    ``time_budget``. Clamping each trial's ``time_limit`` to the remaining
    deadline bounds the total overshoot by preprocessing + one batch instead
    of a full trial. Returns ``None`` when no budget applies; raises
    ``optuna.TrialPruned`` when the deadline has already passed.
    """
    if deadline is None:
        return timeout_per_trial
    remaining: float = deadline - now
    if remaining <= 0:
        raise optuna.TrialPruned()
    return remaining if timeout_per_trial is None else min(timeout_per_trial, remaining)


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

    When ``timeout`` is set it is a hard wall-clock cap on the whole search:
    each trial receives an even share as ``timeout_per_trial``, clamped to
    the remaining overall time, so a running trial never runs past the
    deadline by more than its current batch.

    Raises:
        ValueError: If ``task_type`` is not in ``OBJECTIVE_REGISTRY``.
    """
    from app.ml_engine.hpo.optuna_objectives import OBJECTIVE_REGISTRY

    if task_type not in OBJECTIVE_REGISTRY:
        raise AutoMLConfigError(
            f"Unknown task type '{task_type}'. Supported: {sorted(OBJECTIVE_REGISTRY)}"
        )

    config = load_task_config(task_type)
    objective_fn = OBJECTIVE_REGISTRY[task_type]

    run_dir = workdir / "optuna"
    run_dir.mkdir(exist_ok=True)

    pruner = optuna.pruners.SuccessiveHalvingPruner(
        # Prune from epoch 2 on: with max_epochs of 10-20 across configs,
        # min_resource=10 effectively never pruned anything.
        min_resource=2,
        reduction_factor=3,
        min_early_stopping_rate=0,
    )
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    timeout_per_trial = timeout / max(n_trials, 1) if timeout else None
    deadline: float | None = (time.time() + timeout) if timeout else None

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
        lambda trial: objective_fn(
            trial,
            **{
                **objective_kwargs,
                "timeout_per_trial": _clamp_per_trial_timeout(
                    timeout_per_trial, deadline, time.time()
                ),
            },
        ),
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

    # Objectives attach their best-epoch metrics as a trial user attr, so the
    # search can report what the winning hyperparameters actually achieved.
    best_metrics: dict[str, float] = dict(
        study.best_trial.user_attrs.get("best_metrics") or {}
    )

    return {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_metrics": best_metrics,
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


# ---------------------------------------------------------------------------
# AutoGluon tabular trainer
# ---------------------------------------------------------------------------

load_dotenv(find_dotenv())

_tabular_settings = get_settings()
DEFAULT_TABULAR_TRAIN_TEST_SPLIT_SIZE: float = (
    _tabular_settings.default_tabular_train_test_split_size
)

DEFAULT_TIME_LIMIT: int = _tabular_settings.default_time_limit

if not 0 < DEFAULT_TABULAR_TRAIN_TEST_SPLIT_SIZE < 1:
    raise AutoMLConfigError(
        f"DEFAULT_TABULAR_TRAIN_TEST_SPLIT_SIZE must be in (0, 1), "
        f"got {DEFAULT_TABULAR_TRAIN_TEST_SPLIT_SIZE}"
    )


class AutoGluonTrainer:
    """Wrapper around AutoGluon Tabular training routines."""

    def __init__(
        self,
        save_model_path: Path,
        DatasetClass=TabularDataset,
        PredictorClass=TabularPredictor,
    ):
        if save_model_path == "":
            raise AutoMLConfigError("save_model_path cannot be None or empty")

        self.save_model_path: Path = Path(save_model_path)

        if self.save_model_path.exists() and not self.save_model_path.is_dir():
            raise AutoMLConfigError(
                f"save_model_path must be a directory, got: {self.save_model_path}"
            )

        self.DatasetClass = DatasetClass
        self.PredictorClass = PredictorClass
        logger.debug(f"AutoGluon trainer, model path {self.save_model_path}")

    def train(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame | None,
        target_column: str,
        time_limit: int,
        num_cpus: str | int = "auto",
        num_gpus: str | int = "auto",
    ) -> tuple[pd.DataFrame, TabularPredictor, dict[str, float], float | None]:
        """Train an AutoGluon model on the given DataFrames.

        Validates the target column and time limit, splits train/test if no
        test set was supplied, fits the predictor within the time limit, then
        tries to clone the predictor for deployment (falling back to the
        original on failure), produces the leaderboard on the test set, and
        computes a metrics suite on the same test set (best-effort).

        Returns ``(leaderboard, predictor, test_metrics, train_acc)`` where
        ``test_metrics`` is empty when the suite could not be computed and
        ``train_acc`` is the best model's training-set accuracy (``None``
        for regression or when not computable).
        """
        if not isinstance(train_df, pd.DataFrame):
            raise AutoMLDataError("train_df is not a DataFrame")

        if train_df is None or train_df.empty:
            raise AutoMLDataError("train_df cannot be None or empty")

        if not target_column or not isinstance(target_column, str):
            raise AutoMLDataError("target_column must be a non-empty string")

        if target_column not in train_df.columns:
            raise AutoMLDataError(
                f"target_column '{target_column}' not found in train_df columns: {train_df.columns.tolist()}"
            )

        if test_df is not None and target_column not in test_df.columns:
            raise AutoMLDataError(
                f"target_column '{target_column}' not found in test_df columns: {test_df.columns.tolist()}"
            )

        if not isinstance(time_limit, int) or time_limit <= 0:
            raise AutoMLConfigError(
                f"time_limit must be a positive integer, got {time_limit}"
            )

        try:
            final_train_df, final_test_df = self.train_test_split(
                test_df=test_df, train_df=train_df
            )
        except Exception as e:
            logger.exception("Failed to split train/test data")
            raise AutoMLDataError(f"Train/test split failed: {e}") from e

        try:
            train_dataset = self.DatasetClass(final_train_df)
            test_dataset = self.DatasetClass(final_test_df)
        except Exception as e:
            logger.exception("Failed to create TabularDataset")
            raise AutoMLDataError(f"Dataset creation failed: {e}") from e

        try:
            predictor = self.PredictorClass(
                label=target_column, path=str(self.save_model_path)
            ).fit(
                train_data=train_dataset,
                time_limit=time_limit,
                num_gpus=num_gpus,
                num_cpus=num_cpus,
            )
        except Exception as e:
            logger.exception("AutoGluon training failed")
            raise AutoMLTrainingError(f"Model training failed: {e}") from e

        try:
            save_path_clone_opt = self.save_model_path / "-clone-opt"
            path_clone_opt = predictor.clone_for_deployment(
                path=str(save_path_clone_opt)
            )
            predictor_clone_opt = self.PredictorClass.load(path=str(path_clone_opt))
        except Exception as e:
            logger.warning(
                f"Failed to clone model for deployment, falling back to base predictor: {e}"
            )
            predictor_clone_opt = predictor

        try:
            leaderboard = predictor.leaderboard(test_dataset)
        except Exception as e:
            logger.exception("Failed to generate leaderboard")
            raise AutoMLLeaderboardError(f"Leaderboard generation failed: {e}") from e

        test_metrics: dict[str, float] = _compute_tabular_test_metrics(
            predictor_clone_opt, final_test_df, target_column
        )
        # Accuracy of the best model on its own training data — best-effort
        # (None for regression or when the suite could not be computed).
        train_metrics: dict[str, float] = _compute_tabular_test_metrics(
            predictor_clone_opt, final_train_df, target_column
        )
        train_acc: float | None = train_metrics.get("accuracy")
        return leaderboard, predictor_clone_opt, test_metrics, train_acc

    def train_test_split(
        self, test_df: pd.DataFrame | None, train_df: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the train/test pair to train on.

        When no test DataFrame is given, a fraction of the training data is
        sampled out to serve as the test set; otherwise both frames are used
        unchanged.
        """
        if train_df is None:
            raise AutoMLDataError("train_df cannot be None")

        if train_df.empty:
            raise AutoMLDataError("train_df cannot be empty")

        if test_df is None:
            logger.debug("Test dataset not found, creating split")
            final_train_df = train_df.sample(
                frac=DEFAULT_TABULAR_TRAIN_TEST_SPLIT_SIZE, random_state=42
            )
            final_test_df = train_df.drop(index=final_train_df.index.tolist())
        else:
            logger.debug("Test dataset found")

            if test_df.empty:
                raise AutoMLDataError("test_df cannot be empty")

            final_train_df = train_df
            final_test_df = test_df

        if final_train_df.empty:
            raise AutoMLDataError("Final training DataFrame is empty after split")

        if final_test_df.empty:
            raise AutoMLDataError("Final test DataFrame is empty after split")

        return final_train_df, final_test_df
