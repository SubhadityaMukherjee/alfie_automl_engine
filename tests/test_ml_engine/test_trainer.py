"""Tests for app/ml_engine/trainer.py — EarlyStopping (fast, no ML)."""

import math
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import optuna
import pytest
import torch

from app.core.exceptions import AutoMLConfigError, AutoMLTrainingError
from app.ml_engine.hpo.optuna_objectives import OBJECTIVE_REGISTRY
from app.ml_engine.configs import SUPPORTED_TASK_TYPES
from app.ml_engine.trainer import (
    EarlyStopping,
    FabricTrainer,
    _clamp_per_trial_timeout,
    _compute_classification_metrics,
    _compute_regression_metrics,
    _copy_best_trial_artifacts,
    _resolve_precision,
    run_optuna_search,
)


# ---------------------------------------------------------------------------
# _resolve_precision
# ---------------------------------------------------------------------------


def test_resolve_precision_explicit_values_pass_through():
    assert _resolve_precision("32-true") == "32-true"
    assert _resolve_precision("bf16-mixed") == "bf16-mixed"
    assert _resolve_precision("16-mixed") == "16-mixed"


def test_resolve_precision_auto_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _resolve_precision("auto") == "32-true"


def test_resolve_precision_auto_cuda_bf16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert _resolve_precision("auto") == "bf16-mixed"


def test_resolve_precision_auto_cuda_fp16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert _resolve_precision("auto") == "16-mixed"


# ---------------------------------------------------------------------------
# _compute_classification_metrics
# ---------------------------------------------------------------------------


def test_classification_metrics_binary_perfect():
    probs = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    labels = torch.tensor([0, 1])

    metrics = _compute_classification_metrics(probs, labels)

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["precision_macro"] == 1.0
    assert metrics["recall_macro"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["f1_weighted"] == 1.0
    assert metrics["cohen_kappa"] == 1.0
    assert metrics["matthews_corrcoef"] == 1.0
    assert metrics["roc_auc"] == 1.0
    # Top-k is trivial for 2 classes, so it is not reported.
    assert "top3_accuracy" not in metrics
    assert "top5_accuracy" not in metrics


def test_classification_metrics_multiclass_hand_computed():
    # 3 classes, 5 samples: sample 1 is the only mistake (pred 1, true 0).
    probs = torch.tensor(
        [
            [0.6, 0.2, 0.2],  # true 0 -> pred 0 (correct)
            [0.2, 0.6, 0.2],  # true 0 -> pred 1 (wrong)
            [0.2, 0.6, 0.2],  # true 1 -> pred 1 (correct)
            [0.2, 0.6, 0.2],  # true 1 -> pred 1 (correct)
            [0.2, 0.2, 0.6],  # true 2 -> pred 2 (correct)
        ]
    )
    labels = torch.tensor([0, 0, 1, 1, 2])

    metrics = _compute_classification_metrics(probs, labels)

    assert metrics["accuracy"] == pytest.approx(0.8)
    # Per-class F1: 2/3 (class 0), 0.8 (class 1), 1.0 (class 2).
    assert metrics["f1_macro"] == pytest.approx(2 / 3 / 3 + 0.8 / 3 + 1 / 3)
    # Macro OvR AUC: 3/4 (class 0), 5/6 (class 1), 1.0 (class 2).
    assert metrics["roc_auc"] == pytest.approx(3 / 4 / 3 + 5 / 6 / 3 + 1 / 3)


def test_classification_metrics_topk():
    # 4 classes, all present in labels: top3 reported, top5 not.
    probs = torch.tensor(
        [
            [0.6, 0.2, 0.15, 0.05],  # true 0 -> pred 0 (correct)
            [0.1, 0.5, 0.3, 0.1],  # true 1 -> pred 1 (correct)
            [0.4, 0.35, 0.2, 0.05],  # true 2 -> pred 0 (wrong); 2 in top-3
            [0.55, 0.25, 0.15, 0.05],  # true 3 -> pred 0 (wrong); 3 not in top-3
        ]
    )
    labels = torch.tensor([0, 1, 2, 3])

    metrics = _compute_classification_metrics(probs, labels)

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["top3_accuracy"] == pytest.approx(0.75)
    assert "top5_accuracy" not in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_classification_metrics_degenerate_single_class_labels():
    # Only one class present in labels: ROC-AUC is undefined and must be
    # skipped instead of raising or reporting a misleading value.
    probs = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
    labels = torch.tensor([0, 0, 0])

    metrics = _compute_classification_metrics(probs, labels)

    assert metrics["accuracy"] == 1.0
    assert "roc_auc" not in metrics


# ---------------------------------------------------------------------------
# _compute_regression_metrics
# ---------------------------------------------------------------------------


def test_regression_metrics_hand_computed():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])

    metrics = _compute_regression_metrics(y_true, y_pred)

    assert metrics["r2"] == pytest.approx(-1.0)  # 1 - ss_res/ss_tot = 1 - 4/2
    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["rmse"] == pytest.approx(math.sqrt(4 / 3))
    assert metrics["mape"] == pytest.approx((0.0 + 0.0 + 2 / 3) / 3)
    assert metrics["median_absolute_error"] == pytest.approx(0.0)
    assert "explained_variance" in metrics


def test_regression_metrics_perfect_predictions():
    y = np.array([1.5, -2.0, 7.25])
    metrics = _compute_regression_metrics(y, y.copy())

    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0


# ---------------------------------------------------------------------------
# EarlyStopping — pure logic, no model/HF downloads needed
# ---------------------------------------------------------------------------


def test_early_stopping_init_defaults():
    es = EarlyStopping()
    assert es.monitor == "val_loss"
    assert es.patience == 3
    assert es.min_delta == 0.0
    assert es.best == float("inf")
    assert es.counter == 0


def test_early_stopping_init_custom_params():
    es = EarlyStopping(monitor="train_loss", patience=5, min_delta=0.01)
    assert es.monitor == "train_loss"
    assert es.patience == 5
    assert es.min_delta == 0.01


def test_early_stopping_improvement_updates_best():
    es = EarlyStopping()
    trainer = MagicMock()
    es.on_epoch_end(trainer, 0, {"val_loss": 0.5})
    assert es.best == 0.5
    assert es.counter == 0


def test_early_stopping_continued_improvement_resets_counter():
    es = EarlyStopping()
    trainer = MagicMock()
    es.on_epoch_end(trainer, 0, {"val_loss": 0.5})
    es.on_epoch_end(trainer, 1, {"val_loss": 0.6})  # worse
    es.on_epoch_end(trainer, 2, {"val_loss": 0.3})  # better
    assert es.best == 0.3
    assert es.counter == 0


def test_early_stopping_no_improvement_increments_counter():
    es = EarlyStopping(patience=3)
    trainer = MagicMock()
    es.on_epoch_end(trainer, 0, {"val_loss": 0.5})
    es.on_epoch_end(trainer, 1, {"val_loss": 0.6})
    assert es.counter == 1
    # trainer.epochs should NOT be set before patience is exhausted
    assert not hasattr(trainer, "epochs") or trainer.epochs != 1 + 1


def test_early_stopping_triggers_at_patience():
    es = EarlyStopping(patience=3)
    trainer = MagicMock()
    trainer.epochs = 100
    # First call: improvement
    es.on_epoch_end(trainer, 0, {"val_loss": 0.5})
    # Next 3 calls: no improvement
    es.on_epoch_end(trainer, 1, {"val_loss": 0.6})
    es.on_epoch_end(trainer, 2, {"val_loss": 0.7})
    es.on_epoch_end(trainer, 3, {"val_loss": 0.8})
    # Counter reaches patience=3 on epoch 3, so trainer.epochs = epoch + 1 = 4
    assert trainer.epochs == 4


def test_early_stopping_does_not_trigger_before_patience():
    es = EarlyStopping(patience=3)
    trainer = MagicMock()
    trainer.epochs = 100
    es.on_epoch_end(trainer, 0, {"val_loss": 0.5})
    es.on_epoch_end(trainer, 1, {"val_loss": 0.6})  # counter = 1
    es.on_epoch_end(trainer, 2, {"val_loss": 0.7})  # counter = 2
    # patience not yet reached (counter < 3)
    assert trainer.epochs == 100


def test_early_stopping_missing_metric_skips(caplog):
    import logging

    es = EarlyStopping(monitor="val_loss")
    trainer = MagicMock()
    with caplog.at_level(logging.WARNING):
        es.on_epoch_end(trainer, 0, {"train_loss": 0.5})
    assert es.counter == 0
    assert es.best == float("inf")
    assert "val_loss" in caplog.text


def test_early_stopping_min_delta_prevents_false_improvement():
    es = EarlyStopping(patience=2, min_delta=0.1)
    trainer = MagicMock()
    trainer.epochs = 100
    es.on_epoch_end(trainer, 0, {"val_loss": 0.5})
    # Improvement is only 0.04, less than min_delta=0.1 → not treated as improvement
    es.on_epoch_end(trainer, 1, {"val_loss": 0.46})
    assert es.counter == 1


def test_early_stopping_min_delta_accepts_sufficient_improvement():
    es = EarlyStopping(patience=2, min_delta=0.1)
    trainer = MagicMock()
    es.on_epoch_end(trainer, 0, {"val_loss": 0.5})
    # Improvement is 0.15, greater than min_delta=0.1 → treated as improvement
    es.on_epoch_end(trainer, 1, {"val_loss": 0.35})
    assert es.counter == 0
    assert es.best == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# OBJECTIVE_REGISTRY
# ---------------------------------------------------------------------------


def test_objective_registry_has_all_task_types():
    assert set(OBJECTIVE_REGISTRY.keys()) == SUPPORTED_TASK_TYPES


def test_objective_registry_values_are_callable():
    for task_type, fn in OBJECTIVE_REGISTRY.items():
        assert callable(fn), f"{task_type} objective is not callable"


# ---------------------------------------------------------------------------
# run_optuna_search dispatch
# ---------------------------------------------------------------------------


def test_run_optuna_search_raises_for_unknown_task(tmp_path):
    with pytest.raises(AutoMLConfigError, match="Unknown task type"):
        run_optuna_search(
            task_type="flying_toaster",
            csv_path=tmp_path / "labels.csv",
            images_dir=tmp_path / "images",
            workdir=tmp_path,
        )


# ---------------------------------------------------------------------------
# _copy_best_trial_artifacts
# ---------------------------------------------------------------------------


def test_copy_best_trial_artifacts_copies_existing_files(tmp_path: Path):
    best = tmp_path / "optuna" / "trial_3"
    best.mkdir(parents=True)
    (best / "feature_mapping.json").write_text('{"task_type": "x"}')
    (best / "model.pt").write_text("weights")

    _copy_best_trial_artifacts(best, tmp_path)

    model_dir = tmp_path / "model"
    assert (model_dir / "feature_mapping.json").read_text() == '{"task_type": "x"}'
    assert (model_dir / "model.pt").read_text() == "weights"


def test_copy_best_trial_artifacts_skips_missing_sources(tmp_path: Path):
    best = tmp_path / "optuna" / "trial_0"
    best.mkdir(parents=True)
    # Only feature_mapping.json exists; model.pt missing
    (best / "feature_mapping.json").write_text("{}")

    _copy_best_trial_artifacts(best, tmp_path)

    model_dir = tmp_path / "model"
    assert (model_dir / "feature_mapping.json").exists()
    assert not (model_dir / "model.pt").exists()


def test_copy_best_trial_artifacts_handles_missing_trial_dir(tmp_path: Path):
    # Best-effort: should not raise even if the trial dir doesn't exist.
    _copy_best_trial_artifacts(tmp_path / "optuna" / "trial_99", tmp_path)
    assert (tmp_path / "model").exists()  # model dir still created


# ---------------------------------------------------------------------------
# per-trial timeout enforcement
# ---------------------------------------------------------------------------


class _StubTrainer:
    """Captures constructor kwargs; fit() returns a fixed best-epoch result."""

    last_kwargs: dict | None = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    def fit(self, trial=None):
        return 0.5, {"loss": 0.5, "accuracy": 0.9}


class _StubDataModule:
    def __init__(self, **kwargs):
        pass


class _StubModel:
    def __init__(self, **kwargs):
        pass


def _stub_config() -> dict:
    return {
        "small_models": ["stub-a", "stub-b"],
        "medium_models": ["stub-a"],
        "large_models": ["stub-a"],
        "lr_low": 1e-5,
        "lr_high": 1e-3,
        "batch_sizes": [2],
        "weight_decay_low": 1e-6,
        "weight_decay_high": 0.1,
        "max_epochs": 1,
        "early_stopping_patience": 1,
    }


def test_clamp_per_trial_timeout_no_deadline():
    assert _clamp_per_trial_timeout(30.0, None, now=1000.0) == 30.0
    assert _clamp_per_trial_timeout(None, None, now=1000.0) is None


def test_clamp_per_trial_timeout_even_split_binds():
    assert _clamp_per_trial_timeout(30.0, deadline=1060.0, now=1000.0) == 30.0


def test_clamp_per_trial_timeout_deadline_binds():
    assert _clamp_per_trial_timeout(30.0, deadline=1010.0, now=1000.0) == 10.0


def test_clamp_per_trial_timeout_no_even_split_uses_remaining():
    assert _clamp_per_trial_timeout(None, deadline=1015.0, now=1000.0) == 15.0


def test_clamp_per_trial_timeout_spent_deadline_prunes():
    with pytest.raises(optuna.TrialPruned):
        _clamp_per_trial_timeout(30.0, deadline=999.0, now=1000.0)


def test_objective_base_hands_remaining_budget_to_trainer(monkeypatch):
    from app.ml_engine.hpo import optuna_objectives
    from app.ml_engine.hpo.optuna_objectives import _optuna_objective_base

    monkeypatch.setattr(optuna_objectives, "FabricTrainer", _StubTrainer)
    study = optuna.create_study(direction="minimize")
    trial = study.ask()

    best = _optuna_objective_base(
        trial,
        model_size="small",
        config=_stub_config(),
        datamodule_class=_StubDataModule,
        dm_kwargs={},
        model_class=_StubModel,
        build_model_kwargs=lambda dm, mid: {"model_id": mid},
        timeout_per_trial=60.0,
        workdir=None,
        task_type="image_classification",
    )

    assert best == 0.5
    assert _StubTrainer.last_kwargs is not None
    time_limit = _StubTrainer.last_kwargs["time_limit"]
    assert time_limit is not None
    assert 0 < time_limit <= 60.0


def test_objective_base_prunes_when_budget_already_spent(monkeypatch):
    from app.ml_engine.hpo import optuna_objectives
    from app.ml_engine.hpo.optuna_objectives import _optuna_objective_base

    monkeypatch.setattr(optuna_objectives, "FabricTrainer", _StubTrainer)
    study = optuna.create_study(direction="minimize")
    trial = study.ask()

    with pytest.raises(optuna.TrialPruned):
        _optuna_objective_base(
            trial,
            model_size="small",
            config=_stub_config(),
            datamodule_class=_StubDataModule,
            dm_kwargs={},
            model_class=_StubModel,
            build_model_kwargs=lambda dm, mid: {"model_id": mid},
            timeout_per_trial=0.0,
            workdir=None,
            task_type="image_classification",
        )


def test_objective_base_without_timeout_leaves_trainer_unbounded(monkeypatch):
    from app.ml_engine.hpo import optuna_objectives
    from app.ml_engine.hpo.optuna_objectives import _optuna_objective_base

    monkeypatch.setattr(optuna_objectives, "FabricTrainer", _StubTrainer)
    study = optuna.create_study(direction="minimize")
    trial = study.ask()

    _optuna_objective_base(
        trial,
        model_size="small",
        config=_stub_config(),
        datamodule_class=_StubDataModule,
        dm_kwargs={},
        model_class=_StubModel,
        build_model_kwargs=lambda dm, mid: {"model_id": mid},
        timeout_per_trial=None,
        workdir=None,
        task_type="image_classification",
    )

    assert _StubTrainer.last_kwargs is not None
    assert "time_limit" not in _StubTrainer.last_kwargs


def test_run_optuna_search_passes_per_trial_budget(monkeypatch, tmp_path):
    from app.ml_engine.hpo import optuna_objectives as objectives_module

    received: list[float | None] = []

    def stub_objective(trial, **kwargs):
        received.append(kwargs.get("timeout_per_trial"))
        return 0.5

    monkeypatch.setattr(
        objectives_module,
        "OBJECTIVE_REGISTRY",
        {"image_classification": stub_objective},
    )

    result = run_optuna_search(
        task_type="image_classification",
        csv_path=tmp_path / "labels.csv",
        images_dir=tmp_path / "images",
        workdir=tmp_path,
        n_trials=3,
        timeout=90,
    )

    assert result["best_value"] == 0.5
    assert len(received) == 3
    # Even split: 90s / 3 trials = 30s each.
    assert received[0] == pytest.approx(30.0, rel=0.05)
    # Clamping to the study deadline is monotone non-increasing.
    assert received[1] <= received[0] + 1e-9
    assert received[2] <= received[1] + 1e-9


def test_run_optuna_search_without_timeout_has_no_per_trial_budget(
    monkeypatch, tmp_path
):
    from app.ml_engine.hpo import optuna_objectives as objectives_module

    received: list[float | None] = []

    def stub_objective(trial, **kwargs):
        received.append(kwargs.get("timeout_per_trial"))
        return 0.25

    monkeypatch.setattr(
        objectives_module,
        "OBJECTIVE_REGISTRY",
        {"image_classification": stub_objective},
    )

    result = run_optuna_search(
        task_type="image_classification",
        csv_path=tmp_path / "labels.csv",
        images_dir=tmp_path / "images",
        workdir=tmp_path,
        n_trials=2,
        timeout=None,
    )

    assert result["best_value"] == 0.25
    assert received == [None, None]


# ---------------------------------------------------------------------------
# stopped-before-training guards (no fake metrics from empty passes)
# ---------------------------------------------------------------------------


def _bare_trainer(**attrs) -> FabricTrainer:
    """A FabricTrainer built without __init__, stubbed for loop tests."""
    trainer = FabricTrainer.__new__(FabricTrainer)
    trainer.fabric = SimpleNamespace(device="cpu")
    trainer.model = SimpleNamespace(train=lambda: None, eval=lambda: None)
    trainer.model_computes_loss = False
    trainer.loss_fn = None
    trainer.time_limit = None
    trainer.epochs = 1
    trainer.callbacks = []
    trainer.train_loader = []
    trainer.val_loader = []
    trainer.test_loader = []
    for key, value in attrs.items():
        setattr(trainer, key, value)
    return trainer


def test_train_epoch_without_batches_returns_none():
    trainer = _bare_trainer()
    assert trainer.train_epoch(0, start_time=time.time()) == (None, None)


def test_validate_without_batches_reports_inf_loss():
    trainer = _bare_trainer()
    metrics = trainer.validate(start_time=time.time())
    assert metrics["loss"] == math.inf


def test_test_without_batches_returns_no_metrics():
    trainer = _bare_trainer()
    assert trainer.test() == {}


def test_fit_returns_aliases_after_normal_training():
    trainer = _bare_trainer(epochs=2)
    trainer.train_epoch = lambda epoch, start_time: (0.5, 0.9)
    trainer.validate = lambda start_time: {"loss": 0.4, "accuracy": 0.91}
    trainer.test = lambda: {"loss": 0.45, "accuracy": 0.88, "f1_macro": 0.86}

    best_loss, metrics = trainer.fit()

    assert best_loss == 0.4
    assert metrics["train_loss"] == pytest.approx(0.5)
    assert metrics["train_acc"] == pytest.approx(0.9)
    assert metrics["val_loss"] == pytest.approx(0.4)
    assert metrics["val_acc"] == pytest.approx(0.91)
    assert metrics["test_loss"] == pytest.approx(0.45)
    assert metrics["test_acc"] == pytest.approx(0.88)
    assert metrics["test_f1_macro"] == pytest.approx(0.86)


def test_fit_raises_when_time_limit_fired_before_any_batch():
    trainer = _bare_trainer()
    trainer.train_epoch = lambda epoch, start_time: (None, None)

    with pytest.raises(AutoMLTrainingError):
        trainer.fit()


def test_fit_prunes_trial_when_time_limit_fired_before_any_batch():
    trainer = _bare_trainer()
    trainer.train_epoch = lambda epoch, start_time: (None, None)
    study = optuna.create_study(direction="minimize")
    trial = study.ask()

    with pytest.raises(optuna.TrialPruned):
        trainer.fit(trial=trial)
