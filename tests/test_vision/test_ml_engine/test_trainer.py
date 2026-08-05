"""Tests for app/vision_automl/ml_engine/trainer.py — EarlyStopping (fast, no ML)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.core.exceptions import AutoMLConfigError
from app.vision_automl.hpo.optuna_objectives import OBJECTIVE_REGISTRY
from app.vision_automl.ml_engine.configs import SUPPORTED_TASK_TYPES
from app.vision_automl.ml_engine.trainer import (
    EarlyStopping,
    _copy_best_trial_artifacts,
    run_optuna_search,
)

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
