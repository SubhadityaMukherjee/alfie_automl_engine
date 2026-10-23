"""Tests for app.core.service_helpers.metadata_indicates_zip_download."""

from typing import cast

import pytest

from app.core.service_helpers import metadata_indicates_zip_download


def test_file_type_zip():
    assert metadata_indicates_zip_download({"file_type": "zip"}) is True


def test_file_type_zip_normalization():
    assert metadata_indicates_zip_download({"file_type": " ZIP "}) is True


def test_original_filename_zip():
    assert (
        metadata_indicates_zip_download(
            {"file_type": "csv, jpg", "original_filename": "Dataset.ZIP"}
        )
        is True
    )


def test_is_folder():
    assert metadata_indicates_zip_download({"file_type": "jpg, csv", "is_folder": True})


def test_contents_type_only_is_not_zip():
    assert (
        metadata_indicates_zip_download(
            {"file_type": "csv, jpg", "original_filename": "data.csv"}
        )
        is False
    )


def test_plain_csv_dataset():
    assert (
        metadata_indicates_zip_download(
            {"file_type": "csv", "original_filename": "train.csv", "is_folder": False}
        )
        is False
    )


def test_non_dict_metadata():
    assert metadata_indicates_zip_download(cast(dict, None)) is False
    assert metadata_indicates_zip_download(cast(dict, "zip")) is False


def test_empty_metadata():
    assert metadata_indicates_zip_download({}) is False


def test_falsy_folder_values():
    assert metadata_indicates_zip_download({"is_folder": False}) is False
    assert metadata_indicates_zip_download({"is_folder": None}) is False


# ---------------------------------------------------------------------------
# build_upload_payload: metrics / algorithm mapping
# ---------------------------------------------------------------------------


def test_build_upload_payload_without_metrics_omits_metric_fields():
    from app.core.service_helpers import build_upload_payload

    _, data = build_upload_payload("ds1", "v1", {}, "classification", {})
    for field in (
        "training_accuracy",
        "validation_accuracy",
        "test_accuracy",
        "training_loss",
        "algorithm",
    ):
        assert field not in data


def test_build_upload_payload_maps_metric_fields():
    from app.core.service_helpers import build_upload_payload

    metrics = {
        "train_acc": 0.99,
        "val_acc": 0.91,
        "test_acc": 0.88,
        "train_loss": 0.12,
    }
    _, data = build_upload_payload(
        "ds1", "v1", {}, "classification", {}, metrics=metrics
    )
    assert data["training_accuracy"] == pytest.approx(0.99)
    assert data["validation_accuracy"] == pytest.approx(0.91)
    assert data["test_accuracy"] == pytest.approx(0.88)
    assert data["training_loss"] == pytest.approx(0.12)


def test_build_upload_payload_skips_missing_metric_values():
    from app.core.service_helpers import build_upload_payload

    _, data = build_upload_payload(
        "ds1", "v1", {}, "classification", {}, metrics={"val_acc": 0.9}
    )
    assert data["validation_accuracy"] == pytest.approx(0.9)
    assert "training_accuracy" not in data
    assert "test_accuracy" not in data
    assert "training_loss" not in data


def test_build_upload_payload_algorithm_forwarded():
    from app.core.service_helpers import build_upload_payload

    _, data = build_upload_payload(
        "ds1", "v1", {}, "classification", {}, algorithm="resnet18"
    )
    assert data["algorithm"] == "resnet18"


# ---------------------------------------------------------------------------
# extract_optuna_metrics_for_autodw
# ---------------------------------------------------------------------------


def test_extract_optuna_metrics_maps_best_result():
    from app.core.service_helpers import extract_optuna_metrics_for_autodw

    optuna_result = {
        "best_value": 0.1,
        "best_params": {"model_id": "resnet18", "lr": 1e-3},
        "best_metrics": {
            "loss": 0.1,
            "accuracy": 0.87,
            "f1_macro": 0.85,
            "train_loss": 0.05,
            "train_acc": 0.98,
            "val_loss": 0.1,
            "val_acc": 0.87,
            "test_loss": 0.12,
            "test_acc": 0.84,
        },
        "n_trials": 3,
    }
    metrics, algorithm = extract_optuna_metrics_for_autodw(optuna_result)
    assert metrics["train_acc"] == pytest.approx(0.98)
    assert metrics["val_acc"] == pytest.approx(0.87)
    assert metrics["test_acc"] == pytest.approx(0.84)
    assert metrics["train_loss"] == pytest.approx(0.05)
    assert algorithm == "resnet18"


def test_extract_optuna_metrics_val_acc_falls_back_to_accuracy():
    from app.core.service_helpers import extract_optuna_metrics_for_autodw

    metrics, _ = extract_optuna_metrics_for_autodw({"best_metrics": {"accuracy": 0.75}})
    assert metrics["val_acc"] == pytest.approx(0.75)


def test_extract_optuna_metrics_empty_result_degrades():
    from app.core.service_helpers import extract_optuna_metrics_for_autodw

    metrics, algorithm = extract_optuna_metrics_for_autodw({})
    assert metrics == {
        "train_acc": None,
        "val_acc": None,
        "test_acc": None,
        "train_loss": None,
    }
    assert algorithm is None


def test_extract_optuna_metrics_non_dict_fields_degrade():
    from app.core.service_helpers import extract_optuna_metrics_for_autodw

    metrics, algorithm = extract_optuna_metrics_for_autodw(
        {"best_metrics": None, "best_params": None}
    )
    assert metrics == {
        "train_acc": None,
        "val_acc": None,
        "test_acc": None,
        "train_loss": None,
    }
    assert algorithm is None
