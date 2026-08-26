"""Tests for multimodal helpers in app/vision_automl/services.py."""

import pandas as pd

from app.vision_automl.services import (
    _discover_auxiliary_columns,
    validate_multimodal_inputs,
)

# ---------------------------------------------------------------------------
# _discover_auxiliary_columns
# ---------------------------------------------------------------------------


def test_discover_auxiliary_columns_basic():
    df = pd.DataFrame(
        {
            "filename": ["a.png", "b.png"],
            "label": ["cat", "dog"],
            "age": [2.0, 5.0],
            "weight": [10.0, 20.0],
        }
    )
    result = _discover_auxiliary_columns(df, "filename", "label")
    assert result == ["age", "weight"]


def test_discover_auxiliary_columns_with_exclude():
    df = pd.DataFrame(
        {
            "filename": ["a.png"],
            "label": ["cat"],
            "age": [2.0],
            "weight": [10.0],
            "notes": ["some note"],
        }
    )
    result = _discover_auxiliary_columns(
        df, "filename", "label", exclude_columns=["notes"]
    )
    assert result == ["age", "weight"]


def test_discover_auxiliary_columns_no_aux_columns():
    df = pd.DataFrame({"filename": ["a.png"], "label": ["cat"]})
    result = _discover_auxiliary_columns(df, "filename", "label")
    assert result == []


def test_discover_auxiliary_columns_preserves_column_order():
    df = pd.DataFrame(
        {
            "filename": ["a.png"],
            "label": ["cat"],
            "col_a": [1],
            "col_b": [2],
            "col_c": [3],
        }
    )
    result = _discover_auxiliary_columns(df, "filename", "label")
    assert result == ["col_a", "col_b", "col_c"]


# ---------------------------------------------------------------------------
# validate_multimodal_inputs
# ---------------------------------------------------------------------------


def test_validate_multimodal_inputs_valid(class_structured_images_dir):
    csv_path, images_dir = class_structured_images_dir
    df = pd.read_csv(csv_path)
    df["age"] = range(len(df))
    df.to_csv(csv_path, index=False)
    error, aux_cols = validate_multimodal_inputs(
        csv_path, images_dir, "filename", "label"
    )
    assert error is None
    assert aux_cols == ["age"]


def test_validate_multimodal_inputs_missing_filename_column(
    class_structured_images_dir,
):
    csv_path, images_dir = class_structured_images_dir
    df = pd.read_csv(csv_path)
    df["age"] = range(len(df))
    df.to_csv(csv_path, index=False)
    error, aux_cols = validate_multimodal_inputs(
        csv_path, images_dir, "wrong_col", "label"
    )
    assert error is not None
    assert "Filename" in error
    assert aux_cols == []


def test_validate_multimodal_inputs_missing_label_column(class_structured_images_dir):
    csv_path, images_dir = class_structured_images_dir
    df = pd.read_csv(csv_path)
    df["age"] = range(len(df))
    df.to_csv(csv_path, index=False)
    error, aux_cols = validate_multimodal_inputs(
        csv_path, images_dir, "filename", "wrong_label"
    )
    assert error is not None
    assert "Label" in error
    assert aux_cols == []


def test_validate_multimodal_inputs_no_auxiliary_columns(class_structured_images_dir):
    csv_path, images_dir = class_structured_images_dir
    error, aux_cols = validate_multimodal_inputs(
        csv_path, images_dir, "filename", "label"
    )
    assert error is not None
    assert "No auxiliary columns" in error
    assert aux_cols == []


def test_validate_multimodal_inputs_all_null_aux_column(class_structured_images_dir):
    csv_path, images_dir = class_structured_images_dir
    df = pd.read_csv(csv_path)
    df["all_null"] = [None] * len(df)
    df.to_csv(csv_path, index=False)
    error, aux_cols = validate_multimodal_inputs(
        csv_path, images_dir, "filename", "label"
    )
    assert error is not None
    assert "entirely null" in error
    assert aux_cols == []


def test_validate_multimodal_inputs_missing_images(class_structured_images_dir):
    csv_path, images_dir = class_structured_images_dir
    df = pd.read_csv(csv_path)
    df["age"] = range(len(df))
    df.loc[0, "filename"] = "nonexistent.png"
    df.to_csv(csv_path, index=False)
    error, aux_cols = validate_multimodal_inputs(
        csv_path, images_dir, "filename", "label"
    )
    assert error is not None
    assert "Missing" in error
    assert aux_cols == []


def test_validate_multimodal_inputs_unreadable_csv(tmp_path):
    error, aux_cols = validate_multimodal_inputs(
        tmp_path / "nonexistent.csv", tmp_path, "filename", "label"
    )
    assert error is not None
    assert "Could not read" in error
    assert aux_cols == []


def test_validate_multimodal_inputs_with_exclude_columns(class_structured_images_dir):
    csv_path, images_dir = class_structured_images_dir
    df = pd.read_csv(csv_path)
    df["age"] = range(len(df))
    df["weight"] = range(len(df))
    df["notes"] = ["n"] * len(df)
    df.to_csv(csv_path, index=False)
    error, aux_cols = validate_multimodal_inputs(
        csv_path,
        images_dir,
        "filename",
        "label",
        exclude_columns=["notes"],
    )
    assert error is None
    assert "notes" not in aux_cols
    assert "age" in aux_cols
    assert "weight" in aux_cols
