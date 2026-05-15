"""Service layer for vision AutoML workflows.

Mirrors the structure of tabular_automl/services.py so both pipelines
share a consistent public API consumed by their respective main.py files.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi.concurrency import run_in_threadpool
from huggingface_hub import HfApi

from app.core.exceptions import AutoMLDataError, AutoMLSerializationError
from app.core.service_helpers import (
    build_upload_payload as _core_build_upload_payload,
    download_dataset as _core_download_dataset,
    fetch_dataset_metadata,
    resolve_download_url,
    upload_model,
)
from app.core.utils import jinja_environment, render_template
from app.vision_automl.ml_engine.trainer import run_optuna_search

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def normalize_dataframe_filenames(
    df: pd.DataFrame, filename_column: str, csv_path: Path
) -> pd.DataFrame:
    """Normalize filenames to basenames and persist CSV back to disk."""
    logger.info("Normalizing filenames in column '%s'", filename_column)
    if filename_column in df.columns:
        df[filename_column] = (
            df[filename_column]
            .astype(str)
            .map(lambda s: os.path.basename(str(s).replace("\\", "/")))
        )
        df.to_csv(csv_path, index=False)
        logger.debug("Normalized filenames saved to %s", csv_path)
    else:
        logger.warning(
            "Filename column '%s' not found during normalization", filename_column
        )
    return df


def resolve_images_root(images_dir: Path) -> Path:
    """Resolve common nested packaging patterns inside uploaded image zips."""
    logger.info("Resolving image directory structure at %s", images_dir)
    nested_images_dir = images_dir / "images"
    if nested_images_dir.exists() and nested_images_dir.is_dir():
        logger.debug("Detected nested 'images' folder, using it as root")
        images_dir = nested_images_dir

    try:
        top_level_entries = list(images_dir.iterdir())
        only_dirs = [p for p in top_level_entries if p.is_dir()]
        only_files = [p for p in top_level_entries if p.is_file()]
        if len(only_files) == 0 and len(only_dirs) == 1:
            logger.debug("Detected single top-level directory: %s", only_dirs[0])
            images_dir = only_dirs[0]
    except Exception as e:
        logger.warning("Error resolving image root: %s", e)

    return images_dir


def collect_missing_files(
    df: pd.DataFrame, images_dir: Path, filename_col: str, label_col: str
) -> list[str]:
    """Return a list of filenames referenced in the CSV but absent on disk."""
    missing = []
    for _, row in df.iterrows():
        filename = row[filename_col]

        img_path = images_dir / filename
        if img_path.exists():
            continue

        matches = list(images_dir.rglob(str(filename)))
        if len(matches) == 1:
            continue
        elif len(matches) > 1:
            logger.warning("Multiple matches for %s: %s", filename, matches)

        missing.append(filename)
    return missing


# ---------------------------------------------------------------------------
# Hugging Face model search helpers
# ---------------------------------------------------------------------------


def get_num_params_if_available(
    repo_id: str, revision: str | None = None
) -> int | None:
    """Try to retrieve number of parameters for a HF model, if available."""
    logger.debug("Fetching parameter count for model %s", repo_id)
    api = HfApi()
    try:
        info = api.model_info(repo_id, revision=revision, files_metadata=True)
        num_params = getattr(info, "safetensors", None)
        if num_params is not None:
            return num_params.total
    except Exception as e:
        logger.warning("Failed to retrieve num_params for %s: %s", repo_id, e)
    return None


def search_hf_for_pytorch_models_with_estimated_parameters(
    filter: str = "image-classification", limit: int = 3, sort: str = "downloads"
) -> list[dict[str, Any]]:
    """Search HF for PyTorch image-classification models annotated with param counts."""
    logger.info("Searching Hugging Face models for filter='%s'", filter)
    api = HfApi()
    models = api.list_models(
        filter=filter,
        library="pytorch",
        sort=sort,
        direction=-1,
        limit=limit,
    )

    results: list[dict[str, Any]] = []
    for m in models:
        num_params = get_num_params_if_available(m.id)
        if num_params:
            results.append(
                {
                    "model_id": m.id,
                    "downloads": getattr(m, "downloads", None),
                    "likes": getattr(m, "likes", None),
                    "last_modified": getattr(m, "lastModified", None),
                    "private": getattr(m, "private", None),
                    "num_params": num_params,
                }
            )

    logger.info("Found %d models with parameter info", len(results))
    return results


def sort_models_by_size(
    models: list[dict[str, Any]], size_tier: str
) -> list[dict[str, Any]]:
    """Filter and sort models by size tier based on estimated parameter counts."""
    logger.info("Sorting models by size tier: %s", size_tier)
    tier = str(size_tier).strip().lower()

    SMALL_MAX: int = int(os.getenv("MODEL_SMALL_MAX_PARAM_SIZE", 50_000_000))
    MEDIUM_MIN: int = SMALL_MAX + 1
    MEDIUM_MAX: int = int(os.getenv("MODEL_MEDIUM_MAX_PARAM_SIZE", 200_000_000))
    LARGE_MIN: int = MEDIUM_MAX + 1

    def in_tier(m: dict[str, Any]) -> bool:
        n = m.get("num_params")
        if n is None:
            return False
        if tier == "small":
            return 0 <= n <= SMALL_MAX
        if tier == "medium":
            return MEDIUM_MIN <= n <= MEDIUM_MAX
        if tier == "large":
            return n >= LARGE_MIN
        return True

    filtered = [m for m in models if in_tier(m)]
    if not filtered:
        logger.warning("No models matched tier '%s'; falling back to all models", tier)
        filtered = models

    return sorted(
        filtered, key=lambda m: (m.get("num_params") is None, m.get("num_params", 0))
    )


# ---------------------------------------------------------------------------
# Dataset fetch & extraction
# ---------------------------------------------------------------------------


def download_dataset(download_url: str, workdir: Path, original_filename: str) -> Path:
    """Stream-download the ZIP dataset and return its local path."""
    dest_path = workdir / original_filename
    return _core_download_dataset(download_url, dest_path)


# ---------------------------------------------------------------------------
# ZIP extraction & structure resolution
# ---------------------------------------------------------------------------


def extract_and_locate_dataset(zip_path: Path, workdir: Path) -> tuple[Path, Path]:
    """
    Extract a vision dataset ZIP and return (csv_path, images_dir).

    Raises DatasetValidationError for structural problems.
    """
    extract_dir = workdir / "dataset"
    extract_dir.mkdir(exist_ok=True)
    shutil.unpack_archive(zip_path, extract_dir)

    dataset_root = _find_valid_dataset_root(extract_dir)
    csv_path = _find_csv_file(dataset_root)
    images_dir = _find_or_resolve_images_dir(dataset_root, csv_path)
    return csv_path, images_dir


def _find_valid_dataset_root(extract_dir: Path) -> Path:
    real_dirs = [
        child
        for child in extract_dir.iterdir()
        if child.is_dir()
        and child.name != "__MACOSX"
        and not child.name.startswith(".")
    ]
    if not real_dirs:
        raise AutoMLDataError("No valid dataset folder found in ZIP")
    return real_dirs[0]


def _find_csv_file(dataset_root: Path) -> Path:
    csv_candidates = [
        p
        for p in dataset_root.rglob("*")
        if p.is_file() and p.name in ("labels.csv", "metadata.csv")
    ]
    if not csv_candidates:
        raise AutoMLDataError("labels.csv or metadata.csv not found in dataset")
    return csv_candidates[0]


def _find_or_resolve_images_dir(dataset_root: Path, csv_path: Path) -> Path:
    images_candidates = [
        p for p in dataset_root.rglob("*") if p.is_dir() and p.name == "images"
    ]
    images_dir = (
        images_candidates[0] if images_candidates else (csv_path.parent / "images")
    )
    resolved_dir = resolve_images_root(images_dir)
    if not resolved_dir.exists():
        raise AutoMLDataError("images/ directory not found in dataset ZIP")
    return resolved_dir


# ---------------------------------------------------------------------------
# Validation  (mirrors tabular: validate_tabular_inputs)
# ---------------------------------------------------------------------------


_IMAGE_TASK_TYPES: frozenset[str] = frozenset(
    {
        "image_classification",
        "image_segmentation",
        "object_detection",
        "video_classification",
        "keypoint_detection",
    }
)

_TEXT_REQUIRED_COLUMNS: dict[str, list[str]] = {
    "text_classification": ["text", "label"],
    "question_answering": ["question", "context", "answer_start", "answer_text"],
    "causal_lm": ["text"],
    "seq2seq_lm": ["input_text", "target_text"],
    "masked_lm": ["text"],
}

_DETECTION_EXTRA_COLUMNS: dict[str, list[str]] = {
    "object_detection": ["boxes", "class_labels"],
    "keypoint_detection": ["keypoints"],
}


def validate_vision_inputs(
    csv_path: Path,
    images_dir: Path,
    filename_column: str,
    label_column: str,
    task_type: str = "image_classification",
) -> str | None:
    """Validate dataset structure for the given task type.

    Returns an error string on failure, or None if everything is valid.
    Mirrors the signature/contract of tabular's ``validate_tabular_inputs``.

    Args:
        csv_path: Path to the labels CSV.
        images_dir: Root directory containing image/audio/video files.
            Unused for pure text tasks.
        filename_column: Column name containing file paths (image/audio tasks).
        label_column: Column name containing labels (classification tasks).
        task_type: One of the supported task type slugs.
    """
    # Audio task — validate audio dir + CSV
    if task_type == "audio_classification":
        if not csv_path.exists():
            return f"Labels CSV not found: {csv_path}"
        if not images_dir.exists():
            return f"Audio directory not found: {images_dir}"
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            return f"Could not read labels CSV: {e}"
        for col, role in [(filename_column, "Filename"), (label_column, "Label")]:
            if col not in df.columns:
                return f"{role} column '{col}' not found in labels CSV"
        return None

    # Text tasks — validate CSV + required columns
    if task_type in _TEXT_REQUIRED_COLUMNS:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            return f"Could not read labels CSV: {e}"
        required = _TEXT_REQUIRED_COLUMNS[task_type]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            return f"Required column(s) missing for {task_type}: {missing_cols}"
        return None

    # Image tasks — existing CSV + image presence checks
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Could not read labels CSV: {e}"

    for col, role in [(filename_column, "Filename"), (label_column, "Label")]:
        if col not in df.columns:
            return f"{role} column '{col}' not found in labels CSV"

    # Detection/segmentation tasks — validate annotation columns
    if task_type in _DETECTION_EXTRA_COLUMNS:
        extra = _DETECTION_EXTRA_COLUMNS[task_type]
        missing_cols = [c for c in extra if c not in df.columns]
        if missing_cols:
            return (
                f"Required annotation column(s) missing for {task_type}: {missing_cols}"
            )

    df = normalize_dataframe_filenames(df, filename_column, csv_path)

    missing = collect_missing_files(df, images_dir, filename_column, label_column)
    if missing:
        preview = missing[:5]
        suffix = "..." if len(missing) > 5 else ""
        return f"Missing {len(missing)} image file(s): {preview}{suffix}"

    return None


def _discover_auxiliary_columns(
    df: pd.DataFrame,
    filename_column: str,
    label_column: str,
    exclude_columns: list[str] | None = None,
) -> list[str]:
    """Auto-discover auxiliary columns in *df*.

    All columns except ``filename_column``, ``label_column``, and any
    columns listed in ``exclude_columns`` are treated as auxiliary features.
    """
    exclude = {filename_column, label_column}
    if exclude_columns:
        exclude.update(exclude_columns)
    return [col for col in df.columns if col not in exclude]


def validate_multimodal_inputs(
    csv_path: Path,
    images_dir: Path,
    filename_column: str,
    label_column: str,
    exclude_columns: list[str] | None = None,
) -> tuple[str | None, list[str]]:
    """Validate dataset structure for multimodal image classification.

    Auto-discovers auxiliary columns (all columns except ``filename_column``,
    ``label_column``, and ``exclude_columns``), validates their presence and
    contents, and checks image file existence.

    Returns:
        (error_string_or_None, auxiliary_columns_list).
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Could not read labels CSV: {e}", []

    for col, role in [(filename_column, "Filename"), (label_column, "Label")]:
        if col not in df.columns:
            return f"{role} column '{col}' not found in labels CSV", []

    auxiliary_columns = _discover_auxiliary_columns(
        df, filename_column, label_column, exclude_columns
    )

    if not auxiliary_columns:
        return (
            "No auxiliary columns found in the CSV. "
            "Use the standard /best_model/ endpoint for image-only classification.",
            [],
        )

    for col in auxiliary_columns:
        if df[col].isnull().all():
            return f"Auxiliary column '{col}' is entirely null", []

    df = normalize_dataframe_filenames(df, filename_column, csv_path)

    missing = collect_missing_files(df, images_dir, filename_column, label_column)
    if missing:
        preview = missing[:5]
        suffix = "..." if len(missing) > 5 else ""
        return f"Missing {len(missing)} image file(s): {preview}{suffix}", []

    return None, auxiliary_columns


# ---------------------------------------------------------------------------
# Training  (mirrors tabular: train_automl)
# ---------------------------------------------------------------------------


async def train_automl(
    csv_path: Path,
    images_dir: Path,
    filename_column: str,
    label_column: str,
    time_budget: int,
    model_size: str,
    workdir: Path,
    task_type: str = "image_classification",
) -> dict:
    """Run Optuna-based vision AutoML and return the result dict."""
    return await run_in_threadpool(
        run_optuna_search,
        task_type=task_type,
        csv_path=csv_path,
        images_dir=images_dir,
        filename_column=filename_column,
        label_column=label_column,
        n_trials=max(1, min(25, time_budget // 60)),
        timeout=time_budget,
        model_size=model_size,
        workdir=workdir,
    )


async def train_automl_multimodal(
    csv_path: Path,
    images_dir: Path,
    filename_column: str,
    label_column: str,
    auxiliary_columns: list[str],
    time_budget: int,
    model_size: str,
    workdir: Path,
) -> dict:
    """Run Optuna-based multimodal vision AutoML and return the result dict."""
    return await run_in_threadpool(
        run_optuna_search,
        task_type="image_classification_multimodal",
        csv_path=csv_path,
        images_dir=images_dir,
        filename_column=filename_column,
        label_column=label_column,
        auxiliary_columns=auxiliary_columns,
        n_trials=max(1, min(25, time_budget // 60)),
        timeout=time_budget,
        model_size=model_size,
        workdir=workdir,
    )


# ---------------------------------------------------------------------------
# Artifact packaging  (mirrors tabular: serialize_and_zip_predictor)
# ---------------------------------------------------------------------------


def deployment_instructions() -> str:
    if jinja_environment is not None:
        return render_template(jinja_environment, "vision_deployment_instructions.md")
    else:
        return "No instructions found"


def vision_data_instructions() -> str:
    """Return the instructions from what kind of data is accepted by the vision AutoML engine"""
    if jinja_environment is not None:
        try:
            return render_template(jinja_environment, "vision_accepted_format.md")
        except Exception as e:
            logger.error(f"Failed to render accepted format instructions: {e}")
            return "No accepted format instructions available"
    else:
        logger.warning("jinja_environment is None, returning default formats")
        return "Ask the agent for help"


def serialize_and_zip_model(workdir: Path) -> Path:
    """
    Package the trained model directory into a ZIP archive.

    Returns the path to the ZIP file.
    Mirrors tabular's ``serialize_and_zip_predictor``.
    """
    model_dir = workdir / "model"
    model_dir.mkdir(exist_ok=True)

    try:
        with open(workdir / "vision_deployment_instructions.md", "w") as f:
            f.write(deployment_instructions())
    except Exception as e:
        logger.debug(f"No deployment_instructions found, {e}")

    zip_base = workdir / "vision_model"
    try:
        shutil.make_archive(str(zip_base), "zip", model_dir)
    except Exception as e:
        logger.error(f"Failed to create zip archive: {e}")
        raise AutoMLSerializationError(f"Failed to zip model: {e}") from e
    zip_path = zip_base.with_suffix(".zip")
    logger.debug("Model artifacts zipped to %s", zip_path)
    return zip_path


# ---------------------------------------------------------------------------
# Leaderboard  (mirrors tabular: convert_leaderboard_safely)
# ---------------------------------------------------------------------------


def convert_leaderboard_safely(optuna_result: dict) -> tuple[dict, str]:
    """
    Extract leaderboard information from an Optuna result dict.

    Returns (leaderboard_json, leaderboard_str) — mirrors the tabular
    ``convert_leaderboard_safely`` signature so main.py can treat both
    pipelines identically.
    """
    leaderboard_json = {
        "best_loss": optuna_result.get("best_value"),
        "best_params": optuna_result.get("best_params"),
        "trials": optuna_result.get("n_trials"),
    }
    leaderboard_str = json.dumps(leaderboard_json, indent=2)
    return leaderboard_json, leaderboard_str


# ---------------------------------------------------------------------------
# Upload payload  (mirrors tabular: build_upload_payload + upload_model)
# ---------------------------------------------------------------------------


def build_upload_payload(
    dataset_id: str,
    dataset_version: str | None,
    metadata: dict,
    task_type: str,
    leaderboard_json: dict,
) -> tuple[str, dict]:
    """Return (model_id, form_data_dict) for the AutoDW upload request."""
    return _core_build_upload_payload(
        dataset_id,
        dataset_version,
        metadata,
        task_type,
        leaderboard_json,
        model_id_prefix="vision_automl",
        name=f"Vision AutoML Model - {dataset_id}",
        description="AutoML trained vision model",
        framework="pytorch",
    )
