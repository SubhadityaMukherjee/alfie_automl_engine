"""Service layer for vision AutoML workflows.

Mirrors the structure of tabular_automl/services.py so both pipelines
share a consistent public API consumed by their respective main.py files.
"""

import datetime
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from fastapi.concurrency import run_in_threadpool
from huggingface_hub import HfApi

from app.vision_automl.ml_engine.trainer import run_optuna_search

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / config
# ---------------------------------------------------------------------------

MAX_MODELS_HF = int(os.getenv("MAX_MODELS_HF", 1))
autodw_url = os.getenv("AUTODW_URL", "http://localhost:8000")


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

        matches = list(images_dir.rglob(filename))
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
# Exceptions
# ---------------------------------------------------------------------------


class DatasetValidationError(ValueError):
    """Raised when the uploaded dataset fails structural validation."""


class AutodwError(Exception):
    """Raised on AutoDW communication failures."""


# ---------------------------------------------------------------------------
# Dataset fetch & extraction  (mirrors tabular: fetch_dataset_metadata +
#                               resolve_download_url + download_dataset)
# ---------------------------------------------------------------------------


def _build_metadata_url(
    autodw_base: str, user_id: str, dataset_id: str, dataset_version: str | None
) -> str:
    url = f"{autodw_base}/datasets/{user_id}/{dataset_id}"
    if dataset_version:
        url = f"{url}/version/{dataset_version}"
    return url


def fetch_dataset_metadata(
    autodw_base: str,
    user_id: str,
    dataset_id: str,
    dataset_version: str | None,
) -> dict:
    """Fetch and return dataset metadata from AutoDW."""
    metadata_url = _build_metadata_url(
        autodw_base, user_id, dataset_id, dataset_version
    )
    logger.debug("Fetching dataset metadata: %s", metadata_url)
    resp = requests.get(metadata_url, timeout=15)
    resp.raise_for_status()
    return resp.json()


def resolve_download_url(
    autodw_base: str,
    user_id: str,
    dataset_id: str,
    dataset_version: str | None,
    metadata: dict,
    split: str | None,
) -> str:
    """Determine the correct dataset download URL, accounting for splits."""
    base_url = _build_metadata_url(autodw_base, user_id, dataset_id, dataset_version)
    download_url = f"{base_url}/download"

    has_split = bool(metadata.get("custom_metadata", {}).get("split"))
    if split and has_split:
        download_url = f"{download_url}?split={split}"
        logger.info(
            "Dataset has splits; downloading '%s' split from: %s", split, download_url
        )
    else:
        if split and not has_split:
            logger.warning(
                "split='%s' was requested but dataset has no splits; "
                "downloading full dataset.",
                split,
            )
        logger.debug("Downloading full dataset ZIP: %s", download_url)

    return download_url


def download_dataset(download_url: str, workdir: Path, original_filename: str) -> Path:
    """Stream-download the ZIP dataset and return its local path."""
    zip_path = workdir / original_filename
    with requests.get(
        download_url,
        stream=True,
        timeout=60,
        headers={"Accept-Encoding": "gzip, deflate"},
    ) as resp:
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    logger.info("Dataset ZIP saved to %s", zip_path)
    return zip_path


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
        raise DatasetValidationError("No valid dataset folder found in ZIP")
    return real_dirs[0]


def _find_csv_file(dataset_root: Path) -> Path:
    csv_candidates = [
        p
        for p in dataset_root.rglob("*")
        if p.is_file() and p.name in ("labels.csv", "metadata.csv")
    ]
    if not csv_candidates:
        raise DatasetValidationError("labels.csv or metadata.csv not found in dataset")
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
        raise DatasetValidationError("images/ directory not found in dataset ZIP")
    return resolved_dir


# ---------------------------------------------------------------------------
# Validation  (mirrors tabular: validate_tabular_inputs)
# ---------------------------------------------------------------------------


def validate_vision_inputs(
    csv_path: Path,
    images_dir: Path,
    filename_column: str,
    label_column: str,
) -> str | None:
    """
    Validate CSV structure and image file presence.

    Returns an error string on failure, or None if everything is valid.
    Mirrors the signature/contract of tabular's ``validate_tabular_inputs``.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Could not read labels CSV: {e}"

    for col, role in [(filename_column, "Filename"), (label_column, "Label")]:
        if col not in df.columns:
            return f"{role} column '{col}' not found in labels CSV"

    df = normalize_dataframe_filenames(df, filename_column, csv_path)

    missing = collect_missing_files(df, images_dir, filename_column, label_column)
    if missing:
        preview = missing[:5]
        suffix = "..." if len(missing) > 5 else ""
        return f"Missing {len(missing)} image file(s): {preview}{suffix}"

    return None


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
) -> dict:
    """Run Optuna-based vision AutoML and return the result dict."""
    return await run_in_threadpool(
        run_optuna_search,
        csv_path=csv_path,
        images_dir=images_dir,
        filename_column=filename_column,
        label_column=label_column,
        n_trials=min(25, time_budget // 60),
        timeout=time_budget,
        model_size=model_size,
        workdir=workdir,
    )


# ---------------------------------------------------------------------------
# Artifact packaging  (mirrors tabular: serialize_and_zip_predictor)
# ---------------------------------------------------------------------------


def serialize_and_zip_model(result: dict, workdir: Path) -> Path:
    """
    Package the trained model directory into a ZIP archive.

    Returns the path to the ZIP file.
    Mirrors tabular's ``serialize_and_zip_predictor``.
    """
    model_dir = workdir / "model"
    model_dir.mkdir(exist_ok=True)
    zip_base = workdir / "vision_model"
    shutil.make_archive(str(zip_base), "zip", model_dir)
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
    """
    Return (model_id, form_data_dict) for the AutoDW upload request.

    Mirrors tabular's ``build_upload_payload``.
    """
    model_id = (
        f"vision_automl_{dataset_id}_{int(datetime.datetime.utcnow().timestamp())}"
    )
    data = {
        "model_id": model_id,
        "name": f"Vision AutoML Model - {dataset_id}",
        "description": "AutoML trained vision model",
        "framework": "pytorch",
        "model_type": task_type,
        "training_dataset": str(dataset_id),
        "training_dataset_version": dataset_version or metadata.get("version", "v1"),
        "leaderboard": json.dumps(leaderboard_json),
    }
    return model_id, data


def upload_model(
    upload_url: str,
    zip_path: Path,
    payload: dict,
    task_id: str | None,
) -> requests.Response:
    """Upload the zipped model to AutoDW and return the raw response."""
    headers = {"X-Task-ID": task_id} if task_id else {}
    with open(zip_path, "rb") as f:
        files = {"file": (zip_path.name, f, "application/octet-stream")}
        logger.debug("Uploading vision model to %s", upload_url)
        return requests.post(
            upload_url, headers=headers, files=files, data=payload, timeout=120
        )
