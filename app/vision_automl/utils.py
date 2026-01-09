import datetime
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, UploadFile
from fastapi.concurrency import run_in_threadpool
from huggingface_hub import HfApi

from app.vision_automl.ml_engine.trainer import run_optuna_search
from app.vision_automl.utils import (collect_missing_files,
                                     normalize_dataframe_filenames,
                                     resolve_images_root)

logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())


app = FastAPI()

VISION_AUTOML_PORT = os.getenv("VISION_AUTOML_PORT", "http://localhost:8002")
MAX_MODELS_HF = int(os.getenv("MAX_MODELS_HF", 1))

autodw_port_url = os.getenv("AUTODW_DATASETS_PORT", 8000)
autodw_url = os.getenv("AUTODW_URL", "http://localhost:8000")
# -------------------------------------------------
# Helpers
# -------------------------------------------------
logger = logging.getLogger(__name__)


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
    df: pd.DataFrame,
    images_dir: Path,
    filename_column: str,
    label_column: str,
) -> list[str]:
    """Return list of filenames that do not exist in the extracted images."""
    logger.info("Checking for missing image files...")
    missing_files: list[str] = []
    for _, row in df.iterrows():
        raw_filename = str(row[filename_column])
        label = str(row[label_column])
        basename = os.path.basename(raw_filename.replace("\\", "/"))

        candidates = [
            images_dir / label / basename,
            images_dir / basename,
            images_dir / raw_filename,
        ]

        if any(path.exists() for path in candidates):
            continue

        try:
            found_any = next(images_dir.rglob(basename), None) is not None
        except Exception as e:
            logger.debug("Error searching recursively for %s: %s", basename, e)
            found_any = False

        if not found_any:
            missing_files.append(raw_filename)

    logger.info("Missing %d files", len(missing_files))
    return missing_files


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
    """Search HF for PyTorch models and annotate with estimated parameters."""
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


def save_upload(upload_file: UploadFile, destination: Path) -> None:
    """
    Save a FastAPI UploadFile to a destination path.

    Args:
        upload_file (UploadFile): The uploaded file from a multipart/form-data request.
        destination (Path): The path (including filename) where the file should be saved.

    Raises:
        Exception: If the file cannot be saved.
    """
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    except Exception as e:
        raise RuntimeError(f"Failed to save uploaded file {upload_file.filename}: {e}")


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


class DatasetValidationError(ValueError):
    """Custom exception for dataset validation failures."""


class AutodwError(Exception):
    """Custom exception for AutoDW communication failures."""


async def _fetch_and_extract_dataset(
    user_id: str, dataset_id: str, dataset_version: str | None
) -> tuple[Path, Path]:
    """Download and extract ZIP dataset from AutoDW."""
    autodw_base = autodw_url
    metadata_url = f"{autodw_base}/datasets/{user_id}/{dataset_id}"
    if dataset_version:
        metadata_url += f"/version/{dataset_version}"

    metadata = _fetch_json(metadata_url, timeout=15)
    _validate_zip_dataset(metadata)

    download_url = f"{metadata_url}/download"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        zip_path = await _download_zip(
            download_url, tmp_path / metadata["original_filename"]
        )

        extract_dir = tmp_path / metadata["original_filename"].replace(".zip", "")
        extract_dir.mkdir(exist_ok=True)
        shutil.unpack_archive(zip_path, extract_dir)

        dataset_root = _find_valid_dataset_root(extract_dir)
        csv_path = _find_csv_file(dataset_root)
        images_dir = _find_or_resolve_images_dir(dataset_root, csv_path)

        return csv_path, images_dir


def _validate_zip_dataset(metadata: dict) -> None:
    """Validate dataset metadata for ZIP format."""
    if metadata.get("file_type") != "zip":
        raise DatasetValidationError("Vision AutoML requires a ZIP dataset")


async def _download_zip(url: str, zip_path: Path) -> Path:
    """Download ZIP file with streaming and gzip support."""
    with requests.get(
        url, stream=True, timeout=60, headers={"Accept-Encoding": "gzip, deflate"}
    ) as resp:
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    return zip_path


def _find_valid_dataset_root(extract_dir: Path) -> Path:
    """Find first valid dataset folder (skip __MACOSX, hidden dirs)."""
    real_dirs = [
        child
        for child in extract_dir.iterdir()
        if child.is_dir()
        and child.name != "__MACOSX"
        and not child.name.startswith(".")
    ]
    if not real_dirs:
        raise DatasetValidationError("No valid dataset folder found")
    return real_dirs[0]


def _find_csv_file(dataset_root: Path) -> Path:
    """Find labels.csv or metadata.csv using rglob."""
    csv_candidates = [
        p
        for p in dataset_root.rglob("*")
        if p.is_file() and p.name in ("labels.csv", "metadata.csv")
    ]
    if not csv_candidates:
        raise DatasetValidationError("labels.csv or metadata.csv not found")
    return csv_candidates[0]


def _find_or_resolve_images_dir(dataset_root: Path, csv_path: Path) -> Path:
    """Find images dir or fallback to csv.parent/images."""
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


def _validate_dataset_structure(
    csv_path: Path, images_dir: Path, filename_column: str, label_column: str
) -> tuple[Path, Path]:
    """Load CSV and validate structure and image files."""
    df = pd.read_csv(csv_path)

    _validate_csv_columns(df, filename_column, label_column)
    df = normalize_dataframe_filenames(df, filename_column, csv_path)

    missing_files = collect_missing_files(df, images_dir, filename_column, label_column)
    if missing_files:
        preview = missing_files[:5]
        raise DatasetValidationError(
            f"Missing {len(missing_files)} image files: {preview}{'...' if len(missing_files) > 5 else ''}"
        )

    return csv_path, images_dir


def _validate_csv_columns(
    df: pd.DataFrame, filename_column: str, label_column: str
) -> None:
    """Validate required CSV columns exist."""
    for col, name in [(filename_column, "Filename"), (label_column, "Label")]:
        if col not in df.columns:
            raise DatasetValidationError(
                f"{name} column '{col}' not found in labels.csv"
            )


async def _run_automl_optimization(
    csv_path: Path,
    images_dir: Path,
    filename_column: str,
    label_column: str,
    time_budget: int,
) -> dict:
    """Run Optuna optimization in threadpool."""
    return await run_in_threadpool(
        run_optuna_search,
        csv_path=csv_path,
        images_dir=images_dir,
        filename_column=filename_column,
        label_column=label_column,
        n_trials=25,
        timeout=time_budget,
    )


def _package_model_artifacts(result: dict) -> Path:
    """Package trained model artifacts into ZIP."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        model_dir = tmp_path / "vision_model"
        model_dir.mkdir(exist_ok=True)

        # Note: Assumes run_optuna_search saves model to current working dir
        # You may need to copy files from result or specify model save path

        model_zip_path = tmp_path / "vision_model.zip"
        shutil.make_archive(str(model_zip_path).replace(".zip", ""), "zip", model_dir)
        return model_zip_path


def _prepare_model_metadata(
    dataset_id: str, optuna_result: dict, task_type: str
) -> dict:
    """Prepare metadata for model upload."""
    return {
        "model_id": f"vision_automl_{dataset_id}_{int(datetime.utcnow().timestamp())}",
        "name": f"Vision AutoML Model - {dataset_id}",
        "description": "AutoML trained vision model",
        "framework": "pytorch",
        "model_type": task_type,
        "training_dataset": str(dataset_id),
        "leaderboard": json.dumps(
            {
                "best_loss": optuna_result["best_value"],
                "best_params": optuna_result["best_params"],
                "trials": optuna_result["n_trials"],
            }
        ),
    }


def _upload_model_to_autodw(
    model_zip_path: Path, metadata: dict, task_id: str | None = None
) -> dict:
    """Upload model ZIP to AutoDW."""
    upload_url = f"{autodw_url}/ai-models/upload/single/{metadata['model_id']}"

    headers = {"X-Task-ID": task_id} if task_id else {}
    with open(model_zip_path, "rb") as f:
        files = {"file": (model_zip_path.name, f, "application/octet-stream")}
        resp = requests.post(
            upload_url, headers=headers, files=files, data=metadata, timeout=120
        )
        resp.raise_for_status()

    logger.info("Model uploaded successfully: %s", metadata["model_id"])
    metadata["model_id"] = resp.json().get("model_id", metadata["model_id"])
    return metadata


def _fetch_json(url: str, timeout: float) -> dict:
    """Helper to fetch and validate JSON response."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()
