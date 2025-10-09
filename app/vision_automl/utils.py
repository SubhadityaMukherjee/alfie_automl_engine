
import logging
import os
import shutil
from pathlib import Path

import pandas as pd
from fastapi import UploadFile
from huggingface_hub import HfApi
from typing import Any

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
        logger.warning("Filename column '%s' not found during normalization", filename_column)
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


def get_num_params_if_available(repo_id: str, revision: str|None = None) -> int|None:
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

def sort_models_by_size(models: list[dict[str, Any]], size_tier: str) -> list[dict[str, Any]]:
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

    return sorted(filtered, key=lambda m: (m.get("num_params") is None, m.get("num_params", 0)))


