"""Shared service helpers for AutoML workflows.

Common dataset download, metadata fetching, model upload, and payload
building logic used by both vision and tabular pipelines.
"""

import datetime
import json
import logging
from pathlib import Path
from typing import Any

import requests

from app.core.exceptions import (
    AutoDWDownloadError,
    AutoDWUploadError,
    AutoMLDataError,
    AutoMLSerializationError,
    AutoMLValidationError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset metadata & download
# ---------------------------------------------------------------------------


def metadata_indicates_zip_download(metadata: dict) -> bool:
    """Return True when AutoDW will serve this dataset as a ZIP download.

    AutoDW's ``file_type`` is often the *contents* type after indexing a
    folder upload (e.g. ``"csv, jpg"`` or ``"jpg, csv"``), even though
    ``GET .../download`` still returns a ZIP. Treat a dataset as ZIP-backed
    when any of these hold:

    * ``file_type`` is exactly ``zip``
    * ``original_filename`` ends with ``.zip``
    * ``is_folder`` is true (folder/media uploads are packaged as ZIP on download)
    """
    if not isinstance(metadata, dict):
        return False

    file_type = str(metadata.get("file_type") or "").strip().lower()
    if file_type == "zip":
        return True

    original = str(metadata.get("original_filename") or "").strip().lower()
    if original.endswith(".zip"):
        return True

    return bool(metadata.get("is_folder"))


def build_metadata_url(
    autodw_base: str, user_id: str, dataset_id: str, dataset_version: str | None
) -> str:
    """Assemble the AutoDW metadata URL for a dataset, including version if given."""
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

    if not autodw_base or not isinstance(autodw_base, str):
        raise AutoMLValidationError("autodw_base must be a non-empty string")

    if not user_id or not isinstance(user_id, str):
        raise AutoMLValidationError("user_id must be a non-empty string")

    if not dataset_id or not isinstance(dataset_id, str):
        raise AutoMLValidationError("dataset_id must be a non-empty string")

    metadata_url = build_metadata_url(autodw_base, user_id, dataset_id, dataset_version)
    logger.debug("Fetching dataset metadata: %s", metadata_url)

    try:
        resp = requests.get(metadata_url, timeout=15)
        resp.raise_for_status()
    except requests.Timeout:
        logger.error("Timeout fetching metadata from %s", metadata_url)
        raise AutoDWDownloadError(
            "Timeout fetching dataset metadata from AutoDW"
        ) from None
    except requests.ConnectionError as e:
        logger.error("Connection error fetching metadata: %s", e)
        raise AutoDWDownloadError(f"Failed to connect to AutoDW: {e}") from e
    except requests.HTTPError as e:
        logger.error("HTTP error fetching metadata: %s", e)
        raise AutoDWDownloadError(f"AutoDW returned HTTP error: {e}") from e
    except Exception as e:
        logger.error("Unexpected error fetching metadata: %s", e)
        raise AutoDWDownloadError(f"Unexpected error fetching metadata: {e}") from e

    try:
        metadata = resp.json()
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON response from AutoDW: %s", e)
        raise AutoDWDownloadError(f"Invalid JSON response from AutoDW: {e}") from e

    if not isinstance(metadata, dict):
        logger.error("Metadata is not a dict: %s", type(metadata))
        raise AutoDWDownloadError(
            f"Invalid metadata format: expected dict, got {type(metadata)}"
        )

    return metadata


def resolve_download_url(
    autodw_base: str,
    user_id: str,
    dataset_id: str,
    dataset_version: str | None,
    metadata: dict,
    split: str | None,
) -> str:
    """Determine the correct dataset download URL, accounting for splits."""
    base_url = build_metadata_url(autodw_base, user_id, dataset_id, dataset_version)
    download_url = f"{base_url}/download"

    has_split = bool(metadata.get("custom_metadata", {}).get("split"))
    effective_split = (
        split if (has_split and split in ("train", "test", "drift")) else None
    )

    if effective_split:
        download_url = f"{download_url}?split={effective_split}"
        logger.info(
            "Dataset has splits; downloading '%s' split from: %s",
            effective_split,
            download_url,
        )
    else:
        if split and not has_split:
            logger.warning(
                "split='%s' was requested but dataset has no splits; "
                "downloading full dataset.",
                split,
            )
        logger.debug("Downloading full dataset: %s", download_url)

    return download_url


def download_dataset(download_url: str, dest_path: Path) -> Path:
    """Stream-download a dataset file to *dest_path*. Returns the path."""

    if not download_url or not isinstance(download_url, str):
        raise AutoMLValidationError("download_url must be a non-empty string")

    if dest_path.parent and not dest_path.parent.exists():
        dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # (connect, read) tuple: dataset ZIPs can be large, so give the
        # read phase a generous budget while connect failures stay fast.
        with requests.get(download_url, stream=True, timeout=(10, 720)) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
    except requests.Timeout as e:
        logger.error("Timeout downloading from %s", download_url)
        raise AutoDWDownloadError("Timeout downloading dataset") from e
    except requests.ConnectionError as e:
        logger.error("Connection error downloading dataset: %s", e)
        raise AutoDWDownloadError(f"Failed to connect to download URL: {e}") from e
    except requests.HTTPError as e:
        logger.error("HTTP error downloading dataset: %s", e)
        raise AutoDWDownloadError(f"HTTP error downloading dataset: {e}") from e
    except requests.RequestException as e:
        logger.error("Request error downloading dataset: %s", e)
        raise AutoDWDownloadError(f"Request error downloading dataset: {e}") from e
    except OSError as e:
        logger.error("Failed to write dataset to %s: %s", dest_path, e)
        raise AutoDWDownloadError(f"Failed to save dataset file: {e}") from e
    except Exception as e:
        logger.error("Unexpected error downloading dataset: %s", e)
        raise AutoDWDownloadError(f"Unexpected error downloading dataset: {e}") from e

    if not dest_path.exists():
        raise AutoDWDownloadError(
            f"Download completed but file not created at {dest_path}"
        )

    if dest_path.stat().st_size == 0:
        raise AutoDWDownloadError(f"Downloaded file is empty: {dest_path}")

    logger.info("Dataset saved to %s", dest_path)
    return dest_path


# ---------------------------------------------------------------------------
# Model upload
# ---------------------------------------------------------------------------


def upload_model(
    upload_url: str,
    zip_path: Path,
    payload: dict,
    task_id: str | None,
) -> requests.Response:
    """Upload the zipped model to AutoDW. Returns the raw response."""

    if not upload_url or not isinstance(upload_url, str):
        raise AutoMLValidationError("upload_url must be a non-empty string")

    if not zip_path.exists():
        raise AutoMLDataError(f"Zip file not found: {zip_path}")

    if not isinstance(payload, dict) or not payload:
        raise AutoMLValidationError("payload must be a non-empty dict")

    headers = {"X-Task-ID": task_id} if task_id else {}
    if task_id:
        logger.debug("Including X-Task-ID header: %s", task_id)

    try:
        with open(zip_path, "rb") as f:
            files = {"file": (zip_path.name, f, "application/octet-stream")}
            logger.debug("Uploading model to %s", upload_url)
            return requests.post(
                upload_url, headers=headers, files=files, data=payload, timeout=120
            )
    except requests.Timeout as e:
        logger.error("Timeout uploading to %s", upload_url)
        raise AutoDWUploadError("Timeout uploading model to AutoDW") from e
    except requests.ConnectionError as e:
        logger.error("Connection error uploading model: %s", e)
        raise AutoDWUploadError(f"Failed to connect to upload URL: {e}") from e
    except requests.HTTPError as e:
        logger.error("HTTP error uploading model: %s", e)
        raise AutoDWUploadError(f"HTTP error uploading model: {e}") from e
    except requests.RequestException as e:
        logger.error("Request error uploading model: %s", e)
        raise AutoDWUploadError(f"Request error uploading model: {e}") from e
    except OSError as e:
        logger.error("Failed to read zip file %s: %s", zip_path, e)
        raise AutoDWUploadError(f"Failed to read zip file: {e}") from e
    except Exception as e:
        logger.error("Unexpected error uploading model: %s", e)
        raise AutoDWUploadError(f"Unexpected error uploading model: {e}") from e


def uploaded_model_version(upload_resp: requests.Response) -> str:
    """Extract the AutoDW-assigned version from a *successful* upload response.

    AutoDW reports the created model version in its JSON body; anything
    else (unexpected content type, malformed JSON) falls back to ``"v1"``
    so a successful upload never fails at the parse step. Call this only
    after the response's status code has been checked.
    """
    try:
        upload_data = upload_resp.json()
        version = upload_data.get("version")
    except (ValueError, AttributeError):
        logger.warning(
            "Upload response carried no usable JSON body; defaulting version to v1."
        )
        return "v1"
    return str(version) if version else "v1"


# ---------------------------------------------------------------------------
# Upload payload
# ---------------------------------------------------------------------------


def build_upload_payload(
    dataset_id: str,
    dataset_version: str | None,
    metadata: dict,
    task_type: str,
    leaderboard_json: list | dict,
    *,
    model_id_prefix: str = "automl",
    name: str | None = None,
    description: str = "AutoML trained model",
    framework: str = "sklearn",
    extra_fields: dict | None = None,
    metrics: dict | None = None,
    algorithm: str | None = None,
) -> tuple[str, dict]:
    """Return ``(model_id, form_data_dict)`` for the AutoDW upload request.

    When ``metrics`` is provided, maps common training keys onto AutoDW's
    scalar metadata fields:

    * ``train_acc`` → ``training_accuracy``
    * ``val_acc`` → ``validation_accuracy``
    * ``test_acc`` → ``test_accuracy``
    * ``train_loss`` → ``training_loss``

    ``algorithm`` is forwarded as-is when set (best model id / name).
    Fields are only included when the value is present and parseable, so
    partial metrics never break an upload.
    """

    if not dataset_id or not isinstance(dataset_id, str):
        raise AutoMLValidationError("dataset_id must be a non-empty string")

    if not task_type or not isinstance(task_type, str):
        raise AutoMLValidationError("task_type must be a non-empty string")

    model_id = f"{model_id_prefix}_{dataset_id}_{int(datetime.datetime.now(datetime.timezone.utc).timestamp())}"

    try:
        leaderboard_str = json.dumps(leaderboard_json)
    except TypeError as e:
        logger.error("Failed to serialize leaderboard_json: %s", e)
        raise AutoMLSerializationError(f"Failed to serialize leaderboard: {e}") from e

    version = dataset_version or metadata.get("version", "v1")
    if not isinstance(version, str):
        version = "v1"

    data: dict[str, Any] = {
        "model_id": model_id,
        "name": name or f"AutoML Model - {model_id}",
        "description": description,
        "framework": framework,
        "model_type": task_type,
        "training_dataset": str(dataset_id),
        "training_dataset_version": version,
        "leaderboard": leaderboard_str,
    }

    if algorithm is not None:
        data["algorithm"] = algorithm

    metrics = metrics or {}
    # Optional AutoDW scalar fields — include only when present/parseable.
    if metrics.get("train_acc") is not None:
        data["training_accuracy"] = float(metrics["train_acc"])
    if metrics.get("val_acc") is not None:
        data["validation_accuracy"] = float(metrics["val_acc"])
    if metrics.get("test_acc") is not None:
        data["test_accuracy"] = float(metrics["test_acc"])
    if metrics.get("train_loss") is not None:
        data["training_loss"] = float(metrics["train_loss"])

    if extra_fields:
        data.update(extra_fields)

    return model_id, data


def extract_optuna_metrics_for_autodw(
    optuna_result: dict | None,
) -> tuple[dict, str | None]:
    """Map a generic-engine Optuna result onto AutoDW's scalar metric fields.

    Returns ``(metrics, algorithm)`` where ``metrics`` uses the internal keys
    expected by ``build_upload_payload`` (``val_acc`` / ``test_acc`` /
    ``train_loss``) and ``algorithm`` is the winning model id from the search
    params. Both degrade to empty/``None`` when the result carries nothing —
    metadata extraction must never fail an upload.
    """
    result = optuna_result or {}
    best_metrics = result.get("best_metrics") or {}
    best_params = result.get("best_params") or {}
    if not isinstance(best_metrics, dict):
        best_metrics = {}
    if not isinstance(best_params, dict):
        best_params = {}

    metrics: dict = {
        "train_acc": to_float_or_none(best_metrics.get("train_acc")),
        "val_acc": to_float_or_none(
            best_metrics.get("val_acc", best_metrics.get("accuracy"))
        ),
        "test_acc": to_float_or_none(best_metrics.get("test_acc")),
        "train_loss": to_float_or_none(best_metrics.get("train_loss")),
    }
    algorithm = best_params.get("model_id")
    return metrics, str(algorithm) if algorithm is not None else None


def to_float_or_none(value) -> float | None:
    """Best-effort float conversion for numpy / AutoGluon / torch scalars."""
    try:
        if value is None:
            return None
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except (TypeError, ValueError):
        return None
