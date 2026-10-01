#!/usr/bin/env python3
"""
Kafka Consumer for AutoML Trigger Events (Tabular + Vision) — v5

Listens to automl-trigger-events (from Agentic Core). ``task_category`` selects
the pipeline:
- ``tabular`` -> POST /automl/tabular/best_model/
- ``vision``  -> POST /automl/vision/best_model/

v5 keeps every v3 operational perk (consumer-group parallelism via multiple
replicas, task_id claim/idempotency, 429 busy retries, long poll/read timeouts,
task-type normalization, DW metric enrichment) and targets the *unified*
AutoML engine on a single host/port (default :8001).

Usage:
  KAFKA_BOOTSTRAP_SERVERS=localhost:9092 python kafka_automl_consumer_example_v5.py

  API_BASE=http://localhost:8000 AUTOML_ENGINE_HOST=localhost AUTOML_ENGINE_PORT=8001 \
  python kafka_automl_consumer_example_v5.py

  Scale workers: run multiple processes with the same KAFKA_CONSUMER_GROUP;
  partitions + /jobs/automl/claim keep work from double-running.

  Note: Runs OUTSIDE Docker by default; use host.docker.internal when the
  consumer is in Docker and the engine is on the host.
"""

import os
import re
import asyncio
import json
import logging
import random
from datetime import datetime, timezone
from io import StringIO
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
import requests
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv, find_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("kafka_automl_consumer")

load_dotenv(find_dotenv())

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "alfie.iti.gr:9092")
KAFKA_AUTOML_TRIGGER_TOPIC = os.getenv(
    "KAFKA_AUTOML_TRIGGER_TOPIC", "automl-trigger-events"
)
KAFKA_AUTOML_COMPLETE_TOPIC = os.getenv(
    "KAFKA_AUTOML_COMPLETE_TOPIC", "automl-complete-events"
)
KAFKA_CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "automl-consumer")

# Data Warehouse API configuration
# Default to localhost (for running outside Docker)
# Can be overridden with API_BASE (full URL) or DW_HOST+DW_PORT
DW_HOST = os.getenv("DW_HOST", "localhost")
DW_PORT = os.getenv("DW_PORT", "8000")
# Support both API_BASE (full URL) and DW_HOST+DW_PORT (components)
if os.getenv("API_BASE"):
    API_BASE = os.getenv("API_BASE")
else:
    API_BASE = f"http://{DW_HOST}:{DW_PORT}"

# Unified AutoML engine (single process, one port; paths under /automl/...)
# Prefer AUTOML_ENGINE_HOST / AUTOML_ENGINE_PORT. Legacy TABULAR_/VISION_HOST+PORT
# still work as fallbacks so existing compose env keeps running.
AUTOML_ENGINE_HOST = os.getenv(
    "AUTOML_ENGINE_HOST",
    os.getenv("TABULAR_AUTOML_HOST", os.getenv("VISION_AUTOML_HOST", "localhost")),
)
AUTOML_ENGINE_PORT = os.getenv(
    "AUTOML_ENGINE_PORT",
    os.getenv("TABULAR_AUTOML_PORT", os.getenv("VISION_AUTOML_PORT", "8001")),
)
AUTOML_ENGINE_URL = f"http://{AUTOML_ENGINE_HOST}:{AUTOML_ENGINE_PORT}"
AUTOML_TABULAR_BEST_MODEL_URL = f"{AUTOML_ENGINE_URL}/automl/tabular/best_model/"
AUTOML_VISION_BEST_MODEL_URL = f"{AUTOML_ENGINE_URL}/automl/vision/best_model/"

# Log configuration at module load (so we see which host/port are used)
logger.info("Configuration:")
logger.info(f"  Data Warehouse API: {API_BASE} (host: {DW_HOST}, port: {DW_PORT})")
logger.info(f"  AutoML engine: {AUTOML_ENGINE_URL}")
logger.info(f"  AutoML Tabular: {AUTOML_TABULAR_BEST_MODEL_URL}")
logger.info(f"  AutoML Vision: {AUTOML_VISION_BEST_MODEL_URL}")
if AUTOML_ENGINE_HOST == "localhost" and "kafka" in os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS", ""
):
    logger.warning(
        "  When running in Docker, set AUTOML_ENGINE_HOST=host.docker.internal "
        "so the container can reach the AutoML engine on the host"
    )


# Large dataset downloads from DW (ZIP) can take many minutes; default 2h, override with env.
DW_DOWNLOAD_TIMEOUT_SECONDS = int(
    os.getenv("AUTOML_DW_DOWNLOAD_TIMEOUT_SECONDS", str(2 * 60 * 60))
)

AUTOML_429_MAX_WAIT_SECONDS = int(
    os.getenv("AUTOML_429_MAX_WAIT_SECONDS", "900")
)  # 15 min
AUTOML_429_BASE_SLEEP_SECONDS = float(os.getenv("AUTOML_429_BASE_SLEEP_SECONDS", "5"))
AUTOML_429_MAX_SLEEP_SECONDS = float(os.getenv("AUTOML_429_MAX_SLEEP_SECONDS", "60"))


def compute_request_timeouts(time_budget_seconds: int) -> tuple[float, float]:
    """
    Compute robust timeouts for long AutoML jobs.
    - connect timeout: short (network)
    - read timeout: long (download + unzip + dataloaders + training + upload)

    The AutoML HTTP endpoint may block until training + DW model upload completes, which can be
    far longer than `time_budget` (e.g. vision with large ZIPs). Do not tie read timeout to
    time_budget alone.

    Override with AUTOML_HTTP_READ_TIMEOUT_SECONDS (seconds), e.g. 86400 for 24h.
    """
    try:
        tb = int(time_budget_seconds)
    except Exception:
        tb = 0
    env_read = os.getenv("AUTOML_HTTP_READ_TIMEOUT_SECONDS", "").strip()
    if env_read:
        read_timeout = max(60, int(env_read))
    else:
        # time_budget is a training budget hint, not an upper bound on wall-clock time.
        # Default: at least 6h, or 10x budget + 2h buffer (whichever is larger), capped at 48h.
        read_timeout = max(6 * 60 * 60, tb * 10 + 2 * 60 * 60)
        read_timeout = min(read_timeout, 48 * 60 * 60)
    return (10, float(read_timeout))


# Helpful startup diagnostics (esp. vision + large ZIP training)
_c_demo, _r_demo = compute_request_timeouts(10)
logger.info(
    "AutoML HTTP timeouts: DW download read timeout=%ss, example AutoML read timeout (time_budget=10s)=%ss "
    "(override read with AUTOML_HTTP_READ_TIMEOUT_SECONDS; override DW download with AUTOML_DW_DOWNLOAD_TIMEOUT_SECONDS)",
    DW_DOWNLOAD_TIMEOUT_SECONDS,
    int(_r_demo),
)


def normalize_tabular_task_type(task_type: str | None) -> str | None:
    """
    AutoML Tabular service vNext expects:
      - tabular_classification
      - tabular_regression
      - tabular_time_series

    For backward compatibility, map legacy slugs commonly emitted by orchestrators.
    """
    if task_type is None:
        return None
    t = str(task_type).strip().lower()
    legacy_map = {
        "classification": "tabular_classification",
        "tabular_classification": "tabular_classification",
        "regression": "tabular_regression",
        "tabular_regression": "tabular_regression",
        "time_series": "tabular_time_series",
        "timeseries": "tabular_time_series",
        "tabular_time_series": "tabular_time_series",
    }
    return legacy_map.get(t, t)


def normalize_vision_task_type(task_type: str | None) -> str | None:
    """
    AutoML Vision often uses modality-specific slugs (e.g. image_classification).
    Keep backward compatibility with orchestrators that still send generic 'classification'.
    """
    if task_type is None:
        return None
    t = str(task_type).strip().lower()
    legacy_map = {
        # generic -> canonical (most common)
        "classification": "image_classification",
        # already canonical / accepted
        "image_classification": "image_classification",
        "image_segmentation": "image_segmentation",
        "object_detection": "object_detection",
        "video_classification": "video_classification",
        "keypoint_detection": "keypoint_detection",
        "audio_classification": "audio_classification",
        "text_classification": "text_classification",
    }
    return legacy_map.get(t, t)


async def send_automl_failure_to_kafka(
    producer: AIOKafkaProducer,
    task_id: str,
    user_id: str,
    dataset_id: str,
    error_message: str,
    dataset_version: str = None,
) -> None:
    """Send automl-complete event with failure so orchestrator and Task Manager can continue the flow."""
    if not producer or not task_id:
        return
    try:
        payload = {
            "task_id": task_id,
            "event_type": "automl-complete",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "output": None,
            "failure": {
                "error_type": "AutoMLError",
                "error_message": error_message or "AutoML training failed",
            },
        }
        await producer.send_and_wait(
            KAFKA_AUTOML_COMPLETE_TOPIC,
            value=payload,
            key=task_id,
        )
        logger.info(
            f"Sent AutoML failure event to {KAFKA_AUTOML_COMPLETE_TOPIC}: {error_message[:200]}"
        )
    except Exception as e:
        logger.error(
            f"Failed to send AutoML failure event to Kafka: {e}", exc_info=True
        )


def claim_task_id(
    task_id: str, *, user_id: str, dataset_id: str, dataset_version: str | None
) -> bool:
    """
    Idempotency guard: claim a task_id in the DW so only one consumer replica runs it.
    Returns True when claimed; False when already claimed by another worker.
    """
    if not task_id:
        return True
    url = f"{API_BASE}/jobs/automl/claim"
    payload = {
        "task_id": task_id,
        "user_id": user_id,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 409:
            logger.info(
                "Task already claimed (skipping): task_id=%s user=%s dataset=%s",
                task_id,
                user_id,
                dataset_id,
            )
            return False
        r.raise_for_status()
        return True
    except Exception as e:
        # Fail-open: if claim endpoint is unavailable, we still proceed (keeps backward compatibility),
        # but duplicates may occur. Log loudly.
        logger.warning("Could not claim task_id=%s (proceeding anyway): %s", task_id, e)
        return True


async def _post_with_429_retry(
    url: str, *, data: dict, headers: dict | None
) -> requests.Response:
    """
    POST with bounded retries on 429 (busy worker).
    We intentionally do NOT treat 429 as terminal failure; we retry for a short window so
    parallel requests queue instead of generating immediate automl-complete failures.
    """
    started = __import__("time").time()
    attempt = 0
    while True:
        attempt += 1
        try:
            connect_timeout, read_timeout = compute_request_timeouts(
                int(data.get("time_budget") or 0)
            )
            r = requests.post(
                url, data=data, headers=headers, timeout=(connect_timeout, read_timeout)
            )
            if r.status_code != 429:
                r.raise_for_status()
                return r

            elapsed = __import__("time").time() - started
            if elapsed >= AUTOML_429_MAX_WAIT_SECONDS:
                raise requests.HTTPError(
                    f"429 after waiting {int(elapsed)}s", response=r
                )

            # Exponential backoff + jitter
            sleep_s = min(
                AUTOML_429_MAX_SLEEP_SECONDS,
                AUTOML_429_BASE_SLEEP_SECONDS * (2 ** min(attempt - 1, 6)),
            )
            sleep_s = sleep_s * (0.8 + 0.4 * random.random())
            detail = ""
            try:
                detail = r.text[:200]
            except Exception:
                pass
            logger.warning(
                "AutoML service busy (429). Retrying in %.1fs (attempt=%d elapsed=%ds). %s",
                sleep_s,
                attempt,
                int(elapsed),
                detail,
            )
            await asyncio.sleep(sleep_s)
            continue
        except Exception:
            raise


def fetch_dataset_metadata(user_id: str, dataset_id: str, version: str = None) -> dict:
    """Fetch dataset metadata from the Data Warehouse API (specific version or latest)"""
    if version:
        url = f"{API_BASE}/datasets/{user_id}/{dataset_id}/version/{version}"
    else:
        url = f"{API_BASE}/datasets/{user_id}/{dataset_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def read_csv_with_encoding(file_data: bytes) -> pd.DataFrame:
    """
    Read CSV bytes with robust fallbacks.

    Uses multiple encodings and the Python parser to reduce tokenizer failures on malformed rows.
    """
    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1", "utf-16"]

    for encoding in encodings:
        try:
            df = pd.read_csv(
                BytesIO(file_data),
                encoding=encoding,
                engine="python",
                on_bad_lines="skip",
            )
            logger.info(f"Successfully read CSV with encoding: {encoding}")
            return df
        except Exception:
            continue

    # Last resort: ignore undecodable chars and skip malformed lines
    try:
        df = pd.read_csv(
            BytesIO(file_data),
            encoding="utf-8",
            encoding_errors="ignore",
            engine="python",
            on_bad_lines="skip",
        )
        logger.warning(
            "Read CSV with lenient parser (encoding_errors='ignore', on_bad_lines='skip')"
        )
        return df
    except Exception as e:
        logger.error(f"Failed to read CSV with all encodings: {e}")
        raise


def load_tabular_dataframe(file_bytes: bytes) -> pd.DataFrame:
    """
    Load tabular data from bytes.

    Supports:
    - raw CSV bytes
    - ZIP bytes containing one or more CSV files (uses first CSV)
    """
    import zipfile

    # If this is a ZIP (folder download), do NOT try to parse as CSV first.
    # Pandas can sometimes "successfully" parse ZIP bytes as a 1-column garbage CSV, which then
    # makes downstream validation think the target column is missing.
    if (
        isinstance(file_bytes, (bytes, bytearray))
        and len(file_bytes) >= 2
        and file_bytes[:2] == b"PK"
    ):
        try:
            with zipfile.ZipFile(BytesIO(file_bytes), "r") as zf:
                csv_names = [
                    n
                    for n in zf.namelist()
                    if n.lower().endswith(".csv") and not n.startswith("__")
                ]
                if not csv_names:
                    raise ValueError("ZIP does not contain any CSV files")
                with zf.open(csv_names[0]) as f:
                    return read_csv_with_encoding(f.read())
        except Exception as e:
            raise ValueError(f"Could not load tabular dataframe from ZIP bytes: {e}")

    # Try direct CSV first
    try:
        return read_csv_with_encoding(file_bytes)
    except Exception:
        pass

    # Then try ZIP -> first CSV
    try:
        with zipfile.ZipFile(BytesIO(file_bytes), "r") as zf:
            csv_names = [
                n
                for n in zf.namelist()
                if n.lower().endswith(".csv") and not n.startswith("__")
            ]
            if not csv_names:
                raise ValueError("ZIP does not contain any CSV files")
            with zf.open(csv_names[0]) as f:
                return read_csv_with_encoding(f.read())
    except Exception as e:
        raise ValueError(f"Could not load tabular dataframe from downloaded bytes: {e}")


def download_dataset_file(
    user_id: str, dataset_id: str, version: str = None, split: str = None
) -> bytes:
    """Download dataset file (single file or folder as ZIP). If split is 'train', 'test', or 'drift', download only that subset (for split datasets)."""
    if version:
        url = f"{API_BASE}/datasets/{user_id}/{dataset_id}/version/{version}/download"
    else:
        url = f"{API_BASE}/datasets/{user_id}/{dataset_id}/download"
    if split and split in ("train", "test", "drift"):
        url += f"?split={split}"
    r = requests.get(url, timeout=DW_DOWNLOAD_TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.content


def extract_dataset_folder(zip_bytes: bytes, extract_to: str = "temp_dataset") -> list:
    """
    Extract ZIP file containing dataset folder

    Returns:
        List of extracted file paths
    """
    import zipfile
    from io import BytesIO

    # Create extraction directory
    os.makedirs(extract_to, exist_ok=True)

    # Extract ZIP
    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as zip_ref:
        zip_ref.extractall(extract_to)

    # List extracted files
    extracted_files = []
    for root, dirs, files in os.walk(extract_to):
        for file in files:
            file_path = os.path.join(root, file)
            extracted_files.append(file_path)

    return extracted_files


def upload_model_to_dw(
    user_id: str,
    model_id: str,
    dataset_id: str,
    model_file_path: str,
    model_type: str,
    framework: str = "sklearn",
    accuracy: float = None,
    dataset_version: str = None,
    task_id: str | None = None,
) -> dict:
    """
    Upload trained model to Data Warehouse
    This will automatically trigger an automl-events message
    Version is auto-incremented by the DW
    """
    url = f"{API_BASE}/ai-models/upload/single/{user_id}"

    # Include dataset version in description for data lineage tracking
    description = f"AutoML trained model for {model_type}"
    if dataset_version:
        description += f" (trained on dataset {dataset_id} version {dataset_version})"
    else:
        description += f" (trained on dataset {dataset_id})"

    with open(model_file_path, "rb") as f:
        files = {"file": (os.path.basename(model_file_path), f)}
        data = {
            "model_id": model_id,
            "name": f"AutoML Model - {model_id}",
            "description": description,
            "framework": framework,
            "model_type": model_type,
            "training_dataset": dataset_id,  # Link to dataset
            "training_accuracy": accuracy,
        }

        headers = {"X-Task-ID": task_id} if task_id else None
        r = requests.post(url, files=files, data=data, headers=headers, timeout=120)
        r.raise_for_status()
        return r.json()


def update_model_metadata_in_dw(
    user_id: str,
    model_id: str,
    version: str,
    *,
    leaderboard: str = None,
    test_accuracy: float = None,
    validation_accuracy: float = None,
    training_accuracy: float = None,
    training_loss: float = None,
    custom_metadata: dict = None,
) -> dict:
    """
    Update existing model metadata in the Data Warehouse (e.g. after AutoML returns metrics).
    Use when the AutoML service uploads the model but does not send metrics; we patch them from its response.
    """
    url = f"{API_BASE}/ai-models/{user_id}/{model_id}"
    params = {"version": version}
    payload = {}
    if test_accuracy is not None:
        payload["test_accuracy"] = test_accuracy
    if validation_accuracy is not None:
        payload["validation_accuracy"] = validation_accuracy
    if training_accuracy is not None:
        payload["training_accuracy"] = training_accuracy
    if training_loss is not None:
        payload["training_loss"] = training_loss
    if custom_metadata is not None:
        payload["custom_metadata"] = custom_metadata
    elif leaderboard is not None:
        payload["custom_metadata"] = {"leaderboard": leaderboard}
    if not payload:
        return {}
    r = requests.put(url, params=params, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def get_latest_model_for_user(user_id: str) -> dict | None:
    """Get the most recently created model for a user (by created_at). Used to find the model just uploaded by AutoML."""
    url = f"{API_BASE}/ai-models/{user_id}"
    r = requests.get(url, params={"limit": 1}, timeout=10)
    r.raise_for_status()
    models = r.json()
    return models[0] if models else None


def _coerce_metric_float(value) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _column_lookup(columns: list[str], *needles: str) -> str | None:
    """Find first column whose normalized name contains any needle."""
    for col in columns:
        norm = re.sub(r"\s+", "_", str(col).strip().lower())
        if not norm or norm.startswith("unnamed"):
            continue
        if any(n in norm for n in needles):
            return col
    return None


def _extract_scores_from_leaderboard(leaderboard: str) -> dict[str, float | None]:
    """
    Parse AutoGluon markdown leaderboard (pipe table) and return scores for the best row.
    Maps score_test -> test_accuracy, score_val -> validation_accuracy when present.
    """
    scores: dict[str, float | None] = {
        "test_accuracy": None,
        "validation_accuracy": None,
        "training_accuracy": None,
    }
    if not leaderboard or not isinstance(leaderboard, str):
        return scores

    table_lines: list[str] = []
    for line in leaderboard.strip().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if re.match(r"^\|[\s\-:|]+\|$", line):
            continue
        table_lines.append(line)

    if len(table_lines) < 2:
        return scores

    try:
        df = pd.read_csv(StringIO("\n".join(table_lines)), sep="|", engine="python")
        df = df.dropna(axis=1, how="all")
        df.columns = [str(c).strip() for c in df.columns]
        if df.empty:
            return scores

        col_test = _column_lookup(
            list(df.columns), "score_test", "test_score", "test_accuracy"
        )
        col_val = _column_lookup(
            list(df.columns), "score_val", "val_score", "validation_accuracy"
        )
        col_train = _column_lookup(
            list(df.columns), "score_train", "train_score", "training_accuracy"
        )

        best = df.iloc[0]
        if col_test:
            scores["test_accuracy"] = _coerce_metric_float(best[col_test])
        if col_val:
            scores["validation_accuracy"] = _coerce_metric_float(best[col_val])
        if col_train:
            scores["training_accuracy"] = _coerce_metric_float(best[col_train])
    except Exception as e:
        logger.warning("Could not parse leaderboard table for metrics: %s", e)

    return scores


def _parse_automl_response_metrics(response_data: dict) -> dict:
    """
    Extract metrics from AutoML tabular response for DW metadata update.
    Returns dict with: leaderboard, test_accuracy, validation_accuracy, training_accuracy, custom_metadata.
    """
    out = {
        "leaderboard": None,
        "test_accuracy": None,
        "validation_accuracy": None,
        "training_accuracy": None,
        "custom_metadata": {},
    }
    if not response_data:
        return out
    out["leaderboard"] = response_data.get("leaderboard")
    out["test_accuracy"] = _coerce_metric_float(response_data.get("test_accuracy"))
    out["validation_accuracy"] = _coerce_metric_float(
        response_data.get("validation_accuracy")
    )
    out["training_accuracy"] = _coerce_metric_float(
        response_data.get("training_accuracy")
    )

    best_score = response_data.get("best_score") or response_data.get("score_test")
    if best_score is not None and out["test_accuracy"] is None:
        s = _coerce_metric_float(best_score)
        if s is not None:
            out["test_accuracy"] = s
        else:
            out["custom_metadata"]["best_score"] = best_score

    if out["leaderboard"] and (
        out["test_accuracy"] is None
        or out["validation_accuracy"] is None
        or out["training_accuracy"] is None
    ):
        from_lb = _extract_scores_from_leaderboard(out["leaderboard"])
        if out["test_accuracy"] is None:
            out["test_accuracy"] = from_lb["test_accuracy"]
        if out["validation_accuracy"] is None:
            out["validation_accuracy"] = from_lb["validation_accuracy"]
        if out["training_accuracy"] is None:
            out["training_accuracy"] = from_lb["training_accuracy"]

    for key in ("message", "model_id", "version"):
        if key in response_data and response_data[key] is not None:
            out["custom_metadata"][key] = response_data[key]
    if out["leaderboard"]:
        out["custom_metadata"]["leaderboard"] = out["leaderboard"]
    return out


async def process_automl_trigger(
    event: dict, producer: AIOKafkaProducer = None
) -> None:
    """
    Process an AutoML trigger event from Agentic Core

    Event structure (supports both formats):
    {
        "task_id": "automl_task_<...>",
        "event_type": "automl-trigger",
        "timestamp": "...",
        "input": {
            "dataset_id": "...",
            "dataset_version": "v1",
            "user_id": "...",
            "task_category": "tabular" | "vision",
            "task_type": "classification" | "regression" | ...,
            "target_column_name": "target",        // tabular
            "filename_column": "filename",         // vision (optional)
            "label_column": "label",               // vision (optional)
            "model_size": "small",                 // vision (optional: small/medium/large)
            "time_budget": "10"                    // seconds (both tabular and vision)
        }
    }

    OR (for backward compatibility):
    {
        "event_type": "automl-trigger.reported",
        "dataset_id": "dataset123",
        "user_id": "user123",
        "target_column_name": "target",
        "task_type": "classification",
        "time_budget": "10",
        "timestamp": "2025-10-10T12:00:00.000000"
    }
    """
    try:
        # Extract task_id from event (if present)
        task_id = event.get("task_id")

        # Extract input object (new structure) or use event directly (backward compatibility)
        input_obj = event.get("input", event)

        user_id = input_obj.get("user_id")
        dataset_id = input_obj.get("dataset_id")
        dataset_version = input_obj.get(
            "dataset_version", "v1"
        )  # Default to v1 for backward compatibility
        target_column = input_obj.get("target_column_name")
        task_type = input_obj.get("task_type")
        time_budget = input_obj.get(
            "time_budget", event.get("time_budget", "10")
        )  # Check both places
        task_category = (input_obj.get("task_category") or "tabular").strip().lower()
        filename_column = input_obj.get("filename_column", "filename")
        label_column = input_obj.get("label_column", "label")
        model_size = input_obj.get("model_size", "small")

        if not user_id or not dataset_id:
            logger.warning("Missing user_id or dataset_id in event; skipping")
            if producer and task_id:
                await send_automl_failure_to_kafka(
                    producer,
                    task_id,
                    str(user_id or ""),
                    str(dataset_id or ""),
                    "Missing user_id or dataset_id in event",
                    dataset_version=input_obj.get("dataset_version"),
                )
            return

        # Idempotency claim across replicas (skip if another worker already took it).
        if not claim_task_id(
            task_id,
            user_id=user_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
        ):
            return

        if task_category == "tabular" and (not target_column or not task_type):
            logger.warning(
                "Tabular task requires target_column_name and task_type in event; skipping"
            )
            if producer and task_id:
                await send_automl_failure_to_kafka(
                    producer,
                    task_id,
                    user_id,
                    dataset_id,
                    "Tabular task requires target_column_name and task_type",
                    dataset_version=dataset_version,
                )
            return
        # Vision defaults filename_column, label_column, task_type, model_size so no strict check needed

        logger.info(
            f"Processing AutoML trigger for dataset {dataset_id} version {dataset_version}"
        )
        logger.info(f"  Task ID: {task_id}")
        logger.info(f"  User: {user_id}")
        logger.info(f"  Target column: {target_column}")
        logger.info(f"  Task type: {task_type}")
        logger.info(f"  Task category: {task_category}")
        logger.info(f"  Time budget: {time_budget} seconds")

        # Step 1: Fetch dataset metadata and download file (for validation/preprocessing)
        try:
            logger.info(f"Fetching dataset metadata from: {API_BASE}")
            metadata = fetch_dataset_metadata(user_id, dataset_id, dataset_version)
            logger.info("Dataset metadata fetched successfully")

            # Check if dataset is a folder or single file, and if it has train/test/drift split
            is_folder = metadata.get("is_folder", False)
            file_count = metadata.get("file_count", 1)
            has_split = bool(metadata.get("custom_metadata", {}).get("split"))
            # For tabular split datasets, we expect downstream services to use the train split
            # when downloading from DW (so they get a stable training table).
            dataset_version_for_automl = dataset_version
            # For split tabular datasets, download only the train split so we get a single CSV for validation
            download_split = "train" if has_split else None
            if download_split:
                logger.info(
                    f"Dataset has train/test/drift split; downloading '{download_split}' split for {task_category} AutoML"
                )

            logger.info(f"Dataset type: {'FOLDER' if is_folder else 'SINGLE FILE'}")
            if is_folder:
                logger.info(f"File count: {file_count}")

            logger.info(f"Downloading dataset file from: {API_BASE}")
            file_bytes = download_dataset_file(
                user_id, dataset_id, dataset_version, split=download_split
            )
            logger.info(f"Dataset downloaded: {len(file_bytes)} bytes")
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout while fetching dataset from {API_BASE}")
            logger.error(f"   Error: {e}")
            logger.error(
                "   If running in Docker, ensure DW_HOST=data-warehouse-api is set"
            )
            logger.error(
                f"   If running locally, ensure the Data Warehouse API is running on {DW_HOST}:{DW_PORT}"
            )
            if producer and task_id:
                await send_automl_failure_to_kafka(
                    producer,
                    task_id,
                    user_id,
                    dataset_id,
                    f"Timeout fetching dataset: {e}",
                    dataset_version,
                )
            return
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to Data Warehouse API at {API_BASE}")
            logger.error(f"   Error: {e}")
            logger.error(
                "   If running in Docker, ensure DW_HOST=data-warehouse-api is set"
            )
            logger.error(
                f"   If running locally, ensure the Data Warehouse API is running on {DW_HOST}:{DW_PORT}"
            )
            if producer and task_id:
                await send_automl_failure_to_kafka(
                    producer,
                    task_id,
                    user_id,
                    dataset_id,
                    f"Connection error fetching dataset: {e}",
                    dataset_version,
                )
            return
        except Exception as e:
            logger.error(f"Failed to fetch dataset: {e}")
            logger.error(f"   API Base URL: {API_BASE}")
            if producer and task_id:
                await send_automl_failure_to_kafka(
                    producer, task_id, user_id, dataset_id, str(e), dataset_version
                )
            return

        # Step 2: Call AutoML service with all necessary information
        if task_category == "tabular":
            normalized_task_type = normalize_tabular_task_type(task_type)
            if normalized_task_type != (
                str(task_type).strip().lower() if task_type is not None else None
            ):
                logger.info(
                    f"  Normalized tabular task_type: {task_type} -> {normalized_task_type}"
                )

            # Optional validation before calling AutoML service: ensure bytes are readable tabular data.
            # This catches malformed/incorrect dataset payloads early with clear logs.
            try:
                df_preview = load_tabular_dataframe(file_bytes)
                logger.info(f"Tabular dataset validation OK: shape={df_preview.shape}")
                if target_column and target_column not in df_preview.columns:
                    logger.warning(
                        f"Target column '{target_column}' not found in preview columns; AutoML service may fail."
                    )
            except Exception as e:
                err_msg = f"Tabular dataset could not be parsed locally before AutoML call: {e}"
                logger.error(err_msg)
                if producer and task_id:
                    await send_automl_failure_to_kafka(
                        producer, task_id, user_id, dataset_id, err_msg, dataset_version
                    )
                return

            # Convert time_budget from string to int.
            # IMPORTANT: time_budget is interpreted as SECONDS end-to-end (not minutes).
            # The endpoint expects time_budget in seconds as an integer.
            try:
                time_budget_seconds = int(time_budget)
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid time_budget '{time_budget}', using default 10 seconds"
                )
                time_budget_seconds = 10

            num_cpus = str(input_obj.get("num_cpus", "auto")).strip() or "auto"
            num_gpus = str(input_obj.get("num_gpus", "auto")).strip() or "auto"
            data = {
                "user_id": user_id,
                "dataset_id": dataset_id,
                "dataset_version": dataset_version_for_automl,
                "target_column_name": target_column,
                "time_stamp_column_name": input_obj.get("time_stamp_column_name") or "",
                "task_type": normalized_task_type,
                "time_budget": time_budget_seconds,  # Send as integer in seconds
                "num_cpus": num_cpus,
                "num_gpus": num_gpus,
            }
            # Tell the AutoML tabular service to request the train split when downloading from DW (split datasets)
            if has_split:
                data["dataset_split"] = "train"
                logger.info(
                    "  Passing dataset_split=train so AutoML service uses train split from DW"
                )

            # Include task_id in headers if available (for tracking)
            headers = {"X-Task-ID": task_id} if task_id else None

            logger.info(f"Calling AutoML Tabular: {AUTOML_TABULAR_BEST_MODEL_URL}")
            logger.info(f"   Using engine: {AUTOML_ENGINE_HOST}:{AUTOML_ENGINE_PORT}")

            # Try to verify the service is reachable first
            try:
                health_check_url = f"{AUTOML_ENGINE_URL}/health"
                logger.debug(f"Checking if service is reachable at {health_check_url}")
                health_check = requests.get(health_check_url, timeout=5)
                logger.debug(f"Service health check: {health_check.status_code}")
            except Exception as health_e:
                logger.warning(
                    f"Could not reach AutoML service for health check: {health_e}"
                )
                logger.warning(
                    "   This might indicate the service is not running or not accessible"
                )

            try:
                connect_timeout, read_timeout = compute_request_timeouts(
                    time_budget_seconds
                )
                logger.info(
                    f"Setting request timeout to connect={connect_timeout}s, read={read_timeout}s "
                    f"(time_budget={time_budget_seconds}s + overhead)"
                )
                r = await _post_with_429_retry(
                    AUTOML_TABULAR_BEST_MODEL_URL,
                    data=data,
                    headers=headers,
                )
            except requests.exceptions.ConnectionError as e:
                logger.error(
                    f"Failed to connect to AutoML Tabular at {AUTOML_TABULAR_BEST_MODEL_URL}"
                )
                logger.error(f"   Error: {e}")
                logger.error(
                    f"   Ensure the unified AutoML engine is running on {AUTOML_ENGINE_HOST}:{AUTOML_ENGINE_PORT}"
                )
                logger.error("   Check: docker ps | grep automl")
                logger.error(f"   Verify: curl {AUTOML_ENGINE_URL}/health")
                if producer and task_id:
                    await send_automl_failure_to_kafka(
                        producer,
                        task_id,
                        user_id,
                        dataset_id,
                        f"Connection error: {e}",
                        dataset_version,
                    )
                return
            except requests.exceptions.HTTPError as e:
                logger.error(f"AutoML service returned HTTP error: {e}")
                err_msg = str(e)
                if e.response is not None:
                    logger.error(f"   Response status: {e.response.status_code}")
                    logger.error(f"   URL: {e.response.url}")
                    if e.response.content:
                        try:
                            error_detail = e.response.json()
                            err_body = json.dumps(error_detail)
                            err_msg = (
                                f"{e.response.status_code} {e.response.url}: {err_body}"
                            )
                            logger.error(f"   Response body: {err_body}")
                        except Exception:
                            err_msg = f"{e.response.status_code}: {e.response.text[:500] if e.response.text else 'no body'}"
                            logger.error(f"   Response body: {e.response.text[:500]}")
                    logger.error(
                        "   If running in Docker, ensure AUTOML_ENGINE_HOST points at the engine service"
                    )
                if producer and task_id:
                    await send_automl_failure_to_kafka(
                        producer, task_id, user_id, dataset_id, err_msg, dataset_version
                    )
                return
            except requests.exceptions.Timeout as e:
                logger.error(f"Request to AutoML service timed out (connect/read): {e}")
                logger.error("   Training may have taken longer than expected")
                logger.error(
                    "   Consider increasing time_budget or checking service logs"
                )
                if producer and task_id:
                    await send_automl_failure_to_kafka(
                        producer,
                        task_id,
                        user_id,
                        dataset_id,
                        f"Timeout: {e}",
                        dataset_version,
                    )
                return

            response_data = r.json() if r.content else {}
            logger.info(
                "✅ AutoML processing completed and models uploaded to Data Warehouse"
            )
            logger.info(
                f"   Response: {json.dumps(response_data, indent=2, default=str)}"
            )

            # Enrich DW model metadata with metrics from AutoML response (leaderboard, test/validation accuracy)
            try:
                metrics = _parse_automl_response_metrics(response_data)
                model_id_to_update = response_data.get("model_id")
                version_to_update = response_data.get("version")
                if not model_id_to_update or not version_to_update:
                    latest = get_latest_model_for_user(user_id)
                    if latest:
                        model_id_to_update = latest.get("model_id")
                        version_to_update = latest.get("version", "v1")
                        logger.info(
                            f"   Using latest model for user as target for metrics: {model_id_to_update} {version_to_update}"
                        )
                if (
                    model_id_to_update
                    and version_to_update
                    and (
                        metrics["leaderboard"]
                        or metrics["test_accuracy"] is not None
                        or metrics["validation_accuracy"] is not None
                        or metrics["training_accuracy"] is not None
                        or metrics["custom_metadata"]
                    )
                ):
                    update_model_metadata_in_dw(
                        user_id,
                        model_id_to_update,
                        version_to_update,
                        leaderboard=metrics["leaderboard"],
                        test_accuracy=metrics["test_accuracy"],
                        validation_accuracy=metrics["validation_accuracy"],
                        training_accuracy=metrics["training_accuracy"],
                        custom_metadata=metrics["custom_metadata"] or None,
                    )
                    logger.info(
                        "   Updated model metadata in DW for %s %s (test=%s, val=%s, train=%s)",
                        model_id_to_update,
                        version_to_update,
                        metrics["test_accuracy"],
                        metrics["validation_accuracy"],
                        metrics["training_accuracy"],
                    )
            except Exception as meta_e:
                logger.warning(
                    f"   Could not update model metadata with AutoML metrics: {meta_e}"
                )

            # Note: The AutoML service should handle model upload to DW with task_id
            # which will automatically trigger automl-events
        elif task_category == "vision":
            # Vision AutoML: time_budget in seconds (vision API expects seconds)
            try:
                time_budget_seconds = int(time_budget)
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid time_budget '{time_budget}', using default 10 seconds"
                )
                time_budget_seconds = 10

            normalized_vision_task_type = normalize_vision_task_type(
                task_type or "classification"
            )
            if (task_type or "classification") != normalized_vision_task_type:
                logger.info(
                    f"  Normalized vision task_type: {task_type} -> {normalized_vision_task_type}"
                )

            num_cpus = str(input_obj.get("num_cpus", "auto")).strip() or "auto"
            num_gpus = str(input_obj.get("num_gpus", "auto")).strip() or "auto"
            data = {
                "user_id": user_id,
                "dataset_id": dataset_id,
                "dataset_version": dataset_version or "v1",
                "filename_column": filename_column,
                "label_column": label_column,
                "task_type": normalized_vision_task_type,
                "time_budget": time_budget_seconds,
                "model_size": (model_size or "small").strip().lower(),
                "num_cpus": num_cpus,
                "num_gpus": num_gpus,
            }
            requested_split = (input_obj.get("dataset_split") or "").strip().lower()
            if has_split and requested_split in ("train", "test", "drift"):
                data["dataset_split"] = requested_split
                logger.info(
                    f"  Passing dataset_split={requested_split} so Vision service uses split from DW"
                )
            elif has_split:
                data["dataset_split"] = "train"
                logger.info(
                    "  Passing dataset_split=train so Vision service uses train split from DW"
                )
            headers = {"X-Task-ID": task_id} if task_id else None

            connect_timeout, read_timeout = compute_request_timeouts(
                time_budget_seconds
            )
            logger.info(f"Calling AutoML Vision: {AUTOML_VISION_BEST_MODEL_URL}")
            logger.info(
                f"   filename_column={filename_column}, label_column={label_column}, model_size={model_size}"
            )
            logger.info(
                f"   time_budget={time_budget_seconds}s, timeout=connect={connect_timeout}s read={read_timeout}s"
            )

            try:
                # Same 429 retry path as tabular so parallel workers queue cleanly.
                r = await _post_with_429_retry(
                    AUTOML_VISION_BEST_MODEL_URL,
                    data=data,
                    headers=headers,
                )
            except requests.exceptions.ConnectionError as e:
                logger.error(
                    f"Failed to connect to AutoML Vision at {AUTOML_VISION_BEST_MODEL_URL}: {e}"
                )
                logger.error(
                    "  Ensure the unified AutoML engine is running on "
                    f"{AUTOML_ENGINE_HOST}:{AUTOML_ENGINE_PORT}. "
                    "From Docker use AUTOML_ENGINE_HOST=host.docker.internal"
                )
                if producer and task_id:
                    await send_automl_failure_to_kafka(
                        producer,
                        task_id,
                        user_id,
                        dataset_id,
                        f"Connection error: {e}",
                        dataset_version,
                    )
                return
            except requests.exceptions.HTTPError as e:
                logger.error(f"AutoML Vision HTTP error: {e}")
                err_msg = str(e)
                if e.response is not None:
                    try:
                        body_json = e.response.json() if e.response.content else None
                        if body_json is not None:
                            err_msg = f"{e.response.status_code} {e.response.url}: {json.dumps(body_json)}"
                            logger.error(f"   Response body: {json.dumps(body_json)}")
                        else:
                            err_msg = f"{e.response.status_code} {e.response.url}: <empty body>"
                            logger.error("   Response body: <empty>")
                    except Exception:
                        txt = (e.response.text or "")[:1000]
                        err_msg = f"{e.response.status_code} {e.response.url}: {txt}"
                        logger.error(f"   Response body (text): {txt}")
                if producer and task_id:
                    await send_automl_failure_to_kafka(
                        producer, task_id, user_id, dataset_id, err_msg, dataset_version
                    )
                return
            except requests.exceptions.Timeout as e:
                logger.error(f"AutoML Vision request timed out (connect/read): {e}")
                if producer and task_id:
                    await send_automl_failure_to_kafka(
                        producer,
                        task_id,
                        user_id,
                        dataset_id,
                        f"Timeout: {e}",
                        dataset_version,
                    )
                return

            response_data = r.json() if r.content else {}
            logger.info("✅ Vision AutoML completed; model uploaded to Data Warehouse")
            logger.info(
                f"   Response: {json.dumps(response_data, indent=2, default=str)}"
            )

            # Enrich DW model metadata with metrics from AutoML response
            try:
                metrics = _parse_automl_response_metrics(response_data)
                model_id_to_update = response_data.get("model_id")
                version_to_update = response_data.get("version")
                if not model_id_to_update or not version_to_update:
                    latest = get_latest_model_for_user(user_id)
                    if latest:
                        model_id_to_update = latest.get("model_id")
                        version_to_update = latest.get("version", "v1")
                if (
                    model_id_to_update
                    and version_to_update
                    and (
                        metrics["leaderboard"]
                        or metrics["test_accuracy"] is not None
                        or metrics["validation_accuracy"] is not None
                        or metrics["custom_metadata"]
                    )
                ):
                    update_model_metadata_in_dw(
                        user_id,
                        model_id_to_update,
                        version_to_update,
                        leaderboard=metrics["leaderboard"],
                        test_accuracy=metrics["test_accuracy"],
                        validation_accuracy=metrics["validation_accuracy"],
                        training_accuracy=metrics["training_accuracy"],
                        custom_metadata=metrics["custom_metadata"] or None,
                    )
                    logger.info(
                        "   Updated model metadata in DW for %s %s (test=%s, val=%s, train=%s)",
                        model_id_to_update,
                        version_to_update,
                        metrics["test_accuracy"],
                        metrics["validation_accuracy"],
                        metrics["training_accuracy"],
                    )
            except Exception as meta_e:
                logger.warning(
                    f"   Could not update model metadata with Vision AutoML metrics: {meta_e}"
                )
        else:
            logger.error(f"Unsupported task category: {task_category}")
            return

    except Exception as e:
        logger.error(f"Error processing AutoML trigger event: {e}", exc_info=True)
        try:
            task_id = event.get("task_id")
            input_obj = event.get("input", event)
            user_id = input_obj.get("user_id") or ""
            dataset_id = input_obj.get("dataset_id") or ""
            dataset_version = input_obj.get("dataset_version")
            if producer and task_id:
                await send_automl_failure_to_kafka(
                    producer,
                    task_id,
                    str(user_id),
                    str(dataset_id),
                    str(e),
                    dataset_version,
                )
        except Exception as send_err:
            logger.warning(f"Could not send AutoML failure event: {send_err}")
        return


async def run_consumer() -> None:
    """Main consumer loop"""
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        key_serializer=lambda k: (k.encode("utf-8") if k else None),
    )
    await producer.start()
    logger.info(
        f"AutoML completion producer started (topic: {KAFKA_AUTOML_COMPLETE_TOPIC})"
    )

    consumer = AIOKafkaConsumer(
        KAFKA_AUTOML_TRIGGER_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=KAFKA_CONSUMER_GROUP,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        # Prevent group rebalances during long training jobs (time_budget + overhead).
        # max_poll_interval_ms is the max time between successive polls.
        # Default 4h; training can exceed 10m easily (vision / large ZIP). Override via env.
        max_poll_interval_ms=int(
            os.getenv("KAFKA_MAX_POLL_INTERVAL_MS", str(4 * 60 * 60 * 1000))
        ),
        session_timeout_ms=int(os.getenv("KAFKA_SESSION_TIMEOUT_MS", "30000")),
        heartbeat_interval_ms=int(os.getenv("KAFKA_HEARTBEAT_INTERVAL_MS", "10000")),
        max_poll_records=int(os.getenv("KAFKA_MAX_POLL_RECORDS", "1")),
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        key_deserializer=lambda m: m.decode("utf-8") if m else None,
    )

    logger.info("Starting AutoML Trigger consumer...")
    logger.info(f"Bootstrap servers: {KAFKA_BOOTSTRAP_SERVERS}")
    logger.info(f"Topic: {KAFKA_AUTOML_TRIGGER_TOPIC}")
    logger.info(f"Consumer group: {KAFKA_CONSUMER_GROUP}")
    logger.info(f"Data Warehouse API: {API_BASE} (host: {DW_HOST}, port: {DW_PORT})")
    logger.info(f"AutoML Tabular: {AUTOML_TABULAR_BEST_MODEL_URL}")
    logger.info(f"AutoML Vision: {AUTOML_VISION_BEST_MODEL_URL}")
    logger.info("Waiting for AutoML trigger events from Agentic Core...")

    await consumer.start()

    try:
        async for msg in consumer:
            key = msg.key
            value = msg.value

            logger.info("=" * 80)
            logger.info("AutoML Trigger Message received")
            logger.info(f"  Partition={msg.partition} Offset={msg.offset}")
            logger.info(f"  Key={key}")
            logger.info(f"  Event={json.dumps(value, indent=2)}")
            logger.info("=" * 80)

            # Process the AutoML trigger event
            # The process_automl_trigger function handles both event structures:
            # - New structure with "input" object and "task_id"
            # - Old structure for backward compatibility
            await process_automl_trigger(value, producer)

    finally:
        await consumer.stop()
        await producer.stop()
        logger.info("AutoML Trigger consumer and producer stopped")


if __name__ == "__main__":
    try:
        asyncio.run(run_consumer())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
