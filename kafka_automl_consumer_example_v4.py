#!/usr/bin/env python3
"""
Kafka Consumer Example for AutoML Trigger Events (Tabular + Vision)

This consumer listens to the automl-trigger-events topic (from Agentic Core).
Event field task_category selects the pipeline:
- task_category "tabular" -> calls AutoML Tabular (best_model)
- task_category "vision"  -> calls AutoML Vision (best_model)

When an AutoML trigger event is received, the consumer:
- Fetches dataset metadata and validates download
- Calls the appropriate AutoML service (tabular or vision)
- The AutoML service trains and uploads the model to the Data Warehouse (triggers automl-events)

Usage:
  # Default (localhost; Tabular 8001, Vision 8002)
  KAFKA_BOOTSTRAP_SERVERS=localhost:9092 python kafka_automl_consumer_example_v3.py

  # Override endpoints
  API_BASE=http://localhost:8000 TABULAR_AUTOML_HOST=localhost TABULAR_AUTOML_PORT=8001 \\
  VISION_AUTOML_HOST=localhost VISION_AUTOML_PORT=8002 python kafka_automl_consumer_example_v3.py

  Note: Runs OUTSIDE Docker; connect to Docker services via localhost.
"""

import os
import asyncio
import json
import logging
from datetime import datetime, timezone
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

# AutoML Tabular configuration
# Default to localhost (for running outside Docker)
# Can be overridden with TABULAR_AUTOML_HOST environment variable
TABULAR_AUTOML_HOST = os.getenv("TABULAR_AUTOML_HOST", "localhost")
TABULAR_AUTOML_PORT = os.getenv("TABULAR_AUTOML_PORT", "8001")
TABULAR_AUTOML_URL = f"http://{TABULAR_AUTOML_HOST}:{TABULAR_AUTOML_PORT}"
AUTOML_TABULAR_BEST_MODEL_URL = f"{TABULAR_AUTOML_URL}/automl_tabular/best_model"

# AutoML Vision configuration (runs on 8002; tabular on 8001)
VISION_AUTOML_HOST = os.getenv("VISION_AUTOML_HOST", "localhost")
VISION_AUTOML_PORT = os.getenv("VISION_AUTOML_PORT", "8002")
VISION_AUTOML_URL = f"http://{VISION_AUTOML_HOST}:{VISION_AUTOML_PORT}"
AUTOML_VISION_BEST_MODEL_URL = f"{VISION_AUTOML_URL}/automl_vision/best_model/"

# Log configuration at module load (so we see which host/port are used, e.g. host.docker.internal when in Docker)
logger.info(f"Configuration:")
logger.info(f"  Data Warehouse API: {API_BASE} (host: {DW_HOST}, port: {DW_PORT})")
logger.info(f"  AutoML Tabular: {AUTOML_TABULAR_BEST_MODEL_URL}")
logger.info(f"  AutoML Vision: {AUTOML_VISION_BEST_MODEL_URL}")
if (
    VISION_AUTOML_HOST == "localhost"
    and os.getenv("KAFKA_BOOTSTRAP_SERVERS", "").find("kafka") >= 0
):
    logger.warning(
        "  When running in Docker, set VISION_AUTOML_HOST=host.docker.internal so the container can reach Vision on the host"
    )


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
    r = requests.get(url, timeout=60)
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


def _parse_automl_response_metrics(response_data: dict) -> dict:
    """
    Extract metrics from AutoML tabular response for DW metadata update.
    Returns dict with: leaderboard, test_accuracy, validation_accuracy, custom_metadata.
    """
    out = {
        "leaderboard": None,
        "test_accuracy": None,
        "validation_accuracy": None,
        "custom_metadata": {},
    }
    if not response_data:
        return out
    out["leaderboard"] = response_data.get("leaderboard")
    out["test_accuracy"] = response_data.get("test_accuracy")
    if out["test_accuracy"] is not None and not isinstance(
        out["test_accuracy"], (int, float)
    ):
        try:
            out["test_accuracy"] = float(out["test_accuracy"])
        except (TypeError, ValueError):
            out["test_accuracy"] = None
    out["validation_accuracy"] = response_data.get("validation_accuracy")
    if out["validation_accuracy"] is not None and not isinstance(
        out["validation_accuracy"], (int, float)
    ):
        try:
            out["validation_accuracy"] = float(out["validation_accuracy"])
        except (TypeError, ValueError):
            out["validation_accuracy"] = None
    best_score = response_data.get("best_score") or response_data.get("score_test")
    if best_score is not None and out["test_accuracy"] is None:
        try:
            s = float(best_score)
            if 0 <= s <= 1:
                out["test_accuracy"] = s
            else:
                out["custom_metadata"]["best_score"] = s
        except (TypeError, ValueError):
            out["custom_metadata"]["best_score"] = best_score
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
            "time_budget": "10"                    // minutes (both)
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
        logger.info(
            f"  Time budget: {time_budget} minutes (will be converted to seconds)"
        )

        # Step 1: Fetch dataset metadata and download file (for validation/preprocessing)
        try:
            logger.info(f"Fetching dataset metadata from: {API_BASE}")
            metadata = fetch_dataset_metadata(user_id, dataset_id, dataset_version)
            logger.info(f"Dataset metadata fetched successfully")

            # Check if dataset is a folder or single file, and if it has train/test/drift split
            is_folder = metadata.get("is_folder", False)
            file_count = metadata.get("file_count", 1)
            has_split = bool(metadata.get("custom_metadata", {}).get("split"))
            # For split tabular datasets, download only the train split so we get a single CSV for validation
            download_split = (
                "train" if (has_split and task_category == "tabular") else None
            )
            if download_split:
                logger.info(
                    f"Dataset has train/test/drift split; downloading '{download_split}' split for tabular AutoML"
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
                f"   If running in Docker, ensure DW_HOST=data-warehouse-api is set"
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
                f"   If running in Docker, ensure DW_HOST=data-warehouse-api is set"
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

            # Convert time_budget from string to int, and from minutes to seconds
            # The endpoint expects time_budget in seconds as an integer
            try:
                time_budget_seconds = (
                    int(time_budget) * 60
                )  # Convert minutes to seconds
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid time_budget '{time_budget}', using default 10 minutes (600 seconds)"
                )
                time_budget_seconds = 600

            data = {
                "user_id": user_id,
                "dataset_id": dataset_id,
                "dataset_version": dataset_version,
                "target_column_name": target_column,
                "time_stamp_column_name": "",  # Empty string for non-time-series tasks
                "task_type": task_type,
                "time_budget": time_budget_seconds,  # Send as integer in seconds
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
            logger.info(
                f"   Using host: {TABULAR_AUTOML_HOST}, port: {TABULAR_AUTOML_PORT}"
            )

            # Try to verify the service is reachable first
            try:
                health_check_url = f"{TABULAR_AUTOML_URL}/docs"  # FastAPI docs endpoint
                logger.debug(f"Checking if service is reachable at {health_check_url}")
                health_check = requests.get(health_check_url, timeout=5)
                logger.debug(f"Service health check: {health_check.status_code}")
            except Exception as health_e:
                logger.warning(
                    f"Could not reach AutoML service for health check: {health_e}"
                )
                logger.warning(
                    f"   This might indicate the service is not running or not accessible"
                )

            try:
                # Calculate timeout: training time + buffer for download/upload/processing
                # Add 5 minutes (300 seconds) buffer for dataset download, model upload, and processing overhead
                request_timeout = time_budget_seconds + 300
                logger.info(
                    f"Setting request timeout to {request_timeout} seconds ({request_timeout // 60} minutes) to accommodate {time_budget_seconds // 60} minutes of training"
                )
                r = requests.post(
                    AUTOML_TABULAR_BEST_MODEL_URL,
                    data=data,
                    headers=headers,
                    timeout=request_timeout,
                )
                r.raise_for_status()
            except requests.exceptions.ConnectionError as e:
                logger.error(
                    f"Failed to connect to AutoML Tabular at {AUTOML_TABULAR_BEST_MODEL_URL}"
                )
                logger.error(f"   Error: {e}")
                logger.error(
                    f"   Ensure the tabular Docker service is running and accessible on {TABULAR_AUTOML_HOST}:{TABULAR_AUTOML_PORT}"
                )
                logger.error(f"   Check: docker ps | grep tabular")
                logger.error(
                    f"   Verify port mapping: docker port alfie_automl_tabular"
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
                        f"   If running in Docker, ensure TABULAR_AUTOML_HOST=tabular is set"
                    )
                if producer and task_id:
                    await send_automl_failure_to_kafka(
                        producer, task_id, user_id, dataset_id, err_msg, dataset_version
                    )
                return
            except requests.exceptions.Timeout as e:
                logger.error(
                    f"Request to AutoML service timed out after {request_timeout} seconds"
                )
                logger.error(f"   Training may have taken longer than expected")
                logger.error(
                    f"   Consider increasing time_budget or checking service logs"
                )
                if producer and task_id:
                    await send_automl_failure_to_kafka(
                        producer,
                        task_id,
                        user_id,
                        dataset_id,
                        f"Timeout after {request_timeout}s: {e}",
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
                        custom_metadata=metrics["custom_metadata"] or None,
                    )
                    logger.info(
                        f"   Updated model metadata in DW with leaderboard/metrics for {model_id_to_update} {version_to_update}"
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
                time_budget_seconds = (
                    int(time_budget) * 60
                )  # event in minutes -> seconds
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid time_budget '{time_budget}', using default 10 minutes (600 seconds)"
                )
                time_budget_seconds = 600

            data = {
                "user_id": user_id,
                "dataset_id": dataset_id,
                "dataset_version": dataset_version or "v1",
                "filename_column": filename_column,
                "label_column": label_column,
                "task_type": task_type or "vision_classification",
                "time_budget": time_budget_seconds,
                "model_size": (model_size or "small").strip().lower(),
            }
            if has_split:
                data["dataset_split"] = "train"
                logger.info(
                    "  Passing dataset_split=train so Vision service uses train split from DW"
                )
            headers = {"X-Task-ID": task_id} if task_id else None

            request_timeout = time_budget_seconds + 300
            logger.info(f"Calling AutoML Vision: {AUTOML_VISION_BEST_MODEL_URL}")
            logger.info(
                f"   filename_column={filename_column}, label_column={label_column}, model_size={model_size}"
            )
            logger.info(
                f"   time_budget={time_budget_seconds}s, request_timeout={request_timeout}s"
            )

            try:
                r = requests.post(
                    AUTOML_VISION_BEST_MODEL_URL,
                    data=data,
                    headers=headers,
                    timeout=request_timeout,
                )
                r.raise_for_status()
            except requests.exceptions.ConnectionError as e:
                logger.error(
                    f"Failed to connect to AutoML Vision at {AUTOML_VISION_BEST_MODEL_URL}: {e}"
                )
                logger.error(
                    "  Ensure the AutoML Vision service is running on port 8002. "
                    "Override with VISION_AUTOML_PORT if different. "
                    "From Docker use VISION_AUTOML_HOST=host.docker.internal"
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
                if e.response and e.response.content:
                    try:
                        err_msg = (
                            f"{e.response.status_code}: {json.dumps(e.response.json())}"
                        )
                        logger.error(f"   Response: {e.response.json()}")
                    except Exception:
                        err_msg = f"{e.response.status_code}: {e.response.text[:500]}"
                        logger.error(f"   Response: {e.response.text[:500]}")
                if producer and task_id:
                    await send_automl_failure_to_kafka(
                        producer, task_id, user_id, dataset_id, err_msg, dataset_version
                    )
                return
            except requests.exceptions.Timeout as e:
                logger.error(
                    f"AutoML Vision request timed out after {request_timeout}s"
                )
                if producer and task_id:
                    await send_automl_failure_to_kafka(
                        producer,
                        task_id,
                        user_id,
                        dataset_id,
                        f"Timeout after {request_timeout}s: {e}",
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
                        custom_metadata=metrics["custom_metadata"] or None,
                    )
                    logger.info(
                        f"   Updated model metadata in DW for {model_id_to_update} {version_to_update}"
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
