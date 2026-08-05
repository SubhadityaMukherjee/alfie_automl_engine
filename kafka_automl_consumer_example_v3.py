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
from aiokafka import AIOKafkaConsumer
import requests
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv, find_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("kafka_automl_consumer")

load_dotenv(find_dotenv())

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_AUTOML_TRIGGER_TOPIC = os.getenv(
    "KAFKA_AUTOML_TRIGGER_TOPIC", "automl-trigger-events"
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

# AutoML Vision configuration
VISION_AUTOML_HOST = os.getenv("VISION_AUTOML_HOST", "localhost")
VISION_AUTOML_PORT = os.getenv("VISION_AUTOML_PORT", "8002")
VISION_AUTOML_URL = f"http://{VISION_AUTOML_HOST}:{VISION_AUTOML_PORT}"
AUTOML_VISION_BEST_MODEL_URL = f"{VISION_AUTOML_URL}/automl_vision/best_model/"

# Log configuration at module load
logger.info("Configuration:")
logger.info(f"  Data Warehouse API: {API_BASE} (host: {DW_HOST}, port: {DW_PORT})")
logger.info(f"  AutoML Tabular: {AUTOML_TABULAR_BEST_MODEL_URL}")
logger.info(f"  AutoML Vision: {AUTOML_VISION_BEST_MODEL_URL}")
logger.info(
    "  Note: Running outside Docker - using localhost to connect to Docker services"
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
    Try to read CSV with multiple encodings

    Handles files with different encodings:
    - utf-8: Standard
    - latin-1 (ISO-8859-1): Western European
    - cp1252 (Windows-1252): Windows default
    - utf-16: Some Excel exports
    """

    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1", "utf-16"]

    for encoding in encodings:
        try:
            df = pd.read_csv(BytesIO(file_data), encoding=encoding)
            logger.info(f"Successfully read CSV with encoding: {encoding}")
            return df
        except (UnicodeDecodeError, Exception):
            continue

    # If all encodings fail, try with error handling
    try:
        df = pd.read_csv(BytesIO(file_data), encoding="utf-8", encoding_errors="ignore")
        logger.warning("Read CSV with 'ignore' errors - some characters may be missing")
        return df
    except Exception as e:
        logger.error(f"Failed to read CSV with all encodings: {e}")
        raise


def download_dataset_file(user_id: str, dataset_id: str, version: str = None) -> bytes:
    """Download dataset file (single file or folder as ZIP) - specific version or latest"""
    if version:
        url = f"{API_BASE}/datasets/{user_id}/{dataset_id}/version/{version}/download"
    else:
        url = f"{API_BASE}/datasets/{user_id}/{dataset_id}/download"
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


async def process_automl_trigger(event: dict) -> None:
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
            return

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
            logger.info("Dataset metadata fetched successfully")

            # Check if dataset is a folder or single file
            is_folder = metadata.get("is_folder", False)
            file_count = metadata.get("file_count", 1)

            logger.info(f"Dataset type: {'FOLDER' if is_folder else 'SINGLE FILE'}")
            if is_folder:
                logger.info(f"File count: {file_count}")

            # Download dataset file (AutoML service may do this too, but we validate it exists)
            logger.info(f"Downloading dataset file from: {API_BASE}")
            file_bytes = download_dataset_file(user_id, dataset_id, dataset_version)
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
            return
        except Exception as e:
            logger.error(f"Failed to fetch dataset: {e}")
            logger.error(f"   API Base URL: {API_BASE}")
            return

        # Step 2: Call AutoML service with all necessary information
        if task_category == "tabular":
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
                    "   This might indicate the service is not running or not accessible"
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
                logger.error("   Check: docker ps | grep tabular")
                logger.error("   Verify port mapping: docker port alfie_automl_tabular")
                raise
            except requests.exceptions.HTTPError as e:
                logger.error(f"AutoML service returned HTTP error: {e}")
                if e.response is not None:
                    logger.error(f"   Response status: {e.response.status_code}")
                    logger.error(f"   URL: {e.response.url}")
                    if e.response.content:
                        try:
                            error_detail = e.response.json()
                            logger.error(
                                f"   Response body: {json.dumps(error_detail, indent=2)}"
                            )
                        except Exception:
                            logger.error(f"   Response body: {e.response.text[:500]}")
                logger.error(
                    "   If running in Docker, ensure TABULAR_AUTOML_HOST=tabular is set"
                )
                raise
            except requests.exceptions.Timeout:
                logger.error(
                    f"Request to AutoML service timed out after {request_timeout} seconds"
                )
                logger.error("   Training may have taken longer than expected")
                logger.error(
                    "   Consider increasing time_budget or checking service logs"
                )
                raise

            response_data = r.json() if r.content else {}
            logger.info(
                "✅ AutoML processing completed and models uploaded to Data Warehouse"
            )
            logger.info(
                f"   Response: {json.dumps(response_data, indent=2, default=str)}"
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
                "task_type": task_type or "classification",
                "time_budget": time_budget_seconds,
                "model_size": (model_size or "small").strip().lower(),
            }
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
                raise
            except requests.exceptions.HTTPError as e:
                logger.error(f"AutoML Vision HTTP error: {e}")
                if e.response and e.response.content:
                    try:
                        logger.error(f"   Response: {e.response.json()}")
                    except Exception:
                        logger.error(f"   Response: {e.response.text[:500]}")
                raise
            except requests.exceptions.Timeout:
                logger.error(
                    f"AutoML Vision request timed out after {request_timeout}s"
                )
                raise

            response_data = r.json() if r.content else {}
            logger.info("✅ Vision AutoML completed; model uploaded to Data Warehouse")
            logger.info(
                f"   Response: {json.dumps(response_data, indent=2, default=str)}"
            )
        else:
            logger.error(f"Unsupported task category: {task_category}")
            return

    except Exception as e:
        logger.error(f"Error processing AutoML trigger event: {e}", exc_info=True)
        return


async def run_consumer() -> None:
    """Main consumer loop"""
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
            await process_automl_trigger(value)

    finally:
        await consumer.stop()
        logger.info("AutoML Trigger consumer stopped")


if __name__ == "__main__":
    try:
        asyncio.run(run_consumer())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
