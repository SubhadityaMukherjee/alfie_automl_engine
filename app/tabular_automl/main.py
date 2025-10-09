"""FastAPI endpoints for tabular AutoML workflows.

Provides endpoints to accept user data/config, validate inputs, store
session metadata, and trigger AutoML training using AutoGluon.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.chat_handler import ChatHandler
from app.tabular_automl.modules import AutoMLTrainer
from typing import Annotated
from app.tabular_automl.services import (
    create_session_directory,
    load_table,
    save_upload,
    store_session_in_db,
    validate_tabular_inputs,
)

logger = logging.getLogger(__name__)


load_dotenv(find_dotenv())

app = FastAPI()

TABULAR_AUTOML_PORT = os.getenv("TABULAR_AUTOML_PORT", "http://localhost:8001")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize resources
    await ChatHandler.init()
    yield
    # Cleanup resources
    pass


# # NOTE : I AM NOT SURE IF THE AUTODW WILL HANDLE THIS PART FIRST :/
class SessionRequest(BaseModel):
    """Payload for initiating model search/training for a session."""

    session_id: str

@app.post("/automl_tabular/best_model_mvp/")
async def find_best_model_for_mvp(
    train_file: Annotated[UploadFile, File(..., description="Training dataset file (CSV/TSV/Parquet)")],
    test_file: Annotated[UploadFile | None, File(..., description="Optional test dataset file (CSV/TSV/Parquet)")] = None,
    target_column_name: Annotated[str, Form(..., description="Name of the target column")] = "",
    time_stamp_column_name: Annotated[str | None, Form(..., description="Timestamp column (required for time-series tasks)")] = None,
    task_type: Annotated[str, Form(..., description="Type of ML task", examples=["classification", "regression", "time_series"])] = "classification",
    time_budget: Annotated[int, Form(..., description="Time budget in seconds")] = 10
) -> JSONResponse:
    """
        Create a session, upload data, validate inputs, store metadata,
        train AutoML on stored session data, and return leaderboard.

    ## Endpoint: `/automl_tabular/best_model_mvp/`

    ### Purpose

        This endpoint is a **single-step AutoML orchestration function**.
        It allows a client to:

        1. Upload tabular training (and optionally test) datasets.
        2. Specify metadata (task type, target column, optional timestamp column, time budget).
        3. Validate and persist the uploaded files and metadata in a session.
        4. Launch an AutoML training process using the provided inputs.
        5. Return the AutoML leaderboard (best models and their metrics) together with the session ID for later retrieval.

        ---

    ### Input Parameters

        The endpoint accepts a **multipart/form-data** request because it mixes file uploads and form fields.

    #### **1. `train_file` (required)**

        * **Type:** `UploadFile` (`.csv`, `.tsv`, `.parquet`)
        * **Description:** The primary training dataset containing both features and the target column.
        * **Constraints:**

          * Must end with `.csv`, `.tsv`, or `.parquet`.
          * Must contain the `target_column_name` specified in the request.
          * File must be a valid table (rows = samples, columns = features).
        * **Example:**

          * Filename: `train.csv`
          * Content (simplified):

            ```csv
            age,income,gender,default
            25,50000,Male,0
            42,80000,Female,1
            37,62000,Female,0
            ```

            → Here, `default` could be the `target_column_name`.

        ---

    #### **2. `test_file` (optional)**

        * **Type:** `UploadFile` (`.csv`, `.tsv`, `.parquet`)
        * **Description:** Optional test dataset for evaluating trained models. If omitted, the AutoML system will internally perform cross-validation or split the training file.
        * **Constraints:**

          * If provided, must contain the same feature columns as `train_file`.
          * Target column may be included or omitted depending on use case (if included, used for scoring).
        * **Example:**

          * Filename: `test.csv`
          * Content:

            ```csv
            age,income,gender
            30,55000,Male
            29,47000,Female
            ```

        ---

    #### **3. `target_column_name` (required)**

        * **Type:** `str`
        * **Description:** The name of the column in `train_file` that contains the **supervised learning target**.
        * **Constraints:**

          * Must exactly match a column header in `train_file`.
          * For classification: categorical values (e.g., `"yes"`, `"no"` or integers like `0`, `1`).
          * For regression: numeric values.
          * For time series: numeric or categorical values indexed over `time_stamp_column_name`.
        * **Example:**

          * `"default"` in the dataset above.
          * `"price"` for regression tasks.
          * `"sales"` for time-series tasks.

        ---

    #### **4. `time_stamp_column_name` (optional, required if `task_type="time_series"`)**

        * **Type:** `str | None`
        * **Description:** Column name in `train_file` representing the temporal ordering of samples. Used **only** for time-series tasks.
        * **Constraints:**

          * Must be present in the dataset if task type = `time_series`.
          * Values must represent time (ISO8601 strings, Unix timestamps, or sortable integers).
        * **Example:**

          * `"date"` column containing values like `2021-01-01`, `2021-01-02`, ...
          * `"timestamp"` column with UNIX epoch values (`1632441600`).

        ---

    #### **5. `task_type` (required)**

        * **Type:** `str` (enum)
        * **Description:** Defines what type of supervised learning task the AutoML system should solve.
        * **Allowed Values:**

          * `"classification"` → Predicts discrete categories (binary or multiclass).
          * `"regression"` → Predicts continuous numeric values.
          * `"time_series"` → Predicts future values in a temporal sequence, requires `time_stamp_column_name`.
        * **Examples:**

          * `"classification"` → Predict whether a customer defaults on a loan.
          * `"regression"` → Predict house prices based on features.
          * `"time_series"` → Forecast sales for the next month using past sales history.

        ---

    #### **6. `time_budget` (required)**

        * **Type:** `int` (seconds)
        * **Description:** Maximum wall-clock time (in seconds) that the AutoML system is allowed to train models.
        * **Constraints:**

          * Must be a positive integer.
          * The AutoML engine may terminate early if time budget is exceeded.
        * **Examples:**

          * `600` → Allow up to 10 minutes of training.
          * `3600` → Allow up to 1 hour of training.

        ---

    ### Processing Workflow

        1. **Session Creation**

           * A new session ID and working directory are created.
           * Input files are saved into the session directory (`train.csv`, `test.csv`).

        2. **Validation**

           * File extensions are checked.
           * `target_column_name` and `time_stamp_column_name` (if applicable) are validated against dataset headers.
           * For time-series tasks, timestamp format is checked.

        3. **Persistence**

           * Session metadata is stored in a database for reproducibility:

             * Session ID
             * Train/Test file paths
             * Target column
             * Task type
             * Time budget

        4. **AutoML Training**

           * A new `AutoMLTrainer` instance is initialized.
           * Data is loaded into DataFrames.
           * Models are trained within the `time_budget`.
           * Results (leaderboard) are produced with performance metrics per candidate model.

        5. **Response**

           * Returns both session information and leaderboard results.

        ---

    ### Response Format

    #### **Success Response (200)**

        ```json
        {
          "message": "AutoML training completed successfully.",
          "session_id": "sess_12345",
          "leaderboard": "| model | accuracy | fit_time |\n|-------|----------|----------|\n| XGBoost | 0.89 | 32.5 |\n| RandomForest | 0.85 | 28.1 |"
        }
        ```

        * `leaderboard` is a Markdown table if `pandas.DataFrame`, or a JSON structure if natively returned by AutoML engine.

        ---

    #### **Validation Error (400 / 422)**

        ```json
        {
          "error": "Field 'time_stamp_column_name' is required for time series tasks."
        }
        ```

        ---

    #### **Session Not Found / Internal Error (404 / 500)**

        ```json
        {
          "error": "Session not found"
        }
        ```

        or

        ```json
        {
          "error": "Unexpected error: <details>"
        }
        ```
    """
    try:
        #  Create session directory
        session_id, session_dir = create_session_directory()

        #  Validate training file type
        if train_file.filename is not None:
            if not train_file.filename.endswith((".csv", ".tsv", ".parquet")):
                logger.error("Input file type is wrong")
                return JSONResponse(
                    status_code=400,
                    content={"error": "Training file is invalid, must be CSV/TSV/Parquet"},
                )

        #  Save uploaded files
        provided_filename = train_file.filename or "train.csv"
        train_suffix = Path(provided_filename).suffix or ".csv"
        train_path = session_dir / f"train{train_suffix}"
        logger.debug(f"Train path {train_path}")
        save_upload(train_file, train_path)

        test_path = None
        if test_file:
            test_filename = test_file.filename or "test.csv"
            test_suffix = Path(test_filename).suffix or ".csv"
            test_path = session_dir / f"test{test_suffix}"
            save_upload(test_file, test_path)
            logger.debug(f"Test path {test_path}")

        #  Validate tabular inputs
        validation_error = validate_tabular_inputs(
            train_path=train_path,
            target_column_name=target_column_name,
            time_stamp_column_name=time_stamp_column_name,
            task_type=task_type,
        )
        logger.debug("Tabular input validated")
        if validation_error:
            logger.error(f"Tabular info wrong {validation_error}")
            return JSONResponse(status_code=400, content={"error": validation_error})

        #  Store session metadata in DB
        store_session_in_db(
            session_id=session_id,
            train_path=train_path,
            test_path=test_path,
            target_column_name=target_column_name,
            time_stamp_column_name=time_stamp_column_name,
            task_type=task_type,
            time_budget=time_budget,
        )
        logger.debug("session stored in db")

        #  Train AutoML
        save_model_path = session_dir / "automl_data_path"
        os.makedirs(save_model_path, exist_ok=True)
        logger.debug("Temp model path created")

        trainer = AutoMLTrainer(save_model_path=save_model_path)
        logger.debug("Tabular Trainer created")

        train_df = load_table(train_path)
        test_df = load_table(test_path) if test_path else None
        logger.debug("Tables loaded")

        leaderboard = trainer.train(
            train_df=train_df,
            test_df=test_df,
            target_column=target_column_name,
            time_limit=int(time_budget),
        )
        logger.debug(f"Found best model {leaderboard}")

        #  Return leaderboard + session info
        return JSONResponse(
            status_code=200,
            content={
                "message": "AutoML training completed successfully.",
                "session_id": session_id,
                "leaderboard": (
                    leaderboard.to_markdown()
                    if isinstance(leaderboard, pd.DataFrame)
                    else leaderboard
                ),
            },
        )

    except Exception as e:
        logger.error(f"Could not find best model {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
