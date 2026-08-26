# Available endpoints

All services are exposed by a single FastAPI app behind the unified `/automl`
router (see [architecture](architecture.md)). The app listens on
`AUTOML_ENGINE_PORT` (default `8001`), so the base URL is typically
`http://localhost:8001`. Interactive OpenAPI docs are served at `/docs` and
`/redoc`.

All service endpoints are `POST`. Parameters are sent as
`multipart/form-data` form fields (file uploads included) unless noted
otherwise.

---

## Health (root prefix)

| Method | Path    | Description                                      |
| ------ | ------- | ------------------------------------------------ |
| GET    | `/health` | Liveness probe — reports that the service process is up. |
| GET    | `/ready`  | Readiness probe — reports that the service can handle requests. |

---

## Tabular AutoML (`/automl/tabular`)

Trains AutoGluon models on CSV/TSV/Parquet datasets fetched from AutoDW.

| Method | Path | Description |
| ------ | ---- | ----------- |
| POST | `/automl/tabular/deployment_instructions/` | Return rendered deployment instructions for using a trained tabular model (Jinja2 template). |
| POST | `/automl/tabular/accepted_format/` | Return the accepted tabular dataset formats (CSV, TSV, Parquet) and required column structure. |
| POST | `/automl/tabular/best_model/` | Fetch a dataset from AutoDW, run AutoML training within a time budget, and upload the best model + leaderboard back to AutoDW. |

### `best_model` parameters

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `user_id` | str | *required* | AutoDW user identifier |
| `dataset_id` | str | *required* | AutoDW dataset identifier |
| `dataset_version` | str | `v1` | Dataset version |
| `target_column_name` | str | *required* | Column to predict |
| `time_stamp_column_name` | str | `None` | Timestamp column (required for time-series tasks) |
| `task_type` | str | `classification` | One of the supported tabular task types (see OpenAPI docs) |
| `time_budget` | int | `10` | Training time budget in seconds |
| `num_cpus` | int \| `auto` | `auto` | CPU count for AutoML |
| `num_gpus` | int \| `auto` | `auto` | GPU count for AutoML |
| `dataset_split` | str | `None` | AutoDW dataset split to train on (e.g. `train`) |

Responses: `200` success (message + leaderboard), `400` validation error,
`502` AutoDW communication failure, `500` unexpected error.

---

## Vision AutoML (`/automl/vision`)

Optuna hyperparameter search + Lightning Fabric training over Hugging Face
models, on image/video ZIP datasets fetched from AutoDW. Training is delegated
to the consolidated ML engine (`app/ml_engine`).

| Method | Path | Description |
| ------ | ---- | ----------- |
| POST | `/automl/vision/deployment_instructions/` | Return rendered deployment instructions for using a trained vision model. |
| POST | `/automl/vision/accepted_format/` | Return the accepted vision dataset format (ZIP structure + labels CSV). |
| POST | `/automl/vision/best_model/` | Fetch a vision dataset from AutoDW, train image/video models within a time budget, and upload the best model + leaderboard to AutoDW. |
| POST | `/automl/vision/multimodal_best_model/` | Same pipeline, but trains on images **plus** auxiliary tabular CSV columns (auto-detected) as extra features. |

### `best_model` parameters

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `user_id` | str | *required* | AutoDW user identifier |
| `dataset_id` | str | *required* | AutoDW dataset identifier |
| `dataset_version` | str | `v1` | Dataset version |
| `filename_column` | str | `filename` | CSV column holding image filenames |
| `label_column` | str | `label` | CSV column holding class labels |
| `task_type` | str | `image_classification` | One of the image/video task types: `image_classification`, `image_segmentation`, `object_detection`, `video_classification`, `keypoint_detection` |
| `time_budget` | int | `60` | Training time budget in seconds |
| `model_size` | str | `small` | `small` (≤ 50M params), `medium` (≤ 200M), or `large` |
| `num_cpus` | int \| `auto` | `auto` | CPU count for AutoML |
| `num_gpus` | int \| `auto` | `auto` | GPU count for AutoML |
| `dataset_split` | str | `None` | AutoDW dataset split to train on |

### `multimodal_best_model` additional parameters

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `exclude_columns` | str | `None` | Comma-separated CSV columns to exclude from auxiliary features (all other non-filename/label columns are used; numeric columns are scaled, categorical columns encoded) |

Responses for both endpoints: `200` success (message + leaderboard;
multimodal also returns the detected `auxiliary_columns`), `400` validation
error, `502` AutoDW communication failure, `500` unexpected error.

> **Note:** audio and text task types are no longer accepted on the vision
> endpoint — use `/automl/audio/best_model/` and `/automl/text/best_model/`
> instead.

---

## Audio AutoML (`/automl/audio`)

Audio classification with the same Optuna + Lightning Fabric pipeline over
Hugging Face models, on audio ZIP datasets fetched from AutoDW.

| Method | Path | Description |
| ------ | ---- | ----------- |
| POST | `/automl/audio/deployment_instructions/` | Return rendered deployment instructions for using a trained audio model. |
| POST | `/automl/audio/accepted_format/` | Return the accepted audio dataset format (ZIP structure + labels CSV + audio folder). |
| POST | `/automl/audio/best_model/` | Fetch an audio dataset from AutoDW, train an audio model within a time budget, and upload the best model + leaderboard to AutoDW. |

### `best_model` parameters

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `user_id` | str | *required* | AutoDW user identifier |
| `dataset_id` | str | *required* | AutoDW dataset identifier |
| `dataset_version` | str | `v1` | Dataset version |
| `filename_column` | str | `filename` | CSV column holding audio filenames |
| `label_column` | str | `label` | CSV column holding class labels |
| `task_type` | str | `audio_classification` | Audio task type (`audio_classification`) |
| `time_budget` | int | `60` | Training time budget in seconds |
| `model_size` | str | `small` | `small` (≤ 50M params), `medium` (≤ 200M), or `large` |
| `num_cpus` | int \| `auto` | `auto` | CPU count for AutoML |
| `num_gpus` | int \| `auto` | `auto` | GPU count for AutoML |
| `dataset_split` | str | `None` | AutoDW dataset split to train on |

Responses: `200` success (message + leaderboard), `400` validation error,
`502` AutoDW communication failure, `500` unexpected error.

---

## Text AutoML (`/automl/text`)

Text tasks with the same Optuna + Lightning Fabric pipeline over Hugging Face
models, on CSV-in-ZIP datasets fetched from AutoDW.

| Method | Path | Description |
| ------ | ---- | ----------- |
| POST | `/automl/text/deployment_instructions/` | Return rendered deployment instructions for using a trained text model. |
| POST | `/automl/text/accepted_format/` | Return the accepted text dataset formats per task type. |
| POST | `/automl/text/best_model/` | Fetch a text dataset from AutoDW, train a text model within a time budget, and upload the best model + leaderboard to AutoDW. |

### `best_model` parameters

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `user_id` | str | *required* | AutoDW user identifier |
| `dataset_id` | str | *required* | AutoDW dataset identifier |
| `dataset_version` | str | `v1` | Dataset version |
| `text_column` | str | `text` | CSV column holding the input text (used for `text_classification`) |
| `label_column` | str | `label` | CSV column holding class labels (used for `text_classification`) |
| `task_type` | str | `text_classification` | One of: `text_classification`, `question_answering`, `causal_lm`, `seq2seq_lm`, `masked_lm` |
| `time_budget` | int | `60` | Training time budget in seconds |
| `model_size` | str | `small` | `small` (≤ 50M params), `medium` (≤ 200M), or `large` |
| `num_cpus` | int \| `auto` | `auto` | CPU count for AutoML |
| `num_gpus` | int \| `auto` | `auto` | GPU count for AutoML |
| `dataset_split` | str | `None` | AutoDW dataset split to train on |

Required CSV columns per task type: `text` + `label` for
`text_classification` (the text column name is configurable via
`text_column`); `question`, `context`, `answer_start`, `answer_text` for
`question_answering`; `text` for `causal_lm` / `masked_lm`; `input_text` +
`target_text` for `seq2seq_lm`.

Responses: `200` success (message + leaderboard), `400` validation error,
`502` AutoDW communication failure, `500` unexpected error.

---

## AutoML+ (`/automl/automl_plus`)

LLM/VLM-powered tools — no model training.

| Method | Path | Description |
| ------ | ---- | ----------- |
| POST | `/automl/automl_plus/accepted_format/` | Return the accepted input formats for the AutoML+ tools (images, HTML files, URLs). |
| POST | `/automl/automl_plus/image_tools/image_to_website/` | Convert an uploaded image into a basic HTML website structure. **Currently returns 501 Not implemented.** |
| POST | `/automl/automl_plus/image_tools/run_on_image/` | Run a vision-language model on an image (file upload or URL) with a prompt and return the text output as JSON. |
| POST | `/automl/automl_plus/image_tools/run_on_image_stream/` | Same as `run_on_image`, but streams the model output as plain text (`text/plain`) instead of returning a single JSON blob. |
| POST | `/automl/automl_plus/web_access/check-alt-text/` | Evaluate a provided alt-text string against the referenced image using an LLM. |
| POST | `/automl/automl_plus/web_access/analyze/` | Run WCAG-inspired accessibility checks (and optional readability analysis) on a website, given an uploaded HTML file or a URL. Returns per-element results and an average accessibility score. |

### Key parameters

| Endpoint | Fields |
| -------- | ------ |
| `image_tools/run_on_image/` (and `_stream/`) | `prompt` (required), `model` (optional override), and exactly one of `image_file` (upload) or `image_url` |
| `web_access/check-alt-text/` | `image_url` (required), `alt_text` (required) |
| `web_access/analyze/` | `file` (HTML upload) and/or `url` (website to fetch); optional `extra_file_input` for extra LLM context (e.g. WCAG guidelines) |

Responses: `400` missing/invalid input, `500` internal error (LLM/VLM or
fetch failure).

---

## Common conventions

- **Trailing slashes matter**: endpoints are registered with a trailing
  slash; omitting it triggers a `307` redirect.
- **`X-Task-ID` header**: the five training endpoints (`tabular/best_model`,
  `vision/best_model`, `vision/multimodal_best_model`, `audio/best_model`,
  `text/best_model`) accept an optional `X-Task-ID` header used for request
  tracking.
- **Error shape**: errors are returned as `{"error": "<message>"}` with the
  status codes listed per service above.
- Runnable `curl` examples live in
  [running the services](running_the_services.md); Docker smoke tests in
  `test_services_docker.sh`.
