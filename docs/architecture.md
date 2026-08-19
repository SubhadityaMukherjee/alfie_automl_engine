# Architecture

The ALFIE AutoML Engine is a collection of three independent FastAPI services
plus a shared core library, all living under `app/`. The training services talk
to [AutoDW](autodw.md) to fetch datasets and upload trained models.

![System components](images/flow.png)

## Services

Each service is its own FastAPI app with its own router prefix, run separately
(see [running the services](running_the_services.md)).

| Service | Module | Prefix | What it does |
| --- | --- | --- | --- |
| Tabular AutoML | `app/tabular_automl` | `/automl_tabular` | Trains AutoGluon models on CSV/TSV/Parquet data (classification, regression, time series). |
| Vision AutoML | `app/vision_automl` | `/automl_vision` | Optuna hyperparameter search + Lightning Fabric training over Hugging Face models (image, video, audio, and text tasks, plus multimodal image+tabular). |
| AutoML+ | `app/automlplus` | `/automlplus` | LLM/VLM tools: web accessibility analysis, alt-text checking, image prompts, readability metrics. No model training. |

## Shared core (`app/core`)

- `config.py` — typed settings over the environment (single source of truth for env vars).
- `concurrency.py` — `offload()` runs blocking work in the Starlette threadpool so async endpoints don't block the event loop.
- `exceptions.py` — the `AutoMLError` hierarchy (validation, data, runtime, upload, ...).
- `api_errors.py` — maps those exceptions to HTTP status codes (400 / 500 / 502 policy) in one place.
- `logging.py` — structlog setup plus per-service rotating log files.
- `chat_handler.py` — facade over Azure AI Inference chat models (sync, async, streaming, with images).
- `service_helpers.py` — AutoDW dataset metadata fetch / download / model upload / payload building.
- `health.py` — `/health` and `/ready` endpoints.
- `utils.py` — Jinja2 prompt template rendering.
- `schemas/` — Pydantic task/response models and the base CSV dataset/datamodule classes used by vision.

## Layering inside a service

Every service follows the same layering, so the HTTP layer stays thin:

```
main.py          FastAPI app + lifespan (logging setup)
   └─ router.py           bind HTTP form params, delegate, map exceptions to responses
        └─ orchestrator.py   the multi-step pipeline, raises typed exceptions
             └─ services.py     validation, training wrappers, packaging, templates
```

The training pipelines in both `tabular_automl` and `vision_automl` share the
same shape:

1. Fetch dataset metadata from AutoDW.
2. Resolve the correct download URL (respecting dataset splits).
3. Download the dataset to a temporary directory (vision also extracts the ZIP).
4. Validate user-supplied parameters against the dataset.
5. Train within the given time budget.
6. Serialize and zip the model artifacts.
7. Upload the model and leaderboard back to AutoDW.

Blocking steps (network, training) are pushed off the event loop via `offload`.

## Vision ML engine (`app/vision_automl/ml_engine` and `hpo/`)

- `configs/` — one JSON file per task type with candidate models and
  hyperparameter search ranges.
- `dataset.py` — Torch datasets reading samples from CSV.
- `datamodule.py` — per-task datamodules (splits, preprocessing, DataLoaders)
  and `DATAMODULE_REGISTRY`.
- `model.py` — thin wrappers around Hugging Face `AutoModelFor...` classes.
- `trainer.py` — `FabricTrainer` (Lightning Fabric training loop with early
  stopping, time limits, and Optuna pruning) and `run_optuna_search`.
- `hpo/optuna_objectives.py` — one Optuna objective per task type plus
  `OBJECTIVE_REGISTRY`; `run_optuna_search` dispatches through the registry.

A `feature_mapping.json` (label maps, tokenizer vocab, scaler/encoder state)
is saved alongside each trial so the preprocessing pipeline can be reproduced
when loading a trained model (see [loading the model](loading_model.md)).

## Repository layout

```
alfie_automl_engine/
├── app/
│   ├── core/                  # shared library (config, logging, errors, AutoDW helpers, schemas)
│   │   ├── schemas/           # pydantic models + base dataset/datamodule classes
│   │   └── prompt_templates/  # Jinja2 prompt and instruction templates
│   ├── tabular_automl/        # tabular AutoML service
│   ├── vision_automl/         # vision AutoML service
│   │   ├── ml_engine/         # datasets, datamodules, models, trainer, configs
│   │   │   └── configs/       # per-task hyperparameter JSON configs
│   │   └── hpo/               # Optuna objectives
│   └── automlplus/            # AutoML+ service
│       ├── tools/             # static (readability), text (LLM), vlm tools
│       └── website_accessibility/
├── docs/                      # mkdocs site (this documentation)
├── tests/                     # pytest suite
├── sample_data/               # datasets fetched by download_sample_data.py
├── plans/                     # planning notes
├── docker-compose.yml         # one container per service
├── Dockerfile
├── mkdocs.yml
└── pyproject.toml
```

Runtime output (not in git): `logs/` holds per-service log files and
`uploaded_data/` holds upload artifacts.
