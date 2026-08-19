# Architecture

The ALFIE AutoML Engine is a single FastAPI service that mounts three service
modules plus a shared core library under one unified `/automl` router, all
living under `app/`. The training services talk to [AutoDW](autodw.md) to fetch
datasets and upload trained models.

![System components](images/flow.png)

```mermaid
flowchart TB
    User(["Users / ALFIE web app"])

    subgraph Engine["ALFIE AutoML Engine"]
        subgraph Core["app/core - shared library"]
            direction LR
            COREUTILS["config · logging · typed errors · concurrency"]
            DWHELP["AutoDW helpers"]
            CHAT["ChatHandler (Azure AI)"]
        end

        subgraph Tabular["app/tabular_automl · /automl/tabular"]
            TG["AutoGluon<br>classification · regression · time series"]
        end

        subgraph Vision["app/vision_automl · /automl/vision"]
            direction TB
            HPO["Optuna HPO"]
            FABRIC["Lightning Fabric trainer"]
            HFM["Hugging Face models<br>image · video · audio · text · multimodal"]
            HPO --> FABRIC
            FABRIC --> HFM
        end

        subgraph Plus["app/automlplus · /automl/automl_plus"]
            direction LR
            ACC["web accessibility + readability"]
            VLMT["VLM tools<br>alt text · image prompts · image-to-website"]
        end
    end

    AutoDW[("AutoDW<br>dataset and model store")]

    User --> Tabular
    User --> Vision
    User --> Plus
    Tabular --> Core
    Vision --> Core
    Plus --> Core
    Tabular -- "fetch datasets / upload models" --> AutoDW
    Vision -- "fetch datasets / upload models" --> AutoDW
    Plus -- "LLM/VLM calls" --> CHAT
```

## Unified API

`app/main.py` builds a single FastAPI app and `app/api.py` mounts every
service router under one `/automl` prefix (e.g.
`POST /automl/automl_plus/accepted_format/`). Health endpoints (`/health`,
`/ready`) stay at the root. The app runs as one service on a single port
(see [running the services](running_the_services.md)).

| Service | Module | Prefix | What it does |
| --- | --- | --- | --- |
| Tabular AutoML | `app/tabular_automl` | `/automl/tabular` | Trains AutoGluon models on CSV/TSV/Parquet data (classification, regression, time series). |
| Vision AutoML | `app/vision_automl` | `/automl/vision` | Optuna hyperparameter search + Lightning Fabric training over Hugging Face models (image, video, audio, and text tasks, plus multimodal image+tabular). |
| AutoML+ | `app/automlplus` | `/automl/automl_plus` | LLM/VLM tools: web accessibility analysis, alt-text checking, image prompts, readability metrics. No model training. |

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

Every service module follows the same layering, so the HTTP layer stays thin:

```
app/main.py         FastAPI app + lifespan (logging setup)
   └─ app/api.py            unified router: mounts each service under /automl/<service>
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
│   ├── main.py                # combined FastAPI app (single service)
│   ├── api.py                 # unified router mounting every service under /automl
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
├── docker-compose.yml         # single API container
├── Dockerfile
├── mkdocs.yml
└── pyproject.toml
```

Runtime output (not in git): `logs/` holds per-service log files and
`uploaded_data/` holds upload artifacts.
