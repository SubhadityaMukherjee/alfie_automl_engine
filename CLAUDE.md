# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup

```bash
uv sync
cp .env.template .env  # then fill in credentials
```

### Testing

```bash
uv run pytest -q                                          # fast tests only (default)
uv run pytest -q --full                                   # include @pytest.mark.full tests (slow, disk I/O)
uv run pytest tests/test_vision/ -v                       # specific module
uv run pytest -k "test_name" -v                           # by name pattern
uv run python test_services.py                            # integration tests (requires running services)
```

Tests marked `@pytest.mark.full` are skipped by default — they hit disk (real images, real DataLoaders) and can be slow. Pass `--full` to include them.

### Running services locally

```bash
uv run uvicorn app.tabular_automl.main:app --reload --port 8001
uv run uvicorn app.vision_automl.main:app --reload --port 8002
uv run uvicorn app.automlplus.main:app --reload --port 8003
```

### Docker

```bash
docker-compose up           # all services + Ollama
docker build -f app/Dockerfile -t alfie-automl:latest .
```

### Linting / formatting

```bash
black app/ tests/
isort app/ tests/
ruff check app/ tests/
```

---

## Architecture

Three independent FastAPI services, each on its own port, all integrating with an external **AutoDW** system for dataset retrieval and model storage.

### Service layout

```
app/
  core/               # Shared: ChatHandler (Ollama/Azure), Jinja2 prompt templates
  tabular_automl/     # Port 8001 — AutoGluon tabular AutoML
  vision_automl/      # Port 8002 — PyTorch/HF image classification
  automlplus/         # Port 8003 — web accessibility + image tools
```

Each service follows the same layered pattern:

- **`main.py`** — FastAPI app and route definitions
- **`services.py`** — orchestration logic (calls AutoDW, coordinates steps)
- **`models.py`** — Pydantic request/response schemas
- **`modules.py`** or **`ml_engine/`** — domain/ML logic
- **`db.py`** — SQLAlchemy session tracking (SQLite)

### Tabular service (`app/tabular_automl/`)

Single endpoint `POST /automl_tabular/best_model/`. Workflow: fetch metadata from AutoDW → download CSV/TSV/Parquet → validate columns/task type → `AutoMLTrainer` wraps AutoGluon `TabularPredictor` → zip model → upload to AutoDW. Returns a leaderboard in both JSON and Markdown.

### Vision service (`app/vision_automl/`)

Single endpoint `POST /automl_vision/best_model/`. Expects a ZIP uploaded to AutoDW containing images in class subdirectories (`root/<label>/<filename>`) plus a CSV mapping filenames to labels. Workflow: download ZIP → extract → validate → Optuna hyperparameter search over HF models → best model uploaded to AutoDW.

The `ml_engine/` sub-package contains:

- **`dataset.py`** — `ImageClassificationFromCSVDataset`: reads a CSV/DataFrame; images are resolved as `root_dir / label_name / filename`
- **`datamodule.py`** — `ClassificationData`: stratified train/val/test split, builds DataLoaders with HF `AutoImageProcessor` as the collate function
- **`model.py`** — `ClassificationModel`: thin `nn.Module` wrapping `AutoModelForImageClassification.from_pretrained`; supports backbone freezing
- **`trainer.py`** — `run_optuna_search()`: Optuna trial loop over model IDs and hyperparameters

Model size filtering (`small` ≤50M, `medium` ≤200M, `large` >200M) is driven by `MODEL_SMALL_MAX_PARAM_SIZE` / `MODEL_MEDIUM_MAX_PARAM_SIZE` env vars.

### AutoMLPlus service (`app/automlplus/`)

Endpoints for WCAG accessibility analysis, alt-text evaluation, and image-to-website generation. Uses `ChatHandler` to route LLM calls to either a local **Ollama** model or **Azure OpenAI** depending on `MODEL_BACKEND`. Prompts are Jinja2 templates under `app/core/prompt_templates/`.

### Key cross-cutting concerns

- **AutoDW integration** — all three services call AutoDW (default `localhost:8000`) to fetch dataset metadata, download files, and upload trained models. Tests mock these calls.
- **Environment-driven config** — ports, model IDs, DB paths, split ratios, and LLM backend are all read from `.env` at startup via `python-dotenv`.
- **Logging** — `app/__init__.py` configures a rotating file logger (`alfie_app.log`) on import; level and path are overridable via `ALFIE_LOG_LEVEL` / `ALFIE_LOG_FILE`.
- **Database** — each service tracks job sessions in its own SQLite file; paths are set via `TABULAR_DATABASE_CONFIG` and `VISION_DATABASE_CONFIG`.
