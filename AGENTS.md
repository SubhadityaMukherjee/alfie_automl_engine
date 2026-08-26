# AGENTS.md — Working Guide for the ALFIE AutoML Engine

Everything an AI agent (or new contributor) needs to work in this repo
productively.

## What It Is

A single FastAPI AutoML service for the ALFIE project. One app exposes a
unified `/automl` router with five services (tabular, vision, audio, text,
AutoML+) that all delegate model training to one consolidated ML engine.
Training services talk to AutoDW to fetch datasets and upload trained
models; AutoML+ calls Azure/OpenAI LLMs and VLMs.

- **Python**: `>=3.11` (Docker runtime 3.12)
- **Package manager**: uv — always `uv run <cmd>`, never raw pip/python
- **Branching**: branch off `develop`, never `main`. `develop` merges to
  `main` for releases.

## Architecture

```
app/
├── main.py                  # combined FastAPI app (single service, one port)
├── api.py                   # unified router: mounts services under /automl
│                            #   + GET /automl/endpoints (LLM-readable route listing)
├── core/                    # shared library
│   ├── config.py            # pydantic-settings over the env (single source of truth)
│   ├── exceptions.py        # AutoMLError hierarchy (validation/data/runtime/upload/...)
│   ├── api_errors.py        # exception -> HTTP status mapping (400/500/502 policy)
│   ├── concurrency.py       # offload() for blocking work in async endpoints
│   ├── process_log.py       # per-request process log in response payloads
│   ├── service_helpers.py   # AutoDW metadata fetch / download / upload
│   ├── dataset_extraction.py# shared ZIP+CSV+media extraction (structure only)
│   ├── chat_handler.py      # Azure/OpenAI chat facade
│   ├── logging.py           # structlog + rotating files
│   ├── health.py            # /health, /ready
│   ├── utils.py             # Jinja2 template rendering
│   ├── schemas/             # Pydantic response models (responses.py only)
│   └── prompt_templates/    # Jinja2 prompt + instruction templates
├── ml_engine/               # consolidated ML engine (ALL training code)
│   ├── configs/             # per-task JSON: candidate models + HPO ranges
│   ├── tasks.py             # task Pydantic models + per-modality slug sets
│   ├── dataset.py           # BaseCSVDataset + Torch datasets from CSV
│   ├── datamodule.py        # BaseDataModule + per-task datamodules + registry
│   ├── model.py             # HF AutoModelFor... wrappers
│   ├── trainer.py           # FabricTrainer (Lightning Fabric) + run_optuna_search
│                            #   + AutoGluonTrainer (tabular, AutoGluon)
│   ├── hpo/optuna_objectives.py  # one objective per task + OBJECTIVE_REGISTRY
│   ├── feature_mapping.py   # label maps / tokenizer vocab / scaler state extraction
│   ├── model_search.py      # HF hub discovery + small/medium/large tier filter
├── tabular_automl/          # endpoint package: router -> orchestrator -> services
├── vision_automl/           # endpoint package (image/video + multimodal)
├── audio_automl/            # endpoint package (audio_classification)
├── text_automl/             # endpoint package (text tasks; text_column plumbed)
└── automlplus/              # LLM/VLM tools (no training)
    ├── tools/               # static (readability), text (LLM), vlm
    └── website_accessibility/  # WCAG pipeline
```

### Services & endpoints

| Service        | Prefix                | ML backend                      | Notes                                                                                   |
| -------------- | --------------------- | ------------------------------- | --------------------------------------------------------------------------------------- |
| Tabular AutoML | `/automl/tabular`     | AutoGluon (`AutoGluonTrainer`)  | classification, regression, time series                                                 |
| Vision AutoML  | `/automl/vision`      | generic engine                  | image/video tasks + multimodal (image+tabular); **audio/text task types rejected here** |
| Audio AutoML   | `/automl/audio`       | generic engine                  | `audio_classification`                                                                  |
| Text AutoML    | `/automl/text`        | generic engine                  | classification, QA, causal/masked/seq2seq LM                                            |
| AutoML+        | `/automl/automl_plus` | Azure/OpenAI                    | accessibility, VLM tools                                                                |

Every service follows the same layering: thin `router.py` (bind form params)
→ `orchestrator.py` (fetch → resolve → download → extract → validate → train
→ serialize → upload; raises typed `AutoMLError`s) → `services.py`
(validation, training wrappers, packaging, templates) → `app/ml_engine`.
Blocking work goes through `offload()`.

### Key dependencies

FastAPI/uvicorn · AutoGluon · torch/torchvision/lightning · timm ·
transformers/peft · scikit-learn · pandas · huggingface-hub ·
openai/azure-ai-inference · structlog.

## Testing

Test tree mirrors `app/`: `tests/test_ml_engine`, `test_tabular`,
`test_vision`, `test_audio`, `test_text`, `test_core`, `test_automlplus`,
`test_integration`.

```bash
uv run pytest -q            # fast suite (default; skips @pytest.mark.full)
uv run pytest -q --full     # everything, including long-running tests
uv run pytest -q tests/test_ml_engine                       # one area
uv run pytest -q tests/test_tabular/test_automl_trainer.py  # one file
```

- Router/integration tests patch orchestrator-level functions
  (`@patch("app.<service>.orchestrator.<fn>")`) — no real training or network.
- CI (`.github/workflows/testing.yml`) runs ruff, ruff format, black, mypy,
  pre-commit, and pytest with `--cov-fail-under=60` and
  `JINJAPATH=app/core/prompt_templates`.
- Local env needs the same `JINJAPATH` (copy `.env.template` to `.env`).

### Quality gates — run before pushing

```bash
uv run ruff check app tests
uv run ruff format --check app tests
uv run mypy app/
uv run pre-commit run --all-files   # ruff + format + mypy(app)
```

## Deploying

### Local run

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

Port is `AUTOML_ENGINE_PORT` (default 8001). `test_services.py` boots the
service and fires curl smoke checks per scenario
(`uv run python test_services.py audio`); `test_services_docker.sh` does the
same against an already-running container.

### Docker

```bash
docker compose up --build      # dev: uvicorn --reload, AutoDW via host.docker.internal
```

- `app/Dockerfile`: uv-based image (nonroot, healthcheck on `/health`,
  `HEALTHCHECK_PORT` overridable).
- `.github/workflows/docker-build.yml`: on `v*.*.*` tags, builds
  multi-arch (amd64/arm64) and pushes to the GitLab registry
  (`gitlab.catalink.eu:5050/external/alfie_eu/alfie/automl_engine`).

### Releases

1. Merge feature work to `develop`, then `develop` to `main` (tests green).
2. Tag + release from `main`:

   ```bash
   gh release create vX.Y.Z --target main --title "..." --notes "..."
   ```

   The tag triggers the Docker image build. (Verify the tag points at the
   pushed merge commit — create releases only after pushing.)

### CI/CD summary

- `testing.yml` — lint + type-check + pytest on push/PR to main/develop
- `docker-build.yml` — image build/push on version tags
- `pages.yml` — MkDocs docs to GitHub Pages on push to main touching `docs/`

## Conventions

- uv only (`uv run`, `uv sync --locked`). Lockfile is committed.
- Ruff (lint + format, line-length 88), Black also checked in CI, mypy on
  `app/`. Logging uses lazy `%`-style (`logger.info("x: %s", v)`) — do not
  switch to f-strings in log calls.
- Typed errors: raise `AutoMLValidationError`/`AutoMLDataError`/
  `AutoMLRuntimeError`/`AutoDWDownloadError`/`AutoDWUploadError` from
  `app.core.exceptions`; routers translate via `automl_exception_to_response`.
- New endpoints: follow the router → orchestrator → services layering; add
  task types to `app/ml_engine/tasks.py` slug sets + a config JSON + an
  Optuna objective registered in `OBJECTIVE_REGISTRY`.
- Per-modality file-type validation lives in the modality's services
  (`collect_non_image_files` / `collect_non_audio_files`); shared extraction
  in `app/core/dataset_extraction.py` checks structure only.

## Known gaps / future work

Carried over from the production-readiness and modernization audits — still
open:

- **Security**: no auth/rate limiting on any endpoint; SSRF risk in
  AutoML+ user-supplied URLs; no upload size limits; `user_id`/`dataset_id`
  interpolated into AutoDW URLs unsanitized; tabular predictor serialized
  with pickle; historical secret leakage in git history (rotate if not
  done).
- **Kafka consumers** (`kafka_automl_consumer_example_v*.py`): no TLS/SASL,
  ZIP path traversal in extraction, auto-commit can lose messages.
- **Modernization leftovers**: `Union[...]`/`Optional[...]` in
  `app/ml_engine/dataset.py`; `typing.Dict/List` imports in `automlplus`;
  `ChatHandler` static methods that could be module-level functions;
  `--reload` in the compose command (dev-only).
- **Ops**: no Prometheus metrics, no CORS, no API versioning, no concurrent
  training-job limits.
