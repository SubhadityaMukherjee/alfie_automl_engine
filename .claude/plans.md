# Implementation Plans

Generated from `tasklist.md`. Mark tasks complete here and in tasklist.md when done.

---

## TASK 1 (HIGH) — FastAPI Routers Across All Packages

**Goal:** Extract endpoints into `router.py` files per service. Keep `main.py` thin (app init + lifespan + `include_router`).

**Status:** TODO

### Steps

1. Create `app/tabular_automl/router.py`
   - `router = APIRouter(prefix="/automl_tabular", tags=["tabular"])`
   - Move `POST /automl_tabular/best_model/` handler out of `main.py` (currently lines ~42–174)
   - All imports and dependencies move with it

2. Create `app/vision_automl/router.py`
   - `router = APIRouter(prefix="/automl_vision", tags=["vision"])`
   - Move `POST /automl_vision/best_model/` handler out of `main.py` (lines ~52–179)

3. Create `app/automlplus/router.py`
   - `router = APIRouter(prefix="/automlplus", tags=["automlplus"])`
   - Move all 5 endpoints from `main.py`:
     - `POST /automlplus/image_tools/image_to_website/`
     - `POST /automlplus/web_access/check-alt-text/`
     - `POST /automlplus/image_tools/run_on_image/`
     - `POST /automlplus/image_tools/run_on_image_stream/`
     - `POST /automlplus/web_access/analyze/`

4. Update each `main.py`:
   - Remove moved endpoint handlers
   - Add `from .router import router` + `app.include_router(router)`
   - Keep lifespan, app init, and startup logic only

5. Verify all existing tests still pass (endpoint paths must not change).

---

## TASK 2 (HIGH) — Unified Per-Service Logging to Rotating Log Files

**Goal:** Each service writes to its own rotating log file (`logs/<service>.log`) instead of everything going to one global `alfie_app.log`.

**Status:** TODO

### Steps

1. Create `app/core/logging.py`:

   ```python
   def configure_service_logging(service_name: str) -> None:
       """Set up a RotatingFileHandler for the given service name.
       Reads ALFIE_LOG_DIR (default ./logs) and ALFIE_LOG_LEVEL (default INFO).
       Creates logs/<service_name>.log, maxBytes=10MB, backupCount=5.
       """
   ```

2. Call `configure_service_logging("<name>")` inside each service's lifespan startup:
   - `automlplus/main.py` — lifespan already exists (line ~46), add call there
   - `tabular_automl/main.py` — add lifespan context manager, call there
   - `vision_automl/main.py` — lifespan stub exists (line ~8), add call there

3. Downgrade `app/__init__.py` global logger to a fallback only (no file handler by default); let services set up their own handlers.

4. Add `ALFIE_LOG_DIR` env var to `.env.template` (default `./logs`). Existing `ALFIE_LOG_LEVEL` and `ALFIE_LOG_FILE` remain supported.

5. Verify: run each service and confirm `logs/automlplus.log`, `logs/tabular_automl.log`, `logs/vision_automl.log` are created.

---

## TASK 4 (MEDIUM) — Make AutoML+ Tools More Modular

**Goal:** Separate `app/automlplus/` into distinct Image tools, Language/web tools, and shared utilities so each area can be extended independently.

**Status:** TODO

**Depends on:** TASK 1 (routers) completed first — router refactor makes this easier.

### Steps

1. **Create service layer under `app/automlplus/services/`:**
   - `image_service.py` — logic extracted from image endpoints in `main.py`:
     - `async def run_on_image(request, llm_config) -> dict`
     - `def run_on_image_stream(request, llm_config) -> AsyncGenerator`
     - `def image_to_website(request) -> dict` (currently a stub)

   - `web_accessibility_service.py` — logic extracted from accessibility endpoints:
     - `async def check_alt_text(request, env, llm_config) -> dict`
     - `async def analyze_accessibility(request, env, llm_config) -> dict`

2. **Create `app/automlplus/dependencies.py`** for FastAPI `Depends()` injection:
   - `get_jinja_env()` — returns the Jinja2 environment (currently initialised ad hoc)
   - `get_llm_config()` — returns `(model_id, backend)` from env vars

3. **Extract utilities:**
   - `app/automlplus/utils.py` already exists — move `json_safe()` helper there if not already
   - Any HTML/file loading helpers that appear in multiple endpoints

4. **Update `router.py`** (from TASK 1) to call service functions via `Depends()` rather than embedding logic directly.

5. `main.py` should contain only: lifespan, app init, `include_router`.

6. Verify all existing tests pass; add targeted unit tests for each new service function.

---

## TASK 5 (HIGH) — Remove Old SQL Bits from Tabular API and Tests

**Goal:** Delete the legacy SQLite session-tracking code that was used before AutoDW. Everything is now stored in AutoDW.

**Status:** ✅ DONE

### Steps

1. **Delete `app/tabular_automl/db.py`** (entire file — `AutoMLSession` ORM model, `SessionLocal`, `Base`).

2. **Remove from `app/tabular_automl/services.py`:**
   - Import: `from .db import AutoMLSession, SessionLocal`
   - Function: `store_session_in_db()`
   - Dataclass: `SessionData`
   - Function: `get_session()`
   - (~70 lines total)

3. **Scrub remaining references:**
   - Search `app/` and `tests/` for: `db`, `AutoMLSession`, `SessionLocal`, `session_id`, `sqlite`, `TABULAR_DATABASE_CONFIG`
   - Remove any found references

4. **Update tests:**
   - Root `test_services.py` (integration tests): remove any assertions on `session_id` in tabular responses
   - `tests/test_tabular/`: check for any DB fixture setup in conftest; remove if present

5. **Remove env var** `TABULAR_DATABASE_CONFIG` from `.env.template` and `.env`.

6. Run `uv run pytest tests/test_tabular/ -v` to confirm no regressions.

---

## TASK 6 (LOW) — Environmental Impact Tracking for Vision

**Goal:** Report estimated energy/carbon cost of trained models.

**Status:** TODO (LOW priority — do not start until HIGH tasks are done)

### Notes

- For vision: track compute time × hardware TDP, map to CO2 estimate
- For AutoML+ (LLM calls): estimate token count × published gpt-4o-mini energy-per-token figures
- Likely best surfaced as an optional field in the response payload

---

## Implementation Order

| Order | Task                         | Reason                                                                 |
| ----- | ---------------------------- | ---------------------------------------------------------------------- |
| 1     | TASK 5 — Remove SQL          | Self-contained deletion, no dependencies, unblocks cleaner services.py |
| 2     | TASK 3 — Remove Ollama       | Self-contained, low risk, cleans up core before other work             |
| 3     | TASK 1 — Routers             | Required foundation for TASK 4; test-safe refactor                     |
| 4     | TASK 2 — Per-service logging | Best done after lifespan hooks are in place from TASK 1                |
| 5     | TASK 4 — Modularise AutoML+  | Builds on routers from TASK 1                                          |
| 6     | TASK 6 — Env impact          | Last; low priority                                                     |
