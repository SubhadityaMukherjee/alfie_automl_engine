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

## TASK 2 (MEDIUM) — Make AutoML+ Tools More Modular

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

## TASK 3 (LOW) — Environmental Impact Tracking for Vision

**Goal:** Report estimated energy/carbon cost of trained models.

**Status:** TODO (LOW priority — do not start until HIGH tasks are done)

### Notes

- For vision: track compute time × hardware TDP, map to CO2 estimate
- For AutoML+ (LLM calls): estimate token count × published gpt-4o-mini energy-per-token figures
- Likely best surfaced as an optional field in the response payload

---
