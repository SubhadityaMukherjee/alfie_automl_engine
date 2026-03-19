# Implementation Plans

Generated from `tasklist.md`. Mark tasks complete here and in tasklist.md when done.

---

## TASK 1 (HIGH) — Make AutoMLplus Tools More Modular

**Goal:** Reorganise `app/automlplus/` so tool classes are grouped by technique type (`vlm`, `text`, `static`) rather than by domain (`imagetools`, `website_accessibility`). URLs in `router.py` must not change.

**Status:** DONE

**Depends on:** Router refactor already done — `router.py` and `main.py` are in place.

---

### Current layout (before)

```
app/automlplus/
  main.py                        # lifespan + include_router ✓
  router.py                      # 5 endpoints, inline jinja_env + json_safe
  utils.py                       # ImageConverter only
  imagetools.py                  # ImagePromptRunner (vlm)
  website_accessibility/
    modules.py                   # AltTextChecker (vlm), ReadabilityAnalyzer (static), split_chunks
    services.py                  # pipeline orchestration + extract_text_from_html_bytes
```

### Target layout (after)

```
app/automlplus/
  main.py                        # unchanged
  router.py                      # same 5 URLs, imports from tools/ and utils
  utils.py                       # ImageConverter + extract_text_from_html_bytes + json_safe
  tools/
    __init__.py
    vlm.py                       # ImagePromptRunner (from imagetools.py)
                                 # AltTextChecker (from website_accessibility/modules.py)
    static.py                    # ReadabilityAnalyzer + split_chunks
                                 # (from website_accessibility/modules.py)
    text.py                      # LLM-over-text: ChunkResult + _process_single_chunk
                                 # (from website_accessibility/services.py)
  website_accessibility/
    __init__.py
    pipeline.py                  # run_accessibility_pipeline, resolve_coroutines,
                                 # stream_accessibility_results (thin orchestration layer,
                                 # imports from tools/ instead of modules.py)
```

---

### Steps

1. **Expand `utils.py`** — move two helpers here (no behaviour change):
   - `json_safe()` from `router.py`
   - `extract_text_from_html_bytes()` from `website_accessibility/services.py`

2. **Create `app/automlplus/tools/__init__.py`** (empty).

3. **Create `app/automlplus/tools/vlm.py`** — move classes verbatim:
   - Add documentation to the top of the file about what this fiel does. Explain that vlm is a vision language model task that involves passing images + a prompt of some sort
   - `ImagePromptRunner` from `imagetools.py`
   - `AltTextChecker` from `website_accessibility/modules.py`

4. **Create `app/automlplus/tools/static.py`** — move verbatim:
   - Add documentation to the top of the file about what this fiel does
   - `ReadabilityAnalyzer` from `website_accessibility/modules.py`
   - `split_chunks` from `website_accessibility/modules.py`

5. **Create `app/automlplus/tools/text.py`** — move verbatim:
   - Add documentation to the top of the file about what this fiel does
   - `ChunkResult` dataclass from `website_accessibility/services.py`
   - `_process_single_chunk` from `website_accessibility/services.py`
   - This function uses `AltTextChecker` (import from `tools.vlm`) and `ChatHandler`.

6. **Create `app/automlplus/website_accessibility/pipeline.py`** — thin orchestration:
   - Add documentation to the top of the file about what this fiel does
   - `run_accessibility_pipeline` — imports `split_chunks` from `tools.static`, `_process_single_chunk` from `tools.text`
   - `resolve_coroutines` — no tool dependencies, copy as-is
   - `stream_accessibility_results` — calls `resolve_coroutines`

7. **Update `router.py`** — change imports only, no logic changes:
   - `from app.automlplus.utils import json_safe, extract_text_from_html_bytes`
   - `from app.automlplus.tools.vlm import ImagePromptRunner, AltTextChecker`
   - `from app.automlplus.tools.static import ReadabilityAnalyzer`
   - `from app.automlplus.website_accessibility.pipeline import run_accessibility_pipeline, resolve_coroutines`
   - Remove the inline `json_safe` definition from `router.py`

8. **Delete obsolete files** once imports updated and tests pass:
   - `app/automlplus/imagetools.py`
   - `app/automlplus/website_accessibility/modules.py`
   - `app/automlplus/website_accessibility/services.py` (replaced by `pipeline.py`)

9. **Verify** — run `uv run pytest -q`. No new tests needed; this is a pure move with no behaviour change.

---

## TASK (LOW) — Environmental Impact Tracking for Vision

**Goal:** Report estimated energy/carbon cost of trained models.

**Status:** TODO (LOW priority — do not start until HIGH tasks are done)

### Notes

- For vision: track compute time × hardware TDP, map to CO2 estimate
- For AutoML+ (LLM calls): estimate token count × published gpt-4o-mini energy-per-token figures
- Likely best surfaced as an optional field in the response payload

---
