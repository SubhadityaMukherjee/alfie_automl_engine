# Implementation Plans

Generated from `tasklist.md`. Mark tasks complete here and in tasklist.md when done.

---

## TASK (HIGH) — Refactor Vision Training for Multiple Task Types

**Goal:** Extend the vision service to support multiple HuggingFace task types beyond image classification, each with its own Optuna objective, model class, datamodule, and config.
this referes to files in - app/vision_automl, tests/test_vision and docs/api/vision_automl.md

**Status:** DONE

**Task types to support:**

| Task Type Slug         | HF Auto Class                        | Data Domain       |
| ---------------------- | ------------------------------------ | ----------------- |
| `image_classification` | `AutoModelForImageClassification`    | Images (existing) |
| `image_segmentation`   | `AutoModelForImageSegmentation`      | Images            |
| `object_detection`     | `AutoModelForObjectDetection`        | Images            |
| `video_classification` | `AutoModelForVideoClassification`    | Video frames      |
| `keypoint_detection`   | `AutoModelForKeypointDetection`      | Images            |
| `audio_classification` | `AutoModelForAudioClassification`    | Audio             |
| `text_classification`  | `AutoModelForSequenceClassification` | Text              |
| `question_answering`   | `AutoModelForQuestionAnswering`      | Text              |
| `causal_lm`            | `AutoModelForCausalLM`               | Text              |
| `seq2seq_lm`           | `AutoModelForSeq2SeqLM`              | Text              |
| `masked_lm`            | `AutoModelForMaskedLM`               | Text              |

---

### Step 1 — Create `ml_engine/configs/` with per-task JSON config files

Each JSON file lives at `ml_engine/configs/<task_type>.json` and holds:

- `small_models`: list of HF model IDs (≤50M params)
- `medium_models`: list of HF model IDs (≤200M params)
- `large_models`: list of HF model IDs (>200M params)
- `lr_low` / `lr_high`: float bounds for Optuna log-uniform LR suggestion
- `batch_sizes`: list of ints for Optuna categorical batch_size
- `weight_decay_low` / `weight_decay_high`: bounds for weight decay
- `max_epochs`: int (default 20)
- `early_stopping_patience`: int (default 3)

**Checklist:**

- [ ] Create `app/vision_automl/ml_engine/configs/` directory
- [ ] Write `image_classification.json` (migrate current hardcoded values from `optuna_objective`)
- [ ] Write `image_segmentation.json`
- [ ] Write `object_detection.json`
- [ ] Write `video_classification.json`
- [ ] Write `keypoint_detection.json`
- [ ] Write `audio_classification.json`
- [ ] Write `text_classification.json`
- [ ] Write `question_answering.json`
- [ ] Write `causal_lm.json`
- [ ] Write `seq2seq_lm.json`
- [ ] Write `masked_lm.json`
- [ ] Write `app/vision_automl/ml_engine/configs/__init__.py` with a `load_task_config(task_type: str) -> dict` helper that loads and returns the JSON (raises `ValueError` for unknown task types)

---

### Step 2 — Update `ml_engine/model.py` — Add model wrapper classes per task type

**Pattern:** Each class mirrors `ClassificationModel` — thin `nn.Module` wrapping the corresponding HF `Auto*` class.

**Checklist:**

- [ ] Keep `ClassificationModel` (wraps `AutoModelForImageClassification`) — rename to `ImageClassificationModel` and keep `ClassificationModel` as an alias for backward compat (or just rename outright)
- [ ] Add `ImageSegmentationModel` (wraps `AutoModelForImageSegmentation`)
- [ ] Add `ObjectDetectionModel` (wraps `AutoModelForObjectDetection`)
- [ ] Add `VideoClassificationModel` (wraps `AutoModelForVideoClassification`)
- [ ] Add `KeypointDetectionModel` (wraps `AutoModelForKeypointDetection`)
- [ ] Add `AudioClassificationModel` (wraps `AutoModelForAudioClassification`)
- [ ] Add `SequenceClassificationModel` (wraps `AutoModelForSequenceClassification`)
- [ ] Add `QuestionAnsweringModel` (wraps `AutoModelForQuestionAnswering`)
- [ ] Add `CausalLMModel` (wraps `AutoModelForCausalLM`)
- [ ] Add `Seq2SeqLMModel` (wraps `AutoModelForSeq2SeqLM`)
- [ ] Add `MaskedLMModel` (wraps `AutoModelForMaskedLM`)
- [ ] Add `MODEL_REGISTRY: dict[str, type]` mapping task_type slug → model class (used by trainer dispatch)
- [ ] Forward pass for each: call underlying HF model, return output (logits for classification, full output object for detection/segmentation)

---

### Step 3 — Update `ml_engine/datamodule.py` — Add datamodule classes per task type

**Pattern:** Each datamodule handles loading/collating data for its modality. Existing `ClassificationData` covers image classification.

**Checklist:**

- [ ] Rename `ClassificationData` → `ImageClassificationDataModule` (keep `ClassificationData` alias)
- [ ] Add `ImageSegmentationDataModule` — same CSV+image layout, collate uses `AutoImageProcessor` (segmentation mode)
- [ ] Add `ObjectDetectionDataModule` — CSV with bounding-box annotations; collate pads boxes to same length per batch
- [ ] Add `VideoClassificationDataModule` — CSV pointing to video files or frame directories; decodes frames into tensor clips
- [ ] Add `KeypointDetectionDataModule` — CSV with keypoint coordinates; collate uses `AutoImageProcessor`
- [ ] Add `AudioClassificationDataModule` — CSV pointing to audio files; loads waveforms with `torchaudio`, collate uses `AutoFeatureExtractor`
- [ ] Add `SequenceClassificationDataModule` — CSV with `text` + `label` columns; tokenizes with `AutoTokenizer`
- [ ] Add `QuestionAnsweringDataModule` — CSV with `question`, `context`, `answer_start`, `answer_text`; tokenizes with `AutoTokenizer`
- [ ] Add `CausalLMDataModule` — CSV with `text` column; tokenizes and shifts labels by 1
- [ ] Add `Seq2SeqLMDataModule` — CSV with `input_text` + `target_text`; tokenizes both
- [ ] Add `MaskedLMDataModule` — CSV with `text` column; applies random masking via `DataCollatorForLanguageModeling`
- [ ] Add `DATAMODULE_REGISTRY: dict[str, type]` mapping task_type slug → datamodule class
- [ ] Each new datamodule must expose `train_dataloader()`, `val_dataloader()`, `test_dataloader()`, and `id2label` / `label2id` where applicable

---

### Step 4 — Update `ml_engine/trainer.py` — Split into per-task Optuna objectives

**Checklist:**

- [ ] Rename current `optuna_objective` → `optuna_objective_image_classification`
- [ ] Remove hardcoded model lists and hyperparams; load them from `load_task_config("image_classification")` instead
- [ ] Add `optuna_objective_image_segmentation(trial, data_root, csv_path, ..., config)`
- [ ] Add `optuna_objective_object_detection(trial, ...)`
- [ ] Add `optuna_objective_video_classification(trial, ...)`
- [ ] Add `optuna_objective_keypoint_detection(trial, ...)`
- [ ] Add `optuna_objective_audio_classification(trial, ...)`
- [ ] Add `optuna_objective_text_classification(trial, ...)`
- [ ] Add `optuna_objective_question_answering(trial, ...)`
- [ ] Add `optuna_objective_causal_lm(trial, ...)`
- [ ] Add `optuna_objective_seq2seq_lm(trial, ...)`
- [ ] Add `optuna_objective_masked_lm(trial, ...)`
- [ ] Add `OBJECTIVE_REGISTRY: dict[str, Callable]` mapping task_type slug → objective function
- [ ] Update `run_optuna_search(task_type: str, ...)` to:
  - Load config via `load_task_config(task_type)`
  - Look up objective via `OBJECTIVE_REGISTRY[task_type]`
  - Pass config into the objective via `functools.partial`
  - Raise `ValueError` for unknown task types
- [ ] `FabricTrainer` stays generic — each objective constructs it with the appropriate model + datamodule

---

### Step 5 — Update `models.py` — Pydantic request/response schemas

**Checklist:**

- [ ] Add `SUPPORTED_VISION_TASK_TYPES` literal union / `Enum` covering all 11 task slugs
- [ ] Rename existing `ImageClassificationTask` → keep as-is but add `task_type: Literal["image_classification"]`
- [ ] Add `ImageSegmentationTask(ImageTask)` with `task_type: Literal["image_segmentation"]`
- [ ] Add `ObjectDetectionTask(ImageTask)` with `task_type: Literal["object_detection"]`
- [ ] Add `VideoClassificationTask(ImageTask)` with `task_type: Literal["video_classification"]` — note `label_format` only supports csv
- [ ] Add `KeypointDetectionTask(ImageTask)` with `task_type: Literal["keypoint_detection"]`
- [ ] Add `AudioClassificationTask` — separate base (not ImageTask) — has `audio_dir`, `labels_file`, `task_type: Literal["audio_classification"]`
- [ ] Add `SequenceClassificationTask` — text base, has `data_file` (CSV), `task_type: Literal["text_classification"]`
- [ ] Add `QuestionAnsweringTask` — text base, has `data_file`, `task_type: Literal["question_answering"]`
- [ ] Add `CausalLMTask` — text base, `task_type: Literal["causal_lm"]`
- [ ] Add `Seq2SeqLMTask` — text base, `task_type: Literal["seq2seq_lm"]`
- [ ] Add `MaskedLMTask` — text base, `task_type: Literal["masked_lm"]`
- [ ] Add `VisionTask = Annotated[Union[ImageClassificationTask, ImageSegmentationTask, ...], Field(discriminator="task_type")]` discriminated union

---

### Step 6 — Update `services.py`

**Checklist:**

- [ ] `validate_vision_inputs(task_type, ...)`:
  - Image tasks: existing CSV + image presence checks, but gate on `task_type in IMAGE_TASK_TYPES`
  - Audio tasks: validate audio dir + CSV exists
  - Text tasks: validate CSV exists and required columns are present (per task type — e.g. `text`+`label` for classification, `question`+`context`+`answer_text` for QA)
  - Detection/segmentation tasks: validate annotation columns exist in CSV
- [ ] `build_upload_payload(...)`: add `task_type` field to the returned dict (currently missing)
- [ ] `train_automl(task_type, ...)`: pass `task_type` through to `run_optuna_search`

---

### Step 7 — Update `router.py`

**Checklist:**

- [ ] Change `task_type` Form parameter type from plain `str` to `Literal[<all 11 slugs>]` (or validated via Pydantic/Enum)
- [ ] Pass `task_type` into `validate_vision_inputs()` call
- [ ] Pass `task_type` into `train_automl()` call
- [ ] Pass `task_type` into `build_upload_payload()` call
- [ ] Update docstring/OpenAPI description for the endpoint

---

### Step 8 — Update tests in `tests/test_vision/`

**Checklist:**

- [ ] `test_models.py` — add validation tests for each new Pydantic schema
- [ ] `test_services.py`:
  - [ ] `validate_vision_inputs` — add cases for audio, text, detection tasks (valid + missing columns + missing files)
  - [ ] `build_upload_payload` — add assertion that `task_type` appears in payload
  - [ ] `train_automl` — confirm `task_type` is forwarded to `run_optuna_search`
- [ ] `test_ml_engine/test_dataset.py` — add basic tests for any new dataset classes (can be lightweight — no full loading)
- [ ] `test_ml_engine/test_datamodule.py` — add init + split tests for each new datamodule (mock HF processor/tokenizer/feature extractor)
- [ ] `test_ml_engine/test_trainer.py` — add tests confirming `OBJECTIVE_REGISTRY` has all 11 keys and `run_optuna_search` dispatches correctly (mock the objective itself)
- [ ] `test_ml_engine/` — add `test_configs.py`: test that `load_task_config` returns valid dicts for all 11 task types and raises `ValueError` for an unknown slug

---

### Step 9 — Update `docs/api/vision_automl.md`

**Checklist:**

- [ ] Add `:::` source references for:
  - [ ] `app.vision_automl.ml_engine.configs` (the loader helper)
  - [ ] New model classes in `model.py`
  - [ ] New datamodule classes in `datamodule.py`
  - [ ] New objective functions in `trainer.py`
  - [ ] New Pydantic schemas in `models.py`

---

### Step 10 - Update test_services.py

- [ ] Add tests for these new types (leave the datasets blank for now, I will add my own later. You can suggest some small ones for testing though)

### Implementation Order

1. Step 1 (configs) — everything else reads from here
2. Step 5 (models.py) — defines the task_type vocabulary used everywhere
3. Step 2 (model.py) — model wrappers; needed by trainer objectives
4. Step 3 (datamodule.py) — datamodules; needed by trainer objectives
5. Step 4 (trainer.py) — objectives + registry; needed by services
6. Step 6 (services.py) — wires task_type through the service layer
7. Step 7 (router.py) — exposes task_type at the API boundary
8. Step 8 (tests) — validate everything
9. Step 9 (docs) — update references
10. Step 10 (test_services) - update everything

---

## TASK (LOW) — Environmental Impact Tracking for Vision

**Goal:** Report estimated energy/carbon cost of trained models.

**Status:** TODO (LOW priority — do not start until HIGH tasks are done)

### Notes

- For vision: track compute time × hardware TDP, map to CO2 estimate
- For AutoML+ (LLM calls): estimate token count × published gpt-4o-mini energy-per-token figures
- Likely best surfaced as an optional field in the response payload

---
