
# Running and Testing ALFIE Services

This guide shows how to start each ALFIE service using `uvicorn` directly from the shell, and how to test them with `curl`.

The Python orchestration script is not required — instead, run the commands below in separate shells.

- This page lists the services that exist at the moment and the options available for each
- Note that this might change in the future
- IMPORTANT : The localhost url might change in the future and is not loaded automatically from the .env here, so if something doesnt work, probably look at that first?
---

## Prerequisites

- You followed the setup

---


## Easy way vs manual way
- The easy way to test all of them is to follow the setup and then look at the `test_services.py` script.
    - You can just do `uv run test_services.py im2web` for example for it to test that service on sample data
    - Ports and stuff are directly taken from the `.env` file there so thats the easiest, but if you want to run anything manually, either follow whats below, or refer to the docker instructions

## Killing a Service on a Port

If a port is stuck, kill any process using it:

```bash
lsof -ti tcp:8000 | xargs kill -9
```

Replace `8000` with the relevant service port.

---

## Services Overview (This might change based on the .env file)

| Service     | Port | Uvicorn Target                | Description                             |
| ----------- | ---- | ----------------------------- | --------------------------------------- |
| webfromfile | 8000 | `app.automlplus.main:app`     | Website accessibility (HTML file input) |
| webfromurl  | 8000 | `app.automlplus.main:app`     | Website accessibility (URL input)       |
| im2web      | 8000 | `app.automlplus.main:app`     | Image-to-Website tool                   |
| tabular     | 8001 | `app.tabular_automl.main:app` | AutoML for tabular datasets             |
| vision      | 8002 | `app.vision_automl.main:app`  | AutoML for vision datasets              |

---

## Running Services

Run each service in its own shell:

### Webfromfile / Webfromurl / Im2web (port 8000)

```bash
uv run uvicorn app.automlplus.main:app --reload --host 0.0.0.0 --port 8000
```

### Tabular (port 8001)

```bash
uv run uvicorn app.tabular_automl.main:app --reload --host 0.0.0.0 --port 8001
```

### Vision (port 8002)

```bash
uv run uvicorn app.vision_automl.main:app --reload --host 0.0.0.0 --port 8002
```

---

## Testing Services with Curl

Open another shell and run the tests.
### AutoML Plus
Bits and bobs that are not really "AutoML" but use AI models for a specific use case
#### Test: Website Accessibility (HTML file input)
- This tests the accessibility of a file given an HTML
- Only options here are to enter the html file

```bash
curl -sN -X POST http://localhost:8000/automlplus/web_access/analyze/ \
  -H "Content-Type: multipart/form-data" \
  -F "file=@./sample_data/test.html"
```

---

#### Test: Website Accessibility (URL input)
- This tests the accessibility of a file given a URL (it downloads the html/css)
- Only options here are to enter the url

```bash
curl -s -X POST http://localhost:8000/automlplus/web_access/analyze/ \
  -H "Content-Type: multipart/form-data" \
  -F "url=https://alfie-project.eu"
  # Optionally add: -F "extra_file_input=@./sample_data/wcag_guidelines.txt"
```

---

#### Test: Image-to-Website Tool
- If you upload an image of a website and ask the engine to create a website of it, it will do so
- Options are the prompt and the image file

```bash
curl -sN -X POST http://localhost:8000/automlplus/image_tools/run_on_image_stream/ \
  -H "Content-Type: multipart/form-data" \
  -F "prompt=Recreate this image into a website with HTML/CSS/JS and explain how to run it." \
  -F "image_file=@./sample_data/websample.png"
```

---

### Test: AutoML Tabular
- Tabular AutoML
- The first call for each is to interface with AutoDW (When this is possible)
- Required options are the target_column, and type of task (classification or regression) and the time budget in seconds
- Input can be any file that can be read by pandas
- Train/test/val split will be automatically done

#### Step 1: Start a session

```bash
curl -s -X POST http://localhost:8001/automl_tabular/get_user_input/ \
  -H "Content-Type: multipart/form-data" \
  -F "train_csv=@./sample_data/knot_theory/train.csv" \
  -F "target_column_name=signature" \
  -F "task_type=classification" \
  -F "time_budget=30"
```

This returns a JSON with a `session_id`.

#### Step 2: Train and find best model

```bash
curl -s -X POST http://localhost:8001/automl_tabular/find_best_model/ \
  -H "Content-Type: application/json" \
  -d '{"session_id": "REPLACE_WITH_SESSION_ID"}'
```

---

### Test: AutoML Vision
- AutoML for vision
- Train/test/val split will be automatically done
- The first call for each is to interface with AutoDW (When this is possible)
- The required options are the csv file with labels, a zip of images, the filename column, label column name, task type, time budget and model size
    - image_zip
        - Images in a VERY specific format (for now, probably will be replaced with croissant or similar format later
        - main folder
            - category1
                - image1.png (images of type jpg, png, jpeg)
                - image2.png 
        
    - csv_file should contain two columns - one with the file name (no path, just the file name), and 
        |filename|label|
        |---|---|
        |image1.png|cat|
        |image2.png|dog|
    - time budget in seconds
    - model size is either small,medium or large (based on number of parameters)
        - MODEL_SMALL_MAX_PARAM_SIZE=50000000
        - MODEL_MEDIUM_MAX_PARAM_SIZE=200000000

#### Step 1: Start a session

```bash
curl -s -X POST http://localhost:8002/automl_vision/get_user_input/ \
  -H "Content-Type: multipart/form-data" \
  -F "csv_file=@./sample_data/Garbage_Dataset_Classification/metadata.csv" \
  -F "images_zip=@./sample_data/Garbage_Dataset_Classification/images.zip" \
  -F "filename_column=filename" \
  -F "label_column=label" \
  -F "task_type=classification" \
  -F "time_budget=10" \
  -F "model_size=medium"
```

This returns a JSON with a `session_id`.

#### Step 2: Train and find best model

```bash
curl -s -X POST http://localhost:8002/automl_vision/find_best_model/ \
  -H "Content-Type: application/json" \
  -d '{"session_id": "REPLACE_WITH_SESSION_ID"}'
```

