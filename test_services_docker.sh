#!/bin/bash
inpargs="${1:-all}"
set -euo pipefail

ENGINE_PORT="${AUTOML_ENGINE_PORT:-8001}"
ENGINE_BASE="http://localhost:${ENGINE_PORT}"

wait_for() {
  local name="$1"; shift
  local url="$1"; shift
  local timeout="${1:-60}"
  local start_ts=$(date +%s)
  echo "Waiting for $name at $url ..."
  until curl -fsS "$url" >/dev/null 2>&1; do
    sleep 1
    now=$(date +%s)
    if (( now - start_ts > timeout )); then
      echo "Timed out waiting for $name at $url"
      return 1
    fi
  done
  echo "$name is up"
}

wait_for "automl engine" "${ENGINE_BASE}/health" 240

if [[ $inpargs == "web" || $inpargs == "all" ]]; then
echo "=== Testing Website Accessibility ==="
curl -X POST "${ENGINE_BASE}/automl/automl_plus/web_access/analyze/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@./sample_data/test.html"
echo -e "\n"
fi

if [[ $inpargs == "tabular" || $inpargs == "all" ]]; then
echo "=== Testing AutoML Tabular - best_model ==="
curl -s -X POST "${ENGINE_BASE}/automl/tabular/best_model/" \
  -H "Content-Type: multipart/form-data" \
  -F "user_id=1" \
  -F "dataset_id=4" \
  -F "target_column_name=labels" \
  -F "task_type=tabular_classification" \
  -F "time_budget=10"
echo -e "\n"
fi

if [[ $inpargs == "vision" || $inpargs == "all" ]]; then
echo "=== Testing AutoML Vision - best_model ==="
curl -s -X POST "${ENGINE_BASE}/automl/vision/best_model/" \
  -H "Content-Type: multipart/form-data" \
  -F "user_id=1" \
  -F "dataset_id=2" \
  -F "filename_column=filename" \
  -F "label_column=label" \
  -F "task_type=image_classification" \
  -F "time_budget=10" \
  -F "model_size=medium"
echo -e "\n"

echo "=== Testing AutoML Vision - multimodal_best_model ==="
curl -s -X POST "${ENGINE_BASE}/automl/vision/multimodal_best_model/" \
  -H "Content-Type: multipart/form-data" \
  -F "user_id=1" \
  -F "dataset_id=5" \
  -F "filename_column=image_file_path" \
  -F "label_column=label" \
  -F "time_budget=60" \
  -F "model_size=medium"
echo -e "\n"
fi

echo "=== All tests completed ==="
