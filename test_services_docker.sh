#!/bin/bash
inpargs="${1:-all}"
set -euo pipefail

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

if [[ $inpargs == "web" || $inpargs == "all" ]]; then
wait_for "website" "http://localhost:8000/web_access" 90
echo "=== Testing Website Accessibility ==="
curl -X POST http://localhost:8000/web_access/accessibility/ \
  -H "Content-Type: multipart/form-data" \
  -F "file=@./sample_data/test.html"
echo -e "\n"
fi
if [[ $inpargs == "tabular" || $inpargs == "all" ]]; then
wait_for "tabular" "http://localhost:8001/automl_tabular" 90
echo "=== Testing AutoML Tabular - get_user_input ==="
SESSION_RESPONSE=$(curl -s -X POST http://localhost:8001/automl_tabular/get_user_input/ \
  -H "Content-Type: multipart/form-data" \
  -F "train_csv=@./sample_data/knot_theory/train.csv" \
  -F "target_column_name=signature" \
  -F "task_type=classification" \
  -F "time_budget=30")
echo "$SESSION_RESPONSE"
SESSION_ID=$(echo "$SESSION_RESPONSE" | jq -r '.session_id')

if [[ "$SESSION_ID" != "null" && -n "$SESSION_ID" ]]; then
  echo "=== Testing AutoML Tabular - find_best_model ==="
  curl -X POST http://localhost:8001/automl_tabular/find_best_model/ \
    -H "Content-Type: application/json" \
    -d "{\"session_id\": \"$SESSION_ID\"}"
else
  echo "Failed to get valid session_id from tabular get_user_input"
  exit 1
fi

fi
echo -e "\n"

if [[ $inpargs == "vision" || $inpargs == "all" ]]; then
wait_for "vision" "http://localhost:8002/automl_vision" 120
echo "=== Testing AutoML Vision - get_user_input ==="
VISION_SESSION_RESPONSE=$(curl -s -X POST http://localhost:8002/automl_vision/get_user_input/ \
  -H "Content-Type: multipart/form-data" \
  -F "csv_file=@./sample_data/Garbage_Dataset_Classification/metadata.csv" \
  -F "images_zip=@./sample_data/Garbage_Dataset_Classification/images.zip" \
  -F "filename_column=filename" \
  -F "label_column=label" \
  -F "task_type=classification" \
  -F "time_budget=10" \
  -F "model_size=medium")
echo "$VISION_SESSION_RESPONSE"
VISION_SESSION_ID=$(echo "$VISION_SESSION_RESPONSE" | jq -r '.session_id')

if [[ "$VISION_SESSION_ID" != "null" && -n "$VISION_SESSION_ID" ]]; then
  echo "=== Testing AutoML Vision - find_best_model ==="
  curl -X POST http://localhost:8002/automl_vision/find_best_model/ \
    -H "Content-Type: application/json" \
    -d "{\"session_id\": \"$VISION_SESSION_ID\"}"
else
  echo "Failed to get valid session_id from vision get_user_input"
  exit 1
fi
fi
echo -e "\n"
echo "=== All tests completed ==="
