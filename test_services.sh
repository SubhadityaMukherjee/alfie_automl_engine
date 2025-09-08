#!/bin/bash

PID_FILE="processes.pid"
declare -a PIDS=()

# Start FastAPI apps in background and save their PIDs
# uv run uvicorn app.website_accessibility.main:app --reload --host 0.0.0.0 --port 8000 &
# PIDS+=($!)
# sleep 2  # Allow server to start

uv run uvicorn app.tabular_automl.main:app --reload --host 0.0.0.0 --port 8001 &
PIDS+=($!)
sleep 2  # Allow server to start

uv run uvicorn app.vision_automl.main:app --reload --host 0.0.0.0 --port 8002 &
PIDS+=($!)
sleep 2  # Allow server to start

# Save all PIDs to file
printf "%s\n" "${PIDS[@]}" > "$PID_FILE"

# Function to clean up background processes on exit
cleanup() {
  echo "Stopping servers..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null
  done
  rm -f "$PID_FILE"
}
trap cleanup EXIT

# echo "=== Testing Website Accessibility ==="
# curl -X POST http://localhost:8000/web_access/accessibility/ \
#   -H "Content-Type: multipart/form-data" \
#   -F "file=@./sample_data/test.html"
# echo -e "\n"

# echo "=== Testing AutoML Tabular - get_user_input ==="
# SESSION_RESPONSE=$(curl -s -X POST http://localhost:8001/automl_tabular/get_user_input/ \
#   -H "Content-Type: multipart/form-data" \
#   -F "train_csv=@./sample_data/knot_theory/train.csv" \
#   -F "target_column_name=signature" \
#   -F "task_type=classification" \
#   -F "time_budget=30")
# echo "$SESSION_RESPONSE"
# SESSION_ID=$(echo "$SESSION_RESPONSE" | jq -r '.session_id')

# if [[ "$SESSION_ID" != "null" && -n "$SESSION_ID" ]]; then
#   echo "=== Testing AutoML Tabular - find_best_model ==="
#   curl -X POST http://localhost:8001/automl_tabular/find_best_model/ \
#     -H "Content-Type: application/json" \
#     -d "{\"session_id\": \"$SESSION_ID\"}"
# else
#   echo "Failed to get valid session_id from tabular get_user_input"
# fi

echo -e "\n"
echo "=== Testing AutoML Vision - get_user_input ==="
VISION_SESSION_RESPONSE=$(curl -s -X POST http://localhost:8002/automl_vision/get_user_input/ \
  -H "Content-Type: multipart/form-data" \
  -F "csv_file=@./sample_data/Garbage_Dataset_Classification/metadata.csv" \
  -F "images_zip=@./sample_data/Garbage_Dataset_Classification/images.zip" \
  -F "filename_column=filename" \
  -F "label_column=label" \
  -F "task_type=classification" \
  -F "time_budget=10")
echo "$VISION_SESSION_RESPONSE"
VISION_SESSION_ID=$(echo "$VISION_SESSION_RESPONSE" | jq -r '.session_id')

if [[ "$VISION_SESSION_ID" != "null" && -n "$VISION_SESSION_ID" ]]; then
  echo "=== Testing AutoML Vision - find_best_model ==="
  curl -X POST http://localhost:8002/automl_vision/find_best_model/ \
    -H "Content-Type: application/json" \
    -d "{\"session_id\": \"$VISION_SESSION_ID\"}"
else
  echo "Failed to get valid session_id from vision get_user_input"
fi

echo -e "\n"
echo "=== All tests completed ==="
