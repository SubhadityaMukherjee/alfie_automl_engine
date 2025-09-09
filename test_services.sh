#!/bin/bash
inpargs="${1:-all}"
set -euo pipefail

PID_FILE="processes.pid"
declare -a PIDS=()

# Kill any existing servers on expected ports to avoid hitting old instances
for PORT in 8001 8002 8004; do
  if lsof -ti tcp:$PORT >/dev/null 2>&1; then
    echo "Killing process on port $PORT"
    kill -9 $(lsof -ti tcp:$PORT) || true
  fi
done

# Start FastAPI apps in background and save their PIDs
if [[ $inpargs == "web" || $inpargs == "all" ]]; then
  uv run uvicorn app.website_accessibility.main:app --reload --host 0.0.0.0 --port 8000 &
  PIDS+=($!)
  sleep 2  # Allow server to start
fi

if [[ $inpargs == "tabular" || $inpargs == "all" ]]; then
  uv run uvicorn app.tabular_automl.main:app --reload --host 0.0.0.0 --port 8001 &
  PIDS+=($!)
  sleep 2  # Allow server to start
fi

if [[ $inpargs == "vision" || $inpargs == "all" ]]; then
  uv run uvicorn app.vision_automl.main:app --reload --host 0.0.0.0 --port 8002 &
  PIDS+=($!)
  sleep 2  # Allow server to start
fi

if [[ $inpargs == "general" || $inpargs == "all" ]]; then
  uv run uvicorn app.general_inference_tools.main:app --reload --host 0.0.0.0 --port 8004 &
  PIDS+=($!)
  sleep 2  # Allow server to start
fi
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

if [[ $inpargs == "web" || $inpargs == "all" ]]; then
echo "=== Testing Website Accessibility ==="
curl -X POST http://localhost:8000/web_access/accessibility/ \
  -H "Content-Type: multipart/form-data" \
  -F "file=@./sample_data/test.html"
echo -e "\n"
fi
if [[ $inpargs == "tabular" || $inpargs == "all" ]]; then
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
fi

fi
echo -e "\n"

if [[ $inpargs == "vision" || $inpargs == "all" ]]; then
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
fi
fi
echo -e "\n"


if [[ $inpargs == "general" || $inpargs == "all" ]]; then
echo "=== Testing General Inference - instruction_to_webpage ==="
curl -s -m 5 -X POST http://localhost:8004/general_inference/instruction_to_webpage \
  -H "Content-Type: application/json" \
  -d '{
        "requirements": "Create a simple Tailwind page with a centered blue button",
        "gen_kwargs": {"max_new_tokens": 64}
      }' || true
echo -e "\n"

echo "=== Testing General Inference - screenshot_to_webpage ==="
curl -s -m 5 -X POST http://localhost:8004/general_inference/screenshot_to_webpage \
  -H "Content-Type: application/json" \
  -d '{
        "requirements": "Generate Tailwind HTML for a login form",
        "gen_kwargs": {"max_new_tokens": 64}
      }' || true
echo -e "\n"

echo "=== Testing General Inference - video_understanding (uses placeholder input) ==="
curl -s -m 5 -X POST http://localhost:8004/general_inference/video_understanding \
  -H "Content-Type: application/json" \
  -d '{
        "query": "Describe the video content",
        "inputs": ["PLACEHOLDER_PATH_TO_VIDEO.mp4"],
        "gen_kwargs": {"num_beams": 3, "do_sample": false}
      }' || true
echo -e "\n"

echo "=== Testing General Inference - document_qa (uses placeholder input) ==="
curl -s -m 5 -X POST http://localhost:8004/general_inference/document_qa \
  -H "Content-Type: application/json" \
  -d '{
        "document": "PLACEHOLDER_PATH_TO_DOCUMENT_OR_IMAGE",
        "question": "What is the title?",
        "gen_kwargs": {"max_answer_len": 32}
      }' || true
echo -e "\n"
fi
echo "=== All tests completed ==="
