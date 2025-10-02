#!/bin/bash
# start ollama server and record process id

echo "Starting Ollama server..."
ollama serve &
SERVE_PID=$!


echo "Waiting for Ollama server to be active..."
while [ "$(ollama list | grep 'NAME')" == "" ]; do
  sleep 1
done

ollama pull gemma3:4b
ollama pull qwen2.5vl

wait $SERVE_PID
