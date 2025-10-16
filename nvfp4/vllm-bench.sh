#!/bin/bash

# Configuration
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
DATASET_PATH=${DATASET_PATH:-Aeala/ShareGPT_Vicuna_unfiltered}
NUM_PROMPTS=${NUM_PROMPTS:-128}
RESULT_DIR=${1:-.}
# Strip trailing slash if present
RESULT_DIR=${RESULT_DIR%/}

# Get model name from the server
MODEL=$(curl -s "$BASE_URL/v1/models" | jq -r '.data[0].id')

if [ -z "$MODEL" ] || [ "$MODEL" = "null" ]; then
  echo "Error: Could not detect model from server at $BASE_URL"
  exit 1
fi

echo "Base URL: $BASE_URL"
echo "Detected model: $MODEL"
echo "Dataset: $DATASET_PATH"
echo "Num prompts: $NUM_PROMPTS"
echo "Result directory: $RESULT_DIR"
echo ""

# Create result directory if it doesn't exist
mkdir -p "$RESULT_DIR"

# Run benchmarks across concurrency levels
for c in 1 4 8 16 32; do
  echo "Testing concurrency: $c"
  vllm bench serve \
    --backend vllm \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --hf-name "$DATASET_PATH" \
    --num-prompts "$NUM_PROMPTS" \
    --save-result \
    --result-dir "$RESULT_DIR" \
    --request-rate inf \
    --max-concurrency $c
done
