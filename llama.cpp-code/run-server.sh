#!/usr/bin/bash

LLAMA_SERVER=~/llama.cpp/build/bin/llama-server
LLAMA_SERVER_AMD=~/llama.cpp-hjc4869/build/bin/llama-server

MODEL=/models/gguf/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf 
MODEL_DRAFT=/models/gguf/Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf 

###

# Print usage if no arguments provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <version>"
    echo "Available versions:"
    echo "  w7900        - W7900"
    echo "  w7900-sd     - W7900 with speculative decoding"
    echo "  rtx3090-nofa - RTX3090 without flash attention"
    echo "  rtx3090      - RTX3090"
    echo "  rtx3090-sd   - RTX3090 with speculative decoding"
    exit 1
fi

# Process based on version argument
case "$1" in
    "w7900")
        LLAMA_SERVER=$LLAMA_SERVER_AMD
        ${LLAMA_SERVER} \
            -m ${MODEL} -ngl 99 \
        ;;
    "w7900-sd")
        LLAMA_SERVER=$LLAMA_SERVER_AMD
        ${LLAMA_SERVER} \
            -m ${MODEL} -ngl 99 \
            -md ${MODEL_DRAFT} -ngld 99 \
            --draft-max 24 --draft-min 1 --draft-p-min 0.6
        ;;
    "rtx3090-nofa")
        ${LLAMA_SERVER} \
            -m ${MODEL} -ngl 99 \
        ;;
    "rtx3090")
        ${LLAMA_SERVER} \
            -m ${MODEL} -ngl 99 \
            -fa
        ;;
    "rtx3090-sd")
        ${LLAMA_SERVER} \
            -m ${MODEL} -ngl 99 \
            -fa \
            -md ${MODEL_DRAFT} -ngld 99 \
            --draft-max 24 --draft-min 1 --draft-p-min 0.6
        ;;
    *)
        echo "Error: Invalid version specified"
        echo "Available versions: w7900, w7900-sd, rtx3090-nofa, rtx3090, rtx3090-sd"
        exit 1
        ;;
esac
