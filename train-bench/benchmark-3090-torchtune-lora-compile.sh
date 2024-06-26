#!/bin/bash

# mamba env and GPU
ENV_NAME="torchtune"
DEVICES=1 # 3090

# Create a log file with the current datetime
SCRIPT_NAME=$(basename "$0" .sh)
LOG_FILE="${SCRIPT_NAME}-$(date +%Y%m%d-%H%M%S).log"

echo "Logging results to $LOG_FILE"
echo


# Check if the conda environment already exists
if mamba env list | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating mamba environment '$ENV_NAME'..."
    mamba create --name "$ENV_NAME" python=3.11 -y
fi

# Activate the mamba environment
echo "Activating mamba environment '$ENV_NAME'..."
source "/opt/mambaforge/etc/profile.d/conda.sh"
source "/opt/mambaforge/etc/profile.d/mamba.sh"
mamba activate "$ENV_NAME"

# Run your desired commands or scripts within the activated environment
echo "Running commands within the '$ENV_NAME' environment..."
echo

# Run the get-system-info.sh script and redirect output to the log file
echo "Running get-system-info.sh..."
../get-system-info.sh | tee -a "$LOG_FILE"
echo

# Run the train-unsloth.py script and redirect output to the log file
echo "Running torchtune..."
time CUDA_VISIBLE_DEVICES=$DEVICES tune run lora_finetune_single_device --config torchtune.8B_lora_single_device.compile.yaml
