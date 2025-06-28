#!/bin/bash

MODEL_NAME=${1:?Please provide model name. E.g., minhtien2405/phowhisper-large-central-vi}
CONFIG_PATH="phowhisper/configs/config.yaml"

# Extract region from model name: e.g., phowhisper-large-central-vi -> Central
REGION=$(echo "$MODEL_NAME" | grep -oP 'phowhisper-large-\K(central|south|north|all)(?=-vi)' | awk '{print toupper(substr($0,1,1)) substr($0,2)}')

echo "Running evaluation on model: $MODEL_NAME (Region: $REGION)"
python cli.py evaluate --config $CONFIG_PATH --model-path $MODEL_NAME --region $REGION
