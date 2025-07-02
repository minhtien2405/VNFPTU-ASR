#!/bin/bash

REGION=${1:-All}
CONFIG_PATH="asr/phowhisper/configs/config.yaml"

echo "Starting training for region: $REGION"
CUDA_LAUNCH_BLOCKING=1 python3 asr/phowhisper/cli.py train --config $CONFIG_PATH --region $REGION
