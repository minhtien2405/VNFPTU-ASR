#!/bin/bash

REGION=${1:-All}
CONFIG_PATH="ASR/Phowhisper/configs/config.yaml"

echo "Starting training for region: $REGION"
python ASR/Phowhisper/cli.py train --config $CONFIG_PATH --region $REGION