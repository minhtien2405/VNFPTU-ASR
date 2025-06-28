#!/bin/bash

REGION=${1:-All}
CONFIG_PATH="asr/phowhisper/configs/config.yaml"

echo "Starting training for region: $REGION"
python asr/phowhisper/cli.py train --config $CONFIG_PATH --region $REGION
