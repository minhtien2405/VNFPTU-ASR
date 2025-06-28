#!/bin/bash

REGION=${1:-All}
CONFIG_PATH="phowhisper/configs/config.yaml"

echo "Starting training for region: $REGION"
python cli.py train --config $CONFIG_PATH --region $REGION