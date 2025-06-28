#!/bin/bash

AUDIO_PATH=${1:?Please provide path to audio file (e.g., ./sample.wav)}
MODEL_PATH=${2:-"./models/phowhisper-large-all-vi"}
CONFIG_PATH="phowhisper/configs/config.yaml"

echo "Running inference on audio: $AUDIO_PATH"
python cli.py infer --config $CONFIG_PATH --model-path $MODEL_PATH --audio-path $AUDIO_PATH
