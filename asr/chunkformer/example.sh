#!/bin/bash

MODEL_ID="chunkformer-large-vie"
AUDIO_FILE="data/common_voice_vi_23397238.wav"
TOTAL_BATCH_DURATION=14400
CHUNK_SIZE=64
LEFT_CONTEXT_SIZE=128
RIGHT_CONTEXT_SIZE=128

python decode.py \
--model_checkpoint "$MODEL_ID" \
--long_form_audio "$AUDIO_FILE" \
--total_batch_duration "$TOTAL_BATCH_DURATION" \
--chunk_size "$CHUNK_SIZE" \
--left_context_size "$LEFT_CONTEXT_SIZE" \
--right_context_size "$RIGHT_CONTEXT_SIZE"
