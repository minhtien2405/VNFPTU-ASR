import os
import torch
import time
import logging
import whisperx
from transformers import WhisperProcessor
from typing import List, Dict

logger = logging.getLogger(__name__)

class Inference:
    def __init__(self, config, model_path):
        self.config = config
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading processor and model from: {model_path}")
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = whisperx.load_model(model_path, self.device, compute_type="float16")

    def infer(self, audio_path: str) -> List[Dict]:
        assert os.path.exists(audio_path), f"Audio file not found: {audio_path}"

        logger.info(f"Starting inference on: {audio_path}")
        start_time = time.time()

        result = self.model.transcribe(audio_path, batch_size=16, chunk_size=30)
        segments = result["segments"]

        end_time = time.time()
        total_latency = end_time - start_time
        logger.info(f"Inference completed in {total_latency:.2f} seconds")

        for segment in segments:
            segment["latency"] = total_latency / max(len(segments), 1)

        return segments
