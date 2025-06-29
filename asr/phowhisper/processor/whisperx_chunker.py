import whisperx
import logging
import numpy as np
from typing import List, Dict, Union
import torch

logger = logging.getLogger(__name__)


class WhisperXChunker:
    def __init__(self, model_path: str, device: str, language: str):
        self.model_path = model_path
        self.device = device
        self.language = language
        self._model = None
        logger.debug("[WhisperXChunker] Instance created but model is not loaded yet.")

    @property
    def model(self):
        if self._model is None:
            logger.info(f"[WhisperXChunker] Lazily loading WhisperX model from {self.model_path} on a worker process...")
            self._model = whisperx.load_model(
                self.model_path,
                device=self.device,
                language=self.language,
                compute_type="float16"
            )
        return self._model

    def chunk(self, audio_array: np.ndarray, sampling_rate: int) -> List[Dict[str, Union[np.ndarray, int]]]:
        result = self.model.transcribe(
            audio_array,
            batch_size=16,
            chunk_size=30,
        )
        segments = result.get("segments", [])
        chunks = [
            {
                "array": audio_array[int(segment["start"] * sampling_rate):int(segment["end"] * sampling_rate)],
                "sampling_rate": sampling_rate
            }
            for segment in segments
        ]
        logger.debug(f"[WhisperXChunker] Chunked audio into {len(chunks)} segments")

        return chunks
