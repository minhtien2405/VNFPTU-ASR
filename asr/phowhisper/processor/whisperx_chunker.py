import whisperx
import logging
import numpy as np
from typing import List, Dict, Union

logger = logging.getLogger(__name__)

class WhisperXChunker:
    def __init__(self, model_path: str, device: str, language: str):
        self.device = device
        self.language = language

        self.model = whisperx.load_model(
            model_path,
            device=device,
            language=language,
            compute_type="float16"
        )
        logger.info(f"[WhisperXChunker] Loaded WhisperX model from {model_path}")

    def chunk(self, audio_array: np.ndarray, sampling_rate: int) -> List[Dict[str, Union[np.ndarray, int]]]:
        result = self.model.transcribe(
            {"array": audio_array, "sampling_rate": sampling_rate},
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
