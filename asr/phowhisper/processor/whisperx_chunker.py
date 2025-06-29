import whisperx
import logging
import numpy as np
from typing import List, Dict, Union, Optional
import torch
from pathlib import Path

logger = logging.getLogger(__name__)

class WhisperXChunkerError(Exception):
    """Custom exception for WhisperXChunker errors"""
    pass

class WhisperXChunker:
    def __init__(self, model_path: Union[str, Path], device: str, language: str):
        self.model_path = str(Path(model_path).resolve())
        self.device = device
        self.language = language.lower()
        self._model = None
        self._validate_init_params()
        logger.debug(f"[WhisperXChunker] Initialized with model_path={self.model_path}, device={device}, language={language}")

    def _validate_init_params(self) -> None:
        if not Path(self.model_path).exists():
            raise WhisperXChunkerError(f"Model path does not exist: {self.model_path}")
        if self.device not in ["cpu", "cuda"]:
            raise WhisperXChunkerError(f"Invalid device: {self.device}")
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"

    @property
    def model(self):
        if self._model is None:
            try:
                logger.info(f"Loading WhisperX model from {self.model_path}")
                self._model = whisperx.load_model(
                    self.model_path,
                    device=self.device,
                    language=self.language,
                    compute_type="float16" if self.device == "cuda" else "float32"
                )
                logger.info("Model loaded successfully")
            except Exception as e:
                raise WhisperXChunkerError(f"Failed to load model: {str(e)}")
        return self._model

    def _preprocess_audio(self, audio_array: np.ndarray) -> np.ndarray:
        """Preprocess audio array with proper error handling"""
        try:
            if not isinstance(audio_array, np.ndarray):
                raise ValueError("Input must be a numpy array")
            
            # Convert to float32 to ensure compatibility
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Handle NaN and Inf values
            if np.isnan(audio_array).any() or np.isinf(audio_array).any():
                logger.warning("Found NaN or Inf values in audio array, replacing with zeros")
                audio_array = np.nan_to_num(audio_array)

            # Normalize
            max_val = np.abs(audio_array).max()
            if max_val > 0:
                audio_array = audio_array / max_val
                
            return audio_array
            
        except Exception as e:
            raise WhisperXChunkerError(f"Audio preprocessing failed: {str(e)}")

    def chunk(self, audio_array: np.ndarray, sampling_rate: int) -> List[Dict[str, Union[np.ndarray, int]]]:
        """Chunk audio array into segments"""
        try:
            # Enable TF32 for better compatibility with newer PyTorch versions
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            audio_array = self._preprocess_audio(audio_array)
            
            with torch.cuda.amp.autocast(enabled=True):
                result = self.model.transcribe(
                    audio_array,
                    batch_size=16,
                    chunk_size=30,
                )
            
            segments = result.get("segments", [])
            if not segments:
                logger.warning("No segments found in audio")
                return [{"array": audio_array, "sampling_rate": sampling_rate}]

            chunks = [
                {
                    "array": audio_array[int(segment["start"] * sampling_rate):int(segment["end"] * sampling_rate)],
                    "sampling_rate": sampling_rate
                }
                for segment in segments
            ]
            
            logger.debug(f"Successfully chunked audio into {len(chunks)} segments")
            return chunks

        except Exception as e:
            logger.error(f"Chunking error details: {str(e)}")
            raise WhisperXChunkerError(f"Chunking failed: {str(e)}")
