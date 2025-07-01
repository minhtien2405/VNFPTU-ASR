import whisperx
import logging
import numpy as np
from typing import List, Dict, Union, Optional
import torch
import gc
from pathlib import Path

logger = logging.getLogger(__name__)

class WhisperXChunkerError(Exception):
    """Custom exception for WhisperXChunker errors"""
    pass

class WhisperXChunker:
    def __init__(self, model_path: str, device: str, language: str, 
                batch_size: int = 16, compute_type: str = "float16"):
        self.model_path = model_path
        self.original_device = device
        
        # Normalize device string
        if device == "auto":
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Validate device
        if self.device.startswith('cuda'):
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                self.device = 'cpu'
            else:
                try:
                    device_id = int(self.device.split(':')[1])
                    if device_id >= torch.cuda.device_count():
                        logger.warning(f"CUDA device {device_id} not available. Using device 0.")
                        self.device = 'cuda'
                    logger.info(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
                except (IndexError, ValueError):
                    self.device = 'cuda'
        
        self.language = language.lower()
        self.batch_size = batch_size
        self.compute_type = "float16" if self.device.startswith('cuda') else "float32"
        self._model = None
        self._align_model = None
        self._align_metadata = None

    def _validate_init_params(self) -> None:
        if not self.device.startswith(('cuda', 'cpu')):
            raise WhisperXChunkerError(f"Invalid device: {self.device}")
            
        if self.device.startswith('cuda'):
            # Validate CUDA device ID
            try:
                device_id = int(self.device.split(':')[1])
                if device_id >= torch.cuda.device_count():
                    raise WhisperXChunkerError(f"CUDA device {device_id} not available")
            except (IndexError, ValueError):
                raise WhisperXChunkerError(f"Invalid CUDA device format: {self.device}")

    def _cleanup_gpu(self):
        """Clean up GPU memory"""
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(self, '_model') and self._model is not None:
            del self._model
            self._model = None
        if hasattr(self, '_align_model') and self._align_model is not None:
            del self._align_model
            self._align_model = None

    @property
    def model(self):
        if self._model is None:
            try:
                logger.info(f"Loading WhisperX model from {self.model_path}")
                self._model = whisperx.load_model(
                    self.model_path,
                    device=self.device,
                    compute_type=self.compute_type
                )
                logger.info("Model loaded successfully")
            except Exception as e:
                raise WhisperXChunkerError(f"Failed to load model: {str(e)}")
        return self._model

    def _load_align_model(self):
        """Load alignment model if needed"""
        if self._align_model is None:
            try:
                self._align_model, self._align_metadata = whisperx.load_align_model(
                    language_code=self.language,
                    device=self.device
                )
            except Exception as e:
                logger.warning(f"Failed to load alignment model: {str(e)}")
                self._align_model = None
                self._align_metadata = None

    def _preprocess_audio(self, audio_array: np.ndarray) -> np.ndarray:
        """Preprocess audio array with proper error handling"""
        try:
            if not isinstance(audio_array, np.ndarray):
                raise ValueError("Input must be a numpy array")
            
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
        """Chunk audio array into segments with improved alignment"""
        try:
            # # Enable TF32 
            # torch.backends.cuda.matmul.allow_tf32 = True
            # torch.backends.cudnn.allow_tf32 = True
            
            audio_array = self._preprocess_audio(audio_array)
            
            # Transcribe
            result = self.model.transcribe(
                audio_array,
                batch_size=self.batch_size,
                language=self.language
            )
            
            # Align if possible
            if self.device == "cuda":
                self._load_align_model()
                if self._align_model and self._align_metadata:
                    try:
                        result = whisperx.align(
                            result["segments"],
                            self._align_model,
                            self._align_metadata,
                            audio_array,
                            self.device,
                            return_char_alignments=False
                        )
                    except Exception as e:
                        logger.warning(f"Alignment failed: {str(e)}")

            segments = result.get("segments", [])
            if not segments:
                logger.warning("No segments found in audio")
                return [{"array": audio_array, "sampling_rate": sampling_rate}]

            # Create chunks with aligned timestamps
            chunks = []
            for segment in segments:
                start_sample = int(segment["start"] * sampling_rate)
                end_sample = int(segment["end"] * sampling_rate)
                if start_sample >= end_sample:
                    continue
                
                chunk_array = audio_array[start_sample:end_sample]
                if len(chunk_array) > 0:
                    chunks.append({
                        "array": chunk_array,
                        "sampling_rate": sampling_rate,
                        "start": segment["start"],
                        "end": segment["end"]
                    })

            logger.debug(f"Successfully chunked audio into {len(chunks)} segments")
            return chunks

        except Exception as e:
            logger.error(f"Chunking error: {str(e)}")
            raise WhisperXChunkerError(f"Chunking failed: {str(e)}")
        finally:
            if self.device == "cuda":
                self._cleanup_gpu()
