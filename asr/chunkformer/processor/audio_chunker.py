import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import torch
import torchaudio
from pathlib import Path

logger = logging.getLogger(__name__)

class AudioChunker:
    """
    Audio chunker for Chunkformer that handles long audio files
    by splitting them into chunks with proper context.
    """
    
    def __init__(
        self,
        chunk_size: int = 64,
        left_context_size: int = 128,
        right_context_size: int = 128,
        sampling_rate: int = 16000,
        chunk_overlap: float = 0.1,
        min_chunk_length: float = 1.0,
        max_chunk_length: float = 30.0,
        normalize_audio: bool = True,
        padding_value: float = 0.0
    ):
        """
        Initialize AudioChunker.
        
        Args:
            chunk_size (int): Base chunk size for Chunkformer
            left_context_size (int): Left context size for each chunk
            right_context_size (int): Right context size for each chunk
            sampling_rate (int): Target sampling rate
            chunk_overlap (float): Overlap ratio between chunks (0.0 to 1.0)
            min_chunk_length (float): Minimum chunk length in seconds
            max_chunk_length (float): Maximum chunk length in seconds
            normalize_audio (bool): Whether to normalize audio
            padding_value (float): Value used for padding
        """
        self.chunk_size = chunk_size
        self.left_context_size = left_context_size
        self.right_context_size = right_context_size
        self.sampling_rate = sampling_rate
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        self.normalize_audio = normalize_audio
        self.padding_value = padding_value
        
        # Calculate frame-level parameters
        self.min_chunk_frames = int(min_chunk_length * sampling_rate)
        self.max_chunk_frames = int(max_chunk_length * sampling_rate)
        self.overlap_frames = int(self.max_chunk_frames * chunk_overlap)
        
        logger.info(
            f"AudioChunker initialized: chunk_size={chunk_size}, "
            f"contexts=({left_context_size}, {right_context_size}), "
            f"chunk_length=({min_chunk_length}s, {max_chunk_length}s), "
            f"overlap={chunk_overlap}"
        )

    def chunk_audio(
        self, 
        audio: np.ndarray, 
        sampling_rate: Optional[int] = None,
        audio_length: Optional[float] = None
    ) -> List[Dict]:
        """
        Chunk audio array into overlapping segments.
        
        Args:
            audio (np.ndarray): Input audio array
            sampling_rate (int, optional): Audio sampling rate
            audio_length (float, optional): Audio length in seconds
            
        Returns:
            List[Dict]: List of audio chunks with metadata
        """
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
            
        # Resample if necessary
        if sampling_rate != self.sampling_rate:
            audio = self._resample_audio(audio, sampling_rate, self.sampling_rate)
            sampling_rate = self.sampling_rate
            
        # Normalize audio if requested
        if self.normalize_audio:
            audio = self._normalize_audio(audio)
            
        # Calculate audio length
        if audio_length is None:
            audio_length = len(audio) / sampling_rate
            
        # Decide chunking strategy based on audio length
        if audio_length <= self.max_chunk_length:
            # Short audio: return as single chunk
            return [{
                "array": audio,
                "sampling_rate": sampling_rate,
                "start_time": 0.0,
                "end_time": audio_length,
                "chunk_id": 0,
                "total_chunks": 1
            }]
        else:
            # Long audio: split into chunks
            return self._split_long_audio(audio, sampling_rate, audio_length)

    def _split_long_audio(
        self, 
        audio: np.ndarray, 
        sampling_rate: int,
        audio_length: float
    ) -> List[Dict]:
        """Split long audio into overlapping chunks."""
        chunks = []
        chunk_frames = self.max_chunk_frames
        step_frames = chunk_frames - self.overlap_frames
        
        start_frame = 0
        chunk_id = 0
        
        while start_frame < len(audio):
            end_frame = min(start_frame + chunk_frames, len(audio))
            
            # Extract chunk
            chunk_audio = audio[start_frame:end_frame]
            
            # Pad if necessary (for the last chunk)
            if len(chunk_audio) < self.min_chunk_frames:
                if start_frame == 0:
                    # If even the first chunk is too short, pad it
                    pad_length = self.min_chunk_frames - len(chunk_audio)
                    chunk_audio = np.pad(
                        chunk_audio, 
                        (0, pad_length), 
                        mode='constant', 
                        constant_values=self.padding_value
                    )
                else:
                    # Skip chunks that are too short (except the first one)
                    break
            
            # Calculate timing information
            start_time = start_frame / sampling_rate
            end_time = end_frame / sampling_rate
            
            chunk_dict = {
                "array": chunk_audio,
                "sampling_rate": sampling_rate,
                "start_time": start_time,
                "end_time": end_time,
                "chunk_id": chunk_id,
                "total_chunks": None  # Will be set after all chunks are processed
            }
            
            chunks.append(chunk_dict)
            
            # Move to next chunk
            start_frame += step_frames
            chunk_id += 1
            
            # Break if we've covered the entire audio
            if end_frame >= len(audio):
                break
                
        # Update total_chunks for all chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk["total_chunks"] = total_chunks
            
        logger.debug(f"Split audio ({audio_length:.2f}s) into {total_chunks} chunks")
        return chunks

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        if np.max(np.abs(audio)) > 0:
            return audio / np.max(np.abs(audio))
        return audio

    def _resample_audio(
        self, 
        audio: np.ndarray, 
        orig_sr: int, 
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sampling rate."""
        if orig_sr == target_sr:
            return audio
            
        # Convert to torch tensor for resampling
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr, 
            new_freq=target_sr
        )
        resampled = resampler(audio_tensor).squeeze(0).numpy()
        
        logger.debug(f"Resampled audio from {orig_sr}Hz to {target_sr}Hz")
        return resampled

    def load_and_chunk_audio(self, audio_path: str) -> List[Dict]:
        """
        Load audio file and chunk it.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            List[Dict]: List of audio chunks
        """
        try:
            # Load audio using torchaudio
            audio_tensor, orig_sr = torchaudio.load(audio_path)
            
            # Convert to numpy and handle multi-channel audio
            if audio_tensor.shape[0] > 1:
                # Convert stereo to mono by averaging channels
                audio = torch.mean(audio_tensor, dim=0).numpy()
            else:
                audio = audio_tensor.squeeze(0).numpy()
                
            # Calculate audio length
            audio_length = len(audio) / orig_sr
            
            # Chunk the audio
            chunks = self.chunk_audio(audio, orig_sr, audio_length)
            
            # Add file path to each chunk
            for chunk in chunks:
                chunk["file_path"] = audio_path
                
            logger.info(f"Loaded and chunked audio file: {audio_path} -> {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to load audio file {audio_path}: {str(e)}")
            raise

    def get_chunk_info(self, chunks: List[Dict]) -> Dict:
        """Get summary information about chunks."""
        if not chunks:
            return {}
            
        total_duration = sum(chunk["end_time"] - chunk["start_time"] for chunk in chunks)
        chunk_lengths = [len(chunk["array"]) / chunk["sampling_rate"] for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_duration": total_duration,
            "avg_chunk_length": np.mean(chunk_lengths),
            "min_chunk_length": np.min(chunk_lengths),
            "max_chunk_length": np.max(chunk_lengths),
            "sampling_rate": chunks[0]["sampling_rate"]
        }