import logging
import numpy as np
from typing import List, Dict, Union, Tuple, Optional
from datasets import load_dataset, Audio, Dataset
import torch
from functools import partial
import wandb
from pathlib import Path
import pandas as pd

from processor.audio_chunker import AudioChunker

logger = logging.getLogger(__name__)

class ChunkformerDataProcessorError(Exception):
    """Custom exception for ChunkformerDataProcessor errors"""
    pass

def prepare_chunkformer_dataset(
    batch: Dict,
    chunker: AudioChunker,
    region: str,
    max_text_length: int = 512,
    normalize_text: bool = True
) -> Dict:
    """
    Prepare dataset batch for Chunkformer training.
    
    Args:
        batch: Dataset batch containing audio and text
        chunker: AudioChunker instance
        region: Region filter
        max_text_length: Maximum text length
        normalize_text: Whether to normalize text
        
    Returns:
        Dict: Processed batch ready for training
    """
    try:
        audio = batch["audio"]
        audio_array = audio["array"]
        sampling_rate = audio["sampling_rate"]
        text = batch["text"]

        if not isinstance(audio_array, np.ndarray):
            raise ValueError("Audio array must be numpy.ndarray")

        # Normalize text if requested
        if normalize_text:
            text = _normalize_text(text)

        # Truncate text if too long
        if len(text) > max_text_length:
            text = text[:max_text_length]
            logger.warning(f"Text truncated to {max_text_length} characters")

        # Chunk audio
        chunks = chunker.chunk_audio(audio_array, sampling_rate)

        # Prepare output - each chunk gets the same transcription
        batch["audio_chunks"] = [chunk["array"] for chunk in chunks]
        batch["chunk_metadata"] = [{
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "chunk_id": chunk["chunk_id"],
            "total_chunks": chunk["total_chunks"]
        } for chunk in chunks]
        
        # Each chunk gets the full transcription
        # (in practice, you might want to implement time-aligned transcription)
        batch["transcriptions"] = [text for _ in chunks]
        batch["text_length"] = len(text)
        batch["num_chunks"] = len(chunks)
        
        return batch

    except Exception as e:
        logger.error(f"Failed to prepare dataset batch: {str(e)}")
        raise ChunkformerDataProcessorError(f"Batch processing failed: {str(e)}")

def _normalize_text(text: str) -> str:
    """Normalize text for training."""
    # Basic text normalization
    text = text.strip()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # You can add more normalization steps here:
    # - Convert to lowercase
    # - Remove punctuation
    # - Handle special characters
    # etc.
    
    return text

class ChunkformerDataProcessor:
    """
    Data processor for Chunkformer training on VIMD dataset.
    """
    
    def __init__(self, config: object, device: str = "cuda"):
        """
        Initialize ChunkformerDataProcessor.
        
        Args:
            config: Configuration object with dataset and processing parameters
            device: Device to use for processing
        """
        self.config = config
        self.device = device
        self.region = config.region.lower()
        
        # Initialize audio chunker
        self.chunker = AudioChunker(
            chunk_size=config.model.chunk_size,
            left_context_size=config.model.left_context_size,
            right_context_size=config.model.right_context_size,
            sampling_rate=config.dataset.sampling_rate,
            chunk_overlap=config.data_processing.chunk_overlap,
            min_chunk_length=config.data_processing.min_chunk_length,
            max_chunk_length=config.data_processing.max_chunk_length,
            normalize_audio=config.data_processing.normalize_audio,
            padding_value=config.data_processing.padding_value
        )
        
        # Text processing parameters
        self.max_text_length = config.data_processing.max_text_length
        self.normalize_text = config.data_processing.normalize_text
        
        logger.info(f"ChunkformerDataProcessor initialized for region '{self.region}'")

    def load_dataset(self) -> Tuple[Dataset, Dataset]:
        """Load and prepare the VIMD dataset."""
        try:
            logger.info(f"Loading dataset: {self.config.dataset.name}")
            
            # Load dataset from HuggingFace
            dataset = load_dataset(
                self.config.dataset.name, 
                cache_dir=self.config.dataset.cache_dir
            )
            
            # Cast audio column to correct sampling rate
            train_dataset = dataset["train"].cast_column(
                "audio", 
                Audio(sampling_rate=self.config.dataset.sampling_rate)
            )
            valid_dataset = dataset["valid"].cast_column(
                "audio", 
                Audio(sampling_rate=self.config.dataset.sampling_rate)
            )
            
            # Filter by region if not "all"
            if self.region != "all":
                train_dataset = self._filter_by_region(train_dataset)
                valid_dataset = self._filter_by_region(valid_dataset)
            
            # Log dataset statistics
            self._log_dataset_stats(train_dataset, valid_dataset)
            
            logger.info(
                f"Dataset loaded successfully. Train size: {len(train_dataset)}, "
                f"Valid size: {len(valid_dataset)}"
            )
            
            return train_dataset, valid_dataset

        except Exception as e:
            raise ChunkformerDataProcessorError(f"Failed to load dataset: {str(e)}")

    def _filter_by_region(self, dataset: Dataset) -> Dataset:
        """Filter dataset by specified region."""
        try:
            filtered = dataset.filter(lambda x: x["region"].lower() == self.region)
            logger.info(f"Filtered dataset for region '{self.region}': {len(filtered)} samples")
            return filtered
        except Exception as e:
            logger.error(f"Failed to filter by region '{self.region}': {str(e)}")
            return dataset

    def _log_dataset_stats(self, train_dataset: Dataset, valid_dataset: Dataset):
        """Log dataset statistics to wandb and logger."""
        try:
            # Basic statistics
            stats = {
                "dataset_size_train": len(train_dataset),
                "dataset_size_valid": len(valid_dataset),
                "region": self.region
            }
            
            # Sample a few examples to get audio length statistics
            sample_size = min(100, len(train_dataset))
            sample_indices = np.random.choice(len(train_dataset), sample_size, replace=False)
            sample_lengths = []
            
            for idx in sample_indices:
                try:
                    audio_length = len(train_dataset[idx]["audio"]["array"]) / self.config.dataset.sampling_rate
                    sample_lengths.append(audio_length)
                except:
                    continue
            
            if sample_lengths:
                stats.update({
                    "avg_audio_length": np.mean(sample_lengths),
                    "min_audio_length": np.min(sample_lengths),
                    "max_audio_length": np.max(sample_lengths),
                    "median_audio_length": np.median(sample_lengths)
                })
            
            # Log to wandb if available
            if wandb.run:
                wandb.log(stats)
            
            logger.info(f"Dataset statistics: {stats}")
            
        except Exception as e:
            logger.warning(f"Failed to compute dataset statistics: {str(e)}")

    def process(self, dataset: Dataset) -> Dataset:
        """
        Process the dataset for Chunkformer training.
        
        Args:
            dataset: Raw dataset to process
            
        Returns:
            Dataset: Processed dataset ready for training
        """
        try:
            logger.info("Processing dataset for Chunkformer training")
            
            # Create partial function with fixed parameters
            process_fn = partial(
                prepare_chunkformer_dataset,
                chunker=self.chunker,
                region=self.region,
                max_text_length=self.max_text_length,
                normalize_text=self.normalize_text
            )
            
            # Process dataset
            processed = dataset.map(
                process_fn,
                remove_columns=dataset.column_names,
                num_proc=1,  # Use single process to avoid multiprocessing issues
                desc="Processing dataset for Chunkformer"
            )
            
            logger.info(f"Dataset processing completed: {len(processed)} samples")
            
            # Log chunk statistics
            self._log_chunk_stats(processed)
            
            return processed

        except Exception as e:
            logger.error(f"Dataset processing failed: {str(e)}")
            raise ChunkformerDataProcessorError(f"Dataset processing failed: {str(e)}")

    def _log_chunk_stats(self, processed_dataset: Dataset):
        """Log chunk statistics from processed dataset."""
        try:
            # Sample some examples to get chunk statistics
            sample_size = min(50, len(processed_dataset))
            sample_indices = np.random.choice(len(processed_dataset), sample_size, replace=False)
            
            total_chunks = 0
            chunk_lengths = []
            
            for idx in sample_indices:
                try:
                    sample = processed_dataset[idx]
                    num_chunks = sample["num_chunks"]
                    total_chunks += num_chunks
                    
                    for chunk in sample["audio_chunks"]:
                        chunk_length = len(chunk) / self.config.dataset.sampling_rate
                        chunk_lengths.append(chunk_length)
                except:
                    continue
            
            if chunk_lengths:
                chunk_stats = {
                    "avg_chunks_per_sample": total_chunks / sample_size,
                    "avg_chunk_length": np.mean(chunk_lengths),
                    "min_chunk_length": np.min(chunk_lengths),
                    "max_chunk_length": np.max(chunk_lengths),
                    "total_chunks_sampled": len(chunk_lengths)
                }
                
                # Log to wandb if available
                if wandb.run:
                    wandb.log(chunk_stats)
                
                logger.info(f"Chunk statistics: {chunk_stats}")
                
        except Exception as e:
            logger.warning(f"Failed to compute chunk statistics: {str(e)}")

    def create_data_collator(self):
        """Create data collator for Chunkformer training."""
        from torch.nn.utils.rnn import pad_sequence
        
        def collate_fn(batch):
            """
            Collate function for Chunkformer training.
            Handles variable-length audio chunks and transcriptions.
            """
            try:
                # Flatten all audio chunks from all samples
                all_audio_chunks = []
                all_transcriptions = []
                all_metadata = []
                
                for sample in batch:
                    audio_chunks = sample["audio_chunks"]
                    transcriptions = sample["transcriptions"]
                    metadata = sample["chunk_metadata"]
                    
                    for audio_chunk, transcription, meta in zip(audio_chunks, transcriptions, metadata):
                        all_audio_chunks.append(torch.from_numpy(audio_chunk).float())
                        all_transcriptions.append(transcription)
                        all_metadata.append(meta)
                
                # Pad audio chunks to same length
                if all_audio_chunks:
                    padded_audio = pad_sequence(
                        all_audio_chunks, 
                        batch_first=True, 
                        padding_value=self.chunker.padding_value
                    )
                else:
                    padded_audio = torch.empty(0)
                
                return {
                    "audio": padded_audio,
                    "transcriptions": all_transcriptions,
                    "metadata": all_metadata
                }
                
            except Exception as e:
                logger.error(f"Collation failed: {str(e)}")
                raise
        
        return collate_fn