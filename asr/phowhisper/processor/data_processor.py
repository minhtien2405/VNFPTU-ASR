import logging
import numpy as np
from typing import List, Dict, Union, Tuple, Optional
from datasets import load_dataset, Audio, Dataset
from transformers import WhisperProcessor
from functools import partial
import wandb
from pathlib import Path

from processor.whisperx_chunker import WhisperXChunker, WhisperXChunkerError

logger = logging.getLogger(__name__)

class DataProcessorError(Exception):
    """Custom exception for DataProcessor errors"""
    pass

def prepare_dataset(
    batch: Dict,
    processor: WhisperProcessor,
    chunker: WhisperXChunker,
    region: str,
    chunk_threshold: float = 30.0,
    max_label_length: int = 448  # Add max length parameter
) -> Dict:
    try:
        audio = batch["audio"]
        audio_array = audio["array"]
        sampling_rate = audio["sampling_rate"]

        if not isinstance(audio_array, np.ndarray):
            raise ValueError("Audio array must be numpy.ndarray")

        # Normalize if needed
        if np.abs(audio_array).max() > 0:
            audio_array = audio_array / np.abs(audio_array).max()

        # Decide chunking based on audio length
        audio_length = len(audio_array) / sampling_rate
        chunks = (chunker.chunk(audio_array, sampling_rate) 
                 if audio_length > chunk_threshold 
                 else [{"array": audio_array, "sampling_rate": sampling_rate}])

        # Process each chunk
        batch["input_features"] = [
            processor(
                chunk["array"], 
                sampling_rate=chunk["sampling_rate"], 
                return_tensors="pt"
            ).input_features[0]
            for chunk in chunks
        ]
        
        # Always create a list of labels, one per chunk (even if only one chunk)
        encoded_labels = [
            processor.tokenizer(
                batch["text"], 
                truncation=True,
                max_length=max_label_length,
                return_tensors="pt"
            ).input_ids[0]
            for _ in chunks
        ]
        batch["labels"] = encoded_labels
        
        return batch

    except Exception as e:
        logger.error(f"Failed to prepare dataset batch: {str(e)}")
        raise DataProcessorError(f"Batch processing failed: {str(e)}")

class DataProcessor:
    def __init__(self, config: object, processor: WhisperProcessor, device: str):
        self.config = config
        self.processor = processor
        self.device = device
        self.region = config.region.lower()
        self.max_label_length = 448  # Add as class attribute
        
        try:
            self.chunker = WhisperXChunker(
                model_path=Path("./converted_phowhisper"),
                device=device,
                language=config.model.language,
            )
            logger.info(f"DataProcessor initialized for region '{self.region}'")
            
        except Exception as e:
            raise DataProcessorError(f"Failed to initialize DataProcessor: {str(e)}")

    def load_dataset(self) -> Tuple[Dataset, Dataset]:
        """Load and prepare the dataset"""
        try:
            logger.info(f"Loading dataset: {self.config.dataset.name}")
            dataset = load_dataset(
                self.config.dataset.name, 
                cache_dir=self.config.dataset.cache_dir
            )
            
            train_dataset = dataset["train"].cast_column(
                "audio", 
                Audio(sampling_rate=self.config.dataset.sampling_rate)
            )
            valid_dataset = dataset["valid"].cast_column(
                "audio", 
                Audio(sampling_rate=self.config.dataset.sampling_rate)
            )

            if self.region != "all":
                train_dataset = self._filter_by_region(train_dataset)
                valid_dataset = self._filter_by_region(valid_dataset)

            # Log dataset sizes
            if wandb.run:
                wandb.log({
                    "dataset_size_train": len(train_dataset),
                    "dataset_size_valid": len(valid_dataset)
                })

            logger.info(f"Dataset loaded successfully. Train size: {len(train_dataset)}, Valid size: {len(valid_dataset)}")
            return train_dataset, valid_dataset

        except Exception as e:
            raise DataProcessorError(f"Failed to load dataset: {str(e)}")

    def _filter_by_region(self, dataset: Dataset) -> Dataset:
        """Filter dataset by region"""
        filtered = dataset.filter(lambda x: x["region"].lower() == self.region)
        logger.debug(f"Filtered dataset for region {self.region}: {len(filtered)} samples")
        return filtered

    def process(self, dataset: Dataset) -> Dataset:
        """Process the dataset in single-processing mode"""
        try:
            logger.info("Processing dataset in single-processing mode")
            
            process_fn = partial(
                prepare_dataset,
                processor=self.processor,
                chunker=self.chunker,
                region=self.region,
                chunk_threshold=30.0,
                max_label_length=self.max_label_length  # Pass max length
            )

            processed = dataset.map(
                process_fn,
                remove_columns=dataset.column_names,
                num_proc=1,
                desc="Processing dataset"
            )
            
            logger.info(f"Dataset processing completed successfully: {len(processed)} samples")
            return processed

        except Exception as e:
            logger.error(f"Dataset processing failed: {str(e)}")
            raise DataProcessorError(f"Dataset processing failed: {str(e)}")