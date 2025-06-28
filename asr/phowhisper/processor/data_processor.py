import logging
from typing import List, Dict, Union

import numpy as np
from datasets import load_dataset, Audio, Dataset
from transformers import WhisperProcessor
import whisperx
import wandb

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, config, processor: WhisperProcessor, device: str):
        self.config = config
        self.processor = processor
        self.device = device
        self.region = config.region.lower()

        self.whisperx_model = whisperx.load_model(
            config.model.model_id,
            device=device,
            language=config.model.language,
            compute_type="float16"
        )
        self.vad_model = whisperx.load_vad_model(device)

        logger.info(f"[DataProcessor] Initialized with model '{config.model.model_id}' on region '{self.region}'")

    def load_dataset(self) -> (Dataset, Dataset):
        logger.info(f"[DataProcessor] Loading dataset: {self.config.dataset.name}")
        dataset = load_dataset(self.config.dataset.name, cache_dir=self.config.dataset.cache_dir)

        train_dataset = dataset["train"].cast_column("audio", Audio(sampling_rate=self.config.dataset.sampling_rate))
        valid_dataset = dataset["valid"].cast_column("audio", Audio(sampling_rate=self.config.dataset.sampling_rate))

        if self.region != "all":
            train_dataset = self._filter_by_region(train_dataset, split="train")
            valid_dataset = self._filter_by_region(valid_dataset, split="valid")

        logger.info(f"[DataProcessor] Loaded dataset: train={len(train_dataset)} samples, valid={len(valid_dataset)} samples")

        if wandb.run:
            wandb.log({
                "dataset_size_train": len(train_dataset),
                "dataset_size_valid": len(valid_dataset)
            })

        return train_dataset, valid_dataset

    def _filter_by_region(self, dataset: Dataset, split: str) -> Dataset:
        logger.info(f"[DataProcessor] Filtering '{split}' split for region: {self.region}")
        filtered_dataset = dataset.filter(lambda x: x["region"].lower() == self.region)
        logger.info(f"[DataProcessor] After filtering '{split}': {len(filtered_dataset)} samples")
        return filtered_dataset

    def chunk_audio(self, audio_array: np.ndarray, sampling_rate: int) -> List[Dict[str, Union[np.ndarray, int]]]:
        result = self.whisperx_model.transcribe(
            {"array": audio_array, "sampling_rate": sampling_rate},
            batch_size=16,
            chunk_size=30
        )
        segments = result.get("segments", [])
        chunks = [
            {
                "array": audio_array[int(segment["start"] * sampling_rate):int(segment["end"] * sampling_rate)],
                "sampling_rate": sampling_rate
            }
            for segment in segments
        ]
        logger.debug(f"[DataProcessor] Chunked audio into {len(chunks)} segments")
        return chunks

    def prepare_dataset(self, batch: Dict) -> Dict:
        audio = batch["audio"]
        audio_array = audio["array"]
        sampling_rate = audio["sampling_rate"]
        audio_length = len(audio_array) / sampling_rate

        if audio_length > 30:
            logger.info(f"[DataProcessor] Chunking audio sample of length {audio_length:.2f}s")
            chunks = self.chunk_audio(audio_array, sampling_rate)
        else:
            chunks = [{"array": audio_array, "sampling_rate": sampling_rate}]

        batch["input_features"] = [
            self.processor(chunk["array"], sampling_rate=chunk["sampling_rate"], return_tensors="pt").input_features[0]
            for chunk in chunks
        ]
        batch["labels"] = self.processor.tokenizer(batch["text"]).input_ids

        return batch

    def process(self, dataset: Dataset) -> Dataset:
        logger.info("[DataProcessor] Starting dataset preprocessing")
        processed_dataset = dataset.map(
            self.prepare_dataset,
            remove_columns=dataset.column_names,
            num_proc=4,
            desc="Processing dataset with WhisperX"
        )
        logger.info("[DataProcessor] Finished dataset processing")
        return processed_dataset