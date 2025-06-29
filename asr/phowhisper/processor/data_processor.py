import logging
import numpy as np
from typing import List, Dict, Union
from datasets import load_dataset, Audio, Dataset
from transformers import WhisperProcessor
from functools import partial
import wandb

from processor.whisperx_chunker import WhisperXChunker

logger = logging.getLogger(__name__)

def prepare_dataset(
    batch: Dict,
    processor: WhisperProcessor,
    chunker: WhisperXChunker,
    region: str,
    chunk_threshold: float = 30.0
) -> Dict:
    audio = batch["audio"]
    audio_array = audio["array"]

    # if audio_array.dtype != np.float32:
    #     audio_array = audio_array.astype(np.float32)

    if np.abs(audio_array).max() > 0:
        audio_array = audio_array / np.abs(audio_array).max()

    sampling_rate = audio["sampling_rate"]
    audio_length = len(audio_array) / sampling_rate

    if audio_length > chunk_threshold:
        chunks = chunker.chunk(audio_array, sampling_rate)
    else:
        chunks = [{"array": audio_array, "sampling_rate": sampling_rate}]

    batch["input_features"] = [
        processor(chunk["array"], sampling_rate=chunk["sampling_rate"], return_tensors="pt").input_features[0]
        for chunk in chunks
    ]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch


class DataProcessor:
    def __init__(self, config, processor: WhisperProcessor, device: str):
        self.config = config
        self.processor = processor
        self.device = device
        self.region = config.region.lower()

        self.chunker = WhisperXChunker(
            model_path="./converted_phowhisper",
            device=device,
            language=config.model.language,
        )

        logger.info(f"[DataProcessor] Initialized on region '{self.region}'. Chunker is ready for lazy loading.")

    def load_dataset(self) -> (Dataset, Dataset):
        logger.info(f"[DataProcessor] Loading dataset: {self.config.dataset.name}")
        dataset = load_dataset(self.config.dataset.name, cache_dir=self.config.dataset.cache_dir)
        train_dataset = dataset["train"].cast_column("audio", Audio(sampling_rate=self.config.dataset.sampling_rate))
        valid_dataset = dataset["valid"].cast_column("audio", Audio(sampling_rate=self.config.dataset.sampling_rate))

        if self.region != "all":
            train_dataset = self._filter_by_region(train_dataset, split="train")
            valid_dataset = self._filter_by_region(valid_dataset, split="valid")

        if wandb.run:
            wandb.log({
                "dataset_size_train": len(train_dataset),
                "dataset_size_valid": len(valid_dataset)
            })

        return train_dataset, valid_dataset

    def _filter_by_region(self, dataset: Dataset, split: str) -> Dataset:
        return dataset.filter(lambda x: x["region"].lower() == self.region)
    
    # def process(self, dataset: Dataset) -> Dataset:
    #     import multiprocessing as mp
    #     from processor.parallel_chunker import chunk_worker

    #     num_workers = mp.cpu_count() // 2
    #     logger.info(f"[DataProcessor] Preprocessing dataset with {num_workers} workers using multiprocessing")
    #     mp.set_start_method("spawn", force=True)
    #     queue = mp.Queue()

    #     chunks = [dataset.shard(num_workers, i) for i in range(num_workers)]

    #     config_dict = {
    #         "model_path": "./converted_phowhisper",
    #         "language": self.config.model.language,
    #         "processor": self.processor
    #     }

    #     processes = []
    #     for i in range(num_workers):
    #         p = mp.Process(target=chunk_worker, args=(chunks[i], config_dict, self.device, i, queue))
    #         p.start()   
    #         processes.append(p)

    #     results = []
    #     for _ in processes:
    #         results.extend(queue.get())

    #     for p in processes:
    #         p.join()

    #     return Dataset.from_list(results)


    def process(self, dataset: Dataset) -> Dataset:
        logger.info(f"[DataProcessor] Preprocessing dataset with multiprocessing")

        process_fn = partial(
            prepare_dataset,
            processor=self.processor,
            chunker=self.chunker,
            region=self.region,
            chunk_threshold=30.0
        )

        return dataset.map(
            process_fn,
            remove_columns=dataset.column_names,
            num_proc=1,
            desc="Preprocessing dataset"
        )