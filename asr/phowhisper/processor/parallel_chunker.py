import torch.multiprocessing as mp
import os, json
from datasets import Dataset
from .whisperx_chunker import WhisperXChunker
import numpy as np

def chunk_worker(dataset, config_dict, device, worker_id, queue):
    import torch
    import whisperx
    chunker = WhisperXChunker(
        model_path=config_dict["model_path"],
        device=device,
        language=config_dict["language"],
    )

    output = []
    for i, example in enumerate(dataset):
        try:
            chunks = chunker.chunk(example["audio"]["array"], example["audio"]["sampling_rate"])
            input_features = [
                config_dict["processor"](
                    chunk["array"],
                    sampling_rate=chunk["sampling_rate"],
                    return_tensors="pt"
                ).input_features[0]
                for chunk in chunks
            ]
            output.append({
                "input_features": input_features,
                "labels": config_dict["processor"].tokenizer(example["text"]).input_ids
            })
        except Exception as e:
            print(f"[Worker {worker_id}] Error on sample {i}: {e}")
            continue
    queue.put(output)
