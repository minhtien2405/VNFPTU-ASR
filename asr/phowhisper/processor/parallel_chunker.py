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
            audio_array = example["audio"]["array"]
            if audio_array.dtype != np.float32:
                print(f"[Worker {worker_id}] Warning: Audio dtype {audio_array.dtype}, converting to float32.")
                audio_array = audio_array.astype(np.float32)
            if np.isnan(audio_array).any() or np.isinf(audio_array).any():
                print(f"[Worker {worker_id}] Error: Audio contains NaN/Inf, replacing with zeros.")
                audio_array = np.nan_to_num(audio_array)
            max_val = np.abs(audio_array).max()
            if max_val > 0:
                audio_array = audio_array / max_val
            sampling_rate = example["audio"]["sampling_rate"]

            chunks = chunker.chunk(audio_array, sampling_rate)
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
