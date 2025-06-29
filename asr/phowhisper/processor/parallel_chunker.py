import torch.multiprocessing as mp
import os, json
from datasets import Dataset
from .whisperx_chunker import WhisperXChunker
import numpy as np

def chunk_worker(dataset, config_dict, device, worker_id, queue):
    import torch
    import whisperx
    from tqdm import tqdm

    chunker = WhisperXChunker(
        model_path=config_dict["model_path"],
        device=device,
        language=config_dict["language"],
    )

    output = []
    total = len(dataset)
    with tqdm(total=total, desc=f"Worker {worker_id}", position=worker_id) as pbar:
        for i, example in enumerate(dataset):
            try:
                audio_array = example["audio"]["array"]
                # Silently convert dtype and handle NaN/Inf
                if audio_array.dtype != np.float32:
                    audio_array = audio_array.astype(np.float32)
                if np.isnan(audio_array).any() or np.isinf(audio_array).any():
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
            pbar.update(1)
    print(f"[Worker {worker_id}] Finished processing {total} samples.")
    queue.put(output)
