import torch
import whisperx
from datasets import Dataset
from .whisperx_chunker import WhisperXChunker


def chunk_worker(dataset, language, device, processor, queue, worker_id):
    chunker = WhisperXChunker(
        model_path="./converted_phowhisper",
        device=device,
        language=language
    )

    output = []
    for i, example in enumerate(dataset):
        try:
            # Chuyển đổi audio array sang float32
            audio_array = example["audio"]["array"]
            if isinstance(audio_array, np.ndarray):
                audio_array = audio_array.astype(np.float32)
            elif isinstance(audio_array, torch.Tensor):
                audio_array = audio_array.to(dtype=torch.float32)

            chunks = chunker.chunk(audio_array, example["audio"]["sampling_rate"])
            input_features = [
                processor(
                    chunk["array"],
                    sampling_rate=chunk["sampling_rate"],
                    return_tensors="pt"
                ).input_features[0].to(dtype=torch.float32, device=device)  # Chuyển sang float32 và device
                for chunk in chunks
            ]
            output.append({
                "input_features": input_features,
                "labels": processor.tokenizer(example["text"]).input_ids
            })
        except Exception as e:
            print(f"[Worker {worker_id}] Error on sample {i}: {e}")
            continue
    queue.put(output)