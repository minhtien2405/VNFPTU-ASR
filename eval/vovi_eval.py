import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Union, Optional
import torch
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import evaluate
import numpy as np
import librosa
import requests
import re
from tqdm.auto import tqdm

@dataclass
class EvalConfig:
    model_id: str = "minhtien2405/vovinam-wav2vec2-base-vi"
    dataset_id: str = "minhtien2405/VoviAIDataset"
    split_name: str = "test"
    cache_dir: str = "./cache"
    per_device_eval_batch_size: int = 8
    log_dir: str = "./logs"

def setup_logging(config: EvalConfig) -> logging.Logger:
    os.makedirs(config.log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(config.log_dir, "evaluation.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    return logging.getLogger(__name__)

from utils.vovi_utils import download_audio_from_s3, normalize_text

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

def load_and_prepare_data(config: EvalConfig, logger: logging.Logger):
    logger.info(f"Loading dataset '{config.dataset_id}' for evaluation on split '{config.split_name}'")
    dataset = load_dataset(config.dataset_id, cache_dir=config.cache_dir, split=config.split_name)
    
    def download_and_validate(sample):
        sample["is_valid"] = True
        audio_path = download_audio_from_s3(sample["audioLink"], cache_dir=os.path.join(config.cache_dir, "audio"))
        if not audio_path:
            sample["is_valid"] = False
            return sample
        sample["audio_path"] = audio_path
        try:
            duration = librosa.get_duration(path=audio_path)
            if not (0.1 < duration < 30): sample["is_valid"] = False
        except Exception:
            sample["is_valid"] = False
        text = normalize_text(sample.get("text", ""))
        if len(text) < 2: sample["is_valid"] = False
        sample["text"] = text
        return sample
    
    logger.info("Downloading and validating audio files...")
    dataset = dataset.map(download_and_validate, num_proc=4)
    dataset = dataset.filter(lambda s: s["is_valid"], num_proc=4)
    logger.info(f"Finished validation. Kept {len(dataset)} samples.")
    
    dataset = dataset.remove_columns([col for col in ["audio", "audioLink", "is_valid"] if col in dataset.column_names])
    return dataset

def prepare_for_model(batch, processor: Wav2Vec2Processor):
    try:
        audio_array, sr = librosa.load(batch["audio_path"], sr=16000)
        batch["input_values"] = processor(audio_array, sampling_rate=sr).input_values[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch
    except Exception as e:
        logging.getLogger(__name__).error(f"Error processing file {batch.get('audio_path', 'UNKNOWN')}: {e}")
        return None

def main():
    config = EvalConfig()
    logger = setup_logging(config)
    
    logger.info(f"Loading processor and model from '{config.model_id}'...")
    try:
        processor = Wav2Vec2Processor.from_pretrained(config.model_id)
        model = Wav2Vec2ForCTC.from_pretrained(config.model_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    except Exception as e:
        logger.error(f"Failed to load model/processor: {e}")
        return

    test_dataset = load_and_prepare_data(config, logger)
    
    logger.info("Preparing dataset for the model (tokenizing)...")
    processed_test_dataset = test_dataset.map(
        lambda batch: prepare_for_model(batch, processor),
        num_proc=4
    ).filter(lambda x: x is not None)
    
    wer_metric = evaluate.load("wer")
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    logger.info("Setting up Trainer for evaluation...")
    training_args = TrainingArguments(
        output_dir="./evaluation_results",
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorCTCWithPadding(processor=processor),
    )

    logger.info(f"Starting evaluation on '{config.split_name}' split...")
    results = trainer.evaluate(eval_dataset=processed_test_dataset)

    wer_score = results.get("eval_wer")
    if wer_score is not None:
        logger.info(f"Evaluation finished. WER: {wer_score:.4f}")
        print("\n" + "="*50)
        print(f"✅ Evaluation complete on '{config.split_name}' split!")
        print(f"   Word Error Rate (WER): {wer_score:.4f} ({wer_score*100:.2f}%)")
        print("="*50 + "\n")
    else:
        logger.error("Evaluation failed. Could not retrieve WER score.")
        print("Evaluation failed. Check evaluation.log for details.")


if __name__ == "__main__":
    main()