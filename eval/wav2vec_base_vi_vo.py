import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio
import evaluate
import mlflow
import mlflow.pytorch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
from transformers import TrainerCallback
import numpy as np
import json
import logging

logging.basicConfig(level=logging.INFO)

load_dotenv("./configs/.env")
login(token=os.getenv("HF_TOKEN"))
mlflow.set_experiment("Wav2Vec2_Central_ViMD_FPTU")
cache_dir = os.getcwd() + "/cache"
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

dataset = load_dataset("nguyendv02/ViMD_Dataset", cache_dir=cache_dir)
test_dataset = dataset["test"].filter(lambda x: x["region"] == "Central")
            
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

model_id = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
processor = Wav2Vec2Processor.from_pretrained(model_id)
def prepare_dataset(batch):
	audio = batch["audio"]
	batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
	batch["labels"] = processor.tokenizer(batch["text"]).input_ids
	return batch

model = Wav2Vec2ForCTC.from_pretrained(
	model_id,
	ctc_loss_reduction="mean",
	pad_token_id=processor.tokenizer.pad_token_id,
)

logging.info(f"Model loaded: {model_id}")

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}