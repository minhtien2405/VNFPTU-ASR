import os
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Union, Optional
from datetime import datetime
import torch
from transformers import (
	Wav2Vec2Processor,
	Wav2Vec2ForCTC,
	TrainingArguments,
	Trainer,
	TrainerCallback,
)
from datasets import load_dataset, Audio
import evaluate
from dotenv import load_dotenv
from huggingface_hub import login
import numpy as np
import wandb
import re

os.environ["PYARROW_WITH_INT64"] = "1"

@dataclass
class TrainingConfig:
	model_id: str = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
	hub_model_id: str = "minhtien2405/wav2vec2-base-central-vi"
	dataset_id: str = "nguyendv02/ViMD_Dataset"
	output_dir: str = "./logs/wav2vec2-base-central-vi"
	cache_dir: str = "./cache"
	log_dir: str = "./logs"
	model_save_dir: str = "./models/wav2vec2-base-central-vi"
	per_device_train_batch_size: int = 4
	gradient_accumulation_steps: int = 8
	learning_rate: float = 3e-4
	warmup_steps: int = 20
	num_train_epochs: int = 30
	save_steps: int = 40
	eval_steps: int = 40
	logging_steps: int = 20
	save_total_limit: int = 3
	fp16: bool = True
	project_name: str = "Wav2Vec2_Central_ViMD_FPTU"
	run_name: str = f"wav2vec2_finetune_central_vi_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def setup_logging(config: TrainingConfig) -> logging.Logger:
	os.makedirs(config.log_dir, exist_ok=True)
	logging.basicConfig(
		filename=os.path.join(config.log_dir, "wav2vec_base_central_vi_250h_v0.log"),
		level=logging.INFO,
		format="%(asctime)s - %(levelname)s - %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
		force=True,
	)
	return logging.getLogger(__name__)

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

class WandbCallback(TrainerCallback):
	def on_evaluate(self, args, state, control, metrics, **kwargs):
		if "eval_wer" in metrics:
			wandb.log({"eval_wer": metrics["eval_wer"], "step": state.global_step})

def validate_audio(sample, logger: logging.Logger) -> bool:
	"""Validate audio sample for non-empty and valid format."""
	try:
		audio = sample["audio"]
		if audio["array"] is None or len(audio["array"]) == 0:
			logger.warning(f"Invalid audio sample: {sample.get('path', 'unknown')}")
			return False
		return True
	except Exception as e:
		logger.error(f"Error validating audio sample {sample.get('path', 'unknown')}: {str(e)}")
		return False
	
def validate_text(sample, logger: logging.Logger) -> bool:
	"""Validate text labels for non-empty and reasonable length."""
	try:
		text = sample["text"]
		if not text or len(text) == 0 or len(text) > 1000:  # Adjust max length as needed
			logger.warning(f"Invalid text sample: {sample.get('path', 'unknown')}, text: {text}")
			return False
		return True
	except Exception as e:
		logger.error(f"Error validating text sample {sample.get('path', 'unknown')}: {str(e)}")
		return False

def normalize_text(text: str) -> str:
	"""Clean and normalize text labels."""
	try:
		text = text.lower()  # Convert to lowercase
		text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
		text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
		return text
	except Exception as e:
		logging.getLogger(__name__).error(f"Error normalizing text: {str(e)}")
		return text  # Return original text if normalization fails

def load_and_prepare_data(config: TrainingConfig, logger: logging.Logger):
	"""Load and preprocess dataset with validation."""
	try:
		os.environ["HF_DATASETS_CACHE"] = config.cache_dir
		
		dataset = load_dataset(config.dataset_id, cache_dir=config.cache_dir)
		
		# Filter for Central region
		train_dataset = dataset["train"].filter(lambda x: x["region"] == "Central")
		valid_dataset = dataset["valid"].filter(lambda x: x["region"] == "Central")
		test_dataset = dataset["test"].filter(lambda x: x["region"] == "Central")
		
		# Validate audio samples
		train_dataset = train_dataset.filter(lambda x: validate_audio(x, logger))
		valid_dataset = valid_dataset.filter(lambda x: validate_audio(x, logger))
		test_dataset = test_dataset.filter(lambda x: validate_audio(x, logger))
		
		# Cast audio column to correct sampling rate
		train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
		valid_dataset = valid_dataset.cast_column("audio", Audio(sampling_rate=16000))
		test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
		
		# Clean text labels
		train_dataset = train_dataset.map(
			lambda batch: {"text": [normalize_text(t) for t in batch["text"]]},
			batched=True,
			batch_size=1000
		)
		valid_dataset = valid_dataset.map(
			lambda batch: {"text": [normalize_text(t) for t in batch["text"]]},
			batched=True,
			batch_size=1000
		)
		test_dataset = test_dataset.map(
			lambda batch: {"text": [normalize_text(t) for t in batch["text"]]},
			batched=True,
			batch_size=1000
		)
		
		logger.info(f"Train dataset size after validation: {len(train_dataset)}")
		logger.info(f"Validation dataset size after validation: {len(valid_dataset)}")
		logger.info(f"Test dataset size after validation: {len(test_dataset)}")
		
		return train_dataset, valid_dataset, test_dataset
	except Exception as e:
		logger.error(f"Error loading dataset: {str(e)}")
		raise

def prepare_dataset(batch, processor: Wav2Vec2Processor, logger: logging.Logger):
	"""Prepare dataset for training with error handling."""
	try:
		audio = batch["audio"]
		batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
		batch["labels"] = processor.tokenizer(batch["text"]).input_ids
		return batch
	except Exception as e:
		logger.error(f"Error processing sample: {str(e)}")
		return None

def setup_training_components(config: TrainingConfig, logger: logging.Logger):
	"""Set up model, processor, and metrics with layer freezing."""
	try:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		logger.info(f"Using device: {device}")
		
		processor = Wav2Vec2Processor.from_pretrained(config.model_id)
		model = Wav2Vec2ForCTC.from_pretrained(
			config.model_id,
			ctc_loss_reduction="mean",
			pad_token_id=processor.tokenizer.pad_token_id,
		).to(device)
		
		# Freeze feature extractor layers
		for param in model.wav2vec2.feature_extractor.parameters():
			param.requires_grad = False
		logger.info("Froze feature extractor layers for fine-tuning.")
		
		metric = evaluate.load("wer")
		
		def compute_metrics(pred):
			pred_logits = pred.predictions
			pred_ids = np.argmax(pred_logits, axis=-1)
			pred_str = processor.batch_decode(pred_ids)
			label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
			wer = metric.compute(predictions=pred_str, references=label_str)
			return {"wer": wer}
		
		return processor, model, metric, compute_metrics
	except Exception as e:
		logger.error(f"Error setting up training components: {str(e)}")
		raise

def main():
	config = TrainingConfig()
	logger = setup_logging(config)
	
	try:
		load_dotenv("./configs/.env")
		
		hf_token = os.getenv("HF_TOKEN")
		if not hf_token:
			raise ValueError("HF_TOKEN not found in .env file")
		login(token=hf_token)
		logger.info("Logged in to Hugging Face Hub")
		
		wandb_api_key = os.getenv("WANDB_API_KEY")
		if not wandb_api_key:
			raise ValueError("WANDB_API_KEY not found in .env file")
		wandb.login(key=wandb_api_key)
		logger.info("Logged in to Weights & Biases")
		
		wandb.init(
			project=config.project_name,
			name=config.run_name,
			config=vars(config)
		)
		
		os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
		torch.cuda.empty_cache()

		train_dataset, valid_dataset, test_dataset = load_and_prepare_data(config, logger)
		
				processor, model, metric, compute_metrics = setup_training_components(config, logger)
		logger.info("Starting dataset mapping...")
		train_dataset = train_dataset.map(
			lambda batch: prepare_dataset(batch, processor, logger),
			remove_columns=train_dataset.column_names,
			num_proc=1,
			keep_in_memory=False,
			batch_size=16,
		).filter(lambda x: x is not None)
		valid_dataset = valid_dataset.map(
			lambda batch: prepare_dataset(batch, processor, logger),
			remove_columns=valid_dataset.column_names,
			num_proc=1,
			keep_in_memory=False,
			batch_size=16,
		).filter(lambda x: x is not None)
		test_dataset = test_dataset.map(
			lambda batch: prepare_dataset(batch, processor, logger),
			remove_columns=test_dataset.column_names,
			num_proc=1,
			keep_in_memory=False,
			batch_size=16,
		).filter(lambda x: x is not None)
		logger.info("Dataset mapping completed.")
		
		training_args = TrainingArguments(
			output_dir=config.output_dir,
			per_device_train_batch_size=config.per_device_train_batch_size,
			gradient_accumulation_steps=config.gradient_accumulation_steps,
			learning_rate=config.learning_rate,
			warmup_steps=config.warmup_steps,
			num_train_epochs=config.num_train_epochs,  # Use specified number of epochs
			save_total_limit=config.save_total_limit,
			gradient_checkpointing=True,
			fp16=config.fp16,
			eval_strategy="steps",
			optim="adamw_torch",
			per_device_eval_batch_size=config.per_device_train_batch_size,
			save_steps=config.save_steps,
			eval_steps=config.eval_steps,
			logging_steps=config.logging_steps,
			load_best_model_at_end=True,
			metric_for_best_model="wer",
			greater_is_better=False,
			push_to_hub=True,
			hub_model_id=config.hub_model_id,
			report_to=["wandb"],
		)
		
		trainer = Trainer(
			model=model,
			args=training_args,
			train_dataset=train_dataset,
			eval_dataset=valid_dataset,
			data_collator=DataCollatorCTCWithPadding(processor=processor),
			compute_metrics=compute_metrics,
			callbacks=[WandbCallback()],
		)
		
		logger.info("Starting training...")
		trainer.train()
		logger.info("Training completed.")
		
		test_results = trainer.evaluate(eval_dataset=test_dataset)
		wandb.log({"test_wer": test_results["eval_wer"]})
		logger.info(f"Test evaluation results: {test_results}")
		
		wer_history = [
			(log["step"], log["eval_wer"])
			for log in trainer.state.log_history
			if "eval_wer" in log
		]
		wer_history_path = os.path.join(config.output_dir, "wav2vec_base_central_vi_250h_wer_history.json")
		with open(wer_history_path, "w") as f:
			json.dump(wer_history, f)
		
		artifact = wandb.Artifact("wer_history", type="metrics")
		artifact.add_file(wer_history_path)
		wandb.log_artifact(artifact)
		
		os.makedirs(config.model_save_dir, exist_ok=True)
		trainer.save_model(config.model_save_dir)
		processor.save_pretrained(config.model_save_dir)
		
		trainer.push_to_hub(
			commit_message="Fine-tuned Wav2Vec2_250h on ViMD Central region with layer freezing",
			tags=["speech-recognition", "vietnamese", "central-vietnam"],
			dataset=config.dataset_id,
			language="vi",
			finetuned_from=config.model_id,
			tasks="automatic-speech-recognition",
		)
		processor.push_to_hub(config.hub_model_id)
		
		logger.info(f"Final evaluation results: {eval_results}")
		logger.info("Model and processor saved and pushed to Hugging Face Hub.")
		
		wandb.finish()
		
	except Exception as e:
		logger.error(f"Training failed: {str(e)}")
		wandb.finish()
		raise

if __name__ == "__main__":
	main()