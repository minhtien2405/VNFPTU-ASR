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
from datasets import load_dataset, Audio, Dataset
import evaluate
from dotenv import load_dotenv
from huggingface_hub import login
import numpy as np
import wandb
import re
import requests
import librosa
import tempfile
from pathlib import Path
from tqdm.auto import tqdm
import logging

os.environ["PYARROW_WITH_INT64"] = "1"

@dataclass
class TrainingConfig:
	model_id: str = "minhtien2405/wav2vec2-base-vi"
	hub_model_id: str = "minhtien2405/vovinam-wav2vec2-base-vi"
	dataset_id: str = "minhtien2405/VoviAIDataset"
	output_dir: str = "./logs/vovinam_wav2vec2-base-vi"
	cache_dir: str = "./cache"
	log_dir: str = "./logs"
	model_save_dir: str = "./models/vovinam_wav2vec2-base-vi"
	per_device_train_batch_size: int = 4
	gradient_accumulation_steps: int = 8
	learning_rate: float = 3e-4
	warmup_steps: int = 100
	num_train_epochs: int = 50
	save_steps: int = 100
	eval_steps: int = 100
	logging_steps: int = 50
	save_total_limit: int = 3
	fp16: bool = True
	project_name: str = "Vovinam_Wav2Vec2_All_VoviAI_FPTU"
	run_name: str = f"vovinam_wav2vec2_finetune_voviai_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def setup_logging(config: TrainingConfig) -> logging.Logger:
	os.makedirs(config.log_dir, exist_ok=True)
	logging.basicConfig(
		filename=os.path.join(config.log_dir, "vovinam_wav2vec_base_voviai_v0.log"),
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

def download_audio_from_s3(url: str, cache_dir: str = "./audio_cache") -> Optional[str]:
    """Download audio file from S3 URL and return local path, preserving original extension."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        
        # Use the filename directly from the URL, which is correct (e.g., ends with .aac)
        if '/' in url:
            filename = url.split('/')[-1]
        else:
            # Fallback for unusual URLs
            filename = f"{hash(url)}.audio" 

        local_path = os.path.join(cache_dir, filename)
        
        # Check if file already exists and is valid
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return local_path
            
        # Download the file
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        response = requests.get(url, stream=True, timeout=60, headers=headers)
        response.raise_for_status()
        
        # Write file with temporary name first, then rename (atomic operation)
        temp_path = local_path + '.tmp'
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
        
        os.rename(temp_path, local_path)
        
        # Verify file was downloaded correctly
        if os.path.getsize(local_path) == 0:
            os.remove(local_path)
            raise ValueError("Downloaded file is empty")
            
        return local_path
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error downloading audio from {url}: {str(e)}")
        # Clean up temp file if it exists
        temp_path = os.path.join(cache_dir, filename + '.tmp') if 'filename' in locals() else None
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return None

def load_audio_from_path(audio_path: str, target_sr: int = 16000) -> Optional[Dict]:
	"""Load audio from local path and return audio dict."""
	try:
		if not os.path.exists(audio_path):
			return None
			
		# Load audio using librosa
		audio_array, sr = librosa.load(audio_path, sr=target_sr)
		
		return {
			"array": audio_array,
			"sampling_rate": target_sr,
			"path": audio_path
		}
	except Exception as e:
		logging.getLogger(__name__).error(f"Error loading audio from {audio_path}: {str(e)}")
		return None

def validate_audio(sample, logger: logging.Logger) -> bool:
	"""Validate audio sample for non-empty and valid format."""
	try:
		# VoviAI Dataset: audio is processed and stored in 'audio' field
		audio_data = sample.get("audio")
		
		if audio_data is None:
			logger.warning(f"No audio data found in sample")
			return False
			
		if audio_data["array"] is None or len(audio_data["array"]) == 0:
			logger.warning(f"Invalid audio array in sample")
			return False
			
		# Additional validation: check audio duration (optional)
		duration = len(audio_data["array"]) / audio_data["sampling_rate"]
		if duration < 0.1 or duration > 30:  # 0.1s to 30s reasonable range
			logger.warning(f"Audio duration {duration:.2f}s is outside reasonable range")
			return False
			
		return True
	except Exception as e:
		logger.error(f"Error validating audio sample: {str(e)}")
		return False
	
def validate_text(sample, logger: logging.Logger) -> bool:
	"""Validate text labels for non-empty and reasonable length."""
	try:
		# VoviAI Dataset uses 'text' field
		text = sample.get("text", "")
		
		if not text or len(text) == 0:
			logger.warning(f"Empty text found in sample")
			return False
			
		if len(text) > 1000:  # Reasonable max length for Vietnamese ASR
			logger.warning(f"Text too long ({len(text)} chars): {text[:100]}...")
			return False
			
		# Additional validation: check for reasonable Vietnamese text
		if len(text.strip()) < 2:
			logger.warning(f"Text too short: '{text}'")
			return False
			
		return True
	except Exception as e:
		logger.error(f"Error validating text sample: {str(e)}")
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

def process_audio_sample(sample, logger: logging.Logger) -> Dict:
	"""Process a single audio sample, downloading from S3 if needed."""
	try:
		# VoviAI Dataset uses 'audioLink' field for S3 URLs
		if "audioLink" not in sample:
			logger.error(f"No audioLink field found in sample: {sample.keys()}")
			return None
		
		# Download from S3 URL
		audio_path = download_audio_from_s3(sample["audioLink"])
		if not audio_path:
			logger.error(f"Failed to download audio from: {sample['audioLink']}")
			return None
			
		audio_data = load_audio_from_path(audio_path)
		if audio_data is None:
			logger.error(f"Failed to load audio from path: {audio_path}")
			return None
			
		# Get text transcription (VoviAI Dataset uses 'text' field)
		text = sample.get("text", "")
		if not text:
			logger.warning(f"Empty text found for audio: {sample['audioLink']}")
			return None
		
		return {
			"audio": audio_data,
			"text": normalize_text(text),
			"path": audio_data.get("path", "unknown"),
			# Keep additional metadata for potential use
			"speakerId": sample.get("speakerId", ""),
			"environment": sample.get("environment", ""),
			"gender": sample.get("gender", ""),
			"province_name": sample.get("province_name", ""),
			"region": sample.get("region", "")
		}
	except Exception as e:
		logger.error(f"Error processing audio sample with audioLink {sample.get('audioLink', 'unknown')}: {str(e)}")
		return None

def load_and_prepare_data(config: TrainingConfig, logger: logging.Logger):
    """Load and preprocess dataset with validation and S3 audio download."""
    try:
        os.environ["HF_DATASETS_CACHE"] = config.cache_dir
        
        logger.info(f"Loading dataset: {config.dataset_id}")
        dataset = load_dataset(config.dataset_id, cache_dir=config.cache_dir)
        
        logger.info(f"Dataset structure: {dataset}")
        logger.info(f"Dataset columns: {dataset['train'].column_names if 'train' in dataset else list(dataset.keys())}")
        
        train_split = dataset["train"]
        valid_split = dataset["validation"] 
        test_split = dataset["test"]
        
        logger.info(f"Original train dataset size: {len(train_split)}")
        logger.info(f"Original validation dataset size: {len(valid_split)}")
        logger.info(f"Original test dataset size: {len(test_split)}")
        
        # I have re-enabled the tqdm progress bar for you.
        def process_dataset_split(split_data, split_name):
            logger.info(f"Processing {split_name} split...")
            
            processed_samples = []
            for sample in tqdm(split_data, desc=f"Processing {split_name} split"):
                processed_sample = process_audio_sample(sample, logger)
                if processed_sample and validate_audio(processed_sample, logger) and validate_text(processed_sample, logger):
                    processed_samples.append(processed_sample)
                else:
                    # The original logging is good, let's keep it but reduce frequency if it's too noisy
                    # For now, it's useful for debugging.
                    logger.warning(f"Skipping invalid sample in {split_name} split: {sample['audioLink']}")

            if not processed_samples:
                raise ValueError(f"No valid samples found in {split_name} split after processing. Check download or audio loading logs.")
            
            processed_dataset = Dataset.from_list(processed_samples)
            processed_dataset = processed_dataset.cast_column("audio", Audio(sampling_rate=16000))
            
            logger.info(f"{split_name.capitalize()} dataset size after processing: {len(processed_dataset)}")
            return processed_dataset
        
        logger.info("Starting dataset processing...")
        train_dataset = process_dataset_split(train_split, "train")
        valid_dataset = process_dataset_split(valid_split, "validation")
        test_dataset = process_dataset_split(test_split, "test")
        
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

		logger.info("Starting dataset loading and preparation...")
		train_dataset, valid_dataset, test_dataset = load_and_prepare_data(config, logger)
		logger.info("Dataset loading and preparation completed.")
		
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
		trainer.train()#resume_from_checkpoint=True)
		logger.info("Training completed.")
		
		test_results = trainer.evaluate(eval_dataset=test_dataset)
		wandb.log({"test_wer": test_results["eval_wer"]})
		logger.info(f"Test evaluation results: {test_results}")
		
		wer_history = [
			(log["step"], log["eval_wer"])
			for log in trainer.state.log_history
			if "eval_wer" in log
		]
		wer_history_path = os.path.join(config.output_dir, "vovinam_wav2vec2_base_vi_250hfinetuned_wer_history.json")
		with open(wer_history_path, "w") as f:
			json.dump(wer_history, f)
		
		artifact = wandb.Artifact("wer_history", type="metrics")
		artifact.add_file(wer_history_path)
		wandb.log_artifact(artifact)
		
		os.makedirs(config.model_save_dir, exist_ok=True)
		trainer.save_model(config.model_save_dir)
		processor.save_pretrained(config.model_save_dir)
		
		trainer.push_to_hub(
			commit_message="Fine-tuned Wav2Vec2_250h on VoviAI Dataset with layer freezing",
			tags=["speech-recognition", "vietnamese", "vietnam", "voviai", "vovinam"],
			dataset=config.dataset_id,
			language="vi",
			finetuned_from=config.model_id,
			tasks="automatic-speech-recognition",
		)
		processor.push_to_hub(config.hub_model_id)
		
		logger.info("Model and processor saved and pushed to Hugging Face Hub.")
		
		wandb.finish()
		
	except Exception as e:
		logger.error(f"Training failed: {str(e)}")
		wandb.finish()
		raise

if __name__ == "__main__":
	main()