import os
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Union, Any
from datetime import datetime
import torch
from transformers import (
	WhisperProcessor,
	WhisperForConditionalGeneration,
	Seq2SeqTrainingArguments,
	Seq2SeqTrainer,
	BitsAndBytesConfig,
)
from datasets import load_dataset
import evaluate
from dotenv import load_dotenv
from huggingface_hub import login
import numpy as np
import wandb
import librosa
import peft
from typing import Optional
from transformers import (
	TrainerCallback,
)
import re
import requests


# =================================================================================
# Configuration
# =================================================================================
os.environ["PYARROW_WITH_INT64"] = "1"

@dataclass
class TrainingConfig:
	# Model and Hub IDs
	model_id: str = "minhtien2405/phowhisper-large-all-vi"
	hub_model_id: str = "minhtien2405/vovinam-phowhisper-large-vi"
	
	# Dataset
	dataset_id: str = "minhtien2405/VoviAIDataset"
	
	# Directories
	output_dir: str = "./logs/vovinam_phowhisper-large-vi"
	cache_dir: str = "./cache"
	log_dir: str = "./logs"
	model_save_dir: str = "./models/vovinam_phowhisper-large-vi"
	
	# Training Hyperparameters
	per_device_train_batch_size: int = 4
	gradient_accumulation_steps: int = 8
	learning_rate: float = 1e-5
	warmup_steps: int = 100
	num_train_epochs: int = 3
	save_steps: int = 100
	eval_steps: int = 100
	logging_steps: int = 25
	save_total_limit: int = 3
	
	# Technical Configs
	fp16: bool = True
	gradient_checkpointing: bool = True
	
	# Project Tracking
	project_name: str = "Vovinam_PhoWhisper_Large_Finetune"
	run_name: str = f"vovinam_phowhisper_finetune_voviai_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# =================================================================================
# Logging Setup (specific to this script)
# =================================================================================

def setup_logging(config: TrainingConfig) -> logging.Logger:
	os.makedirs(config.log_dir, exist_ok=True)
	logging.basicConfig(
		filename=os.path.join(config.log_dir, "vovinam_phowhisper_large_voviai_v0.log"),
		level=logging.INFO,
		format="%(asctime)s - %(levelname)s - %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
		force=True,
	)
	return logging.getLogger(__name__)

# =================================================================================
# Data Loading and Preparation
# =================================================================================

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


def load_and_prepare_data(config: TrainingConfig, logger: logging.Logger):
    """
    Load and preprocess dataset efficiently, preparing audio_path for explicit loading later.
    """
    try:
        os.environ["HF_DATASETS_CACHE"] = config.cache_dir
        logger.info(f"Loading dataset from Hugging Face Hub: {config.dataset_id}")
        dataset = load_dataset(config.dataset_id, cache_dir=config.cache_dir)
        logger.info(f"Initial dataset structure: {dataset}")

        def download_and_prepare_sample(sample):
            sample["is_valid"] = True
            audio_path = download_audio_from_s3(sample["audioLink"], cache_dir=os.path.join(config.cache_dir, "audio"))
            if not audio_path:
                logger.warning(f"Failed to download audio, skipping. URL: {sample['audioLink']}")
                sample["is_valid"] = False
                return sample
            sample["audio_path"] = audio_path

            try:
                duration = librosa.get_duration(path=audio_path)
                if duration < 0.1 or duration > 30:
                    logger.warning(f"Invalid duration ({duration:.2f}s), skipping. Path: {audio_path}")
                    sample["is_valid"] = False
            except Exception as e:
                logger.warning(f"Could not get duration, skipping. Path: {audio_path}, Error: {e}")
                sample["is_valid"] = False
            if not sample["is_valid"]: return sample

            text = sample.get("text", "")
            normalized_text = normalize_text(text)
            if not normalized_text or len(normalized_text.strip()) < 2:
                logger.warning(f"Invalid or empty text ('{text}'), skipping. Path: {audio_path}")
                sample["is_valid"] = False
            sample["text"] = normalized_text
            return sample

        for split in dataset.keys():
            logger.info(f"Processing '{split}' split...")
            processed_split = dataset[split].map(
                download_and_prepare_sample, num_proc=4, desc=f"Downloading and validating {split} data"
            )
            original_size = len(processed_split)
            filtered_split = processed_split.filter(
                lambda sample: sample["is_valid"], num_proc=4, desc=f"Filtering invalid samples in {split} split"
            )
            filtered_size = len(filtered_split)
            logger.info(f"Filtered {split} split: {original_size} -> {filtered_size} samples.")
            if filtered_size == 0 and original_size > 0:
                raise ValueError(f"No valid samples remained in '{split}' split after filtering. Check logs for warnings.")

            # THAY ĐỔI QUAN TRỌNG: BỎ cast_column và xóa các cột không cần thiết
            # Chúng ta sẽ giữ lại 'audio_path' để dùng ở bước sau.
            columns_to_remove = [col for col in ["audio", "audioLink", "is_valid", "__index_level_0__"] if col in filtered_split.column_names]
            if columns_to_remove:
                filtered_split = filtered_split.remove_columns(columns_to_remove)
            
            dataset[split] = filtered_split

        logger.info(f"Final processed dataset: {dataset}")
        return dataset["train"], dataset["validation"], dataset["test"]
    except Exception as e:
        logger.error(f"Error during data preparation: {str(e)}")
        raise

def prepare_dataset_for_whisper(batch, processor: WhisperProcessor, logger: logging.Logger):
	try:
		audio_array, sampling_rate = librosa.load(batch["audio_path"], sr=16000)

		batch["input_features"] = processor(audio_array, sampling_rate=sampling_rate).input_values[0]
		batch["labels"] = processor.tokenizer(batch["text"]).input_ids
		return batch
	except Exception as e:
		logger.error(f"Error processing file {batch.get('audio_path', 'UNKNOWN')}: {str(e)}")
		return None

# =================================================================================
# Core Training Components
# =================================================================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
	processor: Any
	def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
		input_features = [{"input_features": feature["input_features"]} for feature in features]
		batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
		
		label_features = [{"input_ids": feature["labels"]} for feature in features]
		labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
		
		labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
		
		if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
			labels = labels[:, 1:]
			
		batch["labels"] = labels
		return batch

def setup_training_components(config: TrainingConfig, logger: logging.Logger):
	"""Thiết lập model, processor, và metrics."""
	try:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		logger.info(f"Sử dụng device: {device}")
		
		processor = WhisperProcessor.from_pretrained(config.model_id, language="vi", task="transcribe")
		
		quantization_config = BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_compute_dtype=torch.float16,
			bnb_4bit_use_double_quant=True,
			bnb_4bit_quant_type="nf4",
		)
		
		model = WhisperForConditionalGeneration.from_pretrained(
			config.model_id,
			use_cache=False,
			device_map="auto",
			quantization_config=quantization_config,
		)
		model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")
		model.config.suppress_tokens = []
		
		model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.gradient_checkpointing)
		lora_config = peft.LoraConfig(
			r=4,
			lora_alpha=8,
			target_modules=["q_proj", "v_proj"],
			lora_dropout=0.05,
			bias="none",
		)
		model = peft.get_peft_model(model, lora_config)
		model.print_trainable_parameters()

		metric = evaluate.load("wer")
		
		def compute_metrics(pred):
			pred_ids = pred.predictions
			label_ids = pred.label_ids
			label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
			
			pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
			label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
			
			wer = 100 * metric.compute(predictions=pred_str, references=label_str)
			return {"wer": wer}
			
		return processor, model, compute_metrics
	except Exception as e:
		logger.error(f"Lỗi khi thiết lập các thành phần training: {str(e)}")
		raise

# =================================================================================
# Main Training Function
# =================================================================================

def main():
	config = TrainingConfig()
	logger = setup_logging(config)
	
	try:
		load_dotenv("./configs/.env")
		
		hf_token = os.getenv("HF_TOKEN")
		if not hf_token: raise ValueError("HF_TOKEN không tìm thấy trong file .env")
		login(token=hf_token)
		logger.info("Đăng nhập Hugging Face Hub thành công")
		
		wandb_api_key = os.getenv("WANDB_API_KEY")
		if not wandb_api_key: raise ValueError("WANDB_API_KEY không tìm thấy trong file .env")
		wandb.login(key=wandb_api_key)
		logger.info("Đăng nhập Weights & Biases thành công")
		
		wandb.init(project=config.project_name, name=config.run_name, config=vars(config))
		
		os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
		torch.cuda.empty_cache()

		logger.info("Bắt đầu tải và chuẩn bị dữ liệu...")
		train_dataset, valid_dataset, test_dataset = load_and_prepare_data(config, logger)
		logger.info("Hoàn tất tải và chuẩn bị dữ liệu.")
		
		processor, model, compute_metrics = setup_training_components(config, logger)
		
		data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
		
		logger.info("Bắt đầu map dataset cho Whisper...")
		map_fn = lambda batch: prepare_dataset_for_whisper(batch, processor, logger)
		
		train_dataset = train_dataset.map(map_fn, num_proc=1, batch_size=16).filter(lambda x: x is not None)
		valid_dataset = valid_dataset.map(map_fn, num_proc=1, batch_size=16).filter(lambda x: x is not None)
		test_dataset = test_dataset.map(map_fn, num_proc=1, batch_size=16).filter(lambda x: x is not None)
		logger.info("Hoàn tất map dataset.")
		
		training_args = Seq2SeqTrainingArguments(
			output_dir=config.output_dir,
			per_device_train_batch_size=config.per_device_train_batch_size,
			gradient_accumulation_steps=config.gradient_accumulation_steps,
			learning_rate=config.learning_rate,
			warmup_steps=config.warmup_steps,
			num_train_epochs=config.num_train_epochs,
			save_total_limit=config.save_total_limit,
			gradient_checkpointing=config.gradient_checkpointing,
			fp16=config.fp16,
			eval_strategy="steps",
			optim="adamw_bnb_8bit",
			per_device_eval_batch_size=config.per_device_train_batch_size,
			save_steps=config.save_steps,
			eval_steps=config.eval_steps,
			logging_steps=config.logging_steps,
			load_best_model_at_end=True,
			metric_for_best_model="wer",
			greater_is_better=False,
			push_to_hub=True,
			remove_unused_columns=False,
			hub_model_id=config.hub_model_id,
			report_to=["wandb"],
		)
		
		trainer = Seq2SeqTrainer(
			model=model,
			args=training_args,
			train_dataset=train_dataset,
			eval_dataset=valid_dataset,
			data_collator=data_collator,
			compute_metrics=compute_metrics,
			callbacks=[WandbCallback()],
		)
		
		logger.info("Bắt đầu training...")
		trainer.train()
		logger.info("Training hoàn tất.")
		
		logger.info("Đánh giá trên tập test...")
		test_results = trainer.evaluate(eval_dataset=test_dataset)
		wandb.log({"test_wer": test_results["eval_wer"]})
		logger.info(f"Kết quả đánh giá trên tập test: {test_results}")
		
		wer_history_path = os.path.join(config.output_dir, "phowhisper_large_vovinam_finetuned_wer_history.json")
		with open(wer_history_path, "w") as f:
			json.dump(trainer.state.log_history, f)
		
		os.makedirs(config.model_save_dir, exist_ok=True)
		trainer.save_model(config.model_save_dir)
		processor.save_pretrained(config.model_save_dir)
		
		trainer.push_to_hub(
			commit_message="Fine-tuned PhoWhisper-large on VoviAI Dataset",
			tags=["speech-recognition", "vietnamese", "vovinam", "phowhisper"],
			dataset=config.dataset_id,
			language="vi",
			finetuned_from=config.model_id,
			tasks="automatic-speech-recognition",
		)
		processor.push_to_hub(config.hub_model_id, commit_message="Update processor for PhoWhisper-large VoviAI finetune")
		
		logger.info("Model và processor đã được lưu và đẩy lên Hugging Face Hub.")
		
		wandb.finish()
		
	except Exception as e:
		logger.error(f"Training thất bại: {str(e)}", exc_info=True)
		if wandb.run:
			wandb.finish()
		raise

if __name__ == "__main__":
	main()