import os
import json
import logging
import re
import time
import requests
import librosa
import peft
import torch
import wandb
import evaluate
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Union, Optional
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
)

# =================================================================================
# 1. Configuration
# =================================================================================
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
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    num_train_epochs: int = 50
    
    # Evaluation and Saving
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    logging_steps: int = 25
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    
    # Technical Configs
    fp16: bool = True
    optim: str = "adamw_bnb_8bit"
    gradient_checkpointing: bool = True
    
    # Project Tracking
    project_name: str = "Vovinam_PhoWhisper_Large_Finetune"
    run_name: str = f"vovinam_phowhisper_finetune_{datetime.now().strftime('%Y%m%d-%H%M')}"

# =================================================================================
# 2. Data Collator
# =================================================================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        valid_features = [f for f in features if "input_features" in f and "labels" in f]
        if not valid_features:
            logging.warning("No valid features found in batch, returning empty dict.")
            return {}

        input_features = [{"input_features": feature["input_features"]} for feature in valid_features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        label_features = [{"input_ids": feature["labels"]} for feature in valid_features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
            
        batch["labels"] = labels
        return batch

# =================================================================================
# 3. Helper Functions
# =================================================================================
def setup_logging(config: TrainingConfig) -> logging.Logger:
    """Cấu hình logging cho project."""
    os.makedirs(config.log_dir, exist_ok=True)
    log_filename = f"vovinam_phowhisper_training_{datetime.now().strftime('%Y%m%d')}.log"
    logging.basicConfig(
        filename=os.path.join(config.log_dir, log_filename),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    return logging.getLogger(__name__)

def setup_environment(config: TrainingConfig, logger: logging.Logger):
    """Thiết lập môi trường, đăng nhập và khởi tạo WandB."""
    load_dotenv("./configs/.env")
    
    try:
        login(token=os.getenv("HF_TOKEN"))
        logger.info("Đăng nhập Hugging Face Hub thành công.")
    except Exception as e:
        logger.error(f"Lỗi đăng nhập Hugging Face: {e}")

    try:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(project=config.project_name, name=config.run_name, config=asdict(config))
        logger.info(f"Đăng nhập và khởi tạo WandB run '{config.run_name}' thành công.")
    except Exception as e:
        logger.error(f"Không thể đăng nhập hoặc khởi tạo W&B: {e}")

    os.makedirs(config.cache_dir, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = config.cache_dir
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
    logger.info("Thiết lập môi trường và cache hoàn tất.")

def download_audio_from_s3(url: str, cache_dir: str, logger: logging.Logger, max_retries: int = 3, base_delay: float = 1.0) -> Optional[str]:
    """Tải file âm thanh từ URL với cơ chế thử lại."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        if not url:
            return None
            
        filename = url.split('/')[-1]
        local_path = os.path.join(cache_dir, filename)
        
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return local_path

        headers = {'User-Agent': 'Mozilla/5.0'}
        for attempt in range(max_retries):
            try:
                with requests.get(url, stream=True, timeout=30, headers=headers) as r:
                    r.raise_for_status()
                    with open(local_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                if os.path.getsize(local_path) > 0:
                    return local_path
                else: # File tải về rỗng
                    os.remove(local_path)
                    logger.warning(f"Tải về file rỗng từ URL: {url}")
                    return None
            except requests.exceptions.RequestException as e:
                logger.warning(f"Lần thử {attempt + 1}/{max_retries} thất bại khi tải {url}. Lỗi: {e}")
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                else:
                    logger.error(f"Không thể tải file từ {url} sau {max_retries} lần thử.")
                    return None
    except Exception as e:
        logger.error(f"Lỗi không mong muốn khi tải {url}: {e}")
        return None
    return None

def normalize_text(text: str) -> str:
    """Chuẩn hóa và làm sạch text."""
    try:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception:
        return text

def load_and_prepare_data(config: TrainingConfig, logger: logging.Logger):
    """Tải và chuẩn bị dữ liệu, tạo cột audio_path."""
    logger.info(f"Đang tải dataset: {config.dataset_id}")
    dataset = load_dataset(config.dataset_id, cache_dir=config.cache_dir)
    audio_cache_dir = os.path.join(config.cache_dir, "audio_vovinam")

    def download_and_validate(sample):
        sample["is_valid"] = False
        audio_path = download_audio_from_s3(sample["audioLink"], audio_cache_dir, logger)
        if not audio_path:
            return sample
        
        try:
            duration = librosa.get_duration(path=audio_path)
            if not (0.1 < duration < 30.0):
                logger.warning(f"Thời lượng không hợp lệ ({duration:.2f}s), bỏ qua: {audio_path}")
                return sample
        except Exception as e:
            logger.warning(f"Không thể lấy thời lượng, bỏ qua: {audio_path}, Lỗi: {e}")
            return sample

        text = normalize_text(sample.get("text", ""))
        if len(text.strip()) < 2:
            logger.warning(f"Text không hợp lệ ('{text}'), bỏ qua: {audio_path}")
            return sample

        sample["audio_path"] = audio_path
        sample["text"] = text
        sample["is_valid"] = True
        return sample

    for split in dataset.keys():
        logger.info(f"Đang xử lý split '{split}'...")
        processed_split = dataset[split].map(download_and_validate, num_proc=4)
        filtered_split = processed_split.filter(lambda s: s["is_valid"], num_proc=4)
        
        logger.info(f"Split '{split}' đã lọc: {len(dataset[split])} -> {len(filtered_split)} mẫu.")
        columns_to_remove = [col for col in ["audio", "audioLink", "is_valid", "__index_level_0__"] if col in filtered_split.column_names]
        dataset[split] = filtered_split.remove_columns(columns_to_remove)

    logger.info(f"Dataset đã xử lý xong: {dataset}")
    return dataset["train"], dataset["validation"], dataset["test"]

def setup_model_and_processor(config: TrainingConfig, logger: logging.Logger):
    """Tải processor và model, áp dụng quantization và PEFT."""
    logger.info(f"Đang tải processor từ: {config.model_id}")
    processor = WhisperProcessor.from_pretrained(config.model_id, language="vi", task="transcribe")

    logger.info(f"Đang tải model từ: {config.model_id} với 4-bit quantization.")
    model = WhisperForConditionalGeneration.from_pretrained(
        config.model_id,
        use_cache=False,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        ),
    )
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")
    model.config.suppress_tokens = []

    logger.info("Áp dụng PEFT/LoRA cho model.")
    model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.gradient_checkpointing)
    lora_config = peft.LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    model = peft.get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, processor

# =================================================================================
# 4. Main Execution
# =================================================================================
def main():
    config = TrainingConfig()
    logger = setup_logging(config)
    
    try:
        setup_environment(config, logger)
        
        model, processor = setup_model_and_processor(config, logger)
        train_dataset, valid_dataset, test_dataset = load_and_prepare_data(config, logger)

        def prepare_for_training(batch):
            try:
                audio_array, _ = librosa.load(batch["audio_path"], sr=16000)
                batch["input_features"] = processor(audio_array, sampling_rate=16000).input_features[0]
                batch["labels"] = processor.tokenizer(batch["text"]).input_ids
                return batch
            except Exception as e:
                logger.error(f"Lỗi khi xử lý file {batch.get('audio_path', 'UNKNOWN')}: {e}")
                return None

        logger.info("Bắt đầu map dataset cho training...")
        train_dataset = train_dataset.map(prepare_for_training).filter(lambda x: x is not None)
        valid_dataset = valid_dataset.map(prepare_for_training).filter(lambda x: x is not None)
        test_dataset = test_dataset.map(prepare_for_training).filter(lambda x: x is not None)
        logger.info("Hoàn tất map dataset.")

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
        metric = evaluate.load("wer")
        def compute_metrics(pred):
            pred_ids = pred.predictions
            label_ids = pred.label_ids
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
            return {"wer": 100 * metric.compute(predictions=pred_str, references=label_str)}

        training_args = Seq2SeqTrainingArguments(
            output_dir=config.output_dir,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            num_train_epochs=config.num_train_epochs,
            eval_strategy=config.eval_strategy,
            eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            save_total_limit=config.save_total_limit,
            logging_steps=config.logging_steps,
            load_best_model_at_end=config.load_best_model_at_end,
            metric_for_best_model=config.metric_for_best_model,
            greater_is_better=config.greater_is_better,
            fp16=config.fp16,
            optim=config.optim,
            gradient_checkpointing=config.gradient_checkpointing,
            hub_model_id=config.hub_model_id,
            run_name=config.run_name,
            remove_unused_columns=False,
            push_to_hub=True,
            report_to="wandb",
            gradient_checkpointing_kwargs={"use_reentrant": False} if config.gradient_checkpointing else None,
        )

        trainer = Seq2SeqTrainer(
            model=model, args=training_args, train_dataset=train_dataset,
            eval_dataset=valid_dataset, data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        logger.info("Bắt đầu quá trình training...")
        trainer.train(resume_from_checkpoint=True)
        logger.info("Quá trình training hoàn tất.")

        logger.info("Bắt đầu đánh giá trên tập test...")
        test_results = trainer.evaluate(eval_dataset=test_dataset)
        logger.info(f"Kết quả WER trên tập test: {test_results['eval_wer']}")
        wandb.log({"test_wer": test_results["eval_wer"]})

        os.makedirs(config.model_save_dir, exist_ok=True)
        trainer.save_model(config.model_save_dir)
        processor.save_pretrained(config.model_save_dir)
        logger.info(f"Model đã được lưu vào {config.model_save_dir}")

        logger.info("Đẩy model lên Hugging Face Hub...")
        trainer.push_to_hub(
			commit_message="Fine-tuned PhoWhisper-large on VoviAI Dataset",
            tag=["phowhisper","vietnamese", "vietnam", "voviai", "vovinam"],
			defataset=config.dataset_id,
			language="vi",
			finetuned_from=config.model_id,
			tasks="automatic-speech-recognition",
		)
        processor.push_to_hub(config.hub_model_id)
        logger.info("Hoàn tất đẩy model lên Hub.")

    except Exception as e:
        logger.error(f"Training thất bại: {e}", exc_info=True)
    finally:
        if wandb.run:
            wandb.finish()
            logger.info("WandB run đã kết thúc.")

if __name__ == "__main__":
    main()
