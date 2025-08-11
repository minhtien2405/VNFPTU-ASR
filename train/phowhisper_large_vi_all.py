import os
import torch
import json
import logging
import peft
import accelerate
import wandb
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Union
from datetime import datetime
from datasets import load_dataset, Audio
import evaluate
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
)

# --- 1. Cấu hình Training ---
@dataclass
class TrainingConfig:
    # Model and Hub
    model_id: str = "vinai/PhoWhisper-large"
    hub_model_id: str = "minhtien2405/phowhisper-large-all-vi"
    # Dataset
    dataset_id: str = "nguyendv02/ViMD_Dataset"
    # Directories
    output_dir: str = "./logs/phowhisper-large-all-vi"
    cache_dir: str = "./cache"
    log_dir: str = "./logs"
    model_save_dir: str = "./models/phowhisper-large-all-vi"
    # Training Hyperparameters
    per_device_train_batch_size: int = 4
    # per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    warmup_steps: int = 400
    num_train_epochs: int = 30
    # Evaluation and Saving
    eval_strategy: str = "no"
    # eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 3
    logging_steps: int = 100
    load_best_model_at_end: bool = True
    # metric_for_best_model: str = "loss"
    greater_is_better: bool = False
    # Technical
    fp16: bool = True
    optim: str = "adamw_bnb_8bit"
    gradient_checkpointing: bool = True
    # WandB
    project_name: str = "PhoWhisper_ViMD_FPTU"
    run_name: str = f"phowhisper-large-vi-all-{datetime.now().strftime('%Y%m%d-%H%M')}"

# --- 2. Data Collator ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# --- 3. Helper Functions ---
def setup_logging(config: TrainingConfig) -> logging.Logger:
    """Cấu hình logging cho project."""
    os.makedirs(config.log_dir, exist_ok=True)
    log_filename = f"phowhisper_training_{datetime.now().strftime('%Y%m%d')}.log"
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
    
    # Hugging Face Login
    try:
        login(token=os.getenv("HF_TOKEN"))
        logger.info("Đăng nhập Hugging Face Hub thành công.")
    except Exception as e:
        logger.error(f"Lỗi đăng nhập Hugging Face: {e}")

    # WandB Login and Init
    try:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(project=config.project_name, name=config.run_name, job_type="training")
        logger.info(f"Đăng nhập và khởi tạo WandB run '{config.run_name}' thành công.")
    except Exception as e:
        logger.error(f"Không thể đăng nhập hoặc khởi tạo W&B: {e}")

    # Cache and CUDA config
    os.makedirs(config.cache_dir, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = config.cache_dir
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
    logger.info("Thiết lập môi trường và cache hoàn tất.")

def load_and_prepare_data(config: TrainingConfig, processor: WhisperProcessor, logger: logging.Logger):
    """Tải, lọc và chuẩn bị dữ liệu cho training."""
    logger.info(f"Đang tải dataset: {config.dataset_id}")
    dataset = load_dataset(config.dataset_id, cache_dir=config.cache_dir)

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    def filter_long_audio(example):
        audio_length = example["audio"]["array"].shape[0] / 16000
        if audio_length > 30:
            logger.warning(f"Bỏ qua audio dài: {example['audio']['path']} ({audio_length:.2f}s)")
            return False
        return True

    train_dataset = dataset["train"].filter(filter_long_audio, num_proc=3)
    valid_dataset = dataset["valid"].filter(filter_long_audio, num_proc=3)
    logger.info(f"Kích thước tập train sau khi lọc: {len(train_dataset)}")
    logger.info(f"Kích thước tập validation sau khi lọc: {len(valid_dataset)}")

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(audio["array"], sampling_rate=16000).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
    valid_dataset = valid_dataset.map(prepare_dataset, remove_columns=valid_dataset.column_names)
    logger.info("Hoàn tất xử lý và chuẩn bị dataset.")
    return train_dataset, valid_dataset

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
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")
    model.config.suppress_tokens = []

    if torch.cuda.device_count() > 1:
        logger.info(f"Phân phối model trên {torch.cuda.device_count()} GPUs.")
        DEV_MAP = model.hf_device_map.copy()
        DEV_MAP["model.decoder.embed_tokens"] = DEV_MAP["model.decoder.embed_positions"] = DEV_MAP["proj_out"] = model._hf_hook.execution_device
        accelerate.dispatch_model(model, device_map=DEV_MAP)
        setattr(model, "model_parallel", True)
        setattr(model, "is_parallelizable", True)

    logger.info("Áp dụng PEFT/LoRA cho model.")
    peft_model = peft.get_peft_model(
        peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing =config.gradient_checkpointing),
        peft.LoraConfig(
            r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none"
        ),
    )
    peft_model.model.model.encoder.conv1.register_forward_hook(lambda module, input, output: output.requires_grad_(True))
    peft_model.print_trainable_parameters()
    
    return peft_model, processor

# --- 4. Main Execution ---
def main():
    config = TrainingConfig()
    logger = setup_logging(config)
    
    try:
        setup_environment(config, logger)
        
        peft_model, processor = setup_model_and_processor(config, logger)
        train_dataset, valid_dataset = load_and_prepare_data(config, processor, logger)

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            decoder_start_token_id=peft_model.config.decoder_start_token_id,
        )

        metric = evaluate.load("wer")
        def compute_metrics(pred):
            pred_ids = pred.predictions
            label_ids = pred.label_ids
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            wer = 100 * metric.compute(predictions=pred_str, references=label_str)
            return {"wer": wer}

        training_args = Seq2SeqTrainingArguments(
            output_dir=config.output_dir,
            per_device_train_batch_size=config.per_device_train_batch_size,
            # per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            num_train_epochs=config.num_train_epochs,
            eval_strategy=config.eval_strategy,
            # eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            save_total_limit=config.save_total_limit,
            logging_steps=config.logging_steps,
            # load_best_model_at_end=config.load_best_model_at_end,
            # metric_for_best_model=config.metric_for_best_model,
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
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        wandb.config.update(training_args.to_dict())
        logger.info("Bắt đầu quá trình training...")
        trainer.train() #resume_from_checkpoint=True)
        logger.info("Quá trình training hoàn tất.")

        logger.info("Bắt đầu đánh giá cuối cùng trên tập validation.")
        eval_results = trainer.evaluate()
        logger.info(f"Kết quả WER cuối cùng: {eval_results['eval_wer']}")
        wandb.log({"final_eval_wer": eval_results["eval_wer"]})

        results_path = os.path.join(config.output_dir, "eval_results.json")
        with open(results_path, "w") as f:
            json.dump(eval_results, f, indent=4)
        logger.info(f"Kết quả đánh giá đã được lưu tại {results_path}")

        logger.info(f"Lưu model và processor vào {config.model_save_dir}")
        os.makedirs(config.model_save_dir, exist_ok=True)
        trainer.save_model(config.model_save_dir)
        processor.save_pretrained(config.model_save_dir)

        logger.info("Lưu model như một artifact trên WandB.")
        model_artifact = wandb.Artifact(
            name=f"{config.hub_model_id.split('/')[-1]}-{wandb.run.id}",
            type="model",
            description="Fine-tuned PhoWhisper-large model on ViMD All region.",
            metadata=training_args.to_dict()
        )
        model_artifact.add_dir(config.model_save_dir)
        wandb.log_artifact(model_artifact)

        logger.info(f"Đẩy model lên Hugging Face Hub: {config.hub_model_id}")
        trainer.push_to_hub(
            commit_message="Fine-tuned PhoWhisper large on ViMD All region",
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
        wandb.finish()
        logger.info("WandB run đã kết thúc.")

if __name__ == "__main__":
    main()
