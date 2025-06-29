import os
import json
import torch
import logging
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import (
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback
)
import evaluate
import wandb
import accelerate
import peft
from thop import profile

logger = logging.getLogger(__name__)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": chunk_features}
            for feature in features
            for chunk_features in feature["input_features"]
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Flatten all label lists (always a list of tensors now)
        label_features = [
            {"input_ids": label}
            for feature in features
            for label in feature["labels"]
        ]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
    
class WandbCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_wer" in metrics:
            wandb.log({"eval_wer": metrics["eval_wer"], "step": state.global_step})
            logger.info(f"Logged eval_wer: {metrics['eval_wer']}")

class Trainer:
    def __init__(self, config, processor, train_dataset, valid_dataset):
        self.config = config
        self.processor = processor
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        wandb.init(project=config.wandb.project, name=config.wandb.run_name, config=config.config)
        self.model = self._load_model()
        self._log_model_stats()
        self.trainer = self._setup_trainer()

    def _load_model(self):
        model = WhisperForConditionalGeneration.from_pretrained(
            self.config.model.model_id,
            use_cache=False,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=self.config.model.quantization.load_in_4bit,
                bnb_4bit_compute_dtype=getattr(torch, self.config.model.quantization.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=self.config.model.quantization.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self.config.model.quantization.bnb_4bit_quant_type,
            ),
        )

        model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.config.model.language, task=self.config.model.task
        )
        model.config.suppress_tokens = []

        if torch.cuda.device_count() > 1:
            device_map = model.hf_device_map.copy()
            device_map.update({
                "model.decoder.embed_tokens": model._hf_hook.execution_device,
                "model.decoder.embed_positions": model._hf_hook.execution_device,
                "proj_out": model._hf_hook.execution_device
            })
            accelerate.dispatch_model(model, device_map=device_map)
            model.model_parallel = True
            model.is_parallelizable = True
            logger.info("Multi-GPU setup configured")

        peft_model = peft.get_peft_model(
            peft.prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            ),
            peft.LoraConfig(
                r=self.config.lora.r,
                lora_alpha=self.config.lora.lora_alpha,
                target_modules=self.config.lora.target_modules,
                lora_dropout=self.config.lora.lora_dropout,
                bias=self.config.lora.bias,
            ),
        )

        peft_model.model.model.encoder.conv1.register_forward_hook(
            lambda module, input, output: output.requires_grad_(True)
        )

        logger.info(f"Trainable parameters: {peft_model.print_trainable_parameters()}")
        return peft_model

    def _log_model_stats(self):
        """Log model statistics to wandb"""
        try:
            # Get parameter counts
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            # Log basic stats without FLOPs calculation
            stats = {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "trainable_percent": (trainable_params / total_params) * 100,
            }
            
            wandb.log(stats)
            logger.info(
                f"Model stats - Total: {total_params:,}, "
                f"Trainable: {trainable_params:,} ({stats['trainable_percent']:.2f}%)"
            )

            # Skip FLOPs calculation for now as it's causing issues
            
        except Exception as e:
            logger.warning(f"Failed to calculate model stats: {str(e)}")
            logger.warning("Continuing training without model stats")

    def _setup_trainer(self):
        # Format output directory with region
        output_dir = self.config.training.output_dir.format(region=self.config.region.lower())
        eval_output_dir = self.config.training.eval_output_dir.format(region=self.config.region.lower())
        hub_model_id = self.config.training.hub_model_id.format(region=self.config.region.lower())

        # Ensure directories exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(eval_output_dir, exist_ok=True)

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=int(self.config.training.per_device_train_batch_size),
            gradient_accumulation_steps=int(self.config.training.gradient_accumulation_steps),
            learning_rate=float(self.config.training.learning_rate),
            warmup_steps=int(self.config.training.warmup_steps),
            gradient_checkpointing=True,
            fp16=bool(self.config.training.fp16),
            optim=self.config.training.optim,
            eval_strategy=self.config.training.eval_strategy,
            eval_steps=int(self.config.training.eval_steps),
            save_steps=int(self.config.training.save_steps),
            save_total_limit=int(self.config.training.save_total_limit),
            per_device_eval_batch_size=int(self.config.training.per_device_eval_batch_size),
            logging_steps=int(self.config.training.logging_steps),
            metric_for_best_model=self.config.training.metric_for_best_model,
            greater_is_better=bool(self.config.training.greater_is_better),
            load_best_model_at_end=True,
            save_strategy="steps",
            push_to_hub=True,
            remove_unused_columns=False,
            hub_model_id=hub_model_id,
        )

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )

        metric = evaluate.load("wer")

        def compute_metrics(pred):
            pred_ids = pred.predictions
            label_ids = pred.label_ids
            
            label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

            pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

            try:
                wer_score = float(metric.compute(predictions=pred_str, references=label_str))
                wer = 100 * wer_score
            except Exception as e:
                logger.error(f"WER computation failed: {e}")
                wer = float('inf')
                
            return {"wer": wer}

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[
                WandbCallback(), 
                EarlyStoppingCallback(
                    early_stopping_patience=10,
                    early_stopping_threshold=0.0
                )
            ],
        )
        
        return trainer

    def train(self):
        logger.info("Starting training...")
        checkpoint_path = os.path.join(self.config.training.output_dir, "checkpoint-last")
        resume = checkpoint_path if os.path.exists(checkpoint_path) else None
        if resume:
            logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        self.trainer.train(resume_from_checkpoint=resume)
        # self.trainer.train(resume_from_checkpoint=True if os.path.exists(self.config.training.output_dir) else None)
        logger.info("Training completed")

        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info("Starting evaluation...")
            
            eval_results = self.trainer.evaluate()
            logger.info(f"Final evaluation WER: {eval_results['eval_wer']}")
            eval_path = os.path.join(self.config.training.eval_output_dir, "eval_results.json")
            with open(eval_path, "w") as f:
                json.dump(eval_results, f, indent=4)
            artifact = wandb.Artifact("eval_results", type="evaluation")
            artifact.add_file(eval_path)
            wandb.log_artifact(artifact)
            logger.info("Evaluation results saved")
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")

        model_dir = os.path.join(os.getcwd(), "models", f"phowhisper-large-{self.config.region.lower()}-vi")
        self.trainer.save_model(model_dir)
        self.processor.save_pretrained(model_dir)
        logger.info("Model and processor saved locally")

        self.trainer.push_to_hub(
            commit_message=f"Fine-tuned PhoWhisper large with WhisperX chunking on ViMD ({self.config.region})",
            tags=["speech-recognition", "vietnamese", "whisperx", f"region-{self.config.region.lower()}"],
            dataset=self.config.dataset.name,
            language=self.config.model.language,
            finetuned_from=self.config.model.model_id,
            tasks="automatic-speech-recognition",
        )
        self.processor.push_to_hub(self.config.training.hub_model_id)
        logger.info("Model and processor pushed to Hugging Face Hub")
        wandb.finish()
