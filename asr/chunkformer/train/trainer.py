import os
import json
import torch
import logging
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional, Tuple
import evaluate
import wandb
from pathlib import Path
import subprocess
import tempfile

from processor.data_processor import ChunkformerDataProcessor

logger = logging.getLogger(__name__)

class ChunkformerTrainerError(Exception):
    """Custom exception for ChunkformerTrainer errors"""
    pass

class ChunkformerTrainer:
    """
    Trainer for Chunkformer model on VIMD dataset.
    Integrates with WeNet framework for ASR training.
    """
    
    def __init__(self, config, device: str = "cuda"):
        """
        Initialize ChunkformerTrainer.
        
        Args:
            config: Configuration object
            device: Device to use for training
        """
        self.config = config
        self.device = device
        self.region = config.region
        
        # Initialize data processor
        self.data_processor = ChunkformerDataProcessor(config, device)
        
        # Initialize wandb
        if hasattr(config, 'wandb'):
            wandb.init(
                project=config.wandb.project, 
                name=config.wandb.run_name.format(region=self.region.lower()),
                config=config.to_dict() if hasattr(config, 'to_dict') else {}
            )
            logger.info("WandB initialized")
        
        # Create output directories
        self.output_dir = config.training.output_dir.format(region=self.region.lower())
        self.eval_output_dir = config.training.eval_output_dir.format(region=self.region.lower())
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.eval_output_dir, exist_ok=True)
        
        logger.info(f"ChunkformerTrainer initialized for region '{self.region}'")

    def prepare_data(self):
        """Load and prepare datasets for training."""
        try:
            logger.info("Preparing data for training...")
            
            # Load datasets
            train_dataset, valid_dataset = self.data_processor.load_dataset()
            
            # Process datasets
            self.train_dataset = self.data_processor.process(train_dataset)
            self.valid_dataset = self.data_processor.process(valid_dataset)
            
            logger.info(
                f"Data preparation completed. "
                f"Train: {len(self.train_dataset)}, Valid: {len(self.valid_dataset)}"
            )
            
            return self.train_dataset, self.valid_dataset
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise ChunkformerTrainerError(f"Data preparation failed: {str(e)}")

    def create_wenet_config(self) -> str:
        """
        Create WeNet configuration file for Chunkformer training.
        
        Returns:
            str: Path to the created configuration file
        """
        try:
            # WeNet configuration template for Chunkformer
            wenet_config = {
                # Model configuration
                "model": {
                    "encoder": "chunkformer",
                    "encoder_conf": {
                        "output_size": 512,
                        "attention_heads": 8,
                        "linear_units": 2048,
                        "num_blocks": 12,
                        "dropout_rate": 0.1,
                        "positional_dropout_rate": 0.1,
                        "attention_dropout_rate": 0.1,
                        "input_layer": "conv2d",
                        "normalize_before": True,
                        "macaron_style": True,
                        "use_cnn_module": True,
                        "cnn_module_kernel": 15,
                        "activation_type": "swish",
                        "pos_enc_layer_type": "rel_pos",
                        "selfattention_layer_type": "rel_selfattn",
                        # Chunkformer specific
                        "chunk_size": self.config.model.chunk_size,
                        "left_context_size": self.config.model.left_context_size,
                        "right_context_size": self.config.model.right_context_size,
                    },
                    "decoder": "transformer",
                    "decoder_conf": {
                        "attention_heads": 8,
                        "linear_units": 2048,
                        "num_blocks": 6,
                        "dropout_rate": 0.1,
                        "positional_dropout_rate": 0.1,
                        "self_attention_dropout_rate": 0.1,
                        "src_attention_dropout_rate": 0.1,
                    },
                    "ctc": {
                        "ctc_weight": 0.3,
                        "ctc_dropoutrate": 0.1,
                        "ctc_grad_norm_type": "instance",
                    },
                    "model_conf": {
                        "length_normalized_loss": False,
                    }
                },
                
                # Dataset configuration
                "dataset_conf": {
                    "filter_conf": {
                        "max_length": int(self.config.dataset.max_audio_length * self.config.dataset.sampling_rate),
                        "min_length": 1000,  # minimum 1 second
                        "token_max_length": self.config.data_processing.max_text_length,
                        "token_min_length": 1,
                    },
                    "resample_conf": {
                        "resample_rate": self.config.dataset.sampling_rate,
                    },
                    "speed_perturb": True,
                    "spec_aug": True,
                    "spec_aug_conf": {
                        "num_t_mask": 2,
                        "num_f_mask": 2,
                        "max_t": 50,
                        "max_f": 10,
                    },
                    "shuffle": True,
                    "shuffle_conf": {
                        "shuffle_size": 2500,
                    },
                    "sort": True,
                    "sort_conf": {
                        "sort_size": 500,
                    },
                    "batch_conf": {
                        "batch_type": "dynamic",
                        "max_frames_in_batch": 12000,
                        "batch_size": self.config.training.per_device_train_batch_size,
                    },
                },
                
                # Training configuration
                "optim": self.config.training.optim,
                "optim_conf": {
                    "lr": float(self.config.training.learning_rate),
                    "weight_decay": 0.000001,
                },
                "scheduler": "warmuplr",
                "scheduler_conf": {
                    "warmup_steps": self.config.training.warmup_steps,
                },
                
                # Training parameters
                "max_epoch": self.config.training.num_train_epochs,
                "accum_grad": self.config.training.gradient_accumulation_steps,
                "grad_clip": self.config.training.max_grad_norm,
                "log_interval": self.config.training.logging_steps,
                "save_interval": self.config.training.save_steps,
                
                # Mixed precision
                "use_amp": self.config.training.fp16 or self.config.training.bf16,
                
                # Model save
                "model_dir": self.output_dir,
                "save_model": {
                    "save_epoch_interval": 1,
                    "save_best_model": True,
                    "best_model_criterion": [
                        ("valid", "wer", "min"),
                    ],
                },
            }
            
            # Save configuration
            config_path = os.path.join(self.output_dir, "wenet_config.yaml")
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(wenet_config, f, default_flow_style=False)
            
            logger.info(f"WeNet configuration saved to {config_path}")
            return config_path
            
        except Exception as e:
            logger.error(f"Failed to create WeNet configuration: {str(e)}")
            raise ChunkformerTrainerError(f"WeNet config creation failed: {str(e)}")

    def prepare_data_for_wenet(self) -> Tuple[str, str]:
        """
        Convert processed datasets to WeNet format.
        
        Returns:
            Tuple[str, str]: Paths to train and validation data lists
        """
        try:
            logger.info("Converting data to WeNet format...")
            
            # Create data directory
            data_dir = os.path.join(self.output_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            
            # Convert datasets
            train_data_list = self._convert_dataset_to_wenet_format(
                self.train_dataset, 
                os.path.join(data_dir, "train_data.list")
            )
            
            valid_data_list = self._convert_dataset_to_wenet_format(
                self.valid_dataset,
                os.path.join(data_dir, "valid_data.list")
            )
            
            logger.info(f"WeNet data format created: {train_data_list}, {valid_data_list}")
            return train_data_list, valid_data_list
            
        except Exception as e:
            logger.error(f"Failed to prepare data for WeNet: {str(e)}")
            raise ChunkformerTrainerError(f"WeNet data preparation failed: {str(e)}")

    def _convert_dataset_to_wenet_format(self, dataset, output_path: str) -> str:
        """Convert dataset to WeNet data list format."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, sample in enumerate(dataset):
                    # Save audio chunks as temporary files
                    audio_chunks = sample["audio_chunks"]
                    transcriptions = sample["transcriptions"]
                    
                    for j, (audio_chunk, transcription) in enumerate(zip(audio_chunks, transcriptions)):
                        # Create temporary audio file
                        temp_audio_dir = os.path.join(os.path.dirname(output_path), "audio_chunks")
                        os.makedirs(temp_audio_dir, exist_ok=True)
                        
                        audio_path = os.path.join(temp_audio_dir, f"sample_{i}_chunk_{j}.wav")
                        
                        # Save audio chunk as WAV file
                        import soundfile as sf
                        sf.write(audio_path, audio_chunk, self.config.dataset.sampling_rate)
                        
                        # Write WeNet format line: audio_path\ttranscription
                        f.write(f"{audio_path}\t{transcription}\n")
            
            logger.info(f"WeNet data list saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to convert dataset to WeNet format: {str(e)}")
            raise

    def create_vocab_file(self) -> str:
        """Create vocabulary file for WeNet training."""
        try:
            logger.info("Creating vocabulary file...")
            
            # Collect all text from datasets
            all_text = []
            for dataset in [self.train_dataset, self.valid_dataset]:
                for sample in dataset:
                    all_text.extend(sample["transcriptions"])
            
            # Create character-level vocabulary
            chars = set()
            for text in all_text:
                chars.update(text)
            
            # Sort characters
            vocab_list = sorted(list(chars))
            
            # Add special tokens
            special_tokens = ["<blank>", "<unk>", "<sos>", "<eos>"]
            vocab_list = special_tokens + vocab_list
            
            # Save vocabulary
            vocab_path = os.path.join(self.output_dir, "vocab.txt")
            with open(vocab_path, 'w', encoding='utf-8') as f:
                for i, token in enumerate(vocab_list):
                    f.write(f"{token}\t{i}\n")
            
            logger.info(f"Vocabulary file created: {vocab_path} ({len(vocab_list)} tokens)")
            return vocab_path
            
        except Exception as e:
            logger.error(f"Failed to create vocabulary file: {str(e)}")
            raise ChunkformerTrainerError(f"Vocabulary creation failed: {str(e)}")

    def train(self):
        """Main training function."""
        try:
            logger.info("Starting Chunkformer training...")
            
            # Prepare data
            self.prepare_data()
            
            # Create WeNet configuration
            wenet_config_path = self.create_wenet_config()
            
            # Prepare data for WeNet
            train_data_list, valid_data_list = self.prepare_data_for_wenet()
            
            # Create vocabulary
            vocab_path = self.create_vocab_file()
            
            # Run WeNet training
            self._run_wenet_training(
                wenet_config_path, 
                train_data_list, 
                valid_data_list, 
                vocab_path
            )
            
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise ChunkformerTrainerError(f"Training failed: {str(e)}")
        finally:
            if wandb.run:
                wandb.finish()

    def _run_wenet_training(
        self, 
        config_path: str, 
        train_data: str, 
        valid_data: str,
        vocab_path: str
    ):
        """Run WeNet training process."""
        try:
            logger.info("Starting WeNet training process...")
            
            # WeNet training command
            cmd = [
                "python", "-m", "wenet.bin.train",
                "--config", config_path,
                "--train_data", train_data,
                "--cv_data", valid_data,
                "--dict", vocab_path,
                "--model_dir", self.output_dir,
                "--ddp.rank", "0",
                "--ddp.world_size", "1",
                "--ddp.dist_backend", "nccl",
                "--num_workers", str(self.config.training.dataloader_num_workers),
                "--pin_memory"
            ]
            
            # Add checkpoint resuming if exists
            checkpoint_path = os.path.join(self.output_dir, "checkpoint")
            if os.path.exists(checkpoint_path):
                cmd.extend(["--checkpoint", checkpoint_path])
                logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            
            # Run training
            logger.info(f"Running command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output
            for line in process.stdout:
                logger.info(line.strip())
                
                # Parse training logs for wandb
                if wandb.run:
                    self._parse_and_log_metrics(line.strip())
            
            # Wait for completion
            return_code = process.wait()
            
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)
                
            logger.info("WeNet training completed successfully!")
            
        except Exception as e:
            logger.error(f"WeNet training failed: {str(e)}")
            raise

    def _parse_and_log_metrics(self, log_line: str):
        """Parse training logs and log metrics to wandb."""
        try:
            # Parse different types of log lines
            if "loss" in log_line.lower():
                # Extract loss values
                if "epoch" in log_line and "step" in log_line:
                    # Training loss
                    parts = log_line.split()
                    for i, part in enumerate(parts):
                        if part == "loss":
                            try:
                                loss_value = float(parts[i + 1])
                                wandb.log({"train_loss": loss_value})
                            except (IndexError, ValueError):
                                pass
                                
            elif "wer" in log_line.lower() or "cer" in log_line.lower():
                # Validation metrics
                parts = log_line.split()
                for i, part in enumerate(parts):
                    if part.lower() == "wer":
                        try:
                            wer_value = float(parts[i + 1])
                            wandb.log({"valid_wer": wer_value})
                        except (IndexError, ValueError):
                            pass
                    elif part.lower() == "cer":
                        try:
                            cer_value = float(parts[i + 1])
                            wandb.log({"valid_cer": cer_value})
                        except (IndexError, ValueError):
                            pass
                            
        except Exception as e:
            # Don't fail training due to logging issues
            logger.debug(f"Failed to parse log line: {log_line}, error: {str(e)}")

    def evaluate(self):
        """Evaluate the trained model."""
        try:
            logger.info("Starting model evaluation...")
            
            # Find the best model
            model_path = os.path.join(self.output_dir, "final.pt")
            if not os.path.exists(model_path):
                model_path = os.path.join(self.output_dir, "best.pt")
            
            if not os.path.exists(model_path):
                logger.warning("No trained model found for evaluation")
                return
            
            # Run evaluation using WeNet's recognition script
            eval_cmd = [
                "python", "-m", "wenet.bin.recognize",
                "--config", os.path.join(self.output_dir, "wenet_config.yaml"),
                "--test_data", os.path.join(self.output_dir, "data", "valid_data.list"),
                "--checkpoint", model_path,
                "--beam_size", "10",
                "--batch_size", str(self.config.training.per_device_eval_batch_size),
                "--penalty", "0.0",
                "--maxlenratio", "0.0",
                "--minlenratio", "0.0",
                "--ctc_weight", "0.3",
                "--result_file", os.path.join(self.eval_output_dir, "recognition_results.txt")
            ]
            
            logger.info(f"Running evaluation: {' '.join(eval_cmd)}")
            subprocess.run(eval_cmd, check=True)
            
            # Calculate WER/CER
            self._calculate_metrics()
            
            logger.info("Evaluation completed!")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")

    def _calculate_metrics(self):
        """Calculate WER and CER from recognition results."""
        try:
            results_path = os.path.join(self.eval_output_dir, "recognition_results.txt")
            
            if not os.path.exists(results_path):
                logger.warning("Recognition results file not found")
                return
            
            # Parse results and calculate metrics
            hypotheses = []
            references = []
            
            with open(results_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            hypotheses.append(parts[1])
                            # Get reference from validation dataset
                            # This is a simplified version - you might need to match by audio file
            
            if hypotheses and references:
                # Calculate WER using jiwer
                import jiwer
                wer = jiwer.wer(references, hypotheses)
                cer = jiwer.cer(references, hypotheses)
                
                metrics = {
                    "final_wer": wer,
                    "final_cer": cer
                }
                
                # Log to wandb
                if wandb.run:
                    wandb.log(metrics)
                
                # Save metrics
                metrics_path = os.path.join(self.eval_output_dir, "metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                logger.info(f"Final metrics - WER: {wer:.4f}, CER: {cer:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to calculate metrics: {str(e)}")