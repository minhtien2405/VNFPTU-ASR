import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import click
import logging
import sys
import torch
import wandb
from huggingface_hub import login
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

from configs.config import Config
from processor.data_processor import DataProcessor
from train.trainer import Trainer
from eval.evaluator import Evaluator
from infer.inference import Inference
from transformers import WhisperProcessor

logger = logging.getLogger(__name__)

class CLIError(Exception):
    """Base exception for CLI errors"""
    pass

def setup_cuda() -> None:
    """Setup CUDA environment"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        logger.info("CUDA setup completed")

def setup_cache() -> None:
    """Setup cache directories"""
    cache_dir = Path.cwd() / "cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    logger.info(f"Cache directory set to {cache_dir}")

def setup_wandb(config_obj: Config) -> None:
    """Setup and configure WandB"""
    if not wandb.run:
        return
    
    params = {
        'training.learning_rate': config_obj.training.learning_rate,
        'training.per_device_train_batch_size': config_obj.training.per_device_train_batch_size,
        'lora.r': config_obj.lora.r,
        'lora.lora_alpha': config_obj.lora.lora_alpha,
        'lora.lora_dropout': config_obj.lora.lora_dropout
    }
    
    for key, default in params.items():
        config_obj.config[key.split('.')[0]][key.split('.')[1]] = wandb.config.get(key, default)
    
    logger.info("WandB configuration applied successfully")

def setup_environment() -> None:
    """Setup complete environment including HF token, WandB, and cache"""
    try:
        load_dotenv(Path(__file__).parent / "configs" / ".env")
        
        if token := os.getenv("HF_TOKEN"):
            login(token)
            logger.info("Hugging Face login successful")
            
        if wandb_token := os.getenv("WANDB_API_KEY"):
            try:
                wandb.login(key=wandb_token)
                logger.info("WandB login successful")
            except Exception as e:
                raise CLIError(f"WandB login failed: {e}")
        
        setup_cache()
        setup_cuda()
        
    except Exception as e:
        raise CLIError(f"Environment setup failed: {e}")

def setup_logging(config_path: str, region: str) -> None:
    """Setup logging configuration"""
    try:
        config = Config(config_path, region)
        log_dir = Path(config.logging.log_dir)
        eval_dir = Path(config.training.eval_output_dir)
        
        log_dir.mkdir(exist_ok=True)
        eval_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            filename=log_dir / config.logging.log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Logging setup completed for region {region} on device {device}")
        
    except Exception as e:
        raise CLIError(f"Logging setup failed: {e}")

@click.group()
def cli():
    """PhoWhisper ASR Model Management CLI"""
    pass

@cli.command()
@click.option('--config', type=click.Path(exists=True), default='phowhisper/configs/config.yaml')
@click.option('--region', type=click.Choice(['All', 'Central', 'South', 'North'], case_sensitive=False))
def train(config: str, region: str) -> None:
    """Fine-tune PhoWhisper model"""
    try:
        setup_environment()
        setup_logging(config, region)
        config_obj = Config(config, region)
        setup_wandb(config_obj)
        
        processor = WhisperProcessor.from_pretrained(
            config_obj.model.model_id,
            language=config_obj.model.language,
            task=config_obj.model.task
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        data_processor = DataProcessor(config_obj, processor, device)
        
        train_dataset, valid_dataset = data_processor.load_dataset()
        train_dataset = data_processor.process(train_dataset)
        valid_dataset = data_processor.process(valid_dataset)
        
        trainer = Trainer(config_obj, processor, train_dataset, valid_dataset)
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--config', type=click.Path(exists=True), default='phowhisper/configs/config.yaml')
@click.option('--model-path', required=True, type=click.Path(exists=True))
def evaluate(config: str, model_path: str) -> None:
    """Evaluate PhoWhisper model"""
    try:
        region = model_path.split('-')[-2].capitalize() if '-' in model_path else 'All'
        setup_logging(config, region)
        config_obj = Config(config, region)
        
        evaluator = Evaluator(config_obj, model_path)
        results = evaluator.evaluate()
        
        click.echo("\n📊 Evaluation Summary")
        click.echo(f"🌍 Region: {region}")
        click.echo(
            f"📈 WER: {results['total_wer']:.2f}% | "
            f"⏱️ Latency: {results['total_latency']:.2f}s | "
            f"🔄 Avg Inference: {results['average_infer_latency']:.2f}s"
        )
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--config', type=click.Path(exists=True), default='phowhisper/configs/config.yaml')
@click.option('--model-path', default='models/phowhisper-large-{region}-vi')
@click.option('--audio-path', required=True, type=click.Path(exists=True))
@click.option('--region', type=click.Choice(['All', 'Central', 'South', 'North'], case_sensitive=False))
def infer(config: str, model_path: str, audio_path: str, region: str) -> None:
    """Run inference on audio file"""
    try:
        setup_logging(config, region)
        config_obj = Config(config, region)
        model_path = model_path.format(region=region.lower())
        
        inference = Inference(config_obj, model_path)
        results = inference.infer(audio_path)
        
        click.echo("\n🔍 Transcription Result")
        for seg in results:
            click.echo(
                f"⏰ [{seg['start_time']:.2f}s - {seg['end_time']:.2f}s]: "
                f"💬 {seg['text']} "
                f"(⚡ Latency: {seg['latency']:.2f}s)"
            )
            
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise click.ClickException(str(e))

if __name__ == '__main__':
    cli()