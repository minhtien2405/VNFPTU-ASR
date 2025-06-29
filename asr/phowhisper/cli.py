import click
import logging
import os
import sys
import torch
import wandb
from huggingface_hub import login


from dotenv import load_dotenv
from pathlib import Path  
from configs.config import Config
from processor.data_processor import DataProcessor
from train.trainer import Trainer
from eval.evaluator import Evaluator
from infer.inference import Inference
from transformers import WhisperProcessor

load_dotenv(os.path.join(os.path.dirname(__file__),  "configs", ".env"))

def validate_file(file_path, description):
    """Validate if a file exists."""
    if not Path(file_path).is_file():
        logging.error(f"{description} not found: {file_path}")
        sys.exit(f"Error: {description} not found: {file_path}")

def setup_environment():
    token = os.getenv("HF_TOKEN")
    if token:
        login(token)
        
    wandb_token = os.getenv("WANDB_API_KEY")
    if wandb_token:
        try:
            wandb.login(key=wandb_token)
        except Exception as e:
            logging.error(f"WandB login failed: {e}")
            sys.exit("Error: WandB login failed. Check your WANDB_API_KEY in .env file.")
        finally:
            logging.info("WandB login successful")
    
    cache_dir = os.path.join(os.getcwd(), "cache")
    os.makedirs(cache_dir, exist_ok=True)

    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@click.group()
def cli():
    """CLI for PhoWhisper speech-to-text model management"""
    pass

@cli.command()
@click.option('--config', default='phowhisper/configs/config.yaml', help='Path to config file')
@click.option('--region', default='All', type=click.Choice(['All', 'Central', 'South', 'North']), help='Region to fine-tune on')
def train(config, region):
    
    setup_environment()
    
    """Fine-tune PhoWhisper model"""
    validate_file(config, "Config file")
    setup_logging(config, region)
    config_obj = Config(config, region)

    if wandb.run:
        config_obj.config['training']['learning_rate'] = wandb.config.get('training.learning_rate', config_obj.training.learning_rate)
        config_obj.config['training']['per_device_train_batch_size'] = wandb.config.get('training.per_device_train_batch_size', config_obj.training.per_device_train_batch_size)
        config_obj.config['lora']['r'] = wandb.config.get('lora.r', config_obj.lora.r)
        config_obj.config['lora']['lora_alpha'] = wandb.config.get('lora.lora_alpha', config_obj.lora.lora_alpha)
        config_obj.config['lora']['lora_dropout'] = wandb.config.get('lora.lora_dropout', config_obj.lora.lora_dropout)
        logging.info("Overridden config with WandB sweep parameters")

    processor = WhisperProcessor.from_pretrained(config_obj.model.model_id, language=config_obj.model.language, task=config_obj.model.task)
    data_processor = DataProcessor(config_obj, processor, device="cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, valid_dataset = data_processor.load_dataset()
    train_dataset = data_processor.process(train_dataset)
    valid_dataset = data_processor.process(valid_dataset)

    trainer = Trainer(config_obj, processor, train_dataset, valid_dataset)
    trainer.train()

@cli.command()
@click.option('--config', default='phowhisper/configs/config.yaml', help='Path to config file')
@click.option('--model-path', required=True, help='Path to trained model or HF model ID')
def evaluate(config, model_path):
    """Evaluate PhoWhisper model"""
    validate_file(config, "Config file")
    validate_file(model_path, "Model file")
    region = model_path.split('-')[-2].capitalize() if '-' in model_path else 'All'
    setup_logging(config, region)
    config_obj = Config(config, region)

    evaluator = Evaluator(config_obj, model_path)
    results = evaluator.evaluate()
    click.echo("\n Evaluation Summary")
    click.echo(f"Region: {region}")
    click.echo(f"WER: {results['total_wer']:.2f}% | Latency: {results['total_latency']:.2f}s | Avg Inference: {results['average_infer_latency']:.2f}s")

@cli.command()
@click.option('--config', default='phowhisper/configs/config.yaml', help='Path to config file')
@click.option('--model-path', default='models/phowhisper-large-{region}-vi', help='Path to trained model or HF ID')
@click.option('--audio-path', required=True, help='Path to audio file')
@click.option('--region', default='All', type=click.Choice(['All', 'Central', 'South', 'North']), help='Region for model')
def infer(config, model_path, audio_path, region):
    """Run inference on audio file"""
    validate_file(config, "Config file")
    validate_file(audio_path, "Audio file")
    setup_logging(config, region)
    config_obj = Config(config, region)
    model_path = model_path.format(region=region.lower())

    inference = Inference(config_obj, model_path)
    results = inference.infer(audio_path)
    click.echo("\n🔍 Transcription Result")
    for seg in results:
        click.echo(f"[{seg['start_time']:.2f}s - {seg['end_time']:.2f}s]: {seg['text']} (Latency: {seg['latency']:.2f}s)")

def setup_logging(config_path, region):
    validate_file(config_path, "Config file")
    config = Config(config_path, region)
    os.makedirs(config.training.eval_output_dir, exist_ok=True)
    os.makedirs(config.logging.log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(config.logging.log_dir, config.logging.log_file),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger(__name__).info(f"Logging setup completed for region {region} on device {'cuda' if torch.cuda.is_available() else 'cpu'}")

if __name__ == '__main__':
    cli()