#!/usr/bin/env python3
"""
Chunkformer Training Script for VIMD Dataset
============================================

This script provides a command-line interface for training Chunkformer
models on the VIMD (Vietnamese Multi-Domain) dataset.

Usage:
    python train_chunkformer.py --config configs/config_vimd.yaml --region All

Features:
- Multi-region training support (All, Central, South, North)
- Integration with WeNet framework
- Automatic data preprocessing and chunking
- WandB integration for experiment tracking
- Comprehensive logging and error handling
"""

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from configs.config import ChunkformerConfig
from train.trainer import ChunkformerTrainer

def setup_logging(log_dir: str, log_file: str, level: str = "INFO"):
    """Set up logging configuration."""
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    log_level = getattr(logging, level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging configured. Log file: {log_path}")
    return logger

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Chunkformer model on VIMD dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train on all regions
    python train_chunkformer.py --config configs/config_vimd.yaml --region All
    
    # Train on Central region only
    python train_chunkformer.py --config configs/config_vimd.yaml --region Central
    
    # Resume training with custom log level
    python train_chunkformer.py --config configs/config_vimd.yaml --region All --log-level DEBUG
    
    # Train without evaluation
    python train_chunkformer.py --config configs/config_vimd.yaml --region All --no-eval
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration YAML file"
    )
    
    parser.add_argument(
        "--region",
        type=str,
        choices=["All", "Central", "South", "North"],
        default="All",
        help="Region to train on (default: All)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (default: cuda)"
    )
    
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation after training"
    )
    
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation on existing model"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run data preparation only without training"
    )
    
    return parser.parse_args()

def validate_environment():
    """Validate that required dependencies are available."""
    required_packages = [
        "torch",
        "torchaudio", 
        "datasets",
        "wandb",
        "numpy",
        "soundfile",
        "jiwer",
        "yaml"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Error: Missing required packages: {missing_packages}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    # Check for WeNet
    try:
        import wenet
    except ImportError:
        print("Warning: WeNet not found. Training will fail without WeNet installation.")
        print("Please install WeNet following the instructions in the README.")
        return False
    
    return True

def main():
    """Main training function."""
    args = parse_arguments()
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    try:
        # Load configuration
        print(f"Loading configuration from {args.config}")
        config = ChunkformerConfig(args.config, args.region)
        
        # Set up logging
        logger = setup_logging(
            config.logging.log_dir,
            config.logging.log_file.format(region=args.region.lower()),
            args.log_level
        )
        
        logger.info("=" * 80)
        logger.info("Starting Chunkformer Training")
        logger.info("=" * 80)
        logger.info(f"Configuration file: {args.config}")
        logger.info(f"Region: {args.region}")
        logger.info(f"Device: {args.device}")
        logger.info(f"Log level: {args.log_level}")
        
        # Log configuration summary
        logger.info("Configuration Summary:")
        logger.info(f"  Model: {config.model.model_id}")
        logger.info(f"  Dataset: {config.dataset.name}")
        logger.info(f"  Chunk size: {config.model.chunk_size}")
        logger.info(f"  Context: ({config.model.left_context_size}, {config.model.right_context_size})")
        logger.info(f"  Learning rate: {config.training.learning_rate}")
        logger.info(f"  Epochs: {config.training.num_train_epochs}")
        logger.info(f"  Batch size: {config.training.per_device_train_batch_size}")
        logger.info(f"  Output dir: {config.training.output_dir}")
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = ChunkformerTrainer(config, args.device)
        
        if args.dry_run:
            logger.info("Dry run mode: preparing data only")
            trainer.prepare_data()
            logger.info("Dry run completed successfully!")
            return
        
        if args.eval_only:
            logger.info("Evaluation-only mode")
            trainer.evaluate()
            logger.info("Evaluation completed successfully!")
            return
        
        # Run training
        logger.info("Starting training process...")
        trainer.train()
        
        # Run evaluation if not disabled
        if not args.no_eval:
            logger.info("Starting evaluation...")
            trainer.evaluate()
        
        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("TRAINING FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}")
        logger.error(f"Type: {type(e).__name__}")
        logger.error("Traceback:")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()