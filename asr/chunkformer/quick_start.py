#!/usr/bin/env python3
"""
Quick Start Script for Chunkformer VIMD Training
================================================

This script provides a quick way to test and validate the Chunkformer setup
before running the full training process.

Usage:
    python quick_start.py [--region Central] [--config configs/config_vimd.yaml]
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Set up basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available."""
    logger = logging.getLogger(__name__)
    logger.info("Checking dependencies...")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("torchaudio", "TorchAudio"),
        ("datasets", "HuggingFace Datasets"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
        ("soundfile", "SoundFile"),
        ("wandb", "Weights & Biases"),
        ("jiwer", "JIWER")
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {name} is available")
        except ImportError:
            logger.error(f"✗ {name} is NOT available")
            missing_packages.append(package)
    
    # Check WeNet separately
    try:
        import wenet
        logger.info("✓ WeNet is available")
    except ImportError:
        logger.error("✗ WeNet is NOT available")
        missing_packages.append("wenet")
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.error("Please install missing packages and try again.")
        return False
    
    logger.info("All dependencies are available!")
    return True

def test_config_loading(config_path: str, region: str):
    """Test configuration loading."""
    logger = logging.getLogger(__name__)
    logger.info(f"Testing configuration loading: {config_path}")
    
    try:
        from configs.config import ChunkformerConfig
        config = ChunkformerConfig(config_path, region)
        
        logger.info(f"✓ Configuration loaded successfully")
        logger.info(f"  - Model: {config.model.model_id}")
        logger.info(f"  - Dataset: {config.dataset.name}")
        logger.info(f"  - Region: {region}")
        logger.info(f"  - Chunk size: {config.model.chunk_size}")
        logger.info(f"  - Learning rate: {config.training.learning_rate}")
        
        return config
        
    except Exception as e:
        logger.error(f"✗ Configuration loading failed: {str(e)}")
        return None

def test_audio_chunker(config):
    """Test audio chunker functionality."""
    logger = logging.getLogger(__name__)
    logger.info("Testing audio chunker...")
    
    try:
        from processor.audio_chunker import AudioChunker
        import numpy as np
        
        # Create chunker
        chunker = AudioChunker(
            chunk_size=config.model.chunk_size,
            left_context_size=config.model.left_context_size,
            right_context_size=config.model.right_context_size,
            sampling_rate=config.dataset.sampling_rate,
            chunk_overlap=config.data_processing.chunk_overlap,
            min_chunk_length=config.data_processing.min_chunk_length,
            max_chunk_length=config.data_processing.max_chunk_length
        )
        
        # Test with synthetic audio (5 minutes)
        duration = 300  # 5 minutes
        sample_rate = config.dataset.sampling_rate
        audio = np.random.randn(duration * sample_rate).astype(np.float32)
        
        chunks = chunker.chunk_audio(audio, sample_rate)
        chunk_info = chunker.get_chunk_info(chunks)
        
        logger.info(f"✓ Audio chunker test passed")
        logger.info(f"  - Input duration: {duration} seconds")
        logger.info(f"  - Number of chunks: {chunk_info['total_chunks']}")
        logger.info(f"  - Average chunk length: {chunk_info['avg_chunk_length']:.2f} seconds")
        logger.info(f"  - Total processed duration: {chunk_info['total_duration']:.2f} seconds")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Audio chunker test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_data_processor(config):
    """Test data processor functionality."""
    logger = logging.getLogger(__name__)
    logger.info("Testing data processor...")
    
    try:
        from processor.data_processor import ChunkformerDataProcessor
        
        # Create data processor
        processor = ChunkformerDataProcessor(config, device="cpu")  # Use CPU for testing
        
        logger.info(f"✓ Data processor created successfully")
        logger.info(f"  - Region: {processor.region}")
        logger.info(f"  - Max text length: {processor.max_text_length}")
        
        # Test collator creation
        collate_fn = processor.create_data_collator()
        logger.info(f"✓ Data collator created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Data processor test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_dataset_access(config):
    """Test dataset access without full loading."""
    logger = logging.getLogger(__name__)
    logger.info("Testing dataset access...")
    
    try:
        from datasets import load_dataset
        
        # Try to access dataset info
        logger.info(f"Attempting to access dataset: {config.dataset.name}")
        
        # Load a small sample
        dataset = load_dataset(
            config.dataset.name,
            streaming=True,  # Use streaming to avoid downloading everything
            split="train"
        )
        
        # Get one sample
        sample = next(iter(dataset))
        
        logger.info(f"✓ Dataset access successful")
        logger.info(f"  - Sample keys: {list(sample.keys())}")
        if "region" in sample:
            logger.info(f"  - Sample region: {sample['region']}")
        if "text" in sample:
            logger.info(f"  - Sample text length: {len(sample['text'])} characters")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Dataset access failed: {str(e)}")
        logger.error("This might be due to network issues or dataset availability")
        return False

def run_full_test(config_path: str, region: str):
    """Run all tests."""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("CHUNKFORMER VIMD QUICK START TEST")
    logger.info("=" * 60)
    
    tests = [
        ("Dependencies", check_dependencies),
        ("Configuration", lambda: test_config_loading(config_path, region)),
    ]
    
    # First check dependencies and config
    config = None
    for test_name, test_func in tests:
        logger.info(f"\n[{test_name.upper()}]")
        result = test_func()
        if not result:
            logger.error(f"❌ {test_name} test failed!")
            return False
        elif test_name == "Configuration":
            config = result
        logger.info(f"✅ {test_name} test passed!")
    
    # Now test components that need config
    if config:
        component_tests = [
            ("Audio Chunker", lambda: test_audio_chunker(config)),
            ("Data Processor", lambda: test_data_processor(config)),
            ("Dataset Access", lambda: test_dataset_access(config)),
        ]
        
        for test_name, test_func in component_tests:
            logger.info(f"\n[{test_name.upper()}]")
            if test_func():
                logger.info(f"✅ {test_name} test passed!")
            else:
                logger.error(f"❌ {test_name} test failed!")
                return False
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 ALL TESTS PASSED!")
    logger.info("🚀 You're ready to start training Chunkformer!")
    logger.info("=" * 60)
    
    # Print next steps
    logger.info("\n📝 NEXT STEPS:")
    logger.info("1. Run a dry run to test data preparation:")
    logger.info(f"   python train_chunkformer.py --config {config_path} --region {region} --dry-run")
    logger.info("\n2. Start training:")
    logger.info(f"   python train_chunkformer.py --config {config_path} --region {region}")
    logger.info("\n3. Or train all regions:")
    logger.info("   bash scripts/train_all_regions.sh")
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Quick start test for Chunkformer VIMD training setup"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_vimd.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--region",
        type=str,
        choices=["All", "Central", "South", "North"],
        default="Central",
        help="Region to test with"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Check if config file exists
        if not Path(args.config).exists():
            logger.error(f"Configuration file not found: {args.config}")
            logger.error("Please make sure the config file exists or specify a different path.")
            sys.exit(1)
        
        # Run tests
        success = run_full_test(args.config, args.region)
        
        if not success:
            logger.error("\n💥 Some tests failed!")
            logger.error("Please fix the issues above before proceeding with training.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n⏹️  Test interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\n💥 Unexpected error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()