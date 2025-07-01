import yaml
import logging
import os
from types import SimpleNamespace
from pathlib import Path

logger = logging.getLogger(__name__)

def dict_to_namespace(d):
    """Recursively convert dict to SimpleNamespace for dot-access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    return d

class ChunkformerConfig:
    def __init__(self, config_path: str, region: str = "All"):
        """
        Initialize Chunkformer configuration.
        
        Args:
            config_path (str): Path to the YAML configuration file
            region (str): Region to train on ("All", "Central", "South", "North")
        """
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)

        self.region = region
        # Replace {region} placeholders
        raw_config = self._replace_region_in_config(raw_config)
        # Convert to dot-accessible object
        self.config = dict_to_namespace(raw_config)

        # Validate configuration
        self._validate_config()
        
        logger.info(f"Loaded Chunkformer configuration from {config_path} for region {region}")

    def _replace_region_in_config(self, config_dict):
        """Replace {region} placeholders with actual region name."""
        def replace_region(obj):
            if isinstance(obj, dict):
                return {k: replace_region(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_region(item) for item in obj]
            elif isinstance(obj, str):
                return obj.format(region=self.region.lower())
            return obj
        return replace_region(config_dict)

    def _validate_config(self):
        """Validate essential configuration parameters."""
        required_sections = ['model', 'dataset', 'training']
        for section in required_sections:
            if not hasattr(self.config, section):
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate model parameters
        if not hasattr(self.config.model, 'model_id'):
            raise ValueError("Missing model.model_id in configuration")
            
        # Validate dataset parameters
        if not hasattr(self.config.dataset, 'name'):
            raise ValueError("Missing dataset.name in configuration")
            
        # Validate training parameters
        if not hasattr(self.config.training, 'output_dir'):
            raise ValueError("Missing training.output_dir in configuration")
            
        # Validate region
        valid_regions = ["All", "Central", "South", "North"]
        if self.region not in valid_regions:
            logger.warning(f"Region '{self.region}' not in standard regions: {valid_regions}")

    def get(self, key: str, default=None):
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = getattr(value, k, default)
            if value is default:
                break
        return value

    def __getattr__(self, name: str):
        """Enable direct access to config attributes."""
        if hasattr(self.config, name):
            return getattr(self.config, name)
        raise AttributeError(f"'ChunkformerConfig' object has no attribute '{name}'")

    def to_dict(self):
        """Convert config back to dictionary."""
        def namespace_to_dict(obj):
            if isinstance(obj, SimpleNamespace):
                return {k: namespace_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [namespace_to_dict(item) for item in obj]
            return obj
        return namespace_to_dict(self.config)

    def save(self, output_path: str):
        """Save configuration to file."""
        config_dict = self.to_dict()
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Configuration saved to {output_path}")

# Convenience function for backward compatibility
Config = ChunkformerConfig