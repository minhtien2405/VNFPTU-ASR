import yaml
import logging
import os
from types import SimpleNamespace

logger = logging.getLogger(__name__)

def dict_to_namespace(d):
    """Recursively convert dict to SimpleNamespace for dot-access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    return d

class Config:
    def __init__(self, config_path: str, region: str = "All"):
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)

        self.region = region
        # Replace {region} placeholders
        raw_config = self._replace_region_in_config(raw_config)
        # Convert to dot-accessible object
        self.config = dict_to_namespace(raw_config)

        logger.info(f"Loaded configuration from {config_path} for region {region}")

    def _replace_region_in_config(self, config_dict):
        def replace_region(obj):
            if isinstance(obj, dict):
                return {k: replace_region(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_region(item) for item in obj]
            elif isinstance(obj, str):
                return obj.format(region=self.region.lower())
            return obj
        return replace_region(config_dict)

    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = getattr(value, k, default)
            if value is default:
                break
        return value

    def __getattr__(self, name: str):
        if hasattr(self.config, name):
            return getattr(self.config, name)
        raise AttributeError(f"'Config' object has no attribute '{name}'")