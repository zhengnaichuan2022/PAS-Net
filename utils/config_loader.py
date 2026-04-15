"""
YAML configuration loader.
Supports loading YAML configs and merging multiple files.
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Configuration loader."""

    def __init__(self, config_path: str = None):
        """
        Initialize the loader.

        Args:
            config_path: Path to the YAML file.
        """
        self.config = {}
        if config_path:
            self.load(config_path)

    def load(self, config_path: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.

        Args:
            config_path: Path to the YAML file.

        Returns:
            Configuration dictionary.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Resolve ${ENV_VAR} placeholders
        self._resolve_env_vars(self.config)

        return self.config

    def _resolve_env_vars(self, config: Dict[str, Any]):
        """Recursively substitute environment variables."""
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                    env_var = value[2:-1]
                    config[key] = os.getenv(env_var, value)
                elif isinstance(value, (dict, list)):
                    self._resolve_env_vars(value)
        elif isinstance(config, list):
            for item in config:
                if isinstance(item, (dict, list)):
                    self._resolve_env_vars(item)

    def merge(self, other_config: Dict[str, Any]):
        """
        Deep-merge another configuration dict.

        Args:
            other_config: Dictionary to merge into the current config.
        """
        self.config = self._deep_merge(self.config, other_config)

    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep-merge two dicts."""
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value by dotted key (e.g. ``dataset.name``).

        Args:
            key: Dotted path.
            default: Default if missing.

        Returns:
            The value at ``key``.
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """
        Set a value by dotted key.

        Args:
            key: Dotted path.
            value: Value to assign.
        """
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def save(self, save_path: str):
        """
        Save the current config to a YAML file.

        Args:
            save_path: Output path.
        """
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

    def __getitem__(self, key: str) -> Any:
        """Dict-style read access."""
        return self.config[key]

    def __setitem__(self, key: str, value: Any):
        """Dict-style write access."""
        self.config[key] = value

    def __contains__(self, key: str) -> bool:
        """``in`` operator."""
        return key in self.config


def load_config(config_path: str = None) -> ConfigLoader:
    """
    Convenience wrapper to load a config file.

    Args:
        config_path: Path to YAML; if None, uses ``snn-config/default_config.yaml``.

    Returns:
        ConfigLoader instance.
    """
    if config_path is None:
        default_path = Path(__file__).parent.parent / "snn-config" / "default_config.yaml"
        config_path = str(default_path)

    loader = ConfigLoader()
    loader.load(config_path)
    return loader
