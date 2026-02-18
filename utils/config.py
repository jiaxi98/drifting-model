"""Configuration helpers for dataset-specific training defaults."""

from pathlib import Path
from typing import Any, Dict

import yaml


DATASET_CONFIG_FILES: Dict[str, str] = {
    "mnist": "config/datasets/mnist.yaml",
    "cifar": "config/datasets/cifar.yaml",
    "imagenet": "config/datasets/imagenet.yaml",
}


def get_default_config(dataset: str) -> Dict[str, Any]:
    """Load YAML defaults for a supported dataset."""
    name = dataset.lower()
    rel_path = DATASET_CONFIG_FILES.get(name)
    if rel_path is None:
        raise ValueError(f"Unsupported dataset: {dataset}")

    config_path = Path(rel_path)
    if not config_path.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        config_path = repo_root / config_path
    config_path = config_path.expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {name} dataset config: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"{name} dataset config must contain a mapping: {config_path}")
    return config.copy()
