"""Configuration file loader."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default config.

    Returns:
        Dictionary containing configuration parameters.
    """
    if config_path is None:
        # Default to robot_config.yaml in configs directory
        config_path = Path(__file__).parent.parent / "configs" / "robot_config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_joint_limits(config: Dict[str, Any], use_soft_limits: bool = True) -> tuple:
    """Extract joint limits from configuration.

    Args:
        config: Configuration dictionary.
        use_soft_limits: If True, use soft limits; otherwise use hard limits.

    Returns:
        Tuple of (lower_limits, upper_limits) as lists.
    """
    limits_key = "soft_joint_limits" if use_soft_limits else "joint_limits"
    limits = config["robot"][limits_key]

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll"]

    lower = [limits[name][0] for name in joint_names]
    upper = [limits[name][1] for name in joint_names]

    return lower, upper


def get_link_lengths(config: Dict[str, Any]) -> Dict[str, float]:
    """Extract link lengths from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Dictionary mapping link names to lengths in meters.
    """
    return config["robot"]["link_lengths"]
