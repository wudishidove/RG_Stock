"""
HorizonConfig dataclass and YAML loader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class HorizonConfig:
    name: str
    horizon_steps: int          # h in 10-min bars; -1 for EOD
    K: int = 100
    alpha: float = 0.9
    A_sparsity: float = 0.15
    rho: float = 0.4
    C_sparsity: float = 0.95
    gamma: float = 0.005
    train_window_bars: int = 3
    buffer_bars: int = 1
    cv_lookback_days: int = 5
    cv_split: float = 0.7
    lambda_ridge: float = 1e-4
    D: int = 6                  # signal dimension


def load_horizon_configs(config_path: str | Path) -> dict[str, HorizonConfig]:
    """Load per-horizon hyperparameter configs from YAML."""
    with open(config_path) as f:
        data = yaml.safe_load(f)

    configs: dict[str, HorizonConfig] = {}
    for name, params in data["horizons"].items():
        configs[name] = HorizonConfig(name=name, **params)
    return configs
