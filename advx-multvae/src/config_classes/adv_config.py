from typing import Optional
from dataclasses import dataclass

from src.config_classes.atk_config import AtkConfig


@dataclass
class AdvConfig(AtkConfig):
    grad_scaling: Optional[float] = 1
