from typing import Optional, List
from dataclasses import dataclass


@dataclass
class AtkConfig:
    feature: str
    type: str
    dims: list
    activation_fn: Optional[str] = "relu"
    input_dropout: Optional[float] = 0.0
    n_parallel: Optional[int] = 1
    loss_fn: Optional[str] = "mse"
    loss_weight: Optional[float] = 1
    loss_class_weights: Optional[List[float]] = None
    scoring_fn: Optional[str] = None
    group_name: Optional[str] = None
