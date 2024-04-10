import torch.nn as nn

from src.config_classes.adv_config import AdvConfig
from src.modules.gradient_reversal import GradientReversalLayer
from src.modules.polylinear import PolyLinearParallel


class Adversary(nn.Module):
    def __init__(self, config: AdvConfig, input_size: int):
        super().__init__()
        self.config = config
        self.input_size = input_size

        self.adv = nn.Sequential(
            GradientReversalLayer(config.grad_scaling),
            PolyLinearParallel(layer_config=[input_size] + config.dims,
                               n_parallel=config.n_parallel,
                               input_dropout=config.input_dropout,
                               activation_fn=config.activation_fn
                               )
        )

    def forward(self, x):
        return self.adv(x)
