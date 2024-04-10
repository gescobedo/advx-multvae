import numpy as np
from typing import List, Union

import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from src.config_classes.adv_config import AdvConfig
from src.config_classes.atk_config import AtkConfig


def accuracy(inputs: torch.Tensor, targets: torch.Tensor):
    predictions = torch.argmax(inputs, dim=-1).detach().cpu().numpy()
    return accuracy_score(targets.cpu().long().numpy(), predictions)


def balanced_accuracy(inputs: torch.Tensor, targets: torch.Tensor):
    predictions = torch.argmax(inputs, dim=-1).detach().cpu().numpy()
    return balanced_accuracy_score(targets.cpu().long().numpy(), predictions)



eval_fn_lookup = {
    "mse": nn.MSELoss(),
    "mae": nn.L1Loss(),
    # need to flip order of inputs and targets for sklearn metrics
    "acc": accuracy,
    "bacc": balanced_accuracy
}


class AdvEval(nn.Module):
    """
    Evaluation function for adversarial networks.
    """

    def __init__(self, adv_config: List[Union[AdvConfig, AtkConfig]]):
        super().__init__()
        self.adv_config = adv_config

    def _eval_adv_group(self, config: Union[AdvConfig, AtkConfig], inputs: List[torch.Tensor], targets: torch.Tensor):
        if config.scoring_fn is None:
            return -1
        return np.mean([self._eval_single_adv(config, inp, targets) for inp in inputs])

    @staticmethod
    def _eval_single_adv(config: Union[AdvConfig, AtkConfig], inputs: torch.Tensor, targets: torch.Tensor):
        if config.scoring_fn in ["mse","mae"]:
            targets = torch.reshape(targets,(-1,1))
            #print("targets:"+str(targets.size()))
        return eval_fn_lookup[config.scoring_fn](inputs, targets).item()

    def forward(self, inputs: List[List[torch.Tensor]], targets: List[torch.Tensor]):
        """
        Evaluates the different adversaries in the different adversarial groups.

        :param inputs: A list of lists of tensors, where the tensors are the results of the individual adversaries
                       and the sublists are the results for a group of adversaries.
        :param targets: A list of tensors, where each tensor is the target value an adversary should achieve
        """

        evaluations = {f"{cfg.group_name}_{cfg.scoring_fn}": self._eval_adv_group(cfg, inp, tar)
                       for cfg, inp, tar in zip(self.adv_config, inputs, targets)
                       if cfg.scoring_fn is not None}

        return evaluations
