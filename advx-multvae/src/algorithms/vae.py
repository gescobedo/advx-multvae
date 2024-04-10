from copy import deepcopy
from typing import List

import torch
from torch.nn import CrossEntropyLoss, MSELoss

from src.config_classes.adv_config import AdvConfig
from src.modules.losses import VAELoss, AdvLosses
from src.modules.mult_vae import MultVAEAdv


def setup_model_and_optimizers_vae(config, n_users, n_items, device, adv_configs=None):
    # create deep copy to ensure that local modifications of dict do not cause problems elsewhere
    config = deepcopy(config)

    # create instance of the model
    model = MultVAEAdv.from_config(config["model"], n_items, adv_configs)
    model.to(device)

    # define loss functions
    loss_fn = VAELoss(**config["loss"])

    default_opt_config = config["optim"]
    # enable possibility to have a different optimizer configuration for each part of the model
    # if not specified, it defaults to the general optimizer config
    encoder_opt_config = config.get("optim_enc", {})
    decoder_opt_config = config.get("optim_dec", {})
    optim = torch.optim.Adam([
        {"params": model.encoder.parameters(), **encoder_opt_config},
        {"params": model.decoder.parameters(), **decoder_opt_config}
    ], **default_opt_config)

    if adv_configs is not None:
        loss_fn_adv = AdvLosses(adv_configs)
        optim_adv = None
        if len(adv_configs) >= 1:
            adv_opt_config = config.get("optim_adv", {})
            optim_adv = torch.optim.Adam([{
                "params": model.adversaries.parameters(), **adv_opt_config
            }], **default_opt_config)
        return model, optim, loss_fn, optim_adv, loss_fn_adv
    return model, optim, loss_fn

def preprocess_data_vae(model_input: List[torch.Tensor], device):
    return [mi.to(device) for mi in model_input]
