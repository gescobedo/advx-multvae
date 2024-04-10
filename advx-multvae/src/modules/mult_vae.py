from typing import Union

import torch
from torch import nn
import torch.nn.functional as F

from src.config_classes.adv_config import AdvConfig
from src.modules.adversary import Adversary
from src.modules.mult_dae import MultDAE
from src.modules.parallel import Parallel, ParallelMode
from src.modules.polylinear import PolyLinear


class MultVAE(MultDAE):
    def __init__(self, encoder_dims, decoder_dims=None, input_dropout=0.5,
                 activation_fn: Union[str, nn.Module] = nn.ReLU(),
                 decoder_dropout=0., normalize_inputs=True, l1_weight_decay=None, **kwargs):
        """
        Variational Autoencoders for Collaborative Filtering - Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, Tony Jebara
        https://arxiv.org/abs/1802.05814
        Attributes
        ---------
        encoder_dims  : list
            list of values that defines the structure of the network on the decoder side
        decoder_dims : list
            list of values that defines the structure of the network on the encoder side (Optional)
        input_dropout: float
            dropout value
        """
        super().__init__(encoder_dims, decoder_dims, input_dropout, activation_fn, l1_weight_decay)
        self.decoder_dropout = nn.Dropout(p=decoder_dropout)

        # Overwrite the encoder
        # variational auto encoder needs two times the hidden size, as the latent
        # space will be split up into a mean and a log(std) vector, with which we
        # sample from the multinomial distribution
        encoder_dims = self.encoder_dims.copy()
        encoder_dims[-1] *= 2

        self.normalize_inputs = normalize_inputs
        l1_weight_decay = kwargs.get("l1_weight_decay")
        self.encoder = PolyLinear(encoder_dims, activation_fn, l1_weight_decay=l1_weight_decay)
        self.apply(self._init_weights)

    def encoder_forward(self, x):
        """
        Performs the encoding step of the variational auto-encoder
        :param x: the unnormalized data to encode
        :return: the sampled encoding + the KL divergence of the generated mean and std params
        """
        x = self.dropout(x)
        if self.normalize_inputs:
            x = F.normalize(x, 2, 1)
        x = self.encoder(x)
        mu, logvar = x[:, :self.latent_size], x[:, self.latent_size:]
        KL = self._calc_KL_div(mu, logvar)

        # Sampling
        z = self._sampling(mu, logvar)
        return z, KL

    def forward(self, x):
        z, KL = self.encoder_forward(x)
        z = self.decoder_dropout(z)
        z = self.decoder(z)

        return z, KL

    def _sampling(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std * self.training
        return z

    @staticmethod
    def _calc_KL_div(mu, logvar):
        """
        Calculates the KL divergence of a multinomial distribution with the generated
        mean and std parameters
        """
        # Calculation for multinomial distribution with multivariate normal and standard normal distribution based on
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
        # Mean is used as we may have different batch sizes, thus possibly have different losses throughout training
        return 0.5 * torch.mean(torch.sum(-logvar + torch.exp(logvar) + mu ** 2 - 1, dim=1))


class MultVAEAdv(MultVAE):
    """
    Adversarial network, based on
    "Adversarial Removal of Demographic Attributes from Text Data"
    https://www.aclweb.org/anthology/D18-1002/
    ========================
    Functionality we want to provide:
        - adjustable l, which controls the intensity of the reversal layer (GRL)
        - adjustable number of adversaries
        - adjustable depth of adversaries
        - (should we allow for different output shapes, e.g., s.t. we can try to predict different demographics?)
    """

    def __init__(self, encoder_dims: list, decoder_dims: list = None, input_dropout: float = 0.5,
                 activation_fn: Union[str, nn.Module] = nn.ReLU(), decoder_dropout: float = 0.,
                 normalize_inputs: bool = True, adv_configs: list[AdvConfig] = None, **kwargs):
        """
        :param encoder_dims, decoder_dims, input_dropout, decoder_dropout, normalize_inputs: ==> see MultVAE
        :param activation_fn: activation function between layers
        :param adv_configs (list[AdvConfig]): each dict contains configurations for the adversaries,
                                              see config_classes/AdvConfig.py for details
        """
        super().__init__(encoder_dims, decoder_dims, input_dropout, activation_fn,
                         decoder_dropout, normalize_inputs, **kwargs)

        self.adv_configs = adv_configs
        self.are_adversaries_enabled = adv_configs is not None and len(adv_configs) > 0

        advs = [Adversary(config, input_size=self.latent_size) for config in adv_configs] if adv_configs else []
        # pack into module list to register values as parameters
        self.adversaries = Parallel(advs, parallel_mode=ParallelMode.SingleInMultiOut)

    def reset_decoder(self):
        self.decoder.apply(self._init_weights)

    def get_encoding_size(self):
        return self.latent_size

    def encode_user(self, x):
        z, _ = self.encoder_forward(x)
        return z

    def forward(self, x):
        z, KL = self.encoder_forward(x)
        dz = self.decoder_dropout(z)
        d = self.decoder(dz)

        adv_results = []
        if self.are_adversaries_enabled:
            adv_results = self.adversaries(z)
        return d, {"KL": KL}, adv_results

    def enable_adversaries(self):
        self.are_adversaries_enabled = True

    def disable_adversaries(self):
        self.are_adversaries_enabled = False

    @classmethod
    def from_config(cls, model_config: dict, n_items: int, adv_configs: list[AdvConfig] = None):
        encoder_dims = model_config.pop("encoder_dims", None) or model_config["dims"]
        decoder_dims = model_config.pop("decoder_dims", None) or model_config["dims"][::-1]
        model_config.pop("dims", None)

        encoder_dims = [n_items] + encoder_dims
        decoder_dims = decoder_dims + [n_items]

        return MultVAEAdv(encoder_dims, decoder_dims, **model_config, adv_configs=adv_configs)
