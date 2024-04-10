from typing import Union

import torch
from torch import nn
import torch.nn.functional as F

from src.modules.polylinear import PolyLinear


class MultDAE(nn.Module):
    def __init__(self, encoder_dims, decoder_dims=None, input_dropout=0.5,
                 activation_fn: Union[str, nn.Module] = nn.ReLU(),
                 l1_weight_decay=None):
        """
        Attributes
        ---------
        encoder_dims  : str
            list of values that defines the structure of the network on the decoder side
        decoder_dims : str
            list of values that defines the structure of the network on the encoder side (Optional)
        input_dropout: float
            dropout value
        """
        super().__init__()

        # Reading Parameters #
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims if decoder_dims is not None else encoder_dims[::-1]
        assert self.encoder_dims[-1] == self.decoder_dims[0], \
            f"Latent dimensions of encoder and decoder networks do not match ({encoder_dims[-1]} vs {decoder_dims[0]})."
        assert self.encoder_dims[0] == self.decoder_dims[-1], \
            f"Input and output dimensions of encoder and decoder networks, respectively, " \
            f"do not match ({encoder_dims[0]} vs {decoder_dims[-1]})."

        self.latent_size = self.encoder_dims[-1]
        self.dropout = nn.Dropout(input_dropout)

        self.encoder = PolyLinear(self.encoder_dims, activation_fn, l1_weight_decay=l1_weight_decay)
        self.decoder = PolyLinear(self.decoder_dims, activation_fn, l1_weight_decay=l1_weight_decay)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        gain = nn.init.calculate_gain('tanh')
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight, gain)
            torch.nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        x = F.normalize(x, 2, 1)
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.decoder(x)

        return x
