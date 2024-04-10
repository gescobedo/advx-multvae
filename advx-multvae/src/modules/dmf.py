# Deep matrix factorization implementation based on
# Xue et al. - 2017 - Deep Matrix Factorization Models for Recommender Systems
# https://www.ijcai.org/proceedings/2017/447
from typing import Union

import torch
from torch import nn

from src.modules.gradient_reversal import GradientReversalLayer
from src.modules.polylinear import PolyLinear
from src.modules.sparse_dropout import SparseDropout


class DMF(nn.Module):
    def __init__(self, user_dims, item_dims, input_dropout=0.5, activation_fn: Union[str, nn.Module] = nn.ReLU(),
                 **kwargs):
        """
        @param user_dims: list of values that defines the structure of the user encoding network
        @param item_dims: list of values that defines the structure of the item encoding network
        @param input_dropout: dropout value applied on both, user and item input vectors
        @param activation_fn: activation function between layers
        """
        super().__init__()

        self.user_dims = user_dims
        self.item_dims = item_dims
        if not self.user_dims[-1] == self.item_dims[-1]:
            raise ValueError("User and item dims need to supply the same last value")

        # Dropout is not supported for sparse tensors, therefore, to ease the usage,
        # we apply normal or sparse dropout, depending on input type
        self.dropout = nn.Dropout(input_dropout)
        self.sparse_dropout = SparseDropout(input_dropout)

        self.user_encoder = PolyLinear(self.user_dims, activation_fn)
        self.item_encoder = PolyLinear(self.item_dims, activation_fn)

        self.sim_net = PolyLinear([self.user_dims[-1] + self.item_dims[-1], self.user_dims[-1], 1], activation_fn)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        gain = nn.init.calculate_gain('tanh')
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight, gain)
            torch.nn.init.constant_(m.bias, 0.01)

    def _apply_dropout(self, x):
        if x.is_sparse:
            return self.sparse_dropout(x)
        return self.dropout(x)

    def similarity(self, first, second):
        return (first @ second.T)
        # return torch.sigmoid(self.sim_net(torch.cat([first, second], dim=-1))).flatten()

    def forward(self, user_data, item_data):

        user_data = self._apply_dropout(user_data)
        item_data = self._apply_dropout(item_data)

        p = self.user_encoder(user_data)
        q = self.item_encoder(item_data)

        return self.similarity(p, q).diag()


class DMFAdv(DMF):
    def __init__(self, user_dims, item_dims, grad_scaling=1, use_adv=False, adv_dims=None,
                 input_dropout=0.5, activation_fn: Union[str, nn.Module] = nn.ReLU(),
                 activation_fn_adv: Union[str, nn.Module] = nn.ReLU(), **kwargs):
        """
        @param user_dims: list of values that defines the structure of the user encoding network
        @param item_dims: list of values that defines the structure of the item encoding network
        @param adv_dims: definition of adversarial network, which will be placed on top of the
                         user encoding. Dimension of user encoding will be automatically inferred.
        @param use_adv: whether data should be passed through adversarial network
        @param input_dropout: dropout value applied on both, user and item input vectors
        @param activation_fn: activation function between layers
        @param activation_fn_adv: activation function between layers of adversarial network
        """
        super().__init__(user_dims, item_dims, input_dropout, activation_fn, **kwargs)

        self.adv_dims = adv_dims
        self.use_adv = use_adv

        self.gru = GradientReversalLayer(grad_scaling)
        self.adv = PolyLinear([self.user_dims[-1]] + self.adv_dims, activation_fn_adv)
        self.apply(self._init_weights)

    def forward(self, user_data, item_data):
        user_data = self._apply_dropout(user_data)
        item_data = self._apply_dropout(item_data)

        user_enc = self.user_encoder(user_data)
        item_enc = self.item_encoder(item_data)

        a = None
        if self.use_adv:
            a = self.adv(self.gru(user_enc))

        return self.similarity(user_enc, item_enc).diag(), {}, a

    def deactivate_adv(self):
        self.use_adv = False
        self.gru = None
        self.adv = None

    @classmethod
    def from_config(cls, model_config, n_users, n_items, adv_config):
        user_dims = model_config.pop("user_dims", None) or model_config["dims"]
        item_dims = model_config.pop("item_dims", None) or model_config["dims"]
        model_config.pop("dims", None)

        # grid search over different user and item dims is difficult as the latent
        # size must match, therefore enable another parameter to specify latent size
        if ldim := model_config.pop("latent_dim", None):
            user_dims += [ldim]
            item_dims += [ldim]

        user_dims = [n_items] + user_dims
        item_dims = [n_users] + item_dims

        if adv_config is None:
            return DMFAdv(user_dims, item_dims, **model_config, use_adv=False)

        return DMFAdv(user_dims, item_dims, **model_config,
                      use_adv=adv_config["perform_adv_training"],
                      adv_dims=adv_config["dims"],
                      activation_fn_adv=adv_config["activation_fn"])
