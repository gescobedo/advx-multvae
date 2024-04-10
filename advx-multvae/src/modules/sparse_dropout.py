import torch
from torch import nn


class SparseDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(SparseDropout, self).__init__()

        if not (0 <= p < 1):
            raise ValueError(f"Dropout probability has to be between 0 and 1, but got {p}")

        if inplace:
            raise NotImplementedError("Inplace operation is not yet supported")

        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return 'p={}, inplace={}'.format(self.p, self.inplace)

    def forward(self, x):
        mask = torch.rand(size=(x._nnz(),)) > self.p

        return torch.sparse_coo_tensor(indices=x._indices()[..., mask],
                                       values=x._values()[..., mask],
                                       size=x.shape
                                       )
