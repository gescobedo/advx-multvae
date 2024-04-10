from enum import Enum
from typing import Iterable, Optional

import torch
import torch.nn as nn


class ParallelMode(Enum):
    MultiInMultiOut = 0
    SingleInMultiOut = 1


class Parallel(nn.ModuleList):
    def __init__(self, modules: Optional[Iterable[nn.Module]] = None,
                 parallel_mode: ParallelMode = ParallelMode.SingleInMultiOut):
        super().__init__(modules)
        self.parallel_mode = parallel_mode

    def forward(self, x):
        if self.parallel_mode == ParallelMode.MultiInMultiOut:
            if len(x) != len(self):
                raise AttributeError(f"Input must be of same size as number of modules ({len(x)} vs {len(self)})")
            module_inputs = x
        else:
            module_inputs = (x,) * len(self)

        futures = [torch.jit.fork(module, model_input) for module, model_input in zip(self, module_inputs)]
        return [torch.jit.wait(future) for future in futures]
