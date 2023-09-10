#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 27/08/2023
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Dict

# External modules
from torch.utils.data import Dataset
import torch

# Internal modules

main_logger = logging.getLogger(__name__)


_mean = 2.34
_std = 3.64


__all__ = [
    "StateDataset"
]


class StateDataset(Dataset):
    def __init__(self, in_tensor: torch.Tensor):
        super().__init__()
        self.state_tensor = in_tensor.view(-1, in_tensor.size(-1))

    def __len__(self) -> int:
        return self.state_tensor.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "data": (self.state_tensor[idx]-_mean) / _std,
            "mask": torch.randint(2, (40,), dtype=torch.float)
        }
