#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 07/09/2023
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Union

# External modules
import torch

# Internal modules

main_logger = logging.getLogger(__name__)


class GaussianEncoder(torch.nn.Module):
    def __init__(
            self,
            mean: Union[float, torch.Tensor] = 0.,
            std: Union[float, torch.Tensor] = 1.,
            eps: float = 1E-9
    ):
        super().__init__()
        self.mean = mean
        self.std = std
        self.eps = eps

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        return (in_tensor-self.mean) / (self.std + self.eps)
