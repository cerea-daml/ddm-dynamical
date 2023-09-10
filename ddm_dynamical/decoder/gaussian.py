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
from typing import Union, Tuple

# External modules
import torch
import torch.nn
from torch.distributions import Normal

# Internal modules

main_logger = logging.getLogger(__name__)


class GaussianDecoder(torch.nn.Module):
    def __init__(
            self,
            mean: Union[float, torch.Tensor] = 0.,
            std: Union[float, torch.Tensor] = 1.,
            data_shape: Tuple[int, ] = (1, ),
            eps: float = 1E-9
    ):
        super().__init__()
        self.mean = mean
        self.std = std
        self.logvar = torch.nn.Parameter(torch.zeros(*data_shape))
        self.eps = eps

    def forward(
            self, in_tensor: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        return in_tensor * self.std + self.mean

    def log_likelihood(
            self,
            in_tensor: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        dist = Normal(target, (self.logvar*0.5).exp())
        return dist.log_prob(in_tensor)
