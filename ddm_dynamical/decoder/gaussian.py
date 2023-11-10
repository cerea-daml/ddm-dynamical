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
from math import log, inf
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
            lower_bound: float = -inf,
            upper_bound: float = inf,
            std_shape: Tuple[int, ...] = (1, 1, 1),
            update_std: bool = False,
            eps: float = 1E-9
    ):
        super().__init__()
        self.mean = mean
        self.std = std
        self.logstd = torch.nn.Parameter(
            torch.ones(*std_shape) * log(std),
            requires_grad=~update_std
        )
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.eps = eps

    def forward(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        prediction = in_tensor * self.std + self.mean
        return prediction.clamp(min=self.lower_bound, max=self.upper_bound)

    def log_likelihood(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        dist = Normal(in_tensor, self.logstd.exp())
        return dist.log_prob(target)
