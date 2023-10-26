#!/bin/env python
# -*- coding: utf-8 -*-
#
#
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}


# System modules
import logging

# External modules
import torch
from torch.func import grad

# Internal modules


logger = logging.getLogger(__name__)


class NoiseScheduler(torch.nn.Module):
    def __init__(
            self,
            gamma_min: float = -10,
            gamma_max: float = 10,
    ):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.eps = 1E-9

    def normalize_gamma(self, gamma: torch.Tensor) -> torch.Tensor:
        return (gamma-self.gamma_min) / (self.gamma_max-self.gamma_min)

    def get_density(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = torch.nn.Parameter(timesteps)
        return -1 / (grad(
            lambda x: self(x).sum()
        )(timesteps) + self.eps)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        pass

    def update(
            self,
            gamma: torch.Tensor,
            target: torch.Tensor
    ) -> None:
        pass
