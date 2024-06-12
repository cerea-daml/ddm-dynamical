#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 26/10/2023
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging

# External modules
import torch

# Internal modules
from .noise_scheduler import NoiseScheduler

main_logger = logging.getLogger(__name__)


class CosineScheduler(NoiseScheduler):
    def __init__(
            self,
            shift: float = 0,
            gamma_min: float = -10,
            gamma_max: float = 10,
            learnable: bool = False
    ):
        super().__init__(
            gamma_min=gamma_min, gamma_max=gamma_max, learnable=learnable
        )
        self.shift = shift

    def inverse_schedule(self, gamma) -> torch.Tensor:
        factor = torch.exp(-torch.tensor(gamma) * 0.5 - self.shift)
        return 2 / torch.pi * torch.arctan(factor)

    def get_density(self, gamma: torch.Tensor) -> torch.Tensor:
        return 1/torch.cosh(gamma * 0.5 - self.shift) / (2 * torch.pi) \
            / self.get_time_scale()

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        truncated_time = self.truncate_time(timesteps)
        gamma = -2 * torch.log(torch.tan(torch.pi*truncated_time*0.5))
        return gamma + 2 * self.shift
