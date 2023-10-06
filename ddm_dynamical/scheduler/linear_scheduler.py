#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 06/09/2023
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


class LinearScheduler(NoiseScheduler):
    def __init__(self, gamma_min: float = -10, gamma_max: float = 10,):
        super().__init__(gamma_min=gamma_min, gamma_max=gamma_max)

    def get_normalized_gamma(self, gamma: torch.Tensor) -> torch.Tensor:
        return gamma / (self.gamma_max-self.gamma_min+self.eps) - self.gamma_min

    def _estimate_gamma(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.gamma_min + (self.gamma_max-self.gamma_min) * timesteps

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self._estimate_gamma(timesteps)
