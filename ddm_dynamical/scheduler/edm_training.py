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


class EDMTrainingScheduler(NoiseScheduler):
    def __init__(
            self,
            gamma_mean: float = -3.82,
            gamma_std: float = 2.,
            gamma_min: float = -10,
            gamma_max: float = 10,
            learnable: bool = False
    ):
        super().__init__(
            gamma_min=gamma_min, gamma_max=gamma_max, learnable=learnable
        )
        self.register_buffer("gamma_mean", torch.tensor(gamma_mean))
        self.register_buffer("gamma_std", torch.tensor(gamma_std))
        self.density_dist = torch.distributions.Normal(
            self.gamma_mean, self.gamma_std
        )

    def inverse_scheduler(
            self,
            gamma,
    ) -> torch.Tensor:
        return self.density_dist.cdf(-torch.tensor(gamma))

    def get_density(self, gamma: torch.Tensor) -> torch.Tensor:
        density = self.density_dist.log_prob(gamma).exp()
        return density / self.get_time_scale()

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        truncated_time = self.truncate_time(timesteps)
        return -self.density_dist.icdf(truncated_time)
