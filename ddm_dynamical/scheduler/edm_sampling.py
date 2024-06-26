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


class EDMSamplingScheduler(NoiseScheduler):
    def __init__(
            self,
            rho: float = 7,
            sigma_min: float = 0.002,
            sigma_max: float = 80,
            gamma_min: float = -10,
            gamma_max: float = 10,
            learnable: bool = False
    ):
        super().__init__(
            gamma_min=gamma_min, gamma_max=gamma_max, learnable=learnable
        )
        self.rho = rho
        self.inv_rho = 1 / rho
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def inverse_scheduler(
            self,
            gamma: torch.Tensor,
    ) -> torch.Tensor:
        numerator = (torch.exp(-gamma * 0.5 * self.inv_rho)
                     - self.sigma_max ** self.inv_rho)
        denom = self.sigma_min ** self.inv_rho - self.sigma_max  ** self.inv_rho
        return 1 - numerator / denom

    def get_density(self, gamma: torch.Tensor) -> torch.Tensor:
        num = torch.exp(-gamma*0.5*self.inv_rho)
        denom = (2*self.rho)*(
                self.sigma_max**self.inv_rho-self.sigma_min**self.inv_rho
        )
        return num / denom / self.get_time_scale()

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        truncated_time = self.truncate_time(timesteps)
        factor = self.sigma_max ** self.inv_rho \
                 + (1-truncated_time) \
                 * (self.sigma_min**self.inv_rho-self.sigma_max**self.inv_rho)
        return -2 * self.rho * torch.log(factor)
