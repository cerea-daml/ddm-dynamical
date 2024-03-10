#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 10/03/2024
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2024}  {Tobias Sebastian Finn}

# System modules
import logging

import torch.nn

# External modules

# Internal modules
from .noise_scheduler import NoiseScheduler

main_logger = logging.getLogger(__name__)


class FlowOTScheduler(NoiseScheduler):
    def get_density(self, gamma: torch.Tensor) -> torch.Tensor:
        # Eq. 68 in Kingma and Gai, 2023
        return 1/torch.cosh(gamma / 4)**2 / 8

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Eq. 63 in Kingma and Gao, 2023
        return 2 * torch.log((1-timesteps) / timesteps)
