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
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return (1-timesteps) * (self.gamma_max-self.gamma_min) + self.gamma_min
