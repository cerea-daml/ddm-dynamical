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
    def __init__(self, gamma_start: float = -5, gamma_end: float = 10):
        super().__init__()
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end

    def get_gamma(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.gamma_start + (self.gamma_end-self.gamma_start) * timestep
