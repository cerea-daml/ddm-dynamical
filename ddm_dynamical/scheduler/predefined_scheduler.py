#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 23/10/2023
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


class PredefinedScheduler(NoiseScheduler):
    def __init__(
            self,
            time_gammas: torch.Tensor,
            gamma_min: float = -15,
            gamma_max: float = 10,
    ):
        """
        To set a predefined scheduler with equally distant time stepping from
        gamma min to gamma max.
        """
        super().__init__(
            gamma_min=gamma_min, gamma_max=gamma_max, learnable=False
        )
        self.register_buffer("time_gammas", time_gammas)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        assert timesteps.size(0) == self.time_gammas.size(0)
        return self.time_gammas
