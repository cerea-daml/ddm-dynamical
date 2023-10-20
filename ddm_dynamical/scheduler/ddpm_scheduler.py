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

# Internal modules
from .noise_scheduler import NoiseScheduler


logger = logging.getLogger(__name__)


class DDPMScheduler(NoiseScheduler):
    def __init__(
            self,
            shift: float = 1E-4,
            scale: float = 10.,
            gamma_min: float = -10,
            gamma_max: float = 10,
    ):
        """
        Noise scheduling as proposed by Ho et al. 2020 and
        approximated in Kingma et al. 2021.
        """
        super().__init__(
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        self.shift = shift
        self.scale = scale

    def _estimate_gamma(self, timesteps: torch.Tensor) -> torch.Tensor:
        return -torch.log(
            torch.expm1(self.shift + self.scale * timesteps.pow(2))
        )