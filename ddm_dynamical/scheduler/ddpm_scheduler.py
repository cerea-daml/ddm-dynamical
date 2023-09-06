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
    def __init__(self):
        """
        Noise scheduling as proposed by Ho et al. 2020 and
        approximated in Kingma et al. 2021.
        """
        super().__init__()
        self.shift = 1E-4
        self.scale = 10.

    def get_gamma(self, timestep: torch.Tensor) -> torch.Tensor:
        return torch.log(
            torch.expm1(self.shift + self.scale * timestep.pow(2))
        )
