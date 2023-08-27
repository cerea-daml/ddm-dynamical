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


class LinearScheduler(NoiseScheduler):
    def __init__(
            self,
            timesteps: int = 1000,
            beta_start: float = 0.0001,
            beta_end: float = 0.03,
    ):
        betas = torch.linspace(beta_start, beta_end, timesteps)
        super(LinearScheduler, self).__init__(betas=betas, timesteps=timesteps)
