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


class CosineScheduler(NoiseScheduler):
    def __init__(
            self,
            timesteps: int = 1000,
            shift: float = 0.008,
            max_beta: float = 0.999
    ):
        time_range = (torch.linspace(0, 1, timesteps+1) + shift) / (1 + shift)
        alphas = torch.cos(time_range * torch.pi * 0.5).pow(2)
        betas = torch.minimum(1-alphas[1:]/alphas[:-1], torch.tensor(max_beta))
        super().__init__(betas=betas, timesteps=timesteps)
