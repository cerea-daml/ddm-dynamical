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


logger = logging.getLogger(__name__)


class NoiseScheduler(torch.nn.Module):
    def __init__(
            self,
            betas: torch.Tensor,
            timesteps: int = 1000,
    ):
        super().__init__()
        assert len(betas) == timesteps, "The betas don't specify all timesteps!"
        alphas_prime = 1-betas
        alphas = torch.cumprod(alphas_prime, dim=0)
        alphas = torch.cat((torch.ones_like(alphas[-1:]), alphas), dim=0)
        sigmas = (1-alphas).sqrt()
        self.register_buffer("alphas", alphas)
        self.register_buffer("sigmas", sigmas)
        self.timesteps = timesteps

    def get_alpha(self, timestep: int) -> float:
        return self.alphas[timestep]

    def get_sigma(self, timestep: int) -> float:
        return self.sigmas[timestep]
