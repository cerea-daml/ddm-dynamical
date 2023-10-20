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
from torch.func import grad

# Internal modules


logger = logging.getLogger(__name__)


class NoiseScheduler(torch.nn.Module):
    def __init__(
            self,
            gamma_min: float = -10,
            gamma_max: float = 10,
            normalize=True
    ):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.normalize = normalize
        self.eps = 1E-9

    def _estimate_gamma(self, timesteps: torch.Tensor) -> torch.Tensor:
        pass

    def normalize_gamma(self, gamma: torch.Tensor) -> torch.Tensor:
        gamma_0, gamma_1 = self._estimate_gamma(
            torch.tensor(
                [0., 1.], dtype=gamma.dtype, device=gamma.device
            )
        )
        return (gamma-gamma_1) / (gamma_0-gamma_1+self.eps)

    def get_gamma_deriv(self, timesteps: torch.Tensor) -> torch.Tensor:
        return grad(
            lambda x: self(x).sum()
        )(timesteps)

    def update(
            self,
            timesteps: torch.Tensor,
            target: torch.Tensor
    ) -> None:
        pass

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        gamma = self._estimate_gamma(timesteps)
        if self.normalize:
            gamma = self.gamma_max + (self.gamma_min-self.gamma_max) * \
                    self.normalize_gamma(gamma)
        return gamma
