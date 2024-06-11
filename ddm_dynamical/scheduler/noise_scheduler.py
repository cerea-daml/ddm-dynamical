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
            gamma_min: float = -10,
            gamma_max: float = 10,
            learnable: bool = False
    ):
        super().__init__()
        self.register_parameter(
            "gamma_min",
            torch.nn.Parameter(
                torch.tensor(gamma_min), requires_grad=learnable
            )
        )
        self.register_parameter(
            "gamma_max",
            torch.nn.Parameter(
                torch.tensor(gamma_max), requires_grad=learnable
            )
        )
        self.eps = 1E-9

    @property
    def time_scale(self) -> torch.Tensor:
        t0 = self.inverse_schedule(self.gamma_max)
        t1 = self.inverse_schedule(self.gamma_min)
        return t1-t0

    @property
    def time_shift(self) -> torch.Tensor:
        return self.inverse_schedule(self.gamma_max)

    def get_density(self, gamma: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        pass

    def update(
            self,
            gamma: torch.Tensor,
            target: torch.Tensor
    ) -> None:
        pass
