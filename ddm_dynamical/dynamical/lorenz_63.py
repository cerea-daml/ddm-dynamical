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
from typing import Union, Tuple

# External modules
import torch

# Internal modules


logger = logging.getLogger(__name__)


class Lorenz63(torch.nn.Module):
    def __init__(
            self,
            sigma: float = 10.,
            rho: float = 28,
            beta: float = 8/3,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x, y, z = state.chunk(3, dim=-1)
        convection = self.sigma * (y-x)
        temp_h = x * (self.rho - z) - y
        temp_z = x * y - self.beta * z
        derivative = torch.concat([convection, temp_h, temp_z], dim=-1)
        return derivative
