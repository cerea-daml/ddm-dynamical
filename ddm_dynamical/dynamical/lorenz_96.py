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
from typing import Union

# External modules
import torch

# Internal modules


logger = logging.getLogger(__name__)


class Lorenz96(object):
    def __init__(self, forcing: Union[float, torch.Tensor] = 8.0) -> None:
        self.forcing = forcing

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        advection = (
                torch.roll(state, shifts=-1, dims=-1) -
                torch.roll(state, shifts=2, dims=-1)
        ) * torch.roll(state, shifts=1, dims=-1)
        dissipation = -state
        derivative = advection + dissipation + self.forcing
        return derivative
