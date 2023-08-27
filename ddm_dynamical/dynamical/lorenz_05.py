#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 04/01/2023
# Created for 2022_ddim_for_attractors
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
from .lorenz_96 import Lorenz96

main_logger = logging.getLogger(__name__)


class Lorenz05(Lorenz96):
    def __init__(
            self,
            forcing: Union[float, torch.Tensor] = 10.0,
            n_fast: int = 10,
            h_coupling: float = 1.,
            c_time_scale: float = 10.,
            b_space_scale: float = 10.
    ) -> None:
        super().__init__(forcing=forcing)
        self.n_fast = n_fast
        self.h_coupling = h_coupling
        self.c_time_scale = c_time_scale
        self.b_space_scale = b_space_scale

    @property
    def coupling_ratio(self):
        return self.h_coupling * self.c_time_scale / self.b_space_scale

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        n_grid = state.shape[-1] // (self.n_fast+1)
        slow_state = state[..., :n_grid]
        fast_state = state[..., n_grid:]

        slow_adv = (
                torch.roll(slow_state, shifts=-1, dims=-1) -
                torch.roll(slow_state, shifts=2, dims=-1)
        ) * torch.roll(slow_state, shifts=1, dims=-1)
        slow_diss = - slow_state
        fast_to_slow = - self.coupling_ratio * fast_state.view(
            *fast_state.shape[:-1], -1, self.n_fast
        ).sum(dim=-1)
        slow_deriv = slow_adv + slow_diss + self.forcing + fast_to_slow

        fast_adv = self.c_time_scale * self.b_space_scale * (
                torch.roll(fast_state, shifts=1, dims=-1) -
                torch.roll(fast_state, shifts=-2, dims=-1)
        ) * torch.roll(fast_state, shifts=-1, dims=-1)
        fast_diss = - self.c_time_scale * fast_state
        slow_to_fast = self.coupling_ratio * slow_state.repeat_interleave(
            self.n_fast, dim=-1
        )
        fast_deriv = fast_adv + fast_diss + slow_to_fast

        derivative = torch.cat((slow_deriv, fast_deriv), dim=-1)
        return derivative
