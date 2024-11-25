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
        self._time_scale = None
        self._time_shift = None
        self.learnable = learnable
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

    def get_time_scale(self) -> torch.Tensor:
        if self.learnable is True or self._time_scale is None:
            t0 = self.inverse_scheduler(self.gamma_max)
            t1 = self.inverse_scheduler(self.gamma_min)
            self._time_scale = t1 - t0
        return self._time_scale

    def get_time_shift(self) -> torch.Tensor:
        if self.learnable is True or self._time_shift is None:
            self._time_shift = self.inverse_scheduler(self.gamma_max)
        return self._time_shift

    def truncate_time(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.get_time_shift() + self.get_time_scale() * timesteps

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
