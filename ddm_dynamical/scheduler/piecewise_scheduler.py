#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 19/10/2023
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Tuple

# External modules
import torch

# Internal modules
from .noise_scheduler import NoiseScheduler

main_logger = logging.getLogger(__name__)


class PiecewiseScheduler(NoiseScheduler):
    def __init__(
            self,
            n_support: int = 101,
            gamma_min: float = -10,
            gamma_max: float = 10,
            normalize: bool = True,
            lr: float = 0.001
    ):
        super().__init__(
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            normalize=normalize
        )
        self.register_buffer(
            "support_t", torch.linspace(0, 1, n_support)
        )
        self.register_buffer(
            "support_density", torch.ones(n_support)
        )
        self.register_buffer(
            "integral_deriv", torch.zeros(n_support)
        )
        self.dt = 1./(n_support-1)
        self.lr = lr
        self.update_integral = True

    @property
    def support_deriv(self) -> torch.Tensor:
        return -1/self.support_density

    @property
    def integral(self) -> torch.Tensor:
        if self.update_integral:
            self.integral_deriv = torch.cat((
                torch.ones(1, device=self.support_t.device),
                torch.cumulative_trapezoid(self.support_deriv, self.support_t)+1
            ), dim=0)
            self.update_integral = False
        return self.integral_deriv

    def get_left(self, timesteps: torch.Tensor) -> torch.Tensor:
        return (
            torch.searchsorted(self.support_t, timesteps, right=True)-1
        ).clamp(min=0, max=len(self.support_t)-2)

    def interp_density(
            self,
            idx_left: torch.Tensor,
            timesteps: torch.Tensor
    ) -> torch.Tensor:
        scale = (self.support_density[1:]-self.support_density[:-1])/self.dt
        shift = self.support_density[:-1]-self.support_t[:-1] * scale
        density = timesteps * scale[idx_left] + shift[idx_left]
        return density

    def update(
            self,
            timesteps: torch.Tensor,
            target: torch.Tensor
    ) -> None:
        idx_left = self.get_left(timesteps)
        weight_right = (timesteps - self.support_t[idx_left]) / self.dt
        weight_left = 1-weight_right
        density = self.interp_density(
            idx_left, timesteps
        )
        diff = density-target
        self.support_density[idx_left] -= self.lr * weight_left * diff
        self.support_density[idx_left+1] -= self.lr * weight_right * diff
        self.update_integral = True

    def normalize_gamma(self, gamma: torch.Tensor) -> torch.Tensor:
        return gamma / self.integral[0]

    def get_gamma_deriv(self, timesteps: torch.Tensor) -> torch.Tensor:
        idx_left = self.get_left(timesteps)
        return -1 / self.interp_density(
            idx_left, timesteps
        )

    def _estimate_gamma(self, timesteps: torch.Tensor) -> torch.Tensor:
        idx_left = self.get_left(timesteps)
        integrals_left = self.integral[idx_left]
        dt = timesteps-self.support_t[idx_left]
        mid_point = (
            self.support_deriv[idx_left] +
            -1/self.interp_density(idx_left, timesteps)
        ) * 0.5
        gamma = integrals_left + mid_point * dt
        return gamma
