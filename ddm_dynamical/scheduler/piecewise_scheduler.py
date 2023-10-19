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
            "support_values", torch.ones(n_support)
        )
        self.register_buffer(
            "integral_deriv", torch.zeros(n_support)
        )
        self.dt = 1./(n_support-1)
        self.lr = lr
        self.update_integral = True

    @property
    def integral(self) -> torch.Tensor:
        if self.update_integral:
            self.integral_deriv = torch.cat((
                torch.zeros(1, device=self.support_t.device),
                torch.cumulative_trapezoid(self.support_deriv, self.support_t)
            ), dim=0)
            self.update_integral = False
        return self.integral_deriv

    @property
    def support_deriv(self) -> torch.Tensor:
        return 1/self.support_values

    def interp_deriv(
            self,
            idx_left: torch.Tensor,
            timesteps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_left = self.support_t[idx_left]
        val_left = self.support_deriv[idx_left]
        idx_right = (idx_left+1).clamp(min=0, max=len(self.support_t)-1)
        t_right = self.support_t[idx_right] + 1E-9
        val_right = self.support_deriv[idx_right]
        total_dist = t_right-t_left
        weight_left = (t_right - timesteps) / total_dist
        weight_right = (timesteps - t_left) / total_dist
        deriv = weight_left * val_left + weight_right * val_right
        return deriv, weight_left, weight_right

    def update(
            self,
            timesteps: torch.Tensor,
            target: torch.Tensor
    ) -> None:
        idx_left = torch.searchsorted(self.support_t, timesteps, right=True) - 1
        deriv, weight_left, weight_right = self.interp_deriv(
            idx_left, timesteps
        )
        diff = 1/deriv-target
        self.support_values[idx_left] -= self.lr * weight_left * diff
        self.support_values[idx_left+1] -= self.lr * weight_right * diff
        self.update_integral = True

    def normalize_gamma(self, gamma: torch.Tensor) -> torch.Tensor:
        return gamma / self.integral[-1]

    def get_gamma_deriv(self, timesteps: torch.Tensor) -> torch.Tensor:
        idx_left = torch.searchsorted(self.support_t, timesteps, right=True) - 1
        return self.interp_deriv(
            idx_left, timesteps
        )[0]

    def _estimate_gamma(self, timesteps: torch.Tensor) -> torch.Tensor:
        idx_left = torch.searchsorted(self.support_t, timesteps, right=True) - 1

        integrals_left = self.integral[idx_left]
        dt = timesteps-self.support_t[idx_left]
        interp_value = self.get_gamma_deriv(timesteps)
        gamma = integrals_left + (
                interp_value + self.support_deriv[idx_left]
        ) * 0.5 * dt
        return gamma
