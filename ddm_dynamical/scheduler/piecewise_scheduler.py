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
        self.dt = 1./(n_support-1)
        self.lr = lr

    @property
    def support_deriv(self) -> torch.Tensor:
        return 1/self.support_values

    def interp_deriv(
            self,
            idx_right: torch.Tensor,
            timesteps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        val_left = self.support_deriv[idx_right-1]
        val_right = self.support_deriv[idx_right]
        t_left = self.support_t[idx_right-1]
        t_right = self.support_t[idx_right]
        weight_left = (t_right - timesteps) / self.dt
        weight_right = (timesteps - t_left) / self.dt
        deriv = weight_left * val_left + weight_right * val_right
        return deriv, weight_left, weight_right

    def update(
            self,
            timesteps: torch.Tensor,
            target: torch.Tensor
    ) -> None:
        idx_right = torch.searchsorted(self.support_t, timesteps, right=True)
        deriv, weight_left, weight_right = self.interp_deriv(
            idx_right, timesteps
        )
        diff = 1/deriv-target
        self.support_values[idx_right-1] -= self.lr * weight_left * diff
        self.support_values[idx_right] -= self.lr * weight_right * diff

    def normalize_gamma(self, gamma: torch.Tensor) -> torch.Tensor:
        return gamma / torch.trapezoid(self.support_deriv, self.support_t)

    def get_gamma_deriv(self, timesteps: torch.Tensor) -> torch.Tensor:
        idx_right = torch.searchsorted(self.support_t, timesteps, right=True)
        return self.interp_deriv(
            idx_right, timesteps
        )[0]

    def _estimate_gamma(self, timesteps: torch.Tensor) -> torch.Tensor:
        idx_right = torch.searchsorted(self.support_t, timesteps, right=True)
        integrals = torch.cat((
            torch.zeros(1, device=self.support_t.device),
            torch.cumulative_trapezoid(self.support_deriv, self.support_t)
        ), dim=0)
        integrals_left = integrals[idx_right-1]
        dt = timesteps-self.support_t[idx_right-1]
        interp_value = self.get_gamma_deriv(timesteps)
        gamma = integrals_left + (
                interp_value + self.support_deriv[idx_right-1]
        ) * 0.5 * dt
        return gamma
