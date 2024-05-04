#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 23/10/2023
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


class BinarizedScheduler(NoiseScheduler):
    def __init__(
            self,
            n_bins: int = 100,
            gamma_min: float = -10,
            gamma_max: float = 10,
            ema_rate: float = 0.999
    ):
        """
        Binarized noise scheduler as proposed in
        Kingma and Guo, `Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation`.
        """
        super().__init__(gamma_min=gamma_min, gamma_max=gamma_max)
        self.n_bins = n_bins
        self.ema_rate = ema_rate
        self.register_buffer(
            "bin_limits", torch.linspace(gamma_min, gamma_max, n_bins+1)
        )
        self.register_buffer("dx", self.bin_limits[1]-self.bin_limits[0])
        self.register_buffer(
            "bin_values", torch.ones(n_bins)
        )
        self.register_buffer(
            "bin_integral", torch.linspace(0, 1, n_bins+1)
        )

    def evaluate_integral(self, gamma: torch.Tensor) -> torch.Tensor:
        idx_left = self.bin_search(gamma, self.bin_limits)
        integral_left = self.bin_integral[idx_left]
        weight = self.bin_values[idx_left] / self.dx
        return integral_left + (gamma - self.bin_limits[idx_left]) * weight

    @property
    def normalization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        int_min = self.evaluate_integral(self.gamma_min)
        int_max = self.evaluate_integral(self.gamma_max)
        return int_min, int_max-int_min

    def bin_search(
            self, value: torch.Tensor, search_in: torch.Tensor
    ) -> torch.Tensor:
        return (
            torch.searchsorted(search_in, value, right=True)-1
        ).clamp(min=0, max=self.n_bins-1)

    def get_density(self, gamma: torch.Tensor) -> torch.Tensor:
        bin_num = self.bin_search(gamma, self.bin_limits)
        return self.bin_values[bin_num] / self.normalization[1]

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        time_shift, time_scale = self.normalization
        times_tilde = (1-timesteps) * time_scale + time_shift
        idx_left = self.bin_search(times_tilde, self.bin_integral)
        gamma_left = self.bin_limits[idx_left]
        delta_gamma = (times_tilde - self.bin_integral[idx_left]) * self.dx / (
                self.bin_integral[idx_left+1]-self.bin_integral[idx_left]
        )
        return gamma_left + delta_gamma

    @torch.no_grad()
    def update(
            self,
            gamma: torch.Tensor,
            target: torch.Tensor
    ) -> None:
        ## EMA update with counting number of indices
        # Get indices
        idx_bin = self.bin_search(gamma, self.bin_limits)

        # Count number of indices
        num_idx = torch.bincount(idx_bin, minlength=self.n_bins)

        # Factor for original value is ema_rate ^ number of appearances
        factors = self.ema_rate ** num_idx
        self.bin_values *= factors

        # Add the target values times a scattered rate taking the
        # power into account.
        scattered_factors = (1-factors[idx_bin]) / num_idx[idx_bin]
        self.bin_values.scatter_add_(
            dim=0, index=idx_bin, src=scattered_factors*target
        )

        # Update the integral values
        self.bin_integral[1:] = torch.cumsum(self.bin_values, dim=0)
