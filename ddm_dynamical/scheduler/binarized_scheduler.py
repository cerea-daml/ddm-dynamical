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
            ema_rate: float = 0.001
    ):
        """
        Binarized noise scheduler as proposed in
        Kingma and Guo, `Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation`.
        """
        super().__init__(gamma_min=gamma_min, gamma_max=gamma_max)
        self.register_buffer(
            "bin_limits", torch.linspace(0, 1, n_bins+1)
        )
        self.register_buffer(
            "bin_values", torch.ones(n_bins)
        )
        self.register_buffer(
            "bin_gamma", 1-self.bin_limits
        )
        self.n_bins = n_bins
        self.ema_rate = ema_rate

    def get_left(self, timesteps: torch.Tensor) -> torch.Tensor:
        return (
            torch.searchsorted(self.bin_limits, timesteps, right=True)-1
        ).clamp(min=0, max=len(self.bin_limits)-2)

    def get_gamma_deriv(self, timesteps: torch.Tensor) -> torch.Tensor:
        idx_left = self.get_left(timesteps)
        return -1 / self.bin_values[idx_left]

    def get_normalized_gamma(self, timesteps: torch.Tensor) -> torch.Tensor:
        idx_left = self.get_left(timesteps)
        gamma_left = self.bin_gamma[idx_left]
        gamma_right = self.bin_gamma[idx_left+1]
        delta_t = (timesteps-self.bin_limits[idx_left]) / self.n_bins
        delta_gamma = delta_t * (gamma_right - gamma_left)
        gamma = gamma_left + delta_gamma
        return gamma

    def update(
            self,
            timesteps: torch.Tensor,
            target: torch.Tensor
    ) -> None:
        ## EMA update with counting number of indices
        # Get indices
        idx_left = self.get_left(timesteps)

        # Count number of indices
        num_idx = torch.bincount(idx_left, minlength=self.bin_values.size(0))

        # Factor for original value is (1-rate) ^ number of appearances
        factors = (1 - self.ema_rate) ** num_idx
        self.bin_values *= factors

        # Add the target values times a scattered rate taking the
        # power into account.
        scattered_factors = (1-factors[idx_left]) / num_idx[idx_left]
        self.bin_values.scatter_add_(
            dim=0, index=idx_left, src=scattered_factors*target
        )

        # Update gamma at bin borders
        new_gamma = torch.cumsum(-1/self.bin_values, dim=0) / self.n_bins
        self.bin_gamma[1:] = (new_gamma[-1]-new_gamma)/new_gamma[-1]
