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
        self.register_buffer(
            "bin_values", torch.ones(n_bins)
        )
        self.register_buffer(
            "_bin_times", torch.linspace(1, 0, n_bins+1)
        )
        self.pdf_norm = 1.

    @property
    def bin_times(self) -> torch.Tensor:
        return self._bin_times

    def _update_times(self):
        bin_times = torch.cumsum(-self.bin_values, dim=0)
        bin_times *= (self.gamma_max-self.gamma_min)/self.n_bins
        bin_times = torch.cat(
            (torch.zeros_like(bin_times[[0]]), bin_times),
            dim=0
        )
        self.pdf_norm = bin_times[-1].abs()
        self._bin_times = bin_times / self.pdf_norm + 1

    def get_bin_num(self, gamma: torch.Tensor) -> torch.Tensor:
        return (
            torch.searchsorted(self.bin_limits, gamma, right=True)-1
        ).clamp(min=0, max=self.n_bins-1)

    def get_left_time(
            self, timesteps: torch.Tensor
    ) -> torch.Tensor:
        ordered_left = (
            torch.searchsorted(-self.bin_times, -timesteps, right=True,)-1
        ).clamp(min=0, max=self.n_bins-1)
        return ordered_left

    def get_density(self, gamma: torch.Tensor) -> torch.Tensor:
        bin_num = self.get_bin_num(gamma)
        return self.bin_values[bin_num] / self.pdf_norm

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        idx_left = self.get_left_time(timesteps)
        gamma_left = self.bin_limits[idx_left]
        delta_gamma = (
                self.bin_limits[idx_left+1]-gamma_left
        ) / (
                self.bin_times[idx_left+1]-self.bin_times[idx_left]
        ) * (timesteps - self.bin_times[idx_left])
        return gamma_left + delta_gamma

    def update(
            self,
            gamma: torch.Tensor,
            target: torch.Tensor
    ) -> None:
        ## EMA update with counting number of indices
        # Get indices
        idx_bin = self.get_bin_num(gamma)

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
        self._update_times()

