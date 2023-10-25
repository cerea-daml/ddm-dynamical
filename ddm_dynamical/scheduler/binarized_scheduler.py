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
            values=None,
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
            "_bin_values", torch.ones(n_bins)
        )
        self.register_buffer(
            "_bin_times", torch.linspace(1, 0, n_bins+1)
        )
        self.bin_values = values


    @property
    def bin_times(self) -> torch.Tensor:
        return self._bin_times

    @property
    def bin_values(self) -> torch.Tensor:
        return self._bin_values

    @bin_values.setter
    def bin_values(self, new_values) -> torch.Tensor:
        if new_values is None:
            new_values = torch.ones(self.n_bins)
        elif not isinstance(new_values, (torch.Tensor, torch.nn.Parameter)):
            new_values = torch.tensor(new_values)
        self._bin_values = new_values

    def _update_times(self):
        bin_times = torch.cumsum(
            torch.flip(self.bin_values, dims=(0,)), dim=0
        )
        bin_times = torch.cat(
            (torch.zeros(1, device=bin_times.device), bin_times),
            dim=0
        )
        self._bin_times = torch.flip(bin_times / bin_times[-1], dims=(0,))

    def get_left(self, keys: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        return (
            torch.searchsorted(keys, query, right=True)-1
        ).clamp(min=0, max=len(keys)-1)

    def get_density(self, gamma: torch.Tensor) -> torch.Tensor:
        return self.bin_values[self.get_left(self.bin_limits, gamma)]

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        idx_left = self.get_left(self.bin_times, timesteps)
        dgamma_dt = (
                self.bin_limits[idx_left+1]-self.bin_limits[idx_left]
        ) / (
                self.bin_times[idx_left+1] - self.bin_times[idx_left]
        )
        delta_gamma = dgamma_dt * (timesteps - self.bin_times[idx_left])
        return self.bin_limits[idx_left] + delta_gamma

    def update(
            self,
            gamma: torch.Tensor,
            target: torch.Tensor
    ) -> None:
        ## EMA update with counting number of indices
        # Get indices
        idx_left = self.get_left(self.bin_limits, gamma)

        # Count number of indices
        num_idx = torch.bincount(idx_left, minlength=self.n_bins)

        # Factor for original value is (1-rate) ^ number of appearances
        factors = (1 - self.ema_rate) ** num_idx
        self.bin_values *= factors

        # Add the target values times a scattered rate taking the
        # power into account.
        scattered_factors = (1-factors[idx_left]) / num_idx[idx_left]
        self.bin_values.scatter_add_(
            dim=0, index=idx_left, src=scattered_factors*target
        )
        self._update_times()

