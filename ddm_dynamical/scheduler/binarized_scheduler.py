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
            ema_rate: float = 0.999,
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
        self.register_buffer("bin_values", torch.ones(n_bins))
        self.register_buffer("bin_integral", torch.arange(n_bins+1))

    def evaluate_integral(self, gamma: torch.Tensor) -> torch.Tensor:
        bin_num = self.bin_search(gamma, self.bin_limits)
        # Get the integral left from given gamma
        integral_left = self.bin_integral[bin_num]
        # Change in the integral per gamma
        weight = self.bin_values[bin_num] / self.dx
        # Integral value = left + additional gamma
        return integral_left + (gamma - self.bin_limits[bin_num]) * weight

    @property
    def normalization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Estimate normalization values such that gamma_min=0 and gamma_max=1
        int_min = self.evaluate_integral(self.gamma_min)
        int_max = self.evaluate_integral(self.gamma_max)
        return int_min, int_max-int_min

    def bin_search(
            self, value: torch.Tensor, search_in: torch.Tensor
    ) -> torch.Tensor:
        # Search for the bin number of a given ˙value˙ in a given `search_in`
        # tensor.
        return (
            torch.searchsorted(search_in, value, right=True)-1
        ).clamp(min=0, max=self.n_bins-1)

    def get_density(self, gamma: torch.Tensor) -> torch.Tensor:
        # Get the probability density for a given `gamma` value. Given as local
        # slope of the CDF, the density corresponds to the bin value scaled by
        # the scale of the whole integral (such that the CDF = 1).
        bin_num = self.bin_search(gamma, self.bin_limits)
        return self.bin_values[bin_num] / self.normalization[1]

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Do a linear interpolation to map from `timesteps` to gamma value.
        time_shift, time_scale = self.normalization
        # The denormalized time steps. Integral goes from gamma_min to
        # gamma_max, so we need inverse value.
        times_tilde = (1-timesteps) * time_scale + time_shift
        bin_num = self.bin_search(times_tilde, self.bin_integral)
        gamma_left = self.bin_limits[bin_num]
        weight = self.dx / self.bin_values[bin_num]
        return gamma_left + weight * (times_tilde - self.bin_integral[bin_num])

    @torch.no_grad()
    def update(
            self,
            gamma: torch.Tensor,
            target: torch.Tensor
    ) -> None:
        ## EMA update with counting number of indices
        # Get indices
        bin_num = self.bin_search(gamma, self.bin_limits)

        # Count number of indices
        num_idx = torch.bincount(bin_num, minlength=self.n_bins)

        # Factor for original value is ema_rate ^ number of appearances
        factors = self.ema_rate ** num_idx
        self.bin_values *= factors

        # Add the target values times a scattered rate taking the
        # power into account.
        scattered_factors = (1-factors[bin_num]) / num_idx[bin_num]
        self.bin_values.scatter_add_(
            dim=0, index=bin_num, src=scattered_factors*target
        )

        # Update the integral values
        self.bin_integral[1:] = torch.cumsum(self.bin_values, dim=0)
