#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 30/10/2023
# Created for diffusion_nextsim
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Iterable

# External modules
import torch

# Internal modules

main_logger = logging.getLogger(__name__)


class OptimalModel(torch.nn.Module):
    def __init__(
            self,
            training_data: torch.Tensor,
            gamma_min: float = -15,
            gamma_max: float = 10,
            dims: Iterable = (-3, -2, -1)
    ) -> None:
        """
        The optimal denoising model following Karras et al. (2022). Assumes that
        all training data fits into memory.
        """
        super().__init__()
        self.training_data = training_data
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.dims = dims

    def optimal_denoiser(self, state, var, mask) -> torch.Tensor:
        """
        Estimation of the optimal denoiser.
        """
        error = state[:, None, ...] - self.training_data
        error = error * mask
        sse = error.pow(2).sum(dim=self.dims, keepdims=True)
        log_likeli = -sse / (2 * var)
        prob = torch.softmax(log_likeli, dim=1)
        denoised_state = (prob * self.training_data).sum(dim=1)
        return denoised_state

    def forward(
            self,
            in_tensor: torch.Tensor,
            normalized_gamma: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate the optimal epsilon for given input and normalized gamma tensor.
        """
        if normalized_gamma.dim() > 0:
            normalized_gamma = normalized_gamma[0]
        gamma = normalized_gamma * (
                self.gamma_max-self.gamma_min
        ) + self.gamma_min
        variance = torch.sigmoid(-gamma)
        alpha = (1 - variance).sqrt()
        sigma = variance.sqrt()
        var_tilde = 1 / gamma.exp()

        in_tensor_exploded = in_tensor * (1 + var_tilde).sqrt()
        state = self.optimal_denoiser(in_tensor_exploded, var_tilde, mask)

        eps = (in_tensor - alpha * state) / sigma
        return eps * mask
