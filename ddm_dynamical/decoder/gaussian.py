#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 07/09/2023
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from math import inf
from typing import Union

# External modules
import torch
import torch.nn
from torch.distributions import Normal

# Internal modules
from ..utils import masked_average

main_logger = logging.getLogger(__name__)


class GaussianDecoder(torch.nn.Module):
    n_dims = 1

    def __init__(
            self,
            mean: Union[float, torch.Tensor] = 0.,
            std: Union[float, torch.Tensor] = 1.,
            lower_bound: float = -inf,
            upper_bound: float = inf,
            std_dims: int = 3,
            ema_rate: float = 1.,
            stochastic: bool = False
    ):
        super().__init__()
        self.mean = mean
        self.std = std
        self.ema_rate = ema_rate
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.stochastic = stochastic
        self.register_buffer(
            "scale", torch.ones(*[1]*std_dims) * std,
        )

    def to_mean(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor
    ) -> torch.Tensor:
        return in_tensor * self.std + self.mean

    def forward(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        prediction = self.to_mean(in_tensor, first_guess)
        if self.stochastic:
            prediction.add_(torch.randn_like(prediction) * self.scale)
        prediction = prediction * mask
        return prediction.clamp(min=self.lower_bound, max=self.upper_bound)

    def update(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ) -> None:
        mean = self.to_mean(in_tensor, first_guess)
        squared_error = (mean-target) ** 2
        mse = masked_average(squared_error, mask).detach()
        self.scale = (
            self.ema_rate * self.scale ** 2
            + (1-self.ema_rate) * mse
        ).sqrt()

    def loss(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        mean = self.to_mean(in_tensor, first_guess)
        dist = Normal(mean, self.scale)
        nll = -dist.log_prob(target)
        return masked_average(nll, mask)
