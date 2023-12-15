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
from typing import Union, Tuple, Callable

# External modules
import torch
import torch.nn
from torch.distributions import Normal

# Internal modules
from .base_decoder import BaseDecoder
from .mean_funcs import delta_mean
from ..utils import masked_average

main_logger = logging.getLogger(__name__)


class GaussianDecoder(BaseDecoder):
    n_dims = 1

    def __init__(
            self,
            mean: Union[float, torch.Tensor] = 0.,
            std: Union[float, torch.Tensor] = 1.,
            lower_bound: float = -inf,
            upper_bound: float = inf,
            std_dims: int = 3,
            ema_rate: float = 1.,
            stochastic: bool = False,
            mean_func: Callable = delta_mean
    ):
        super().__init__(stochastic=stochastic)
        self.mean = mean
        self.std = std
        self.ema_rate = ema_rate
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mean_func = mean_func
        self.register_buffer("scale", torch.ones(*[1]*std_dims) * std)
        self.register_buffer("fixed_scale", torch.ones(*[1]*std_dims) * std)

    def forward(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        prediction = self.mean_func(in_tensor, first_guess)
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
        mean = self.mean_func(in_tensor, first_guess)
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_func(in_tensor, first_guess)
        dist = Normal(mean, self.scale)
        nll = -dist.log_prob(target)
        dist_clim = Normal(mean, self.scale)
        nll_clim = -dist_clim.log_prob(target)
        return masked_average(nll, mask), masked_average(nll_clim, mask)
