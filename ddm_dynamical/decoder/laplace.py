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
from types import MethodType

# External modules
import torch
import torch.nn
from torch.distributions import Normal, Laplace

# Internal modules
from .base_decoder import BaseDecoder
from .prediction_funcs import delta_prediction
from ..utils import masked_average

main_logger = logging.getLogger(__name__)


class LaplaceDecoder(BaseDecoder):
    n_dims = 1

    def __init__(
            self,
            mean: Union[float, torch.Tensor] = 0.,
            std: Union[float, torch.Tensor] = 1.,
            lower_bound: float = -inf,
            upper_bound: float = inf,
            ema_rate: float = 1.,
            stochastic: bool = False,
            prediction_func: Callable = delta_prediction,
            **kwargs
    ):
        super().__init__(stochastic=stochastic)
        self.mean = mean
        self.std = std
        self.ema_rate = ema_rate
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.to_prediction = MethodType(prediction_func, self)
        self.register_buffer("scale", torch.ones(1) * std)

    def forward(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        prediction = self.to_prediction(in_tensor, first_guess)
        if self.stochastic:
            u = torch.rand_like(prediction)*2-1
            prediction.sub_(self.scale * u.sign() * torch.log1p(-u.abs()))
        prediction = prediction * mask
        return prediction.clamp(min=self.lower_bound, max=self.upper_bound)

    def update(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ) -> None:
        prediction = self.to_prediction(in_tensor, first_guess)
        mae = masked_average((target-prediction).abs(), mask).detach()
        self.scale = self.ema_rate * self.scale + (1-self.ema_rate) * mae

    def loss(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prediction = self.to_prediction(in_tensor, first_guess)
        dist = Laplace(prediction, self.scale)
        loss = masked_average(-dist.log_prob(target), mask)

        # Climatological loss
        prediction = self(in_tensor, first_guess, mask)
        loss_clim = ((prediction-target)/self.std).pow(2)
        loss_clim = masked_average(loss_clim, mask)
        return loss, loss_clim
