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
import math
from typing import Union, Tuple, Callable
from types import MethodType

# External modules
import torch
import torch.nn
from torch.distributions import Normal

# Internal modules
from .base_decoder import BaseDecoder
from .mean_funcs import delta_mean
from ..utils import masked_average

main_logger = logging.getLogger(__name__)


_const = math.log(math.sqrt(2 * math.pi))


class CensoredDecoder(BaseDecoder):
    n_dims = 1

    def __init__(
            self,
            mean: Union[float, torch.Tensor] = 0.,
            std: Union[float, torch.Tensor] = 1.,
            lower_bound: float = -math.inf,
            upper_bound: float = math.inf,
            std_dims: int = 3,
            stochastic: bool = False,
            mean_func: Callable = delta_mean
    ):
        super().__init__(stochastic=stochastic)
        self.mean = mean
        self.std = std
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.to_mean = MethodType(mean_func, self)
        self.register_parameter(
            "scale", torch.nn.Parameter(torch.ones(*[1]*std_dims) * std)
        )
        self.register_buffer("fixed_scale", torch.ones(*[1]*std_dims) * std)

    def forward(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        prediction = self.to_mean(in_tensor, first_guess)
        if self.stochastic:
            prediction.add_(self.scale * torch.randn_like(prediction))
        prediction = prediction * mask
        return prediction.clamp(min=self.lower_bound, max=self.upper_bound)

    def update(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ) -> None:
        pass

    def get_loss(
            self,
            mean: torch.Tensor,
            scale: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        loss = 0
        logpdf_mask = mask
        if self.lower_bound > -torch.inf:
            logcdf_lower = torch.special.log_ndtr(
                (self.lower_bound-mean)/scale
            )
            mask_lower = target <= self.lower_bound
            loss = loss-masked_average(
                logcdf_lower, mask * mask_lower
            )
            logpdf_mask = logpdf_mask * ~mask_lower
        if self.upper_bound < torch.inf:
            logcdf_upper = torch.special.log_ndtr(
                -(self.upper_bound-mean)/scale
            )
            mask_upper = target >= self.upper_bound
            loss = loss-masked_average(logcdf_upper, mask * mask_upper)
            logpdf_mask = logpdf_mask * ~mask_upper
        norm_val = (target-mean)/scale
        logpdf = -0.5 * (norm_val.pow(2)) - scale.log() - _const
        return loss-masked_average(logpdf, logpdf_mask)

    def loss(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.to_mean(in_tensor, first_guess)
        loss = self.get_loss(mean, self.scale, target, mask)C
        loss_clim = -Normal(mean, self.fixed_scale).log_prob(target)
        loss_clim = masked_average(loss_clim, mask)
        return loss, loss_clim
