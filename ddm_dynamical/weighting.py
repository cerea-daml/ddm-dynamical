#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 09/03/2024
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2024}  {Tobias Sebastian Finn}

# System modules
import logging
import math

# External modules
import torch

# Internal modules

main_logger = logging.getLogger(__name__)


__all__ = [
    "SigmoidWeighting",
    "ExponentialWeighting",
    "ELBOWeighting",
    "EDMWeighting"
]


class SigmoidWeighting(torch.nn.Module):
    def __init__(self, shift: float = 2.):
        super().__init__()
        self.shift = shift

    def forward(self, gamma: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(-gamma + self.shift)


class ExponentialWeighting(torch.nn.Module):
    def __init__(self, multiplier: float = 0.5, shift: float = 0.):
        super().__init__()
        self.multiplier = multiplier
        self.shift = shift

    def forward(self, gamma: torch.Tensor) -> torch.Tensor:
        return torch.exp(-gamma*self.multiplier-self.shift)


class ELBOWeighting(torch.nn.Module):
    def forward(self, gamma: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(gamma)


class EDMWeighting(torch.nn.Module):
    def __init__(self, mean=2.4, scale=2.4) -> None:
        super().__init__()
        self.mean = mean
        self.scale = scale
        self.constant = -math.log(self.scale)-math.log(math.sqrt(2 * math.pi))

    def forward(self, gamma):
        data_part = -((gamma - self.mean) ** 2) / (2 * self.scale**2)
        pdf = (self.constant + data_part).exp()
        return pdf * (torch.exp(-gamma)+0.25)
