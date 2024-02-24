#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 24/02/2024
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2024}  {Tobias Sebastian Finn}

# System modules
import logging

import torch.nn

# External modules

# Internal modules

main_logger = logging.getLogger(__name__)


class ExponentialWeighting(torch.nn.Module):
    def __init__(self, multiplier: float = 0.5, shift: float = 0.):
        super().__init__()
        self.multiplier = multiplier
        self.shift = shift

    def forward(self, gamma: torch.Tensor) -> torch.Tensor:
        return torch.exp(-gamma*self.multiplier-self.shift)
