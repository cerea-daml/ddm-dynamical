#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 26/10/2023
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
import math

# External modules
import torch
import torch.nn

# Internal modules

main_logger = logging.getLogger(__name__)


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
