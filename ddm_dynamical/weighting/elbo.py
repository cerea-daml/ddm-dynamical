#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 25/10/2023
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging

# External modules
import torch.nn

# Internal modules

main_logger = logging.getLogger(__name__)


class ELBOWeighting(torch.nn.Module):
    def forward(self, gamma: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(gamma)
