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

import torch.nn

# External modules

# Internal modules

main_logger = logging.getLogger(__name__)


class SigmoidWeighting(torch.nn.Module):
    def __init__(self, shift: float = 2.):
        super().__init__()
        self.shift = shift

    def forward(self, gamma: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(-gamma + self.shift)
