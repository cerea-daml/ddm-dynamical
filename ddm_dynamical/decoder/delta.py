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

# External modules
import torch.nn
from torch.distributions import Normal

# Internal modules
from .gaussian import GaussianDecoder

main_logger = logging.getLogger(__name__)


class DeltaDecoder(GaussianDecoder):
    def to_mean(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor
    ) -> torch.Tensor:
        dynamics = in_tensor * self.std + self.mean
        return first_guess + dynamics
