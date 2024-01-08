#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 15/12/2023
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging

# External modules
import torch

# Internal modules

main_logger = logging.getLogger(__name__)


def state_prediction(
        self,
        in_tensor: torch.Tensor,
        first_guess: torch.Tensor
):
    return in_tensor * self.std + self.mean


def delta_prediction(
        self,
        in_tensor: torch.Tensor,
        first_guess: torch.Tensor
):
    dynamics = in_tensor * self.std + self.mean
    return first_guess + dynamics
