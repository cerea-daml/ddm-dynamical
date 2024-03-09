#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 08/03/2024
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2024}  {Tobias Sebastian Finn}

# System modules
import logging

# External modules
import torch

# Internal modules

main_logger = logging.getLogger(__name__)


class EPSParam(torch.nn.Module):
    def get_errors(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            target: torch.Tensor,
            noise: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor
    ) -> torch.Tensor:
        v_weight = (torch.exp(-gamma) + 1)
        return (noise - prediction).pow(2) / v_weight

    def forward(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor
    ):
        return alpha * in_data - sigma * prediction
