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

# External modules
import torch

# Internal modules

main_logger = logging.getLogger(__name__)


__all__ = [
    "EPSParam",
    "VParam",
    "DataParam"
]


class EPSParam(torch.nn.Module):
    def estimate_errors(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            target: torch.Tensor,
            noise: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        return (noise - prediction).pow(2)

    def forward(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            *args, **kwargs
    ):
        return alpha * in_data - sigma * prediction


class VParam(torch.nn.Module):
    def estimate_errors(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            target: torch.Tensor,
            noise: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        v_target = alpha * noise - sigma * target
        v_weight = (torch.exp(-gamma) + 1)
        return (v_target - prediction).pow(2) / v_weight

    def forward(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            *args, **kwargs
    ):
        return alpha * in_data - sigma * prediction


class DataParam(torch.nn.Module):
    def estimate_errors(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            target: torch.Tensor,
            noise: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        data_weight = torch.exp(gamma)
        return data_weight * (target - prediction).pow(2)

    def forward(
            self,
            prediction: torch.Tensor,
            in_data: torch.Tensor,
            alpha: torch.Tensor,
            sigma: torch.Tensor,
            gamma: torch.Tensor,
            *args, **kwargs
    ):
        return prediction
