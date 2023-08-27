#!/bin/env python
# -*- coding: utf-8 -*-
#
#
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Any

# External modules
import torch

# Internal modules
from .head import HeadParam


logger = logging.getLogger(__name__)


class VelocityHead(HeadParam):
    def get_state(
            self,
            latent_state: torch.Tensor,
            prediction: torch.Tensor,
            alpha_sqrt: float,
            sigma: float
    ) -> torch.Tensor:
        return alpha_sqrt * latent_state - sigma * prediction

    def get_noise(
            self,
            latent_state: torch.Tensor,
            prediction: torch.Tensor,
            alpha_sqrt: float,
            sigma: float
    ) -> torch.Tensor:
        return sigma * latent_state + alpha_sqrt * prediction

    def get_velocity(
            self,
            latent_state: torch.Tensor,
            prediction: torch.Tensor,
            alpha_sqrt: float,
            sigma: float
    ) -> torch.Tensor:
        return prediction

    def forward(
            self,
            state: torch.Tensor,
            noise: torch.Tensor,
            prediction: torch.Tensor,
            alpha_sqrt: Any,
            sigma: Any
    ):
        target = alpha_sqrt*noise-sigma*state
        loss = self.loss_func(prediction, target)
        return loss
