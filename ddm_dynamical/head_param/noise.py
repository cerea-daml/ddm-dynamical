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


class NoiseHead(HeadParam):
    def get_state(
            self,
            latent_state: torch.Tensor,
            prediction: torch.Tensor,
            alpha_sqrt: float,
            sigma: float
    ) -> torch.Tensor:
        return (latent_state-sigma*prediction) / alpha_sqrt

    def get_noise(
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
        loss = self.loss_func(prediction, noise)
        return loss
