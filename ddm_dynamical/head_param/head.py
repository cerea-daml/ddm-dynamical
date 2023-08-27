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
import abc
from typing import Any

# External modules
import torch

# Internal modules


logger = logging.getLogger(__name__)


class HeadParam(abc.ABC, torch.nn.Module):
    def __init__(self, loss_func: torch.nn.Module):
        super().__init__()
        self.loss_func = loss_func

    @abc.abstractmethod
    def forward(
            self,
            state: torch.Tensor,
            noise: torch.Tensor,
            prediction: torch.Tensor,
            alpha_sqrt: Any,
            sigma: Any
    ):
        pass

    @abc.abstractmethod
    def get_state(
            self,
            latent_state: torch.Tensor,
            prediction: torch.Tensor,
            alpha_sqrt: float,
            sigma: float
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_noise(
            self,
            latent_state: torch.Tensor,
            prediction: torch.Tensor,
            alpha_sqrt: float,
            sigma: float
    ) -> torch.Tensor:
        pass

    def get_velocity(
            self,
            latent_state: torch.Tensor,
            prediction: torch.Tensor,
            alpha_sqrt: float,
            sigma: float
    ) -> torch.Tensor:
        noise_part = alpha_sqrt*self.get_noise(
            latent_state=latent_state, prediction=prediction, alpha_sqrt=alpha_sqrt,
            sigma=sigma
        )
        state_part = -sigma * self.get_state(
            latent_state=latent_state, prediction=prediction, alpha_sqrt=alpha_sqrt,
            sigma=sigma
        )
        return noise_part+state_part
