#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 10/11/2023
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
from abc import abstractmethod
from typing import Tuple

# External modules
import torch.nn

# Internal modules

main_logger = logging.getLogger(__name__)


class BaseDecoder(torch.nn.Module):
    n_dims = 0

    def __init__(self, stochastic: bool = False):
        super().__init__()
        self.stochastic = stochastic

    @abstractmethod
    def forward(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def update(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ) -> None:
        pass

    @abstractmethod
    def loss(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
