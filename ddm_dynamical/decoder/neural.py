#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 28/11/2023
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
from .base_decoder import BaseDecoder


main_logger = logging.getLogger(__name__)


class NeuralDecoder(BaseDecoder):
    def __init__(
            self,
            network: torch.nn.Module,
            physical_decoder: BaseDecoder,
            stochastic: bool = True
    ):
        super().__init__()
        self.network = network
        self.physical_decoder = physical_decoder
        self.physical_decoder.stochastic = stochastic

    def forward(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        decoded = self.network(in_tensor, mask)
        return self.physical_decoder(decoded, first_guess, mask)

    def update(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ) -> None:
        decoded = self.network(in_tensor, mask)
        _ = self.physical_decoder.update(decoded, first_guess, target, mask)

    def loss(
            self,
            in_tensor: torch.Tensor,
            first_guess: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        decoded = self.network(in_tensor, mask)
        return self.physical_decoder.loss(decoded, first_guess, target, mask)
